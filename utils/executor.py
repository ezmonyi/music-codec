"""Executor: one-epoch train loop and CV with DDP join and save."""

import logging
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from tqdm import tqdm

from utils.train_utils import (
    update_parameter_and_lr,
    log_per_step,
    log_per_save,
    batch_forward,
    batch_backward,
    save_model_opt,
    cosyvoice_join,
    get_scheduled_vq_weights,
)


class Executor:
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.rank = int(__import__("os").environ.get("RANK", 0))
        self.device = torch.device("cuda:{}".format(self.rank))

    def train_one_epoc(
        self,
        model,
        optimizer,
        scheduler,
        train_data_loader,
        cv_data_loader,
        writer,
        info_dict,
        group_join,
    ):
        lr = optimizer.param_groups[0]["lr"]
        logging.info("Epoch {} TRAIN info lr {} rank {}".format(self.epoch, lr, self.rank))
        logging.info(
            "using accumulate grad, new batch size is {} times larger".format(
                info_dict["accum_grad"]
            )
        )
        # Initialize progress bar for rank 0
        if self.rank == 0:
            # IterableDataset reports len=0; avoid calling len(dataloader) to prevent warning
            dataset = getattr(train_data_loader, "dataset", None)
            if dataset is not None and isinstance(dataset, IterableDataset):
                total_batches = None
            else:
                try:
                    total_batches = len(train_data_loader)
                except (TypeError, AttributeError):
                    total_batches = None
            pbar = tqdm(
                desc=f"epoch-{self.epoch}",
                total=total_batches,
                unit="batch",
                ncols=120,
                dynamic_ncols=True,
            )

        model.train()
        model_context = model.join if info_dict.get("train_engine") == "torch_ddp" else nullcontext
        
        # Log from all ranks (DataLoader workers run on each rank)
        logging.info(f"[Rank {self.rank}] Starting training loop, entering DataLoader iteration...")
        if self.rank == 0:
            logging.info(f"[Rank {self.rank}] Progress bar will show batch processing status")
        
        steps_per_epoch = info_dict.get("steps_per_epoch") or 0  # 0 = no limit; set >0 to keep all ranks in sync (avoid NCCL timeout)

        # Suppress PyTorch UserWarning about IterableDataset length when using multiprocessing
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Length of IterableDataset.*was reported to be 0.*",
                category=UserWarning,
            )
            with model_context():
                for batch_idx, batch_dict in enumerate(train_data_loader):
                    # Fixed epoch size: keep all ranks in sync to avoid NCCL collective timeout
                    if steps_per_epoch > 0 and batch_idx >= steps_per_epoch:
                        break
                    if batch_idx == 0:
                        logging.info(f"[Rank {self.rank}] First batch loaded, batch keys: {list(batch_dict.keys())}")
                        if "whisper_feat" in batch_dict:
                            logging.info(f"[Rank {self.rank}] Batch shapes - whisper: {batch_dict['whisper_feat'].shape}, "
                                        f"wavlm: {batch_dict['wavlm_feat'].shape}, muq: {batch_dict['muq_feat'].shape}, "
                                        f"mel: {batch_dict['mel'].shape}")
                    info_dict["tag"] = "TRAIN"
                    info_dict["step"] = self.step
                    info_dict["epoch"] = self.epoch
                    info_dict["batch_idx"] = batch_idx
                    info_dict.update(get_scheduled_vq_weights(self.step, info_dict))
                    if cosyvoice_join(group_join, info_dict):
                        logging.warning("cosyvoice_join break this epoch")
                        break

                    if (
                        info_dict.get("train_engine") == "torch_ddp"
                        and (batch_idx + 1) % info_dict["accum_grad"] != 0
                    ):
                        context = model.no_sync
                    else:
                        context = nullcontext

                    with context():
                        info_dict = batch_forward(model, batch_dict, info_dict)
                        info_dict = batch_backward(model, info_dict)

                    if self.rank == 0:
                        _s = lambda v: v.item() if torch.is_tensor(v) else v
                        msgs = [
                            "{}={:.5f}".format(k, _s(v))
                            for k, v in info_dict["loss_dict"].items()
                        ]
                        pbar.set_postfix_str(";".join(msgs))

                    info_dict = update_parameter_and_lr(
                        model, optimizer, scheduler, info_dict
                    )
                    log_per_step(writer, info_dict)

                    save_per_step = info_dict.get("save_per_step", 0)
                    if (
                        save_per_step > 0
                        and (self.step + 1) % save_per_step == 0
                        and (batch_idx + 1) % info_dict["accum_grad"] == 0
                    ):
                        dist.barrier()
                        self.cv(
                            model,
                            optimizer,
                            cv_data_loader,
                            writer,
                            info_dict,
                            on_batch_end=False,
                        )
                        model.train()

                    if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                        self.step += 1
                        if self.rank == 0:
                            pbar.update(1)  # Update progress bar after gradient accumulation step
        
        # Close progress bar and log completion
        if self.rank == 0:
            pbar.close()
        logging.info(f"[Rank {self.rank}] Completed epoch {self.epoch}, processed {batch_idx + 1} batches")

        dist.barrier()
        self.cv(
            model,
            optimizer,
            cv_data_loader,
            writer,
            info_dict,
            on_batch_end=True,
        )

    @torch.inference_mode()
    def cv(
        self,
        model,
        optimizer,
        cv_data_loader,
        writer,
        info_dict,
        on_batch_end=True,
    ):
        logging.info(
            "Epoch {} Step {} on_batch_end {} CV rank {}".format(
                self.epoch, self.step + 1, on_batch_end, self.rank
            )
        )
        model.eval()

        if cv_data_loader is not None:
            total_num = 0
            total_loss_dict = {}
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                info_dict.update(get_scheduled_vq_weights(self.step, info_dict))
                num_utts = batch_dict["mel"].shape[0]
                total_num += num_utts
                info_dict = batch_forward(model, batch_dict, info_dict)
                for k, v in info_dict["loss_dict"].items():
                    if k not in total_loss_dict:
                        total_loss_dict[k] = []
                    total_loss_dict[k].append(v.item() * num_utts)
            for k in total_loss_dict:
                total_loss_dict[k] = sum(total_loss_dict[k]) / max(total_num, 1)
            info_dict["loss_dict"] = total_loss_dict
            log_per_save(writer, info_dict)
        else:
            info_dict["tag"] = "SkipedCV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch

        if on_batch_end:
            model_name = "epoch_{}_whole".format(self.epoch)
        else:
            model_name = "epoch_{}_step_{}".format(self.epoch, self.step + 1)
        save_model_opt(model, optimizer, model_name, info_dict)
