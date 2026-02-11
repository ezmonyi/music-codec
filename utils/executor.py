"""Executor: one-epoch train loop and CV with DDP join and save."""

import logging
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.train_utils import (
    update_parameter_and_lr,
    log_per_step,
    log_per_save,
    batch_forward,
    batch_backward,
    save_model_opt,
    cosyvoice_join,
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
        if self.rank == 0:
            pbar = tqdm(desc="epoch-{}".format(self.epoch), ncols=100)

        model.train()
        model_context = model.join if info_dict.get("train_engine") == "torch_ddp" else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
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
                    msgs = [
                        "{}={:.5f}".format(k, v.item())
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
                        pbar.update(1)

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
