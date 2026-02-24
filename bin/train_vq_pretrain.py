"""
VQ module pretraining: train encoder (resamplers + in_proj) + VQ codebook
as a feature autoencoder, without flow matching loss.

Architecture:
    features -> resample -> concat -> in_proj -> VQ -> decoder_MLP -> reconstruct concat
    Loss: recon_weight * L1(recon, target) + commit_weight * commit_loss
         + entropy_weight * entropy_loss [+ codebook_weight * codebook_loss (RVQ)]

The reconstruction target is stride-2 downsampled raw features concatenated
(no encoder gradient on target side), so the loss cleanly trains the encoder
to produce VQ-friendly representations.

The decoder MLP is discarded after pretraining. Exported checkpoints
(encoder_vq_*.pt) contain the full AudioReconModel state_dict and can be
loaded via train_conf.restore_model_path in the main training config.

Usage:
    ./bin/run_vq_pretrain.sh [conf/vq_pretrain.yaml]
"""

from __future__ import print_function

import os
import sys
import argparse
import datetime
import logging
import warnings
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.train_utils import (
    init_distributed,
    check_modify_and_save_config,
    find_last_state,
)
from utils.scheduler import WarmupLR, ConstantLR
from dataset.codec_dataset import init_dataset_and_dataloader


class VQPretrainDecoder(nn.Module):
    """Lightweight MLP decoder: VQ output -> reconstructed concat features.
    Discarded after pretraining."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [1024]
        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class VQPretrainWrapper(nn.Module):
    """Wraps AudioReconModel (encoder+VQ) and throwaway decoder for DDP."""

    def __init__(self, model, decoder):
        super().__init__()
        self.model = model
        self.decoder = decoder

    def forward(self, whisper_feat, wavlm_feat, muq_feat):
        """Returns (recon, codes, commit_loss, codebook_loss_or_None, entropy_loss)."""
        z_q_st, codes, *vq_losses = self.model.encode(
            whisper_feat, wavlm_feat, muq_feat
        )
        recon = self.decoder(z_q_st)
        if self.model.use_rvq:
            commit_loss, codebook_loss, entropy_loss = vq_losses
        else:
            commit_loss, entropy_loss = vq_losses
            codebook_loss = None
        return recon, codes, commit_loss, codebook_loss, entropy_loss


def get_args():
    parser = argparse.ArgumentParser(description="VQ module pretraining")
    parser.add_argument(
        "--train_engine", default="torch_ddp", choices=["torch_ddp"]
    )
    parser.add_argument("--model", default="model")
    parser.add_argument("--config", required=True, help="config yaml (hyperpyyaml)")
    parser.add_argument("--local-rank", help="placeholder for DDP")
    parser.add_argument(
        "--dataset_conf", required=True, help="dataset yaml or manifest"
    )
    parser.add_argument("--model_dir", required=True, help="checkpoint save dir")
    parser.add_argument("--tensorboard_dir", default="tensorboard")
    parser.add_argument(
        "--ddp.dist_backend",
        dest="dist_backend",
        default="nccl",
        choices=["nccl", "gloo"],
    )
    parser.add_argument("--timeout", default=300, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--prefetch", default=2, type=int)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    return parser.parse_args()


def build_recon_target(whisper_feat, wavlm_feat, muq_feat):
    """Stride-2 downsample 50Hz features, concat with 25Hz muq -> (B, T25, 3328).

    This is the reconstruction target for VQ pretraining. Using simple stride-2
    instead of learned Conv1d so the target is independent of encoder parameters.
    """
    T25 = muq_feat.shape[1]
    w25 = whisper_feat[:, ::2, :][:, :T25, :]
    wl25 = wavlm_feat[:, ::2, :][:, :T25, :]
    return torch.cat([w25, wl25, muq_feat[:, :T25, :]], dim=-1)


def main():
    args = get_args()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # ── Load config ──────────────────────────────────────────────────
    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f)

    configs["train_conf"] = configs.get("train_conf", {})
    orig_batch = configs["train_conf"].get("batch_size")
    configs["train_conf"].update(vars(args))
    if args.batch_size is not None:
        configs["train_conf"]["batch_size"] = args.batch_size
    elif orig_batch is not None:
        configs["train_conf"]["batch_size"] = orig_batch

    if "dataset_conf" not in configs:
        configs["dataset_conf"] = {}
    if isinstance(args.dataset_conf, str) and os.path.isfile(args.dataset_conf):
        if args.dataset_conf.endswith((".yaml", ".yml")):
            import yaml

            with open(args.dataset_conf, "r") as f:
                configs["dataset_conf"].update(yaml.safe_load(f) or {})
        else:
            configs["dataset_conf"]["manifest_path"] = args.dataset_conf

    # ── DDP ──────────────────────────────────────────────────────────
    init_distributed(args)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # ── Dataset ──────────────────────────────────────────────────────
    if rank == 0:
        logging.info("Initializing dataset...")
    train_dataset, _, train_loader, _ = init_dataset_and_dataloader(args, configs)
    if rank == 0:
        logging.info("Dataset ready")

    configs = check_modify_and_save_config(args, configs)
    train_conf = configs["train_conf"]

    # ── TensorBoard ──────────────────────────────────────────────────
    writer = None
    if rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

    # ── Build model ──────────────────────────────────────────────────
    model = configs[args.model]

    # Freeze CFM entirely (not used in VQ pretraining)
    for name, p in model.named_parameters():
        if name.startswith("cfm."):
            p.requires_grad = False

    # Decoder input dim = encode() output dim (256 for both single VQ and RVQ after cond_proj)
    if model.use_rvq and model.cond_proj is not None:
        vq_output_dim = model.cond_proj.out_features
    else:
        vq_output_dim = model.codebook_dim

    concat_dim = 1280 + 1024 + 1024  # whisper + wavlm + muq = 3328
    decoder_hidden = train_conf.get("decoder_hidden_dims", [1024])
    decoder = VQPretrainDecoder(vq_output_dim, concat_dim, decoder_hidden)

    wrapper = VQPretrainWrapper(model, decoder)

    if rank == 0:
        trainable = sum(
            p.numel() for p in wrapper.parameters() if p.requires_grad
        )
        frozen = sum(
            p.numel() for p in wrapper.parameters() if not p.requires_grad
        )
        logging.info(
            f"VQ pretrain: trainable={trainable / 1e6:.2f}M, "
            f"frozen(CFM)={frozen / 1e6:.2f}M"
        )

    # ── Resume / Restore ─────────────────────────────────────────────
    last_model_path, last_opt_path, last_step, last_epoch = find_last_state(
        args.model_dir
    )
    if last_model_path and os.path.isfile(last_model_path):
        ckpt = torch.load(last_model_path, map_location="cpu")
        wrapper.load_state_dict(ckpt, strict=True)
        logging.info(
            f"Resume VQ pretrain from {last_model_path} "
            f"epoch={last_epoch} step={last_step}"
        )
    else:
        restore_path = train_conf.get("restore_model_path")
        if restore_path and os.path.isfile(restore_path):
            ckpt = torch.load(restore_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            logging.info(
                f"Restored encoder/VQ from {restore_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    # ── DDP wrap ─────────────────────────────────────────────────────
    wrapper.cuda()
    wrapper = torch.nn.parallel.DistributedDataParallel(
        wrapper, device_ids=[local_rank], find_unused_parameters=True
    )

    # ── Optimizer (only trainable params) ────────────────────────────
    optim_conf = train_conf.get("optim_conf", {"lr": 1e-3})
    optim_name = train_conf.get("optim", "adamw")
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, **optim_conf)
    else:
        optimizer = torch.optim.Adam(trainable_params, **optim_conf)

    if last_opt_path and os.path.isfile(last_opt_path):
        opt_ckpt = torch.load(last_opt_path, map_location="cpu")
        optimizer.load_state_dict(opt_ckpt["optim"])
        logging.info(f"Resume optimizer from {last_opt_path}")

    # ── Scheduler ────────────────────────────────────────────────────
    sched_conf = dict(train_conf.get("scheduler_conf", {}))
    sched_conf["last_epoch"] = last_epoch if last_epoch else -1
    if train_conf.get("scheduler", "warmuplr") == "warmuplr":
        scheduler = WarmupLR(optimizer, **sched_conf)
    else:
        scheduler = ConstantLR(optimizer)

    step = last_step or 0
    last_epoch = last_epoch or 0

    # ── GPU feature extractor (when dataset returns mel only) ────────
    dataset_conf = configs.get("dataset_conf", {})
    feature_extractor = None
    if dataset_conf.get("use_mel_extractor") and dataset_conf.get(
        "feature_extraction_on_gpu"
    ):
        from dataset.mel_to_features import CodecFeatureExtractor

        fe_conf = dict(dataset_conf.get("feature_extractor", {}))
        fe_conf["device"] = str(device)
        if fe_conf.get("wavlm_ckpt") == "":
            fe_conf["wavlm_ckpt"] = None
        feature_extractor = CodecFeatureExtractor(**fe_conf)
        if rank == 0:
            logging.info(f"GPU feature extractor on {device}")

    # ── Training hyperparams ─────────────────────────────────────────
    max_epoch = train_conf.get("max_epoch", 20)
    save_per_step = train_conf.get("save_per_step", 2000)
    log_interval = train_conf.get("log_interval", 50)
    accum_grad = train_conf.get("accum_grad", 1)
    grad_clip = train_conf.get("grad_clip", 5.0)
    recon_w = train_conf.get("recon_loss_weight", 1.0)
    commit_w = train_conf.get("commit_loss_weight", 1.0)
    entropy_w = train_conf.get("entropy_loss_weight", 0.1)
    codebook_w = train_conf.get("codebook_loss_weight", 1.0)
    steps_per_epoch = train_conf.get("steps_per_epoch", 0)
    dtype_str = train_conf.get("dtype", "fp32")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        dtype_str, torch.float32
    )

    base_model = wrapper.module.model
    codebook_size = base_model.codebook_size
    use_rvq = base_model.use_rvq

    def _codebook_util(codes):
        if use_rvq and codes.dim() == 3:
            sizes = base_model.rvq.codebook_sizes
            utils = [
                codes[..., i].unique().numel() / float(sizes[i])
                for i in range(codes.shape[-1])
            ]
            return sum(utils) / len(utils)
        return codes.unique().numel() / float(codebook_size)

    def save_ckpt(epoch, step, is_epoch_end=False):
        if rank != 0:
            return
        tag = (
            f"epoch_{epoch}_whole"
            if is_epoch_end
            else f"epoch_{epoch}_step_{step}"
        )
        model_path = os.path.join(args.model_dir, f"model_{tag}.pt")
        opt_path = os.path.join(args.model_dir, f"opt_{tag}.pt")
        export_path = os.path.join(args.model_dir, f"encoder_vq_{tag}.pt")

        torch.save(wrapper.module.state_dict(), model_path)
        torch.save(
            {"optim": optimizer.state_dict(), "step": step, "epoch": epoch},
            opt_path,
        )
        # Full AudioReconModel state for main training's restore_model_path
        torch.save(wrapper.module.model.state_dict(), export_path)
        logging.info(
            f"Checkpoint step={step}: {os.path.basename(model_path)}, "
            f"export: {os.path.basename(export_path)}"
        )

    # ── Training loop ────────────────────────────────────────────────
    logging.info(
        f"[Rank {rank}] VQ pretrain: epochs {last_epoch}..{max_epoch - 1}, "
        f"from step {step}"
    )

    for epoch in range(last_epoch, max_epoch):
        wrapper.train()
        if hasattr(train_loader, "sampler") and hasattr(
            train_loader.sampler, "set_epoch"
        ):
            train_loader.sampler.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=args.timeout),
        )

        pbar = None
        if rank == 0:
            ds = getattr(train_loader, "dataset", None)
            total = (
                None
                if ds and isinstance(ds, IterableDataset)
                else (len(train_loader) if ds else None)
            )
            pbar = tqdm(
                desc=f"vq-pretrain-e{epoch}",
                total=total,
                unit="batch",
                ncols=120,
                dynamic_ncols=True,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Length of IterableDataset.*",
                category=UserWarning,
            )
            with wrapper.join():
                for batch_idx, batch in enumerate(train_loader):
                    if steps_per_epoch > 0 and batch_idx >= steps_per_epoch:
                        break

                    # ── Prepare features ──
                    mel = batch["mel"].to(device, dtype=dtype)
                    if "whisper_feat" not in batch:
                        if feature_extractor is None:
                            raise RuntimeError(
                                "Batch has only mel but no GPU feature_extractor"
                            )
                        with torch.no_grad():
                            wf, wlf, mf = feature_extractor.extract_batch(mel)
                        wf = wf.to(dtype)
                        wlf = wlf.to(dtype)
                        mf = mf.to(dtype)
                    else:
                        wf = batch["whisper_feat"].to(device, dtype=dtype)
                        wlf = batch["wavlm_feat"].to(device, dtype=dtype)
                        mf = batch["muq_feat"].to(device, dtype=dtype)

                    # Target: stride-2 downsampled features (detached from encoder)
                    with torch.no_grad():
                        target = build_recon_target(wf, wlf, mf)

                    # ── Gradient accumulation sync context ──
                    no_sync = (
                        wrapper.no_sync
                        if accum_grad > 1
                        and (batch_idx + 1) % accum_grad != 0
                        else nullcontext
                    )

                    with no_sync():
                        with torch.amp.autocast(
                            enabled=(dtype != torch.float32),
                            device_type="cuda",
                            dtype=dtype,
                        ):
                            recon, codes, commit_loss, codebook_loss, entropy_loss = (
                                wrapper(wf, wlf, mf)
                            )

                        min_t = min(recon.shape[1], target.shape[1])
                        recon_loss = F.l1_loss(
                            recon[:, :min_t], target[:, :min_t]
                        )

                        loss = recon_w * recon_loss + commit_w * commit_loss
                        if codebook_loss is not None:
                            loss = loss + codebook_w * codebook_loss
                        if entropy_w > 0 and entropy_loss is not None:
                            loss = loss + entropy_w * entropy_loss

                        (loss / accum_grad).backward()

                    # ── Optimizer step ──
                    if (batch_idx + 1) % accum_grad == 0:
                        grad_norm = clip_grad_norm_(
                            wrapper.parameters(), grad_clip
                        )
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        step += 1

                        cb_util = _codebook_util(codes)

                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix_str(
                                f"recon={recon_loss.item():.4f};"
                                f"commit={commit_loss.item():.4f};"
                                f"cb_util={cb_util:.3f};"
                                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                            )

                        # TensorBoard logging
                        if writer and step % log_interval == 0:
                            writer.add_scalar(
                                "train/total_loss", loss.item(), step
                            )
                            writer.add_scalar(
                                "train/recon_loss", recon_loss.item(), step
                            )
                            writer.add_scalar(
                                "train/commit_loss", commit_loss.item(), step
                            )
                            writer.add_scalar(
                                "train/codebook_util", cb_util, step
                            )
                            writer.add_scalar(
                                "train/lr",
                                optimizer.param_groups[0]["lr"],
                                step,
                            )
                            gn = (
                                grad_norm.item()
                                if torch.is_tensor(grad_norm)
                                else grad_norm
                            )
                            writer.add_scalar("train/grad_norm", gn, step)
                            if codebook_loss is not None:
                                writer.add_scalar(
                                    "train/codebook_loss",
                                    codebook_loss.item(),
                                    step,
                                )
                            if entropy_loss is not None:
                                writer.add_scalar(
                                    "train/entropy_loss",
                                    entropy_loss.item(),
                                    step,
                                )
                            writer.add_histogram(
                                "vq/code_usage",
                                codes.detach().flatten(),
                                step,
                            )

                        # Periodic checkpoint
                        if save_per_step > 0 and step % save_per_step == 0:
                            dist.barrier()
                            save_ckpt(epoch, step)

        if pbar is not None:
            pbar.close()
        logging.info(f"[Rank {rank}] Epoch {epoch} done, step {step}")

        dist.barrier()
        save_ckpt(epoch, step, is_epoch_end=True)
        dist.destroy_process_group(group_join)

    if writer:
        writer.close()
    logging.info(
        f"VQ pretraining complete. Final step: {step}. "
        f"Use encoder_vq_*.pt as restore_model_path in main training."
    )


if __name__ == "__main__":
    main()
