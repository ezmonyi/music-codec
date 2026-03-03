"""
Training utilities: DDP init, optimizer/scheduler, checkpoint save/load,
batch forward/backward for codec (flow_loss + commit_loss), logging.

We use native torch DDP (not PyTorch Lightning) to match flowmatching2/cosyvoice:
explicit control over checkpoint format, resume logic, and training loop.
"""

from contextlib import nullcontext
import datetime
import logging
import os
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from utils.scheduler import WarmupLR, ConstantLR


def get_scheduled_vq_weights(step, info_dict):
    """Compute flow_loss_weight, commit_loss_weight and entropy_loss_weight with optional linear decay.

    Config (in train_conf):
      - flow_loss_weight_start, flow_loss_weight_end, flow_loss_decay_steps
      - commit_loss_weight_start, commit_loss_weight_end, commit_loss_decay_steps
      - entropy_loss_weight_start, entropy_loss_weight_end, entropy_loss_decay_steps
    If decay_steps is 0 or missing, use constant weights.
    """
    out = {}
    # Flow loss weight schedule (reconstruction loss - minimize early to focus on codebook)
    start = info_dict.get("flow_loss_weight_start")
    end = info_dict.get("flow_loss_weight_end")
    decay_steps = info_dict.get("flow_loss_decay_steps", 0)
    if decay_steps > 0 and start is not None and end is not None:
        progress = min(1.0, step / decay_steps)
        out["flow_loss_weight"] = end + (start - end) * (1.0 - progress)
    else:
        out["flow_loss_weight"] = info_dict.get("flow_loss_weight", 0.1)
    # Commit loss weight schedule
    start = info_dict.get("commit_loss_weight_start")
    end = info_dict.get("commit_loss_weight_end")
    decay_steps = info_dict.get("commit_loss_decay_steps", 0)
    if decay_steps > 0 and start is not None and end is not None:
        progress = min(1.0, step / decay_steps)
        out["commit_loss_weight"] = end + (start - end) * (1.0 - progress)
    else:
        out["commit_loss_weight"] = info_dict.get("commit_loss_weight", 0.25)
    # Entropy loss weight schedule
    start = info_dict.get("entropy_loss_weight_start")
    end = info_dict.get("entropy_loss_weight_end")
    decay_steps = info_dict.get("entropy_loss_decay_steps", 0)
    if decay_steps > 0 and start is not None and end is not None:
        progress = min(1.0, step / decay_steps)
        out["entropy_loss_weight"] = end + (start - end) * (1.0 - progress)
    else:
        out["entropy_loss_weight"] = info_dict.get("entropy_loss_weight", 0.0)
    return out


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    logging.info(
        "training on multiple gpus, this gpu {}, rank {}, world_size {}".format(
            local_rank, rank, world_size
        )
    )
    if args.train_engine == "torch_ddp":
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    else:
        raise NotImplementedError("only torch_ddp is supported")
    return world_size, local_rank, rank


def check_modify_and_save_config(args, configs):
    """Optionally adjust config for DDP; save to model_dir for reproducibility."""
    if args.train_engine == "torch_ddp":
        pass
    return configs


def wrap_cuda_model(args, model):
    assert torch.cuda.is_available()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    return model


def init_optimizer(args, configs, model):
    train_conf = configs["train_conf"]
    if train_conf["optim"] == "adam":
        optimizer = optim.Adam(model.parameters(), **train_conf["optim_conf"])
    elif train_conf["optim"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), **train_conf["optim_conf"])
    else:
        raise ValueError("unknown optimizer: " + str(train_conf["optim"]))
    return optimizer


def init_scheduler(args, configs, optimizer):
    train_conf = configs["train_conf"]
    sched = train_conf["scheduler"]
    sched_conf = train_conf.get("scheduler_conf", {})
    if sched == "warmuplr":
        scheduler = WarmupLR(optimizer, **sched_conf)
    elif sched == "constantlr":
        scheduler = ConstantLR(optimizer)
    else:
        raise ValueError("unknown scheduler: " + str(sched))
    return scheduler


def init_summarywriter(args):
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer = None
    return writer


def save_model_opt(model, optimizer, name, info_dict):
    rank = int(os.environ.get("RANK", 0))
    model_dir = info_dict["model_dir"]
    if rank != 0:
        return
    save_model_path = os.path.join(model_dir, "model_{}.pt".format(name))
    save_opt_path = os.path.join(model_dir, "opt_{}.pt".format(name))
    torch.save(model.module.state_dict(), save_model_path)
    torch.save(
        {
            "optim": optimizer.state_dict(),
            "step": info_dict["step"],
            "epoch": info_dict["epoch"],
        },
        save_opt_path,
    )
    info_path = re.sub(r"\.pt$", ".yaml", save_model_path)
    info_dict_copy = dict(info_dict)
    # Remove non-serializable objects before YAML dump
    info_dict_copy.pop("feature_extractor", None)
    info_dict_copy.pop("loss_dict", None)
    info_dict_copy.pop("_pred_mel", None)
    info_dict_copy.pop("_mel_recon_loss", None)
    info_dict_copy.pop("codes_for_hist", None)
    info_dict_copy["save_time"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    import yaml
    with open(info_path, "w") as fout:
        fout.write(yaml.dump(info_dict_copy))
    logging.info("[Rank {}] Checkpoint: save to checkpoint {}".format(rank, save_model_path))


def find_last_state(model_dir):
    """Return (model_path, opt_path, step, epoch) of latest step checkpoint.
    Only considers files matching model_epoch_N_step_M.pt so that epoch_whole
    checkpoints do not trigger resume (and we avoid asserting when no step ckpt)."""
    last_model_path = last_opt_path = last_step = last_epoch = None
    if not os.path.isdir(model_dir):
        return last_model_path, last_opt_path, last_step, last_epoch
    ckpt_path_list = [
        f
        for f in os.listdir(model_dir)
        if f.endswith(".pt") and f.startswith("model_") and "init" not in f
    ]
    step_style = [f for f in ckpt_path_list if re.match(r"^model_epoch_(\d+)_step_(\d+)\.pt$", f)]
    if not step_style:
        return last_model_path, last_opt_path, last_step, last_epoch

    def _parse(name):
        g = re.match(r"^model_epoch_([0-9]+)_step_([0-9]+)\.pt$", name)
        return (int(g.group(1)), int(g.group(2))) if g else (0, 0)

    last_name = max(step_style, key=lambda x: _parse(x)[1])
    last_epoch, last_step = _parse(last_name)
    last_model_path = os.path.join(model_dir, last_name)
    last_opt_path = os.path.join(model_dir, last_name.replace("model_", "opt_"))
    return last_model_path, last_opt_path, last_step, last_epoch


def sync_ddp_ranks_or_break_epoch(group_join, info_dict):
    """Barrier to sync DDP ranks each batch; return True to break epoch if uneven workload detected."""
    if info_dict["batch_idx"] != 0:
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.barrier(group=group_join, device_ids=[local_rank])
            return False
        except RuntimeError as e:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            logging.info(
                "Detected uneven workload: {}\nBreak to join; world_size {} rank {}".format(
                    e, world_size, rank
                )
            )
            return True
    return False


def compute_eval_mel_recon_loss(model, batch_dict, info_dict, n_steps=8):
    """Run one batch through decode_from_features (no grad), return L1 mel recon loss.
    Used for periodic eval logging when training the FM model (joint or fm_only).
    """
    device = torch.device("cuda:{}".format(int(os.environ.get("LOCAL_RANK", 0))))
    dtype = info_dict.get("dtype", "fp32")
    dtype = torch.float16 if dtype == "fp16" else torch.bfloat16 if dtype == "bf16" else torch.float32

    mel = batch_dict["mel"].to(device, dtype=dtype)
    mel_mask = batch_dict["mel_mask"].to(device)

    if "audio" in batch_dict:
        extractor = info_dict.get("feature_extractor")
        if extractor is None:
            return None
        audio = batch_dict["audio"].to(device)
        sr = batch_dict["sample_rate"][0].item()
        wf_l, wlf_l, mf_l = [], [], []
        with torch.no_grad():
            for i in range(audio.shape[0]):
                # T_mel from audio duration (50Hz) to align feature length; mel is GT only
                T_mel = int(audio.shape[1] / sr * 50)
                w, wl, m_ = extractor.extract_from_waveform(
                    audio[i], sr, T_mel=T_mel
                )
                wf_l.append(w)
                wlf_l.append(wl)
                mf_l.append(m_)
        whisper_feat = torch.stack(wf_l).to(dtype)
        wavlm_feat = torch.stack(wlf_l).to(dtype)
        muq_feat = torch.stack(mf_l).to(dtype)
    elif "whisper_feat" not in batch_dict:
        # Mel is GT only; cannot use mel for feature extraction. Need pre-computed features.
        logging.debug(
            "compute_eval_mel_recon_loss: skipping eval (no audio, no features; "
            "mel cannot be used for feature extraction)"
        )
        return None
    else:
        whisper_feat = batch_dict["whisper_feat"].to(device, dtype=dtype)
        wavlm_feat = batch_dict["wavlm_feat"].to(device, dtype=dtype)
        muq_feat = batch_dict["muq_feat"].to(device, dtype=dtype)

    m = model.module if hasattr(model, "module") else model
    m.eval()
    try:
        with torch.no_grad():
            pred_mel, _ = m.decode_from_features(
                whisper_feat, wavlm_feat, muq_feat,
                mel_mask=mel_mask,
                n_timesteps=n_steps,
                cfg=0.0,
            )
            loss = (
                F.l1_loss(pred_mel, mel, reduction="none").float() * mel_mask.unsqueeze(-1)
            ).sum() / (mel_mask.sum() * mel.shape[-1] + 1e-8)
            return loss.item()
    finally:
        m.train()


def batch_forward(model, batch, info_dict):
    """Codec: move batch to device, run model, compute flow_loss + commit_loss; optional mel_recon.
    Mel is used ONLY as ground truth for flow matching reconstruction.
    Batch must have pre-computed features (whisper_feat/wavlm_feat/muq_feat) or audio."""
    device = torch.device("cuda:{}".format(int(os.environ.get("LOCAL_RANK", 0))))
    dtype = info_dict.get("dtype", "fp32")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    mel = batch["mel"].to(device, dtype=dtype)
    mel_mask = batch["mel_mask"].to(device)

    if "whisper_feat" in batch:
        whisper_feat = batch["whisper_feat"].to(device, dtype=dtype)
        wavlm_feat = batch["wavlm_feat"].to(device, dtype=dtype)
        muq_feat = batch["muq_feat"].to(device, dtype=dtype)
        # Fix malformed shape from webdataset packing: (B, 1, T, D) -> (B, T, D)
        def _squeeze_to_3d(x):
            if x.dim() == 4:
                if x.numel() == 0:
                    return None  # corrupt batch, skip
                x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
            return x
        whisper_feat = _squeeze_to_3d(whisper_feat)
        wavlm_feat = _squeeze_to_3d(wavlm_feat)
        muq_feat = _squeeze_to_3d(muq_feat)
        if whisper_feat is None or wavlm_feat is None or muq_feat is None:
            return None  # corrupt batch
    elif "audio" in batch:
        extractor = info_dict.get("feature_extractor")
        if extractor is None:
            raise RuntimeError(
                "Batch has audio but info_dict has no feature_extractor "
                "(set feature_extractor config in dataset_conf)"
            )
        audio = batch["audio"].to(device)
        sr = batch["sample_rate"][0].item()
        wf_l, wlf_l, mf_l = [], [], []
        with torch.no_grad():
            for i in range(audio.shape[0]):
                # T_mel from audio duration (50Hz); mel is GT only
                T_mel = int(audio.shape[1] / sr * 50)
                w, wl, m_ = extractor.extract_from_waveform(
                    audio[i], sr, T_mel=T_mel
                )
                wf_l.append(w)
                wlf_l.append(wl)
                mf_l.append(m_)
        whisper_feat = torch.stack(wf_l).to(dtype)
        wavlm_feat = torch.stack(wlf_l).to(dtype)
        muq_feat = torch.stack(mf_l).to(dtype)
    else:
        raise RuntimeError(
            "Batch must contain either pre-computed features "
            "(whisper_feat/wavlm_feat/muq_feat) or audio"
        )
    mel_recon_weight = info_dict.get("mel_recon_weight", 0.0)
    mel_recon_n_steps = info_dict.get("mel_recon_n_steps", 4)
    need_pred_mel = mel_recon_weight > 0

    with torch.amp.autocast(enabled=(dtype != torch.float32), device_type="cuda", dtype=dtype):
        out = model(
            whisper_feat, wavlm_feat, muq_feat, mel, mel_mask,
            return_pred_mel=need_pred_mel,
            mel_recon_n_steps=mel_recon_n_steps,
        )
    noise, x, flow_pred, mask = out["cfm_output"]
    flow_gt = x - (1 - 1e-5) * noise
    flow_loss = (F.l1_loss(flow_pred, flow_gt, reduction="none").float() * mask).sum() / (
        mask.sum() * mel.shape[-1] + 1e-8
    )
    flow_loss_weight = info_dict.get("flow_loss_weight", 0.1)
    commit_loss = out["commit_loss"]
    commit_weight = info_dict.get("commit_loss_weight", 0.25)
    loss = flow_loss_weight * flow_loss + commit_weight * commit_loss
    codebook_loss = out.get("codebook_loss")
    if codebook_loss is not None:
        codebook_weight = info_dict.get("codebook_loss_weight", 1.0)
        loss = loss + codebook_weight * codebook_loss
    entropy_loss = out.get("entropy_loss")
    m = model.module if hasattr(model, "module") else model
    fm_only = getattr(m, "fm_only", False)
    if entropy_loss is not None and not fm_only:
        entropy_loss_weight = info_dict.get("entropy_loss_weight", getattr(m, "entropy_loss_weight", 0.0))
        if entropy_loss_weight > 0:
            loss = loss + entropy_loss_weight * entropy_loss

    pred_mel = out.get("pred_mel")
    mel_recon_loss = None
    if need_pred_mel and pred_mel is not None:
        mel_recon_loss = (F.l1_loss(pred_mel, mel, reduction="none") * mel_mask.unsqueeze(-1)).sum() / (
            mel_mask.sum() * mel.shape[-1] + 1e-8
        )
        loss = loss + mel_recon_weight * mel_recon_loss
        info_dict["_pred_mel"] = pred_mel
        info_dict["_mel_recon_loss"] = mel_recon_loss

    # Codebook utilization (skip when fm_only: no VQ, pre-VQ features go directly to CFM)
    loss_dict = {
        "total_loss": loss,
        "flow_matching_loss": flow_loss,
        "commit_loss": commit_loss,
    }
    if not fm_only and out["codes"] is not None:
        codes = out["codes"]
        if codes.dim() == 3 and getattr(m, "use_rvq", False):
            codebook_sizes = m.rvq.codebook_sizes
            utils = []
            for i in range(codes.shape[-1]):
                num_used = codes[..., i].unique().numel()
                utils.append(num_used / float(codebook_sizes[i]))
            codebook_util = sum(utils) / len(utils)
        else:
            codebook_size = m.codebook_size
            num_used = codes.unique().numel()
            codebook_util = num_used / float(codebook_size)
        loss_dict["codebook_util"] = torch.tensor(codebook_util, device=loss.device, dtype=torch.float32)
        info_dict["codes_for_hist"] = codes.detach().flatten()
    if codebook_loss is not None:
        loss_dict["codebook_loss"] = codebook_loss
    if entropy_loss is not None:
        loss_dict["entropy_loss"] = entropy_loss
    if mel_recon_loss is not None:
        loss_dict["mel_recon_loss"] = mel_recon_loss
    info_dict["loss_dict"] = loss_dict
    return info_dict


def batch_backward(model, info_dict):
    if info_dict["train_engine"] != "torch_ddp":
        raise NotImplementedError
    loss_dict = info_dict["loss_dict"]
    accum_grad = info_dict["accum_grad"]
    scaled_loss = loss_dict["total_loss"] / accum_grad
    scaled_loss.backward()
    loss_dict["total_loss"] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, info_dict):
    grad_norm = 0.0
    if (info_dict["batch_idx"] + 1) % info_dict["accum_grad"] == 0:
        grad_norm = clip_grad_norm_(model.parameters(), info_dict["grad_clip"])
        if torch.isfinite(grad_norm):
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]["lr"]
    info_dict["grad_norm"] = grad_norm
    return info_dict


def _loss_scalar(v):
    """Convert loss value to Python scalar for logging (tensor or float)."""
    return v.item() if torch.is_tensor(v) else v


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict["loss_dict"]
    rank = int(os.environ.get("RANK", 0))
    if writer is not None:
        if (info_dict["batch_idx"] + 1) % info_dict["accum_grad"] == 0:
            for k in ["epoch", "lr", "grad_norm"]:
                writer.add_scalar("{}/{}".format(tag, k), info_dict.get(k, 0), step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar("{}/{}".format(tag, k), _loss_scalar(v), step + 1)
            # Code usage histogram: which code indices are used (spread vs collapse)
            if "codes_for_hist" in info_dict:
                writer.add_histogram("vq/code_usage", info_dict["codes_for_hist"], step + 1)
                info_dict.pop("codes_for_hist", None)
    if (batch_idx + 1) % info_dict["log_interval"] == 0:
        log_str = "{} Batch {} ".format(tag, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += "{} {:.6f} ".format(name, _loss_scalar(value))
        if tag == "TRAIN":
            log_str += "lr {:.8f} grad_norm {:.6f}".format(
                info_dict["lr"], info_dict["grad_norm"]
            )
        log_str += " rank {}".format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    rank = int(os.environ.get("RANK", 0))
    _scalar = _loss_scalar
    logging.info(
        "Epoch {} Step {} CV info lr {} rank {} {}".format(
            epoch,
            step + 1,
            info_dict["lr"],
            rank,
            " ".join(["{}_{}".format(k, _scalar(v)) for k, v in loss_dict.items()]),
        )
    )
    if writer is not None:
        writer.add_scalar("{}/epoch".format(tag), epoch, step + 1)
        writer.add_scalar("{}/lr".format(tag), info_dict["lr"], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar("{}/{}".format(tag, k), _scalar(v), step + 1)
