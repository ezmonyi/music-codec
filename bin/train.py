"""
Codec training: load_hyperpyyaml, init_distributed, dataset, model, resume,
DDP, optimizer/scheduler, Executor loop.

We use native torch DDP (not PyTorch Lightning) to match flowmatching2/cosyvoice:
explicit checkpoint format (model_*.pt / opt_*.pt), resume from latest step,
and full control over the training loop. Lightning would require wrapping the
model in LightningModule and adapting save/load; DDP keeps the pipeline aligned
with your existing cosyvoice scripts.
"""

import os
import sys
import argparse
import datetime
import logging
import warnings
from copy import deepcopy

# Suppress FutureWarning for torch.nn.utils.weight_norm deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

import torch
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml

# Repo root = parent of bin/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch.optim as optim
from utils.train_utils import (
    init_distributed,
    init_optimizer,
    init_scheduler,
    init_summarywriter,
    wrap_cuda_model,
    save_model_opt,
    find_last_state,
    check_modify_and_save_config,
)
from utils.executor import Executor
from dataset.audio_webdataset import init_dataset_and_dataloader


def get_args():
    parser = argparse.ArgumentParser(description="Codec training")
    parser.add_argument(
        "--train_engine",
        default="torch_ddp",
        choices=["torch_ddp"],
        help="Engine for paralleled training",
    )
    parser.add_argument("--model", default="model", help="model key in config (must match yaml)")
    parser.add_argument("--config", required=True, help="config yaml (hyperpyyaml)")
    parser.add_argument("--local-rank", "--local_rank", help="placeholder for DDP (set by torchrun/launch)")
    parser.add_argument("--dataset_conf", default=None, help="dataset yaml or manifest path (optional; dataset_conf in training yaml is used by default)")
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument("--tensorboard_dir", default="tensorboard", help="tensorboard log dir")
    parser.add_argument(
        "--ddp.dist_backend",
        dest="dist_backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="distributed backend",
    )
    parser.add_argument(
        "--timeout",
        default=60,
        type=int,
        help="timeout (seconds) for DDP join",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="override train_conf batch_size")
    parser.add_argument("--num_workers", default=4, type=int, help="dataloader workers")
    parser.add_argument("--prefetch", default=2, type=int, help="prefetch factor")
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--overrides", default=None, help="YAML overrides string prepended to config (e.g. 'estimator_type: llama')")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # DataLoader workers run mel_to_features on CUDA; must use spawn, not fork
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f, overrides=args.overrides)

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
        if args.dataset_conf.endswith(".yaml") or args.dataset_conf.endswith(".yml"):
            import yaml
            with open(args.dataset_conf, "r") as f:
                configs["dataset_conf"].update(yaml.safe_load(f) or {})
        else:
            configs["dataset_conf"]["manifest_path"] = args.dataset_conf

    init_distributed(args)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        logging.info(f"[Rank {rank}] Initializing dataset and dataloader...")
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = init_dataset_and_dataloader(
        args, configs
    )
    if rank == 0:
        logging.info(f"[Rank {rank}] Dataset and dataloader initialized successfully")

    configs = check_modify_and_save_config(args, configs)

    writer = init_summarywriter(args)

    model = configs[args.model]
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        nparams = sum(p.numel() for p in model.parameters())
        print("Model parameters: {:.3f}M".format(nparams / 1e6))

    last_model_path, last_opt_path, last_step, last_epoch = find_last_state(args.model_dir)
    if last_model_path is not None and os.path.isfile(last_model_path):
        ckpt = torch.load(last_model_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        print(
            "Resume model from {} at epoch {} step {}".format(
                last_model_path, last_epoch, last_step
            )
        )

    restore_model_path = configs.get("train_conf", {}).get("restore_model_path")
    if restore_model_path and os.path.isfile(restore_model_path):
        ckpt = torch.load(restore_model_path, map_location="cpu")
        model_state = model.state_dict()
        loaded_keys, skipped_keys = [], []
        for k, v in ckpt.items():
            if k not in model_state:
                skipped_keys.append(k)
                continue
            if model_state[k].shape != v.shape:
                skipped_keys.append(k)
                continue
            model_state[k] = v
            loaded_keys.append(k)
        model.load_state_dict(model_state, strict=False)
        print("Restore from {} ({} loaded, {} skipped)".format(
            restore_model_path, len(loaded_keys), len(skipped_keys)))
        if loaded_keys:
            print("  Loaded: {}".format(", ".join(loaded_keys[:10]) + ("..." if len(loaded_keys) > 10 else "")))
        if skipped_keys:
            print("  Skipped: {}".format(", ".join(skipped_keys[:10]) + ("..." if len(skipped_keys) > 10 else "")))

    model = wrap_cuda_model(args, model)

    optimizer = init_optimizer(args, configs, model)
    if last_opt_path is not None and os.path.isfile(last_opt_path):
        opt_ckpt = torch.load(last_opt_path, map_location="cpu")
        optimizer.load_state_dict(opt_ckpt["optim"])
        print(
            "Resume optimizer from {} at epoch {} step {}".format(
                last_opt_path, last_epoch, last_step
            )
        )

    sched_conf = configs["train_conf"].get("scheduler_conf", {})
    if last_epoch is not None:
        sched_conf["last_epoch"] = last_epoch
    else:
        sched_conf["last_epoch"] = -1
    configs["train_conf"]["scheduler_conf"] = sched_conf
    scheduler = init_scheduler(args, configs, optimizer)

    last_step = last_step if last_step is not None else 0
    last_epoch = last_epoch if last_epoch is not None else 0
    print("last_step: {}, last_epoch: {}".format(last_step, last_epoch))

    info_dict = deepcopy(configs["train_conf"])
    # Expose dataset statistics (e.g., mel mean/std) to training loop via info_dict
    # so that batch_forward can normalize mel before FM noising / loss.
    if "stat" in configs and isinstance(configs["stat"], dict):
        info_dict.update(configs["stat"])
    info_dict["step"] = last_step
    info_dict["epoch"] = last_epoch

    # Load EAR_VAE for earvae_latent mode (encode waveform -> latent on GPU)
    ear_vae_conf = configs.get("ear_vae")
    if ear_vae_conf is not None:
        source_dir = ear_vae_conf.get("source_dir")
        if source_dir and source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("ear_vae requires huggingface_hub; pip install huggingface_hub") from e
        import json as _json
        config_path = os.path.join(source_dir or ".", ear_vae_conf.get("config_json", "config/ear_vae_v2.json"))
        with open(config_path, "r") as f:
            vae_config = _json.load(f)
        ckpt_path = hf_hub_download(
            repo_id=ear_vae_conf.get("hf_repo", "earlab/EAR_VAE"),
            filename=ear_vae_conf.get("hf_filename", "pretrained_weight/ear_vae_v2_48k.pyt"),
        )
        from model.ear_vae import EAR_VAE
        ear_vae = EAR_VAE(model_config=vae_config)
        ear_vae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        ear_vae.eval()
        for p in ear_vae.parameters():
            p.requires_grad = False
        ear_vae.cuda()
        info_dict["ear_vae_model"] = ear_vae
        info_dict["ear_vae_downsample_ratio"] = int(ear_vae_conf.get("downsampling_ratio", 960))
        if rank == 0:
            logging.info(f"[Rank {rank}] Loaded EAR_VAE from {ckpt_path}")

    save_model_opt(model, optimizer, "init", info_dict)

    executor = Executor()
    executor.step = last_step

    rank = int(os.environ.get("RANK", 0))
    logging.info(f"[Rank {rank}] Starting training loop: epochs {last_epoch} to {info_dict['max_epoch']-1}, "
                f"starting from step {last_step}")

    for epoch in range(last_epoch, info_dict["max_epoch"]):
        executor.epoch = epoch
        if rank == 0:
            logging.info(f"[Rank {rank}] Starting epoch {epoch}/{info_dict['max_epoch']-1}")
        if hasattr(train_data_loader, "sampler") and hasattr(
            train_data_loader.sampler, "set_epoch"
        ):
            train_data_loader.sampler.set_epoch(epoch)
        dist.barrier()
        if rank == 0:
            logging.info(f"[Rank {rank}] Creating DDP group for epoch {epoch}")
        group_join = dist.new_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=args.timeout),
        )
        if rank == 0:
            logging.info(f"[Rank {rank}] Entering train_one_epoc for epoch {epoch}")
        executor.train_one_epoc(
            model,
            optimizer,
            scheduler,
            train_data_loader,
            cv_data_loader,
            writer,
            info_dict,
            group_join,
        )
        if rank == 0:
            logging.info(f"[Rank {rank}] Completed epoch {epoch}")
        dist.destroy_process_group(group_join)


if __name__ == "__main__":
    main()
