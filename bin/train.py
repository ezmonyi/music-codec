"""
Codec training: load_hyperpyyaml, init_distributed, dataset, model, resume,
DDP, optimizer/scheduler, Executor loop.

We use native torch DDP (not PyTorch Lightning) to match flowmatching2/cosyvoice:
explicit checkpoint format (model_*.pt / opt_*.pt), resume from latest step,
and full control over the training loop. Lightning would require wrapping the
model in LightningModule and adapting save/load; DDP keeps the pipeline aligned
with your existing cosyvoice scripts.
"""

from __future__ import print_function

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
from dataset.codec_dataset import init_dataset_and_dataloader


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
    parser.add_argument("--local-rank", help="placeholder for DDP")
    parser.add_argument("--dataset_conf", required=True, help="dataset yaml or manifest path")
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument(
        "--restore_model_path",
        type=str,
        default=None,
        help="restore from this ckpt (overrides resume from model_dir)",
    )
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

    if getattr(args, "restore_model_path", None) and os.path.isfile(args.restore_model_path):
        ckpt = torch.load(args.restore_model_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        print("Restore model from {}".format(args.restore_model_path))

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
    info_dict["step"] = last_step
    info_dict["epoch"] = last_epoch
    train_conf = configs["train_conf"]
    mel_recon_weight = train_conf.get("mel_recon_weight", 0.0)
    use_disc = train_conf.get("use_disc", False)
    if use_disc:
        from utils.balancer import Balancer
        bw = train_conf.get("balancer_weights", {"mel_recon": 1.0, "disc_gen": 1.0})
        info_dict["balancer"] = Balancer(
            bw,
            rescale_grads=train_conf.get("balancer_rescale_grads", True),
            total_norm=train_conf.get("balancer_total_norm", 1.0),
            ema_decay=train_conf.get("balancer_ema_decay", 0.999),
        )
    else:
        info_dict["balancer"] = None
    if use_disc:
        from utils.ms_stft_disc import MultiScaleSTFTDiscriminator
        disc = MultiScaleSTFTDiscriminator(
            filters=train_conf.get("disc_filters", 32),
            n_ffts=train_conf.get("disc_n_ffts", [1024, 2048, 512]),
            hop_lengths=train_conf.get("disc_hop_lengths", [256, 512, 128]),
            win_lengths=train_conf.get("disc_win_lengths", [1024, 2048, 512]),
        )
        disc = disc.cuda().train()
        info_dict["discriminator"] = disc
        info_dict["D_optimizer"] = optim.Adam(disc.parameters(), lr=train_conf.get("disc_lr", 1e-4))
    else:
        info_dict["discriminator"] = None
        info_dict["D_optimizer"] = None
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
