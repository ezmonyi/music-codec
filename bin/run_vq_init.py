"""
K-means VQ codebook initialization.

Uses a dataset of pre-VQ features (concat of Whisper/WavLM/MuQ through frozen
resamplers + in_proj), runs K-means with K=codebook_size, and initializes the
single-VQ codebook with the centroids. Saves the model checkpoint with the new
codebook so training can start from it.

Single process (from repo root):
  PYTHONPATH=$PWD python bin/run_vq_init.py \\
    --config conf/single_vq.yaml \\
    --dataset_conf /path/to/manifest.jsonl \\
    --out exp/vq_init_ckpt/model_0.pt

8-GPU distributed (abundant data; each GPU collects a shard, rank 0 runs K-means):
  PYTHONPATH=$PWD torchrun --nproc_per_node=8 bin/run_vq_init.py \\
    --config conf/single_vq.yaml \\
    --dataset_conf /path/to/manifest.jsonl \\
    --out /mnt/yi-jfs/checkpoints/codec/vq_init/model_0.pt \\
    --max_samples 500000 \\
    --max_vectors 5000000 \\
    --kmeans_max_iter 100
"""

from __future__ import print_function

import argparse
import glob
import logging
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hyperpyyaml import load_hyperpyyaml
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from dataset.codec_dataset import CodecDataset, CodecCollateFn, CodecWebDataset


def get_args():
    p = argparse.ArgumentParser(description="K-means VQ codebook initialization")
    p.add_argument("--config", required=True, help="Hyperpyyaml config (e.g. conf/single_vq.yaml)")
    p.add_argument("--dataset_conf", required=True, help="Dataset manifest JSONL or dataset yaml with manifest_path")
    p.add_argument("--out", required=True, help="Output path: model state_dict or full checkpoint .pt")
    p.add_argument("--max_samples", type=int, default=200000, help="Max dataset samples to use (default 200000)")
    p.add_argument("--max_vectors", type=int, default=2_000_000, help="Max vectors for K-means (subsampled); for 8-GPU abundant use e.g. 10000000")
    p.add_argument("--max_batches", type=int, default=None, help="Stop after this many batches (default: no limit)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    p.add_argument("--kmeans_max_iter", type=int, default=100, help="K-means max iterations (default 100)")
    p.add_argument("--kmeans_n_init", type=int, default=1, help="K-means n_init (default 1)")
    p.add_argument("--kmeans_method", type=str, default="minibatch", choices=["minibatch", "full"], help="K-means method: minibatch (faster) or full (slower, more accurate)")
    p.add_argument("--kmeans_batch_size", type=int, default=50000, help="MiniBatchKMeans batch_size (default 50000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for model forward (single-process only)")
    p.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (default 0 single-process, 4 distributed)")
    p.add_argument("--chunk_dir", type=str, default=None, help="Dir for per-rank chunk files (default: dirname(out)/vq_init_chunks)")
    p.add_argument("--keep_chunks", action="store_true", help="Keep chunk files after run (for debugging)")
    return p.parse_args()


def init_distributed(logger):
    """Initialize process group when WORLD_SIZE > 1. Returns (world_size, rank, local_rank, device)."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        device = torch.device("cuda", local_rank)
        logger.info("Distributed: rank {} local_rank {} world_size {} device {}".format(rank, local_rank, world_size, device))
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return world_size, rank, local_rank, device


def load_config_and_model(config_path, device):
    with open(config_path, "r") as f:
        config_str = f.read()
    configs = load_hyperpyyaml(config_str, overrides=None)
    model = configs["model"]
    if hasattr(model, "module"):
        model = model.module
    model = model.to(device)
    return configs, model


def freeze_encoder_for_init(model):
    """Freeze resamplers and in_proj so we only use them to get pre-VQ features."""
    for name, p in model.named_parameters():
        if "whisper_resample" in name or "wavlm_resample" in name or "in_proj" in name:
            p.requires_grad = False


def build_dataloader(dataset_conf_arg, configs, max_samples, batch_size, seed, world_size=1, rank=0, num_workers=0):
    dataset_conf = dict(configs.get("dataset_conf", {}))
    if isinstance(dataset_conf, str):
        with open(dataset_conf, "r") as f:
            dataset_conf = yaml.safe_load(f) or {}
    if os.path.isfile(dataset_conf_arg):
        if dataset_conf_arg.endswith(".yaml") or dataset_conf_arg.endswith(".yml"):
            with open(dataset_conf_arg, "r") as f:
                dataset_conf.update(yaml.safe_load(f) or {})
        else:
            dataset_conf["manifest_path"] = dataset_conf_arg

    urls = dataset_conf.get("urls")
    webdataset_path = dataset_conf.get("webdataset_path")
    shard_pattern = dataset_conf.get("shard_pattern")
    if urls or (webdataset_path and shard_pattern):
        # WebDataset (tar shards). Use extract_on_gpu=True so dataset returns only mel (fast I/O);
        # each rank runs feature extraction on its GPU in the collection loop (parallel across GPUs).
        if urls is None:
            base = webdataset_path.rstrip("/")
            urls = os.path.join(base, shard_pattern)
        if isinstance(urls, str) and "*" in urls:
            expanded = sorted(glob.glob(urls))
            if not expanded:
                raise ValueError("No shards found for pattern: {}".format(urls))
            urls = expanded
        elif isinstance(urls, str):
            urls = [urls]
        max_frames = dataset_conf.get("max_frames_50hz", 1500)
        use_mel_extractor = dataset_conf.get("use_mel_extractor", False)
        ds = CodecWebDataset(
            urls=urls,
            max_frames_50hz=max_frames,
            seed=seed,
            feature_keys=dataset_conf.get("feature_keys"),
            use_mel_extractor=use_mel_extractor,
            feature_extractor_conf=None,
            extract_on_gpu=True,
        )
        collate = CodecCollateFn(pad_value=0.0)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=False,
        )
        return loader, None, use_mel_extractor, dataset_conf

    manifest_path = dataset_conf.get("manifest_path")
    if not manifest_path or not os.path.isfile(manifest_path):
        raise ValueError(
            "dataset_conf must provide urls (webdataset) or a valid manifest_path (JSONL). Got manifest_path: {!r}".format(manifest_path)
        )
    max_frames = dataset_conf.get("max_frames_50hz", 1500)
    ds = CodecDataset(
        manifest_path=manifest_path,
        max_frames_50hz=max_frames,
        random_crop=True,
        seed=seed,
    )
    n_total = len(ds)
    if max_samples is not None and n_total > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=max_samples, replace=False)
        ds = Subset(ds, indices.tolist())
        n_total = max_samples
    collate = CodecCollateFn(pad_value=0.0)
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=False,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=False,
        )
    return loader, n_total, False, None


def collect_pre_vq_features(model, loader, device, max_vectors, max_batches, seed, logger, extractor=None, rank=0):
    """Collect pre-VQ features (z_e). If extractor is set and batch has mel only, run extraction on device (GPU)."""
    model.eval()
    list_ze = []
    n_vectors = 0
    rng = np.random.default_rng(seed)
    start = time.time()
    batch_idx = -1
    # total=None avoids len(loader); suppress IterableDataset length warning during iteration
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*", category=UserWarning)
        iterator = tqdm(loader, desc="Loading features", unit="batch", total=None, disable=(rank != 0), dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                if "whisper_feat" in batch:
                    w = batch["whisper_feat"].to(device)
                    wl = batch["wavlm_feat"].to(device)
                    m = batch["muq_feat"].to(device)
                else:
                    if extractor is None:
                        raise RuntimeError("Batch has mel only but no feature extractor provided.")
                    mel = batch["mel"].to(device)
                    w, wl, m = extractor.extract_batch(mel)
                z_e = model.get_pre_vq_features(w, wl, m)
                z_e_flat = z_e.reshape(-1, z_e.shape[-1]).cpu().numpy()
                n_new = z_e_flat.shape[0]
                if n_vectors + n_new <= max_vectors:
                    list_ze.append(z_e_flat)
                    n_vectors += n_new
                else:
                    need = max_vectors - n_vectors
                    if need > 0:
                        idx = rng.choice(n_new, size=need, replace=False)
                        list_ze.append(z_e_flat[idx])
                    n_vectors = max_vectors
                    break
                if rank == 0:
                    iterator.set_postfix(vectors=n_vectors, refresh=False)
    if list_ze:
        X = np.concatenate(list_ze, axis=0)
    else:
        X = np.zeros((0, 0), dtype=np.float32)
    elapsed = time.time() - start
    logger.info("Collected {} vectors in {:.1f}s ({} batches)".format(X.shape[0], elapsed, batch_idx + 1))
    return X


def save_chunk(chunk_dir, rank, X, logger):
    """Write per-rank feature chunk to disk."""
    os.makedirs(chunk_dir, exist_ok=True)
    path = os.path.join(chunk_dir, "vq_init_chunk_rank{}.npy".format(rank))
    np.save(path, X)
    logger.info("Saved chunk rank {} shape {} to {}".format(rank, X.shape, path))


def load_and_concat_chunks(chunk_dir, world_size, max_vectors, seed, logger):
    """Load all rank chunks, concatenate, optionally subsample to max_vectors. Returns X (n, dim)."""
    list_X = []
    for r in range(world_size):
        path = os.path.join(chunk_dir, "vq_init_chunk_rank{}.npy".format(r))
        if not os.path.isfile(path):
            raise FileNotFoundError("Chunk file not found: {}".format(path))
        X_r = np.load(path)
        list_X.append(X_r)
    X = np.concatenate(list_X, axis=0)
    if X.shape[0] > max_vectors:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_vectors, replace=False)
        X = X[idx]
        logger.info("Subsampled to {} vectors".format(max_vectors))
    return X


def run_kmeans(X, codebook_size, max_iter, n_init, seed, logger, method="minibatch", batch_size=50000):
    """Run K-means (mini-batch or full) with progress bar; return centroids (codebook_size, dim)."""
    if X.shape[0] < codebook_size:
        raise ValueError(
            "Not enough vectors for K-means: have {} but codebook_size is {}. "
            "Increase max_samples / max_vectors or use a larger dataset.".format(X.shape[0], codebook_size)
        )
    logger.info("Running K-means ({}): K={}, n_samples={}, dim={}, max_iter={}, n_init={}".format(
        method, codebook_size, X.shape[0], X.shape[1], max_iter, n_init))
    start = time.time()
    
    if method == "full":
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=codebook_size,
            init="k-means++",
            max_iter=max_iter,
            n_init=n_init,
            random_state=seed,
            verbose=0,
        )
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_.astype(np.float32)
    else:
        # MiniBatchKMeans with progress bar
        mbk = MiniBatchKMeans(
            n_clusters=codebook_size,
            init="k-means++",
            max_iter=max_iter,
            batch_size=min(batch_size, X.shape[0]),
            random_state=seed,
            n_init=n_init,
        )
        n_chunks = (X.shape[0] + batch_size - 1) // batch_size
        ranges = [(i * batch_size, min((i + 1) * batch_size, X.shape[0])) for i in range(n_chunks)]
        for start_idx, end_idx in tqdm(ranges, desc="K-means centroids", unit="chunk", dynamic_ncols=True):
            mbk.partial_fit(X[start_idx:end_idx])
        centroids = mbk.cluster_centers_.astype(np.float32)
    
    elapsed = time.time() - start
    logger.info("K-means done in {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))
    return centroids


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world_size, rank, local_rank, device = init_distributed(logger)
    if world_size == 1:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = args.num_workers if args.num_workers is not None else (4 if world_size > 1 else 0)
    chunk_dir = args.chunk_dir
    if chunk_dir is None:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        chunk_dir = os.path.join(out_dir, "vq_init_chunks") if out_dir else "vq_init_chunks"

    logger.info("Loading config and model from {}".format(args.config))
    configs, model = load_config_and_model(args.config, device)

    if getattr(model, "use_rvq", False):
        raise ValueError("This script only supports single-VQ codebook init. Model is RVQ.")

    freeze_encoder_for_init(model)
    codebook_size = model.codebook_size
    codebook_dim = model.codebook_dim

    logger.info("Building dataloader (max_samples={}, max_vectors={}, world_size={}, num_workers={})".format(
        args.max_samples, args.max_vectors, world_size, num_workers))
    loader, n_samples, use_mel_extractor_gpu, dataset_conf_for_extractor = build_dataloader(
        args.dataset_conf, configs, args.max_samples, args.batch_size, args.seed,
        world_size=world_size, rank=rank, num_workers=num_workers,
    )
    logger.info("Dataset size: {} (this rank sees a shard)".format(n_samples if n_samples is not None else "WebDataset (iterable)"))

    extractor = None
    if use_mel_extractor_gpu and dataset_conf_for_extractor is not None:
        from dataset.mel_to_features import CodecFeatureExtractor
        fe_conf = dict(dataset_conf_for_extractor.get("feature_extractor", {}))
        fe_conf["device"] = str(device)
        if fe_conf.get("wavlm_ckpt") == "":
            fe_conf["wavlm_ckpt"] = None
        extractor = CodecFeatureExtractor(**fe_conf)
        logger.info("Feature extractor on {} for GPU extraction".format(device))

    if world_size > 1:
        max_vectors_per_rank = args.max_vectors // world_size
        X = collect_pre_vq_features(
            model, loader, device, max_vectors_per_rank, args.max_batches, args.seed, logger,
            extractor=extractor, rank=rank,
        )
        if X.shape[0] == 0:
            logger.warning("Rank {} collected 0 vectors".format(rank))
        save_chunk(chunk_dir, rank, X, logger)
        dist.barrier()

        # After barrier: other ranks exit; rank 0 continues alone (K-means is CPU-only, no NCCL needed)
        # This avoids NCCL timeout while rank 0 does K-means (~10+ minutes)
        if rank != 0:
            dist.destroy_process_group()
            return

        # Rank 0: destroy process group before CPU-bound K-means to avoid NCCL timeout
        dist.destroy_process_group()
        logger.info("Process group destroyed; rank 0 continuing with K-means (CPU-only)")

        X = load_and_concat_chunks(chunk_dir, world_size, args.max_vectors, args.seed, logger)
        if X.shape[0] == 0:
            raise RuntimeError("No features in chunks. Check dataset and batch size.")
        centroids = run_kmeans(
            X, codebook_size, args.kmeans_max_iter, args.kmeans_n_init, args.seed, logger,
            method=args.kmeans_method, batch_size=args.kmeans_batch_size,
        )
        with torch.no_grad():
            model.vq_codebook.weight.data = torch.from_numpy(centroids).to(device, dtype=model.vq_codebook.weight.dtype)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        # Save only codebook weights (not conv1d/MLP - keep them random for training)
        codebook_state = {"vq_codebook.weight": model.vq_codebook.weight.data.cpu()}
        torch.save(codebook_state, args.out)
        logger.info("Saved codebook weights (only) with K-means-initialized centroids to {}".format(args.out))
        if not args.keep_chunks:
            for r in range(world_size):
                path = os.path.join(chunk_dir, "vq_init_chunk_rank{}.npy".format(r))
                if os.path.isfile(path):
                    os.remove(path)
            try:
                os.rmdir(chunk_dir)
            except OSError:
                pass
        return

    # Single-process path
    X = collect_pre_vq_features(
        model, loader, device, args.max_vectors, args.max_batches, args.seed, logger,
        extractor=extractor, rank=rank,
    )
    if X.shape[0] == 0:
        raise RuntimeError("No features collected. Check dataset and batch size.")

    centroids = run_kmeans(
        X, codebook_size, args.kmeans_max_iter, args.kmeans_n_init, args.seed, logger,
        method=args.kmeans_method, batch_size=args.kmeans_batch_size,
    )

    with torch.no_grad():
        model.vq_codebook.weight.data = torch.from_numpy(centroids).to(device, dtype=model.vq_codebook.weight.dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    state = model.state_dict()
    torch.save(state, args.out)
    logger.info("Saved model state_dict with K-means-initialized codebook to {}".format(args.out))


if __name__ == "__main__":
    main()
