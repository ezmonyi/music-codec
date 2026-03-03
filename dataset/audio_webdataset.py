"""
AudioWebDataset: read webdataset tar shards containing JSON metadata,
precomputed mel spectrogram, and pre-extracted whisper/wavlm/muq features.

Design: mel is used ONLY as ground truth for flow matching reconstruction.
Features (whisper/wavlm/muq) are conditioning; mel is never used to drive
feature extraction or alignment.

Tar sample structure:
    {id}.json                -- metadata: audio_filepath, audio_basename,
                                segment_index, segment_start_time, segment_end_time,
                                segment_duration, total_duration, sample_rate
    {id}.mel.npz             -- precomputed mel spectrogram (channels, mel_dim, T)
    {id}.whisper_feature.npy -- pre-extracted Whisper feature (T, 1280)
    {id}.wavlm_feature.npy  -- pre-extracted WavLM feature  (T, 1024)
    {id}.muq_feature.npy     -- pre-extracted MuQ feature    (T, 1024)
"""

import glob
import json
import logging
import os
import random
from io import BytesIO

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

try:
    import webdataset as wds
except ImportError:
    wds = None


def _decode_npz_mel(data):
    """Decode mel.npz or mel.npy bytes to tensor.
    npz yields NpzFile (dict-like); npy yields ndarray directly.
    """
    if data is None:
        return None
    buf = BytesIO(data) if isinstance(data, bytes) else data
    z = np.load(buf, allow_pickle=True)
    if isinstance(z, np.ndarray):
        x = z
    else:
        keys = list(z.keys())
        if not keys:
            return None
        x = z[keys[0]] if len(keys) == 1 else z.get("mel", z.get("arr_0", z[keys[0]]))
    return torch.from_numpy(np.asarray(x)).float()


def _decode_npy_feature(data):
    """Decode .npy bytes to tensor (T, D)."""
    if data is None:
        return None
    buf = BytesIO(data) if isinstance(data, bytes) else data
    arr = np.load(buf)
    return torch.from_numpy(np.asarray(arr)).float()


def _normalize_mel_shape(mel: torch.Tensor) -> torch.Tensor:
    """Normalize mel to (T, 128). Handles (C, mel_dim, T), (mel_dim, T), etc."""
    if mel.dim() == 3:
        # (channels, mel_dim, T) -> mean over channels -> (mel_dim, T)
        mel = mel.mean(dim=0)
    if mel.dim() == 2:
        if mel.shape[0] <= 200 and mel.shape[1] > mel.shape[0]:
            mel = mel.transpose(0, 1)
    if mel.dim() != 2:
        raise ValueError(f"mel must be 2D (T, mel_dim) after reshape, got {mel.shape}")
    return mel


class AudioWebDataset(IterableDataset):
    """Iterable dataset over tar shards with JSON metadata, mel.npz,
    and pre-extracted whisper/wavlm/muq features.

    Each sample yields::

        {"mel": (T, 128), "mel_mask": (T,),
         "whisper_feat": (T_w, 1280), "wavlm_feat": (T_wl, 1024),
         "muq_feat": (T_m, 1024)}
    """

    def __init__(
        self,
        urls,
        max_frames_50hz: int = 1500,
        seed: int = 42,
        handler=None,
    ):
        if wds is None:
            raise ImportError("webdataset is required; pip install webdataset")

        self.max_frames_50hz = max_frames_50hz
        self.rng = random.Random(seed)
        self.handler = handler or wds.handlers.warn_and_continue

        if isinstance(urls, str):
            if "*" in urls:
                expanded = sorted(glob.glob(urls))
                if not expanded:
                    raise ValueError(f"No shards found for pattern: {urls}")
                urls = expanded
            else:
                urls = [urls]
        self.urls = urls

        self.rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else int(os.environ.get("RANK", 0))
        )
        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", 1))
        )
        if self.rank == 0:
            logger.info(
                f"[AudioWebDataset] {len(self.urls)} shards, "
                f"world_size={self.world_size}"
            )

    def _decode_sample(self, sample):
        json_data = None
        mel_data = None
        whisper_data = None
        wavlm_data = None
        muq_data = None

        for key, val in sample.items():
            if key in ("__key__", "__url__"):
                continue
            if key.endswith(".json") or key == "json":
                json_data = val
            elif "mel" in key and (key.endswith(".npz") or key.endswith(".npy")):
                mel_data = val
            elif "whisper_feature" in key and key.endswith(".npy"):
                whisper_data = val
            elif "wavlm_feature" in key and key.endswith(".npy"):
                wavlm_data = val
            elif "muq_feature" in key and key.endswith(".npy"):
                muq_data = val

        if json_data is None:
            logger.warning("[AudioWebDataset] sample missing JSON metadata")
            return None

        if isinstance(json_data, bytes):
            json_data = json_data.decode("utf-8")
        if isinstance(json_data, str):
            try:
                json.loads(json_data)
            except json.JSONDecodeError as e:
                logger.warning(f"[AudioWebDataset] bad JSON: {e}")
                return None

        # --- decode mel ---
        mel = _decode_npz_mel(mel_data)
        if mel is None:
            logger.warning("[AudioWebDataset] failed to decode mel")
            return None
        mel = _normalize_mel_shape(mel)

        # --- decode pre-extracted features ---
        whisper_feat = _decode_npy_feature(whisper_data)
        wavlm_feat = _decode_npy_feature(wavlm_data)
        muq_feat = _decode_npy_feature(muq_data)
        if whisper_feat is None or wavlm_feat is None or muq_feat is None:
            logger.warning("[AudioWebDataset] missing one or more feature files")
            return None

        # --- length limit: truncate mel from start (mel is GT only; features used as-is) ---
        if self.max_frames_50hz > 0 and mel.shape[0] > self.max_frames_50hz:
            mel = mel[: self.max_frames_50hz]

        mel_mask = torch.ones(mel.shape[0], dtype=torch.float32)

        return {
            "mel": mel,
            "mel_mask": mel_mask,
            "whisper_feat": whisper_feat,
            "wavlm_feat": wavlm_feat,
            "muq_feat": muq_feat,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        urls = self.urls.copy()
        random.Random(self.rank * 1000 + worker_id).shuffle(urls)
        if self.world_size > 1:
            urls = [u for i, u in enumerate(urls) if i % self.world_size == self.rank]

        if worker_id == 0 and self.rank == 0:
            logger.info(
                f"[AudioWebDataset] Worker {worker_id} (rank {self.rank}): "
                f"{len(urls)} shards"
            )

        pipeline = wds.DataPipeline(
            wds.SimpleShardList(urls),
            wds.tarfile_to_samples(handler=self.handler),
            wds.shuffle(1000),
        )

        sample_count = 0
        for sample in pipeline:
            out = self._decode_sample(sample)
            if out is not None:
                sample_count += 1
                if sample_count == 1 and worker_id == 0 and self.rank == 0:
                    logger.info(
                        f"[AudioWebDataset] First sample: "
                        f"mel={out['mel'].shape}, "
                        f"whisper={out['whisper_feat'].shape}, "
                        f"wavlm={out['wavlm_feat'].shape}, "
                        f"muq={out['muq_feat'].shape}"
                    )
                yield out

    def __len__(self):
        return 0


class AudioCollateFn:
    """Pad mel and features to max length in batch, stack into tensors."""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch_list):
        mel_list = [b["mel"] for b in batch_list]
        mask_list = [b["mel_mask"] for b in batch_list]
        whisper_list = [b["whisper_feat"] for b in batch_list]
        wavlm_list = [b["wavlm_feat"] for b in batch_list]
        muq_list = [b["muq_feat"] for b in batch_list]

        def _time_dim(x):
            return x.shape[1] if x.dim() == 3 and x.shape[0] == 1 else x.shape[0]

        max_mel_t = max(x.shape[0] for x in mel_list)
        max_whisper_t = max(_time_dim(x) for x in whisper_list)
        max_wavlm_t = max(_time_dim(x) for x in wavlm_list)
        max_muq_t = max(_time_dim(x) for x in muq_list)

        def _pad_2d(x, max_t, pad_val=0.0):
            if x.dim() != 2:
                raise ValueError(f"_pad_2d expects 2D (T, D), got {x.shape}")
            if x.shape[0] < max_t:
                pad = torch.full(
                    (max_t - x.shape[0], x.shape[1]), pad_val, dtype=x.dtype,
                )
                return torch.cat([x, pad], dim=0)
            return x[:max_t]

        def _pad_1d(x, max_t):
            if x.shape[0] < max_t:
                pad = torch.zeros(max_t - x.shape[0], dtype=x.dtype)
                return torch.cat([x, pad], dim=0)
            return x[:max_t]

        def _pad_feat(x, max_t, pad_val=0.0):
            """(T, D) or (1, T, D) -> pad time dim -> (1, max_t, D) for concat."""
            if x.dim() == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            if x.shape[0] < max_t:
                pad = torch.full((max_t - x.shape[0], x.shape[1]), pad_val, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[:max_t]
            return x.unsqueeze(0)

        mel = torch.stack([_pad_2d(x, max_mel_t, self.pad_value) for x in mel_list])
        mel_mask = torch.stack([_pad_1d(x, max_mel_t) for x in mask_list])
        whisper_feat = torch.cat([_pad_feat(x, max_whisper_t) for x in whisper_list], dim=0)
        wavlm_feat = torch.cat([_pad_feat(x, max_wavlm_t) for x in wavlm_list], dim=0)
        muq_feat = torch.cat([_pad_feat(x, max_muq_t) for x in muq_list], dim=0)

        return {
            "mel": mel,
            "mel_mask": mel_mask,
            "whisper_feat": whisper_feat,
            "wavlm_feat": wavlm_feat,
            "muq_feat": muq_feat,
        }


def init_dataset_and_dataloader(args, configs):
    """Build AudioWebDataset + DataLoader from tar shards for train and optionally cv.
    Requires dataset_conf urls (or webdataset_path + shard_pattern).
    Optional: cv_ratio (float, e.g. 0.05) to use the last N% of shards as validation;
    or cv_num_shards (int) to use the last N shards for cv. Train uses the rest.
    Returns (audio_webdataset, cv_dataset, train_loader, cv_loader); cv can be None.
    """
    import yaml as _yaml

    dataset_conf = configs.get("dataset_conf", {})
    if isinstance(dataset_conf, str):
        with open(dataset_conf, "r") as f:
            dataset_conf = _yaml.safe_load(f) or {}
    train_conf = configs.get("train_conf", {})
    max_frames = dataset_conf.get("max_frames_50hz", 1500)
    seed = dataset_conf.get("seed", 42)
    batch_size = train_conf.get("batch_size", 8)
    cv_batch_size = dataset_conf.get("cv_batch_size", batch_size)

    urls = dataset_conf.get("urls")
    webdataset_path = dataset_conf.get("webdataset_path")
    shard_pattern = dataset_conf.get("shard_pattern")
    if not (urls or (webdataset_path and shard_pattern)):
        raise ValueError(
            "dataset_conf: set urls or (webdataset_path + shard_pattern) for tar shards."
        )

    if urls is None:
        base = webdataset_path.rstrip("/")
        urls = os.path.join(base, shard_pattern)

    # Normalize to list so we can split train/cv from the same shards
    if isinstance(urls, str):
        if "*" in urls:
            all_urls = sorted(glob.glob(urls))
            if not all_urls:
                raise ValueError(f"No shards found for pattern: {urls}")
        else:
            all_urls = [urls]
    else:
        all_urls = list(urls)

    cv_ratio = dataset_conf.get("cv_ratio", 0.0)
    cv_num_shards = dataset_conf.get("cv_num_shards")
    if cv_num_shards is not None:
        n_cv = min(max(0, int(cv_num_shards)), len(all_urls) - 1)
    elif cv_ratio > 0 and len(all_urls) >= 2:
        n_cv = min(max(1, int(len(all_urls) * cv_ratio)), len(all_urls) - 1)
    else:
        n_cv = 0

    if n_cv > 0:
        train_urls, cv_urls = all_urls[:-n_cv], all_urls[-n_cv:]
        logging.info(
            "[DataLoader] Split shards: train=%d, cv=%d (from same urls)",
            len(train_urls),
            len(cv_urls),
        )
    else:
        train_urls = all_urls
        cv_urls = []

    train_dataset = AudioWebDataset(
        urls=train_urls,
        max_frames_50hz=max_frames,
        seed=seed,
    )
    num_workers_val = getattr(args, "num_workers", 4)
    audio_collate_fn = AudioCollateFn(pad_value=0.0)
    logging.info(
        f"[DataLoader] AudioWebDataset: batch_size={batch_size}, "
        f"num_workers={num_workers_val}"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers_val,
        collate_fn=audio_collate_fn,
        pin_memory=getattr(args, "pin_memory", False),
        prefetch_factor=getattr(args, "prefetch", 2) if num_workers_val > 0 else None,
        drop_last=True,
    )
    logging.info("[DataLoader] AudioWebDataset DataLoader created")

    val_dataset = None
    val_loader = None
    if cv_urls:
        val_dataset = AudioWebDataset(
            urls=cv_urls,
            max_frames_50hz=max_frames,
            seed=seed + 1,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cv_batch_size,
            num_workers=num_workers_val,
            collate_fn=AudioCollateFn(pad_value=0.0),
            pin_memory=getattr(args, "pin_memory", False),
            prefetch_factor=getattr(args, "prefetch", 2) if num_workers_val > 0 else None,
            drop_last=False,
        )
        logging.info(
            "[DataLoader] CV AudioWebDataset DataLoader created (batch_size=%s, %d shards)",
            cv_batch_size,
            len(cv_urls),
        )

    return train_dataset, val_dataset, train_loader, val_loader
