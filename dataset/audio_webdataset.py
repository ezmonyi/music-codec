"""
AudioWebDataset: read webdataset tar shards containing JSON metadata + mel.npz,
fetch audio waveform from S3 via audio_filepath, and return raw audio + mel.

Tar sample structure:
    {id}.json       -- metadata: audio_filepath, segment_start_time, segment_end_time,
                       sample_rate, hop_length, mel_shape, ...
    {id}.mel.npz    -- precomputed mel spectrogram (channels, mel_dim, T)

Feature extraction (whisper/wavlm/muq) is NOT done here; the training loop
runs CodecFeatureExtractor.extract_from_waveform() on GPU.
"""

import glob
import json
import logging
import os
import random
import threading
from collections import OrderedDict
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


class _AudioLRUCache:
    """Thread-safe LRU cache for S3 audio downloads.

    Each DataLoader worker should hold its own instance so that the cache
    is not shared across processes (fork-safe).
    """

    def __init__(self, maxsize: int = 64):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, filepath: str) -> torch.Tensor:
        with self._lock:
            if filepath in self._cache:
                self._cache.move_to_end(filepath)
                return self._cache[filepath]
        wav = self._load(filepath)
        with self._lock:
            self._cache[filepath] = wav
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return wav

    @staticmethod
    def _load(filepath: str) -> torch.Tensor:
        from oss_cli import read_audio
        return read_audio(filepath)


def _decode_npz_mel(data):
    """Decode mel.npz bytes to tensor."""
    if data is None:
        return None
    buf = BytesIO(data) if isinstance(data, bytes) else data
    z = np.load(buf)
    keys = list(z.keys())
    if not keys:
        return None
    x = z[keys[0]] if len(keys) == 1 else z.get("mel", z.get("arr_0", z[keys[0]]))
    return torch.from_numpy(np.asarray(x)).float()


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
    """Iterable dataset over tar shards with JSON metadata + mel.npz.

    Each sample yields::

        {"audio": (1, samples), "sample_rate": int,
         "mel": (T, 128), "mel_mask": (T,)}
    """

    def __init__(
        self,
        urls,
        max_frames_50hz: int = 1500,
        seed: int = 42,
        audio_cache_size: int = 64,
        handler=None,
    ):
        if wds is None:
            raise ImportError("webdataset is required; pip install webdataset")

        self.max_frames_50hz = max_frames_50hz
        self.rng = random.Random(seed)
        self.handler = handler or wds.handlers.warn_and_continue
        self.audio_cache_size = audio_cache_size
        self._audio_cache: _AudioLRUCache | None = None

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
                f"world_size={self.world_size}, cache_size={audio_cache_size}"
            )

    def _get_audio_cache(self) -> _AudioLRUCache:
        if self._audio_cache is None:
            self._audio_cache = _AudioLRUCache(maxsize=self.audio_cache_size)
        return self._audio_cache

    def _decode_sample(self, sample):
        # --- parse JSON metadata ---
        json_data = None
        mel_data = None
        for key, val in sample.items():
            if key == "__key__" or key == "__url__":
                continue
            if key.endswith(".json") or key == "json":
                json_data = val
            elif "mel" in key and (key.endswith(".npz") or key.endswith(".npy")):
                mel_data = val

        if json_data is None:
            logger.warning("[AudioWebDataset] sample missing JSON metadata")
            return None

        if isinstance(json_data, bytes):
            json_data = json_data.decode("utf-8")
        if isinstance(json_data, str):
            try:
                meta = json.loads(json_data)
            except json.JSONDecodeError as e:
                logger.warning(f"[AudioWebDataset] bad JSON: {e}")
                return None
        else:
            meta = json_data

        audio_filepath = meta.get("audio_filepath")
        if not audio_filepath:
            logger.warning("[AudioWebDataset] JSON missing audio_filepath")
            return None

        seg_start = meta.get("segment_start_time", 0)
        seg_end = meta.get("segment_end_time")
        sr = meta.get("sample_rate", 24000)
        hop_length = meta.get("hop_length", 240)

        # --- decode mel ---
        mel = _decode_npz_mel(mel_data)
        if mel is None:
            logger.warning("[AudioWebDataset] failed to decode mel")
            return None
        mel = _normalize_mel_shape(mel)

        # --- fetch audio from S3 (cached) ---
        try:
            wav = self._get_audio_cache().get(audio_filepath)
        except Exception as e:
            logger.warning(f"[AudioWebDataset] audio read failed for {audio_filepath}: {e}")
            return None

        # slice segment
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if seg_end is not None:
            wav = wav[:, seg_start:seg_end]
        elif seg_start > 0:
            wav = wav[:, seg_start:]

        # --- crop mel to max_frames_50hz ---
        T_mel = mel.shape[0]
        if self.max_frames_50hz > 0 and T_mel > self.max_frames_50hz:
            start = self.rng.randint(0, T_mel - self.max_frames_50hz)
            end = start + self.max_frames_50hz
            mel = mel[start:end]
            # crop audio to match
            audio_start = start * hop_length
            audio_end = end * hop_length
            wav = wav[:, audio_start:audio_end]
        else:
            # trim audio to match mel duration
            expected_samples = T_mel * hop_length
            if wav.shape[1] > expected_samples:
                wav = wav[:, :expected_samples]

        mel_mask = torch.ones(mel.shape[0], dtype=torch.float32)

        return {
            "audio": wav.float(),
            "sample_rate": sr,
            "mel": mel,
            "mel_mask": mel_mask,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # each worker gets a fresh cache (fork-safe)
        self._audio_cache = None

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
                        f"audio={out['audio'].shape}, mel={out['mel'].shape}, sr={out['sample_rate']}"
                    )
                yield out

    def __len__(self):
        return 0


class AudioCollateFn:
    """Pad audio and mel to max length in batch, stack into tensors."""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch_list):
        mel_list = [b["mel"] for b in batch_list]
        mask_list = [b["mel_mask"] for b in batch_list]
        audio_list = [b["audio"] for b in batch_list]
        sr_list = [b["sample_rate"] for b in batch_list]

        max_mel_t = max(x.shape[0] for x in mel_list)
        max_audio_t = max(x.shape[1] for x in audio_list)

        def pad_mel(x):
            if x.shape[0] < max_mel_t:
                pad = torch.full(
                    (max_mel_t - x.shape[0], x.shape[1]),
                    self.pad_value,
                    dtype=x.dtype,
                )
                return torch.cat([x, pad], dim=0)
            return x[:max_mel_t]

        def pad_mask(x):
            if x.shape[0] < max_mel_t:
                pad = torch.zeros(max_mel_t - x.shape[0], dtype=x.dtype)
                return torch.cat([x, pad], dim=0)
            return x[:max_mel_t]

        def pad_audio(x):
            if x.shape[1] < max_audio_t:
                pad = torch.zeros(
                    x.shape[0], max_audio_t - x.shape[1], dtype=x.dtype
                )
                return torch.cat([x, pad], dim=1)
            return x[:, :max_audio_t]

        mel = torch.stack([pad_mel(x) for x in mel_list])
        mel_mask = torch.stack([pad_mask(x) for x in mask_list])
        audio = torch.stack([pad_audio(x) for x in audio_list])
        sample_rate = torch.tensor(sr_list, dtype=torch.int32)

        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "mel": mel,
            "mel_mask": mel_mask,
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

    audio_cache_size = dataset_conf.get("audio_cache_size", 64)
    train_dataset = AudioWebDataset(
        urls=train_urls,
        max_frames_50hz=max_frames,
        seed=seed,
        audio_cache_size=audio_cache_size,
    )
    num_workers_val = getattr(args, "num_workers", 4)
    audio_collate_fn = AudioCollateFn(pad_value=0.0)
    logging.info(
        f"[DataLoader] AudioWebDataset: batch_size={batch_size}, "
        f"num_workers={num_workers_val}, audio_cache_size={audio_cache_size}"
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
            audio_cache_size=min(audio_cache_size, 16),
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
