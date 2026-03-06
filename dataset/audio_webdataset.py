"""
AudioWebDataset: read webdataset tar shards containing JSON metadata,
precomputed mel spectrogram, and pre-extracted whisper/wavlm/muq features.

Design: mel is used ONLY as ground truth for flow matching reconstruction.
Features (whisper/wavlm/muq) are conditioning; mel is never used to drive
feature extraction or alignment.

When target_type="earvae_latent", reads waveform from OSS (audio_filepath),
caches full-song waveform, chunks by segment time, resamples to 48kHz stereo.
EAR_VAE encoding happens in batch_forward on GPU.

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
import torchaudio
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

try:
    import webdataset as wds
except ImportError:
    wds = None

# OSS and EAR_VAE waveform logic only when target_type=earvae_latent
def _read_waveform_from_oss(file_path: str, oss_pool) -> tuple:
    """Read audio from OSS s3 path, return (audio [C, T], sample_rate) or None."""
    try:
        oss = oss_pool.get_conn()
        bucket = file_path[5:].split("/", 1)[0]
        name = file_path[5:].split("/", 1)[1]
        audio_bytes = oss.get_file(bucket, name).read()
        oss_pool.release_conn(oss)
    except Exception as e:
        logger.warning(f"[AudioWebDataset] OSS read failed {file_path}: {e}")
        return None
    try:
        buf = BytesIO(audio_bytes)
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        fmt = "mp4" if ext in ("m4a", "mp4") else None
        audio, sr = torchaudio.load(buf, format=fmt)
        return audio, sr
    except Exception as e:
        logger.warning(f"[AudioWebDataset] torchaudio load failed {file_path}: {e}")
        return None


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

    target_type:
        "mel": yield mel + mel_mask + features (default).
        "earvae_latent": read waveform from OSS, cache per song, chunk by segment,
            resample to 48kHz stereo; yield waveform_48k + waveform_mask + features.
            EAR_VAE encode happens in batch_forward on GPU.

    Each sample yields (mel mode)::

        {"mel": (T, 128), "mel_mask": (T,),
         "whisper_feat": (T_w, 1280), "wavlm_feat": (T_wl, 1024),
         "muq_feat": (T_m, 1024)}

    Or (earvae_latent mode)::

        {"waveform_48k": (2, T_audio), "waveform_mask": (T_audio,),
         "whisper_feat", "wavlm_feat", "muq_feat"}
    """

    def __init__(
        self,
        urls,
        max_frames_50hz: int = 1500,
        seed: int = 42,
        handler=None,
        target_type: str = "mel",
        target_sr: int = 48000,
        oss_pool=None,
    ):
        if wds is None:
            raise ImportError("webdataset is required; pip install webdataset")

        self.max_frames_50hz = max_frames_50hz
        self.rng = random.Random(seed)
        self.handler = handler or wds.handlers.warn_and_continue
        self.target_type = target_type
        self.target_sr = target_sr
        self._waveform_cache = {}

        if target_type == "earvae_latent":
            if oss_pool is None:
                try:
                    from oss_cli import OSS_POOL
                    oss_pool = OSS_POOL
                except ImportError as e:
                    raise ImportError(
                        "earvae_latent requires oss_cli; pip install or set oss_pool"
                    ) from e
            self._oss_pool = oss_pool

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
                f"world_size={self.world_size}, target_type={target_type}"
            )

    def _read_waveform_segment(self, metadata: dict):
        """Read waveform from OSS, cache full song, chunk by segment, resample to target_sr stereo."""
        audio_filepath = metadata.get("audio_filepath")
        if not audio_filepath or not str(audio_filepath).startswith("s3://"):
            return None
        start_sample = int(metadata.get("segment_start_time", 0))
        end_sample = int(metadata.get("segment_end_time", 0))
        orig_sr = int(metadata.get("sample_rate", 24000))

        # Cache: when path changes, clear and load new song
        if self._waveform_cache.get("path") != audio_filepath:
            result = _read_waveform_from_oss(audio_filepath, self._oss_pool)
            if result is None:
                return None
            full_audio, full_sr = result
            self._waveform_cache = {"path": audio_filepath, "audio": full_audio, "sr": full_sr}

        full_audio = self._waveform_cache["audio"]
        full_sr = self._waveform_cache["sr"]

        # Chunk: segment indices at original sr
        start_idx = int(start_sample * full_sr / orig_sr)
        end_idx = int(end_sample * full_sr / orig_sr)
        if start_idx >= full_audio.shape[1] or end_idx <= 0:
            return None
        start_idx = max(0, start_idx)
        end_idx = min(full_audio.shape[1], end_idx)
        segment = full_audio[:, start_idx:end_idx]

        # Resample to target_sr
        if full_sr != self.target_sr:
            segment = torchaudio.functional.resample(segment, full_sr, self.target_sr)

        # Mono -> stereo
        if segment.shape[0] == 1:
            segment = torch.cat([segment, segment], dim=0)

        return segment

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
        try:
            meta = json.loads(json_data) if isinstance(json_data, str) else json_data
        except json.JSONDecodeError as e:
            logger.warning(f"[AudioWebDataset] bad JSON: {e}")
            return None

        # --- decode pre-extracted features (required for both modes) ---
        whisper_feat = _decode_npy_feature(whisper_data)
        wavlm_feat = _decode_npy_feature(wavlm_data)
        muq_feat = _decode_npy_feature(muq_data)
        if whisper_feat is None or wavlm_feat is None or muq_feat is None:
            logger.warning("[AudioWebDataset] missing one or more feature files")
            return None

        if self.target_type == "earvae_latent":
            waveform_48k = self._read_waveform_segment(meta)
            if waveform_48k is None:
                return None
            T = waveform_48k.shape[1]
            if self.max_frames_50hz > 0:
                max_samples = int(self.max_frames_50hz * 960)  # 50Hz * downsample 960
                if T > max_samples:
                    waveform_48k = waveform_48k[:, :max_samples]
                    T = max_samples
            waveform_mask = torch.ones(T, dtype=torch.float32)
            return {
                "waveform_48k": waveform_48k,
                "waveform_mask": waveform_mask,
                "whisper_feat": whisper_feat,
                "wavlm_feat": wavlm_feat,
                "muq_feat": muq_feat,
            }

        # --- mel mode: decode mel ---
        mel = _decode_npz_mel(mel_data)
        if mel is None:
            logger.warning("[AudioWebDataset] failed to decode mel")
            return None
        mel = _normalize_mel_shape(mel)

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
                    if "mel" in out:
                        logger.info(
                            f"[AudioWebDataset] First sample: "
                            f"mel={out['mel'].shape}, "
                            f"whisper={out['whisper_feat'].shape}, "
                            f"wavlm={out['wavlm_feat'].shape}, "
                            f"muq={out['muq_feat'].shape}"
                        )
                    else:
                        logger.info(
                            f"[AudioWebDataset] First sample: "
                            f"waveform_48k={out['waveform_48k'].shape}, "
                            f"whisper={out['whisper_feat'].shape}, "
                            f"wavlm={out['wavlm_feat'].shape}, "
                            f"muq={out['muq_feat'].shape}"
                        )
                yield out

    def __len__(self):
        return 0


class AudioCollateFn:
    """Pad mel/waveform and features to max length in batch, stack into tensors."""

    def __init__(self, pad_value: float = 0.0, target_type: str = "mel"):
        self.pad_value = pad_value
        self.target_type = target_type

    def __call__(self, batch_list):
        whisper_list = [b["whisper_feat"] for b in batch_list]
        wavlm_list = [b["wavlm_feat"] for b in batch_list]
        muq_list = [b["muq_feat"] for b in batch_list]

        def _time_dim(x):
            return x.shape[1] if x.dim() == 3 and x.shape[0] == 1 else x.shape[0]

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

        max_whisper_t = max(_time_dim(x) for x in whisper_list)
        max_wavlm_t = max(_time_dim(x) for x in wavlm_list)
        max_muq_t = max(_time_dim(x) for x in muq_list)

        whisper_feat = torch.cat([_pad_feat(x, max_whisper_t) for x in whisper_list], dim=0)
        wavlm_feat = torch.cat([_pad_feat(x, max_wavlm_t) for x in wavlm_list], dim=0)
        muq_feat = torch.cat([_pad_feat(x, max_muq_t) for x in muq_list], dim=0)

        out = {"whisper_feat": whisper_feat, "wavlm_feat": wavlm_feat, "muq_feat": muq_feat}

        if "waveform_48k" in batch_list[0]:
            # earvae_latent mode: pad waveform (2, T) to max T
            wf_list = [b["waveform_48k"] for b in batch_list]
            mask_list = [b["waveform_mask"] for b in batch_list]
            max_t = max(w.shape[1] for w in wf_list)

            def _pad_wf(w, max_t_):
                if w.shape[1] < max_t_:
                    pad = torch.zeros(2, max_t_ - w.shape[1], dtype=w.dtype)
                    return torch.cat([w, pad], dim=1)
                return w[:, :max_t_]

            def _pad_1d(x, max_t_):
                if x.shape[0] < max_t_:
                    pad = torch.zeros(max_t_ - x.shape[0], dtype=x.dtype)
                    return torch.cat([x, pad], dim=0)
                return x[:max_t_]

            out["waveform_48k"] = torch.stack([_pad_wf(w, max_t) for w in wf_list])
            out["waveform_mask"] = torch.stack([_pad_1d(m, max_t) for m in mask_list])
        else:
            # mel mode
            mel_list = [b["mel"] for b in batch_list]
            mask_list = [b["mel_mask"] for b in batch_list]
            max_mel_t = max(x.shape[0] for x in mel_list)

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

            out["mel"] = torch.stack([_pad_2d(x, max_mel_t, self.pad_value) for x in mel_list])
            out["mel_mask"] = torch.stack([_pad_1d(x, max_mel_t) for x in mask_list])

        return out


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

    target_type = dataset_conf.get("target_type", "mel")
    target_sr = dataset_conf.get("target_sr", 48000)

    train_dataset = AudioWebDataset(
        urls=train_urls,
        max_frames_50hz=max_frames,
        seed=seed,
        target_type=target_type,
        target_sr=target_sr,
    )
    num_workers_val = getattr(args, "num_workers", 4)
    audio_collate_fn = AudioCollateFn(pad_value=0.0, target_type=target_type)
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
            target_type=target_type,
            target_sr=target_sr,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cv_batch_size,
            num_workers=num_workers_val,
            collate_fn=AudioCollateFn(pad_value=0.0, target_type=target_type),
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
