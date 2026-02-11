"""
Codec dataset: load precomputed whisper / wavlm / muq features and mel.

Two modes:
1. Manifest JSONL: each line has whisper_path, wavlm_path, muq_path, mel_path.
2. WebDataset: tar dir with urls pattern (e.g. .../shard_48*.tar); each sample in tar
   has keys whisper.npy, wavlm.npy, muq.npy, mel.npy (or .pt).
Shapes: whisper (T50, 1280), wavlm (T50, 1024), muq (T25, 1024), mel (T50, 128).
"""

import glob
import json
import os
import random
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist

try:
    import webdataset as wds
except ImportError:
    wds = None


def _load_array(path):
    if path is None or not os.path.isfile(path):
        return None
    if path.endswith(".npy"):
        x = np.load(path)
    elif path.endswith(".pt") or path.endswith(".pth"):
        x = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(x, torch.Tensor):
            x = x.numpy()
    else:
        raise ValueError("Unknown extension: {}".format(path))
    return x


class CodecDataset(Dataset):
    """Each item: load one sample's features and mel; lengths may differ."""

    def __init__(
        self,
        manifest_path,
        max_frames_50hz=1500,
        random_crop=True,
        seed=42,
    ):
        self.max_frames_50hz = max_frames_50hz
        self.random_crop = random_crop
        self.rng = random.Random(seed)
        self.samples = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        whisper = _load_array(s.get("whisper_path"))
        wavlm = _load_array(s.get("wavlm_path"))
        muq = _load_array(s.get("muq_path"))
        mel = _load_array(s.get("mel_path"))
        if whisper is None or wavlm is None or muq is None or mel is None:
            raise RuntimeError("Missing feature for sample idx {}".format(idx))
        whisper = torch.from_numpy(whisper).float()
        wavlm = torch.from_numpy(wavlm).float()
        muq = torch.from_numpy(muq).float()
        mel = torch.from_numpy(mel).float()
        # whisper, wavlm, mel: (T50, D); muq: (T25, D)
        t50 = min(whisper.shape[0], wavlm.shape[0], mel.shape[0])
        t25 = min(muq.shape[0], t50 // 2)
        t50 = t25 * 2
        if self.max_frames_50hz > 0 and t50 > self.max_frames_50hz:
            if self.random_crop:
                start = self.rng.randint(0, t50 - self.max_frames_50hz)
            else:
                start = 0
            end = start + self.max_frames_50hz
            whisper = whisper[start:end]
            wavlm = wavlm[start:end]
            mel = mel[start:end]
            t25_half = self.max_frames_50hz // 2
            start25 = start // 2
            muq = muq[start25 : start25 + t25_half]
        else:
            whisper = whisper[:t50]
            wavlm = wavlm[:t50]
            mel = mel[:t50]
            muq = muq[:t25]
        mel_mask = torch.ones(mel.shape[0], dtype=torch.float32)
        return {
            "whisper_feat": whisper,
            "wavlm_feat": wavlm,
            "muq_feat": muq,
            "mel": mel,
            "mel_mask": mel_mask,
        }


class CodecCollateFn:
    """Pad to max length in batch and stack."""

    def __init__(self, pad_value=0.0):
        self.pad_value = pad_value

    def __call__(self, batch_list):
        w = [b["whisper_feat"] for b in batch_list]
        wl = [b["wavlm_feat"] for b in batch_list]
        m = [b["muq_feat"] for b in batch_list]
        mel = [b["mel"] for b in batch_list]
        mask = [b["mel_mask"] for b in batch_list]
        t50 = max(x.shape[0] for x in w)
        t25 = max(x.shape[0] for x in m)
        def pad_50(x):
            if x.shape[0] < t50:
                pad = torch.full(
                    (t50 - x.shape[0], x.shape[1]),
                    self.pad_value,
                    dtype=x.dtype,
                )
                return torch.cat([x, pad], dim=0)
            return x[:t50]

        def pad_25(x):
            if x.shape[0] < t25:
                pad = torch.full(
                    (t25 - x.shape[0], x.shape[1]),
                    self.pad_value,
                    dtype=x.dtype,
                )
                return torch.cat([x, pad], dim=0)
            return x[:t25]

        def pad_mask(x):
            if x.shape[0] < t50:
                pad = torch.zeros(t50 - x.shape[0], dtype=x.dtype)
                return torch.cat([x, pad], dim=0)
            return x[:t50]

        whisper_feat = torch.stack([pad_50(x) for x in w])
        wavlm_feat = torch.stack([pad_50(x) for x in wl])
        muq_feat = torch.stack([pad_25(x) for x in m])
        mel = torch.stack([pad_50(x) for x in mel])
        mel_mask = torch.stack([pad_mask(x) for x in mask])
        return {
            "whisper_feat": whisper_feat,
            "wavlm_feat": wavlm_feat,
            "muq_feat": muq_feat,
            "mel": mel,
            "mel_mask": mel_mask,
        }


def _decode_npy_or_pt(data, key):
    """Decode bytes from webdataset sample (key = 'whisper.npy', 'mel.npz' etc.) to tensor."""
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    if isinstance(data, torch.Tensor):
        return data.float()
    buf = BytesIO(data) if isinstance(data, bytes) else data
    if key.endswith(".npz"):
        z = np.load(buf)
        keys = list(z.keys())
        if not keys:
            return None
        x = z[keys[0]] if len(keys) == 1 else z.get("mel", z.get("arr_0", z[keys[0]]))
        return torch.from_numpy(np.asarray(x)).float()
    if key.endswith(".npy"):
        x = np.load(buf)
        return torch.from_numpy(x).float()
    if key.endswith(".pt") or key.endswith(".pth"):
        x = torch.load(buf, map_location="cpu", weights_only=True)
        return x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
    return None


def _align_and_crop_codec(whisper, wavlm, muq, mel, max_frames_50hz, rng):
    """Align T50/T25 and optionally crop to max_frames_50hz."""
    t50 = min(whisper.shape[0], wavlm.shape[0], mel.shape[0])
    t25 = min(muq.shape[0], t50 // 2)
    t50 = t25 * 2
    whisper = whisper[:t50]
    wavlm = wavlm[:t50]
    mel = mel[:t50]
    muq = muq[:t25]
    if max_frames_50hz > 0 and t50 > max_frames_50hz:
        start = rng.randint(0, t50 - max_frames_50hz) if rng else 0
        end = start + max_frames_50hz
        whisper = whisper[start:end]
        wavlm = wavlm[start:end]
        mel = mel[start:end]
        muq = muq[start // 2 : start // 2 + max_frames_50hz // 2]
    mel_mask = torch.ones(mel.shape[0], dtype=torch.float32)
    return whisper, wavlm, muq, mel, mel_mask


class CodecWebDataset(IterableDataset):
    """Iterable dataset over tarred webdataset shards (e.g. shard_48*.tar).
    When use_mel_extractor=True, only mel.npz is loaded; whisper/wavlm/muq are computed on-the-fly from mel.
    """

    def __init__(
        self,
        urls,
        max_frames_50hz=1500,
        seed=42,
        handler=wds.handlers.warn_and_continue if wds else None,
        feature_keys=None,
        use_mel_extractor=False,
        feature_extractor=None,
        feature_extractor_conf=None,
    ):
        if wds is None:
            raise ImportError("webdataset is required for CodecWebDataset; pip install webdataset")
        self.max_frames_50hz = max_frames_50hz
        self.rng = random.Random(seed)
        self.handler = handler or wds.handlers.warn_and_continue
        self.feature_keys = feature_keys or {
            "whisper": "whisper.npy",
            "wavlm": "wavlm.npy",
            "muq": "muq.npy",
            "mel": "mel.npz",
        }
        self.use_mel_extractor = use_mel_extractor
        self._feature_extractor = feature_extractor
        self._feature_extractor_conf = feature_extractor_conf or {}
        if isinstance(urls, str):
            if "*" in urls:
                expanded = sorted(glob.glob(urls))
                if not expanded:
                    raise ValueError("No shards found for pattern: {}".format(urls))
                urls = expanded
            else:
                urls = [urls]
        self.urls = urls
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))
        if self.rank == 0:
            print("[INFO] CodecWebDataset: {} shards, world_size={}, use_mel_extractor={}".format(
                len(self.urls), self.world_size, self.use_mel_extractor
            ))

    def _get_feature_extractor(self):
        if self._feature_extractor is not None:
            return self._feature_extractor
        if not self.use_mel_extractor or not self._feature_extractor_conf:
            return None
        from dataset.mel_to_features import CodecFeatureExtractor
        conf = dict(self._feature_extractor_conf)
        if conf.get("wavlm_ckpt") == "":
            conf["wavlm_ckpt"] = None
        self._feature_extractor = CodecFeatureExtractor(**conf)
        return self._feature_extractor

    def _decode_sample(self, sample):
        melk = self.feature_keys.get("mel", "mel.npz")
        mel = _decode_npy_or_pt(sample.get(melk), melk)
        if mel is None:
            return None

        if self.use_mel_extractor:
            extractor = self._get_feature_extractor()
            if extractor is None:
                return None
            whisper, wavlm, muq = extractor.extract(mel)
            if whisper is None or wavlm is None or muq is None:
                return None
        else:
            wk = self.feature_keys.get("whisper", "whisper.npy")
            wlk = self.feature_keys.get("wavlm", "wavlm.npy")
            mk = self.feature_keys.get("muq", "muq.npy")
            whisper = _decode_npy_or_pt(sample.get(wk), wk)
            wavlm = _decode_npy_or_pt(sample.get(wlk), wlk)
            muq = _decode_npy_or_pt(sample.get(mk), mk)
            if whisper is None or wavlm is None or muq is None:
                return None

        whisper, wavlm, muq, mel, mel_mask = _align_and_crop_codec(
            whisper, wavlm, muq, mel, self.max_frames_50hz, self.rng
        )
        return {
            "whisper_feat": whisper,
            "wavlm_feat": wavlm,
            "muq_feat": muq,
            "mel": mel,
            "mel_mask": mel_mask,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        urls = self.urls.copy()
        random.Random(self.rank * 1000 + worker_id).shuffle(urls)
        if self.world_size > 1:
            urls = [u for i, u in enumerate(urls) if i % self.world_size == self.rank]
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(urls),
            wds.tarfile_to_samples(handler=self.handler),
            wds.shuffle(1000),
        )
        for sample in pipeline:
            out = self._decode_sample(sample)
            if out is not None:
                yield out

    def __len__(self):
        return 0


def init_dataset_and_dataloader(args, configs):
    """Build CodecDataset (manifest) or CodecWebDataset (tar shards) + DataLoader."""
    import yaml as _yaml
    dataset_conf = configs.get("dataset_conf", {})
    if isinstance(dataset_conf, str):
        with open(dataset_conf, "r") as f:
            dataset_conf = _yaml.safe_load(f) or {}
    train_conf = configs.get("train_conf", {})
    max_frames = dataset_conf.get("max_frames_50hz", 1500)
    seed = dataset_conf.get("seed", 42)
    collate_fn = CodecCollateFn(pad_value=0.0)
    batch_size = train_conf.get("batch_size", 8)

    urls = dataset_conf.get("urls")
    webdataset_path = dataset_conf.get("webdataset_path")
    shard_pattern = dataset_conf.get("shard_pattern")
    if urls or (webdataset_path and shard_pattern):
        if urls is None:
            base = webdataset_path.rstrip("/")
            urls = os.path.join(base, shard_pattern)
        use_mel_extractor = dataset_conf.get("use_mel_extractor", False)
        feature_extractor_conf = dataset_conf.get("feature_extractor", {})
        if use_mel_extractor and feature_extractor_conf:
            feature_extractor_conf = dict(feature_extractor_conf)
            num_workers = getattr(args, "num_workers", 4)
            # Force CPU in DataLoader workers to avoid cuFFT/OOM; main process moves batch to GPU
            if num_workers > 0:
                feature_extractor_conf["device"] = "cpu"
            elif "device" not in feature_extractor_conf:
                feature_extractor_conf["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        train_dataset = CodecWebDataset(
            urls=urls,
            max_frames_50hz=max_frames,
            seed=seed,
            feature_keys=dataset_conf.get("feature_keys"),
            use_mel_extractor=use_mel_extractor,
            feature_extractor_conf=feature_extractor_conf if use_mel_extractor else None,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=getattr(args, "num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=getattr(args, "pin_memory", False),
            prefetch_factor=getattr(args, "prefetch", 2) if getattr(args, "num_workers", 0) > 0 else None,
            drop_last=True,
        )
        cv_dataset = None
        cv_loader = None
        return train_dataset, cv_dataset, train_loader, cv_loader

    manifest_path = dataset_conf.get("manifest_path") or getattr(args, "dataset_conf", None)
    if not manifest_path or not os.path.isfile(manifest_path):
        raise ValueError(
            "dataset_conf: set urls (or webdataset_path + shard_pattern) for webdataset, "
            "or manifest_path for JSONL manifest"
        )
    train_dataset = CodecDataset(
        manifest_path=manifest_path,
        max_frames_50hz=max_frames,
        random_crop=True,
        seed=seed,
    )
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(args, "num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=getattr(args, "pin_memory", False),
        prefetch_factor=getattr(args, "prefetch", 2) if getattr(args, "num_workers", 0) > 0 else None,
        drop_last=True,
    )
    cv_dataset = None
    cv_loader = None
    return train_dataset, cv_dataset, train_loader, cv_loader
