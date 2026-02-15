"""
On-the-fly extraction of whisper / wavlm / muq features from mel during training.

Pipeline: mel (T, 128) → waveform (inverse mel, codec params) → Whisper(16k), WavLM(16k), MuQ(24k) → features.
Models are loaded once per process (or lazily) and reused.
"""

import glob
import logging
import os
import sys

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

# Repo root and MuQ src for imports (workers may not inherit PYTHONPATH)
_CODEC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MUQ_SRC = os.path.join(_CODEC_ROOT, "MuQ", "src")
for _p in (_CODEC_ROOT, _MUQ_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def mel_to_waveform(
    mel: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1920,
    hop_length: int = 480,
    n_mels: int = 128,
    n_iter: int = 32,
) -> torch.Tensor:
    """Invert log-mel (T, n_mels) to waveform (1, T*hop_length) using Griffin-Lim."""
    # mel: (T, n_mels) -> (n_mels, T) for InverseMelScale
    if mel.dim() == 2:
        mel = mel.T  # (n_mels, T)
    mel = mel.float()
    # Assume log-scale; convert to linear for inverse
    mel_linear = torch.clamp(mel.exp(), min=1e-10)
    inv_mel = T.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=8000.0,
    ).to(mel.device)
    griffin = T.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_iter=n_iter,
    ).to(mel.device)
    # InverseMelScale expects (..., n_mels, time)
    spec = inv_mel(mel_linear.unsqueeze(0))
    wav = griffin(spec)
    return wav


def _resolve_local_ckpt(path: str, ext: str = ".pt") -> str:
    """If path is a directory, return first file with given extension inside; else return path."""
    if not path:
        return path
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = glob.glob(os.path.join(path, "*" + ext))
        if candidates:
            return sorted(candidates)[0]
        # HuggingFace style
        for name in ("pytorch_model.bin", "model.safetensors"):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                return p
    return path


class CodecFeatureExtractor:
    """Extract whisper, wavlm, muq features from mel (T, 128) on-the-fly."""

    def __init__(
        self,
        device: str = "cuda",
        whisper_name: str = "large-v3",
        whisper_download_root: str = None,
        wavlm_ckpt: str = None,
        muq_name: str = "OpenMuQ/MuQ-large-msd-iter",
        hf_cache_dir: str = None,
        # codec mel params for mel→wav
        sample_rate: int = 24000,
        n_fft: int = 1920,
        hop_length: int = 480,
        n_mels: int = 128,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._whisper_model = None
        self._whisper_name = whisper_name
        self._whisper_download_root = whisper_download_root
        self._wavlm_model = None
        self._wavlm_cfg = None
        self._wavlm_ckpt = wavlm_ckpt
        self._wavlm_is_hf = False
        self._muq_model = None
        self._muq_name = muq_name
        self._hf_cache_dir = hf_cache_dir
        self._resample_24k_16k = None

    def _get_whisper(self):
        if self._whisper_model is None:
            logger.info(f"[FeatureExtractor] Loading Whisper model: {self._whisper_name} on {self.device}")
            from whisper import whisper
            # Use official model name (e.g. large-v3); download to whisper_download_root if set
            name = self._whisper_name
            kwargs = {"device": self.device}
            if self._whisper_download_root:
                kwargs["download_root"] = os.path.expanduser(self._whisper_download_root)
            self._whisper_model = whisper.load_model(name, **kwargs)
            self._whisper_model = self._whisper_model.eval()
            logger.info(f"[FeatureExtractor] Whisper model loaded successfully")
        return self._whisper_model

    def _get_wavlm(self):
        if self._wavlm_model is None and self._wavlm_ckpt:
            logger.info(f"[FeatureExtractor] Loading WavLM model: {self._wavlm_ckpt} on {self.device}")
            path = os.path.expanduser(self._wavlm_ckpt)
            resolved = path if os.path.isfile(path) else (_resolve_local_ckpt(self._wavlm_ckpt, ".pt") if os.path.isdir(path) else None)
            if resolved:
                # Local checkpoint (Microsoft .pt format)
                logger.info(f"[FeatureExtractor] Loading WavLM from local checkpoint: {resolved}")
                from WavLM import WavLM, WavLMConfig
                ckpt = torch.load(resolved, map_location=self.device, weights_only=False)
                self._wavlm_cfg = WavLMConfig(ckpt["cfg"])
                self._wavlm_model = WavLM(self._wavlm_cfg)
                self._wavlm_model.load_state_dict(ckpt["model"])
                self._wavlm_model = self._wavlm_model.to(self.device).eval()
                self._wavlm_is_hf = False
                logger.info(f"[FeatureExtractor] WavLM model loaded successfully")
            else:
                # HuggingFace id (e.g. microsoft/wavlm-large) → download to hf_cache_dir if set
                logger.info(f"[FeatureExtractor] Loading WavLM from HuggingFace: {self._wavlm_ckpt}")
                # Suppress "some weights not initialized" / "some weights not used" (pos_conv format mismatch)
                import logging
                from transformers import WavLMModel
                tlog = logging.getLogger("transformers")
                old_level = tlog.level
                tlog.setLevel(logging.ERROR)
                try:
                    kwargs = {}
                    if self._hf_cache_dir:
                        kwargs["cache_dir"] = os.path.expanduser(self._hf_cache_dir)
                    self._wavlm_model = WavLMModel.from_pretrained(self._wavlm_ckpt, **kwargs)
                finally:
                    tlog.setLevel(old_level)
                self._wavlm_model = self._wavlm_model.to(self.device).eval()
                self._wavlm_is_hf = True
                logger.info(f"[FeatureExtractor] WavLM model loaded successfully")
        return self._wavlm_model

    def _get_muq(self):
        if self._muq_model is None:
            logger.info(f"[FeatureExtractor] Loading MuQ model: {self._muq_name} on {self.device}")
            from muq import MuQ
            # from_pretrained: HuggingFace id or local dir; cache to hf_cache_dir if set
            kwargs = {}
            if self._hf_cache_dir:
                kwargs["cache_dir"] = os.path.expanduser(self._hf_cache_dir)
            self._muq_model = MuQ.from_pretrained(self._muq_name, **kwargs)
            self._muq_model = self._muq_model.to(self.device).eval()
            logger.info(f"[FeatureExtractor] MuQ model loaded successfully")
        return self._muq_model

    def _resample_24_to_16(self, wav_24k: torch.Tensor) -> torch.Tensor:
        if self._resample_24k_16k is None:
            self._resample_24k_16k = torchaudio.transforms.Resample(
                self.sample_rate, 16000
            ).to(wav_24k.device)
        return self._resample_24k_16k(wav_24k)

    def extract(
        self,
        mel: torch.Tensor,
    ) -> tuple:
        """
        mel: (T, 128) or (1, T, 128), float, log-scale.
        Returns (whisper_feat, wavlm_feat, muq_feat) with shapes (T, 1280), (T, 1024), (T25, 1024).
        T25 = T // 2. Features are trimmed to match mel length.
        """
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        mel = mel.float().to(self.device)
        T_mel = mel.shape[0]
        wav_24k = mel_to_waveform(
            mel,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        wav_24k = wav_24k.to(self.device)
        wav_16k = self._resample_24_to_16(wav_24k)

        whisper_feat = None
        wavlm_feat = None
        muq_feat = None

        with torch.no_grad():
            # Whisper: 16k waveform → pad_or_trim → log_mel → encoder
            whisper_model = self._get_whisper()
            if whisper_model is not None:
                from whisper import whisper
                from whisper.decoding import DecodingTask, DecodingOptions
                audio_np = wav_16k.squeeze(0).cpu().numpy()
                audio = whisper.pad_or_trim(audio_np)
                mel_whisper = whisper.log_mel_spectrogram(
                    audio, n_mels=whisper_model.dims.n_mels
                ).to(self.device)
                # Encoder expects (batch, n_mels, n_frames); log_mel can return (n_mels, T) or (B, n_mels, T)
                if mel_whisper.dim() == 3:
                    mel_whisper = mel_whisper[0]
                mel_whisper = mel_whisper.unsqueeze(0)
                options = DecodingOptions()
                task = DecodingTask(whisper_model, options)
                audio_features = task._get_audio_features(mel_whisper)
                # (1, T_whisper, 1280) -> trim to T_mel
                audio_features = audio_features.squeeze(0)
                if audio_features.shape[0] > T_mel:
                    audio_features = audio_features[:T_mel]
                elif audio_features.shape[0] < T_mel:
                    audio_features = torch.nn.functional.pad(
                        audio_features, (0, 0, 0, T_mel - audio_features.shape[0])
                    )
                whisper_feat = audio_features.float() if self.device == "cpu" else audio_features.float().to(self.device)

            # WavLM: 16k waveform
            wavlm_model = self._get_wavlm()
            if wavlm_model is not None:
                wav_input = wav_16k
                if getattr(self, "_wavlm_is_hf", False):
                    # HuggingFace WavLM expects (B, samples) mono; ensure 2D
                    wav_input = wav_input.squeeze()
                    if wav_input.dim() == 1:
                        wav_input = wav_input.unsqueeze(0)
                    elif wav_input.dim() > 1:
                        wav_input = wav_input[0:1].reshape(1, -1)
                    rep = wavlm_model(wav_input).last_hidden_state
                else:
                    # Microsoft WavLM expects (batch, length) for extract_features
                    # Handle various input shapes: flatten to (batch, length)
                    while wav_input.dim() > 2:
                        wav_input = wav_input.squeeze()
                    if wav_input.dim() == 1:
                        wav_input = wav_input.unsqueeze(0)  # (length,) -> (1, length)
                    elif wav_input.dim() == 2:
                        # (batch, length) or (channels, length)
                        if wav_input.shape[0] > 1 and wav_input.shape[0] <= 2:
                            wav_input = wav_input[0:1]  # Take first channel if stereo
                        # Now should be (1, length)
                    if wavlm_model.cfg.normalize:
                        wav_input = torch.nn.functional.layer_norm(
                            wav_input, wav_input.shape
                        )
                    rep = wavlm_model.extract_features(wav_input)[0]
                # rep: (1, seq_len, 1024). WavLM 50Hz -> seq_len ≈ T_mel
                rep = rep.squeeze(0)
                if rep.shape[0] > T_mel:
                    rep = rep[:T_mel]
                elif rep.shape[0] < T_mel:
                    rep = torch.nn.functional.pad(
                        rep, (0, 0, 0, T_mel - rep.shape[0])
                    )
                wavlm_feat = rep.float() if self.device == "cpu" else rep.float().to(self.device)

            # MuQ: 24k waveform, 25Hz
            muq_model = self._get_muq()
            if muq_model is not None:
                # MuQ expects (batch, time) 2D mono input
                muq_input = wav_24k
                # Remove all size-1 dimensions
                while muq_input.dim() > 1 and any(s == 1 for s in muq_input.shape):
                    muq_input = muq_input.squeeze()
                # If stereo (2 channels), average channels to mono
                if muq_input.dim() == 2 and muq_input.shape[0] == 2:
                    muq_input = muq_input.mean(dim=0, keepdim=True)  # (2, time) -> (1, time)
                elif muq_input.dim() == 3 and muq_input.shape[1] == 2:
                    muq_input = muq_input.mean(dim=1, keepdim=True).squeeze(1)  # (batch, 2, time) -> (batch, time)
                elif muq_input.dim() == 1:
                    muq_input = muq_input.unsqueeze(0)  # (time,) -> (1, time)
                elif muq_input.dim() > 2:
                    # Higher dim: take first channel/index and flatten
                    muq_input = muq_input[:, 0, ...] if muq_input.dim() == 3 else muq_input[0, 0, ...]
                    muq_input = muq_input.flatten().unsqueeze(0)
                # Ensure (1, time)
                if muq_input.dim() != 2 or muq_input.shape[0] != 1:
                    muq_input = muq_input.flatten().unsqueeze(0)
                out = muq_model(muq_input, output_hidden_states=True)
                rep = out.last_hidden_state.squeeze(0)
                T25 = T_mel // 2
                if rep.shape[0] > T25:
                    rep = rep[:T25]
                elif rep.shape[0] < T25:
                    rep = torch.nn.functional.pad(
                        rep, (0, 0, 0, T25 - rep.shape[0])
                    )
                muq_feat = rep.float() if self.device == "cpu" else rep.float().to(self.device)

        return whisper_feat, wavlm_feat, muq_feat

    def extract_batch(self, mel_batch):
        """Extract features for a batch of mel. mel_batch: (B, T, 128). Returns (whisper_feat, wavlm_feat, muq_feat) as (B, T, 1280), (B, T, 1024), (B, T//2, 1024)."""
        B = mel_batch.shape[0]
        whisper_list, wavlm_list, muq_list = [], [], []
        for i in range(B):
            w, wl, m = self.extract(mel_batch[i])
            whisper_list.append(w)
            wavlm_list.append(wl)
            muq_list.append(m)
        device = mel_batch.device
        whisper_feat = torch.stack(whisper_list, dim=0).to(device)
        wavlm_feat = torch.stack(wavlm_list, dim=0).to(device)
        muq_feat = torch.stack(muq_list, dim=0).to(device)
        return whisper_feat, wavlm_feat, muq_feat
