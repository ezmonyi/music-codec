"""
On-the-fly extraction of whisper / wavlm / muq features from mel or from audio waveform.

Two pipelines:
- From mel (legacy): mel (T, 128) → waveform (inverse mel) → Whisper/WavLM/MuQ → features.
- From audio (preferred): read waveform from webdataset audio field → Whisper(16k), WavLM(16k), MuQ(24k) → features.
Models are loaded once per process (or lazily) and reused.
"""

import logging
import os
import torch
import torchaudio

logger = logging.getLogger(__name__)


class CodecFeatureExtractor:
    """Extract whisper, wavlm, muq features from mel (T, 128) on-the-fly."""

    def __init__(
        self,
        device: str = "cuda",
    ):
        self.device = device
        self._whisper_model = None
        self._wavlm_model = None
        self._muq_model = None

    def _get_whisper(self):
        if self._whisper_model is None:
            import whisper
            model_path = "/mnt/yi-jfs/pretrained_models/whisper/large-v3.pt"
            self._whisper_model = whisper.load_model(model_path)
            self._whisper_model = self._whisper_model.to(self.device).eval()
            logger.info("Whisper model loaded successfully.")
        return self._whisper_model

    def _get_wavlm(self):
        if self._wavlm_model is None:
            from WavLM import WavLM, WavLMConfig
            checkpoint = torch.load(
                "/mnt/yi-jfs/pretrained_models/wavlm/WavLM-Large.pt",
                map_location=self.device,
                weights_only=False,
            )
            cfg = WavLMConfig(checkpoint["cfg"])
            self._wavlm_model = WavLM(cfg)
            self._wavlm_model.load_state_dict(checkpoint["model"])
            self._wavlm_model = self._wavlm_model.to(self.device).eval()
            logger.info("WavLM model loaded successfully.")
        return self._wavlm_model

    def _get_muq(self):
        if self._muq_model is None:
            logger.info(f"[FeatureExtractor] Loading MuQ model: {self._muq_name} on {self.device}")
            from muq import MuQ
            self._muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
            self._muq_model = self._muq_model.to(self.device).eval()
            logger.info(f"[FeatureExtractor] MuQ model loaded successfully")
        return self._muq_model

    def _mel_to_muq_input(self, mel: torch.Tensor):
        """Convert mel (T, 128) or (1, T, 128) to MuQ input dict: {\"melspec_2048\": (B, 128, T)}."""
        if mel.dim() == 3:
            mel = mel.mean(dim=0)  # stereo (C, T, 128) -> mono (T, 128)
        # (T, 128) -> (1, 128, T) for Conv2dSubsampling
        mel_bft = mel.float().to(self.device).transpose(0, 1).unsqueeze(0)
        return {"melspec_2048": mel_bft}

    def _run_models_on_waveform(
        self,
        mel: torch.Tensor,
        T_mel: int = None,
    ) -> tuple:
        """Run Whisper/WavLM on waveform; run MuQ on mel spec (mel_for_muq) when provided, else compute mel from wav_24k.
        MuQ expects x[\"melspec_2048\"] with shape (B, 128, T). Optionally trim to T_mel (50Hz) and T_mel//2 (25Hz)."""
        T_trim = T_mel  # for 50Hz outputs; T25 = T_mel // 2 for MuQ
        whisper_feat = None
        wavlm_feat = None
        muq_feat = None

        with torch.no_grad():
            # Whisper: 16k waveform
            whisper_model = self._get_whisper()
            if whisper_model is not None:
                from whisper import whisper
                from whisper.decoding import DecodingTask, DecodingOptions
                audio_np = wav_16k.squeeze(0).cpu().numpy()
                audio = whisper.pad_or_trim(audio_np)
                mel_whisper = whisper.log_mel_spectrogram(
                    audio, n_mels=whisper_model.dims.n_mels
                ).to(self.device)
                if mel_whisper.dim() == 3:
                    mel_whisper = mel_whisper[0]
                mel_whisper = mel_whisper.unsqueeze(0)
                options = DecodingOptions()
                task = DecodingTask(whisper_model, options)
                audio_features = task._get_audio_features(mel_whisper)
                audio_features = audio_features.squeeze(0)
                if T_trim is not None:
                    if audio_features.shape[0] > T_trim:
                        audio_features = audio_features[:T_trim]
                    elif audio_features.shape[0] < T_trim:
                        audio_features = torch.nn.functional.pad(
                            audio_features, (0, 0, 0, T_trim - audio_features.shape[0])
                        )
                whisper_feat = audio_features.float() if self.device == "cpu" else audio_features.float().to(self.device)

            # WavLM: 16k waveform
            wavlm_model = self._get_wavlm()
            if wavlm_model is not None:
                wav_input = wav_16k
                if getattr(self, "_wavlm_is_hf", False):
                    wav_input = wav_input.squeeze()
                    if wav_input.dim() == 1:
                        wav_input = wav_input.unsqueeze(0)
                    elif wav_input.dim() > 1:
                        wav_input = wav_input[0:1].reshape(1, -1)
                    rep = wavlm_model(wav_input).last_hidden_state
                else:
                    while wav_input.dim() > 2:
                        wav_input = wav_input.squeeze()
                    if wav_input.dim() == 1:
                        wav_input = wav_input.unsqueeze(0)
                    elif wav_input.dim() == 2:
                        if wav_input.shape[0] > 1 and wav_input.shape[0] <= 2:
                            wav_input = wav_input[0:1]
                    if wavlm_model.cfg.normalize:
                        wav_input = torch.nn.functional.layer_norm(
                            wav_input, wav_input.shape
                        )
                    rep = wavlm_model.extract_features(wav_input)[0]
                rep = rep.squeeze(0)
                if T_trim is not None:
                    if rep.shape[0] > T_trim:
                        rep = rep[:T_trim]
                    elif rep.shape[0] < T_trim:
                        rep = torch.nn.functional.pad(
                            rep, (0, 0, 0, T_trim - rep.shape[0])
                        )
                wavlm_feat = rep.float() if self.device == "cpu" else rep.float().to(self.device)

            # MuQ: take mel spec (melspec_2048); no waveform reshaping
            muq_model = self._get_muq()
            if muq_model is not None:
                if mel_for_muq is not None:
                    muq_x = self._mel_to_muq_input(mel_for_muq)
                else:
                    # extract_from_waveform path: compute mel from wav_24k (MuQ uses n_fft=2048, hop_length=240, n_mels=128, is_db=True)
                    if getattr(self, "_muq_mel_from_wav", None) is None:
                        self._muq_mel_stft = torchaudio.transforms.MelSpectrogram(
                            sample_rate=24000, n_fft=2048, hop_length=240, n_mels=128
                        ).to(self.device)
                        self._muq_amp_to_db = torchaudio.transforms.AmplitudeToDB().to(self.device)
                    mel_wav = self._muq_mel_stft(wav_24k.float())
                    mel_wav = self._muq_amp_to_db(mel_wav)
                    if mel_wav.dim() == 3:
                        mel_wav = mel_wav[0]
                    # (n_mels, T) -> (1, 128, T)
                    mel_wav = mel_wav.unsqueeze(0)
                    muq_x = {"melspec_2048": mel_wav}
                out = muq_model(muq_x, output_hidden_states=True)
                rep = out.last_hidden_state.squeeze(0)
                T25_trim = (T_trim // 2) if T_trim is not None else None
                if T25_trim is not None:
                    if rep.shape[0] > T25_trim:
                        rep = rep[:T25_trim]
                    elif rep.shape[0] < T25_trim:
                        rep = torch.nn.functional.pad(
                            rep, (0, 0, 0, T25_trim - rep.shape[0])
                        )
                muq_feat = rep.float() if self.device == "cpu" else rep.float().to(self.device)

        return whisper_feat, wavlm_feat, muq_feat

    def extract(
        self,
        mel: torch.Tensor,
    ) -> tuple:
        """
        mel: (T, 128) or (1, T, 128), float, log-scale.
        Returns (whisper_feat, wavlm_feat, muq_feat) with shapes (T, 1280), (T, 1024), (T25, 1024).
        Uses mel→waveform (Griffin-Lim) then runs models; prefer extract_from_waveform when audio is in dataset.
        """
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        mel = mel.float().to(self.device)
        T_mel = mel.shape[0]
        return self._run_models_on_waveform(mel, T_mel=T_mel)

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
