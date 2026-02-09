"""
AudioReconModel: Audio reconstruction codec using Conditional Flow Matching.

Architecture (see config.yaml for hyperparameters):

    Whisper feat (B, 1500, 1280) @ 50Hz ─┐
    WavLM  feat (B, 1500, 1024) @ 50Hz ──┤ resample ──→ all @ 25Hz
    MuQ    feat (B,  750, 1024) @ 25Hz ──┘
                                          │
                                    concat dim=-1
                                   (B, 750, 3328)
                                          │
                                    in_proj Linear
                                   (B, 750, 256)
                                          │
                                    VQ codebook
                                   ┌──────┴──────┐
                              codes (B,750)    z_q_st (B,750,256)
                              (for LM target)       │
                                          ┌─────────┘
                                FlowMatchingTransformer
                                  cond_emb: 256 → 1024
                                  upsample: 25Hz → 50Hz
                                  DiffLlama: predict flow
                                          │
                                    mel (B, 1500, 128) @ 50Hz
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_matching import FlowMatchingTransformer


class AudioReconModel(nn.Module):
    """Audio reconstruction codec: features → VQ tokens → CFM → mel."""

    def __init__(
        self,
        # Feature dimensions
        whisper_dim=1280,
        wavlm_dim=1024,
        muq_dim=1024,
        # VQ
        codebook_size=8192,
        codebook_dim=256,
        # CFM
        mel_dim=128,
        hidden_size=1024,
        num_layers=22,
        num_heads=16,
        cfg_drop_prob=0.2,
        sigma=1e-5,
        time_scheduler="cos",
        cond_scale_factor=2,  # 25Hz → 50Hz
    ):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.mel_dim = mel_dim

        # ── Feature resamplers: 50Hz → 25Hz ──────────────────────────
        self.whisper_resample = nn.Conv1d(
            whisper_dim, whisper_dim, kernel_size=4, stride=2, padding=1
        )
        self.wavlm_resample = nn.Conv1d(
            wavlm_dim, wavlm_dim, kernel_size=4, stride=2, padding=1
        )
        # MuQ is already at 25Hz — no resampling needed

        # ── Projection: concat dim → codebook dim ────────────────────
        concat_dim = whisper_dim + wavlm_dim + muq_dim  # 3328
        self.in_proj = nn.Linear(concat_dim, codebook_dim)

        # ── VQ codebook ──────────────────────────────────────────────
        self.vq_codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.normal_(self.vq_codebook.weight, mean=0.0, std=0.02)

        # ── CFM decoder ──────────────────────────────────────────────
        self.cfm = FlowMatchingTransformer(
            mel_dim=mel_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            cfg_drop_prob=cfg_drop_prob,
            cond_dim=codebook_dim,
            cond_scale_factor=cond_scale_factor,
            sigma=sigma,
            time_scheduler=time_scheduler,
        )

    # ==================================================================
    #  VQ quantization
    # ==================================================================

    def _vq_quantize(self, z_e):
        """Vector-quantize projected features.

        Args:
            z_e: (B, T, codebook_dim) projected features

        Returns:
            z_q_st:      (B, T, codebook_dim) quantized (straight-through)
            codes:       (B, T) discrete codebook indices
            commit_loss: scalar commitment loss
        """
        B, T, D = z_e.shape
        z_e_flat = z_e.reshape(-1, D)  # (B*T, D)

        # Find nearest codebook entry
        distances = torch.cdist(z_e_flat, self.vq_codebook.weight)  # (B*T, K)
        codes = distances.argmin(dim=-1)  # (B*T,)

        # Codebook lookup
        z_q = self.vq_codebook(codes).reshape(B, T, D)  # (B, T, D)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss: push encoder output towards codebook entries
        commit_loss = F.mse_loss(z_e, z_q.detach())

        codes = codes.reshape(B, T)
        return z_q_st, codes, commit_loss

    # ==================================================================
    #  Encode
    # ==================================================================

    def encode(self, whisper_feat, wavlm_feat, muq_feat):
        """Encode features → VQ codes.

        Args:
            whisper_feat: (B, T_w,  1280) @ 50Hz
            wavlm_feat:  (B, T_wl, 1024) @ 50Hz
            muq_feat:    (B, T_m,  1024) @ 25Hz

        Returns:
            z_q_st:      (B, T, codebook_dim) quantized embeddings (for training)
            codes:       (B, T) discrete VQ codes
            commit_loss: scalar
        """
        # Resample whisper & wavlm from 50Hz → 25Hz  (Conv1d expects B,C,T)
        w = self.whisper_resample(whisper_feat.transpose(1, 2)).transpose(1, 2)
        wl = self.wavlm_resample(wavlm_feat.transpose(1, 2)).transpose(1, 2)
        m = muq_feat  # already 25Hz

        # Align lengths (take minimum)
        min_len = min(w.shape[1], wl.shape[1], m.shape[1])
        w = w[:, :min_len, :]
        wl = wl[:, :min_len, :]
        m = m[:, :min_len, :]

        # Concat along feature dimension
        concat_feat = torch.cat([w, wl, m], dim=-1)  # (B, T, 3328)

        # Project to codebook dimension
        z_e = self.in_proj(concat_feat)  # (B, T, codebook_dim)

        # Vector quantize
        z_q_st, codes, commit_loss = self._vq_quantize(z_e)

        return z_q_st, codes, commit_loss

    # ==================================================================
    #  Training forward
    # ==================================================================

    def forward(self, whisper_feat, wavlm_feat, muq_feat, mel, mel_mask):
        """Training forward pass.

        Args:
            whisper_feat: (B, T_w,  1280) @ 50Hz
            wavlm_feat:  (B, T_wl, 1024) @ 50Hz
            muq_feat:    (B, T_m,  1024) @ 25Hz
            mel:         (B, T_mel, mel_dim) target mel spectrogram @ 50Hz
            mel_mask:    (B, T_mel) mask (1 = valid)

        Returns:
            dict:
                "cfm_output": (noise, x, flow_pred, mask) for flow loss
                "commit_loss": VQ commitment loss (scalar)
                "codes": (B, T) discrete VQ codes
        """
        # Encode: resample → concat → proj → VQ
        z_q_st, codes, commit_loss = self.encode(whisper_feat, wavlm_feat, muq_feat)

        # CFM: z_q_st (256-dim) → cond_emb (1024) → upsample → DiffLlama
        cfm_results = self.cfm(mel, mel_mask, z_q_st)

        return {
            "cfm_output": cfm_results["output"],
            "commit_loss": commit_loss,
            "codes": codes,
        }

    # ==================================================================
    #  Inference
    # ==================================================================

    @torch.no_grad()
    def decode_from_codes(
        self, codes, mel_mask=None, n_timesteps=32, cfg=1.0, rescale_cfg=0.75
    ):
        """Decode mel from discrete VQ codes (for LM inference pipeline).

        Args:
            codes:       (B, T) discrete VQ codes
            mel_mask:    (B, T_mel) mask for output mel
            n_timesteps: Euler ODE steps
            cfg:         classifier-free guidance scale
            rescale_cfg: CFG rescaling weight

        Returns:
            mel: (B, T_mel, mel_dim) generated mel spectrogram
        """
        z_q = self.vq_codebook(codes)  # (B, T, codebook_dim)
        return self.cfm.generate(z_q, mel_mask, n_timesteps, cfg, rescale_cfg)

    @torch.no_grad()
    def decode_from_features(
        self,
        whisper_feat,
        wavlm_feat,
        muq_feat,
        mel_mask=None,
        n_timesteps=32,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        """Encode features then decode to mel (reconstruction inference).

        Returns:
            mel:   (B, T_mel, mel_dim) generated mel spectrogram
            codes: (B, T) VQ codes
        """
        z_q_st, codes, _ = self.encode(whisper_feat, wavlm_feat, muq_feat)
        mel = self.cfm.generate(z_q_st, mel_mask, n_timesteps, cfg, rescale_cfg)
        return mel, codes


# ======================================================================
#  Test
# ======================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2

    # --- Simulated 30s audio features ---
    whisper_feat = torch.randn(B, 1500, 1280).to(device)  # 50Hz
    wavlm_feat = torch.randn(B, 1500, 1024).to(device)  # 50Hz
    muq_feat = torch.randn(B, 750, 1024).to(device)  # 25Hz

    # --- Target mel (50Hz, 128 bins) ---
    mel = torch.randn(B, 1500, 128).to(device)
    mel_mask = torch.ones(B, 1500).to(device)

    # --- Build model (small for testing) ---
    model = AudioReconModel(
        mel_dim=128,
        hidden_size=256,  # small for quick test
        num_layers=4,
        num_heads=8,
        codebook_size=8192,
        codebook_dim=256,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Training forward ---
    print("\n--- Training forward ---")
    results = model(whisper_feat, wavlm_feat, muq_feat, mel, mel_mask)

    noise, x, flow_pred, mask = results["cfm_output"]
    print(f"  flow_pred shape: {flow_pred.shape}")  # (2, 1500, 128)
    print(f"  mask shape:      {mask.shape}")  # (2, 1500, 1)
    print(f"  commit_loss:     {results['commit_loss'].item():.4f}")
    print(f"  codes shape:     {results['codes'].shape}")  # (2, 750)

    # Compute flow matching loss
    flow_gt = x - (1 - 1e-5) * noise
    diff_loss = F.l1_loss(flow_pred, flow_gt, reduction="none").float() * mask
    diff_loss = diff_loss.sum() / (mask.sum() * mel.shape[-1])
    print(f"  flow_loss:       {diff_loss.item():.4f}")

    # --- Decode from codes ---
    print("\n--- Decode from codes ---")
    codes = results["codes"]
    gen_mel = model.decode_from_codes(codes, n_timesteps=4)
    print(f"  generated mel shape: {gen_mel.shape}")  # (2, 1500, 128)

    # --- Decode from features ---
    print("\n--- Decode from features ---")
    gen_mel2, codes2 = model.decode_from_features(
        whisper_feat, wavlm_feat, muq_feat, n_timesteps=4
    )
    print(f"  generated mel shape: {gen_mel2.shape}")  # (2, 1500, 128)
    print(f"  codes match: {torch.equal(codes, codes2)}")

    print("\nAll tests passed!")
