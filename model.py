"""
AudioReconModel: Audio reconstruction codec using Conditional Flow Matching.

Architecture (see config.yaml for hyperparameters):

    Whisper feat (B, 1500, 1280) @ 50Hz ─┐
    WavLM  feat (B, 1500, 1024) @ 50Hz ──┤ resample ──→ all @ 25Hz
    MuQ    feat (B,  750, 1024) @ 25Hz ──┘
                                          │
                                    concat dim=-1
                                   (B, 750, 4352)
                                          │
                                    in_proj (Linear or MLP)
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
from typing import Optional, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_matching import FlowMatchingTransformer


class ResidualVQ(nn.Module):
    """
    Residual VQ aligned with SoundStream / mucodec descript_quantize3 style:
    - Per layer: proj_down → VQ(codebook_dim) → proj_up; residual = residual - proj_up(z_q).
    - Commitment loss: encoder → codebook. Codebook loss: codebook → encoder (Improved VQGAN).
    - Optional L2-normalized lookup (ViT-VQGAN) for stability and codebook usage.
    - Output for condition: concat of quantized 16-dim per layer (time-aligned).
    """

    def __init__(
        self,
        rvq_hidden_dim: int,
        codebook_sizes: List[int],
        codebook_dim: int = 16,
        use_l2_norm: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        entropy_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.rvq_hidden_dim = rvq_hidden_dim
        self.codebook_sizes = list(codebook_sizes)
        self.codebook_dim = codebook_dim
        self.n_layers = len(self.codebook_sizes)
        self.use_l2_norm = use_l2_norm
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.entropy_loss_weight = entropy_loss_weight

        self.proj_downs = nn.ModuleList()
        self.codebooks = nn.ModuleList()
        self.proj_ups = nn.ModuleList()
        for K in self.codebook_sizes:
            self.proj_downs.append(nn.Linear(rvq_hidden_dim, codebook_dim))
            self.codebooks.append(nn.Embedding(K, codebook_dim))
            self.proj_ups.append(nn.Linear(codebook_dim, rvq_hidden_dim))
        for m in self.proj_downs:
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in self.proj_ups:
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for emb in self.codebooks:
            nn.init.normal_(emb.weight, 0.0, 0.02)

    def forward(self, z_e):
        """z_e: (B, T, rvq_hidden_dim). Returns z_q_concat (B,T,n*16), codes (B,T,n), commitment_loss, codebook_loss, entropy_loss."""
        B, T, D = z_e.shape
        residual = z_e
        z_q_list = []
        codes_list = []
        commitment_losses = []
        codebook_losses = []
        entropy_losses = []
        for i in range(self.n_layers):
            z_e_i = self.proj_downs[i](residual)  # (B, T, codebook_dim)
            z_e_flat = z_e_i.reshape(-1, self.codebook_dim)
            cb = self.codebooks[i].weight  # (K, codebook_dim)
            if self.use_l2_norm:
                z_e_flat = F.normalize(z_e_flat, dim=-1)
                cb = F.normalize(cb, dim=-1)
            distances = torch.cdist(z_e_flat, cb)
            codes_i = distances.argmin(dim=-1)
            z_q_i = self.codebooks[i](codes_i).reshape(B, T, self.codebook_dim)
            # Straight-through (reference: z_q = z_e + (z_q - z_e).detach())
            z_q_st_i = z_e_i + (z_q_i - z_e_i).detach()
            commitment_losses.append(F.mse_loss(z_e_i, z_q_i.detach()))
            codebook_losses.append(F.mse_loss(z_q_i, z_e_i.detach()))
            
            # Entropy loss for this layer
            if self.entropy_loss_weight > 0 and self.training:
                flat_z_e = rearrange(z_e_i, "b t d -> (b t) d")
                z_e_norm = F.normalize(flat_z_e, dim=-1)
                cb_norm = F.normalize(self.codebooks[i].weight, dim=-1)
                # cosine similarity → soft assignment（temperature 控制锐度）
                logits = z_e_norm @ cb_norm.t()  # (BT, codebook_size)，可微
                temperature = 0.1  # 较小温度 → 接近 hard assignment 但保留梯度
                soft_assign = F.softmax(logits / temperature, dim=-1)  # (BT, codebook_size)
                # 求平均使用分布
                avg_probs = soft_assign.mean(0)  # (codebook_size,)
                # 最大化 entropy = -sum(p * log(p))
                max_entropy = math.log(self.codebook_sizes[i])
                entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
                entropy_loss_i = (max_entropy - entropy) / max_entropy  # 归一化到 [0, 1]
                entropy_losses.append(entropy_loss_i)
            else:
                entropy_losses.append(torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype))
            
            residual = residual - self.proj_ups[i](z_q_i.detach())
            if self.use_ema and self.training:
                self._ema_update_codebook(i, z_e_i.detach(), codes_i, B * T)
            z_q_list.append(z_q_st_i)
            codes_list.append(codes_i.reshape(B, T))
        z_q_concat = torch.cat(z_q_list, dim=-1)
        codes = torch.stack(codes_list, dim=-1)
        commitment_loss = sum(commitment_losses)
        codebook_loss = sum(codebook_losses)
        entropy_loss = sum(entropy_losses) * self.entropy_loss_weight if self.entropy_loss_weight > 0 else torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        return z_q_concat, codes, commitment_loss, codebook_loss, entropy_loss

    def _ema_update_codebook(self, layer_idx: int, z_e_flat: torch.Tensor, codes_flat: torch.Tensor, n: int):
        """EMA update: codebook[k] = (1-decay)*codebook[k] + decay*mean(z_e where codes==k)."""
        K = self.codebook_sizes[layer_idx]
        D = self.codebook_dim
        with torch.no_grad():
            one_hot = F.one_hot(codes_flat, K).float()  # (n, K)
            count = one_hot.sum(0).clamp(min=1)  # (K,)
            sum_ = one_hot.T @ z_e_flat  # (K, D)
            mean = sum_ / count.unsqueeze(-1)
            emb = self.codebooks[layer_idx].weight
            emb.data = (1 - self.ema_decay) * emb.data + self.ema_decay * mean

    def lookup_codes(self, codes: torch.Tensor):
        """codes: (B, T, n_layers). Returns (B, T, n_layers * codebook_dim)."""
        z_list = [self.codebooks[i](codes[..., i]) for i in range(self.n_layers)]
        return torch.cat(z_list, dim=-1)


class AudioReconModel(nn.Module):
    """Audio reconstruction codec: features → VQ tokens (single VQ or RVQ) → CFM → mel."""

    def __init__(
        self,
        # Feature dimensions
        whisper_dim=1280,
        wavlm_dim=1024,
        muq_dim=2048,
        # VQ: single-layer
        codebook_size=8192,
        codebook_dim=256,
        # RVQ: 8-layer residual VQ (overrides single VQ when use_rvq=True)
        use_rvq: bool = False,
        rvq_codebook_sizes: Optional[Sequence[int]] = None,
        rvq_hidden_dim: int = 256,
        rvq_codebook_dim: int = 16,
        # in_proj: concat_dim → codebook_dim (single) or rvq_hidden_dim (RVQ)
        in_proj_hidden_dims: Optional[Sequence[int]] = None,
        in_proj_dropout: float = 0.0,
        # CFM
        mel_dim=128,
        hidden_size=1024,
        num_layers=22,
        num_heads=16,
        cfg_drop_prob=0.2,
        sigma=1e-5,
        time_scheduler="cos",
        cond_scale_factor=2,  # 25Hz → 50Hz
        cfm_cond_dim: int = 256,  # condition dim fed to CFM (256 for single VQ; RVQ uses cond_proj 128→256)
        use_codebook_ema: bool = False,
        ema_decay: float = 0.99,
        entropy_loss_weight: float = 0.0,
        codebook_init_std: float = 0.02,
        codebook_init: str = "normal",
        vq_pre_batch_norm: bool = False,
        cfm_gradient_checkpointing: bool = False,
        fm_only: bool = False,
        post_vq_proj_dims: Optional[Sequence[int]] = None,
        estimator_type: str = "dit",
    ):
        super().__init__()

        self.fm_only = fm_only
        self.use_rvq = use_rvq
        self.use_codebook_ema = use_codebook_ema
        self.ema_decay = ema_decay
        self.mel_dim = mel_dim
        self.entropy_loss_weight = entropy_loss_weight
        concat_dim = whisper_dim + wavlm_dim + muq_dim  # 4352

        if use_rvq:
            rvq_sizes = list(rvq_codebook_sizes) if rvq_codebook_sizes else [1024] * 8
            self.n_rvq_layers = len(rvq_sizes)
            self.codebook_dim = self.n_rvq_layers * rvq_codebook_dim  # concat cond dim, e.g. 128
            self.codebook_size = sum(rvq_sizes)  # for logging; codes are (B,T,n_layers)
            self.rvq_cond_dim = self.n_rvq_layers * rvq_codebook_dim
        else:
            self.codebook_dim = codebook_dim
            self.codebook_size = codebook_size
            self.n_rvq_layers = 0
            self.rvq_cond_dim = 0

        # ── Feature resamplers: 50Hz → 25Hz ──────────────────────────
        self.whisper_resample = nn.Conv1d(
            whisper_dim, whisper_dim, kernel_size=4, stride=2, padding=1
        )
        self.wavlm_resample = nn.Conv1d(
            wavlm_dim, wavlm_dim, kernel_size=4, stride=2, padding=1
        )

        # ── Projection: concat → single-VQ dim or RVQ hidden dim ────────────────────
        out_dim = rvq_hidden_dim if use_rvq else codebook_dim
        self.vq_pre_batch_norm = vq_pre_batch_norm
        if vq_pre_batch_norm:
            self.pre_vq_bn = nn.BatchNorm1d(out_dim)
        else:
            self.pre_vq_bn = None
        if not in_proj_hidden_dims:
            self.in_proj = nn.Linear(concat_dim, out_dim)
        else:
            layers = []
            dims = [concat_dim] + list(in_proj_hidden_dims) + [out_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
                    if in_proj_dropout > 0:
                        layers.append(nn.Dropout(in_proj_dropout))
            self.in_proj = nn.Sequential(*layers)
            for m in self.in_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if use_rvq:
            self.rvq = ResidualVQ(
                rvq_hidden_dim, rvq_sizes, rvq_codebook_dim,
                use_ema=use_codebook_ema, ema_decay=ema_decay,
                entropy_loss_weight=entropy_loss_weight,
            )
            self.cond_proj = nn.Linear(self.rvq_cond_dim, cfm_cond_dim)
            nn.init.normal_(self.cond_proj.weight, 0.0, 0.02)
            if self.cond_proj.bias is not None:
                nn.init.zeros_(self.cond_proj.bias)
            self.vq_codebook = None
        else:
            self.vq_codebook = nn.Embedding(codebook_size, codebook_dim)
            self._init_codebook(self.vq_codebook.weight, codebook_init, codebook_init_std)
            self.rvq = None
            self.cond_proj = None

        # ── Post-VQ projection: codebook_dim → cfm_cond_dim (when dims differ) ──────────────────────
        if post_vq_proj_dims:
            layers = []
            dims = [codebook_dim] + list(post_vq_proj_dims)
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
            self.post_vq_proj = nn.Sequential(*layers)
            for m in self.post_vq_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            self.post_vq_proj = None

        # ── CFM decoder (always receives cfm_cond_dim) ──────────────────────────────────────────────
        self.cfm = FlowMatchingTransformer(
            mel_dim=mel_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            cfg_drop_prob=cfg_drop_prob,
            cond_dim=cfm_cond_dim,
            cond_scale_factor=cond_scale_factor,
            sigma=sigma,
            time_scheduler=time_scheduler,
            gradient_checkpointing=cfm_gradient_checkpointing,
            estimator_type=estimator_type,
        )

    @staticmethod
    def _init_codebook(weight: torch.Tensor, init_type: str, std: float):
        """Initialize codebook for better utilization.
        - normal: Gaussian(0, std). Use larger std (e.g. 0.5, 1.0) to spread entries so more get used early.
        - uniform_unit: sample normal then L2-normalize rows (uniform on sphere); scale by std for magnitude.
        """
        if init_type == "normal":
            nn.init.normal_(weight, mean=0.0, std=std)
        elif init_type == "uniform_unit":
            nn.init.normal_(weight, mean=0.0, std=1.0)
            with torch.no_grad():
                weight.data = F.normalize(weight.data, dim=-1) * std
        else:
            nn.init.normal_(weight, mean=0.0, std=std)

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
            entropy_loss: scalar entropy loss
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

        # Entropy loss
        if self.entropy_loss_weight > 0 and self.training:
            flat_z_e = rearrange(z_e, "b t d -> (b t) d")
            z_e_norm = F.normalize(flat_z_e, dim=-1)
            cb_norm = F.normalize(self.vq_codebook.weight, dim=-1)
            # cosine similarity → soft assignment（temperature 控制锐度）
            logits = z_e_norm @ cb_norm.t()  # (BT, codebook_size)，可微
            temperature = 0.1  # 较小温度 → 接近 hard assignment 但保留梯度
            soft_assign = F.softmax(logits / temperature, dim=-1)  # (BT, codebook_size)
            # 求平均使用分布
            avg_probs = soft_assign.mean(0)  # (codebook_size,)
            # 最大化 entropy = -sum(p * log(p))
            max_entropy = math.log(self.codebook_size)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
            entropy_loss = (max_entropy - entropy) / max_entropy  # 归一化到 [0, 1]
            entropy_loss = entropy_loss * self.entropy_loss_weight
        else:
            entropy_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)

        if self.use_codebook_ema and self.training:
            self._ema_update_vq_codebook(z_e_flat.detach(), codes, B * T)

        codes = codes.reshape(B, T)
        return z_q_st, codes, commit_loss, entropy_loss

    def _ema_update_vq_codebook(self, z_e_flat: torch.Tensor, codes_flat: torch.Tensor, n: int):
        """EMA update for single VQ codebook."""
        K = self.vq_codebook.num_embeddings
        D = self.vq_codebook.embedding_dim
        with torch.no_grad():
            one_hot = F.one_hot(codes_flat, K).float()  # (n, K)
            count = one_hot.sum(0).clamp(min=1)  # (K,)
            sum_ = one_hot.T @ z_e_flat  # (K, D)
            mean = sum_ / count.unsqueeze(-1)
            self.vq_codebook.weight.data = (
                (1 - self.ema_decay) * self.vq_codebook.weight.data + self.ema_decay * mean
            )

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
            z_q_st:      (B, T, codebook_dim) or (B, T, rvq_cond_dim) after cond_proj for CFM
            codes:       (B, T) single VQ or (B, T, n_layers) RVQ
            commit_loss: scalar
            codebook_loss: scalar (RVQ only)
            entropy_loss: scalar
        """
        # Ensure all inputs are 3D (B, T, D)
        if whisper_feat.dim() != 3:
            raise ValueError(f"whisper_feat must be 3D (B, T, D), got shape {whisper_feat.shape}")
        if wavlm_feat.dim() != 3:
            raise ValueError(f"wavlm_feat must be 3D (B, T, D), got shape {wavlm_feat.shape}")
        if muq_feat.dim() != 3:
            raise ValueError(f"muq_feat must be 3D (B, T, D), got shape {muq_feat.shape}")
        
        w = self.whisper_resample(whisper_feat.transpose(1, 2)).transpose(1, 2)
        wl = self.wavlm_resample(wavlm_feat.transpose(1, 2)).transpose(1, 2)
        m = muq_feat
        min_len = min(w.shape[1], wl.shape[1], m.shape[1])
        w = w[:, :min_len, :]
        wl = wl[:, :min_len, :]
        m = m[:, :min_len, :]
        concat_feat = torch.cat([w, wl, m], dim=-1)  # (B, T, 4352)
        z_e = self.in_proj(concat_feat)
        if self.pre_vq_bn is not None:
            # BatchNorm1d expects (N, C, L); z_e is (B, T, D) -> (B, D, T)
            z_e = self.pre_vq_bn(z_e.transpose(1, 2)).transpose(1, 2)

        if self.fm_only:
            z_out = self.post_vq_proj(z_e) if self.post_vq_proj is not None else z_e
            return z_out, None, None, None
        if self.use_rvq:
            z_q_concat, codes, commitment_loss, codebook_loss, entropy_loss = self.rvq(z_e)
            z_q_st = self.cond_proj(z_q_concat)  # (B, T, cfm_cond_dim)
            return z_q_st, codes, commitment_loss, codebook_loss, entropy_loss
        else:
            z_q_st, codes, commit_loss, entropy_loss = self._vq_quantize(z_e)
            if self.post_vq_proj is not None:
                z_q_st = self.post_vq_proj(z_q_st)
            return z_q_st, codes, commit_loss, entropy_loss

    def get_pre_vq_features(self, whisper_feat, wavlm_feat, muq_feat):
        """Return encoder output z_e (B, T, codebook_dim) before VQ. For k-means codebook init."""
        if whisper_feat.dim() != 3 or wavlm_feat.dim() != 3 or muq_feat.dim() != 3:
            raise ValueError("All feature inputs must be 3D (B, T, D)")
        w = self.whisper_resample(whisper_feat.transpose(1, 2)).transpose(1, 2)
        wl = self.wavlm_resample(wavlm_feat.transpose(1, 2)).transpose(1, 2)
        m = muq_feat
        min_len = min(w.shape[1], wl.shape[1], m.shape[1])
        w, wl, m = w[:, :min_len, :], wl[:, :min_len, :], m[:, :min_len, :]
        concat_feat = torch.cat([w, wl, m], dim=-1)
        z_e = self.in_proj(concat_feat)
        if self.pre_vq_bn is not None:
            z_e = self.pre_vq_bn(z_e.transpose(1, 2)).transpose(1, 2)
        return z_e

    # ==================================================================
    #  Training forward
    # ==================================================================

    def forward(
        self,
        whisper_feat,
        wavlm_feat,
        muq_feat,
        mel,
        mel_mask,
        return_pred_mel: bool = False,
        mel_recon_n_steps: int = 4,
    ):
        """Training forward pass.

        Args:
            whisper_feat: (B, T_w,  1280) @ 50Hz
            wavlm_feat:  (B, T_wl, 1024) @ 50Hz
            muq_feat:    (B, T_m,  1024) @ 25Hz
            mel:         (B, T_mel, mel_dim) target mel spectrogram @ 50Hz
            mel_mask:    (B, T_mel) mask (1 = valid)
            return_pred_mel: if True, run short ODE and return pred_mel (for mel_recon / disc loss)
            mel_recon_n_steps: ODE steps when return_pred_mel (e.g. 4)

        Returns:
            dict:
                "cfm_output": (noise, x, flow_pred, mask) for flow loss
                "commit_loss": VQ commitment loss (scalar)
                "codes": (B, T) single VQ or (B, T, n_layers) RVQ
                "entropy_loss": entropy loss (scalar)
                "pred_mel": (B, T_mel, mel_dim) when return_pred_mel (keeps grad)
        """
        z_q_st, codes, *vq_losses = self.encode(whisper_feat, wavlm_feat, muq_feat)
        cfm_results = self.cfm(mel, mel_mask, z_q_st)
        out = {
            "cfm_output": cfm_results["output"],
            "codes": codes,
        }
        if self.fm_only:
            out["commit_loss"] = mel.new_zeros(1).squeeze()
            out["entropy_loss"] = mel.new_zeros(1).squeeze()
        elif self.use_rvq:
            out["commit_loss"] = vq_losses[0]
            out["codebook_loss"] = vq_losses[1]
            out["entropy_loss"] = vq_losses[2]
        else:
            out["commit_loss"] = vq_losses[0]
            out["entropy_loss"] = vq_losses[1]
        if return_pred_mel:
            cond = self.cfm.process_cond(z_q_st)
            out["pred_mel"] = self.cfm.reverse_diffusion_train(
                cond, mel_mask, n_timesteps=mel_recon_n_steps, cfg=0.0
            )
        return out

    # ==================================================================
    #  Inference
    # ==================================================================

    @torch.no_grad()
    def decode_from_codes(
        self, codes, mel_mask=None, n_timesteps=32, cfg=1.0, rescale_cfg=0.75
    ):
        """Decode mel from discrete VQ codes (for LM inference pipeline).

        Args:
            codes:       (B, T) single VQ or (B, T, n_layers) RVQ
            mel_mask:    (B, T_mel) mask for output mel
            n_timesteps: Euler ODE steps
            cfg:         classifier-free guidance scale
            rescale_cfg: CFG rescaling weight

        Returns:
            mel: (B, T_mel, mel_dim) generated mel spectrogram
        """
        if self.use_rvq:
            z_q = self.rvq.lookup_codes(codes)  # (B, T, rvq_cond_dim)
            z_q = self.cond_proj(z_q)  # (B, T, cfm_cond_dim)
        else:
            z_q = self.vq_codebook(codes)
            if self.post_vq_proj is not None:
                z_q = self.post_vq_proj(z_q)
        return self.cfm.generate(
            z_q, x_mask=mel_mask, n_timesteps=n_timesteps,
            cfg=cfg, rescale_cfg=rescale_cfg,
        )

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
        z_q_st, codes, *_ = self.encode(whisper_feat, wavlm_feat, muq_feat)
        mel = self.cfm.generate(
            z_q_st, x_mask=mel_mask, n_timesteps=n_timesteps,
            cfg=cfg, rescale_cfg=rescale_cfg,
        )
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

    # --- in_proj MLP variant ---
    print("\n--- in_proj MLP (hidden_dims=[1024, 512]) ---")
    model_mlp = AudioReconModel(
        mel_dim=128,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        codebook_size=8192,
        codebook_dim=256,
        in_proj_hidden_dims=[1024, 512],
        in_proj_dropout=0.1,
    ).to(device)
    z_q_st, codes_mlp, _, _ = model_mlp.encode(whisper_feat, wavlm_feat, muq_feat)
    assert z_q_st.shape == (B, 750, 256), f"z_q_st shape {z_q_st.shape}"
    assert codes_mlp.shape == (B, 750), f"codes shape {codes_mlp.shape}"
    results_mlp = model_mlp(whisper_feat, wavlm_feat, muq_feat, mel, mel_mask)
    assert results_mlp["cfm_output"][1].shape == (B, 1500, 128)
    print(f"  z_q_st shape: {z_q_st.shape}")
    print(f"  codes shape: {codes_mlp.shape}")
    print("  MLP in_proj test passed.")

    # --- RVQ 8-layer variant ---
    print("\n--- RVQ 8x1024 ---")
    model_rvq = AudioReconModel(
        mel_dim=128,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        use_rvq=True,
        rvq_codebook_sizes=[1024] * 8,
        rvq_hidden_dim=256,
        rvq_codebook_dim=16,
        cfm_cond_dim=256,
    ).to(device)
    z_rvq, codes_rvq, commit_rvq, codebook_rvq, _ = model_rvq.encode(whisper_feat, wavlm_feat, muq_feat)
    assert z_rvq.shape == (B, 750, 256), f"z_rvq {z_rvq.shape}"
    assert codes_rvq.shape == (B, 750, 8), f"codes_rvq {codes_rvq.shape}"
    out_rvq = model_rvq(whisper_feat, wavlm_feat, muq_feat, mel, mel_mask)
    assert out_rvq["codes"].shape == (B, 750, 8)
    gen_rvq = model_rvq.decode_from_codes(codes_rvq, n_timesteps=4)
    assert gen_rvq.shape == (B, 1500, 128)
    print(f"  codes shape: {codes_rvq.shape}, decode mel: {gen_rvq.shape}")
    print("  RVQ test passed.")

    print("\nAll tests passed!")
