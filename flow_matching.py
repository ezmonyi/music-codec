"""
Conditional Flow Matching with DiT backbone.

Adapted from SoulX-Singer (Amphion) for audio codec reconstruction.
Uses DiT (Diffusion Transformer) like CosyVoice, compatible with DiffLlama interface.
"""

import os
import sys
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dit import DiT


class FlowMatchingTransformer(nn.Module):
    """Conditional Flow Matching decoder.

    Condition flow:
        cond_code (B, T_cond, cond_dim)
        → cond_emb: Linear(cond_dim, hidden_size)
        → DiT: cross attention to cond (no upsampling)
    """

    def __init__(
        self,
        mel_dim=128,
        hidden_size=1024,
        num_layers=22,
        num_heads=16,
        cfg_drop_prob=0.2,
        cond_dim=256,
        cond_scale_factor=2,
        sigma=1e-5,
        time_scheduler="cos",
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.cfg_drop_prob = cfg_drop_prob
        self.sigma = sigma
        self.time_scheduler = time_scheduler
        self.cond_scale_factor = cond_scale_factor

        # Condition embedding: cond_dim → hidden_size
        self.cond_emb = nn.Linear(cond_dim, hidden_size)

        # Flow velocity estimator (DiT with cross attention to cond)
        dim_head = hidden_size // num_heads
        self.diff_estimator = DiT(
            mel_dim=mel_dim,
            cond_dim=hidden_size,
            dim=hidden_size,
            depth=num_layers,
            heads=num_heads,
            dim_head=dim_head,
            dropout=0.1,
            ff_mult=4,
        )

        self.reset_parameters()

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)
            elif isinstance(
                m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    # ------------------------------------------------------------------
    #  Condition processing
    # ------------------------------------------------------------------

    def process_cond(self, cond_code):
        """Embed condition. No upsampling; cross attention handles different lengths.

        Args:
            cond_code: (B, T_cond, cond_dim) e.g. VQ embeddings at 25Hz

        Returns:
            cond: (B, T_cond, hidden_size)
        """
        return self.cond_emb(cond_code)

    # ------------------------------------------------------------------
    #  Forward diffusion (noising)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_diffusion(self, x, t):
        """Add noise along the flow matching interpolation path.

        Args:
            x: (B, T, mel_dim) clean mel
            t: (B,) timestep in [0, 1]

        Returns:
            xt: noised mel
            z:  noise sample
        """
        # Ensure t is 1D: (B,)
        # Handle various input shapes: (B,), (B, 1), (1, B), etc.
        if t.dim() == 0:
            t = t.unsqueeze(0)
        elif t.dim() > 1:
            # If t is (B, T) or similar, take the first element along time dim
            # or flatten and take first B elements
            if t.shape[0] == x.shape[0]:
                t = t[:, 0] if t.shape[1] > 1 else t.squeeze(-1)
            else:
                t = t.flatten()[:x.shape[0]]
        
        # Ensure t has shape (B,)
        assert t.shape[0] == x.shape[0], f"Batch size mismatch: t.shape={t.shape}, x.shape={x.shape}"
        
        # Ensure x is 3D (B, T, mel_dim)
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, T, mel_dim), got shape {x.shape}")
        
        # Expand t to (B, 1, 1) for broadcasting with (B, T, mel_dim)
        t_expand = t.view(-1, 1, 1)  # (B, 1, 1)
        z = torch.randn_like(x)  # (B, T, mel_dim) - same shape as x

        # xt = (1 - (1-sigma)*t) * z + t * x
        # Ensure shapes match for broadcasting
        assert z.shape == x.shape, f"Shape mismatch: z.shape={z.shape}, x.shape={x.shape}"
        xt = (1 - (1 - self.sigma) * t_expand) * z + t_expand * x

        return xt, z

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def loss_t(self, x, x_mask, t, cond, cond_mask=None):
        """Compute flow matching loss at a given timestep.

        Args:
            x:         (B, T_mel, mel_dim) target mel
            x_mask:    (B, T_mel) mask (1 = valid, 0 = padding)
            t:         (B,) timestep
            cond:      (B, T_cond, hidden_size) processed condition
            cond_mask: (B, T_cond) mask for cond, 1=valid; optional, defaults to all valid

        Returns:
            dict with "output": (noise, x, flow_pred, final_mask)
        """
        assert x.shape[1] == x_mask.shape[1], \
            f"Time dimension mismatch: x.shape={x.shape}, x_mask.shape={x_mask.shape}"

        xt, z = self.forward_diffusion(x, t)

        # CFG dropout: randomly zero out condition during training
        if self.training and self.cfg_drop_prob > 0:
            keep = (torch.rand(x.shape[0], device=x.device) > self.cfg_drop_prob).float()
            cond = cond * keep[:, None, None]

        flow_pred = self.diff_estimator(xt, t, cond, x_mask, cond_mask=cond_mask)

        final_mask = x_mask[..., None]  # (B, T_mel, 1)

        return {"output": (z, x, flow_pred, final_mask)}

    def compute_loss(self, x, x_mask, cond, cond_mask=None):
        """Sample timestep and compute flow matching loss."""
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)

        if self.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)

        return self.loss_t(x, x_mask, t, cond, cond_mask=cond_mask)

    def forward(self, x, x_mask, cond_code, cond_mask=None):
        """Training forward: embed condition → compute loss.

        Args:
            x:         (B, T_mel, mel_dim) target mel spectrogram
            x_mask:    (B, T_mel) mask
            cond_code: (B, T_cond, cond_dim) raw condition (e.g. VQ embeddings)
            cond_mask: (B, T_cond) mask for cond, 1=valid; optional
        """
        cond = self.process_cond(cond_code)
        return self.compute_loss(x, x_mask, cond, cond_mask=cond_mask)

    # ------------------------------------------------------------------
    #  Inference / training-time generation
    # ------------------------------------------------------------------

    def reverse_diffusion_train(
        self, cond, x_mask=None, cond_mask=None, n_timesteps=32, cfg=0.0, rescale_cfg=0.75
    ):
        """Generate mel via Euler ODE (keeps graph for gradients). Use for mel_recon / disc loss.

        Same as reverse_diffusion but without no_grad, so pred_mel has requires_grad.
        """
        h = 1.0 / n_timesteps
        B, T_cond, _ = cond.shape
        T = int(round(T_cond * self.cond_scale_factor)) if x_mask is None else x_mask.shape[1]

        if x_mask is None:
            x_mask = torch.ones(B, T, device=cond.device, dtype=cond.dtype)

        z = torch.randn(
            (B, T, self.mel_dim), dtype=cond.dtype, device=cond.device
        )
        xt = z

        for i in range(n_timesteps):
            t = (i + 0.5) * h * torch.ones(B, dtype=cond.dtype, device=cond.device)
            flow_pred = self.diff_estimator(xt, t, cond, x_mask, cond_mask=cond_mask)

            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(
                    xt, t, torch.zeros_like(cond), x_mask, cond_mask=cond_mask
                )
                pos_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescaled = flow_pred_cfg * pos_std / flow_pred_cfg.std()
                flow_pred = rescale_cfg * rescaled + (1 - rescale_cfg) * flow_pred_cfg

            xt = xt + flow_pred * h

        return xt

    @torch.no_grad()
    def reverse_diffusion(
        self, cond, x_mask=None, cond_mask=None, n_timesteps=32, cfg=1.0, rescale_cfg=0.75
    ):
        """Generate mel via Euler ODE from processed condition.

        Args:
            cond:        (B, T_cond, hidden_size) processed condition
            x_mask:      (B, T_mel) mask; if None, T_mel = T_cond * cond_scale_factor
            cond_mask:   (B, T_cond) mask for cond; optional
            n_timesteps: Euler integration steps
            cfg:         classifier-free guidance scale (0 = no guidance)
            rescale_cfg: CFG rescaling weight

        Returns:
            xt: (B, T_mel, mel_dim) generated mel spectrogram
        """
        h = 1.0 / n_timesteps
        B, T_cond, _ = cond.shape
        T = int(round(T_cond * self.cond_scale_factor)) if x_mask is None else x_mask.shape[1]

        if x_mask is None:
            x_mask = torch.ones(B, T, device=cond.device)

        z = torch.randn(
            (B, T, self.mel_dim), dtype=cond.dtype, device=cond.device
        )
        xt = z

        for i in range(n_timesteps):
            t = (i + 0.5) * h * torch.ones(B, dtype=cond.dtype, device=cond.device)
            flow_pred = self.diff_estimator(xt, t, cond, x_mask, cond_mask=cond_mask)

            # Classifier-free guidance
            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(
                    xt, t, torch.zeros_like(cond), x_mask, cond_mask=cond_mask
                )
                pos_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescaled = flow_pred_cfg * pos_std / flow_pred_cfg.std()
                flow_pred = rescale_cfg * rescaled + (1 - rescale_cfg) * flow_pred_cfg

            xt = xt + flow_pred * h

        return xt

    @torch.no_grad()
    def generate(
        self, cond_code, x_mask=None, cond_mask=None, n_timesteps=32, cfg=1.0, rescale_cfg=0.75
    ):
        """Generate mel from raw condition code (convenience wrapper).

        Args:
            cond_code: (B, T_cond, cond_dim) raw condition (e.g. VQ embeddings at 25Hz)
            x_mask:    (B, T_mel) mask for output mel; if None, T_mel = T_cond * cond_scale_factor
            cond_mask: (B, T_cond) mask for cond; optional
            n_timesteps: Euler steps
            cfg:       CFG scale
            rescale_cfg: CFG rescaling weight

        Returns:
            mel: (B, T_mel, mel_dim)
        """
        cond = self.process_cond(cond_code)  # (B, T_cond, hidden_size)
        return self.reverse_diffusion(cond, x_mask, cond_mask, n_timesteps, cfg, rescale_cfg)
