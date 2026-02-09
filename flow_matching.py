"""
Conditional Flow Matching with DiffLlama backbone.

Adapted from SoulX-Singer (Amphion) for audio codec reconstruction.
Simplified: no prompt mechanism, no REPA/CTC auxiliary losses.
"""

import os
import sys
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama import DiffLlama


class FlowMatchingTransformer(nn.Module):
    """Conditional Flow Matching decoder.

    Condition flow:
        cond_code (B, T_cond, cond_dim)
        → cond_emb: Linear(cond_dim, hidden_size)
        → resampling_layers: ConvTranspose1d (optional, e.g. 25Hz→50Hz)
        → DiffLlama: predict flow velocity
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

        # Condition upsampling (e.g., 25Hz → 50Hz)
        if cond_scale_factor != 1:
            self.do_resampling = True
            assert math.log2(cond_scale_factor).is_integer()
            up_layers = []
            for _ in range(int(math.log2(cond_scale_factor))):
                up_layers.extend(
                    [
                        nn.ConvTranspose1d(
                            hidden_size,
                            hidden_size,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        nn.GELU(),
                    ]
                )
            self.resampling_layers = nn.Sequential(*up_layers)
        else:
            self.do_resampling = False

        # Flow velocity estimator
        self.diff_estimator = DiffLlama(
            mel_dim=mel_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
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

    def process_cond(self, cond_code, target_len=None):
        """Embed condition and upsample to mel frame rate.

        Args:
            cond_code: (B, T_cond, cond_dim) e.g. VQ embeddings at 25Hz
            target_len: optional target time dim (mel frames) for alignment

        Returns:
            cond: (B, T_mel, hidden_size)
        """
        cond = self.cond_emb(cond_code)  # (B, T_cond, hidden_size)

        if self.do_resampling:
            cond = self.resampling_layers(cond.transpose(1, 2)).transpose(
                1, 2
            )  # upsample

        if target_len is not None:
            if cond.shape[1] >= target_len:
                cond = cond[:, :target_len, :]
            else:
                padding_frames = target_len - cond.shape[1]
                last_frame = cond[:, -1:, :]
                padding = last_frame.repeat(1, padding_frames, 1)
                cond = torch.cat([cond, padding], dim=1)

        return cond

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
        t_expand = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        z = torch.randn(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)

        # xt = (1 - (1-sigma)*t) * z + t * x
        xt = (1 - (1 - self.sigma) * t_expand) * z + t_expand * x

        return xt, z

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def loss_t(self, x, x_mask, t, cond):
        """Compute flow matching loss at a given timestep.

        Args:
            x:      (B, T, mel_dim) target mel
            x_mask: (B, T) mask (1 = valid, 0 = padding)
            t:      (B,) timestep
            cond:   (B, T, hidden_size) processed condition

        Returns:
            dict with "output": (noise, x, flow_pred, final_mask)
        """
        xt, z = self.forward_diffusion(x, t)

        # CFG dropout: randomly zero out condition during training
        if self.training and self.cfg_drop_prob > 0:
            keep = (torch.rand(x.shape[0], device=x.device) > self.cfg_drop_prob).float()
            cond = cond * keep[:, None, None]

        flow_pred = self.diff_estimator(xt, t, cond, x_mask)  # (B, T, mel_dim)

        final_mask = x_mask[..., None]  # (B, T, 1)

        return {"output": (z, x, flow_pred, final_mask)}

    def compute_loss(self, x, x_mask, cond):
        """Sample timestep and compute flow matching loss."""
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)

        if self.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)

        return self.loss_t(x, x_mask, t, cond)

    def forward(self, x, x_mask, cond_code):
        """Training forward: embed condition → compute loss.

        Args:
            x:         (B, T_mel, mel_dim) target mel spectrogram
            x_mask:    (B, T_mel) mask
            cond_code: (B, T_cond, cond_dim) raw condition (e.g. VQ embeddings)
        """
        T = x.shape[1]
        cond = self.process_cond(cond_code, target_len=T)
        return self.compute_loss(x, x_mask, cond)

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reverse_diffusion(
        self, cond, x_mask=None, n_timesteps=32, cfg=1.0, rescale_cfg=0.75
    ):
        """Generate mel via Euler ODE from processed condition.

        Args:
            cond:        (B, T, hidden_size) processed condition
            x_mask:      (B, T) mask
            n_timesteps: Euler integration steps
            cfg:         classifier-free guidance scale (0 = no guidance)
            rescale_cfg: CFG rescaling weight

        Returns:
            xt: (B, T, mel_dim) generated mel spectrogram
        """
        h = 1.0 / n_timesteps
        B, T, _ = cond.shape

        if x_mask is None:
            x_mask = torch.ones(B, T, device=cond.device)

        z = torch.randn(
            (B, T, self.mel_dim), dtype=cond.dtype, device=cond.device
        )
        xt = z

        for i in range(n_timesteps):
            t = (i + 0.5) * h * torch.ones(B, dtype=cond.dtype, device=cond.device)
            flow_pred = self.diff_estimator(xt, t, cond, x_mask)

            # Classifier-free guidance
            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(
                    xt, t, torch.zeros_like(cond), x_mask
                )
                pos_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescaled = flow_pred_cfg * pos_std / flow_pred_cfg.std()
                flow_pred = rescale_cfg * rescaled + (1 - rescale_cfg) * flow_pred_cfg

            xt = xt + flow_pred * h

        return xt

    @torch.no_grad()
    def generate(
        self, cond_code, x_mask=None, n_timesteps=32, cfg=1.0, rescale_cfg=0.75
    ):
        """Generate mel from raw condition code (convenience wrapper).

        Args:
            cond_code: (B, T_cond, cond_dim) raw condition (e.g. VQ embeddings at 25Hz)
            x_mask:    (B, T_mel) mask for output mel
            n_timesteps: Euler steps
            cfg:       CFG scale
            rescale_cfg: CFG rescaling weight

        Returns:
            mel: (B, T_mel, mel_dim)
        """
        cond = self.process_cond(cond_code)  # (B, T_mel, hidden_size)
        T = cond.shape[1]

        if x_mask is None:
            x_mask = torch.ones(cond.shape[0], T, device=cond.device)

        return self.reverse_diffusion(cond, x_mask, n_timesteps, cfg, rescale_cfg)
