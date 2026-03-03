"""
DiT (Diffusion Transformer) for Flow Matching.

Architecture adapted from CosyVoice cosyvoice/flow/DiT/dit.py.
Interface compatible with DiffLlama: forward(x, t, cond, x_mask) -> output.

Used as the flow velocity estimator in ConditionalCFM / FlowMatchingTransformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dit_modules import (
    TimestepEmbedding,
    CausalConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    FlashAttn3Processor,
    RotaryEmbedding,
)


class InputEmbedding(nn.Module):
    """Project noised mel x to model dim. Condition is fed via cross attention."""

    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x):
        # x: (B, T, mel_dim)
        x = self.proj(x)
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    """Diffusion Transformer for flow matching.

    Interface matches DiffLlama for drop-in replacement:
        forward(x, t, cond, x_mask, return_dict=False) -> output or dict

    Args:
        mel_dim: mel spectrogram dimension
        cond_dim: condition embedding dimension (e.g. hidden_size from cond_emb)
        dim: transformer hidden dimension
        depth: number of DiT blocks
        heads: attention heads
        dim_head: dimension per head
        dropout: dropout rate
        ff_mult: FFN expansion factor
        long_skip_connection: use long skip connection
    """

    def __init__(
        self,
        mel_dim=128,
        cond_dim=1024,
        dim=1024,
        depth=12,
        heads=16,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        long_skip_connection=False,
        use_flash_attn_3=False,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        self.use_flash_attn_3 = use_flash_attn_3
        self.gradient_checkpointing = gradient_checkpointing

        attn_processor = FlashAttn3Processor() if use_flash_attn_3 else None

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, heads=heads, dim_head=dim_head,
                ff_mult=ff_mult, dropout=dropout,
                attn_processor=attn_processor,
                context_dim=cond_dim,
            )
            for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(self, x, t, cond, x_mask, cond_mask=None, return_dict=False):
        """Predict flow velocity.

        Args:
            x: (B, T_mel, mel_dim) noised mel
            t: (B,) diffusion timestep in [0, 1]
            cond: (B, T_cond, cond_dim) condition embedding (can differ from T_mel)
            x_mask: (B, T_mel) mask, 1=valid, 0=padding
            cond_mask: (B, T_cond) mask for cond, 1=valid, 0=padding; defaults to all valid
            return_dict: if True, return dict with 'output' and 'hidden_states'

        Returns:
            output: (B, T_mel, mel_dim) predicted flow velocity
            if return_dict: {"output": ..., "hidden_states": [...]}
        """
        batch, seq_len = x.shape[0], x.shape[1]

        if t.ndim == 0:
            t = t.unsqueeze(0).expand(batch)

        t_emb = self.time_embed(t)
        x = self.input_embed(x)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if x_mask is not None:
            if self.use_flash_attn_3:
                attn_mask = x_mask.bool()  # (B, T) – FA3 processor handles varlen packing
            else:
                mask_bool = x_mask.bool()
                attn_mask = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)  # (B, T, T)
        else:
            attn_mask = None

        ctx_mask = cond_mask.bool() if cond_mask is not None else None  # True=valid for cross attn

        if self.long_skip_connection is not None:
            residual = x

        all_hidden = []
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                x = torch_checkpoint(
                    block, x, t_emb, attn_mask, rope, cond, ctx_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, t_emb, mask=attn_mask, rope=rope, context=cond, context_mask=ctx_mask)
            if return_dict:
                all_hidden.append(x.clone())

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t_emb)
        output = self.proj_out(x)

        if x_mask is not None:
            output = output.masked_fill(~x_mask.unsqueeze(-1).bool(), 0.0)

        if return_dict:
            return {"output": output, "hidden_states": all_hidden}
        return output
