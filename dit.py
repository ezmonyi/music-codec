"""
DiT (Diffusion Transformer) for Flow Matching.

Architecture adapted from CosyVoice cosyvoice/flow/DiT/dit.py.
Interface compatible with DiffLlama: forward(x, t, cond, x_mask) -> output.

Used as the flow velocity estimator in ConditionalCFM / FlowMatchingTransformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dit_modules import (
    TimestepEmbedding,
    CausalConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    RotaryEmbedding,
)


class InputEmbedding(nn.Module):
    """Project noised mel x and condition cond to model dim."""

    def __init__(self, mel_dim, cond_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + cond_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond):
        # x: (B, T, mel_dim), cond: (B, T, cond_dim)
        x = self.proj(torch.cat([x, cond], dim=-1))
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
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, cond_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(self, x, t, cond, x_mask, return_dict=False):
        """Predict flow velocity.

        Args:
            x: (B, T, mel_dim) noised mel
            t: (B,) diffusion timestep in [0, 1]
            cond: (B, T, cond_dim) condition embedding
            x_mask: (B, T) mask, 1=valid, 0=padding
            return_dict: if True, return dict with 'output' and 'hidden_states'

        Returns:
            output: (B, T, mel_dim) predicted flow velocity
            if return_dict: {"output": ..., "hidden_states": [...]}
        """
        batch, seq_len = x.shape[0], x.shape[1]

        if t.ndim == 0:
            t = t.unsqueeze(0).expand(batch)

        t_emb = self.time_embed(t)
        x = self.input_embed(x, cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        # x_mask: 1=valid, 0=padding. Attention mask: True=attend, False=mask out.
        # We need both query and key positions valid to attend.
        if x_mask is not None:
            # (B, T) -> (B, 1, T, T): valid[b,q,k] = mask[b,q] & mask[b,k]
            mask_bool = x_mask.bool()
            attn_mask = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)  # (B, T, T)
        else:
            attn_mask = None

        if self.long_skip_connection is not None:
            residual = x

        all_hidden = []
        for block in self.transformer_blocks:
            x = block(x, t_emb, mask=attn_mask, rope=rope)
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
