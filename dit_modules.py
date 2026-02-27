"""
DiT (Diffusion Transformer) building blocks for flow matching.

Adapted from CosyVoice cosyvoice/flow/DiT/modules.py.
Self-contained: no x_transformers dependency. Includes RoPE implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rearrange(x, pattern, **kwargs):
    """Minimal rearrange for RoPE; avoid einops dependency in modules."""
    if " (d r) -> ... d r" in pattern:
        # (... (d r) -> ... d r), r=2
        b, n, d = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(b, n, d // 2, 2)
        return x
    if "... d r -> ... (d r)" in pattern:
        return x.reshape(*x.shape[:-2], -1)
    raise NotImplementedError(pattern)


# ---------------------------------------------------------------------------
# Sinusoidal position embedding (for timestep)
# ---------------------------------------------------------------------------


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim, scale=1000):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = self.scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Causal conv position embedding
# ---------------------------------------------------------------------------


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv1(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv2(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """RoPE for sequence positions. No xpos by default."""

    def __init__(self, dim, base=10000, base_rescale_factor=1.0):
        super().__init__()
        base = base * (base_rescale_factor ** (dim / (dim - 2)))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        freqs = torch.einsum("bi,j->bij", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = freqs.reshape(*freqs.shape[:-2], -1)
        return freqs, 1.0


def _rotate_half(x):
    """x: (..., d) where d is even. Rotate half for RoPE."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, scale=1.0):
    """Apply RoPE to t. t: (B, seq, dim), freqs: (B, seq, rot_dim)."""
    rot_dim = freqs.shape[-1]
    seq_len = t.shape[-2]
    orig_dtype = t.dtype

    freqs = freqs[:, -seq_len:, :]
    if isinstance(scale, torch.Tensor):
        scale = scale[:, -seq_len:, :]

    t_rot = t[..., :rot_dim]
    t_unrot = t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos() * scale) + (_rotate_half(t_rot) * freqs.sin() * scale)
    out = torch.cat((t_rot, t_unrot), dim=-1)

    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# AdaLayerNormZero (for DiT blocks)
# ---------------------------------------------------------------------------


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate="none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# ---------------------------------------------------------------------------
# Attention (self-attention with RoPE)
# ---------------------------------------------------------------------------


class AttnProcessor:
    def __call__(self, attn, x, mask=None, rope=None):
        batch_size = x.shape[0]

        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale = xpos_scale if isinstance(xpos_scale, (int, float)) else 1.0
            k_xpos_scale = 1.0 / q_xpos_scale if q_xpos_scale != 1.0 else 1.0
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask.unsqueeze(1).unsqueeze(1)
            else:
                attn_mask = mask
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        if mask is not None:
            if mask.dim() == 2:
                mask_exp = mask.unsqueeze(-1)
            else:
                mask_exp = mask[:, 0, -1].unsqueeze(-1) if mask.dim() == 4 else mask[:, :, -1].unsqueeze(-1)
            x = x.masked_fill(~mask_exp, 0.0)

        return x


def _extract_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Extract (B, T) boolean padding mask from various mask formats."""
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask.bool()
    if mask.dim() == 3:
        return mask[:, 0, :].bool()
    if mask.dim() == 4:
        return mask[:, 0, 0, :].bool()
    return None


class FlashAttn3Processor:
    """Attention processor using Flash Attention 3 (Hopper GPU, sm90).

    Requires ``flash-attn >= 2.7`` compiled with Hopper support.
    Falls back to the standard FA interface when the Hopper-specific
    ``flash_attn_interface`` module is not on the Python path.

    Padding is handled via ``flash_attn_varlen_func`` (packed sequences) so
    that padding tokens never participate in the softmax.
    """

    def __init__(self):
        try:
            from flash_attn_interface import (
                flash_attn_func,
                flash_attn_varlen_func,
            )
        except ImportError:
            from flash_attn.flash_attn_interface import (
                flash_attn_func,
                flash_attn_varlen_func,
            )
        self._fa_func = flash_attn_func
        self._fa_varlen_func = flash_attn_varlen_func

    # --------------------------------------------------------------------- #

    def __call__(
        self,
        attn: "Attention",
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope=None,
    ) -> torch.Tensor:
        B, S = x.shape[:2]

        q = attn.to_q(x)
        k = attn.to_k(x)
        v = attn.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_scale = xpos_scale if isinstance(xpos_scale, (int, float)) else 1.0
            k_scale = 1.0 / q_scale if q_scale != 1.0 else 1.0
            q = apply_rotary_pos_emb(q, freqs, q_scale)
            k = apply_rotary_pos_emb(k, freqs, k_scale)

        head_dim = attn.inner_dim // attn.heads
        q = q.view(B, S, attn.heads, head_dim)
        k = k.view(B, S, attn.heads, head_dim)
        v = v.view(B, S, attn.heads, head_dim)

        padding_mask = _extract_padding_mask(mask)

        if padding_mask is not None:
            out = self._varlen_forward(q, k, v, padding_mask)
        else:
            out = self._fa_func(q, k, v, dropout_p=0.0, causal=False)
            if isinstance(out, tuple):
                out = out[0]

        x = out.reshape(B, S, attn.inner_dim).to(q.dtype)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        return x

    # --------------------------------------------------------------------- #

    def _varlen_forward(self, q, k, v, padding_mask):
        """Run flash_attn_varlen_func on packed (unpadded) sequences."""
        B, S, H, D = q.shape

        seqlens = padding_mask.sum(-1, dtype=torch.int32)
        cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
        max_seqlen = seqlens.max().item()

        idx = padding_mask.nonzero(as_tuple=False)
        q_pack = q[idx[:, 0], idx[:, 1]]
        k_pack = k[idx[:, 0], idx[:, 1]]
        v_pack = v[idx[:, 0], idx[:, 1]]

        out = self._fa_varlen_func(
            q_pack, k_pack, v_pack,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            dropout_p=0.0,
            causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]

        result = q.new_zeros(B, S, H, D)
        result[idx[:, 0], idx[:, 1]] = out
        return result


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, processor=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        ])

        self.processor = processor or AttnProcessor()

    def forward(self, x, mask=None, rope=None):
        return self.processor(self, x, mask=mask, rope=rope)


# ---------------------------------------------------------------------------
# Cross Attention (query from x, key/value from context)
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Cross attention: q from x, k/v from context. No RoPE on context."""

    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        context_dim = context_dim or dim
        self.heads = heads
        self.inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(context_dim, self.inner_dim)
        self.to_v = nn.Linear(context_dim, self.inner_dim)
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        ])

    def forward(self, x, context, key_padding_mask=None):
        """x: (B, T_q, dim), context: (B, T_kv, context_dim), key_padding_mask: (B, T_kv) bool, True=valid."""
        B, T_q, _ = x.shape
        T_kv = context.shape[1]

        q = self.to_q(x)  # (B, T_q, inner_dim)
        k = self.to_k(context)  # (B, T_kv, inner_dim)
        v = self.to_v(context)

        head_dim = self.inner_dim // self.heads
        q = q.view(B, T_q, self.heads, head_dim).transpose(1, 2)  # (B, H, T_q, D)
        k = k.view(B, T_kv, self.heads, head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.heads, head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            # key_padding_mask: True=valid, False=pad. SDPA expects True=attend, False=mask out.
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_kv)
            attn_mask = attn_mask.expand(B, self.heads, T_q, T_kv)
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(B, T_q, self.inner_dim)

        x = self.to_out[0](x)
        x = self.to_out[1](x)
        return x


# ---------------------------------------------------------------------------
# DiT Block
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, attn_processor=None, context_dim=None):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, processor=attn_processor)

        self.cross_attn = None
        if context_dim is not None:
            self.cross_attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None, context=None, context_mask=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        if self.cross_attn is not None and context is not None:
            cx = self.cross_attn_norm(x)
            cx_out = self.cross_attn(cx, context, key_padding_mask=context_mask)
            x = x + self.cross_attn_gate * cx_out

        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)
