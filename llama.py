"""
DiffLlama: Non-causal Llama with Adaptive RMSNorm for flow matching.

Copied from SoulX-Singer (Amphion). Key components:
- SinusoidalPosEmb: sinusoidal timestep embedding
- LlamaAdaptiveRMSNorm: LayerNorm conditioned on diffusion timestep
- LlamaNARDecoderLayer: bidirectional attention layer
- DiffLlama: full model (mel_mlp + cond_mlp + N x LlamaNARDecoderLayer + mel_out_mlp)
"""

from transformers import LlamaConfig, LlamaModel
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        """Override to adaptive layer norm"""
        super().__init__(config, layer_idx)
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DiffLlama(LlamaModel):
    """Non-causal Llama for flow velocity estimation.

    Input flow:
        mel_mlp(xt) + cond_mlp(cond) → hidden_states
        diffusion_step → sinusoidal → MLP → AdaptiveRMSNorm (per layer)
        N x LlamaNARDecoderLayer (bidirectional attention)
        → mel_out_mlp → flow prediction
    """

    def __init__(
        self,
        mel_dim=128,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        config=LlamaConfig(0, 256, 1024, 1, 1),
    ):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    LlamaConfig(
                        hidden_size=hidden_size,
                        num_attention_heads=num_heads,
                        max_position_embeddings=4096,
                        intermediate_size=hidden_size * 4,
                    ),
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_mlp = nn.Sequential(
            nn.Linear(mel_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, mel_dim),
        )

        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )

        self.embed_tokens = None
        self.post_init()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Create NON-CAUSAL (bidirectional) attention mask."""
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ):
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len
            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        x,
        diffusion_step,
        cond,
        x_mask,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Args:
            x: (B, T, mel_dim) noised mel
            diffusion_step: (B,) timestep
            cond: (B, T, hidden_size) condition embedding
            x_mask: (B, T) mask
        Returns:
            flow prediction (B, T, mel_dim)
        """
        batch_size, seq_length, _ = x.shape

        # Condition MLP
        cond_embedding = self.cond_mlp(cond)  # (B, T, hidden_size)

        # Mel MLP
        x = self.mel_mlp(x)  # (B, T, hidden_size)

        # Diffusion step embedding → AdaptiveRMSNorm
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step = self.diff_step_mlp(diffusion_step)  # (B, hidden_size)

        # Add condition to mel
        x = x + cond_embedding

        inputs_embeds = x
        attention_mask = x_mask

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cond_embedding=diffusion_step,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step)

        # Output projection: hidden_size → mel_dim
        hidden_states = self.mel_out_mlp(hidden_states)

        return hidden_states
