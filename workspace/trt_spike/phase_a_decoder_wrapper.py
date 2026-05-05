#!/usr/bin/env python3
"""
Phase A VQA-specialized decoder wrapper.

This wrapper converts wall-x decoder into an expert0-only dense decoder path
for VQA-specialized experiments:

  - attention_moe is ignored (current config uses attention_moe = false)
  - mlp_moe is collapsed to expert0-only dense MLP
  - no permute/unpermute
  - no expert1 / action route

The wrapper is intended as the Python-side reference implementation before
attempting TensorRT export/build.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl import Qwen2_5_VLAttention


class VQASpecializedDecoderLayer(nn.Module):
    def __init__(self, src_layer: nn.Module, dim_input0: int, force_manual_attention: bool = False):
        super().__init__()
        self.self_attn = src_layer.self_attn
        self.input_layernorm = src_layer.input_layernorm
        self.post_attention_layernorm = src_layer.post_attention_layernorm
        self.expert0 = src_layer.moe.experts[0] if src_layer.moe is not None else src_layer.mlp
        self.dim_input0 = dim_input0
        self.force_manual_attention = force_manual_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
    ):
        residual = hidden_states

        hidden_states, _ = self.input_layernorm(hidden_states)
        if self.force_manual_attention:
            hidden_states, _, present_key_value = Qwen2_5_VLAttention.forward(
                self.self_attn,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        else:
            hidden_states, _, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states, _ = self.post_attention_layernorm(hidden_states)

        expert_input = hidden_states[..., : self.dim_input0]
        expert_output = self.expert0(expert_input)
        mlp_out = torch.zeros_like(hidden_states)
        mlp_out[..., : self.dim_input0] = expert_output[..., : self.dim_input0]

        hidden_states = residual + mlp_out
        return hidden_states, present_key_value


class VQASpecializedDecoder(nn.Module):
    def __init__(self, wallx_model: nn.Module, force_manual_attention: bool = False):
        super().__init__()
        self.config = wallx_model.config
        self.base_model = wallx_model.model
        self.rotary_emb = wallx_model.model.rotary_emb
        self.norm = wallx_model.model.norm
        self.lm_head = wallx_model.lm_head

        dim_input0 = self.config.dim_inputs[0]
        self.layers = nn.ModuleList(
            [
                VQASpecializedDecoderLayer(
                    layer, dim_input0, force_manual_attention=force_manual_attention
                )
                for layer in wallx_model.model.layers
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        return_hidden_states: bool = False,
    ):
        hidden_states = inputs_embeds
        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        causal_mask = self.base_model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions=False,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        next_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = None
            if past_key_values is not None:
                if isinstance(past_key_values, (list, tuple)):
                    past = past_key_values[i]
                else:
                    past = past_key_values
            hidden_states, present = layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if use_cache:
                next_cache.append(present)

        hidden_states, _ = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_hidden_states and use_cache:
            return logits, hidden_states, next_cache
        if return_hidden_states:
            return logits, hidden_states
        if use_cache:
            return logits, next_cache
        return logits
