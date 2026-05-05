#!/usr/bin/env python3
"""
Export detailed Phase A layer-0 reference tensors.

This sits between the full decoder reference and the future TRT single-layer block:

  - hidden_states in
  - causal mask
  - rotary cos/sin
  - precomputed mrope-ready cos/sin
  - layer0 output

The goal is to lock down the exact tensor interface for a TRT-LLM single-layer
manual-attention block.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from phase_a_decoder_wrapper import VQASpecializedDecoder


def prepare_mrope_ready(cos: torch.Tensor, sin: torch.Tensor, mrope_section):
    mrope_section = mrope_section * 2
    cos_m = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    sin_m = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(1)
    return cos_m, sin_m


def build_explicit_causal_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    # Use a finite large negative value instead of dtype.min to avoid NaNs
    # when reproducing manual attention outside SDPA.
    neg_large = -1e4
    mask = torch.full((batch_size, 1, seq_len, seq_len), neg_large, dtype=dtype, device=device)
    tri = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    mask = torch.where(tri.unsqueeze(0).unsqueeze(0), torch.zeros_like(mask), mask)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Export Phase A layer0 reference tensors")
    parser.add_argument("--model-path", required=True, help="Path to wall-x model")
    parser.add_argument("--ref-dir", required=True, help="Directory with decoder_reference.safetensors")
    parser.add_argument("--output", required=True, help="Output safetensors path")
    args = parser.parse_args()

    ref_path = Path(args.ref_dir) / "decoder_reference.safetensors"
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    print(f"[LOAD] model from {args.model_path}")
    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()

    wrapper = VQASpecializedDecoder(model).eval().to("cuda").bfloat16()
    layer0 = wrapper.layers[0]
    attn = layer0.self_attn

    tensors = load_file(str(ref_path), device="cuda")
    inputs_embeds = tensors["inputs_embeds"]
    position_ids = tensors["position_ids"]
    attention_mask = tensors["attention_mask"]
    if attention_mask.numel() == 0:
        attention_mask = torch.ones(
            inputs_embeds.shape[0], inputs_embeds.shape[1],
            dtype=torch.long, device=inputs_embeds.device
        )

    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    causal_mask = wrapper.base_model._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, None, output_attentions=False
    )
    if causal_mask is None:
        causal_mask = build_explicit_causal_mask(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            inputs_embeds.dtype,
            inputs_embeds.device,
        )
    cos, sin = wrapper.rotary_emb(inputs_embeds, position_ids)
    cos_m, sin_m = prepare_mrope_ready(
        cos, sin, model.config.rope_scaling["mrope_section"]
    )

    with torch.no_grad():
        # Reproduce the exact math intended for the TRT single-layer block:
        # norm -> qkv -> manual mrope -> manual attention -> o_proj -> residual
        # -> norm -> expert0-only dense MLP -> residual
        h1, _ = layer0.input_layernorm(inputs_embeds)

        q = attn.q_proj(h1).view(1, inputs_embeds.shape[1], attn.num_heads, attn.head_dim).transpose(1, 2).float()
        k = attn.k_proj(h1).view(1, inputs_embeds.shape[1], attn.num_key_value_heads, attn.head_dim).transpose(1, 2).float()
        v = attn.v_proj(h1).view(1, inputs_embeds.shape[1], attn.num_key_value_heads, attn.head_dim).transpose(1, 2).float()
        cos_m_f = cos_m.float()
        sin_m_f = sin_m.float()
        causal_mask_f = causal_mask.float()

        q_rot = (q * cos_m_f) + (torch.cat([-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]], dim=-1) * sin_m_f)
        k_rot = (k * cos_m_f) + (torch.cat([-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]], dim=-1) * sin_m_f)

        k_rep = k_rot.repeat_interleave(attn.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(attn.num_key_value_groups, dim=1)

        scores = torch.matmul(q_rot, k_rep.transpose(-1, -2)) / (attn.head_dim ** 0.5)
        scores = scores + causal_mask_f
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_rep)
        attn_out = attn_out.transpose(1, 2).contiguous().view(1, inputs_embeds.shape[1], attn.num_heads * attn.head_dim)
        attn_out = attn.o_proj(attn_out.to(h1.dtype))
        x2 = inputs_embeds + attn_out

        h2, _ = layer0.post_attention_layernorm(x2)
        gate = layer0.expert0.gate_proj(h2[..., : layer0.dim_input0])
        up = layer0.expert0.up_proj(h2[..., : layer0.dim_input0])
        hidden = F.silu(gate) * up
        mlp = layer0.expert0.down_proj(hidden)
        mlp_out = torch.zeros_like(x2)
        mlp_out[..., : layer0.dim_input0] = mlp[..., : layer0.dim_input0]
        layer0_out = x2 + mlp_out

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "hidden_in": inputs_embeds.cpu(),
            "position_ids": position_ids.cpu(),
            "attention_mask_2d": attention_mask.cpu(),
            "causal_mask_4d": causal_mask.cpu()
            if causal_mask is not None
            else torch.empty(0, dtype=torch.float32),
            "cos": cos.cpu(),
            "sin": sin.cpu(),
            "cos_mrope": cos_m.cpu(),
            "sin_mrope": sin_m.cpu(),
            "layer0_out": layer0_out.cpu(),
        },
        str(out_path),
    )
    print(f"[SAVE] {out_path}")
    print(f"  hidden_in:     {list(inputs_embeds.shape)}")
    print(
        f"  causal_mask:   {list(causal_mask.shape) if causal_mask is not None else 'None'}"
    )
    print(f"  cos:           {list(cos.shape)}")
    print(f"  cos_mrope:     {list(cos_m.shape)}")
    print(f"  layer0_out:    {list(layer0_out.shape)}")


if __name__ == "__main__":
    main()
