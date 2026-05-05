#!/usr/bin/env python3
"""
Export Phase A layer-0 weights in TRT-LLM-friendly layout.

Focuses on the first decoder layer of the VQA-specialized expert0-only path:

  - input_layernorm.weight
  - qkv fused weight / bias
  - o_proj weight
  - post_attention_layernorm.weight
  - expert0 gate/up/down weights
  - final lm_head / final norm optionally for later use
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Export Phase A layer-0 weights")
    parser.add_argument("--model-path", required=True, help="Path to wall-x model")
    parser.add_argument("--output", required=True, help="Output safetensors path")
    args = parser.parse_args()

    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    print(f"[LOAD] model from {args.model_path}")
    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cpu")

    layer0 = model.model.layers[0]
    attn = layer0.self_attn

    q_w = attn.q_proj.weight.detach().cpu().contiguous()
    k_w = attn.k_proj.weight.detach().cpu().contiguous()
    v_w = attn.v_proj.weight.detach().cpu().contiguous()
    q_b = attn.q_proj.bias.detach().cpu().contiguous()
    k_b = attn.k_proj.bias.detach().cpu().contiguous()
    v_b = attn.v_proj.bias.detach().cpu().contiguous()

    qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
    qkv_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()

    expert0 = layer0.moe.experts[0]

    tensors = {
        "layer0.input_layernorm.weight": layer0.input_layernorm.weight.detach().cpu().contiguous(),
        "layer0.qkv.weight": qkv_w,
        "layer0.qkv.bias": qkv_b,
        "layer0.o_proj.weight": attn.o_proj.weight.detach().cpu().contiguous(),
        "layer0.post_attention_layernorm.weight": layer0.post_attention_layernorm.weight.detach().cpu().contiguous(),
        "layer0.expert0.gate_proj.weight": expert0.gate_proj.weight.detach().cpu().contiguous(),
        "layer0.expert0.up_proj.weight": expert0.up_proj.weight.detach().cpu().contiguous(),
        "layer0.expert0.down_proj.weight": expert0.down_proj.weight.detach().cpu().contiguous(),
        "final.norm.weight": model.model.norm.weight.detach().cpu().contiguous(),
        "lm_head.weight": model.lm_head.weight.detach().cpu().contiguous(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"[SAVE] {out_path}")

    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
