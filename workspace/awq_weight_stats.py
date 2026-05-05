#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from llm_loader import AutoModel


def move_known_tensors_to_device(model, device):
    for module in model.modules():
        for attr in ("bias", "qweight", "qzeros", "scales", "weight", "weight_scale", "pre_quant_scale"):
            value = getattr(module, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(module, attr, value.to(device))


def tensor_stats(t: torch.Tensor):
    x = t.detach().float().cpu().reshape(-1)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(x.abs().mean().item()),
    }


def awq_module_stats(model):
    layer = model.model.layers[0].self_attn
    out = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        mod = getattr(layer, name)
        entry = {}
        for attr in ("qweight", "qzeros", "scales", "bias"):
            value = getattr(mod, attr, None)
            if isinstance(value, torch.Tensor):
                entry[attr] = tensor_stats(value)
        out[name] = entry
    return out


def fp16_module_stats(model):
    layer = model.model.layers[0].self_attn
    out = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        mod = getattr(layer, name)
        entry = {}
        for attr in ("weight", "bias"):
            value = getattr(mod, attr, None)
            if isinstance(value, torch.Tensor):
                entry[attr] = tensor_stats(value)
        out[name] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--awq", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    fp16 = AutoModel.from_pretrained(args.fp16)
    awq = AutoModel.from_pretrained(args.awq)
    fp16.to(device)
    awq.to(device)
    move_known_tensors_to_device(fp16, device)
    move_known_tensors_to_device(awq, device)

    print(json.dumps({"fp16": fp16_module_stats(fp16), "awq": awq_module_stats(awq)}, indent=2))


if __name__ == "__main__":
    main()
