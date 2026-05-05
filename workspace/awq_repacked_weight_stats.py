#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from llm_loader import AutoModel
from llm_loader.models.linear import AWQLinear


def tensor_stats(t: torch.Tensor):
    a = t.float().reshape(-1).cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": float(a.mean().item()),
        "std": float(a.std(unbiased=False).item()),
        "min": float(a.min().item()),
        "max": float(a.max().item()),
        "abs_mean": float(a.abs().mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model = AutoModel.from_pretrained(args.model, device="cpu")
    layer = model.model.layers[0].self_attn
    out = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        mod = getattr(layer, name)
        assert isinstance(mod, AWQLinear), type(mod)
        entry = {}
        for attr in ("qweight", "qzeros", "scales", "bias"):
            value = getattr(mod, attr, None)
            if isinstance(value, torch.Tensor):
                entry[attr] = tensor_stats(value)
        out[name] = entry
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
