#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

BIT_TO_CH = [0, 2, 4, 6, 1, 3, 5, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    in_features, out_div8 = qweight.shape
    out_features = out_div8 * 8
    group_size = in_features // qzeros.shape[0]

    qw = qweight.to(torch.int32)
    qz = qzeros.to(torch.int32)

    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[:, BIT_TO_CH[k]::8] = (qw >> (4 * k)) & 0xF

    zeros = torch.zeros(in_features // group_size, out_features, dtype=torch.int32)
    for k in range(8):
        zeros[:, BIT_TO_CH[k]::8] = (qz >> (4 * k)) & 0xF

    zeros_expanded = zeros.repeat_interleave(group_size, dim=0)
    scales_expanded = scales.float().repeat_interleave(group_size, dim=0)
    dequant_in_out = (nibbles.float() - zeros_expanded.float()) * scales_expanded
    return dequant_in_out.t().contiguous()  # [out, in]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(aa, bb, dim=0).item())


def stats(t: torch.Tensor):
    x = t.reshape(-1).float().cpu().numpy()
    return {
        "shape": list(t.shape),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "abs_mean": float(np.abs(x).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--awq", required=True)
    args = ap.parse_args()

    fp16_dir = Path(args.fp16)
    awq_dir = Path(args.awq)
    fp16_index = json.load(open(fp16_dir / "model.safetensors.index.json"))["weight_map"]
    fp16_file = fp16_dir / fp16_index["model.layers.0.self_attn.q_proj.weight"]
    awq_file = awq_dir / "model.safetensors"

    result = {}
    projs = ("q_proj", "k_proj", "v_proj", "o_proj")
    with safe_open(str(fp16_file), framework="pt", device="cpu") as ff, safe_open(str(awq_file), framework="pt", device="cpu") as af:
        for proj in projs:
            w_fp = ff.get_tensor(f"model.layers.0.self_attn.{proj}.weight").float()
            qweight = af.get_tensor(f"model.layers.0.self_attn.{proj}.qweight")
            qzeros = af.get_tensor(f"model.layers.0.self_attn.{proj}.qzeros")
            scales = af.get_tensor(f"model.layers.0.self_attn.{proj}.scales")
            w_awq = unpack_awq(qweight, qzeros, scales)
            diff = (w_awq - w_fp).float()
            result[proj] = {
                "fp16": stats(w_fp),
                "awq_dequant": stats(w_awq),
                "cosine": cosine(w_fp, w_awq),
                "mae": float(diff.abs().mean().item()),
                "max_abs": float(diff.abs().max().item()),
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
