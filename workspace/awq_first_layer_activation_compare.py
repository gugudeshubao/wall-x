#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

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
    x = t.reshape(-1).float().cpu()
    return {
        "shape": list(t.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(x.abs().mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--awq", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", default="What is 2+2?")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    token_ids = tok.encode(args.text).ids

    fp16_dir = Path(args.fp16)
    awq_dir = Path(args.awq)
    fp16_index = json.load(open(fp16_dir / "model.safetensors.index.json"))["weight_map"]
    shard = fp16_dir / fp16_index["model.embed_tokens.weight"]

    out = {"token_ids": token_ids, "qkv": {}}
    with safe_open(str(shard), framework="pt", device="cpu") as ff, safe_open(str(awq_dir / "model.safetensors"), framework="pt", device="cpu") as af:
        embed = ff.get_tensor("model.embed_tokens.weight").float()[token_ids]  # [seq, hidden]
        for proj in ("q_proj", "k_proj", "v_proj"):
            w_fp = ff.get_tensor(f"model.layers.0.self_attn.{proj}.weight").float()  # [out,in]
            b_fp = ff.get_tensor(f"model.layers.0.self_attn.{proj}.bias").float() if f"model.layers.0.self_attn.{proj}.bias" in ff.keys() else None
            qweight = af.get_tensor(f"model.layers.0.self_attn.{proj}.qweight")
            qzeros = af.get_tensor(f"model.layers.0.self_attn.{proj}.qzeros")
            scales = af.get_tensor(f"model.layers.0.self_attn.{proj}.scales")
            b_awq = af.get_tensor(f"model.layers.0.self_attn.{proj}.bias").float() if f"model.layers.0.self_attn.{proj}.bias" in af.keys() else None
            w_awq = unpack_awq(qweight, qzeros, scales)
            y_fp = embed @ w_fp.t()
            if b_fp is not None:
                y_fp = y_fp + b_fp
            y_awq = embed @ w_awq.t()
            if b_awq is not None:
                y_awq = y_awq + b_awq
            out["qkv"][proj] = {
                "cosine": cosine(y_fp, y_awq),
                "mae": float((y_awq - y_fp).abs().mean().item()),
                "max_abs": float((y_awq - y_fp).abs().max().item()),
                "fp16": stats(y_fp),
                "awq": stats(y_awq),
            }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
