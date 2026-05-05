#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def patched_repack_awq_to_plugin(qweight: torch.Tensor, qzeros: torch.Tensor) -> torch.Tensor:
    import numpy as np

    in_features, out_div8 = qweight.shape
    out_features = out_div8 * 8
    group_size = in_features // qzeros.shape[0]

    qw = qweight.cpu().to(torch.int32)
    qz = qzeros.cpu().to(torch.int32)

    bit_to_ch = [0, 2, 4, 6, 1, 3, 5, 7]

    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[:, bit_to_ch[k]::8] = (qw >> (4 * k)) & 0xF

    zeros = torch.zeros(in_features // group_size, out_features, dtype=torch.int32)
    for k in range(8):
        zeros[:, bit_to_ch[k]::8] = (qz >> (4 * k)) & 0xF

    zeros_expanded = zeros.repeat_interleave(group_size, dim=0)
    nibbles = (nibbles - zeros_expanded + 8).clamp(0, 15)
    nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)

    packed_u16 = np.asarray(_pack_intweights(nibbles_nk), dtype=np.uint16)
    packed_u16 = np.ascontiguousarray(packed_u16)
    rows, cols = packed_u16.shape
    packed_int8 = np.frombuffer(packed_u16.tobytes(), dtype=np.int8).reshape(rows * 2, cols)
    return torch.tensor(packed_int8, dtype=torch.int8)


def _pack_intweights(unpacked_qweight: torch.Tensor):
    # unpacked_qweight: [N, K] values 0..15, N % 4 == 0
    import numpy as np
    arr = np.asarray(unpacked_qweight, dtype=np.uint16)
    N, K = arr.shape
    out = np.zeros((N // 4, K), dtype=np.uint16)
    out |= (arr[0::4, :] & 0xF) << 0
    out |= (arr[1::4, :] & 0xF) << 4
    out |= (arr[2::4, :] & 0xF) << 8
    out |= (arr[3::4, :] & 0xF) << 12
    return out


def stats(t: torch.Tensor):
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

    model_file = Path(args.model) / "model.safetensors"
    result = {}
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            qweight = f.get_tensor(f"model.layers.0.self_attn.{proj}.qweight")
            qzeros = f.get_tensor(f"model.layers.0.self_attn.{proj}.qzeros")
            scales = f.get_tensor(f"model.layers.0.self_attn.{proj}.scales")
            repacked = patched_repack_awq_to_plugin(qweight, qzeros)
            result[proj] = {
                "raw_qweight": stats(qweight),
                "raw_qzeros": stats(qzeros),
                "raw_scales": stats(scales),
                "repacked_qweight": stats(repacked),
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
