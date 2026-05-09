#!/usr/bin/env python3
"""
Validate INT8 quantization accuracy for wall-x model.

Compares bf16 linear outputs vs INT8 linear outputs per layer,
using random inputs at the actual sequence lengths used during inference.

Usage:
    python validate_int8.py --model-dir /path/to/bf16/model --int8-dir /path/to/int8/model
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open


def pad_rows_for_int_mm(x_2d: torch.Tensor):
    """Pad [M, K] rows to satisfy torch._int_mm CUDA shape constraints."""
    m = x_2d.shape[0]
    min_m = 24
    pad_m = 0
    if m < min_m:
        pad_m = min_m - m
    elif m % 8 != 0:
        pad_m = 8 - (m % 8)
    if pad_m > 0:
        x_2d = F.pad(x_2d, (0, 0, 0, pad_m))
    return x_2d, pad_m


def get_orig_shape(int8_weights, weight_key, weight_int8):
    shape_key = weight_key.replace(".weight", ".weight_orig_shape")
    if shape_key in int8_weights:
        shape = int8_weights[shape_key].to(torch.int64).cpu().tolist()
        return int(shape[0]), int(shape[1])
    return int(weight_int8.shape[0]), int(weight_int8.shape[1])


class INT8Linear:
    """Per-token dynamic activation + per-channel static weight INT8 linear."""

    def __init__(self, weight_int8, weight_scale, bias=None, orig_shape=None):
        self.weight_int8 = weight_int8        # [N, K] int8
        self.weight_int8_t = weight_int8.t().contiguous()  # [K, N] for _int_mm
        self.weight_scale = weight_scale      # [N] f32
        self.bias = bias
        if orig_shape is None:
            self.orig_out_features = int(weight_int8.shape[0])
            self.orig_in_features = int(weight_int8.shape[1])
        else:
            self.orig_out_features = int(orig_shape[0])
            self.orig_in_features = int(orig_shape[1])

    def forward(self, x):
        orig_shape = list(x.shape)
        K = x.shape[-1]
        flat = x.reshape(-1, K)  # [M, K]
        valid_m = flat.shape[0]
        padded_k = self.weight_int8_t.shape[0]

        if K != padded_k:
            if K > padded_k:
                raise ValueError(f"Input K={K} exceeds padded K={padded_k}")
            flat = F.pad(flat, (0, padded_k - K))

        # Per-token dynamic activation quantization
        act_f32 = flat.float()
        act_scale = (act_f32.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-10)
        act_int8 = (act_f32 / act_scale).round().clamp(-128, 127).to(torch.int8)
        act_int8, pad_m = pad_rows_for_int_mm(act_int8)
        if pad_m > 0:
            act_scale = F.pad(act_scale, (0, 0, 0, pad_m))

        # INT8 GEMM: [M, K] @ [K, N] -> [M, N] int32
        out_i32 = torch._int_mm(act_int8, self.weight_int8_t)

        # Dequantize
        out = out_i32.float() * act_scale * self.weight_scale.unsqueeze(0)

        if self.orig_out_features != self.weight_int8.shape[0]:
            out = out[:, :self.orig_out_features]

        if self.bias is not None:
            out = out + self.bias.float()

        if pad_m > 0:
            out = out[:valid_m]

        orig_shape[-1] = self.orig_out_features
        return out.to(x.dtype).reshape(orig_shape)


def validate(model_dir: str, int8_dir: str, device: str = "cuda"):
    model_dir = Path(model_dir)
    int8_dir = Path(int8_dir)

    # Load both weight sets
    bf16_weights = {}
    int8_weights = {}

    print("Loading bf16 weights...")
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device=device) as f:
            for key in f.keys():
                bf16_weights[key] = f.get_tensor(key)
    print(f"  Loaded {len(bf16_weights)} tensors")

    print("Loading int8 weights...")
    for sf in sorted(int8_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device=device) as f:
            for key in f.keys():
                int8_weights[key] = f.get_tensor(key)
    print(f"  Loaded {len(int8_weights)} tensors")

    # Find quantized layers (weight dtype is int8)
    quantized_keys = [
        k for k in bf16_weights
        if k.endswith(".weight") and k in int8_weights
        and int8_weights[k].dtype == torch.int8
    ]
    print(f"\nFound {len(quantized_keys)} quantized layers to validate")

    # Test configurations matching wall-x inference:
    #   Decode:   M=1   (single-token VQA decode, padded internally for _int_mm)
    #   Postfix:  M=32  (action tokens, ODE loop)
    #   Prefill:  M=488 (420 prefix + 68 vision)
    test_configs = [
        ("Decode-1", 1),       # VQA decode
        ("Postfix-32", 32),     # ODE loop
        ("Prefill-488", 488),   # Full prefill
    ]

    print(f"\n{'Key':<65} {'M':>4} {'MSE':>12} {'MaxErr':>10} {'CosSim':>10}")
    print("-" * 110)

    all_errors = []

    for key in quantized_keys:
        W_bf16 = bf16_weights[key]
        W_int8 = int8_weights[key]
        scale_key = key.replace(".weight", ".weight_scale")
        W_scale = int8_weights[scale_key]
        orig_shape = get_orig_shape(int8_weights, key, W_int8)

        bias_key = key.replace(".weight", ".bias")
        bias = bf16_weights.get(bias_key)

        N, K = W_bf16.shape

        for config_name, M in test_configs:
            x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

            # bf16 reference
            ref = F.linear(x, W_bf16, bias)

            # INT8 quantized
            int8_op = INT8Linear(W_int8, W_scale, bias, orig_shape=orig_shape)
            out = int8_op.forward(x)

            # Error metrics
            diff = (ref.float() - out.float())
            mse = (diff ** 2).mean().item()
            max_err = diff.abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                ref.float().reshape(1, -1), out.float().reshape(1, -1)
            ).item()

            all_errors.append({
                "key": key, "M": M,
                "mse": mse, "max_err": max_err, "cos_sim": cos_sim
            })

            # Only print first 40 entries to keep output manageable
            if len(all_errors) <= 40:
                print(f"  {key:<63} {M:>4} {mse:>12.4e} {max_err:>10.4f} {cos_sim:>10.6f}")

    if len(all_errors) > 40:
        print(f"  ... ({len(all_errors) - 40} more entries)")

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    for config_name, M in test_configs:
        entries = [e for e in all_errors if e["M"] == M]
        if not entries:
            continue
        avg_cos = sum(e["cos_sim"] for e in entries) / len(entries)
        min_cos = min(e["cos_sim"] for e in entries)
        avg_mse = sum(e["mse"] for e in entries) / len(entries)
        max_max_err = max(e["max_err"] for e in entries)

        print(f"\n  {config_name} (M={M}, {len(entries)} layers):")
        print(f"    Avg CosSim: {avg_cos:.6f}  (min: {min_cos:.6f})")
        print(f"    Avg MSE:    {avg_mse:.6e}")
        print(f"    Max Error:  {max_max_err:.4f}")

        if avg_cos > 0.999:
            print(f"    Verdict:    PASS - Excellent accuracy")
        elif avg_cos > 0.995:
            print(f"    Verdict:    OK - Good accuracy, monitor ODE accumulation")
        elif avg_cos > 0.99:
            print(f"    Verdict:    WARNING - Consider SmoothQuant")
        else:
            print(f"    Verdict:    FAIL - SmoothQuant strongly recommended")

    # Chain test: simulate 5-step ODE with multiple linear layers
    print(f"\n{'='*60}")
    print("ODE Chain Simulation (5 steps x 2 linear layers per step)")
    print(f"{'='*60}")

    # Pick q_proj and o_proj from layer 0 (both are [2048, 2048])
    chain_keys = [k for k in quantized_keys
                  if "layers.0." in k and ("q_proj" in k or "o_proj" in k)]
    if len(chain_keys) >= 2:
        K = bf16_weights[chain_keys[0]].shape[1]
        x_bf16 = torch.randn(32, K, dtype=torch.bfloat16, device=device)
        x_int8 = x_bf16.clone()

        for step in range(5):
            for key in chain_keys[:2]:
                W_bf16 = bf16_weights[key]
                W_int8 = int8_weights[key]
                scale_key = key.replace(".weight", ".weight_scale")
                W_scale = int8_weights[scale_key]
                orig_shape = get_orig_shape(int8_weights, key, W_int8)
                bias = bf16_weights.get(key.replace(".weight", ".bias"))

                x_bf16 = F.linear(x_bf16, W_bf16, bias)
                int8_op = INT8Linear(W_int8, W_scale, bias, orig_shape=orig_shape)
                x_int8 = int8_op.forward(x_int8)

        cos_sim = torch.nn.functional.cosine_similarity(
            x_bf16.float().reshape(1, -1), x_int8.float().reshape(1, -1)
        ).item()
        mse = ((x_bf16.float() - x_int8.float()) ** 2).mean().item()
        print(f"  After 5 ODE steps x 2 layers (10 INT8 GEMMs):")
        print(f"    CosSim: {cos_sim:.6f}")
        print(f"    MSE:    {mse:.6e}")
        if cos_sim > 0.99:
            print(f"    Verdict: PASS - ODE chain error is acceptable")
        else:
            print(f"    Verdict: WARNING - ODE chain error may be visible")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate INT8 quantization accuracy")
    parser.add_argument("--model-dir", required=True, help="Path to original bf16 model")
    parser.add_argument("--int8-dir", required=True, help="Path to INT8 quantized model")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()
    validate(args.model_dir, args.int8_dir, args.device)
