#!/usr/bin/env python3
"""
Compare original wall-x decode trace logits with TRT runner step logits.
"""

import argparse
import json

import torch
from safetensors.torch import load_file


def top2_info(x: torch.Tensor):
    vals, idx = torch.topk(x, k=2, dim=-1)
    vals = vals.squeeze(0)
    idx = idx.squeeze(0)
    margin = (vals[0] - vals[1]).item()
    return int(idx[0].item()), float(vals[0].item()), int(idx[1].item()), float(vals[1].item()), margin


def main():
    parser = argparse.ArgumentParser(description="Compare original decode trace vs TRT per-step logits")
    parser.add_argument("--trace", required=True, help="decode_trace.safetensors")
    parser.add_argument("--trt-json", required=True, help="trt_step_logits.json")
    args = parser.parse_args()

    trace = load_file(args.trace, device="cpu")
    with open(args.trt_json) as f:
        trt = json.load(f)

    num_steps = min(len(trt["step_logits"]), len([k for k in trace.keys() if k.startswith("step_logits_")]))
    for step in range(num_steps):
        ref = trace[f"step_logits_{step}"].float()
        trt_logits = torch.tensor(trt["step_logits"][step], dtype=torch.float32)
        diff = (trt_logits - ref).abs()
        cos = torch.nn.functional.cosine_similarity(trt_logits.flatten(), ref.flatten(), dim=0).item()
        r1, rv1, r2, rv2, rmargin = top2_info(ref)
        t1, tv1, t2, tv2, tmargin = top2_info(trt_logits)

        print(f"\n[STEP {step}]")
        print(f"  cosine:   {cos:.8f}")
        print(f"  mean abs: {diff.mean().item():.8e}")
        print(f"  max abs:  {diff.max().item():.8e}")
        print(f"  REF top1/top2: {r1} ({rv1:.4f}) / {r2} ({rv2:.4f}), margin={rmargin:.6f}")
        print(f"  TRT top1/top2: {t1} ({tv1:.4f}) / {t2} ({tv2:.4f}), margin={tmargin:.6f}")


if __name__ == "__main__":
    main()
