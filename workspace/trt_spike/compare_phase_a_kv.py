#!/usr/bin/env python3
"""
Compare TRT prefill/decode KV tensors against wall-x reference KV tensors.
"""

import argparse

import torch
from safetensors.torch import load_file

import build_phase_a_prefill_trtllm as prefill_mod
import build_phase_a_decode_trtllm as decode_mod


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float().cpu(), b.flatten().float().cpu(), dim=0
    ).item()


def stats(name, a: torch.Tensor, b: torch.Tensor):
    diff = (a.cpu().float() - b.cpu().float()).abs()
    print(f"[{name}] cosine:   {cosine(a, b):.8f}")
    print(f"[{name}] mean abs: {diff.mean().item():.8e}")
    print(f"[{name}] max abs:  {diff.max().item():.8e}")


def main():
    parser = argparse.ArgumentParser(description="Compare Phase A TRT KV tensors")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer0-ref", required=True)
    parser.add_argument("--decoder-ref", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    layer0_ref = load_file(args.layer0_ref, device="cpu")
    decoder_ref = load_file(args.decoder_ref, device="cpu")

    ckpt1 = prefill_mod.CheckpointReader(args.checkpoint)
    prefill_engine = prefill_mod.build_engine(
        ckpt1, layer0_ref, num_layers=args.num_layers,
        precision=args.precision, include_head=True, output_kv=True
    )
    prefill_out, outputs, ok, _ = prefill_mod.run_engine(
        prefill_engine, layer0_ref, precision=args.precision,
        include_head=True, output_kv=True, num_layers=args.num_layers,
        warmup=1, iters=1
    )
    if not ok:
        raise RuntimeError("prefill run failed")

    # compare prefill kv
    for idx in [0, args.num_layers - 1]:
        stats(
            f"prefill_key_{idx}",
            outputs[f"present_past_key_{idx}"],
            decoder_ref[f"prefill_past_key_{idx}"].to(outputs[f"present_past_key_{idx}"].dtype),
        )
        stats(
            f"prefill_value_{idx}",
            outputs[f"present_past_value_{idx}"],
            decoder_ref[f"prefill_past_value_{idx}"].to(outputs[f"present_past_value_{idx}"].dtype),
        )

    # build decode engine
    ckpt2 = decode_mod.CheckpointReader(args.checkpoint)
    decode_engine = decode_mod.build_engine(
        ckpt2, decoder_ref, num_layers=args.num_layers, precision=args.precision
    )

    # run one decode step using reference prefill past
    dec_outputs, ok, _ = decode_mod.run_engine(
        decode_engine, decoder_ref, args.num_layers,
        precision=args.precision, warmup=1, iters=1
    )
    if not ok:
        raise RuntimeError("decode run failed")

    for idx in [0, args.num_layers - 1]:
        stats(
            f"decode_present_key_{idx}",
            dec_outputs[f"present_past_key_{idx}"],
            decoder_ref[f"present_past_key_{idx}"].to(dec_outputs[f"present_past_key_{idx}"].dtype),
        )
        stats(
            f"decode_present_value_{idx}",
            dec_outputs[f"present_past_value_{idx}"],
            decoder_ref[f"present_past_value_{idx}"].to(dec_outputs[f"present_past_value_{idx}"].dtype),
        )


if __name__ == "__main__":
    main()
