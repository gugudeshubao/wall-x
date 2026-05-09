#!/usr/bin/env python3
"""
Pad a real-image Flow Action reference to match the dummy reference's sequence
length so that existing engines (built for the dummy 456-prefix / 488-total)
can accept it.

What we do:
- Prefix KV cache: pad from real_prefix_len to target_prefix_len (zeros at end)
- postfix_attention_mask_additive_4d: pad total-seq dimension from
  real_total to target_total with -1e4 (masked / unreachable positions)
- prefix_length field: update to target_prefix_len

Everything else (noise, postfix embeddings, etc.) is left unchanged.

Usage:
    python pad_real_reference.py \
        --input  /data/wy/wall-x/workspace/trt_spike/tmp/flow_real_1/flow_real_reference.safetensors \
        --output /data/wy/wall-x/workspace/trt_spike/tmp/flow_real_1/flow_real_padded_reference.safetensors \
        --target-prefix-len 456 \
        --postfix-len 32
"""

import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-prefix-len", type=int, default=456)
    parser.add_argument("--postfix-len", type=int, default=32)
    args = parser.parse_args()

    refs = load_file(args.input, device="cpu")
    real_prefix = int(refs["prefix_length"][0].item())
    pad = args.target_prefix_len - real_prefix
    target_total = args.target_prefix_len + args.postfix_len

    if pad < 0:
        raise ValueError(
            f"real prefix ({real_prefix}) > target ({args.target_prefix_len}); shrinking not supported"
        )

    print(f"real prefix_len = {real_prefix}")
    print(f"target prefix_len = {args.target_prefix_len}  (pad = {pad} tokens)")
    print(f"target total = {target_total}")

    out = {}
    for key, tensor in refs.items():
        if key.startswith("prefix_past_key_") or key.startswith("prefix_past_value_"):
            # shape: [1, n_heads, seq, head_dim] — pad seq dimension (dim=2)
            padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad))
            out[key] = padded
        elif key == "postfix_attention_mask_additive_4d":
            # shape: [1, 1, postfix, total_seq] — pad total_seq (last dim) with -1e4
            padded = torch.nn.functional.pad(tensor, (0, pad), value=-1e4)
            out[key] = padded
        elif key == "prefix_length":
            out[key] = torch.tensor([args.target_prefix_len], dtype=tensor.dtype)
        else:
            out[key] = tensor

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.output)
    print(f"Saved padded reference to {args.output}")

    # Quick sanity check
    kv = out["prefix_past_key_0"]
    print(f"prefix_past_key_0 shape after pad: {list(kv.shape)}")
    mask = out["postfix_attention_mask_additive_4d"]
    print(f"attention_mask shape after pad: {list(mask.shape)}")


if __name__ == "__main__":
    main()
