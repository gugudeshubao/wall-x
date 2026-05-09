#!/usr/bin/env python3
"""
Compare Python vs C++ logits step-by-step.

Runs Python inference manually (step-by-step greedy) and saves
logits/top-k at each step. Then reads C++ output tokens and
compares where divergence starts and how close the logits are.

Usage (on Orin):
    python scripts/compare_logits.py \
        --model-path /data/wy/models/wall-oss-flow \
        --input-dir /data/wy/wall-x/vqa_test_inputs \
        --image fruits_on_table \
        --max-new-tokens 20

Usage (INT8: compare C++ INT8 against Python INT8 generated via
bf16 model + patched INT8 weights):
    python scripts/compare_logits.py \
        --model-path /data/wy/models/wall-oss-flow-int8-pad-moe \
        --python-model-path /data/wy/models/wall-oss-flow \
        --python-int8-path /data/wy/models/wall-oss-flow-int8-pad-moe \
        --cpp-binary ./cpp_infer/build/wallx_infer \
        --cpp-model-path /data/wy/models/wall-oss-flow-int8-pad-moe \
        --input-dir /data/wy/wall-x/vqa_test_inputs \
        --image real_tabletop_2 \
        --max-new-tokens 20
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file


def run_cpp_engine(binary, model_path, input_dir, max_new_tokens, kernels_dir=None):
    """Run C++ wallx_infer on a single input directory, writing cpp_output_tokens.txt."""
    cmd = [
        binary,
        "--model", model_path,
        "--mode", "vqa",
        "--input", input_dir,
        "--max_new_tokens", str(max_new_tokens),
        "--warmup", "0",
    ]
    if kernels_dir:
        cmd.extend(["--kernels", kernels_dir])

    print(f"Running C++: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ engine failed (exit code {result.returncode}):\n{result.stderr[:1000]}"
        )
    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python vs C++ logits step-by-step")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--python-model-path", default=None,
                        help="Optional bf16 Python model path for generating baseline")
    parser.add_argument("--python-int8-path", default=None,
                        help="Optional INT8 weights path to patch the Python baseline model")
    parser.add_argument("--cpp-binary", default=None,
                        help="Optional C++ wallx_infer binary to regenerate cpp_output_tokens.txt")
    parser.add_argument("--cpp-model-path", default=None,
                        help="Optional C++ model path (defaults to --model-path)")
    parser.add_argument("--cpp-kernels", default=None,
                        help="Optional Triton kernel directory for C++ engine")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--image", required=True,
                        help="Image name (e.g. fruits_on_table)")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    img_dir = Path(args.input_dir) / args.image
    tensors_path = img_dir / "inputs.safetensors"
    cpp_tokens_path = img_dir / "cpp_output_tokens.txt"

    # Load model
    load_model_path = args.python_model_path or args.model_path
    print(f"\nLoading model from {load_model_path}...")
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    model = Qwen2_5_VLMoEForAction.from_pretrained(load_model_path)
    model.eval().to("cuda").bfloat16()
    processor = model.processor

    if args.python_int8_path:
        print(f"Patching Python model with INT8 weights from {args.python_int8_path} ...")
        try:
            from scripts.bench_int8_vqa import load_int8_weights, patch_model_int8
        except ModuleNotFoundError:
            from bench_int8_vqa import load_int8_weights, patch_model_int8
        int8_weights = load_int8_weights(args.python_int8_path, device="cuda")
        patched = patch_model_int8(model, int8_weights)
        print(f"Patched {patched} Linear layers")
        del int8_weights
        torch.cuda.empty_cache()

    # Run C++ engine first if requested, so cpp_output_tokens.txt exists.
    if args.cpp_binary:
        cpp_model_path = args.cpp_model_path or args.model_path
        print(f"\nRunning C++ engine from {args.cpp_binary} ...")
        run_cpp_engine(
            args.cpp_binary,
            cpp_model_path,
            str(img_dir),
            args.max_new_tokens,
            args.cpp_kernels,
        )

    # Load C++ output tokens
    if cpp_tokens_path.exists():
        with open(cpp_tokens_path) as f:
            cpp_tokens = [int(x) for x in f.read().strip().split()]
        print(f"C++ tokens ({len(cpp_tokens)}): {cpp_tokens}")
    else:
        print(f"WARNING: {cpp_tokens_path} not found")
        cpp_tokens = None

    # Load inputs from safetensors
    print(f"Loading inputs from {tensors_path}...")
    tensors = load_file(str(tensors_path))
    input_ids = tensors["input_ids"].to("cuda")
    pixel_values = tensors["pixel_values"].to("cuda")
    image_grid_thw = tensors["image_grid_thw"].to("cuda")

    print(f"input_ids: {input_ids.shape}, pixel_values: {pixel_values.shape}")

    # --- Step-by-step greedy generation ---
    # Using model's internal generate would be cleaner, but we want to
    # capture logits at each step. Use manual prefill + decode.

    attention_mask = torch.ones_like(input_ids)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    # Run full generation with output_scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # outputs.scores is a tuple of (num_generated_tokens,) tensors, each [batch, vocab]
    gen_ids = outputs.sequences[0][input_ids.shape[1]:]
    py_tokens = gen_ids.tolist()
    print(f"\nPython tokens ({len(py_tokens)}): {py_tokens}")

    # Compare step by step
    print(f"\n{'='*80}")
    print(f"{'Step':>4} | {'Py Token':>10} | {'C++ Token':>10} | {'Match':>5} "
          f"| {'C++ rank':>8} | {'C++ logit':>10} | {'Py top1 logit':>14}")
    print(f"{'='*80}")

    for step_idx, score in enumerate(outputs.scores):
        # score: [1, vocab_size] — logits for this step
        logits = score[0]  # [vocab_size]

        py_tok = py_tokens[step_idx] if step_idx < len(py_tokens) else None
        cpp_tok = cpp_tokens[step_idx] if cpp_tokens and step_idx < len(cpp_tokens) else None

        # Get top-5
        top_vals, top_ids = logits.topk(5)

        match = "YES" if (py_tok is not None and cpp_tok is not None and py_tok == cpp_tok) else "NO"

        # Find where C++ token ranks in Python logits
        if cpp_tok is not None:
            cpp_logit = logits[cpp_tok].item()
            # Compute rank
            rank = (logits > logits[cpp_tok]).sum().item() + 1
        else:
            cpp_logit = float('nan')
            rank = -1

        py_top1_logit = top_vals[0].item()

        print(f"{step_idx:>4} | {py_tok:>10} | {cpp_tok if cpp_tok is not None else 'N/A':>10} "
              f"| {match:>5} | {rank:>8} | {cpp_logit:>10.3f} | {py_top1_logit:>14.3f}")

        # Show top-5 for diverging steps
        if match == "NO":
            print(f"      Python top-5: {[(tid.item(), f'{tv.item():.3f}') for tid, tv in zip(top_ids, top_vals)]}")
            if cpp_tok is not None:
                py_tok_decoded = processor.decode([py_tok])
                cpp_tok_decoded = processor.decode([cpp_tok])
                print(f"      Py: '{py_tok_decoded}' (id={py_tok})  vs  C++: '{cpp_tok_decoded}' (id={cpp_tok})")

    # Summary
    if cpp_tokens:
        n_match = sum(1 for i in range(min(len(py_tokens), len(cpp_tokens)))
                      if py_tokens[i] == cpp_tokens[i])
        n_total = min(len(py_tokens), len(cpp_tokens))
        print(f"\nToken match: {n_match}/{n_total}")

        # Check first divergence
        for i in range(n_total):
            if py_tokens[i] != cpp_tokens[i]:
                logits = outputs.scores[i][0]
                cpp_logit_val = logits[cpp_tokens[i]].item()
                py_logit_val = logits[py_tokens[i]].item()
                diff = py_logit_val - cpp_logit_val
                print(f"\nFirst divergence at step {i}:")
                print(f"  Python chose {py_tokens[i]} with logit {py_logit_val:.4f}")
                print(f"  C++ chose {cpp_tokens[i]} with logit {cpp_logit_val:.4f}")
                print(f"  Logit difference: {diff:.4f}")
                print(f"  C++ token's rank in Python logits: "
                      f"{(logits > logits[cpp_tokens[i]]).sum().item() + 1}")
                break


if __name__ == "__main__":
    main()
