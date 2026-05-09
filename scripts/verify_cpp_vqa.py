#!/usr/bin/env python3
"""
Verify C++ inference engine accuracy against Python baseline.

Reads the exported tensor directories (from export_vqa_inputs.py),
runs the C++ wallx_infer engine on each, and compares output tokens
with the Python baseline.

Prerequisites:
    1. Run export_vqa_inputs.py to create test inputs + Python baselines
    2. Rebuild wallx_infer with --input support
    3. Run this script

Usage (on Orin):
    python scripts/verify_cpp_vqa.py \
        --binary ./cpp_infer/build/wallx_infer \
        --model-path /data/wy/models/wall-oss-flow \
        --input-dir /data/wy/wall-x/vqa_test_inputs \
        --max-new-tokens 20

Usage (INT8: compare C++ INT8 against Python INT8 baseline generated
from the same exported inputs):
    python scripts/verify_cpp_vqa.py \
        --binary ./cpp_infer/build/wallx_infer \
        --model-path /data/wy/models/wall-oss-flow-int8-pad-moe \
        --input-dir /data/wy/wall-x/vqa_test_inputs \
        --max-new-tokens 20 \
        --python-model-path /data/wy/models/wall-oss-flow \
        --python-int8-path /data/wy/models/wall-oss-flow-int8-pad-moe
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file


def decode_tokens(token_ids, model_path):
    """Decode token IDs to text using the model's tokenizer."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    return processor.decode(token_ids, skip_special_tokens=True)


def run_cpp_engine(binary, model_path, input_dir, max_new_tokens, kernels_dir=None):
    """Run C++ wallx_infer on a single input directory, return generated token IDs."""
    cmd = [
        binary,
        "--model", model_path,
        "--mode", "vqa",
        "--input", input_dir,
        "--max_new_tokens", str(max_new_tokens),
        "--warmup", "0",  # no warmup for accuracy test
    ]
    if kernels_dir:
        cmd.extend(["--kernels", kernels_dir])

    print(f"    Running: {' '.join(cmd[-6:])}")  # show key args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"    ERROR: C++ engine failed (exit code {result.returncode})")
        print(f"    stderr: {result.stderr[:500]}")
        return None, result.stdout

    return result.stdout, result.stderr


def load_python_baseline_model(model_path, int8_path=None, device="cuda"):
    """Load optional Python baseline model for on-the-fly token generation."""
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    print(f"\nLoading Python baseline model from {model_path} ...")
    model = Qwen2_5_VLMoEForAction.from_pretrained(model_path)
    model.eval().to(device).bfloat16()
    processor = model.processor

    if int8_path:
        print(f"Patching Python baseline with INT8 weights from {int8_path} ...")
        try:
            from scripts.bench_int8_vqa import load_int8_weights, patch_model_int8
        except ModuleNotFoundError:
            from bench_int8_vqa import load_int8_weights, patch_model_int8

        int8_weights = load_int8_weights(int8_path, device=device)
        patched = patch_model_int8(model, int8_weights)
        print(f"  Patched {patched} Linear layers")
        del int8_weights
        torch.cuda.empty_cache()

    return model, processor


def run_python_baseline(model, input_dir, max_new_tokens, device="cuda"):
    """Generate Python baseline tokens from exported inputs.safetensors."""
    tensors_path = Path(input_dir) / "inputs.safetensors"
    tensors = load_file(str(tensors_path))

    input_ids = tensors["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "pixel_values" in tensors:
        inputs["pixel_values"] = tensors["pixel_values"].to(device)
    if "image_grid_thw" in tensors:
        inputs["image_grid_thw"] = tensors["image_grid_thw"].to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return outputs[0][input_ids.shape[1]:].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Verify C++ VQA inference accuracy against Python baseline")
    parser.add_argument("--binary", required=True,
                        help="Path to wallx_infer binary")
    parser.add_argument("--model-path", required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--input-dir", required=True,
                        help="Root directory with exported test inputs")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max tokens for C++ generation (default: 20)")
    parser.add_argument("--kernels", default=None,
                        help="Path to Triton cubin kernels (optional)")
    parser.add_argument("--python-model-path", default=None,
                        help="Optional Python model path for dynamic baseline generation")
    parser.add_argument("--python-int8-path", default=None,
                        help="Optional INT8 weights path to patch the Python baseline model")
    args = parser.parse_args()

    input_root = Path(args.input_dir)

    # Load manifest
    manifest_path = input_root / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found in {input_root}")
        print("       Run export_vqa_inputs.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Found {len(manifest['images'])} test cases")
    print(f"Question: {manifest['question']}")
    print(f"Max new tokens: {args.max_new_tokens}")

    python_model = None
    baseline_label = "Python manifest baseline"
    if args.python_model_path:
        python_model, processor = load_python_baseline_model(
            args.python_model_path,
            int8_path=args.python_int8_path,
            device="cuda",
        )
        baseline_label = "Python generated baseline"
        if args.python_int8_path:
            baseline_label = "Python INT8 generated baseline"
    else:
        # Load tokenizer for decoding
        print(f"\nLoading tokenizer from {args.model_path} ...")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)

    results = []
    exact_match = 0

    for entry in manifest['images']:
        img_name = Path(entry['image']).stem
        img_dir = str(input_root / img_name)
        if python_model is not None:
            py_token_ids = run_python_baseline(
                python_model, img_dir, args.max_new_tokens, device="cuda")
            py_answer = processor.decode(py_token_ids, skip_special_tokens=True)
        else:
            py_token_ids = entry['token_ids']
            py_answer = entry['answer']

        print(f"\n{'='*70}")
        print(f"  Image: {entry['image']}  (seq_len={entry['input_seq_len']})")
        print(f"{'='*70}")
        print(f"  {baseline_label}: {py_answer[:65]}{'...' if len(py_answer)>65 else ''}")
        print(f"  Baseline tokens:   {py_token_ids}")

        # Run C++ engine
        stdout, stderr = run_cpp_engine(
            args.binary, args.model_path, img_dir,
            args.max_new_tokens, args.kernels)

        if stdout is None:
            results.append({"image": entry['image'], "match": False, "error": True})
            continue

        # Load C++ output tokens (plain text format: space-separated ints)
        cpp_tokens_path = Path(img_dir) / "cpp_output_tokens.txt"
        if not cpp_tokens_path.exists():
            print(f"    ERROR: {cpp_tokens_path} not found")
            results.append({"image": entry['image'], "match": False, "error": True})
            continue

        with open(cpp_tokens_path) as f:
            cpp_token_ids = [int(x) for x in f.read().strip().split()]
        cpp_answer = processor.decode(cpp_token_ids, skip_special_tokens=True)

        print(f"  C++ output:      {cpp_answer[:65]}{'...' if len(cpp_answer)>65 else ''}")
        print(f"  C++ tokens:      {cpp_token_ids}")

        # Compare
        tokens_match = py_token_ids == cpp_token_ids
        text_match = py_answer.strip() == cpp_answer.strip()

        if tokens_match:
            print(f"  Result: ✓ EXACT TOKEN MATCH")
            exact_match += 1
        elif text_match:
            print(f"  Result: ~ Text matches, tokens differ")
        else:
            # Show token-level diff
            min_len = min(len(py_token_ids), len(cpp_token_ids))
            first_diff = -1
            for i in range(min_len):
                if py_token_ids[i] != cpp_token_ids[i]:
                    first_diff = i
                    break
            if first_diff == -1:
                first_diff = min_len
            print(f"  Result: ✗ DIFFERENT (first diff at token {first_diff})")

        results.append({
            "image": entry['image'],
            "py_answer": py_answer,
            "cpp_answer": cpp_answer,
            "py_tokens": py_token_ids,
            "cpp_tokens": cpp_token_ids,
            "tokens_match": tokens_match,
            "text_match": text_match,
        })

    # Summary
    n = len(results)
    errors = sum(1 for r in results if r.get("error"))
    valid = n - errors
    token_matches = sum(1 for r in results if r.get("tokens_match"))
    text_matches = sum(1 for r in results if r.get("text_match") or r.get("tokens_match"))

    print(f"\n{'='*70}")
    print(f"SUMMARY ({n} images)")
    print(f"{'='*70}")
    print(f"  Exact token match:  {token_matches}/{valid}")
    print(f"  Text match:         {text_matches}/{valid}")
    print(f"  Errors/failures:    {errors}")
    if token_matches == valid:
        print(f"  Verdict: ✓ PASS — C++ engine output is identical to Python")
    elif text_matches == valid:
        print(f"  Verdict: ~ ACCEPTABLE — Same text, minor numerical differences")
    else:
        print(f"  Verdict: ✗ NEEDS INVESTIGATION — C++ output differs from Python")


if __name__ == "__main__":
    main()
