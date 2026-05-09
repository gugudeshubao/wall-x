#!/usr/bin/env python3
"""
INT8 Quantization VQA Comparison — bf16 vs INT8 on real test images.

Loads wall-x model, runs VQA on test images with bf16 weights,
then patches Linear layers with INT8 weights and re-runs same tests.
Compares answer quality and latency.

Usage (on Orin):
    python scripts/bench_int8_vqa.py \
        --model-path /data/wy/models/wall-oss-flow \
        --int8-path /data/wy/models/wall-oss-flow-int8 \
        --image-dir /data/wy/wall-x/test_images

Usage (on 5090):
    python scripts/bench_int8_vqa.py \
        --model-path /home/ubuntu/project/models/wall-oss-flow \
        --int8-path /home/ubuntu/project/models/wall-oss-flow-int8 \
        --image-dir /home/ubuntu/project/github/wall-x/test_images
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open


# ---------------------------------------------------------------------------
# INT8 Linear replacement
# ---------------------------------------------------------------------------

def get_orig_shape(int8_weights, weight_key, weight_int8):
    shape_key = weight_key.replace(".weight", ".weight_orig_shape")
    if shape_key in int8_weights:
        shape = int8_weights[shape_key].to(torch.int64).cpu().tolist()
        return int(shape[0]), int(shape[1])
    return int(weight_int8.shape[0]), int(weight_int8.shape[1])

class INT8Linear(nn.Module):
    """Drop-in replacement for nn.Linear using W8A8 INT8 GEMM."""

    def __init__(self, weight_int8, weight_scale, bias=None, orig_shape=None):
        super().__init__()
        self.register_buffer('weight_int8', weight_int8)          # [N, K] int8
        self.register_buffer('weight_int8_t',
                             weight_int8.t().contiguous())         # [K, N] for _int_mm
        self.register_buffer('weight_scale', weight_scale)        # [N] f32
        if orig_shape is None:
            self.orig_out_features = int(weight_int8.shape[0])
            self.orig_in_features = int(weight_int8.shape[1])
        else:
            self.orig_out_features = int(orig_shape[0])
            self.orig_in_features = int(orig_shape[1])
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        orig_dtype = x.dtype
        orig_shape = list(x.shape)
        K = x.shape[-1]
        flat = x.reshape(-1, K)                                   # [M, K]
        M = flat.shape[0]
        padded_k = self.weight_int8_t.shape[0]

        if K != padded_k:
            if K > padded_k:
                raise ValueError(f"Input K={K} exceeds padded K={padded_k}")
            flat = torch.nn.functional.pad(flat, (0, padded_k - K))

        # torch._int_mm requires M > 16 and M % 8 == 0
        # Minimum usable M is 24 (next multiple of 8 above 16)
        min_m = 24
        pad_m = 0
        if M < min_m:
            pad_m = min_m - M
        elif M % 8 != 0:
            pad_m = 8 - (M % 8)
        if pad_m > 0:
            flat = torch.nn.functional.pad(flat, (0, 0, 0, pad_m))

        # Per-token dynamic activation quantization
        act_f32 = flat.float()
        act_scale = (act_f32.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-10)
        act_int8 = (act_f32 / act_scale).round().clamp(-128, 127).to(torch.int8)

        # INT8 GEMM: [M', K] @ [K, N] -> [M', N] int32
        out_i32 = torch._int_mm(act_int8, self.weight_int8_t)

        # Dequantize: int32 -> f32 -> orig_dtype
        out = out_i32.float() * act_scale * self.weight_scale.unsqueeze(0)

        if self.orig_out_features != self.weight_int8.shape[0]:
            out = out[:, :self.orig_out_features]

        if self.bias is not None:
            out = out + self.bias.float()

        # Remove padding
        if pad_m > 0:
            out = out[:M]

        orig_shape[-1] = self.orig_out_features
        return out.to(orig_dtype).reshape(orig_shape)


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def load_int8_weights(int8_dir, device="cuda"):
    """Load all tensors from INT8 safetensors directory."""
    int8_dir = Path(int8_dir)
    weights = {}
    for sf in sorted(int8_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device=device) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    print(f"  Loaded {len(weights)} tensors from {int8_dir}")
    return weights


def patch_model_int8(model, int8_weights):
    """Replace bf16 Linear layers with INT8 versions where quantized."""
    patched = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        weight_key = f"{name}.weight"
        scale_key = f"{name}.weight_scale"

        if weight_key not in int8_weights or scale_key not in int8_weights:
            continue
        if int8_weights[weight_key].dtype != torch.int8:
            continue

        w_int8 = int8_weights[weight_key]
        w_scale = int8_weights[scale_key]
        orig_shape = get_orig_shape(int8_weights, weight_key, w_int8)
        bias = module.bias.data if module.bias is not None else None

        int8_mod = INT8Linear(w_int8, w_scale, bias, orig_shape=orig_shape)

        # Replace in parent
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], int8_mod)
        patched += 1

    return patched


# ---------------------------------------------------------------------------
# VQA runner
# ---------------------------------------------------------------------------

def run_vqa(model, processor, image_paths, questions,
            max_new_tokens=20, warmup=1, label=""):
    """Run VQA on all images × questions, return structured results."""
    results = []

    for img_idx, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)

        for q_idx, question in enumerate(questions):
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]}
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], images=[image],
                padding=True, return_tensors="pt",
            ).to("cuda")

            # Warmup on first combo only
            if img_idx == 0 and q_idx == 0:
                for _ in range(warmup):
                    with torch.no_grad():
                        model.generate(**inputs, max_new_tokens=max_new_tokens)
                    torch.cuda.synchronize()

            # Timed run
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            gen_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            answer = processor.decode(gen_ids, skip_special_tokens=True)
            n_tok = len(gen_ids)

            results.append({
                "image": img_name,
                "question": question,
                "answer": answer,
                "tokens": n_tok,
                "latency_ms": round(latency * 1000, 1),
                "tok_per_s": round(n_tok / latency, 2) if latency > 0 else 0,
            })

            print(f"  [{label}] {img_name} | Q{q_idx+1}: {question[:35]}...")
            print(f"         A: {answer[:70]}{'...' if len(answer)>70 else ''}")
            print(f"         {n_tok} tok, {latency*1000:.0f}ms, "
                  f"{n_tok/latency:.1f} tok/s")

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(bf16_results, int8_results):
    """Side-by-side comparison of bf16 vs INT8 answers and latency."""
    print(f"\n{'='*80}")
    print("COMPARISON: bf16 vs INT8")
    print(f"{'='*80}\n")

    exact_match = 0
    total_bf16_lat = 0
    total_int8_lat = 0

    for bf, i8 in zip(bf16_results, int8_results):
        same = bf['answer'].strip() == i8['answer'].strip()
        if same:
            exact_match += 1
        speedup = bf['latency_ms'] / i8['latency_ms'] if i8['latency_ms'] > 0 else 0

        print(f"[{bf['image']}] {bf['question'][:50]}")
        print(f"  bf16: {bf['answer'][:65]}{'...' if len(bf['answer'])>65 else ''}")
        print(f"  INT8: {i8['answer'][:65]}{'...' if len(i8['answer'])>65 else ''}")
        print(f"  Match: {'✓' if same else '✗'}  |  "
              f"{bf['latency_ms']:.0f}ms → {i8['latency_ms']:.0f}ms "
              f"({speedup:.2f}x)\n")

        total_bf16_lat += bf['latency_ms']
        total_int8_lat += i8['latency_ms']

    n = len(bf16_results)
    print(f"{'='*80}")
    print(f"SUMMARY ({n} test cases)")
    print(f"{'='*80}")
    print(f"  Exact answer match:  {exact_match}/{n} "
          f"({exact_match/n*100:.0f}%)")
    print(f"  Avg latency bf16:    {total_bf16_lat/n:.0f} ms")
    print(f"  Avg latency INT8:    {total_int8_lat/n:.0f} ms")
    if total_int8_lat > 0:
        print(f"  Overall speedup:     {total_bf16_lat/total_int8_lat:.2f}x")
    print(f"  Peak GPU memory:     "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="INT8 VQA Accuracy & Latency Comparison")
    parser.add_argument("--model-path", required=True,
                        help="Path to bf16 model directory")
    parser.add_argument("--int8-path", required=True,
                        help="Path to INT8 quantized model directory")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing test images")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max tokens to generate (default: 20)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs (default: 1)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Discover images
    image_dir = Path(args.image_dir)
    image_paths = sorted([
        str(p) for p in image_dir.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    ])
    if not image_paths:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} test images:")
    for p in image_paths:
        print(f"  {os.path.basename(p)}")

    questions = [
        "Describe what you see in this image.",
        "What objects are on the table?",
    ]

    # ---- Load model (bf16) ----
    print(f"\nLoading bf16 model from {args.model_path} ...")
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
        Qwen2_5_VLMoEForAction,
    )

    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()
    processor = model.processor
    print(f"Model loaded. Peak GPU: "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # ---- Phase 1: bf16 baseline ----
    print(f"\n{'='*80}")
    print(f"PHASE 1: bf16 baseline (max_new_tokens={args.max_new_tokens})")
    print(f"{'='*80}")
    bf16_results = run_vqa(
        model, processor, image_paths, questions,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup, label="bf16")

    # ---- Phase 2: Patch to INT8 ----
    print(f"\n{'='*80}")
    print(f"PHASE 2: Patching model with INT8 weights")
    print(f"{'='*80}")
    print(f"Loading INT8 weights from {args.int8_path} ...")
    int8_weights = load_int8_weights(args.int8_path, device="cuda")
    n_patched = patch_model_int8(model, int8_weights)
    print(f"Patched {n_patched} Linear layers → INT8")
    del int8_weights
    torch.cuda.empty_cache()

    # ---- Phase 3: INT8 inference ----
    print(f"\n{'='*80}")
    print(f"PHASE 3: INT8 inference (max_new_tokens={args.max_new_tokens})")
    print(f"{'='*80}")
    int8_results = run_vqa(
        model, processor, image_paths, questions,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup, label="INT8")

    # ---- Compare ----
    compare(bf16_results, int8_results)

    # ---- Save ----
    if args.output:
        data = {
            "config": {
                "model_path": args.model_path,
                "int8_path": args.int8_path,
                "max_new_tokens": args.max_new_tokens,
                "num_images": len(image_paths),
                "num_questions": len(questions),
                "gpu": torch.cuda.get_device_name(0),
            },
            "bf16": bf16_results,
            "int8": int8_results,
        }
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
