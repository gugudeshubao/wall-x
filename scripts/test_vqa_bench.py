#!/usr/bin/env python3
"""VQA Batch Benchmark — wall-x cross-platform comparison.

Usage:
    python test_vqa_bench.py --model_path /path/to/model --image_dir /path/to/images
"""
import os, sys, time, json, argparse
import torch
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    # ---- load model ----
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval().to("cuda").bfloat16()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s, peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    processor = model.processor

    # ---- discover images ----
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted([
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not images:
        print("No images found!"); sys.exit(1)
    print(f"\nFound {len(images)} images\n")

    # ---- define questions per image ----
    questions = [
        "Describe what you see in this image.",
        "What objects are on the table?",
        "What action should the robot take?",
    ]

    # ---- run VQA ----
    results = []
    for img_name in images:
        img_path = os.path.join(args.image_dir, img_name)
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size

        for q_idx, question in enumerate(questions):
            # Build chat messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[pil_img],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # Warmup
            for _ in range(args.warmup):
                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=128)
                torch.cuda.synchronize()

            # Timed runs
            times_ms = []
            output_text = ""
            for r in range(args.repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=128)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000)

                if r == 0:
                    # Decode only the generated part
                    input_len = inputs["input_ids"].shape[1]
                    output_ids = generated[0][input_len:]
                    output_text = processor.decode(output_ids, skip_special_tokens=True)

            avg_ms = np.mean(times_ms)
            tok_count = generated.shape[1] - inputs["input_ids"].shape[1]
            tok_per_s = tok_count / (avg_ms / 1000) if avg_ms > 0 else 0

            result = {
                "image": img_name,
                "image_size": f"{w}x{h}",
                "question": question,
                "answer": output_text[:200],
                "output_tokens": int(tok_count),
                "avg_ms": round(avg_ms, 1),
                "min_ms": round(min(times_ms), 1),
                "max_ms": round(max(times_ms), 1),
                "tokens_per_sec": round(tok_per_s, 1),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            }
            results.append(result)
            print(f"[{len(results):2d}] {img_name:30s} | Q{q_idx+1} | {avg_ms:7.1f} ms | {tok_count:3d} tok | {tok_per_s:6.1f} tok/s | {output_text[:60]}")

    # ---- summary ----
    print("\n" + "=" * 90)
    all_avg = [r["avg_ms"] for r in results]
    all_tps = [r["tokens_per_sec"] for r in results]
    print(f"Total test cases:  {len(results)}")
    print(f"Avg latency:       {np.mean(all_avg):.1f} ms")
    print(f"Median latency:    {np.median(all_avg):.1f} ms")
    print(f"Min latency:       {np.min(all_avg):.1f} ms")
    print(f"Max latency:       {np.max(all_avg):.1f} ms")
    print(f"Avg tokens/sec:    {np.mean(all_tps):.1f}")
    print(f"Peak GPU memory:   {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # ---- save JSON ----
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    out_file = f"vqa_bench_{gpu_name}.json"
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_path": args.model_path,
        "num_images": len(images),
        "num_tests": len(results),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "avg_latency_ms": round(np.mean(all_avg), 1),
        "median_latency_ms": round(np.median(all_avg), 1),
        "avg_tokens_per_sec": round(np.mean(all_tps), 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
