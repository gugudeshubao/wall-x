#!/usr/bin/env python3
"""Benchmark Flash Attention 2 vs SDPA on wall-x VQA inference (Orin).

Usage:
    # Run SDPA benchmark
    python bench_fa2_vs_sdpa.py --model_path /data/wy/models/wall-oss-flow --attn sdpa

    # Run FA2 benchmark
    python bench_fa2_vs_sdpa.py --model_path /data/wy/models/wall-oss-flow --attn flash_attention_2

    # Run both back-to-back (loads model twice)
    python bench_fa2_vs_sdpa.py --model_path /data/wy/models/wall-oss-flow --attn both
"""
import os, sys, time, argparse, gc, glob
import torch
import numpy as np


def load_model(model_path, attn_impl, device="cuda"):
    """Load wall-x VQA model with specified attention implementation."""
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
    from safetensors.torch import load_file

    print(f"\n{'='*60}")
    print(f"Loading model with attn_implementation = '{attn_impl}'")
    print(f"{'='*60}")

    t0 = time.time()

    # Load config
    config_path = os.path.join(model_path, "config.json")
    model_config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(config_path)

    # Set attention implementation BEFORE model construction
    model_config._attn_implementation = attn_impl

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    # Construct model with chosen attention
    model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
    model.resize_token_embeddings(len(processor.tokenizer))

    # Load weights
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        sd = load_file(f, device="cpu")
        state_dict.update(sd)
    model.load_state_dict(state_dict, strict=False)

    model.eval().to(device, dtype=torch.bfloat16)
    load_time = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Model loaded in {load_time:.1f}s, peak GPU: {peak_mem:.2f} GB")
    print(f"Attention: {model_config._attn_implementation}")

    return model, processor


def prepare_input(processor, image_path, question, device="cuda"):
    """Prepare VQA input."""
    from PIL import Image

    pil_img = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path} ({pil_img.size[0]}x{pil_img.size[1]})")

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
    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to(device)
    return inputs


def benchmark_inference(model, inputs, max_new_tokens, warmup_runs, bench_runs):
    """Run inference benchmark and return latency stats."""
    input_len = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_len}, max_new_tokens: {max_new_tokens}")

    # Warmup
    print(f"Warmup ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
    print("Warmup done.")

    # Benchmark
    latencies = []
    output_tokens_list = []
    print(f"Benchmarking ({bench_runs} runs)...")
    for i in range(bench_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        out_tokens = generated.shape[1] - input_len
        latencies.append(elapsed_ms)
        output_tokens_list.append(out_tokens)
        print(f"  Run {i+1}/{bench_runs}: {elapsed_ms:.1f} ms, {out_tokens} tokens")

    latencies = np.array(latencies)
    avg_tokens = np.mean(output_tokens_list)
    stats = {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "avg_tokens": avg_tokens,
        "avg_tok_per_s": avg_tokens / (np.mean(latencies) / 1000),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    return stats


def print_stats(name, stats):
    """Pretty-print benchmark stats."""
    print(f"\n{'='*60}")
    print(f"  {name} Results")
    print(f"{'='*60}")
    print(f"  Mean latency:    {stats['mean_ms']:.1f} ms  (std: {stats['std_ms']:.1f})")
    print(f"  Median latency:  {stats['median_ms']:.1f} ms")
    print(f"  Min / Max:       {stats['min_ms']:.1f} / {stats['max_ms']:.1f} ms")
    print(f"  P95 latency:     {stats['p95_ms']:.1f} ms")
    print(f"  Avg tokens:      {stats['avg_tokens']:.1f}")
    print(f"  Throughput:      {stats['avg_tok_per_s']:.1f} tok/s")
    print(f"  Peak GPU:        {stats['peak_gpu_gb']:.2f} GB")
    print(f"{'='*60}")


def unload_model(model):
    """Unload model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(2)  # let GPU memory settle


def main():
    parser = argparse.ArgumentParser(description="Benchmark FA2 vs SDPA on wall-x VQA")
    parser.add_argument("--model_path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--image", default="/data/wy/wall-x/test_images/fruits_on_table.png")
    parser.add_argument("--question", default="Describe what you see in this image.")
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free GPU: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f} GB")

    # Check FA2 availability
    try:
        import flash_attn
        print(f"Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("Flash Attention: NOT AVAILABLE")
        if args.attn in ("flash_attention_2", "both"):
            print("ERROR: flash_attn not installed, cannot benchmark FA2")
            sys.exit(1)

    attn_modes = []
    if args.attn == "both":
        attn_modes = ["sdpa", "flash_attention_2"]
    else:
        attn_modes = [args.attn]

    all_stats = {}
    for attn_impl in attn_modes:
        torch.cuda.reset_peak_memory_stats()
        model, processor = load_model(args.model_path, attn_impl)
        inputs = prepare_input(processor, args.image, args.question)

        stats = benchmark_inference(model, inputs, args.max_new_tokens, args.warmup, args.runs)
        all_stats[attn_impl] = stats
        print_stats(attn_impl.upper(), stats)

        unload_model(model)
        del processor, inputs
        gc.collect()
        torch.cuda.empty_cache()

    # Comparison
    if len(all_stats) == 2:
        sdpa = all_stats["sdpa"]
        fa2 = all_stats["flash_attention_2"]
        speedup = sdpa["mean_ms"] / fa2["mean_ms"]
        mem_saving = sdpa["peak_gpu_gb"] - fa2["peak_gpu_gb"]

        print(f"\n{'#'*60}")
        print(f"  FA2 vs SDPA Comparison (Orin)")
        print(f"{'#'*60}")
        print(f"  SDPA mean:  {sdpa['mean_ms']:.1f} ms")
        print(f"  FA2  mean:  {fa2['mean_ms']:.1f} ms")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Mem saving: {mem_saving:+.2f} GB ({sdpa['peak_gpu_gb']:.2f} -> {fa2['peak_gpu_gb']:.2f})")
        print(f"  SDPA tok/s: {sdpa['avg_tok_per_s']:.1f}")
        print(f"  FA2  tok/s: {fa2['avg_tok_per_s']:.1f}")
        print(f"{'#'*60}")


if __name__ == "__main__":
    main()
