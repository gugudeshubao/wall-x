#!/usr/bin/env python3
"""Benchmark bitsandbytes INT8 quantization on Orin.

Usage:
    python bench_bnb_int8.py --model_path /data/wy/models/wall-oss-flow --mode baseline
    python bench_bnb_int8.py --model_path /data/wy/models/wall-oss-flow --mode int8
    python bench_bnb_int8.py --model_path /data/wy/models/wall-oss-flow --mode all
"""
import os, sys, time, argparse, gc, glob
import torch
import numpy as np


def load_model_baseline(model_path, device="cuda"):
    """Load wall-x VQA model in bf16 (baseline)."""
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
    from safetensors.torch import load_file

    config_path = os.path.join(model_path, "config.json")
    model_config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(config_path)
    model_config._attn_implementation = "flash_attention_2"

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
    model.resize_token_embeddings(len(processor.tokenizer))

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        sd = load_file(f, device="cpu")
        state_dict.update(sd)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, processor


def load_model_int8_bnb(model_path, device="cuda"):
    """Load wall-x VQA model with bitsandbytes INT8 quantization."""
    import bitsandbytes as bnb
    from transformers import AutoProcessor
    from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
    from safetensors.torch import load_file

    print(f"\n[BNB-INT8] bitsandbytes version: {bnb.__version__}")

    config_path = os.path.join(model_path, "config.json")
    model_config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(config_path)
    model_config._attn_implementation = "flash_attention_2"

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
    model.resize_token_embeddings(len(processor.tokenizer))

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        sd = load_file(f, device="cpu")
        state_dict.update(sd)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    # First move to bf16 on CPU
    model = model.to(dtype=torch.bfloat16)

    # Replace nn.Linear with bnb.nn.Linear8bitLt
    t0 = time.time()
    replaced = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            # Skip very small layers (embedding, heads)
            if module.in_features < 256 or module.out_features < 256:
                skipped += 1
                continue

            # Create bnb INT8 linear
            has_bias = module.bias is not None
            int8_linear = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=6.0,  # outlier threshold
            )
            # Copy weights
            int8_linear.weight = bnb.nn.Int8Params(
                module.weight.data.to(torch.float16),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if has_bias:
                int8_linear.bias = module.bias

            # Replace in parent module
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], int8_linear)
            replaced += 1

    # Move to CUDA (this triggers INT8 quantization)
    model = model.to(device)
    model.eval()

    elapsed = time.time() - t0
    print(f"[BNB-INT8] Replaced {replaced} Linear layers, skipped {skipped} (took {elapsed:.1f}s)")
    return model, processor


def prepare_input(processor, image_path, question, device="cuda"):
    """Prepare VQA input."""
    from PIL import Image
    pil_img = Image.open(image_path).convert("RGB").resize((448, 448))
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


def benchmark_inference(model, inputs, max_new_tokens, warmup_runs, bench_runs, label=""):
    """Run inference benchmark."""
    input_len = inputs["input_ids"].shape[1]
    print(f"\n--- Benchmark: {label} ---")
    print(f"Input tokens: {input_len}, max_new_tokens: {max_new_tokens}")

    # Warmup
    print(f"Warmup ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        try:
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{warmup_runs} OK")
        except Exception as e:
            print(f"  Warmup {i+1} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None

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
        "label": label,
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "avg_tokens": avg_tokens,
        "avg_tok_per_s": avg_tokens / (np.mean(latencies) / 1000),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark BNB INT8")
    parser.add_argument("--model_path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--image", default="/data/wy/wall-x/test_images/fruits_on_table.png")
    parser.add_argument("--question", default="Describe what you see in this image.")
    parser.add_argument("--mode", choices=["baseline", "int8", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    modes = [args.mode] if args.mode != "all" else ["baseline", "int8"]
    all_stats = []

    for mode in modes:
        print(f"\n{'#'*80}")
        print(f"  Testing mode: {mode}")
        print(f"{'#'*80}")

        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        if mode == "baseline":
            model, processor = load_model_baseline(args.model_path)
            label = "bf16-baseline"
        elif mode == "int8":
            model, processor = load_model_int8_bnb(args.model_path)
            label = "bnb-int8"

        inputs = prepare_input(processor, args.image, args.question)
        stats = benchmark_inference(model, inputs, args.max_new_tokens,
                                    args.warmup, args.runs, label)
        if stats:
            all_stats.append(stats)

        del model, processor, inputs
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    # Print comparison
    if len(all_stats) > 1:
        print(f"\n{'='*80}")
        print(f"  Comparison")
        print(f"{'='*80}")
        print(f"{'Mode':<25} {'Mean(ms)':>10} {'Std':>8} {'tok/s':>8} {'GPU(GB)':>8} {'Speedup':>8}")
        print("-" * 80)
        baseline_ms = all_stats[0]["mean_ms"]
        for s in all_stats:
            speedup = baseline_ms / s["mean_ms"]
            print(f"{s['label']:<25} {s['mean_ms']:>10.1f} {s['std_ms']:>8.1f} "
                  f"{s['avg_tok_per_s']:>8.1f} {s['peak_gpu_gb']:>8.2f} {speedup:>7.2f}x")
        print(f"{'='*80}")
    elif len(all_stats) == 1:
        s = all_stats[0]
        print(f"\n{s['label']}: {s['mean_ms']:.1f}ms, {s['avg_tok_per_s']:.1f} tok/s, {s['peak_gpu_gb']:.2f} GB")


if __name__ == "__main__":
    main()
