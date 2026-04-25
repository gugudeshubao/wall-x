#!/usr/bin/env python3
"""Benchmark GEMM optimizations: baseline vs torch.compile vs INT8 weight-only.

Usage:
    python bench_gemm_opt.py --model_path /data/wy/models/wall-oss-flow --mode baseline
    python bench_gemm_opt.py --model_path /data/wy/models/wall-oss-flow --mode compile
    python bench_gemm_opt.py --model_path /data/wy/models/wall-oss-flow --mode int8
    python bench_gemm_opt.py --model_path /data/wy/models/wall-oss-flow --mode all
"""
import os, sys, time, argparse, gc, glob
import torch
import numpy as np


def load_model(model_path, device="cuda"):
    """Load wall-x VQA model with FA2 attention."""
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


def apply_torch_compile(model):
    """Apply torch.compile to the model."""
    print("\n[torch.compile] Compiling model...")
    t0 = time.time()
    try:
        # Try reduce-overhead first (uses CUDA graphs)
        compiled = torch.compile(model, mode="reduce-overhead")
        print(f"[torch.compile] mode=reduce-overhead, took {time.time()-t0:.1f}s")
        return compiled, "compile(reduce-overhead)"
    except Exception as e:
        print(f"[torch.compile] reduce-overhead failed: {e}")
        try:
            compiled = torch.compile(model, mode="default")
            print(f"[torch.compile] mode=default, took {time.time()-t0:.1f}s")
            return compiled, "compile(default)"
        except Exception as e2:
            print(f"[torch.compile] default also failed: {e2}")
            try:
                compiled = torch.compile(model, backend="eager")
                print(f"[torch.compile] backend=eager (no optimization), took {time.time()-t0:.1f}s")
                return compiled, "compile(eager)"
            except Exception as e3:
                print(f"[torch.compile] All modes failed: {e3}")
                return model, "compile(FAILED)"


def quantize_int8_weight_only(model):
    """Apply INT8 weight-only quantization to nn.Linear layers.

    Manual per-channel INT8: store weights as int8 + fp32 scales.
    At inference: dequantize to bf16 before matmul.
    This tests if cuBLAS picks INT8 GEMM automatically.
    """
    print("\n[INT8] Quantizing nn.Linear weights to INT8...")
    t0 = time.time()
    count = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data  # [out, in], bf16
            # Per-channel quantization (per output channel)
            scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / 127.0
            weight_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
            # Store quantized weight and scale
            module.weight_int8 = weight_int8
            module.weight_scale = scale.to(torch.bfloat16)
            module.weight.data = (weight_int8.to(torch.bfloat16) * scale.to(torch.bfloat16))
            count += 1
            total_params += weight.numel()

    elapsed = time.time() - t0
    print(f"[INT8] Quantized {count} Linear layers ({total_params/1e6:.1f}M params) in {elapsed:.1f}s")
    print(f"[INT8] Note: This is dequantize-on-store, testing weight compression effect.")
    return model


def try_native_int8_mm(model):
    """Try to use torch._int_mm for actual INT8 GEMM.

    Replace Linear forward to use int8 weights + _int_mm.
    """
    print("\n[INT8-MM] Attempting native INT8 matmul via torch._int_mm...")

    # Check if _int_mm is available
    if not hasattr(torch, '_int_mm'):
        print("[INT8-MM] torch._int_mm not available in this PyTorch version")
        return model, False

    # Test if _int_mm works on this device
    try:
        a = torch.randint(-128, 127, (16, 32), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 127, (32, 64), dtype=torch.int8, device="cuda")
        c = torch._int_mm(a, b)
        print(f"[INT8-MM] torch._int_mm works! Output dtype: {c.dtype}")
    except Exception as e:
        print(f"[INT8-MM] torch._int_mm failed on CUDA: {e}")
        return model, False

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight_int8'):
            original_forward = module.forward

            def make_int8_forward(mod):
                w_int8 = mod.weight_int8  # [out, in]
                w_scale = mod.weight_scale  # [out, 1]
                bias = mod.bias

                def int8_forward(x):
                    # x: [..., in_features] bf16
                    orig_shape = x.shape
                    x_2d = x.reshape(-1, x.shape[-1])  # [batch, in]

                    # Quantize input per-token
                    x_scale = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / 127.0
                    x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

                    # INT8 GEMM: [batch, in] @ [in, out] -> [batch, out] (int32)
                    out_int32 = torch._int_mm(x_int8, w_int8.t())

                    # Dequantize: multiply by scales
                    out_bf16 = out_int32.to(torch.bfloat16) * (x_scale * w_scale.t())

                    if bias is not None:
                        out_bf16 = out_bf16 + bias

                    return out_bf16.reshape(*orig_shape[:-1], -1)

                return int8_forward

            module.forward = make_int8_forward(module)
            count += 1

    print(f"[INT8-MM] Replaced {count} Linear.forward with INT8 matmul")
    return model, count > 0


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
        "median_ms": np.median(latencies),
        "avg_tokens": avg_tokens,
        "avg_tok_per_s": avg_tokens / (np.mean(latencies) / 1000),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    return stats


def print_comparison(all_stats):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"  GEMM Optimization Comparison")
    print(f"{'='*80}")
    print(f"{'Mode':<30} {'Mean(ms)':>10} {'Std':>8} {'tok/s':>8} {'GPU(GB)':>8} {'Speedup':>8}")
    print("-" * 80)

    baseline_ms = None
    for s in all_stats:
        if s is None:
            continue
        if baseline_ms is None:
            baseline_ms = s["mean_ms"]
        speedup = baseline_ms / s["mean_ms"] if s["mean_ms"] > 0 else 0
        print(f"{s['label']:<30} {s['mean_ms']:>10.1f} {s['std_ms']:>8.1f} "
              f"{s['avg_tok_per_s']:>8.1f} {s['peak_gpu_gb']:>8.2f} {speedup:>7.2f}x")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM optimizations")
    parser.add_argument("--model_path", default="/data/wy/models/wall-oss-flow")
    parser.add_argument("--image", default="/data/wy/wall-x/test_images/fruits_on_table.png")
    parser.add_argument("--question", default="Describe what you see in this image.")
    parser.add_argument("--mode", choices=["baseline", "compile", "int8", "int8mm", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")
    print(f"torch._int_mm available: {hasattr(torch, '_int_mm')}")

    modes = [args.mode] if args.mode != "all" else ["baseline", "compile"]
    all_stats = []

    for mode in modes:
        print(f"\n{'#'*80}")
        print(f"  Testing mode: {mode}")
        print(f"{'#'*80}")

        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        model, processor = load_model(args.model_path)
        inputs = prepare_input(processor, args.image, args.question)

        label = mode
        if mode == "compile":
            model, label = apply_torch_compile(model)
        elif mode == "int8":
            model = quantize_int8_weight_only(model)
            label = "int8-weight-only(dequant)"
        elif mode == "int8mm":
            model = quantize_int8_weight_only(model)
            model, success = try_native_int8_mm(model)
            label = "int8-native-mm" if success else "int8mm(FAILED->dequant)"

        stats = benchmark_inference(model, inputs, args.max_new_tokens,
                                    args.warmup, args.runs, label)
        if stats:
            all_stats.append(stats)

        # Cleanup
        del model, processor, inputs
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    if len(all_stats) > 1:
        print_comparison(all_stats)
    elif len(all_stats) == 1:
        s = all_stats[0]
        print(f"\n{s['label']}: {s['mean_ms']:.1f}ms, {s['avg_tok_per_s']:.1f} tok/s, {s['peak_gpu_gb']:.2f} GB")


if __name__ == "__main__":
    main()
