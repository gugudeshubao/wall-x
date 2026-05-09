"""
Triton Kernel Compilation Pipeline for SM 8.7 (Jetson Orin)

Compiles Triton kernels into cubin files that can be loaded by the C++ runtime
via CUDA Driver API. Also saves launch metadata (grid, block, shared_mem).

Usage:
  python compile_kernels.py --output-dir ../cpp_infer/kernels/
"""

import argparse
import json
import os

import torch

from fused_add_rmsnorm import (
    fused_add_rmsnorm_single_pass_kernel,
    rmsnorm_kernel,
)
from fused_silu_mul import fused_silu_mul_kernel


def get_device_capability():
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return f"{cap[0]}.{cap[1]}"
    return "8.7"  # Default for Orin


def compile_kernel(kernel_fn, name, example_args, warmup_kwargs, output_dir):
    """Compile a Triton kernel to cubin and save metadata."""
    print(f"Compiling {name}...")
    compiled = kernel_fn.warmup(*example_args, grid=(1,), **warmup_kwargs)

    # Save cubin
    cubin_path = os.path.join(output_dir, f"{name}.cubin")
    with open(cubin_path, "wb") as f:
        f.write(compiled.asm["cubin"])

    # Save metadata for C++ launcher
    meta = {
        "name": name,
        "kernel_name": getattr(compiled.metadata, "name", getattr(compiled, "name", name)),
        "num_warps": getattr(compiled.metadata, "num_warps", 0),
        "num_stages": getattr(compiled.metadata, "num_stages", 0),
        "shared_mem": getattr(compiled.metadata, "shared", 0),
        "constants": {k: v for k, v in warmup_kwargs.items() if isinstance(v, (int, float))},
    }
    meta_path = os.path.join(output_dir, f"{name}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    cubin_size = os.path.getsize(cubin_path)
    print(f"  -> {cubin_path} ({cubin_size} bytes)")
    print(f"  -> {meta_path}")
    return cubin_path, meta_path


def main():
    parser = argparse.ArgumentParser(description="Compile Triton kernels to cubin")
    parser.add_argument("--output-dir", default="../cpp_infer/kernels/",
                        help="Output directory for cubin files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = get_device_capability()
    print(f"Target device: SM {cap}")
    print(f"Output directory: {args.output_dir}")
    print()

    # --- Model dimensions ---
    HIDDEN_SIZE = 2048
    INTERMEDIATE_SIZE_0 = 11008  # Expert 0
    INTERMEDIATE_SIZE_1 = 2048   # Expert 1
    EPS = 1e-6

    results = []

    x = torch.empty((1, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    residual = torch.empty((1, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((HIDDEN_SIZE,), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((1, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)

    # 1. fused_add_rmsnorm (single pass, BLOCK_SIZE=2048 for hidden=2048)
    try:
        cubin, meta = compile_kernel(
            kernel_fn=fused_add_rmsnorm_single_pass_kernel,
            name="fused_add_rmsnorm_h2048",
            example_args=(x, residual, weight, out, 1, HIDDEN_SIZE, EPS, 2048),
            warmup_kwargs={},
            output_dir=args.output_dir,
        )
        results.append(("fused_add_rmsnorm_h2048", cubin, meta))
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. rmsnorm standalone (for first layer input norm)
    try:
        cubin, meta = compile_kernel(
            kernel_fn=rmsnorm_kernel,
            name="rmsnorm_h2048",
            example_args=(x, weight, out, 1, HIDDEN_SIZE, EPS, 2048),
            warmup_kwargs={},
            output_dir=args.output_dir,
        )
        results.append(("rmsnorm_h2048", cubin, meta))
    except Exception as e:
        print(f"  FAILED: {e}")

    gate0 = torch.empty((1, INTERMEDIATE_SIZE_0), device="cuda", dtype=torch.bfloat16)
    up0 = torch.empty((1, INTERMEDIATE_SIZE_0), device="cuda", dtype=torch.bfloat16)
    out0 = torch.empty((1, INTERMEDIATE_SIZE_0), device="cuda", dtype=torch.bfloat16)
    # 3. fused_silu_mul for expert 0 (intermediate=11008)
    try:
        cubin, meta = compile_kernel(
            kernel_fn=fused_silu_mul_kernel,
            name="fused_silu_mul_n11008",
            example_args=(gate0, up0, out0, 1, INTERMEDIATE_SIZE_0, 4096),
            warmup_kwargs={},
            output_dir=args.output_dir,
        )
        results.append(("fused_silu_mul_n11008", cubin, meta))
    except Exception as e:
        print(f"  FAILED: {e}")

    gate1 = torch.empty((1, INTERMEDIATE_SIZE_1), device="cuda", dtype=torch.bfloat16)
    up1 = torch.empty((1, INTERMEDIATE_SIZE_1), device="cuda", dtype=torch.bfloat16)
    out1 = torch.empty((1, INTERMEDIATE_SIZE_1), device="cuda", dtype=torch.bfloat16)
    # 4. fused_silu_mul for expert 1 (intermediate=2048)
    try:
        cubin, meta = compile_kernel(
            kernel_fn=fused_silu_mul_kernel,
            name="fused_silu_mul_n2048",
            example_args=(gate1, up1, out1, 1, INTERMEDIATE_SIZE_1, 2048),
            warmup_kwargs={},
            output_dir=args.output_dir,
        )
        results.append(("fused_silu_mul_n2048", cubin, meta))
    except Exception as e:
        print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"Compiled {len(results)} / 4 kernels successfully")
    for name, cubin_path, _ in results:
        print(f"  {name}: {cubin_path}")


if __name__ == "__main__":
    main()
