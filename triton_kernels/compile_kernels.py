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
import sys

import torch
import triton

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


def compile_kernel(kernel_fn, name, signature, constants, num_warps, num_stages, output_dir):
    """Compile a Triton kernel to cubin and save metadata."""
    print(f"Compiling {name}...")

    compiled = triton.compile(
        fn=kernel_fn,
        signature=signature,
        constants=constants,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Save cubin
    cubin_path = os.path.join(output_dir, f"{name}.cubin")
    with open(cubin_path, "wb") as f:
        f.write(compiled.asm["cubin"])

    # Save metadata for C++ launcher
    meta = {
        "name": name,
        "kernel_name": compiled.name if hasattr(compiled, "name") else name,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "shared_mem": compiled.shared if hasattr(compiled, "shared") else 0,
        "constants": {k: v for k, v in constants.items() if isinstance(v, (int, float))},
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

    # 1. fused_add_rmsnorm (single pass, BLOCK_SIZE=2048 for hidden=2048)
    try:
        cubin, meta = compile_kernel(
            kernel_fn=fused_add_rmsnorm_single_pass_kernel,
            name="fused_add_rmsnorm_h2048",
            signature={
                0: "*bf16",  # X_ptr
                1: "*bf16",  # Residual_ptr
                2: "*bf16",  # Weight_ptr
                3: "*bf16",  # Out_ptr
                4: "i32",    # M
            },
            constants={
                "N": HIDDEN_SIZE,
                "eps": EPS,
                "BLOCK_SIZE": 2048,
            },
            num_warps=8,
            num_stages=2,
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
            signature={
                0: "*bf16",  # X_ptr
                1: "*bf16",  # Weight_ptr
                2: "*bf16",  # Out_ptr
                3: "i32",    # M
            },
            constants={
                "N": HIDDEN_SIZE,
                "eps": EPS,
                "BLOCK_SIZE": 2048,
            },
            num_warps=8,
            num_stages=2,
            output_dir=args.output_dir,
        )
        results.append(("rmsnorm_h2048", cubin, meta))
    except Exception as e:
        print(f"  FAILED: {e}")

    # 3. fused_silu_mul for expert 0 (intermediate=11008)
    try:
        cubin, meta = compile_kernel(
            kernel_fn=fused_silu_mul_kernel,
            name="fused_silu_mul_n11008",
            signature={
                0: "*bf16",  # Gate_ptr
                1: "*bf16",  # Up_ptr
                2: "*bf16",  # Out_ptr
                3: "i32",    # M
                4: "i32",    # N
            },
            constants={
                "BLOCK_SIZE": 4096,
            },
            num_warps=8,
            num_stages=2,
            output_dir=args.output_dir,
        )
        results.append(("fused_silu_mul_n11008", cubin, meta))
    except Exception as e:
        print(f"  FAILED: {e}")

    # 4. fused_silu_mul for expert 1 (intermediate=2048)
    try:
        cubin, meta = compile_kernel(
            kernel_fn=fused_silu_mul_kernel,
            name="fused_silu_mul_n2048",
            signature={
                0: "*bf16",  # Gate_ptr
                1: "*bf16",  # Up_ptr
                2: "*bf16",  # Out_ptr
                3: "i32",    # M
                4: "i32",    # N
            },
            constants={
                "BLOCK_SIZE": 2048,
            },
            num_warps=4,
            num_stages=2,
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
