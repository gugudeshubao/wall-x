"""
Fused SiLU Activation + Elementwise Multiply Triton Kernel for SM 8.7

Fuses the MLP gate activation:
  out = silu(gate) * up
  where silu(x) = x * sigmoid(x)

In Qwen2.5 MLP:
  gate = gate_proj(x)   # [M, intermediate_size]  -- cuBLAS GEMM
  up   = up_proj(x)     # [M, intermediate_size]  -- cuBLAS GEMM
  hidden = silu(gate) * up                         -- THIS KERNEL
  out  = down_proj(hidden)                         -- cuBLAS GEMM

Model dimensions (wall-x 3B):
  intermediate_size = 11008 (expert 0) or 2048 (expert 1)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def fused_silu_mul_kernel(
    Gate_ptr,       # [M, N] gate projection output
    Up_ptr,         # [M, N] up projection output
    Out_ptr,        # [M, N] output
    M,              # number of rows
    N,              # intermediate_size
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU(gate) * up. One program per row."""
    row_idx = tl.program_id(0)
    row_offset = row_idx * N

    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        gate = tl.load(Gate_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(Up_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        silu_gate = gate * tl.sigmoid(gate)

        # Elementwise multiply
        out = silu_gate * up

        tl.store(Out_ptr + row_offset + col_offsets, out.to(tl.bfloat16), mask=mask)


# Optimized version: 2D grid for large intermediate_size
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def fused_silu_mul_2d_kernel(
    Gate_ptr,
    Up_ptr,
    Out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D tiled version for better parallelism across rows and columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    for m in range(BLOCK_M):
        row = row_start + m
        if row < M:
            row_offset = row * N
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N

            gate = tl.load(Gate_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(Up_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)

            silu_gate = gate * tl.sigmoid(gate)
            out = silu_gate * up

            tl.store(Out_ptr + row_offset + col_offsets, out.to(tl.bfloat16), mask=mask)


def fused_silu_mul(
    gate: torch.Tensor,    # [M, N] bf16
    up: torch.Tensor,      # [M, N] bf16
) -> torch.Tensor:
    """
    Fused SiLU(gate) * up.
    Returns: [M, N] bf16
    """
    assert gate.shape == up.shape
    M = gate.shape[0] if gate.dim() == 2 else gate.numel() // gate.shape[-1]
    N = gate.shape[-1]
    out = torch.empty_like(gate)

    grid = (M,)
    fused_silu_mul_kernel[grid](
        gate, up, out,
        M, N,
    )
    return out


# --- Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    for M, N, name in [(128, 11008, "expert0"), (128, 2048, "expert1"), (1, 11008, "decode")]:
        gate = torch.randn(M, N, device=device, dtype=dtype)
        up = torch.randn(M, N, device=device, dtype=dtype)

        # Reference
        ref = torch.nn.functional.silu(gate.float()) * up.float()
        ref = ref.to(dtype)

        # Triton
        tri = fused_silu_mul(gate, up)

        max_diff = (tri - ref).abs().max().item()
        print(f"[{name}] M={M}, N={N}, max_diff={max_diff:.6f}")
        assert max_diff < 0.1, f"Output mismatch for {name}!"

    print("PASSED: fused_silu_mul matches PyTorch reference")
