"""
Fused Residual Add + RMSNorm Triton Kernel for SM 8.7 (Jetson Orin)

Fuses two operations into one kernel:
  1. residual = residual + x  (residual connection)
  2. out = rmsnorm(residual, weight, eps)

Saves one global memory round-trip vs separate add + rmsnorm.

Model dimensions (wall-x 3B):
  hidden_size = 2048
  rms_norm_eps = 1e-6
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def fused_add_rmsnorm_kernel(
    # Pointers
    X_ptr,          # [M, N] input from attention/mlp output
    Residual_ptr,   # [M, N] residual stream (read+write, updated in-place)
    Weight_ptr,     # [N]    rmsnorm weight
    Out_ptr,        # [M, N] normalized output
    # Dimensions
    M,              # number of rows (batch * seq_len)
    N: tl.constexpr,  # hidden_size (2048)
    # Params
    eps: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Each program instance processes one row."""
    row_idx = tl.program_id(0)
    row_offset = row_idx * N

    # Accumulator for sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    residual_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Pass 1: Load x and residual, compute residual += x, accumulate x^2
    # Process in blocks if N > BLOCK_SIZE
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        x_vals = tl.load(X_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
        res_vals = tl.load(Residual_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # Fused add: residual += x
        new_res = res_vals + x_vals

        # Store updated residual back
        tl.store(Residual_ptr + row_offset + col_offsets, new_res.to(tl.bfloat16), mask=mask)

        # Accumulate for RMS: sum(x^2)
        sum_sq += new_res * new_res

    # Compute RMS
    mean_sq = tl.sum(sum_sq, axis=0) / N
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # Pass 2: Normalize and apply weight
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Re-load updated residual
        res_vals = tl.load(Residual_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(Weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # RMSNorm: out = (residual / rms) * weight
        out_vals = res_vals * rrms * w_vals

        tl.store(Out_ptr + row_offset + col_offsets, out_vals.to(tl.bfloat16), mask=mask)


# Single-pass version: avoids re-loading residual by caching in registers
# Works when BLOCK_SIZE >= N (true for hidden_size=2048)
@triton.jit
def fused_add_rmsnorm_single_pass_kernel(
    X_ptr,
    Residual_ptr,
    Weight_ptr,
    Out_ptr,
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass version: BLOCK_SIZE must be >= N.
    Keeps residual in registers to avoid second global memory read."""
    row_idx = tl.program_id(0)
    row_offset = row_idx * N

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load x and residual
    x_vals = tl.load(X_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
    res_vals = tl.load(Residual_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused add
    new_res = res_vals + x_vals

    # Store updated residual
    tl.store(Residual_ptr + row_offset + col_offsets, new_res.to(tl.bfloat16), mask=mask)

    # Compute RMS (in registers, no re-load)
    sum_sq = tl.sum(new_res * new_res, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / N + eps)

    # Load weight and normalize
    w_vals = tl.load(Weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    out_vals = new_res * rrms * w_vals

    tl.store(Out_ptr + row_offset + col_offsets, out_vals.to(tl.bfloat16), mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor,           # [M, N] bf16
    residual: torch.Tensor,    # [M, N] bf16 (modified in-place)
    weight: torch.Tensor,      # [N] bf16
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused residual add + RMSNorm.
    Updates residual in-place: residual += x
    Returns: rmsnorm(residual, weight, eps)
    """
    assert x.shape == residual.shape
    assert x.shape[-1] == weight.shape[0]
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    out = torch.empty_like(x)

    # Use single-pass kernel when hidden_size fits in one block (N <= 2048)
    if N <= 2048:
        BLOCK_SIZE = triton.next_power_of_2(N)
        grid = (M,)
        fused_add_rmsnorm_single_pass_kernel[grid](
            x, residual, weight, out,
            M, N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        grid = (M,)
        fused_add_rmsnorm_kernel[grid](
            x, residual, weight, out,
            M, N, eps,
        )
    return out


# --- Standalone RMSNorm (no residual add) for pre-attention norm ---

@triton.jit
def rmsnorm_kernel(
    X_ptr,
    Weight_ptr,
    Out_ptr,
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm without residual add. BLOCK_SIZE must be >= N."""
    row_idx = tl.program_id(0)
    row_offset = row_idx * N

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x_vals = tl.load(X_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x_vals * x_vals, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / N + eps)

    w_vals = tl.load(Weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    out_vals = x_vals * rrms * w_vals

    tl.store(Out_ptr + row_offset + col_offsets, out_vals.to(tl.bfloat16), mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standalone RMSNorm via Triton."""
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_kernel[grid](
        x, weight, out,
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# --- Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    M, N = 128, 2048  # batch*seq=128, hidden=2048
    eps = 1e-6

    x = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.ones(N, device=device, dtype=dtype)

    # Reference: PyTorch
    ref_residual = residual.clone()
    ref_residual += x
    ref_rms = torch.sqrt(ref_residual.float().pow(2).mean(-1, keepdim=True) + eps)
    ref_out = (ref_residual.float() / ref_rms * weight.float()).to(dtype)

    # Triton
    tri_residual = residual.clone()
    tri_out = fused_add_rmsnorm(x, tri_residual, weight, eps)

    # Check
    print(f"residual max diff: {(tri_residual - ref_residual).abs().max().item():.6f}")
    print(f"output max diff:   {(tri_out - ref_out).abs().max().item():.6f}")
    assert (tri_out - ref_out).abs().max().item() < 0.05, "Output mismatch!"
    assert (tri_residual - ref_residual).abs().max().item() < 0.01, "Residual mismatch!"
    print("PASSED: fused_add_rmsnorm matches PyTorch reference")
