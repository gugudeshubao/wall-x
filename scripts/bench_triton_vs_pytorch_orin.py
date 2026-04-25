"""
Triton vs PyTorch operator benchmark on Jetson Orin (SM 8.7)
Tests: vector_add, softmax, matmul, layernorm, fused_add_rmsnorm
Purpose: evaluate Triton MLIR codegen quality on SM 8.7
"""
import torch
import triton
import triton.language as tl
import time
import sys

torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================
# 1. Vector Add (pure memory-bound baseline)
# ============================================================
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def triton_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

# ============================================================
# 2. Softmax (reduction + exp + normalize)
# ============================================================
@triton.jit
def _softmax_kernel(input_ptr, output_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    row_ptr = input_ptr + row * stride
    x = tl.load(row_ptr + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    out = num / den
    out_ptr = output_ptr + row * stride
    tl.store(out_ptr + offs, out, mask=mask)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), BLOCK=BLOCK)
    return out

# ============================================================
# 3. GEMM (compute-bound, tests Triton's tiling/MMA codegen)
# ============================================================
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)

def triton_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

# ============================================================
# 4. LayerNorm (reduction + normalize + affine)
# ============================================================
@triton.jit
def _layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    n_cols, eps,
    stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    out = xhat * w + b
    tl.store(out_ptr + row * stride + offs, out.to(tl.float16), mask=mask)

def triton_layernorm(x, weight, bias, eps=1e-5):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    _layernorm_kernel[(n_rows,)](
        x, weight, bias, out, n_cols, eps, x.stride(0), BLOCK=BLOCK
    )
    return out

# ============================================================
# 5. Fused Add + RMSNorm (common in LLM, tests fusion benefit)
# ============================================================
@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, residual_ptr, w_ptr, out_ptr,
    n_cols, eps,
    stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    ms = tl.sum(h * h, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    out = h * rstd * w
    tl.store(out_ptr + row * stride + offs, out.to(tl.float16), mask=mask)

def triton_fused_add_rmsnorm(x, residual, weight, eps=1e-5):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    _fused_add_rmsnorm_kernel[(n_rows,)](
        x, residual, weight, out, n_cols, eps, x.stride(0), BLOCK=BLOCK
    )
    return out

def pytorch_rmsnorm(x, residual, weight, eps=1e-5):
    h = x + residual
    ms = h.float().pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(ms + eps)
    return (h * weight).to(x.dtype)

# ============================================================
# Benchmark harness
# ============================================================
def bench(fn, warmup=10, repeat=100, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeat * 1000  # ms
    return elapsed

def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM:  {torch.cuda.get_device_capability()}")
    print(f"Triton: {triton.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 80)

    results = []

    # --- 1. Vector Add ---
    print("\n[1] Vector Add  (n=1M, fp16, pure memory-bound)")
    n = 1_000_000
    x = torch.randn(n, device=device, dtype=torch.float16)
    y = torch.randn(n, device=device, dtype=torch.float16)

    t_pt = bench(lambda: x + y, label="pytorch")
    t_tr = bench(lambda: triton_add(x, y), label="triton")
    ratio = t_pt / t_tr
    results.append(("vector_add (1M)", t_pt, t_tr, ratio))
    print(f"  PyTorch: {t_pt:.4f} ms  |  Triton: {t_tr:.4f} ms  |  ratio: {ratio:.2f}x")

    # --- 2. Softmax ---
    print("\n[2] Softmax  (2048 rows x 2048 cols, fp16)")
    x_sm = torch.randn(2048, 2048, device=device, dtype=torch.float16)

    t_pt = bench(lambda: torch.softmax(x_sm, dim=-1), label="pytorch")
    t_tr = bench(lambda: triton_softmax(x_sm), label="triton")
    ratio = t_pt / t_tr
    results.append(("softmax (2048x2048)", t_pt, t_tr, ratio))
    print(f"  PyTorch: {t_pt:.4f} ms  |  Triton: {t_tr:.4f} ms  |  ratio: {ratio:.2f}x")

    # wall-x actual: decode softmax (1 x 420)
    print("\n[2b] Softmax  (16 rows x 420 cols, fp16, wall-x decode scale)")
    x_sm2 = torch.randn(16, 420, device=device, dtype=torch.float16)

    t_pt = bench(lambda: torch.softmax(x_sm2, dim=-1))
    t_tr = bench(lambda: triton_softmax(x_sm2))
    ratio = t_pt / t_tr
    results.append(("softmax (16x420)", t_pt, t_tr, ratio))
    print(f"  PyTorch: {t_pt:.4f} ms  |  Triton: {t_tr:.4f} ms  |  ratio: {ratio:.2f}x")

    # --- 3. GEMM ---
    print("\n[3a] GEMM  (2048x2048 x 2048x2048, fp16, compute-bound)")
    M, N, K = 2048, 2048, 2048
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)

    t_pt = bench(lambda: torch.mm(a, b))
    t_tr = bench(lambda: triton_matmul(a, b))
    ratio = t_pt / t_tr
    results.append(("GEMM (2048x2048)", t_pt, t_tr, ratio))
    print(f"  PyTorch: {t_pt:.4f} ms  |  Triton: {t_tr:.4f} ms  |  ratio: {ratio:.2f}x")

    # wall-x actual: decode GEMV (1x2048 x 2048x5632)
    print("\n[3b] GEMV  (1x2048 x 2048x5632, fp16, wall-x decode)")
    a2 = torch.randn(1, 2048, device=device, dtype=torch.float16)
    b2 = torch.randn(2048, 5632, device=device, dtype=torch.float16)

    t_pt = bench(lambda: torch.mm(a2, b2))
    t_tr = bench(lambda: triton_matmul(a2, b2))
    ratio = t_pt / t_tr
    results.append(("GEMV (1x2048x5632)", t_pt, t_tr, ratio))
    print(f"  PyTorch: {t_pt:.4f} ms  |  Triton: {t_tr:.4f} ms  |  ratio: {ratio:.2f}x")

    # --- 4. LayerNorm ---
    print("\n[4] LayerNorm  (420x2048, fp16)")
    x_ln = torch.randn(420, 2048, device=device, dtype=torch.float16)
    w_ln = torch.ones(2048, device=device, dtype=torch.float16)
    b_ln = torch.zeros(2048, device=device, dtype=torch.float16)
    ln = torch.nn.LayerNorm(2048, device=device, dtype=torch.float16)

    t_pt = bench(lambda: ln(x_ln))
    t_tr = bench(lambda: triton_layernorm(x_ln, w_ln, b_ln))
    ratio = t_pt / t_tr
    results.append(("layernorm (420x2048)", t_pt, t_tr, ratio))
    print(f"  PyTorch: {t_pt:.4f} ms  |  Triton: {t_tr:.4f} ms  |  ratio: {ratio:.2f}x")

    # --- 5. Fused Add + RMSNorm ---
    print("\n[5] Fused Add+RMSNorm  (420x2048, fp16, tests fusion benefit)")
    x_rms = torch.randn(420, 2048, device=device, dtype=torch.float16)
    r_rms = torch.randn(420, 2048, device=device, dtype=torch.float16)
    w_rms = torch.ones(2048, device=device, dtype=torch.float16)

    t_pt = bench(lambda: pytorch_rmsnorm(x_rms, r_rms, w_rms))
    t_tr = bench(lambda: triton_fused_add_rmsnorm(x_rms, r_rms, w_rms))
    ratio = t_pt / t_tr
    results.append(("fused_add_rmsnorm (420x2048)", t_pt, t_tr, ratio))
    print(f"  PyTorch: {t_pt:.4f} ms  |  Triton: {t_tr:.4f} ms  |  ratio: {ratio:.2f}x")

    # --- Summary ---
    print("\n" + "=" * 80)
    print(f"{'Operator':<30} {'PyTorch':>10} {'Triton':>10} {'Ratio':>10}")
    print("-" * 60)
    for name, pt, tr, ratio in results:
        winner = "Triton" if ratio > 1.0 else "PyTorch"
        print(f"{name:<30} {pt:>9.4f}ms {tr:>9.4f}ms {ratio:>8.2f}x  <- {winner}")
    print("=" * 80)
    print("ratio > 1.0 = Triton faster, < 1.0 = PyTorch faster")

if __name__ == "__main__":
    main()
