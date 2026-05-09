"""Check which SDPA backend is actually used on Orin."""
import torch
import torch.nn.functional as F

print(f"PyTorch: {torch.__version__}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"SM: {torch.cuda.get_device_capability()}")
print()

# wall-x actual dimensions (GQA: 16 heads, 2 KV heads)
q = torch.randn(1, 16, 420, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, 2, 420, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, 2, 420, 128, dtype=torch.bfloat16, device="cuda")

# Some SDPA builds do not accept dense GQA inputs directly for backend forcing.
# Expand KV heads explicitly so backend probing still works across versions.
kv_groups = q.size(1) // k.size(1)
k_sdpa = k.repeat_interleave(kv_groups, dim=1)
v_sdpa = v.repeat_interleave(kv_groups, dim=1)

# Test each backend individually
print("=== Testing individual backends ===")
backends_available = {}

# 1. flash
try:
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
    backends_available['flash'] = True
    print("flash_attention: AVAILABLE")
except Exception as e:
    backends_available['flash'] = False
    print(f"flash_attention: UNAVAILABLE - {e}")

# 2. mem_efficient 
try:
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
    backends_available['mem_efficient'] = True
    print("efficient_attention: AVAILABLE")
except Exception as e:
    backends_available['mem_efficient'] = False
    print(f"efficient_attention: UNAVAILABLE - {e}")

# 3. cudnn
try:
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
    backends_available['cudnn'] = True
    print("cudnn_attention: AVAILABLE")
except Exception as e:
    backends_available['cudnn'] = False
    print(f"cudnn_attention: UNAVAILABLE - {e}")

# 4. math
try:
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
    backends_available['math'] = True
    print("math: AVAILABLE")
except Exception as e:
    backends_available['math'] = False
    print(f"math: UNAVAILABLE - {e}")

# Benchmark each available backend
import time
print("\n=== Benchmark ===")
for name, backend_enum in [
    ("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
    ("efficient", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
    ("cudnn", torch.nn.attention.SDPBackend.CUDNN_ATTENTION),
    ("math", torch.nn.attention.SDPBackend.MATH),
]:
    if not backends_available.get(name.replace("efficient","mem_efficient").replace("cudnn","cudnn").replace("flash","flash").replace("math","math"), False):
        continue
    try:
        with torch.nn.attention.sdpa_kernel([backend_enum]):
            # warmup
            for _ in range(10):
                out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / 100 * 1000
        print(f"{name:20s}: {elapsed:.4f} ms")
    except Exception as e:
        print(f"{name:20s}: FAILED - {e}")

# Default backend benchmark
print("\ndefault (auto-select):")
for _ in range(10):
    out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) / 100 * 1000
print(f"{'default':20s}: {elapsed:.4f} ms")
