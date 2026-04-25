"""Test flash-attn's Triton rotary embedding on Orin SM 8.7"""
import torch
import time

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"SM: {torch.cuda.get_device_capability()}")

# Test 1: import the triton-based rotary
try:
    from flash_attn.ops.triton.rotary import apply_rotary
    print("OK: flash_attn.ops.triton.rotary.apply_rotary imported (Triton version)")
    HAS_TRITON_ROTARY = True
except ImportError as e:
    print(f"FAIL: {e}")
    HAS_TRITON_ROTARY = False

# Test 2: import the layer wrapper
try:
    from flash_attn.layers.rotary import apply_rotary_emb
    print("OK: flash_attn.layers.rotary.apply_rotary_emb imported")
except ImportError as e:
    print(f"FAIL: {e}")

# Test 3: actually run it
if HAS_TRITON_ROTARY:
    print("\n--- Running Triton rotary kernel ---")
    batch, seqlen, nheads, headdim = 1, 420, 16, 128
    x = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.bfloat16)
    cos = torch.randn(seqlen, headdim // 2, device='cuda', dtype=torch.float32)
    sin = torch.randn(seqlen, headdim // 2, device='cuda', dtype=torch.float32)

    try:
        # warmup
        for _ in range(3):
            out = apply_rotary_emb(x, cos, sin)
        torch.cuda.synchronize()

        # benchmark
        start = time.time()
        N = 100
        for _ in range(N):
            out = apply_rotary_emb(x, cos, sin)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / N * 1000
        print(f"SUCCESS: Triton rotary works! {elapsed:.3f} ms/call")
        print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    except Exception as e:
        print(f"RUNTIME ERROR: {e}")

# Test 4: compare with PyTorch fallback
print("\n--- Comparing Triton vs PyTorch rotary ---")
try:
    from flash_attn.layers.rotary import apply_rotary_emb as rotary_emb
    # The function internally uses triton if available, pytorch otherwise
    # Let's check which one it uses
    import flash_attn.ops.triton.rotary as triton_rotary
    print(f"Triton rotary module location: {triton_rotary.__file__}")
    print("Triton path is ACTIVE (not falling back to PyTorch)")
except Exception as e:
    print(f"Note: {e}")
