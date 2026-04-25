"""Minimal Triton kernel test on Jetson Orin (SM 8.7)"""
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def test():
    n = 1024
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.randn(n, device='cuda', dtype=torch.float32)
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print(f"Triton: {triton.__version__}")
    print("Launching Triton kernel...")
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    # verify
    expected = x + y
    if torch.allclose(output, expected):
        print("SUCCESS: Triton kernel works on Orin!")
    else:
        print("FAIL: Results mismatch")
        print(f"  max diff: {(output - expected).abs().max().item()}")

if __name__ == "__main__":
    test()
