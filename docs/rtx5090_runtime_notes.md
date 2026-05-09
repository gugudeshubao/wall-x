# RTX 5090 Runtime Notes

## Login / Environment

- Host: private, see external local context
- Workdirs observed:
  - runtime source snapshot: `/home/ubuntu/project/wall-x`
  - lightweight helper dir: `/home/ubuntu/project/github/wall-x`
  - models: `/home/ubuntu/project/models/wall-oss-flow`
  - test images: `/home/ubuntu/project/test_images`
- GPU: `NVIDIA GeForce RTX 5090`
- Driver: `580.126.09`
- CUDA: `13.0`
- Python: `3.12.3`
- venv: `/home/ubuntu/project/github/wall-x/wallx-venv`

## Source Tree Notes

- `/home/ubuntu/project/wall-x` contains the main synced source tree (`wall_x/`, `scripts/`, `csrc/`, `workspace/`).
- `/home/ubuntu/project/github/wall-x` currently contains only a lightweight helper layout (`scripts/`, `wallx-venv`, and profiling artifacts).
- Neither of the two 5090-side directories is currently a git clone; they behave like synced working copies rather than branch-tracking repos.

## Packages

- `torch 2.11.0+cu130`
- `flash_attn 2.8.3`

## GPU State During Inspection

- Before benchmarking, one long-lived service was occupying most VRAM:
  - `python scripts/serve_policy.py --env DROID --port 8001`
  - GPU memory usage: about `24.6 GiB`
- The service was stopped before running the benchmarks below.

## Python Flow Action Benchmarks

All numbers below were measured on 5090 with:

- model: `wall-oss-flow`
- precision: `bf16`
- attention backend: `sdpa`
- ODE steps: `5`

### Dummy benchmark

Conditions:

- synthetic Flow Action input
- total sequence length `488`
- action horizon `32`
- benchmark runs `5`

Results:

- total: `151.8 ms`
- embed + ViT: `50.7 ms`
- prefill: `34.5 ms`
- ODE: `65.3 ms`
- other: `1.3 ms`
- throughput: `6.59 infer/s`
- peak GPU memory: `8.17 GB`

Warmup:

- run1: `423.3 ms`
- run2: `152.2 ms`

### Real-image benchmark

Conditions:

- image: `fruits_on_table.png`
- prompt: `"Pick up the red object on the table."`
- prefix length: `427`
- total sequence length: `459`
- action horizon `32`
- benchmark runs `5`

Results:

- total: `199.0 ms`
- embed + ViT: `100.8 ms`
- prefill: `31.0 ms`
- ODE: `65.8 ms`
- other: `1.4 ms`
- throughput: `5.02 infer/s`
- peak GPU memory: `8.40 GB`

Warmup:

- run1: `358.5 ms`
- run2: `199.3 ms`

## Quick Comparison

| Route | Input shape | Prefix | Total | ViT/embed | Prefill | ODE | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python Flow dummy | `488` | `456` | `151.8 ms` | `50.7 ms` | `34.5 ms` | `65.3 ms` | `6.59 infer/s` |
| Python Flow real image | `459` | `427` | `199.0 ms` | `100.8 ms` | `31.0 ms` | `65.8 ms` | `5.02 infer/s` |

## Current Takeaways

- On 5090, Python Flow Action is already fast enough that the ODE portion is no longer the dominant cost in the real-image case; `embed + ViT` is the largest block.
- Dummy vs real-image gap is mainly in the vision/input embedding stage:
  - dummy `50.7 ms`
  - real image `100.8 ms`
- Prefill and ODE are very stable across repeated runs:
  - prefill stays around `31-35 ms`
  - ODE stays around `65-66 ms`
