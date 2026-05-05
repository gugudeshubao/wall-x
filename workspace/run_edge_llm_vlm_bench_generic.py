#!/usr/bin/env python3
"""
Generic end-to-end benchmark for TensorRT-Edge-LLM VLM models on Orin.

It can benchmark both original and quantized checkpoints with the same flow:
  1. export LLM and visual ONNX if missing
  2. build llm.engine + visual.engine if missing
  3. run llm_inference N times and record wall-clock
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_REPO_ROOT = Path("/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM")
DEFAULT_WORK_ROOT = Path("/data/wy/wall-x/workspace/edge_llm_exp")


def make_env(repo_root: Path) -> dict[str, str]:
    e = os.environ.copy()
    e["PATH"] = "/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin:" + e.get("PATH", "")
    e["HOME"] = "/data/wy/hf_home"
    e["XDG_CACHE_HOME"] = "/data/wy/hf_home/.cache"
    e["HF_HOME"] = "/data/wy/hf_cache"
    e["HF_HUB_CACHE"] = "/data/wy/hf_cache/hub"
    e["HF_ENDPOINT"] = "https://hf-mirror.com"
    e["HF_HUB_ETAG_TIMEOUT"] = "60"
    e["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    e["EDGELLM_PLUGIN_PATH"] = str(repo_root / "build_orin" / "libNvInfer_edgellm_plugin.so")
    return e


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def write_input(path: Path, image: str, prompt: str, max_generate_length: int) -> None:
    payload = {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": max_generate_length,
        "requests": [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one Edge-LLM VLM model on Orin")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--visual-model-id", default="", help="Use a different checkpoint for visual export if needed.")
    parser.add_argument("--export-device", default="cpu", choices=["cpu", "cuda"], help="Device used for ONNX export.")
    parser.add_argument("--image", default="/data/wy/wall-x/test_images/fruits_on_table.png")
    parser.add_argument("--prompt", default="Please describe the image.")
    parser.add_argument("--max-generate-length", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    build_dir = repo_root / "build_orin"
    work_root = Path(args.work_root)
    env = make_env(repo_root)

    visual_model_id = args.visual_model_id or args.model_id

    onnx_root = work_root / "work_onnx" / args.model_name
    llm_onnx_dir = onnx_root
    visual_onnx_dir = onnx_root / "visual_enc_onnx"
    engine_root = work_root / "edge_engines" / args.model_name
    input_json = work_root / f"input_vlm_{args.model_name}.json"
    output_json = work_root / f"output_vlm_{args.model_name}.json"
    summary_json = work_root / f"benchmark_vlm_{args.model_name}.json"

    onnx_root.mkdir(parents=True, exist_ok=True)
    engine_root.mkdir(parents=True, exist_ok=True)
    write_input(input_json, args.image, args.prompt, args.max_generate_length)

    llm_onnx = llm_onnx_dir / "model.onnx"
    if not llm_onnx.exists():
        run(
            [
                "tensorrt-edgellm-export-llm",
                "--model_dir",
                args.model_id,
                "--output_dir",
                str(llm_onnx_dir),
                "--device",
                args.export_device,
            ],
            cwd=repo_root,
            env=env,
        )
    else:
        print(f"[SKIP] existing llm onnx: {llm_onnx}", flush=True)

    visual_onnx = visual_onnx_dir / "model.onnx"
    if not visual_onnx.exists():
        run(
            [
                "tensorrt-edgellm-export-visual",
                "--model_dir",
                visual_model_id,
                "--output_dir",
                str(visual_onnx_dir),
                "--device",
                args.export_device,
            ],
            cwd=repo_root,
            env=env,
        )
    else:
        print(f"[SKIP] existing visual onnx: {visual_onnx}", flush=True)

    llm_engine = engine_root / "llm.engine"
    if not llm_engine.exists():
        run(
            [
                "./examples/llm/llm_build",
                "--onnxDir",
                str(llm_onnx_dir),
                "--engineDir",
                str(engine_root),
                "--maxBatchSize",
                "1",
                "--maxInputLen",
                "1024",
                "--maxKVCacheCapacity",
                "4096",
            ],
            cwd=build_dir,
            env=env,
        )
    else:
        print(f"[SKIP] existing llm engine: {llm_engine}", flush=True)

    visual_engine = engine_root / "visual" / "visual.engine"
    if not visual_engine.exists():
        run(
            [
                "./examples/multimodal/visual_build",
                "--onnxDir",
                str(visual_onnx_dir),
                "--engineDir",
                str(engine_root),
                "--minImageTokens",
                "128",
                "--maxImageTokens",
                "512",
                "--maxImageTokensPerImage",
                "512",
            ],
            cwd=build_dir,
            env=env,
        )
    else:
        print(f"[SKIP] existing visual engine: {visual_engine}", flush=True)

    cmd = [
        "./examples/llm/llm_inference",
        "--engineDir",
        str(engine_root),
        "--multimodalEngineDir",
        str(engine_root / "visual"),
        "--inputFile",
        str(input_json),
        "--outputFile",
        str(output_json),
    ]
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        run(cmd, cwd=build_dir, env=env)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times.append(dt_ms)
        print(f"[RUN] {i + 1}: {dt_ms:.3f} ms", flush=True)

    summary = {
        "model_id": args.model_id,
        "visual_model_id": visual_model_id,
        "model_name": args.model_name,
        "image": args.image,
        "prompt": args.prompt,
        "times_ms": times,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "output_json": str(output_json),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
