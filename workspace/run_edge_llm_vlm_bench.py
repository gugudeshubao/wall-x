#!/usr/bin/env python3
"""
End-to-end benchmark for TensorRT-Edge-LLM VLM on Orin.

Default target:
  Qwen/Qwen3-VL-2B-Instruct

The script will:
  1. export LLM and visual ONNX if missing
  2. build llm.engine + visual.engine if missing
  3. run llm_inference 3 times and record wall-clock time

All cache locations are redirected away from /home to /data.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path("/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM")
WORK_ROOT = Path("/data/wy/wall-x/workspace/edge_llm_exp")
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_NAME = "qwen3-vl-2b"
IMAGE_PATH = "/data/wy/wall-x/test_images/fruits_on_table.png"
PROMPT = "Please describe the image."

ONNX_ROOT = WORK_ROOT / "work_onnx" / MODEL_NAME
LLM_ONNX_DIR = ONNX_ROOT
VISUAL_ONNX_DIR = ONNX_ROOT / "visual_enc_onnx"
ENGINE_ROOT = WORK_ROOT / "edge_engines" / MODEL_NAME
LOG_DIR = WORK_ROOT / "logs"
INPUT_JSON = WORK_ROOT / f"input_vlm_{MODEL_NAME}.json"
OUTPUT_JSON = WORK_ROOT / f"output_vlm_{MODEL_NAME}.json"
SUMMARY_JSON = WORK_ROOT / f"benchmark_vlm_{MODEL_NAME}.json"


def env():
    e = os.environ.copy()
    e["PATH"] = "/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin:" + e.get("PATH", "")
    e["HOME"] = "/data/wy/hf_home"
    e["XDG_CACHE_HOME"] = "/data/wy/hf_home/.cache"
    e["HF_HOME"] = "/data/wy/hf_cache"
    e["HF_HUB_CACHE"] = "/data/wy/hf_cache/hub"
    e["HF_ENDPOINT"] = "https://hf-mirror.com"
    e["HF_HUB_ETAG_TIMEOUT"] = "60"
    e["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    e["EDGELLM_PLUGIN_PATH"] = str(REPO_ROOT / "build_orin" / "libNvInfer_edgellm_plugin.so")
    return e


def run(cmd, cwd=None):
    print(f"\n[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env(), check=True)


def ensure_dirs():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_ROOT.mkdir(parents=True, exist_ok=True)
    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_exports():
    llm_onnx = LLM_ONNX_DIR / "model.onnx"
    visual_onnx = VISUAL_ONNX_DIR / "model.onnx"
    if not llm_onnx.exists():
        run([
            "tensorrt-edgellm-export-llm",
            "--model_dir", MODEL_ID,
            "--output_dir", str(LLM_ONNX_DIR),
            "--device", "cpu",
        ], cwd=str(REPO_ROOT))
    else:
        print(f"[SKIP] LLM export exists: {llm_onnx}", flush=True)

    if not visual_onnx.exists():
        run([
            "tensorrt-edgellm-export-visual",
            "--model_dir", MODEL_ID,
            "--output_dir", str(VISUAL_ONNX_DIR),
            "--device", "cpu",
        ], cwd=str(REPO_ROOT))
    else:
        print(f"[SKIP] Visual export exists: {visual_onnx}", flush=True)


def ensure_builds():
    llm_engine = ENGINE_ROOT / "llm.engine"
    visual_engine = ENGINE_ROOT / "visual" / "visual.engine"
    if not llm_engine.exists():
        run([
            "./examples/llm/llm_build",
            "--onnxDir", str(LLM_ONNX_DIR),
            "--engineDir", str(ENGINE_ROOT),
            "--maxBatchSize", "1",
            "--maxInputLen", "1024",
            "--maxKVCacheCapacity", "4096",
        ], cwd=str(REPO_ROOT / "build_orin"))
    else:
        print(f"[SKIP] LLM engine exists: {llm_engine}", flush=True)

    if not visual_engine.exists():
        run([
            "./examples/multimodal/visual_build",
            "--onnxDir", str(VISUAL_ONNX_DIR),
            "--engineDir", str(ENGINE_ROOT),
            "--minImageTokens", "128",
            "--maxImageTokens", "512",
            "--maxImageTokensPerImage", "512",
        ], cwd=str(REPO_ROOT / "build_orin"))
    else:
        print(f"[SKIP] Visual engine exists: {visual_engine}", flush=True)


def write_input():
    payload = {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": 32,
        "requests": [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": IMAGE_PATH},
                            {"type": "text", "text": PROMPT},
                        ],
                    },
                ]
            }
        ],
    }
    INPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def benchmark():
    cmd = [
        "./examples/llm/llm_inference",
        "--engineDir", str(ENGINE_ROOT),
        "--multimodalEngineDir", str(ENGINE_ROOT / "visual"),
        "--inputFile", str(INPUT_JSON),
        "--outputFile", str(OUTPUT_JSON),
    ]
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        run(cmd, cwd=str(REPO_ROOT / "build_orin"))
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        print(f"[RUN] {i+1}: {dt:.3f} ms", flush=True)
    summary = {
        "model_id": MODEL_ID,
        "times_ms": times,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ensure_dirs()
    write_input()
    ensure_exports()
    ensure_builds()
    benchmark()


if __name__ == "__main__":
    main()
