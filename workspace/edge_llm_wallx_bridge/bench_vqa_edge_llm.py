#!/usr/bin/env python3
"""
Run a minimal wall-x VQA benchmark through TensorRT-Edge-LLM.

This script is intentionally narrow:
- single image
- single prompt
- repeated llm_inference calls
- record wall-clock

It is meant to answer the practical question:

> Can TensorRT-Edge-LLM be used as the VQA backend for wall-x?
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def write_input_json(path: Path, image: str, prompt: str, max_generate_length: int):
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


def run_cmd(cmd, cwd, env):
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (time.perf_counter() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark wall-x VQA via TensorRT-Edge-LLM")
    parser.add_argument("--repo-root", default="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM")
    parser.add_argument("--build-dir", default="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin")
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--visual-engine-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Please describe the image.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-generate-length", type=int, default=32)
    parser.add_argument("--log-json", required=True)
    args = parser.parse_args()

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = str(Path(args.build_dir) / "libNvInfer_edgellm_plugin.so")

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    log_json = Path(args.log_json)
    input_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_json.parent.mkdir(parents=True, exist_ok=True)

    write_input_json(input_json, args.image, args.prompt, args.max_generate_length)

    cmd = [
        "./examples/llm/llm_inference",
        "--engineDir", args.engine_dir,
        "--multimodalEngineDir", args.visual_engine_dir,
        "--inputFile", str(input_json),
        "--outputFile", str(output_json),
    ]

    for _ in range(args.warmup):
        run_cmd(cmd, cwd=args.build_dir, env=env)

    times = []
    for i in range(args.runs):
        dt = run_cmd(cmd, cwd=args.build_dir, env=env)
        times.append(dt)
        print(f"run_{i+1}_ms={dt:.3f}", flush=True)

    summary = {
        "engine_dir": args.engine_dir,
        "visual_engine_dir": args.visual_engine_dir,
        "image": args.image,
        "prompt": args.prompt,
        "runs": args.runs,
        "warmup": args.warmup,
        "times_ms": times,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }
    log_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

