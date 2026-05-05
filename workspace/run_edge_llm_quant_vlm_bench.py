#!/usr/bin/env python3
"""
Quantize a VLM checkpoint with TensorRT-Edge-LLM's standalone quantization
pipeline, then run the generic VLM export/build/infer benchmark.

This is the supported path to test official quantization formats such as
fp8, nvfp4, mxfp8, and int8_sq without relying on the broken AWQ branch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_REPO_ROOT = Path("/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM")
DEFAULT_WORK_ROOT = Path("/data/wy/wall-x/workspace/edge_llm_exp")
DEFAULT_QUANT_PYTHON_BIN = Path("/data/wy/wall-x/venv/bin/python")
DEFAULT_BENCH_PYTHON_BIN = Path("/data/wy/wall-x/venv/bin/python")


def make_env(repo_root: Path, python_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(python_bin.parent) + ":" + env.get("PATH", "")
    env["HOME"] = "/data/wy/hf_home"
    env["XDG_CACHE_HOME"] = "/data/wy/hf_home/.cache"
    env["HF_HOME"] = "/data/wy/hf_cache"
    env["HF_HUB_CACHE"] = "/data/wy/hf_cache/hub"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_ETAG_TIMEOUT"] = "60"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    env["EDGELLM_PLUGIN_PATH"] = str(repo_root / "build_orin" / "libNvInfer_edgellm_plugin.so")
    return env


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def quant_tag(quantization: str | None, lm_head_quantization: str | None, kv_cache_quantization: str | None) -> str:
    q = quantization or "fp16"
    lh = lm_head_quantization or "none"
    kv = kv_cache_quantization or "none"
    return f"{q}_lmhead_{lh}_kv_{kv}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a VLM checkpoint and run the Edge-LLM VLM benchmark"
    )
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--quant-python-bin", default=str(DEFAULT_QUANT_PYTHON_BIN))
    parser.add_argument("--bench-python-bin", default=str(DEFAULT_BENCH_PYTHON_BIN))
    parser.add_argument("--model-id", required=True, help="Base model ID or local snapshot path")
    parser.add_argument("--model-name", required=True, help="Name used for local output directories")
    parser.add_argument(
        "--visual-model-id",
        default="",
        help="Use a different checkpoint for visual export if needed.",
    )
    parser.add_argument(
        "--quantization",
        default="nvfp4",
        choices=["fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"],
        help="Backbone quantization method",
    )
    parser.add_argument(
        "--lm-head-quantization",
        default="none",
        choices=["none", "fp8", "int4_awq", "nvfp4", "mxfp8"],
        help="LM-head quantization method (use none for backbone-only methods like int8_sq)",
    )
    parser.add_argument(
        "--kv-cache-quantization",
        default="none",
        choices=["none", "fp8"],
        help="KV-cache quantization method (use none for methods like int8_sq)",
    )
    parser.add_argument("--export-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dataset", default="cnn_dailymail")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--image", default="/data/wy/wall-x/test_images/fruits_on_table.png")
    parser.add_argument("--prompt", default="Please describe the image.")
    parser.add_argument("--max-generate-length", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    work_root = Path(args.work_root)
    quant_python = Path(args.quant_python_bin)
    bench_python = Path(args.bench_python_bin)
    env = make_env(repo_root, quant_python)
    visual_model_id = args.visual_model_id or args.model_id

    quant_root = work_root / "work_quant" / args.model_name / quant_tag(
        args.quantization, args.lm_head_quantization, args.kv_cache_quantization
    )
    quant_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_quantize:
        quant_cmd = [
            str(quant_python),
            "-m",
            "experimental.quantization.cli",
            "llm",
            "--model_dir",
            args.model_id,
            "--output_dir",
            str(quant_root),
            "--quantization",
            args.quantization,
            "--dtype",
            "fp16",
            "--device",
            args.export_device,
            "--dataset",
            args.dataset,
            "--num_samples",
            str(args.num_samples),
        ]
        if args.lm_head_quantization != "none":
            quant_cmd.extend(["--lm_head_quantization", args.lm_head_quantization])
        if args.kv_cache_quantization != "none":
            quant_cmd.extend(["--kv_cache_quantization", args.kv_cache_quantization])
        run(quant_cmd, cwd=repo_root, env=env)
    else:
        print(f"[SKIP] quantization step skipped, reusing: {quant_root}", flush=True)

    bench_cmd = [
        str(bench_python),
        str(Path(__file__).with_name("run_edge_llm_vlm_bench_generic.py")),
        "--repo-root",
        str(repo_root),
        "--work-root",
        str(work_root),
        "--python-bin",
        str(bench_python),
        "--export-tool",
        "llm_loader",
        "--model-id",
        str(quant_root),
        "--model-name",
        f"{args.model_name}_{quant_tag(args.quantization, args.lm_head_quantization, args.kv_cache_quantization)}",
        "--visual-model-id",
        visual_model_id,
        "--export-device",
        args.export_device,
        "--image",
        args.image,
        "--prompt",
        args.prompt,
        "--max-generate-length",
        str(args.max_generate_length),
        "--runs",
        str(args.runs),
    ]
    run(bench_cmd, cwd=work_root, env=env)


if __name__ == "__main__":
    main()
