#!/usr/bin/env python3
"""
Batch benchmark for TensorRT-Edge-LLM VQA models.

This script is the reusable bridge layer for testing whether a model can be
treated as a wall-x VQA backend on Orin or on another embedded target.

It benchmarks multiple model entries from a JSON manifest, records wall-clock,
and keeps the generated output JSON for later comparison.
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelSpec:
    name: str
    model_id: str
    model_name: str
    engine_dir: str
    visual_engine_dir: str
    image: str
    prompt: str
    max_generate_length: int = 32


def load_models(path: Path) -> list[ModelSpec]:
    items = json.loads(path.read_text())
    models: list[ModelSpec] = []
    for item in items:
        models.append(
            ModelSpec(
                name=item["name"],
                model_id=item["model_id"],
                model_name=item["model_name"],
                engine_dir=item["engine_dir"],
                visual_engine_dir=item["visual_engine_dir"],
                image=item["image"],
                prompt=item.get("prompt", "Please describe the image."),
                max_generate_length=int(item.get("max_generate_length", 32)),
            )
        )
    return models


def make_env(repo_root: Path, build_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin:{env.get('PATH', '')}"
    env["HOME"] = "/data/wy/hf_home"
    env["XDG_CACHE_HOME"] = "/data/wy/hf_home/.cache"
    env["HF_HOME"] = "/data/wy/hf_cache"
    env["HF_HUB_CACHE"] = "/data/wy/hf_cache/hub"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_ETAG_TIMEOUT"] = "60"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    env["EDGELLM_PLUGIN_PATH"] = str(build_dir / "libNvInfer_edgellm_plugin.so")
    env["EDGE_LLM_REPO_ROOT"] = str(repo_root)
    env["EDGE_LLM_BUILD_DIR"] = str(build_dir)
    return env


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def write_input_json(path: Path, image: str, prompt: str, max_generate_length: int) -> None:
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


def load_output_text(output_json: Path) -> str:
    data = json.loads(output_json.read_text())
    responses = data.get("responses", [])
    if not responses:
        return ""
    return str(responses[0].get("output_text", ""))


def compare_with_reference(candidate_text: str, reference_json: Path | None) -> dict[str, Any]:
    if reference_json is None or not reference_json.exists():
        return {}
    ref_text = load_output_text(reference_json)
    return {
        "reference_text": ref_text,
        "exact_match": candidate_text == ref_text,
        "normalized_match": candidate_text.strip() == ref_text.strip(),
    }


def benchmark_one(
    spec: ModelSpec,
    repo_root: Path,
    build_dir: Path,
    work_root: Path,
    env: dict[str, str],
    runs: int,
    warmup: int,
    reference_json: Path | None,
    prompt_override: str,
) -> dict[str, Any]:
    model_dir = Path(spec.engine_dir)
    visual_dir = Path(spec.visual_engine_dir)
    input_json = work_root / f"input_vlm_{spec.name}.json"
    output_json = work_root / f"output_vlm_{spec.name}.json"
    summary_json = work_root / f"benchmark_vlm_{spec.name}.json"

    input_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    prompt = prompt_override or spec.prompt
    write_input_json(input_json, spec.image, prompt, spec.max_generate_length)

    cmd = [
        "./examples/llm/llm_inference",
        "--engineDir",
        str(model_dir),
        "--multimodalEngineDir",
        str(visual_dir),
        "--inputFile",
        str(input_json),
        "--outputFile",
        str(output_json),
    ]

    for _ in range(warmup):
        run_cmd(cmd, cwd=build_dir, env=env)

    times: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        run_cmd(cmd, cwd=build_dir, env=env)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times.append(dt_ms)
        print(f"[RUN] {spec.name} #{i+1}: {dt_ms:.3f} ms", flush=True)

    output_text = load_output_text(output_json)
    compare = compare_with_reference(output_text, reference_json)
    summary = {
        "name": spec.name,
        "model_id": spec.model_id,
        "model_name": spec.model_name,
        "engine_dir": spec.engine_dir,
        "visual_engine_dir": spec.visual_engine_dir,
        "image": spec.image,
        "prompt": prompt,
        "runs": runs,
        "warmup": warmup,
        "times_ms": times,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "output_json": str(output_json),
        "output_text": output_text,
        **compare,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple TensorRT-Edge-LLM VQA models")
    parser.add_argument("--repo-root", default="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM")
    parser.add_argument("--build-dir", default="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin")
    parser.add_argument("--work-root", default="/data/wy/wall-x/workspace/edge_llm_exp")
    parser.add_argument("--models-file", default=str(Path(__file__).with_name("models.edge_llm.json")))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt-override", default="", help="Override prompt for all models in the suite.")
    parser.add_argument(
        "--reference-json",
        default="",
        help="Optional reference JSON to compare output text against.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    build_dir = Path(args.build_dir)
    work_root = Path(args.work_root)
    models = load_models(Path(args.models_file))
    env = make_env(repo_root, build_dir)
    reference_json = Path(args.reference_json) if args.reference_json else None

    work_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for spec in models:
        print(f"\n=== Benchmark: {spec.name} ({spec.model_id}) ===", flush=True)
        summary = benchmark_one(
            spec=spec,
            repo_root=repo_root,
            build_dir=build_dir,
            work_root=work_root,
            env=env,
            runs=args.runs,
            warmup=args.warmup,
            reference_json=reference_json,
            prompt_override=args.prompt_override,
        )
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    fleet_summary = {
        "models": [
            {
                "name": item["name"],
                "model_id": item["model_id"],
                "mean_ms": item["mean_ms"],
                "min_ms": item["min_ms"],
                "max_ms": item["max_ms"],
                "exact_match": item.get("exact_match"),
                "normalized_match": item.get("normalized_match"),
            }
            for item in all_summaries
        ]
    }
    (work_root / "edge_llm_bridge_suite_summary.json").write_text(
        json.dumps(fleet_summary, indent=2, ensure_ascii=False)
    )
    print("\n=== Fleet summary ===", flush=True)
    print(json.dumps(fleet_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
