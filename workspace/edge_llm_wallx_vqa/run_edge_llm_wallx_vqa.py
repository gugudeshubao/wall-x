#!/usr/bin/env python3
"""Run TensorRT-Edge-LLM on wall-x VQA cases and compare to wall-x baseline."""

import argparse
import json
import os
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path


def make_env(build_dir: Path) -> dict[str, str]:
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
    return env


def load_models(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["cases"]


def write_input_json(
    path: Path,
    image: str,
    prompt: str,
    max_generate_length: int,
    system_prompt: str,
) -> None:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    )
    payload = {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": max_generate_length,
        "requests": [
            {
                "messages": messages
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def run_inference(build_dir: Path, engine_dir: str, visual_engine_dir: str, input_file: Path, output_file: Path, env: dict[str, str]) -> float:
    cmd = [
        "./examples/llm/llm_inference",
        "--engineDir",
        engine_dir,
        "--multimodalEngineDir",
        visual_engine_dir,
        "--inputFile",
        str(input_file),
        "--outputFile",
        str(output_file),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(build_dir), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (time.perf_counter() - t0) * 1000.0


def load_output_text(path: Path) -> str:
    data = json.loads(path.read_text())
    responses = data.get("responses", [])
    if not responses:
        return ""
    return str(responses[0].get("output_text", ""))


def similarity(a: str, b: str) -> float:
    a_norm = " ".join(a.split())
    b_norm = " ".join(b.split())
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run wall-x VQA cases through TensorRT-Edge-LLM")
    parser.add_argument("--build-dir", default="/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/build_orin")
    parser.add_argument("--image-root", default="/data/wy/wall-x/test_images")
    parser.add_argument("--models-file", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model-name", default="", help="Only run one model entry by name")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of cases to run")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-generate-length", type=int, default=32)
    parser.add_argument(
        "--compat-mode",
        choices=["default", "wallx_vqa"],
        default="default",
        help="Preset input organization. wallx_vqa currently means no system prompt + reference token length.",
    )
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--use-reference-token-length", action="store_true")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    image_root = Path(args.image_root)
    work_root = Path(args.work_root)
    env = make_env(build_dir)

    models = load_models(Path(args.models_file))
    cases = load_cases(Path(args.cases_file))
    if args.model_name:
        models = [m for m in models if m["name"] == args.model_name]
    if args.limit > 0:
        cases = cases[: args.limit]

    if args.compat_mode == "wallx_vqa":
        args.no_system_prompt = True
        args.use_reference_token_length = True

    system_prompt = "" if args.no_system_prompt else args.system_prompt

    work_root.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model in models:
        model_name = model["name"]
        model_dir = work_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_results = []

        for case in cases:
            case_dir = model_dir / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            input_json = case_dir / "input.json"
            output_json = case_dir / "output.json"
            image_path = image_root / case["image"]
            max_generate_length = (
                int(case["reference_tokens"]) if args.use_reference_token_length else args.max_generate_length
            )
            write_input_json(
                input_json,
                str(image_path),
                case["question"],
                max_generate_length,
                system_prompt,
            )

            for _ in range(args.warmup):
                run_inference(
                    build_dir,
                    model["engine_dir"],
                    model["visual_engine_dir"],
                    input_json,
                    output_json,
                    env,
                )

            latency_ms = run_inference(
                build_dir,
                model["engine_dir"],
                model["visual_engine_dir"],
                input_json,
                output_json,
                env,
            )
            output_text = load_output_text(output_json)
            ratio = similarity(case["reference_answer"], output_text)
            result = {
                "case_id": case["case_id"],
                "image": case["image"],
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "candidate_answer": output_text,
                "latency_ms": latency_ms,
                "similarity": ratio,
                "reference_latency_ms": case["reference_latency_ms"],
                "max_generate_length": max_generate_length,
                "system_prompt": system_prompt,
            }
            model_results.append(result)
            print(
                f"[{model_name}] {case['case_id']} | {latency_ms:.1f} ms | sim={ratio:.4f} | {output_text[:80]}",
                flush=True,
            )

        summary = {
            "model_name": model_name,
            "model_id": model["model_id"],
            "num_cases": len(model_results),
            "mean_latency_ms": sum(x["latency_ms"] for x in model_results) / len(model_results),
            "mean_similarity": sum(x["similarity"] for x in model_results) / len(model_results),
            "system_prompt": system_prompt,
            "use_reference_token_length": args.use_reference_token_length,
            "compat_mode": args.compat_mode,
            "cases": model_results,
        }
        (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        all_results.append(summary)

    fleet = {"models": all_results}
    (work_root / "fleet_summary.json").write_text(json.dumps(fleet, indent=2, ensure_ascii=False))
    print(json.dumps(fleet, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
