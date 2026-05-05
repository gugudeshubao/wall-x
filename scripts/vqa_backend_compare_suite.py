#!/usr/bin/env python3
"""Compare wall-x and Edge-LLM backends on a batch of VQA cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_cases(cases_json: Path) -> list[dict]:
    data = json.loads(cases_json.read_text())
    return data["cases"]


def filter_cases(cases: list[dict], image_names: list[str], questions: list[str]) -> list[dict]:
    selected = []
    for case in cases:
        if image_names and case["image"] not in image_names:
            continue
        if questions and case["question"] not in questions:
            continue
        selected.append(case)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare wall-x and Edge-LLM backends across multiple cases")
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--image-root", default="/data/wy/wall-x/test_images")
    parser.add_argument("--edge-backend-config", required=True)
    parser.add_argument("--wallx-model-path", required=True)
    parser.add_argument("--wallx-train-config", default="")
    parser.add_argument("--python-bin", default="/data/wy/wall-x/venv/bin/python")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--images", nargs="*", default=[], help="Optional subset of image filenames")
    parser.add_argument("--questions", nargs="*", default=[], help="Optional subset of questions")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases_json))
    cases = filter_cases(cases, args.images, args.questions)
    if args.limit > 0:
        cases = cases[: args.limit]

    tmp_dir = REPO_ROOT / "workspace" / "edge_llm_wallx_vqa" / "compare_suite_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for idx, case in enumerate(cases):
        case_out = tmp_dir / f"case_{idx:02d}.json"
        cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts" / "vqa_backend_compare.py"),
            "--edge-backend-config",
            args.edge_backend_config,
            "--wallx-model-path",
            args.wallx_model_path,
            "--image",
            str(Path(args.image_root) / case["image"]),
            "--question",
            case["question"],
            "--output-json",
            str(case_out),
        ]
        if args.wallx_train_config:
            cmd.extend(["--wallx-train-config", args.wallx_train_config])

        import subprocess

        subprocess.run(cmd, check=True)
        result = json.loads(case_out.read_text())
        result["reference_answer"] = case["reference_answer"]
        result["reference_latency_ms"] = case["reference_latency_ms"]
        all_results.append(result)
        print(
            f"[{idx+1}/{len(cases)}] {case['image']} | {case['question'][:30]}... | sim={result['similarity']:.4f}",
            flush=True,
        )

    summary = {
        "num_cases": len(all_results),
        "mean_similarity": sum(r["similarity"] for r in all_results) / len(all_results) if all_results else 0.0,
        "mean_edge_latency_ms": sum(r["edge"]["latency_ms"] for r in all_results) / len(all_results) if all_results else 0.0,
        "results": all_results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
