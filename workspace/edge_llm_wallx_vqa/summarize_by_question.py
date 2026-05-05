#!/usr/bin/env python3
"""Summarize a fleet_summary.json by question type."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Edge-LLM wall-x VQA results by question")
    parser.add_argument("--fleet-summary", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    data = json.loads(Path(args.fleet_summary).read_text())
    summary = {"models": []}
    for model in data["models"]:
        buckets = {}
        for case in model["cases"]:
            q = case["question"]
            bucket = buckets.setdefault(q, {"n": 0, "latency_ms": 0.0, "similarity": 0.0})
            bucket["n"] += 1
            bucket["latency_ms"] += case["latency_ms"]
            bucket["similarity"] += case["similarity"]
        question_summary = {}
        for q, bucket in buckets.items():
            question_summary[q] = {
                "n": bucket["n"],
                "mean_latency_ms": bucket["latency_ms"] / bucket["n"],
                "mean_similarity": bucket["similarity"] / bucket["n"],
            }
        summary["models"].append(
            {
                "model_name": model["model_name"],
                "model_id": model["model_id"],
                "questions": question_summary,
            }
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
