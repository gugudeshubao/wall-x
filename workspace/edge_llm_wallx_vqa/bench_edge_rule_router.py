#!/usr/bin/env python3
"""Benchmark the rule-based Edge-LLM router on wall-x VQA cases."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from edge_rule_router import EdgeRuleRouter, EdgeRuleRouterConfig


def similarity(a: str, b: str) -> float:
    a_norm = " ".join(a.split())
    b_norm = " ".join(b.split())
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the rule-based Edge-LLM router on wall-x VQA cases")
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--image-root", default="/data/wy/wall-x/test_images")
    parser.add_argument("--qwen25-backend-config", required=True)
    parser.add_argument("--qwen3-backend-config", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    data = json.loads(Path(args.cases_json).read_text())
    router = EdgeRuleRouter(
        EdgeRuleRouterConfig(
            qwen25_backend_config=args.qwen25_backend_config,
            qwen3_backend_config=args.qwen3_backend_config,
        )
    )

    results = []
    for case in data["cases"]:
        image = Image.open(Path(args.image_root) / case["image"]).convert("RGB")
        result = router.generate_with_metadata(image, case["question"], max_new_tokens=case["reference_tokens"])
        score = similarity(case["reference_answer"], result["output_text"])
        row = {
            "image": case["image"],
            "question": case["question"],
            "reference_answer": case["reference_answer"],
            "candidate_answer": result["output_text"],
            "latency_ms": result["latency_ms"],
            "similarity": score,
            "routed_backend": result["routed_backend"],
        }
        results.append(row)
        print(f"{case['image']} | {case['question'][:32]}... | {result['routed_backend']} | {result['latency_ms']:.1f} ms | sim={score:.4f}")

    summary = {
        "num_cases": len(results),
        "mean_latency_ms": sum(r["latency_ms"] for r in results) / len(results) if results else 0.0,
        "mean_similarity": sum(r["similarity"] for r in results) / len(results) if results else 0.0,
        "results": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
