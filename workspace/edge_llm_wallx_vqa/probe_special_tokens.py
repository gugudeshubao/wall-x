#!/usr/bin/env python3
"""Probe how Edge-LLM-backed tokenizers and models behave with wall-x special tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_llm_vqa_backend import EdgeLLMVQABackend, EdgeLLMVQABackendConfig


def load_backend_config(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe special-token behavior on Edge-LLM VQA backends")
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    cfg = EdgeLLMVQABackendConfig(**load_backend_config(Path(args.backend_config)))
    backend = EdgeLLMVQABackend(cfg)

    prompts = [
        "What objects are on the table?",
        "What objects are on the table?\nProprioception: <|propri|>",
        "What objects are on the table?\n<|action|>",
        "Predict the next action in robot action.\nProprioception: <|propri|>\n<|action|>",
    ]

    results = []
    for prompt in prompts:
        result = backend.generate_with_metadata(args.image, prompt, max_new_tokens=20)
        results.append(
            {
                "prompt": prompt,
                "latency_ms": result["latency_ms"],
                "output_text": result["output_text"],
                "formatted_complete_request": result["raw_response"]["responses"][0]["formatted_complete_request"],
            }
        )
        print(json.dumps(results[-1], indent=2, ensure_ascii=False))

    summary = {
        "backend_config": args.backend_config,
        "image": args.image,
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
