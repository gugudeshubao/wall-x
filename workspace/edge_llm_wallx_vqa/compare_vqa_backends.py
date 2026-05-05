#!/usr/bin/env python3
"""Compare one Edge-LLM VQA backend run against a wall-x reference case."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from vqa_wrapper_edge_llm import EdgeLLMVQAWrapper


def find_case(cases_path: Path, image_name: str, question: str) -> dict:
    data = json.loads(cases_path.read_text())
    for case in data["cases"]:
        if case["image"] == image_name and case["question"] == question:
            return case
    raise ValueError(f"No matching case for image={image_name!r}, question={question!r}")


def similarity(a: str, b: str) -> float:
    a_norm = " ".join(a.split())
    b_norm = " ".join(b.split())
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare an Edge-LLM wrapper output against wall-x VQA reference")
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--image", required=True, help="Image filename, e.g. fruits_on_table.png")
    parser.add_argument("--question", required=True)
    parser.add_argument("--image-root", default="/data/wy/wall-x/test_images")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    case = find_case(Path(args.cases_json), args.image, args.question)
    wrapper = EdgeLLMVQAWrapper(backend_config_path=args.backend_config)
    image = Image.open(Path(args.image_root) / args.image).convert("RGB")
    edge = wrapper.generate_with_metadata(image, args.question)
    ratio = similarity(case["reference_answer"], edge["output_text"])

    result = {
        "image": args.image,
        "question": args.question,
        "reference_answer": case["reference_answer"],
        "reference_latency_ms": case["reference_latency_ms"],
        "edge_latency_ms": edge["latency_ms"],
        "edge_output_text": edge["output_text"],
        "similarity": ratio,
        "formatted_complete_request": edge["raw_response"]["responses"][0]["formatted_complete_request"],
        "system_prompt": edge["system_prompt"],
        "max_new_tokens": edge["max_new_tokens"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
