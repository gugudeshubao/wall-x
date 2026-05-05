#!/usr/bin/env python3
"""CLI for the EdgeLLMVQAWrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vqa_wrapper_edge_llm import EdgeLLMVQAWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Edge-LLM VQA wrapper on one image/question.")
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    wrapper = EdgeLLMVQAWrapper(backend_config_path=args.backend_config)
    image = Image.open(args.image).convert("RGB")
    result = wrapper.generate_with_metadata(
        image,
        args.question,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
