#!/usr/bin/env python3
"""Build a wall-x style VQA request JSON for Edge-LLM comparisons."""

import argparse
import json
from pathlib import Path


def make_request(image: str, question: str, system_prompt: str, max_generate_length: int) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    )
    return {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": max_generate_length,
        "requests": [{"messages": messages}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a wall-x style VQA request JSON")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    request = make_request(
        image=args.image,
        question=args.question,
        system_prompt=args.system_prompt,
        max_generate_length=args.max_generate_length,
    )
    Path(args.output_json).write_text(json.dumps(request, indent=2, ensure_ascii=False))
    print(json.dumps(request, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
