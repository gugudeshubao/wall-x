#!/usr/bin/env python3
"""
Extract a minimal Edge-LLM-style reference JSON from a wall-x benchmark dump.

This is useful when we want to compare Edge-LLM output text against an existing
wall-x answer for the exact same image/question pair.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wall-x reference output JSON")
    parser.add_argument("--source-json", required=True, help="wall-x benchmark JSON")
    parser.add_argument("--image", required=True, help="Image filename, e.g. fruits_on_table.png")
    parser.add_argument("--question", required=True, help="Question text to match exactly")
    parser.add_argument("--output-json", required=True, help="Path to write the minimal reference JSON")
    args = parser.parse_args()

    data = json.loads(Path(args.source_json).read_text())
    results = data.get("bf16") or data.get("int8") or data.get("results") or []
    matched = None
    for item in results:
        if item.get("image") == args.image and item.get("question") == args.question:
            matched = item
            break
    if matched is None:
        raise SystemExit(
            f"Could not find a matching entry for image={args.image!r} question={args.question!r}"
        )

    reference = {
        "responses": [
            {
                "output_text": matched.get("answer", ""),
                "image": matched.get("image"),
                "question": matched.get("question"),
                "source": str(args.source_json),
            }
        ]
    }
    Path(args.output_json).write_text(json.dumps(reference, indent=2, ensure_ascii=False))
    print(json.dumps(reference, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
