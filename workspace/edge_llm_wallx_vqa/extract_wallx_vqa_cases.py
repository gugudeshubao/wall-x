#!/usr/bin/env python3
"""Extract wall-x VQA cases into a compact JSON manifest."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wall-x VQA cases")
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--bucket", choices=["bf16", "int8"], default="bf16")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.source_json).read_text())
    rows = data[args.bucket]
    cases = []
    for idx, row in enumerate(rows):
        cases.append(
            {
                "case_id": f"{idx:02d}_{Path(row['image']).stem}",
                "image": row["image"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "reference_tokens": row["tokens"],
                "reference_latency_ms": row["latency_ms"],
                "reference_tok_per_s": row["tok_per_s"],
            }
        )

    output = {
        "source_json": args.source_json,
        "bucket": args.bucket,
        "num_cases": len(cases),
        "cases": cases,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
