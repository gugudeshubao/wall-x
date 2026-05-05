#!/usr/bin/env python3
"""Compare a wall-x VQA reference output to TensorRT-Edge-LLM output."""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare wall-x and Edge-LLM outputs")
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", default="", help="Optional path to write the comparison summary as JSON.")
    args = parser.parse_args()

    ref = json.loads(Path(args.reference_json).read_text())
    cand = json.loads(Path(args.candidate_json).read_text())

    ref_text = ref["responses"][0]["output_text"]
    cand_text = cand["responses"][0]["output_text"]
    ref_norm = " ".join(ref_text.split())
    cand_norm = " ".join(cand_text.split())
    ratio = SequenceMatcher(None, ref_norm, cand_norm).ratio()
    summary = {
        "reference_json": args.reference_json,
        "candidate_json": args.candidate_json,
        "reference_text": ref_text,
        "candidate_text": cand_text,
        "exact_match": ref_text == cand_text,
        "normalized_match": ref_norm == cand_norm,
        "ratio": ratio,
    }

    print("[REF]", ref_text)
    print("[CAND]", cand_text)
    print("[MATCH]", summary["exact_match"])
    print("[MATCH_STRIP]", summary["normalized_match"])
    print("[RATIO]", f"{ratio:.4f}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
