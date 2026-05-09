#!/usr/bin/env python3
"""Compare Python bf16, C++ bf16, and C++ INT8 tokens side by side."""
import json
from pathlib import Path

input_dir = Path("vqa_test_inputs")
with open(input_dir / "manifest.json") as f:
    manifest = json.load(f)

header = "{:<20} {:>20} {:>20} {:>22}".format(
    "Image", "Py-bf16 vs C++-bf16", "Py-bf16 vs C++-INT8", "C++-bf16 vs C++-INT8")
print(header)
print("=" * 85)

total_pb, total_pi, total_bi = 0, 0, 0
total_n_pb, total_n_pi, total_n_bi = 0, 0, 0

for entry in manifest["images"]:
    name = Path(entry["image"]).stem
    py_toks = entry["token_ids"]

    bf16_path = input_dir / name / "cpp_output_tokens.txt"
    int8_path = input_dir / name / "cpp_int8_tokens.txt"

    bf16_toks = [int(x) for x in bf16_path.read_text().strip().split()]
    int8_toks = [int(x) for x in int8_path.read_text().strip().split()]

    def match_count(a, b):
        n = min(len(a), len(b))
        m = sum(1 for i in range(n) if a[i] == b[i])
        return m, n

    m_pb, n_pb = match_count(py_toks, bf16_toks)
    m_pi, n_pi = match_count(py_toks, int8_toks)
    m_bi, n_bi = match_count(bf16_toks, int8_toks)

    total_pb += m_pb; total_n_pb += n_pb
    total_pi += m_pi; total_n_pi += n_pi
    total_bi += m_bi; total_n_bi += n_bi

    row = "{:<20} {:>20} {:>20} {:>22}".format(
        name,
        "{}/{}".format(m_pb, n_pb),
        "{}/{}".format(m_pi, n_pi),
        "{}/{}".format(m_bi, n_bi))
    print(row)

print("=" * 85)
row = "{:<20} {:>20} {:>20} {:>22}".format(
    "TOTAL",
    "{}/{} ({:.0%})".format(total_pb, total_n_pb, total_pb/total_n_pb if total_n_pb else 0),
    "{}/{} ({:.0%})".format(total_pi, total_n_pi, total_pi/total_n_pi if total_n_pi else 0),
    "{}/{} ({:.0%})".format(total_bi, total_n_bi, total_bi/total_n_bi if total_n_bi else 0))
print(row)
