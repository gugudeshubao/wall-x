#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys

USER_SITE = "/home/dog/.local/lib/python3.10/site-packages"
EDGE_VENV_SITE = "/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/lib/python3.10/site-packages"
EDGE_REPO_ROOT = "/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM"
EDGE_EXPERIMENTAL = "/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/experimental"


def prepare_hybrid_imports() -> None:
    tmpdir = "/data/wy/tmp"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ.setdefault("TMPDIR", tmpdir)
    os.environ.setdefault("TMP", tmpdir)
    os.environ.setdefault("TEMP", tmpdir)

    if USER_SITE not in sys.path:
        sys.path.insert(0, USER_SITE)
    import torch  # noqa: F401

    sys.path = [p for p in sys.path if "/home/dog/.local" not in p]
    for key in list(sys.modules.keys()):
        if key == "numpy" or key.startswith("numpy."):
            del sys.modules[key]
        if key == "typing_extensions" or key.startswith("typing_extensions."):
            del sys.modules[key]

    sys.path.insert(0, EDGE_VENV_SITE)
    sys.path.insert(0, EDGE_REPO_ROOT)
    sys.path.insert(0, EDGE_EXPERIMENTAL)
    sys.path.insert(0, "/Users/sam/project/github/wall-x/workspace")


def main() -> None:
    prepare_hybrid_imports()
    from run_edge_llm_llm_loader_hybrid import patch_llm_loader_awq_repack

    patch_llm_loader_awq_repack()
    runpy.run_module("awq_repacked_weight_stats", run_name="__main__")


if __name__ == "__main__":
    main()
