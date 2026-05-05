#!/usr/bin/env python3
"""
Debug helper for llm_loader AWQ repacking on Orin.

It uses the same hybrid import idea as `run_edge_llm_llm_loader_hybrid.py`,
then monkey-patches AWQ repacking to print the exact failing module and tensor
shapes during checkpoint load.
"""

from __future__ import annotations

import argparse
import sys


USER_SITE = "/home/dog/.local/lib/python3.10/site-packages"
EDGE_VENV_SITE = "/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/lib/python3.10/site-packages"
EDGE_EXPERIMENTAL = "/data/wy/wall-x/workspace/edge_llm_exp/TensorRT-Edge-LLM/experimental"


def prepare_hybrid_imports() -> None:
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
    sys.path.insert(0, EDGE_EXPERIMENTAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    prepare_hybrid_imports()

    import torch
    from llm_loader.model import AutoModel
    import llm_loader.checkpoint.repacking as repacking

    original = repacking.repack_awq_to_plugin
    current_module_name = {"name": None}

    def wrapped(qweight, qzeros):
        try:
            return original(qweight, qzeros)
        except Exception as e:  # pragma: no cover - debug only
            print(
                "FAIL_MODULE",
                current_module_name["name"],
                "qweight",
                tuple(qweight.shape),
                str(qweight.dtype),
                "qweight_stride",
                tuple(qweight.stride()),
                "qweight_contig",
                qweight.is_contiguous(),
                "qzeros",
                tuple(qzeros.shape),
                str(qzeros.dtype),
                "qzeros_stride",
                tuple(qzeros.stride()),
                "qzeros_contig",
                qzeros.is_contiguous(),
                repr(e),
                flush=True,
            )
            try:
                fixed = original(qweight.contiguous(), qzeros.contiguous())
                print(
                    "CONTIGUOUS_RETRY_OK",
                    current_module_name["name"],
                    tuple(fixed.shape),
                    str(fixed.dtype),
                    flush=True,
                )
            except Exception as e2:
                print("CONTIGUOUS_RETRY_FAIL", current_module_name["name"], repr(e2), flush=True)
            raise

    repacking.repack_awq_to_plugin = wrapped

    model = AutoModel.from_pretrained(args.model_dir, device="cpu")

    # Manually reproduce the named_modules repacking loop so we can attribute the failure.
    from llm_loader.models.linear import AWQLinear

    for name, module in model.named_modules():
        if isinstance(module, AWQLinear):
            current_module_name["name"] = name
            qw = module._buffers.get("qweight")
            qz = module._buffers.get("qzeros")
            if qw is not None and qw.dtype == torch.int32 and qz is not None:
                module._buffers["qweight"] = repacking.repack_awq_to_plugin(qw, qz)
    print("ALL_REPACK_OK", flush=True)


if __name__ == "__main__":
    main()
