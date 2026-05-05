#!/usr/bin/env python3
"""Edge-LLM VQA backend wrapper for wall-x style experiments."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional at import time
    Image = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - optional at import time
    np = None  # type: ignore


@dataclass
class EdgeLLMVQABackendConfig:
    build_dir: str
    engine_dir: str
    multimodal_engine_dir: str
    plugin_path: str
    work_root: str
    compat_mode: str = "wallx_vqa"
    system_prompt: str = "You are a helpful assistant."
    max_new_tokens: int = 20
    hf_home: str = "/data/wy/hf_cache"
    hf_endpoint: str = "https://hf-mirror.com"
    venv_bin: str = "/data/wy/wall-x/workspace/edge_llm_exp/venv_edge/bin"


class EdgeLLMVQABackend:
    """Thin subprocess wrapper around TensorRT-Edge-LLM llm_inference."""

    def __init__(self, config: EdgeLLMVQABackendConfig):
        self.config = config
        self.build_dir = Path(config.build_dir)
        self.engine_dir = Path(config.engine_dir)
        self.multimodal_engine_dir = Path(config.multimodal_engine_dir)
        self.work_root = Path(config.work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

    def _make_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PATH"] = f"{self.config.venv_bin}:{env.get('PATH', '')}"
        env["HOME"] = "/data/wy/hf_home"
        env["XDG_CACHE_HOME"] = "/data/wy/hf_home/.cache"
        env["HF_HOME"] = self.config.hf_home
        env["HF_HUB_CACHE"] = f"{self.config.hf_home}/hub"
        env["HF_ENDPOINT"] = self.config.hf_endpoint
        env["HF_HUB_ETAG_TIMEOUT"] = "60"
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
        env["EDGELLM_PLUGIN_PATH"] = self.config.plugin_path
        return env

    def _resolve_generation_config(self, max_new_tokens: int | None, system_prompt: str | None) -> tuple[int, str]:
        resolved_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        if self.config.compat_mode == "wallx_vqa":
            resolved_system = "" if system_prompt is None else system_prompt
        else:
            resolved_system = self.config.system_prompt if system_prompt is None else system_prompt
        return resolved_tokens, resolved_system

    def _save_temp_image(self, image: Any, temp_dir: Path) -> Path:
        if Image is None:
            raise RuntimeError("Pillow is required when passing a PIL image object")
        if isinstance(image, str):
            return Path(image)
        if isinstance(image, Path):
            return image
        if np is not None and isinstance(image, np.ndarray):
            array = image
            if array.ndim > 3:
                array = array.squeeze()
            if array.ndim == 3 and (array.shape[0] == 1 or array.shape[0] == 3):
                array = array.transpose(1, 2, 0)
            if array.dtype != np.uint8:
                array = (array * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(array)
        else:
            try:
                import torch

                if isinstance(image, torch.Tensor):
                    tensor = image.detach().cpu()
                    if tensor.ndim > 3:
                        tensor = tensor.squeeze()
                    if tensor.ndim == 3 and (tensor.shape[0] == 1 or tensor.shape[0] == 3):
                        tensor = tensor.permute(1, 2, 0)
                    if tensor.dtype != torch.uint8:
                        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
                    image = Image.fromarray(tensor.numpy())
            except Exception:
                pass
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)!r}")
        temp_path = temp_dir / "input_image.png"
        image.save(temp_path)
        return temp_path

    def _build_request(self, image_path: str, question: str, max_new_tokens: int, system_prompt: str) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        )
        return {
            "batch_size": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": max_new_tokens,
            "requests": [{"messages": messages}],
        }

    def generate_with_metadata(
        self,
        image: Any,
        text: str,
        *,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        resolved_tokens, resolved_system = self._resolve_generation_config(max_new_tokens, system_prompt)
        env = self._make_env()

        with tempfile.TemporaryDirectory(dir=str(self.work_root), prefix="edge_llm_vqa_") as tmp:
            tmp_dir = Path(tmp)
            image_path = self._save_temp_image(image, tmp_dir)
            input_json = tmp_dir / "input.json"
            output_json = tmp_dir / "output.json"
            request = self._build_request(str(image_path), text, resolved_tokens, resolved_system)
            input_json.write_text(json.dumps(request, ensure_ascii=False, indent=2))

            cmd = [
                "./examples/llm/llm_inference",
                "--engineDir",
                str(self.engine_dir),
                "--multimodalEngineDir",
                str(self.multimodal_engine_dir),
                "--inputFile",
                str(input_json),
                "--outputFile",
                str(output_json),
            ]

            t0 = time.perf_counter()
            subprocess.run(
                cmd,
                cwd=str(self.build_dir),
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            response_payload = json.loads(output_json.read_text())
            output_text = ""
            responses = response_payload.get("responses", [])
            if responses:
                output_text = str(responses[0].get("output_text", ""))

            return {
                "question": text,
                "image_path": str(image_path),
                "max_new_tokens": resolved_tokens,
                "system_prompt": resolved_system,
                "latency_ms": latency_ms,
                "output_text": output_text,
                "raw_response": response_payload,
            }

    def generate(
        self,
        image: Any,
        text: str,
        **kwargs: Any,
    ) -> str:
        return self.generate_with_metadata(image, text, **kwargs)["output_text"]
