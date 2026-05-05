from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError:
    logging.warning(
        "msgpack-numpy not installed. Install with: pip install msgpack-numpy"
    )
    msgpack = None

try:
    import websockets
except ImportError:
    logging.warning(
        "websockets not installed. Install with: pip install websockets"
    )
    websockets = None

logger = logging.getLogger(__name__)


class VQAClient:
    """Client for the VQA websocket server."""

    def __init__(self, uri: str = "ws://localhost:8000"):
        self.uri = uri
        self.websocket = None
        self.metadata = None
        self._loop = None
        self._thread = None

    async def connect(self):
        if websockets is None or msgpack is None:
            raise RuntimeError("websockets/msgpack-numpy dependencies are missing")
        self.websocket = await websockets.connect(
            self.uri,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        )
        self.metadata = msgpack.unpackb(await self.websocket.recv())
        return self.metadata

    async def infer(self, image: Any, prompt: str) -> Dict[str, Any]:
        if self.websocket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        elif not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        obs = {
            "image": image,
            "prompt": prompt,
        }
        await self.websocket.send(msgpack.packb(obs))
        return msgpack.unpackb(await self.websocket.recv())

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    def _start_background_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_loop(self):
        if self._loop is None or not self._loop.is_running():
            self._thread = threading.Thread(
                target=self._start_background_loop, daemon=True
            )
            self._thread.start()
            import time

            while self._loop is None:
                time.sleep(0.01)

    def _run_async(self, coro):
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def connect_sync(self):
        return self._run_async(self.connect())

    def infer_sync(self, image: Any, prompt: str) -> Dict[str, Any]:
        return self._run_async(self.infer(image, prompt))

    def close_sync(self):
        result = self._run_async(self.close())
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        return result
