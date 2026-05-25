"""WebSocket client for ODISS /ws/chat (STT in, filler/response out)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


OnWsEvent = Callable[..., None]


@dataclass
class CloudChatConfig:
    endpoint: str = "http://localhost:8000"
    websocket_path: str = "/ws/chat"
    speaker_id: str = "jetson_live"
    timeout_sec: float = 120.0
    on_ws_event: OnWsEvent | None = None


def _http_to_ws_url(endpoint: str, websocket_path: str) -> str:
    parsed = urlparse(endpoint.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.netloc or parsed.path
    path = websocket_path if websocket_path.startswith("/") else f"/{websocket_path}"
    return f"{scheme}://{host}{path}"


class CloudChatClient:
    """Send stt_result to ai-server and stream server messages until final response."""

    def __init__(self, config: CloudChatConfig | None = None) -> None:
        self._config = config or CloudChatConfig()

    @property
    def ws_url(self) -> str:
        return _http_to_ws_url(self._config.endpoint, self._config.websocket_path)

    async def send_stt(self, text: str) -> AsyncIterator[dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_sec)
        payload = {
            "type": "stt_result",
            "text": text,
            "speaker_id": self._config.speaker_id,
        }
        on_event = self._config.on_ws_event
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(self.ws_url, heartbeat=30) as ws:
                await ws.send_json(payload)
                if on_event is not None:
                    on_event(
                        "cloud_ws_send",
                        text_len=len(text),
                        speaker_id=self._config.speaker_id,
                    )
                while True:
                    msg = await ws.receive(timeout=self._config.timeout_sec)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        if on_event is not None:
                            msg_type = str(data.get("type", "unknown"))
                            on_event(
                                f"cloud_ws_recv_{msg_type}",
                                type=msg_type,
                                text_len=len(self.spoken_text(data)),
                            )
                        yield data
                        if data.get("type") in {
                            "response",
                            "error",
                            "identity_check",
                        }:
                            break
                    elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                        logger.warning("CloudChat WebSocket closed: %s", msg)
                        yield {"type": "error", "message": "WebSocket closed"}
                        break

    @staticmethod
    def spoken_text(message: dict[str, Any]) -> str:
        return (
            message.get("response_text")
            or message.get("text")
            or message.get("message")
            or ""
        ).strip()
