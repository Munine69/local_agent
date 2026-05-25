"""
서버(WebSocket) 메시지 → TTS → Jabra 스피커 E2E 검증.

ai-server가 떠 있어야 하며, agent_config의 cloud.endpoint / audio 설정을 사용한다.
실제 마이크/STT 없이 고정 질의를 WS로 보내고 filler/response를 스피커로 재생한다.

사용법:
    cd ~/local_agent
    source .env   # ODISS_CLOUD_URL 등
    PYTHONPATH=src python3 -m scripts.test_server_to_speaker
    PYTHONPATH=src python3 -m scripts.test_server_to_speaker --text "심장병약이랑 같이 먹어도 되나요?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cloud_server.chat_client import CloudChatClient, CloudChatConfig
from src.config_loader import load_config
from src.edge_node.tts import GTTSEngine, StubTTS, TTSPriority
from src.home_environment.speaker import LocalSpeaker

logger = logging.getLogger(__name__)

DEFAULT_QUERY = "심장병약이랑 같이 먹어도 되나요?"


async def play_server_messages(
    *,
    query: str,
    cloud: CloudChatConfig,
    tts: GTTSEngine | StubTTS,
) -> dict[str, float | int | str]:
    client = CloudChatClient(cloud)
    played = 0
    last_spoken = ""
    t0 = time.perf_counter()

    print(f"[e2e] WS URL: {client.ws_url}")
    print(f"[e2e] speaker_id={cloud.speaker_id}")
    print(f"[e2e] query={query!r}")

    async for message in client.send_stt(query):
        msg_type = message.get("type", "")
        spoken = CloudChatClient.spoken_text(message)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[e2e] +{elapsed:0.0f}ms type={msg_type} text={spoken[:120]!r}")

        if msg_type == "filler" and spoken:
            await tts.speak(spoken, TTSPriority.NORMAL)
            played += 1
        elif msg_type in {"response", "identity_check", "reminder", "ocr_processed"}:
            if spoken and message.get("requires_tts", True):
                await tts.speak(spoken, TTSPriority.HIGH)
                last_spoken = spoken
                played += 1
        elif msg_type == "error":
            err = message.get("message", "unknown")
            raise RuntimeError(f"server error: {err}")

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "played_count": played,
        "last_spoken": last_spoken,
        "total_ms": round(total_ms, 1),
    }


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_QUERY)
    parser.add_argument("--stub-tts", action="store_true", help="스피커 없이 TTS 로그만")
    args = parser.parse_args()

    cfg = load_config()
    cloud_cfg = cfg.get("cloud", {})
    endpoint = str(cloud_cfg.get("endpoint", "http://localhost:8000")).rstrip("/")
    cloud = CloudChatConfig(
        endpoint=endpoint,
        websocket_path=cloud_cfg.get("websocket_path", "/ws/chat"),
        speaker_id=str(cloud_cfg.get("speaker_id", "jetson_live")),
        timeout_sec=float(cloud_cfg.get("websocket_timeout_sec", 120.0)),
    )

    audio_cfg = cfg.get("audio", {})
    tts_cfg = cfg.get("tts", {})
    if args.stub_tts:
        tts: GTTSEngine | StubTTS = StubTTS()
    else:
        speaker = LocalSpeaker(
            device=audio_cfg.get("output_device", "default"),
            sample_rate=tts_cfg.get("sample_rate", audio_cfg.get("sample_rate", 22050)),
            channels=tts_cfg.get("channels", audio_cfg.get("channels", 1)),
        )
        tts = GTTSEngine(
            speaker=speaker,
            lang=tts_cfg.get("lang", "ko"),
            tld=tts_cfg.get("tld", "co.kr"),
            slow=tts_cfg.get("slow", False),
            sample_rate=tts_cfg.get("sample_rate", 22050),
            channels=tts_cfg.get("channels", 1),
        )

    summary = await play_server_messages(query=args.text, cloud=cloud, tts=tts)
    print(
        f"[e2e] OK played={summary['played_count']} total_ms={summary['total_ms']} "
        f"last={str(summary['last_spoken'])[:80]!r}"
    )
    if summary["played_count"] == 0:
        raise SystemExit("재생된 음성이 없습니다. WS 응답 또는 TTS 경로를 확인하세요.")


if __name__ == "__main__":
    asyncio.run(main())
