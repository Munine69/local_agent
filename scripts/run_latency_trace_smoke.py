"""
Latency trace smoke: one WS turn + stub TTS, writes JSONL.

Usage:
    cd local_agent
    PYTHONPATH=src python3 scripts/run_latency_trace_smoke.py
    PYTHONPATH=src python3 scripts/run_latency_trace_smoke.py --endpoint http://192.168.0.12:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cloud_server.chat_client import CloudChatClient, CloudChatConfig
from src.config_loader import load_config
from src.edge_node.tts import StubTTS, TTSPriority
from src.runtime.latency_trace import configure_latency_trace, get_latency_recorder

DEFAULT_QUERY = "안녕하세요. 오늘은 그냥 인사만 하고 싶어요."


async def run_smoke(
    *, endpoint: str, query: str, output: Path, speaker_id: str
) -> Path:
    configure_latency_trace({"enabled": True, "output_path": str(output)})
    rec = get_latency_recorder()

    cloud = CloudChatConfig(
        endpoint=endpoint.rstrip("/"),
        speaker_id=speaker_id,
        on_ws_event=rec.mark,
    )
    tts = StubTTS()

    rec.begin_turn("wake_dialogue", smoke=True)
    rec.mark("wake_detected", keyword="오디스")
    rec.mark("question_wait_start")
    rec.mark("utterance_end", audio_duration_sec=3.2)
    rec.mark("stt_start", audio_duration_sec=3.2)
    rec.mark("stt_end", audio_duration_sec=3.2, text_preview=query[:120])
    rec.mark("stt_question_received", text_preview=query[:120])
    rec.mark("cloud_dialogue_start", text_preview=query[:120])

    client = CloudChatClient(cloud)
    status = "ok"
    try:
        async for message in client.send_stt(query):
            msg_type = message.get("type", "")
            spoken = CloudChatClient.spoken_text(message)
            if msg_type == "filler" and spoken:
                rec.mark("tts_queue_filler", text_len=len(spoken))
                await tts.speak(spoken, TTSPriority.NORMAL, trace_role="filler")
            elif msg_type in {"response", "identity_check", "reminder"}:
                if spoken and message.get("requires_tts", True):
                    await tts.speak(spoken, TTSPriority.HIGH, trace_role="response")
            elif msg_type == "error":
                status = "cloud_error"
                break
    except Exception as exc:
        status = f"exception:{type(exc).__name__}"
        rec.mark("cloud_dialogue_error", error=str(exc)[:200])
    finally:
        rec.mark("cloud_dialogue_end", status=status)
        rec.end_turn(status=status, text_preview=query[:120], endpoint=endpoint)

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=None, help="ai-server base URL")
    parser.add_argument("--text", default=DEFAULT_QUERY)
    parser.add_argument("--speaker-id", default="latency_smoke_001")
    parser.add_argument(
        "--output",
        default="runtime/latency_trace_smoke.jsonl",
        help="JSONL output path (relative to cwd)",
    )
    args = parser.parse_args()

    cfg = load_config()
    endpoint = args.endpoint or str(
        cfg.get("cloud", {}).get("endpoint", "http://localhost:8000")
    ).rstrip("/")

    out = Path(args.output)
    if not out.is_absolute():
        out = Path.cwd() / out

    print(f"[smoke] endpoint={endpoint}")
    print(f"[smoke] output={out}")
    asyncio.run(
        run_smoke(
            endpoint=endpoint,
            query=args.text,
            output=out,
            speaker_id=args.speaker_id,
        )
    )

    if not out.exists():
        print("[smoke] FAIL: no output file")
        sys.exit(1)

    line = out.read_text(encoding="utf-8").strip().splitlines()[-1]
    row = json.loads(line)
    print(f"[smoke] turn_id={row['turn_id']} status={row['status']} total_ms={row['total_ms']}")
    print("[smoke] deltas_ms:", json.dumps(row.get("deltas_ms", {}), ensure_ascii=False))
    print("[smoke] last record:")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
