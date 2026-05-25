"""End-to-end latency timestamps for voice dialogue (JSONL append)."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_recorder: LatencyTraceRecorder | None = None


def configure_latency_trace(config: dict[str, Any] | None) -> LatencyTraceRecorder:
    """Create or replace the process-wide recorder from config."""
    global _recorder
    cfg = config or {}
    _recorder = LatencyTraceRecorder(
        enabled=bool(cfg.get("enabled", False)),
        output_path=str(cfg.get("output_path", "runtime/latency_trace.jsonl")),
    )
    return _recorder


def get_latency_recorder() -> LatencyTraceRecorder:
    if _recorder is None:
        return configure_latency_trace({"enabled": False})
    return _recorder


def _resolve_output_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# Named deltas (from_event -> to_event) for summary fields on each turn.
_DELTA_PAIRS: tuple[tuple[str, str], ...] = (
    ("utterance_end", "stt_end"),
    ("stt_end", "cloud_ws_send"),
    ("cloud_ws_send", "cloud_ws_recv_filler"),
    ("cloud_ws_recv_filler", "tts_play_start_filler"),
    ("cloud_ws_send", "cloud_ws_recv_response"),
    ("cloud_ws_recv_response", "tts_synthesize_start_response"),
    ("tts_synthesize_start_response", "tts_play_start_response"),
    ("tts_play_start_response", "tts_play_end_response"),
    ("wake_detected", "tts_play_end_wake"),
    ("tts_play_end_wake", "stt_end"),
    ("stt_end", "tts_play_start_response"),
)


class LatencyTraceRecorder:
    """Records monotonic/wall timestamps per dialogue turn; appends one JSONL line per turn."""

    def __init__(self, *, enabled: bool, output_path: str) -> None:
        self.enabled = enabled
        self._output_path = _resolve_output_path(output_path)
        self._lock = threading.Lock()
        self._turn_id: str | None = None
        self._phase: str = ""
        self._t0: float | None = None
        self._events: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = {}

    @property
    def output_path(self) -> Path:
        return self._output_path

    def begin_turn(self, phase: str, **meta: Any) -> str:
        """Start a new turn; returns turn_id."""
        if not self.enabled:
            return ""
        turn_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._turn_id = turn_id
            self._phase = phase
            self._t0 = time.perf_counter()
            self._events = []
            self._meta = dict(meta)
        self.mark("turn_start", phase=phase)
        return turn_id

    def mark(self, event: str, **meta: Any) -> None:
        if not self.enabled or self._t0 is None:
            return
        now = time.perf_counter()
        entry: dict[str, Any] = {
            "event": event,
            "t_wall": datetime.now(timezone.utc).isoformat(),
            "delta_ms": round((now - self._t0) * 1000, 1),
        }
        if meta:
            entry.update(meta)
        with self._lock:
            self._events.append(entry)

    def end_turn(self, status: str = "ok", **meta: Any) -> None:
        if not self.enabled or self._t0 is None:
            return
        self.mark("turn_end", status=status)
        ended_at = time.perf_counter()
        with self._lock:
            events = list(self._events)
            turn_id = self._turn_id or ""
            phase = self._phase
            turn_meta = dict(self._meta)
            t0 = self._t0
            self._turn_id = None
            self._phase = ""
            self._t0 = None
            self._events = []
            self._meta = {}

        record: dict[str, Any] = {
            "turn_id": turn_id,
            "phase": phase,
            "status": status,
            "started_at": events[0]["t_wall"] if events else "",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "total_ms": round((ended_at - t0) * 1000, 1),
            "events": events,
            "deltas_ms": _compute_deltas(events),
            **turn_meta,
            **meta,
        }
        self._append_jsonl(record)
        logger.info(
            "[LatencyTrace] turn_id=%s phase=%s status=%s total_ms=%.1f -> %s",
            turn_id,
            phase,
            status,
            record["total_ms"],
            self._output_path,
        )

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        try:
            with open(self._output_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            logger.exception("LatencyTrace JSONL write failed: %s", self._output_path)


def _compute_deltas(events: list[dict[str, Any]]) -> dict[str, float]:
    """First occurrence of each event name -> delta_ms."""
    by_event: dict[str, float] = {}
    for e in events:
        name = str(e.get("event", ""))
        if name and name not in by_event:
            by_event[name] = float(e.get("delta_ms", 0))

    out: dict[str, float] = {}
    for start, end in _DELTA_PAIRS:
        if start in by_event and end in by_event:
            key = f"{start}__{end}"
            out[key] = round(by_event[end] - by_event[start], 1)
    return out


def latency_hook(event: str, **meta: Any) -> None:
    get_latency_recorder().mark(event, **meta)
