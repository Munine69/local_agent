"""
VAD [Wake-word 엔진]

mermaid 노드: VAD
mermaid 엣지:
  - User --> STT --> VAD                  (음성 입력에서 Wake-word 감지)
  - VAD --> Wait_UX --> TTS --> Speaker    (즉시 응답 트리거)
  - VAD --> State1 --> STT                (대화 대기 루프)
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WAKE_WORDS = [
    "오디스야",
    "오디세이",
    "오딧세이",
    "오디스요",
    "오디스아",
    "오디스여",
    "오디스",
    "오디세",
    "오딧스",
    "오딧세",
    "오디즈",
    "오디쓰",
    "오디수",
    "오티스",
    "오티즈",
    "오티쓰",
    "오티세",
    "오티세이",
    "오지스",
    "오지즈",
    "오지쓰",
    "우디스",
    "우디즈",
    "오리스",
    "보리스",
    "보리쓰",
    "보디스",
    "보디즈",
    "어디스",
    "어딧스",
    "오디서",
    "저기",
    "얘야",
]

WAKE_WORD_ONLY_ALIASES = (
    "야",
    "오디",
    "여보세요",
    "들려",
    "잘들려",
    "듣고있어",
    "듣고있니",
    "내말들려",
    "어디서",
)

# Whisper가 자주 내는 오인식 패턴 (공백/띄어쓰기 제거 후 비교)
_WAKE_FUZZY_PATTERNS = (
    re.compile(r"[오어우]\s*디\s*[스즈쓰수sS]"),
    re.compile(r"오\s*딧\s*[스세]"),
    re.compile(r"오\s*티\s*[스즈쓰]"),
    re.compile(r"오\s*지\s*[스즈쓰]"),
    re.compile(r"보\s*리\s*[스쓰]"),
    re.compile(r"보\s*디\s*[스즈]"),
)


def detect_wake_in_text(text: str) -> str | None:
    """STT 전사문에서 웨이크워드를 찾는다. 없으면 None."""
    raw = (text or "").strip()
    if not raw:
        return None
    compact = _compact_wake_text(raw)
    if compact in WAKE_WORD_ONLY_ALIASES or _has_prefix_alias(raw):
        return "오디스"
    for wake in sorted(WAKE_WORDS, key=len, reverse=True):
        if wake in raw:
            return wake
    for wake in WAKE_WORDS:
        if wake in {"오디", "저기", "얘야"}:
            continue
        if wake in compact:
            return "오디스"
    for pattern in _WAKE_FUZZY_PATTERNS:
        if pattern.search(raw):
            return "오디스"
    return None


def strip_wake_from_text(text: str, wake_hit: str) -> str:
    """웨이크 구간을 제거한 나머지 발화."""
    raw = (text or "").strip()
    compact = _compact_wake_text(raw)
    if compact in WAKE_WORD_ONLY_ALIASES:
        return ""
    if _has_prefix_alias(raw):
        return re.sub(r"^\s*(야|오디|여보세요|들려|잘\s*들려|듣고\s*있어|듣고\s*있니|내\s*말\s*들려)[\s,.!?~·]+", "", raw, count=1).strip(" ,.!?~")
    if wake_hit in raw:
        return raw.replace(wake_hit, "", 1).strip(" ,.!?~")
    for pattern in _WAKE_FUZZY_PATTERNS:
        match = pattern.search(raw)
        if match:
            return (raw[: match.start()] + raw[match.end() :]).strip(" ,.!?~")
    for wake in sorted(WAKE_WORDS, key=len, reverse=True):
        if wake in raw:
            return raw.replace(wake, "", 1).strip(" ,.!?~")
    return raw.strip(" ,.!?~")


def _compact_wake_text(text: str) -> str:
    return re.sub(r"[\s,.!?~·]+", "", (text or "").strip().lower())


def _has_prefix_alias(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(야|오디|여보세요|들려|잘\s*들려|듣고\s*있어|듣고\s*있니|내\s*말\s*들려)[\s,.!?~·]+",
            text or "",
        )
    )


@dataclass
class WakeWordResult:
    detected: bool
    keyword: str
    confidence: float


class VAD(ABC):
    """Wake-word 엔진 추상 인터페이스.

    실제 구현은 Silero VAD, Porcupine, 또는 커스텀 KWS 모델로 대체한다.
    """

    @abstractmethod
    async def start(self) -> None:
        """Wake-word 감지를 시작한다."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Wake-word 감지를 중단한다."""
        ...

    @abstractmethod
    async def wait_for_wakeword(self) -> WakeWordResult:
        """Wake-word가 감지될 때까지 대기한다.

        mermaid: User --> STT --> VAD
        """
        ...


class StubVAD(VAD):
    """테스트/개발용 VAD 스텁.

    외부에서 이벤트 큐를 통해 Wake-word를 시뮬레이션한다.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[WakeWordResult] = asyncio.Queue()
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("StubVAD 시작")

    async def stop(self) -> None:
        self._running = False
        logger.info("StubVAD 중단")

    async def wait_for_wakeword(self) -> WakeWordResult:
        return await self._queue.get()

    async def simulate_wakeword(self, keyword: str = "오디스야") -> None:
        await self._queue.put(WakeWordResult(detected=True, keyword=keyword, confidence=1.0))


class PipelineVAD(VAD):
    """`AudioPipeline`이 발행한 wake-word 이벤트를 소비하는 VAD 어댑터.

    실제 마이크 캡처/세그먼테이션은 AudioPipeline이 수행하고,
    이 클래스는 큐에서 이벤트를 꺼내 인터페이스로 노출한다.
    """

    def __init__(self, pipeline: "object") -> None:  # AudioPipeline (forward ref)
        self._pipeline = pipeline
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("PipelineVAD 시작")

    async def stop(self) -> None:
        self._running = False
        logger.info("PipelineVAD 중단")

    async def wait_for_wakeword(self) -> WakeWordResult:
        return await self._pipeline.wake_queue.get()
