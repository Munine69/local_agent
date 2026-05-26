"""
TTS [대화형 TTS 모델]

mermaid 노드: TTS
mermaid 엣지:
  - Wait_UX --> TTS --> Speaker                                  (고정 멘트 출력)
  - Timer --> TTS --> Speaker                                    (카운트다운 음성)
  - Cloud_Server --> TTS --> Speaker                             (실시간 응답)
  - OCR_Engine --실패--> Wait_UX --> TTS -- "약봉투 다시..." --> Speaker
  - TTS -- "즉시 응답 (네 어르신 말씀해주세요!)" --> Speaker
  - TTS -- "에이전트: 어르신, XX약은 녹용이랑 드시면 안되요~" --> Speaker

설계 노트:
  Qwen3-TTS는 0.9B BF16 + flash_attention_2 요구 + ~2GB 메모리로
  Jetson Orin Nano 8GB에서는 실용성이 낮다. 동일한 한국어 여성 음색을
  내기 위해 gTTS(Google Text-to-Speech)를 기본 RealTTS로 사용한다.
  추후 로컬 모델(MeloTTS / Piper 호환 ko 모델 등) 교체가 쉽도록
  Speaker 인터페이스에 PCM bytes만 전달한다.
"""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TTSPriority(IntEnum):
    """TTS 출력 우선순위 (낮을수록 높은 우선순위)."""
    URGENT = 0       # 실패/재요청
    HIGH = 1         # 즉시 응답, 카운트다운
    NORMAL = 2       # 에이전트 응답
    LOW = 3          # 대기 멘트


@dataclass
class TTSRequest:
    text: str
    priority: TTSPriority = TTSPriority.NORMAL


class TTS(ABC):
    """대화형 TTS 추상 인터페이스 (Qwen3-TTS 기반).

    실제 구현은 Qwen3-TTS 로컬 모델 또는 API로 대체한다.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """텍스트를 음성 데이터(PCM/WAV bytes)로 변환한다."""
        ...

    @abstractmethod
    async def speak(self, text: str, priority: TTSPriority = TTSPriority.NORMAL) -> None:
        """텍스트를 음성으로 변환하고 Speaker로 출력한다.

        mermaid: TTS --> Speaker
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """현재 TTS 출력을 중단한다."""
        ...


class StubTTS(TTS):
    """테스트/개발용 TTS 스텁.

    실제 음성 합성 없이 텍스트 로깅만 수행한다.
    """

    def __init__(self) -> None:
        self._output_log: list[TTSRequest] = []

    async def synthesize(self, text: str) -> bytes:
        logger.info("StubTTS synthesize: %s", text)
        return b""

    async def speak(
        self,
        text: str,
        priority: TTSPriority = TTSPriority.NORMAL,
        *,
        trace_role: str = "generic",
    ) -> None:
        request = TTSRequest(text=text, priority=priority)
        self._output_log.append(request)
        logger.info("StubTTS speak [%s] role=%s: %s", priority.name, trace_role, text)

    async def stop(self) -> None:
        logger.info("StubTTS 중단")

    @property
    def output_log(self) -> list[TTSRequest]:
        return self._output_log


class GTTSEngine(TTS):
    """gTTS(Google Text-to-Speech) 기반 실제 TTS 구현.

    - text -> mp3(in-memory) -> sox로 PCM(S16_LE) 디코딩 -> Speaker.play()
    - sox에 mp3 디코더(libsox-fmt-mp3)가 설치되어 있어야 한다.
    - Speaker는 LocalSpeaker(또는 동일 인터페이스) 인스턴스를 주입.
    """

    def __init__(
        self,
        speaker: Any,
        lang: str = "ko",
        tld: str = "co.kr",
        slow: bool = False,
        sample_rate: int = 22050,
        channels: int = 1,
        on_latency_event: Callable[..., None] | None = None,
    ) -> None:
        self._speaker = speaker
        self._lang = lang
        self._tld = tld
        self._slow = slow
        self._sample_rate = sample_rate
        self._channels = channels
        self._stop_flag = False
        self._cancel_version = 0
        self._on_latency_event = on_latency_event
        self._pcm_cache: dict[str, bytes] = {}

    async def synthesize(self, text: str) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    async def prepare_texts(self, texts: list[str] | tuple[str, ...]) -> None:
        """Pre-synthesize fixed prompts so first wake acknowledgement is instant."""
        for text in texts:
            if not text or text in self._pcm_cache:
                continue
            try:
                await self.synthesize(text)
                logger.info("GTTSEngine cache warmed: %s", text[:80])
            except Exception:
                logger.exception("GTTSEngine cache warm failed: %s", text[:80])

    def _synthesize_sync(self, text: str) -> bytes:
        cached = self._pcm_cache.get(text)
        if cached is not None:
            return cached

        from gtts import gTTS  # lazy

        buf = io.BytesIO()
        tts = gTTS(text=text, lang=self._lang, tld=self._tld, slow=self._slow)
        tts.write_to_fp(buf)
        mp3_bytes = buf.getvalue()

        proc = subprocess.run(
            [
                "sox",
                "-t", "mp3", "-",
                "-t", "raw",
                "-r", str(self._sample_rate),
                "-c", str(self._channels),
                "-e", "signed-integer",
                "-b", "16",
                "-",
            ],
            input=mp3_bytes,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.error(
                "sox mp3->pcm 디코딩 실패 (rc=%d): %s",
                proc.returncode,
                proc.stderr.decode(errors="replace")[:300],
            )
            return b""
        self._pcm_cache[text] = proc.stdout
        return proc.stdout

    async def speak(
        self,
        text: str,
        priority: TTSPriority = TTSPriority.NORMAL,
        *,
        trace_role: str = "generic",
    ) -> None:
        if not text:
            return
        self._stop_flag = False
        cancel_version = self._cancel_version
        logger.info("GTTSEngine speak [%s]: %s", priority.name, text)
        hook = self._on_latency_event
        if hook is not None:
            hook(
                f"tts_synthesize_start_{trace_role}",
                text_len=len(text),
                priority=priority.name,
            )
        try:
            pcm = await self.synthesize(text)
        except Exception:
            logger.exception("TTS 합성 실패")
            return
        if not pcm:
            logger.error("TTS 합성 결과 PCM이 비어 있음: %s", text[:120])
            return
        if self._stop_flag or cancel_version != self._cancel_version:
            logger.info("TTS 재생 전 중단됨: %s", text[:120])
            return
        logger.info("GTTSEngine PCM 생성 완료: %d bytes role=%s", len(pcm), trace_role)
        if hook is not None:
            hook(
                f"tts_synthesize_end_{trace_role}",
                pcm_bytes=len(pcm),
            )
        try:
            if hook is not None:
                hook(f"tts_play_start_{trace_role}", pcm_bytes=len(pcm))
            await self._speaker.play(pcm)
            if hook is not None:
                hook(f"tts_play_end_{trace_role}")
            logger.info("GTTSEngine 재생 완료 role=%s", trace_role)
        except Exception:
            logger.exception("Speaker 재생 실패")

    async def stop(self) -> None:
        self._stop_flag = True
        self._cancel_version += 1
        try:
            await self._speaker.stop()
        except Exception:
            pass
