"""
Wait_UX [**고정 멘트 엔진**]

mermaid 노드: Wait_UX
mermaid 엣지:
  - VAD --> Wait_UX --> TTS -- "즉시 응답 (ex: 네 어르신 말씀해주세요!)" --> Speaker
  - Wait_UX --> TTS -- "고정 멘트: '오디스가 생각하고 있어요. 잠시만 기다려 주세요~'" --> Speaker
  - OCR_Engine --실패 시 재요청--> Wait_UX --> TTS -- "어르신 약봉투를 다시 들어주시겠어요?" --> Speaker

멘트 유형:
  - IMMEDIATE_RESPONSE: 즉시 응답 멘트
  - WAITING: 처리 중 대기 멘트 (시간 경과에 따라 에스컬레이션)
  - RETRY_REQUEST: 재촬영/재확인 요청 멘트
  - QUALITY_GUIDE: 품질 미달 촬영 가이드 멘트
  - CONFIDENCE_CONFIRM: 신뢰도 미달 확인 요청 멘트
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.edge_node.capture_mode.buffer import QualityFailReason

logger = logging.getLogger(__name__)


class MentType(Enum):
    IMMEDIATE_RESPONSE = auto()
    WAITING = auto()
    RETRY_REQUEST = auto()
    QUALITY_GUIDE = auto()
    CONFIDENCE_CONFIRM = auto()
    COUNTDOWN = auto()


_TEMPLATES: dict[MentType, list[str]] = {
    MentType.IMMEDIATE_RESPONSE: [
        "네, 어르신! 말씀해 주세요!",
        "네, 어르신 말씀해주세요!",
        "네! 오디스가 듣고 있어요!",
    ],
    MentType.WAITING: [
        "오디스가 생각하고 있어요. 잠시만 기다려 주세요~",
        "잠시만요, 어르신. 확인하고 있어요!",
        "거의 다 됐어요, 어르신. 조금만 기다려 주세요~",
    ],
    MentType.RETRY_REQUEST: [
        "찍힌 사진에서 글씨를 정확히 인식하기 어려웠어요. 약봉투나 처방전을 다시 한 번 보여주세요.",
        "방금 사진의 인식률이 낮아서 다시 촬영이 필요해요. 글자가 잘 보이게 한 번만 더 보여주세요.",
        "촬영된 사진을 읽기 어려웠어요. 약봉투를 카메라 앞에 다시 들어주세요.",
    ],
    MentType.QUALITY_GUIDE: [],
    MentType.CONFIDENCE_CONFIRM: [],
    MentType.COUNTDOWN: [
        "하나",
        "둘",
        "셋",
        "찰칵!",
    ],
}

_QUALITY_GUIDE_TEMPLATES: dict[str, list[str]] = {
    "BLUR": [
        "어르신, 사진이 조금 흔들렸습니다. 카메라를 조금만 더 멀리 떨어뜨려 주시겠어요?",
        "어르신, 사진이 좀 흔들렸어요. 손을 가만히 잡고 계셔 보세요~",
    ],
    "TOO_DARK": [
        "어르신, 화면이 조금 어두워요. 불을 켜주시거나 약봉투를 조금 더 밝은 곳으로 옮겨주세요.",
        "어르신, 조금 어두운 것 같아요. 밝은 곳에서 다시 보여주시겠어요?",
    ],
    "TOO_BRIGHT": [
        "어르신, 빛이 좀 많이 반사돼요. 그늘진 곳에서 다시 보여주시겠어요?",
        "어르신, 너무 밝아서 글씨가 잘 안 보여요. 살짝 가려주시겠어요?",
    ],
    "GLARE": [
        "어르신, 빛 반사가 있어서 글씨가 잘 안 보여요. 약봉투 각도를 살짝 바꿔주시겠어요?",
        "어르신, 반사광이 좀 있어요. 각도를 조금만 바꿔주세요~",
    ],
}

_CONFIDENCE_CONFIRM_TEMPLATE = "어르신, 방금 보여주신 약 이름이 '{drug_name}'이 맞으실까요? 이름이 잘 안 보여서 다시 여쭤봐요."


class WaitUX:
    """고정 멘트 엔진.

    모든 출력은 TTS 큐를 통해 TTS --> Speaker 경로로 전달된다.
    """

    def __init__(
        self,
        escalation_intervals: list[float] | None = None,
        quality_guide_cooldown_sec: float = 5.0,
    ) -> None:
        self._escalation_intervals = escalation_intervals or [3.0, 7.0, 15.0]
        self._quality_guide_cooldown_sec = quality_guide_cooldown_sec
        self._last_quality_guide_at: dict[str, float] = {}
        self._tts_queue: asyncio.Queue[str] | None = None
        self._waiting_task: asyncio.Task[None] | None = None

    def set_tts_queue(self, queue: asyncio.Queue[str]) -> None:
        """Wait_UX --> TTS: TTS 출력 큐 연결."""
        self._tts_queue = queue

    @staticmethod
    def immediate_response_templates() -> tuple[str, ...]:
        return tuple(_TEMPLATES[MentType.IMMEDIATE_RESPONSE])

    async def immediate_response(self) -> None:
        """VAD --> Wait_UX --> TTS: 즉시 응답 멘트 출력."""
        await self._speak(MentType.IMMEDIATE_RESPONSE)

    async def start_waiting(self) -> None:
        """처리 지연 시 단계별 대기 멘트를 출력한다."""
        if self._waiting_task is not None and not self._waiting_task.done():
            return
        self._waiting_task = asyncio.create_task(self._waiting_loop())

    async def stop_waiting(self) -> None:
        """대기 멘트를 중단한다."""
        if self._waiting_task is not None and not self._waiting_task.done():
            self._waiting_task.cancel()
            try:
                await self._waiting_task
            except asyncio.CancelledError:
                pass
            self._waiting_task = None

    async def retry_request(self) -> None:
        """OCR_Engine --실패--> Wait_UX --> TTS: 재촬영 요청 멘트."""
        await self._speak(MentType.RETRY_REQUEST)

    async def quality_guide(self, reason: QualityFailReason | str) -> None:
        """Buffer 품질 미달 시 촬영 가이드 멘트.

        reason 은 QualityFailReason enum 이거나 그 name 문자열.
        """
        reason_key = reason.name if hasattr(reason, "name") else str(reason)
        now = time.monotonic()
        last_at = self._last_quality_guide_at.get(reason_key, 0.0)
        if now - last_at < self._quality_guide_cooldown_sec:
            logger.debug("품질 가이드 쿨다운 중: %s", reason_key)
            return
        templates = _QUALITY_GUIDE_TEMPLATES.get(reason_key, [])
        if templates:
            self._last_quality_guide_at[reason_key] = now
            text = random.choice(templates)
            await self._send_to_tts(text)
        else:
            logger.warning("품질 가이드 템플릿 없음: %s", reason_key)

    async def confidence_confirm(self, drug_name: str) -> None:
        """신뢰도 미달 시 사용자 확인 요청."""
        text = _CONFIDENCE_CONFIRM_TEMPLATE.format(drug_name=drug_name)
        await self._send_to_tts(text)

    async def _speak(self, ment_type: MentType) -> None:
        templates = _TEMPLATES.get(ment_type, [])
        if not templates:
            return
        text = random.choice(templates)
        await self._send_to_tts(text)

    async def _send_to_tts(self, text: str) -> None:
        logger.info("Wait_UX --> TTS: %s", text)
        if self._tts_queue is not None:
            await self._tts_queue.put(text)

    async def _waiting_loop(self) -> None:
        templates = _TEMPLATES[MentType.WAITING]
        for i, interval in enumerate(self._escalation_intervals):
            await asyncio.sleep(interval)
            idx = min(i, len(templates) - 1)
            await self._send_to_tts(templates[idx])
