"""
Timer [촬영 타이머: 하나 둘 셋 찰칵]

mermaid 노드: Timer
mermaid 엣지:
  - Buffer --> Timer             (버퍼로부터 프레임 공급)
  - Timer --> TTS                (카운트다운 음성 "하나, 둘, 셋, 찰칵!")
  - Timer --> OCR_Engine         (확정된 BestShot 프레임을 OCR에 전달)
  - Timer --> TTS --> Speaker    (촬영 실행 흐름의 일부)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.edge_node.capture_mode.buffer import Buffer, BufferedFrame

logger = logging.getLogger(__name__)

@dataclass
class CaptureResult:
    frame: np.ndarray
    timestamp: float
    quality_score: float


class Timer:
    """촬영 타이머 노드.

    카운트다운("하나, 둘, 셋, 찰칵!")을 진행하며,
    각 카운트를 TTS 콜백으로 전달하고,
    카운트 완료 시 Buffer에서 BestShot을 확정하여
    OCR_Engine에 전달할 CaptureResult를 생성한다.
    """

    def __init__(
        self,
        buffer: Buffer,
        timer_seconds: int = 3,
    ) -> None:
        self._buffer = buffer
        self._timer_seconds = timer_seconds
        self._tts_queue: asyncio.Queue[str] | None = None
        self._ocr_queue: asyncio.Queue[CaptureResult] | None = None

    def set_tts_queue(self, queue: asyncio.Queue[str]) -> None:
        """Timer --> TTS: 카운트다운 음성 출력 큐."""
        self._tts_queue = queue

    def set_ocr_queue(self, queue: asyncio.Queue[CaptureResult]) -> None:
        """Timer --> OCR_Engine: 확정 프레임 전달 큐."""
        self._ocr_queue = queue

    async def run_countdown(self) -> CaptureResult | None:
        """BestShot을 확정한다.

        Returns:
            CaptureResult: 확정된 프레임 + 메타데이터, 실패 시 None.
        """
        logger.info("BestShot 확정 시작")
        capture_started = time.time()
        bestshot = self._buffer.get_bestshot()
        if bestshot is None:
            logger.warning("BestShot 확보 실패: 버퍼 비어있음")
            return None

        result = CaptureResult(
            frame=bestshot.frame,
            timestamp=bestshot.timestamp,
            quality_score=bestshot.quality.composite_score,
        )

        if self._ocr_queue is not None:
            await self._ocr_queue.put(result)

        logger.info(
            "BestShot 확정: timestamp=%.3f, trigger=%.3f, delta=%.3fs, quality=%.3f",
            result.timestamp,
            capture_started,
            capture_started - result.timestamp,
            result.quality_score,
        )
        return result
