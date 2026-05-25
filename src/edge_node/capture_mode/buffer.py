"""
Buffer [RTSP 프레임 버퍼링]

mermaid 노드: Buffer
mermaid 엣지:
  - Cam --"RTSP 통신 지원"--> Capture_Mode  (Cam 프레임이 Buffer로 유입)
  - State3 --"촬영 실행"--> Buffer           (촬영 대기 -> 버퍼 활성화)
  - Buffer --> Timer                        (품질 통과 프레임을 Timer에 전달)

실시간 프리뷰 프레임 품질 평가 (블러/조도/글레어) 및 BestShot 후보 관리.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.home_environment.cam import Cam

logger = logging.getLogger(__name__)


class QualityFailReason(Enum):
    BLUR = auto()
    TOO_DARK = auto()
    TOO_BRIGHT = auto()
    GLARE = auto()


@dataclass
class QualityConfig:
    blur_threshold: float = 100.0
    brightness_min: int = 50
    brightness_max: int = 230
    glare_max_ratio: float = 0.15


@dataclass
class QualityReport:
    blur_score: float
    brightness_mean: float
    glare_ratio: float
    is_acceptable: bool
    fail_reasons: list[QualityFailReason] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """BestShot 선별에 사용할 복합 품질 점수 (높을수록 좋음)."""
        norm_blur = min(self.blur_score / 500.0, 1.0)
        norm_brightness = 1.0 - abs(self.brightness_mean - 140.0) / 140.0
        norm_glare = 1.0 - min(self.glare_ratio / 0.3, 1.0)
        return norm_blur * 0.5 + max(norm_brightness, 0.0) * 0.3 + norm_glare * 0.2


@dataclass
class BufferedFrame:
    frame: np.ndarray
    quality: QualityReport
    timestamp: float


class Buffer:
    """RTSP 프레임 버퍼링 노드.

    Cam으로부터 프레임을 수신하여 링 버퍼에 저장하면서
    각 프레임의 품질을 실시간 평가한다.
    품질 미달 시 콜백을 통해 Wait_UX에 피드백 신호를 전달한다.
    """

    def __init__(
        self,
        cam: Cam,
        quality_config: QualityConfig | None = None,
        buffer_size: int = 30,
        quality_fail_cooldown_sec: float = 5.0,
    ) -> None:
        self._cam = cam
        self._quality_config = quality_config or QualityConfig()
        self._buffer: deque[BufferedFrame] = deque(maxlen=buffer_size)
        self._active = False
        self._on_quality_fail: asyncio.Queue[QualityFailReason] | None = None
        self._quality_fail_cooldown_sec = quality_fail_cooldown_sec
        self._last_quality_fail_at: dict[QualityFailReason, float] = {}

    def set_quality_fail_queue(self, queue: asyncio.Queue[QualityFailReason]) -> None:
        self._on_quality_fail = queue

    def activate(self) -> None:
        """State3 --"촬영 실행"--> Buffer: 버퍼링 활성화."""
        self._active = True
        self._buffer.clear()
        self._last_quality_fail_at.clear()
        logger.info("Buffer 활성화: 프레임 수집 시작")

    def deactivate(self) -> None:
        self._active = False
        logger.info("Buffer 비활성화")

    @property
    def is_active(self) -> bool:
        return self._active

    async def capture_loop(self) -> None:
        """활성 상태에서 Cam 프레임을 지속적으로 버퍼링한다."""
        while self._active:
            try:
                frame = await self._cam.read_frame()
                report = self._assess_quality(frame)
                buffered = BufferedFrame(
                    frame=frame,
                    quality=report,
                    timestamp=time.time(),
                )
                self._buffer.append(buffered)

                if not report.is_acceptable and self._on_quality_fail is not None:
                    now = time.monotonic()
                    for reason in report.fail_reasons:
                        last_at = self._last_quality_fail_at.get(reason, 0.0)
                        if now - last_at >= self._quality_fail_cooldown_sec:
                            self._last_quality_fail_at[reason] = now
                            await self._on_quality_fail.put(reason)

            except IOError:
                logger.warning("프레임 읽기 실패, 재시도...")
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

    def get_bestshot(self) -> BufferedFrame | None:
        """Buffer --> Timer: 현재 버퍼에서 최고 품질 프레임을 반환한다."""
        if not self._buffer:
            return None
        return max(self._buffer, key=lambda bf: bf.quality.composite_score)

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    async def wait_until_ready(
        self,
        *,
        min_frames: int = 3,
        timeout_sec: float = 5.0,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if len(self._buffer) >= min_frames:
                return True
            await asyncio.sleep(0.1)
        return False

    def _assess_quality(self, frame: np.ndarray) -> QualityReport:
        """실시간 프레임 품질 평가 (블러 / 조도 / 글레어)."""
        cfg = self._quality_config
        fail_reasons: list[QualityFailReason] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < cfg.blur_threshold:
            fail_reasons.append(QualityFailReason.BLUR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        brightness_mean = float(np.mean(v_channel))
        if brightness_mean < cfg.brightness_min:
            fail_reasons.append(QualityFailReason.TOO_DARK)
        elif brightness_mean > cfg.brightness_max:
            fail_reasons.append(QualityFailReason.TOO_BRIGHT)

        total_pixels = v_channel.size
        glare_pixels = int(np.count_nonzero(v_channel > 250))
        glare_ratio = glare_pixels / total_pixels if total_pixels > 0 else 0.0
        if glare_ratio > cfg.glare_max_ratio:
            fail_reasons.append(QualityFailReason.GLARE)

        return QualityReport(
            blur_score=blur_score,
            brightness_mean=brightness_mean,
            glare_ratio=glare_ratio,
            is_acceptable=len(fail_reasons) == 0,
            fail_reasons=fail_reasons,
        )
