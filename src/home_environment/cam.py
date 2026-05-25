"""
Cam [홈캠 RTSP 스트리밍]

mermaid 노드: Cam
mermaid 엣지:
  - Cam --"RTSP 통신 지원"--> Capture_Mode
  - Speaker <--> User <--> Cam
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import quote, urlparse, urlunparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GSTREAMER_PIPELINE_TEMPLATE = (
    "rtspsrc location=\"{url}\" latency=100 ! "
    "rtph264depay ! h264parse ! "
    "nvv4l2decoder ! nvvidconv ! "
    "video/x-raw,format=BGRx ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 sync=0"
)


@dataclass
class CamConfig:
    url: str = "rtsp://192.168.0.100:554/stream"
    reconnect_backoff_sec: float = 2.0
    max_reconnect_attempts: int = 10


class Cam:
    """홈캠 RTSP 스트리밍 노드.

    Jetson GStreamer HW 디코더를 우선 사용하고,
    불가능할 경우 OpenCV 기본 백엔드로 폴백한다.
    """

    def __init__(self, config: CamConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._open_sync)

    def _open_sync(self) -> None:
        url = (self._config.url or "").strip()
        if not url:
            raise ConnectionError("RTSP URL이 비어 있습니다. .env의 RTSP_URL을 설정하세요.")
        if urlparse(url).scheme != "rtsp":
            raise ConnectionError(f"RTSP URL 형식이 아닙니다: {url}")

        candidates = _rtsp_url_candidates(url)
        pipeline = GSTREAMER_PIPELINE_TEMPLATE.format(url=candidates[0])
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logger.warning("GStreamer 파이프라인 실패, OpenCV 기본 백엔드로 폴백")
            cap.release()
            for candidate in candidates:
                cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    url = candidate
                    break
                cap.release()
        if not cap.isOpened():
            raise ConnectionError(f"RTSP 연결 실패: {url}")
        self._cap = cap
        self._running = True
        logger.info("Cam 연결 성공: %s", url)

    async def read_frame(self) -> np.ndarray:
        """단일 프레임을 비동기로 읽어 반환한다."""
        if self._cap is None or not self._running:
            raise RuntimeError("Cam이 열려 있지 않습니다. open()을 먼저 호출하세요.")
        loop = asyncio.get_running_loop()
        ret, frame = await loop.run_in_executor(None, self._cap.read)
        if not ret or frame is None:
            raise IOError("프레임 읽기 실패")
        return frame

    async def reconnect(self) -> None:
        """지수 백오프를 적용한 재연결."""
        backoff = self._config.reconnect_backoff_sec
        for attempt in range(1, self._config.max_reconnect_attempts + 1):
            logger.info("Cam 재연결 시도 %d/%d ...", attempt, self._config.max_reconnect_attempts)
            try:
                self.close_sync()
                await self.open()
                return
            except ConnectionError:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        raise ConnectionError("최대 재연결 시도 초과")

    def close_sync(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._running = False

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close_sync)

    @property
    def is_opened(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()


def _rtsp_url_candidates(url: str) -> list[str]:
    parsed = urlparse(url)
    if not parsed.username:
        return [url]

    username = quote(parsed.username, safe="")
    password = quote(parsed.password or "", safe="")
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    encoded = urlunparse(
        (
            parsed.scheme,
            f"{username}:{password}@{host}",
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )
    return [url] if encoded == url else [url, encoded]
