"""Verify the configured RTSP camera can open and capture a real frame.

Usage:
    python -m scripts.verify_rtsp_capture
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2

from src.config_loader import load_config
from src.home_environment.cam import Cam, CamConfig


async def main() -> None:
    cfg = load_config()
    rtsp_cfg = cfg.get("rtsp", {})
    output_path = Path("runtime/rtsp_probe/latest_frame.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cam = Cam(
        CamConfig(
            url=rtsp_cfg.get("url", ""),
            reconnect_backoff_sec=rtsp_cfg.get("reconnect_backoff_sec", 2.0),
            max_reconnect_attempts=rtsp_cfg.get("max_reconnect_attempts", 1),
        )
    )
    try:
        await cam.open()
        frame = await cam.read_frame()
        ok = cv2.imwrite(str(output_path), frame)
        if not ok:
            raise RuntimeError(f"프레임 저장 실패: {output_path}")
        h, w = frame.shape[:2]
        print(f"RTSP capture OK: {w}x{h} -> {output_path}")
    finally:
        await cam.close()


if __name__ == "__main__":
    asyncio.run(main())
