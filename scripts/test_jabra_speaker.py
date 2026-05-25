"""
Jabra SPEAK 510 USB 스피커 재생 테스트.

agent_config.yaml의 audio.output_device 설정을 그대로 사용해
LocalSpeaker로 1초짜리 사인파(440Hz)를 재생한다.

사용법:
    cd ~/local_agent
    python -m scripts.test_jabra_speaker
"""

from __future__ import annotations

import asyncio
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.home_environment.speaker import LocalSpeaker


def generate_sine_pcm(
    freq_hz: float = 440.0,
    duration_sec: float = 1.0,
    sample_rate: int = 22050,
    amplitude: float = 0.3,
) -> bytes:
    """16-bit signed PCM mono 사인파 데이터를 생성한다."""
    n_samples = int(sample_rate * duration_sec)
    max_amp = int(32767 * amplitude)
    samples = (
        int(max_amp * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        for i in range(n_samples)
    )
    return b"".join(struct.pack("<h", s) for s in samples)


async def main() -> None:
    cfg = load_config()

    audio_cfg = cfg.get("audio", {})
    device = audio_cfg.get("output_device", "default")
    sample_rate = audio_cfg.get("sample_rate", 22050)
    channels = audio_cfg.get("channels", 1)

    print(f"[test] device={device} rate={sample_rate} ch={channels}")
    print("[test] Jabra SPEAK 510에서 440Hz 비프음 1초 재생합니다...")

    speaker = LocalSpeaker(device=device, sample_rate=sample_rate, channels=channels)
    pcm = generate_sine_pcm(sample_rate=sample_rate)
    await speaker.play(pcm)

    print("[test] 완료. 소리가 들렸으면 Jabra 연결 OK.")


if __name__ == "__main__":
    asyncio.run(main())
