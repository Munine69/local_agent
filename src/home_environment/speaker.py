"""
Speaker [홈 스피커 / 캠 내장 스피커]

mermaid 노드: Speaker
mermaid 엣지:
  - TTS --> Speaker
  - Speaker <--> User
"""

from __future__ import annotations

import asyncio
import logging
import math
import wave
from pathlib import Path
import struct
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

DEFAULT_CAPTURE_SHUTTER_PATH = Path("runtime/audio/camera_shutter.wav")


class Speaker(ABC):
    """홈 스피커 오디오 출력 추상 인터페이스."""

    @abstractmethod
    async def play(self, audio_data: bytes) -> None:
        """TTS로부터 받은 오디오 데이터를 스피커로 출력한다."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """현재 재생 중인 오디오를 중단한다."""
        ...

    async def play_capture_shutter(self) -> None:
        """촬영 완료를 알려주는 짧은 비프/찰칵 효과음을 재생한다."""
        sample_rate = 22050
        audio = _make_capture_shutter_pcm(sample_rate=sample_rate)
        await self.play(audio)


def _tone_pcm(
    *,
    frequency: float,
    duration_sec: float,
    sample_rate: int,
    channels: int,
    volume: float,
) -> bytes:
    frames = int(duration_sec * sample_rate)
    pcm = bytearray()
    for i in range(frames):
        t = i / sample_rate
        env = min(1.0, i / max(1, int(sample_rate * 0.01)))
        env *= min(1.0, (frames - i) / max(1, int(sample_rate * 0.02)))
        value = int(32767 * volume * env * math.sin(2 * math.pi * frequency * t))
        frame = struct.pack("<h", value)
        pcm.extend(frame * channels)
    return bytes(pcm)


def _silence_pcm(duration_sec: float, sample_rate: int, channels: int) -> bytes:
    return b"\x00\x00" * int(duration_sec * sample_rate) * channels


def _make_capture_shutter_pcm(sample_rate: int = 22050, channels: int = 1) -> bytes:
    """Two quick beeps followed by a crisp shutter-like click."""
    return b"".join(
        [
            _tone_pcm(
                frequency=880,
                duration_sec=0.08,
                sample_rate=sample_rate,
                channels=channels,
                volume=0.35,
            ),
            _silence_pcm(0.04, sample_rate, channels),
            _tone_pcm(
                frequency=1320,
                duration_sec=0.07,
                sample_rate=sample_rate,
                channels=channels,
                volume=0.32,
            ),
            _silence_pcm(0.05, sample_rate, channels),
            _tone_pcm(
                frequency=2200,
                duration_sec=0.035,
                sample_rate=sample_rate,
                channels=channels,
                volume=0.45,
            ),
            _silence_pcm(0.018, sample_rate, channels),
            _tone_pcm(
                frequency=1500,
                duration_sec=0.045,
                sample_rate=sample_rate,
                channels=channels,
                volume=0.38,
            ),
        ]
    )


class LocalSpeaker(Speaker):
    """로컬 오디오 디바이스를 통한 스피커 구현.

    Jetson 환경에서 ALSA/PulseAudio 디바이스로 출력한다.
    실제 오디오 출력은 aplay 또는 pyaudio를 사용할 수 있으며,
    여기서는 subprocess 기반으로 구현한다.

    Jabra SPEAK 510 USB 같은 USB 오디오를 쓰려면 device를
    `plughw:CARD=USB,DEV=0` 처럼 ALSA 식별자로 지정한다.
    """

    def __init__(
        self,
        device: str = "default",
        sample_rate: int = 22050,
        channels: int = 1,
        capture_shutter_path: str | None = None,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._capture_shutter_path = Path(
            capture_shutter_path or DEFAULT_CAPTURE_SHUTTER_PATH
        )
        self._process: asyncio.subprocess.Process | None = None
        self._play_lock = asyncio.Lock()

    async def play_capture_shutter(self) -> None:
        path = self._ensure_capture_shutter_file()
        async with self._play_lock:
            logger.info(
                "스피커 실제 셔터음 파일 재생: path=%s device=%s",
                path,
                self._device,
            )
            self._process = await asyncio.create_subprocess_exec(
                "aplay",
                "-D",
                self._device,
                str(path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await self._process.communicate()
            if self._process.returncode != 0:
                err = stderr.decode(errors="replace")[:500] if stderr else ""
                raise RuntimeError(
                    f"aplay shutter failed rc={self._process.returncode}: {err}"
                )
            self._process = None

    def _ensure_capture_shutter_file(self) -> Path:
        if self._capture_shutter_path.exists():
            return self._capture_shutter_path
        self._capture_shutter_path.parent.mkdir(parents=True, exist_ok=True)
        audio = _make_capture_shutter_pcm(
            sample_rate=self._sample_rate,
            channels=self._channels,
        )
        with wave.open(str(self._capture_shutter_path), "wb") as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(audio)
        logger.info("기본 셔터음 WAV 생성: %s", self._capture_shutter_path)
        return self._capture_shutter_path

    async def play(self, audio_data: bytes) -> None:
        if not audio_data:
            logger.warning("빈 오디오 데이터 재생 요청 무시")
            return
        async with self._play_lock:
            self._process = await asyncio.create_subprocess_exec(
                "aplay",
                "-D", self._device,
                "-f", "S16_LE",
                "-r", str(self._sample_rate),
                "-c", str(self._channels),
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            assert self._process.stdin is not None
            self._process.stdin.write(audio_data)
            self._process.stdin.close()
            _, stderr = await self._process.communicate()
            if self._process.returncode != 0:
                err = stderr.decode(errors="replace")[:500] if stderr else ""
                raise RuntimeError(
                    f"aplay failed rc={self._process.returncode} device={self._device}: {err}"
                )
            logger.info(
                "오디오 재생 완료 (%d bytes, device=%s rate=%d channels=%d)",
                len(audio_data),
                self._device,
                self._sample_rate,
                self._channels,
            )
            self._process = None

    async def stop(self) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
            logger.debug("오디오 재생 중단")
        if self._process is process:
            self._process = None
