"""
오디오 파이프라인 - mermaid의 STT/VAD 노드를 실제 하드웨어로 구현.

흐름:
    Jabra MIC --(arecord 16kHz mono PCM16)--> WebRTC VAD (utterance segmentation)
        --> faster-whisper STT --> (wake_word_queue | transcription_queue)

mermaid 매핑:
    User --> STT --> VAD     : utterance 추출
    User --"약 가져왔어"--> STT : wake-word 이후의 transcription
    STT --> Instruction_Log  : transcription 큐 소비자가 처리
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from collections import deque
from dataclasses import dataclass, field

import webrtcvad

from src.edge_node.stt import TranscriptionResult
from src.edge_node.stt_sanitize import (
    is_prompt_echo_transcript,
    is_short_noise_hallucination,
    is_tts_echo_transcript,
    normalize_transcript,
)
from src.edge_node.vad import WakeWordResult, detect_wake_in_text, strip_wake_from_text
from src.runtime.latency_trace import get_latency_recorder

logger = logging.getLogger(__name__)


@dataclass
class AudioPipelineConfig:
    """오디오 파이프라인 설정."""

    input_device: str = "plughw:CARD=USB,DEV=0"
    sample_rate: int = 16000  # webrtcvad/whisper 표준
    frame_ms: int = 30  # webrtcvad: 10/20/30ms 만 허용
    vad_aggressiveness: int = 2  # 0~3 (높을수록 민감)
    silence_tail_ms: int = 700  # 발화 종료 판정 무음 길이
    min_utterance_ms: int = 300  # 너무 짧은 잡음 무시
    max_utterance_ms: int = 12000  # utterance 최대 길이

    whisper_backend: str = "faster_whisper"  # faster_whisper | openai_whisper | gemini_stt
    whisper_model: str = "small"  # tiny/base/small/medium/large-v3
    whisper_compute_type: str = "int8"  # CPU/Jetson 권장
    whisper_device: str = "cpu"  # cpu | cuda
    gemini_stt_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_stt_prompt: str = (
        "이 오디오는 한국어 음성입니다. 들리는 발화만 그대로 전사하세요. "
        "추측하거나 문장을 보완하지 마세요. 음성이 없거나 불명확하면 빈 문자열만 반환하세요."
    )
    language: str = "ko"
    initial_prompt: str = "오디스 오디스야 김영수 약"
    wake_focus_prompt: str = "오디스 오디스야"
    wake_focus_scan_enabled: bool = True
    wake_focus_scan_min_duration_sec: float = 2.5
    wake_focus_tail_sec: float = 2.8
    wake_focus_head_sec: float = 2.5
    wake_rolling_scan_enabled: bool = True
    wake_rolling_scan_interval_ms: int = 2000
    wake_rolling_scan_min_speech_ms: int = 2500
    wake_fast_path_enabled: bool = True
    wake_fast_path_max_duration_sec: float = 1.8
    wake_stt_backend: str = "same"  # same | faster_whisper | openai_whisper | gemini_stt
    wake_whisper_model: str = "tiny"
    wake_whisper_compute_type: str = "int8"
    wake_whisper_device: str = "cpu"
    discard_non_wake_background: bool = True


@dataclass
class Utterance:
    """VAD가 분리한 한 발화 단위 PCM."""

    pcm_bytes: bytes
    sample_rate: int
    started_at: float
    duration_sec: float = field(default=0.0)


class MicrophoneCapture:
    """`arecord` subprocess 기반 마이크 캡처.

    Jabra SPEAK 510 같은 USB 오디오를 ALSA로 직접 읽는다.
    """

    def __init__(
        self,
        device: str,
        sample_rate: int,
        frame_bytes: int,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._frame_bytes = frame_bytes
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            "arecord",
            "-q",
            "-D", self._device,
            "-f", "S16_LE",
            "-r", str(self._sample_rate),
            "-c", "1",
            "-t", "raw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("MIC 캡처 시작: device=%s rate=%d", self._device, self._sample_rate)

    async def read_frame(self) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("MicrophoneCapture가 시작되지 않음")
        buf = await self._proc.stdout.readexactly(self._frame_bytes)
        return buf

    async def stop(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
            logger.info("MIC 캡처 종료")
        self._proc = None


class AudioPipeline:
    """마이크 -> VAD -> STT 파이프라인.

    한 번 시작하면 단일 백그라운드 태스크에서 utterance를 계속 추출하고,
    Whisper로 전사한 결과를 큐로 발행한다.

    - wake-word(WAKE_WORDS)가 포함되면 wake_queue에 push
    - 포함되지 않으면 transcription_queue에 push
    - 한 발화에 wake-word + 명령이 모두 있으면 둘 다 push
    """

    def __init__(self, config: AudioPipelineConfig | None = None) -> None:
        self._cfg = config or AudioPipelineConfig()
        bytes_per_sample = 2  # S16_LE
        self._frame_bytes = (
            self._cfg.sample_rate * self._cfg.frame_ms // 1000
        ) * bytes_per_sample

        self._mic = MicrophoneCapture(
            device=self._cfg.input_device,
            sample_rate=self._cfg.sample_rate,
            frame_bytes=self._frame_bytes,
        )
        self._vad = webrtcvad.Vad(self._cfg.vad_aggressiveness)
        self._whisper = None  # lazy load
        self._wake_whisper = None
        self._wake_backend = self._cfg.whisper_backend

        self.wake_queue: asyncio.Queue[WakeWordResult] = asyncio.Queue()
        self.transcription_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()

        self._task: asyncio.Task | None = None
        self._running = False
        self._utterance_wake_triggered = False
        self._rolling_scan_task: asyncio.Task | None = None
        self._dialogue_capture_active = False
        self._dialogue_capture_grace_until = 0.0
        self._wake_suppressed = False
        self._recent_tts_texts: deque[tuple[float, str]] = deque(maxlen=8)

    async def start(self) -> None:
        if self._running:
            return
        self._load_whisper()
        await self._mic.start()
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="audio_pipeline")
        logger.info("AudioPipeline 시작")

    def _load_whisper(self) -> None:
        if self._whisper is not None:
            return
        if self._cfg.whisper_backend == "gemini_stt":
            api_key = self._cfg.gemini_api_key or os.environ.get(
                self._cfg.gemini_api_key_env,
                "",
            )
            if not api_key.strip():
                raise RuntimeError(
                    f"gemini_stt requires {self._cfg.gemini_api_key_env}"
                )
            from google import genai

            self._whisper = genai.Client(api_key=api_key.strip())
            logger.info("Gemini STT 로드 완료: model=%s", self._cfg.gemini_stt_model)
            self._load_wake_model()
            return
        if self._cfg.whisper_backend == "openai_whisper":
            logger.info(
                "openai-whisper 로드: model=%s device=%s",
                self._cfg.whisper_model,
                self._cfg.whisper_device,
            )
            import torch
            import whisper

            if self._cfg.whisper_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("whisper_device=cuda 이지만 torch CUDA를 사용할 수 없습니다")
            self._whisper = whisper.load_model(
                self._cfg.whisper_model,
                device=self._cfg.whisper_device,
            )
            logger.info("openai-whisper 로드 완료")
            self._load_wake_model()
            return

        logger.info(
            "faster-whisper 로드: model=%s device=%s compute=%s",
            self._cfg.whisper_model,
            self._cfg.whisper_device,
            self._cfg.whisper_compute_type,
        )
        from faster_whisper import WhisperModel  # lazy

        self._whisper = WhisperModel(
            self._cfg.whisper_model,
            device=self._cfg.whisper_device,
            compute_type=self._cfg.whisper_compute_type,
        )
        logger.info("faster-whisper 로드 완료")
        self._load_wake_model()

    def _load_wake_model(self) -> None:
        backend = (self._cfg.wake_stt_backend or "same").strip()
        if backend == "same":
            self._wake_backend = self._cfg.whisper_backend
            self._wake_whisper = self._whisper
            return
        try:
            if backend == "gemini_stt":
                self._wake_backend = "gemini_stt"
                self._wake_whisper = self._whisper
                return
            if backend == "openai_whisper":
                import torch
                import whisper

                if self._cfg.wake_whisper_device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("wake_whisper_device=cuda 이지만 torch CUDA를 사용할 수 없습니다")
                self._wake_whisper = whisper.load_model(
                    self._cfg.wake_whisper_model,
                    device=self._cfg.wake_whisper_device,
                )
                self._wake_backend = "openai_whisper"
                logger.info(
                    "wake 전용 openai-whisper 로드 완료: model=%s device=%s",
                    self._cfg.wake_whisper_model,
                    self._cfg.wake_whisper_device,
                )
                return
            if backend == "faster_whisper":
                from faster_whisper import WhisperModel

                self._wake_whisper = WhisperModel(
                    self._cfg.wake_whisper_model,
                    device=self._cfg.wake_whisper_device,
                    compute_type=self._cfg.wake_whisper_compute_type,
                )
                self._wake_backend = "faster_whisper"
                logger.info(
                    "wake 전용 faster-whisper 로드 완료: model=%s device=%s compute=%s",
                    self._cfg.wake_whisper_model,
                    self._cfg.wake_whisper_device,
                    self._cfg.wake_whisper_compute_type,
                )
                return
            raise ValueError(f"지원하지 않는 wake_stt_backend={backend}")
        except Exception:
            logger.exception(
                "wake 전용 STT 로드 실패. 기본 STT로 fallback: backend=%s",
                backend,
            )
            self._wake_backend = self._cfg.whisper_backend
            self._wake_whisper = self._whisper

    def set_dialogue_capture_active(self, active: bool) -> None:
        """웨이크 대화 세션 중에는 웨이크어 없는 사용자 발화도 transcription 큐로 보낸다."""
        self._dialogue_capture_active = active
        if active:
            self._wake_suppressed = True
        logger.info("dialogue_capture_active=%s", active)

    def set_wake_suppressed(self, suppressed: bool) -> None:
        self._wake_suppressed = suppressed

    def extend_dialogue_capture_grace(self, ttl_sec: float) -> None:
        self._dialogue_capture_grace_until = max(
            self._dialogue_capture_grace_until,
            time.monotonic() + ttl_sec,
        )
        logger.info("dialogue_capture_grace_until=%.3f", self._dialogue_capture_grace_until)

    def remember_tts_text(self, text: str, *, ttl_sec: float = 45.0) -> None:
        """Remember local TTS text so STT can drop speaker echo by content."""
        cleaned = normalize_transcript(text)
        if not cleaned:
            return
        self._recent_tts_texts.append((time.time() + ttl_sec, cleaned))
        logger.info("TTS 에코 필터 등록: '%s'", cleaned[:80])

    def drain_wake_queue(self) -> int:
        drained = 0
        while True:
            try:
                self.wake_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        return drained

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._mic.stop()
        logger.info("AudioPipeline 중단")

    async def _run_loop(self) -> None:
        """메인 루프: VAD로 utterance를 분리하고 STT 실행."""
        loop = asyncio.get_running_loop()

        silence_frames_to_close = max(
            1, self._cfg.silence_tail_ms // self._cfg.frame_ms
        )
        min_frames = max(1, self._cfg.min_utterance_ms // self._cfg.frame_ms)
        max_frames = max(1, self._cfg.max_utterance_ms // self._cfg.frame_ms)

        in_speech = False
        speech_frames: list[bytes] = []
        silence_count = 0
        speech_started_at = 0.0

        while self._running:
            try:
                frame = await self._mic.read_frame()
            except (asyncio.IncompleteReadError, asyncio.CancelledError):
                break
            except Exception:
                logger.exception("MIC 프레임 읽기 오류")
                await asyncio.sleep(0.1)
                continue

            try:
                is_speech = self._vad.is_speech(frame, self._cfg.sample_rate)
            except Exception:
                is_speech = False

            if is_speech:
                if not in_speech:
                    in_speech = True
                    speech_started_at = time.time()
                    speech_frames = []
                    self._utterance_wake_triggered = False
                speech_frames.append(frame)
                silence_count = 0

                if (
                    self._cfg.wake_rolling_scan_enabled
                    and not self._utterance_wake_triggered
                    and len(speech_frames) * self._cfg.frame_ms
                    >= self._cfg.wake_rolling_scan_min_speech_ms
                ):
                    interval_frames = max(
                        1, self._cfg.wake_rolling_scan_interval_ms // self._cfg.frame_ms
                    )
                    if len(speech_frames) % interval_frames == 0:
                        self._schedule_rolling_wake_scan(loop, list(speech_frames))

                if len(speech_frames) >= max_frames:
                    await self._flush_utterance(
                        loop, speech_frames, speech_started_at
                    )
                    in_speech = False
                    speech_frames = []
                    silence_count = 0
            else:
                if in_speech:
                    speech_frames.append(frame)
                    silence_count += 1
                    if silence_count >= silence_frames_to_close:
                        if len(speech_frames) >= min_frames:
                            await self._flush_utterance(
                                loop, speech_frames, speech_started_at
                            )
                        in_speech = False
                        speech_frames = []
                        silence_count = 0

    def _schedule_rolling_wake_scan(
        self,
        loop: asyncio.AbstractEventLoop,
        frames: list[bytes],
    ) -> None:
        if self._rolling_scan_task and not self._rolling_scan_task.done():
            return
        self._rolling_scan_task = asyncio.create_task(
            self._rolling_wake_scan(loop, frames),
            name="rolling_wake_scan",
        )

    async def _rolling_wake_scan(
        self,
        loop: asyncio.AbstractEventLoop,
        frames: list[bytes],
    ) -> None:
        if self._utterance_wake_triggered or self._wake_suppressed:
            return
        pcm = b"".join(frames)
        tail_pcm = self._slice_pcm_tail(pcm, self._cfg.wake_focus_tail_sec)
        if not tail_pcm:
            return
        text = await loop.run_in_executor(
            None, self._transcribe_wake_focused_sync, tail_pcm
        )
        wake_hit = detect_wake_in_text(text)
        if not wake_hit:
            return
        logger.info(
            "롤링 웨이크 스캔(꼬리): STT='%s' -> '%s'",
            (text or "")[:80],
            wake_hit,
        )
        await self._emit_wake(wake_hit, text or "", started_at=time.time())

    async def _flush_utterance(
        self,
        loop: asyncio.AbstractEventLoop,
        frames: list[bytes],
        started_at: float,
    ) -> None:
        already_woken = self._utterance_wake_triggered
        if self._rolling_scan_task and not self._rolling_scan_task.done():
            self._rolling_scan_task.cancel()
            try:
                await self._rolling_scan_task
            except asyncio.CancelledError:
                pass
        self._rolling_scan_task = None

        pcm = b"".join(frames)
        duration = len(pcm) / 2 / self._cfg.sample_rate
        logger.info("utterance 감지: duration=%.2fs", duration)

        trace = get_latency_recorder()
        trace.mark("utterance_end", audio_duration_sec=round(duration, 2))
        if self._should_try_fast_wake_scan(duration, already_woken):
            trace.mark("wake_fast_scan_start", audio_duration_sec=round(duration, 2))
            wake_hit, focused_text = await self._scan_short_wake_utterance(loop, pcm)
            trace.mark(
                "wake_fast_scan_end",
                audio_duration_sec=round(duration, 2),
                text_preview=(focused_text or "")[:120],
                wake_detected=bool(wake_hit),
                source="short",
            )
            if wake_hit:
                logger.info(
                    "짧은 발화 wake fast-path: '%s' -> '%s'",
                    (focused_text or "")[:80],
                    wake_hit,
                )
                await self._emit_wake(wake_hit, focused_text or wake_hit, started_at=started_at)
                return

        trace.mark("stt_start", audio_duration_sec=round(duration, 2))
        text = await loop.run_in_executor(None, self._transcribe_sync, pcm)
        trace.mark(
            "stt_end",
            audio_duration_sec=round(duration, 2),
            text_preview=(text or "")[:120],
        )
        text = normalize_transcript(text)
        if is_prompt_echo_transcript(text):
            logger.info("STT 프롬프트 환각으로 추정, 무시: '%s'", text[:100])
            return
        if is_short_noise_hallucination(text, duration):
            logger.info(
                "짧은 잡음 STT 환각으로 추정, 무시: '%s' duration=%.2fs",
                text[:100],
                duration,
            )
            return
        if self._is_recent_tts_echo(text):
            logger.info("TTS 에코 STT로 추정, 무시: '%s'", text[:100])
            return

        wake_hit = detect_wake_in_text(text)
        wake_source = "full"

        if (
            not wake_hit
            and not self._wake_suppressed
            and self._cfg.wake_focus_scan_enabled
            and duration >= self._cfg.wake_focus_scan_min_duration_sec
        ):
            wake_hit, focused_text, wake_source = await self._scan_wake_in_noisy_utterance(
                loop, pcm
            )
            if wake_hit and focused_text:
                text = focused_text if not text else f"{text} {focused_text}".strip()

        if not text and not wake_hit:
            logger.info("STT 결과 비어 있음, 무시")
            return

        if text:
            logger.info("STT 결과: '%s'", text)

        if wake_hit and not already_woken and not self._wake_suppressed:
            if wake_source != "full":
                logger.info(
                    "TV/배경 구간 웨이크 스캔(%s): '%s' -> '%s'",
                    wake_source,
                    (text or "")[:80],
                    wake_hit,
                )
            elif wake_hit not in text:
                logger.info("웨이크 퍼지 매칭: STT='%s' -> '%s'", text[:80], wake_hit)
            await self._emit_wake(wake_hit, text, started_at=started_at)
            return

        if (
            (
                self._dialogue_capture_active
                or time.monotonic() < self._dialogue_capture_grace_until
            )
            and text
        ):
            await self.transcription_queue.put(
                TranscriptionResult(text=text, confidence=1.0, timestamp=started_at)
            )
            return

        if already_woken and text:
            await self.transcription_queue.put(
                TranscriptionResult(text=text, confidence=1.0, timestamp=started_at)
            )
            return

        if self._cfg.discard_non_wake_background:
            logger.info(
                "웨이크 없음 — TV/배경 발화로 추정, transcription 큐에 넣지 않음 (%.1fs)",
                duration,
            )
            return

        await self.transcription_queue.put(
            TranscriptionResult(text=text, confidence=1.0, timestamp=started_at)
        )

    def _should_try_fast_wake_scan(self, duration: float, already_woken: bool) -> bool:
        return (
            self._cfg.wake_fast_path_enabled
            and not already_woken
            and not self._wake_suppressed
            and self._cfg.wake_focus_scan_enabled
            and duration <= self._cfg.wake_fast_path_max_duration_sec
        )

    def _is_recent_tts_echo(self, text: str) -> bool:
        now = time.time()
        while self._recent_tts_texts and self._recent_tts_texts[0][0] < now:
            self._recent_tts_texts.popleft()
        return is_tts_echo_transcript(text, [item[1] for item in self._recent_tts_texts])

    async def _emit_wake(
        self,
        wake_hit: str,
        text: str,
        *,
        started_at: float,
    ) -> None:
        if self._wake_suppressed:
            if text.strip():
                await self.transcription_queue.put(
                    TranscriptionResult(
                        text=strip_wake_from_text(text, wake_hit) or text,
                        confidence=1.0,
                        timestamp=started_at,
                    )
                )
            return
        self._utterance_wake_triggered = True
        await self.wake_queue.put(
            WakeWordResult(detected=True, keyword=wake_hit, confidence=1.0)
        )
        remainder = strip_wake_from_text(text, wake_hit)
        if remainder:
            await self.transcription_queue.put(
                TranscriptionResult(
                    text=remainder, confidence=1.0, timestamp=started_at
                )
        )

    async def _scan_short_wake_utterance(
        self,
        loop: asyncio.AbstractEventLoop,
        pcm: bytes,
    ) -> tuple[str | None, str]:
        focused = await loop.run_in_executor(
            None,
            self._transcribe_wake_focused_sync,
            pcm,
        )
        wake_hit = detect_wake_in_text(focused)
        return wake_hit, (focused or "").strip()

    async def _scan_wake_in_noisy_utterance(
        self,
        loop: asyncio.AbstractEventLoop,
        pcm: bytes,
    ) -> tuple[str | None, str, str]:
        """긴 발화(TV+사용자)에서 앞/꼬리 구간만 다시 STT해 웨이크를 찾는다."""
        scans: list[tuple[str, bytes]] = []
        head_pcm = self._slice_pcm_head(pcm, self._cfg.wake_focus_head_sec)
        tail_pcm = self._slice_pcm_tail(pcm, self._cfg.wake_focus_tail_sec)
        if head_pcm:
            scans.append(("head", head_pcm))
        if tail_pcm and tail_pcm != head_pcm:
            scans.append(("tail", tail_pcm))

        for source, chunk in scans:
            focused = await loop.run_in_executor(
                None, self._transcribe_wake_focused_sync, chunk
            )
            wake_hit = detect_wake_in_text(focused)
            if wake_hit:
                return wake_hit, (focused or "").strip(), source
        return None, "", ""

    def _slice_pcm_tail(self, pcm: bytes, seconds: float) -> bytes:
        if seconds <= 0:
            return pcm
        max_bytes = int(self._cfg.sample_rate * 2 * seconds)
        return pcm[-max_bytes:] if len(pcm) > max_bytes else pcm

    def _slice_pcm_head(self, pcm: bytes, seconds: float) -> bytes:
        if seconds <= 0:
            return pcm
        max_bytes = int(self._cfg.sample_rate * 2 * seconds)
        return pcm[:max_bytes] if len(pcm) > max_bytes else pcm

    def _transcribe_wake_focused_sync(self, pcm_bytes: bytes) -> str:
        model = self._wake_whisper or self._whisper
        backend = self._wake_backend or self._cfg.whisper_backend
        if model is None or not pcm_bytes:
            return ""
        import numpy as np

        if backend == "gemini_stt":
            return self._transcribe_gemini_sync(
                pcm_bytes,
                prompt=(
                    self._cfg.gemini_stt_prompt
                    + "\n웨이크워드 후보는 '오디스', '오디스야'입니다."
                ),
            )

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if backend == "openai_whisper":
            result = model.transcribe(
                audio,
                language=self._cfg.language,
                initial_prompt=self._cfg.wake_focus_prompt,
                fp16=self._cfg.wake_whisper_device == "cuda",
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=1,
            )
            return normalize_transcript(str(result.get("text", "")))

        segments, _info = model.transcribe(
            audio,
            language=self._cfg.language,
            initial_prompt=self._cfg.wake_focus_prompt,
            vad_filter=False,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        return normalize_transcript(" ".join(seg.text for seg in segments))

    def drain_transcription_queue(self) -> int:
        drained = 0
        while True:
            try:
                self.transcription_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        return drained

    def _transcribe_sync(self, pcm_bytes: bytes) -> str:
        if self._whisper is None:
            return ""
        import numpy as np

        if self._cfg.whisper_backend == "gemini_stt":
            return self._transcribe_gemini_sync(
                pcm_bytes,
                prompt=self._cfg.gemini_stt_prompt,
            )

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self._cfg.whisper_backend == "openai_whisper":
            result = self._whisper.transcribe(
                audio,
                language=self._cfg.language,
                initial_prompt=self._cfg.initial_prompt,
                fp16=self._cfg.whisper_device == "cuda",
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=1,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.5,
            )
            return normalize_transcript(str(result.get("text", "")))

        segments, _info = self._whisper.transcribe(
            audio,
            language=self._cfg.language,
            initial_prompt=self._cfg.initial_prompt,
            vad_filter=False,
            beam_size=1,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.5,
        )
        return normalize_transcript(" ".join(seg.text for seg in segments))

    def _transcribe_gemini_sync(self, pcm_bytes: bytes, *, prompt: str) -> str:
        from google.genai import types

        wav_bytes = self._pcm_to_wav_bytes(pcm_bytes)
        response = self._whisper.models.generate_content(
            model=self._cfg.gemini_stt_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                candidate_count=1,
            ),
        )
        return normalize_transcript(getattr(response, "text", "") or "")

    def _pcm_to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._cfg.sample_rate)
            wav.writeframes(pcm_bytes)
        return buf.getvalue()
