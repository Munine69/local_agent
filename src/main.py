"""
로컬 에이전트 메인 엔트리포인트.

OCR_Agent.mermaid에 정의된 모든 노드를 조립하고,
엣지(데이터 흐름)를 asyncio 이벤트 루프 위에서 연결한다.

=== mermaid 엣지 매핑 ===
1-2. User --> STT --> VAD --> Wait_UX --> TTS --> Speaker   (호출 및 즉시 응답)
3.   STT --> Instruction_Log                               (STT 로그를 클라우드로)
4.   Cloud_Server --> State3                               (주기적 처방 레포트 요청)
5.   User --"약 가져왔어"--> STT                             (사용자 촬영 트리거)
6.   State3 --> Buffer --> Timer --> TTS --> Speaker        (촬영 실행 흐름)
7.   Timer --> OCR_Engine                                  (타이머 완료 후 OCR)
8.   VAD --> State1 --> STT                                (대화 대기 루프)
9.   Wait_UX --> TTS --> Speaker                           (고정 멘트)
10.  OCR_Engine --성공--> Drug_Parser --> DB                 (클라우드 전송)
11.  OCR_Engine --실패--> Wait_UX --> TTS --> Speaker        (재요청)
12.  Cam --> Capture_Mode                                  (RTSP 스트림)
13.  Cloud_Server --> TTS --> Speaker                       (클라우드 실시간 응답)
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from asyncio import QueueEmpty
from pathlib import Path
from typing import Any

from src.config_loader import load_config
from src.home_environment.cam import Cam, CamConfig
from src.home_environment.speaker import LocalSpeaker
from src.edge_node.vad import StubVAD, PipelineVAD, WAKE_WORDS
from src.edge_node.stt import StubSTT, PipelineSTT, TranscriptionResult
from src.edge_node.tts import StubTTS, GTTSEngine, TTSPriority
from src.edge_node.audio_pipeline import AudioPipeline, AudioPipelineConfig
from src.edge_node.stt_sanitize import compact_korean_text, is_filler_only_transcript
from src.edge_node.wait_ux import WaitUX
from src.edge_node.capture_mode.buffer import Buffer, QualityConfig, QualityFailReason
from src.edge_node.capture_mode.timer import Timer
from src.edge_node.capture_mode.ocr_engine import (
    OCREngine,
    OCREngineConfig,
    OCRResult,
    ActionRequired,
)
from src.edge_node.dialogue_manager.state_machine import (
    StateMachine,
    DialogueState,
    DialogueEvent,
)
from src.cloud_server.chat_client import CloudChatClient, CloudChatConfig
from src.cloud_server.drug_parser import (
    DrugParserConfig,
    HttpDrugParserClient,
    StubDrugParserClient,
    to_server_ocr_payload,
)
from src.cloud_server.instruction_log import (
    HttpInstructionLogClient,
    InstructionEntry,
    InstructionLogConfig,
    StubInstructionLogClient,
)
from src.runtime import turboquant_runtime
from src.runtime.latency_trace import configure_latency_trace, get_latency_recorder

logger = logging.getLogger(__name__)

IDENTITY_CONFIRMATION_MARKERS = (
    "지금 말씀하시는 분이",
    "본인이 맞으신가요",
    "다시 확인할게요",
)
IDENTITY_NEGATIVE_RESPONSES = (
    "아니",
    "아니야",
    "아니요",
    "아뇨",
    "틀려",
    "틀렸",
    "다른",
)
IDENTITY_AFFIRMATIVE_RESPONSES = (
    "맞아",
    "맞아요",
    "응",
    "네",
    "그래",
    "본인",
)
IDENTITY_RESET_PROMPT = "알겠습니다. 다시 등록할게요. 이름, 나이, 성별을 한 번에 말씀해 주세요."
PRIVACY_SAFE_SPOKEN_REPLACEMENTS = (
    (
        "이름, 나이, 성별을 한 번에 말씀해 주세요. 예: 김영수 72살 남자.",
        "이름, 나이, 성별을 한 번에 말씀해 주세요.",
    ),
)

def _strip_wake_words(text: str) -> str:
    """웨이크워드는 서버 약물/DUR 추출에서 제외한다."""
    import re

    cleaned = (text or "").strip()
    for wake in sorted(WAKE_WORDS, key=len, reverse=True):
        cleaned = cleaned.replace(wake, " ")
    return re.sub(r"\s+", " ", cleaned).strip(" ,.!?~")


def _is_capture_trigger_text(text: str) -> bool:
    if any(kw in text for kw in CAPTURE_TRIGGER_KEYWORDS):
        return True
    compact = compact_korean_text(text)
    has_capture_target = any(token in compact for token in ("처방전", "약봉투", "약사진"))
    has_photo = "사진" in compact or "촬영" in compact
    has_capture_verb = any(token in compact for token in ("찍", "찍자", "찍어", "보여"))
    return (has_capture_target and (has_photo or has_capture_verb)) or (
        has_photo and has_capture_verb
    )


CAPTURE_TRIGGER_KEYWORDS = [
    "약 가져왔어",
    "약 찍어",
    "약 보여줄게",
    "사진 찍어",
    "사진 좀 찍",
    "처방전 사진",
    "처방전 찍",
    "약 사진",
    "약봉투",
    "찍을게",
]
QUALITY_GUIDE_TEXT_MARKERS = (
    "사진이 조금 흔들",
    "사진이 좀 흔들",
    "카메라를 조금만 더 멀리",
    "손을 가만히",
    "화면이 조금 어두",
    "조금 어두",
    "너무 밝",
    "빛 반사",
    "반사광",
)
OCR_RECAPTURE_MARKERS = (
    "다시 촬영",
    "다시 찍",
    "다시 보여",
    "읽기 어려",
    "인식하기 어려",
    "인식률이 낮",
    "흐리게 인식",
    "정확해야 하므로",
)
OCR_CONFIRMATION_MARKERS = (
    "저장",
    "저장해",
    "저장해줘",
    "등록",
    "등록해",
    "맞아",
    "확인",
    "응",
    "네",
)


class LocalAgent:
    """mermaid 아키텍처에 따른 로컬 에이전트 오케스트레이터.

    모든 mermaid 노드를 생성하고, 엣지를 asyncio 태스크/큐로 연결한다.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_config()

        turboquant_runtime.install(cfg.get("turboquant"))
        self._latency = configure_latency_trace(cfg.get("latency_trace"))
        self._pending_tts_role: str | None = None
        self._wake_session_active = False
        self._recent_cloud_turns: dict[str, float] = {}
        self._last_spoken_text = ""
        self._last_spoken_at = 0.0
        self._identity_confirmation_pending = False
        self._identity_prompt_spoken_at = 0.0
        self._last_tts_completed_at = 0.0
        self._ocr_exchange_active = False
        self._ocr_confirmation_pending = False
        self._pending_cloud_filler_texts: set[str] = set()

        # --- Home_Environment 노드 ---
        rtsp_cfg = cfg.get("rtsp", {})
        self.cam = Cam(CamConfig(
            url=rtsp_cfg.get("url", "rtsp://192.168.0.100:554/stream"),
            reconnect_backoff_sec=rtsp_cfg.get("reconnect_backoff_sec", 2.0),
            max_reconnect_attempts=rtsp_cfg.get("max_reconnect_attempts", 10),
        ))
        audio_cfg = cfg.get("audio", {})
        self.speaker = LocalSpeaker(
            device=audio_cfg.get("output_device", "default"),
            sample_rate=audio_cfg.get("sample_rate", 22050),
            channels=audio_cfg.get("channels", 1),
            capture_shutter_path=audio_cfg.get("capture_shutter_path"),
        )

        # --- Edge_Node 노드 ---
        stt_cfg = cfg.get("stt", {})
        self._stt_cfg = stt_cfg
        if stt_cfg.get("enabled", False):
            pipeline_cfg = AudioPipelineConfig(
                input_device=stt_cfg.get(
                    "input_device", audio_cfg.get("input_device", "default")
                ),
                sample_rate=stt_cfg.get("sample_rate", 16000),
                frame_ms=stt_cfg.get("frame_ms", 30),
                vad_aggressiveness=stt_cfg.get("vad_aggressiveness", 2),
                silence_tail_ms=stt_cfg.get("silence_tail_ms", 700),
                min_utterance_ms=stt_cfg.get("min_utterance_ms", 300),
                max_utterance_ms=stt_cfg.get("max_utterance_ms", 12000),
                whisper_backend=stt_cfg.get("whisper_backend", "faster_whisper"),
                whisper_model=stt_cfg.get("whisper_model", "small"),
                whisper_compute_type=stt_cfg.get("whisper_compute_type", "int8"),
                whisper_device=stt_cfg.get("whisper_device", "cpu"),
                gemini_stt_model=stt_cfg.get("gemini_stt_model", "gemini-2.5-flash"),
                gemini_api_key=stt_cfg.get("gemini_api_key", ""),
                gemini_api_key_env=stt_cfg.get("gemini_api_key_env", "GEMINI_API_KEY"),
                gemini_stt_prompt=stt_cfg.get(
                    "gemini_stt_prompt",
                    (
                        "이 오디오는 한국어 음성입니다. 들리는 발화만 그대로 전사하세요. "
                        "추측하거나 문장을 보완하지 마세요. 음성이 없거나 불명확하면 빈 문자열만 반환하세요."
                    ),
                ),
                language=stt_cfg.get("language", "ko"),
                initial_prompt=stt_cfg.get(
                    "initial_prompt",
                    "오디스, 오디세, 오딧스, 오티스, 오지스, 보리스, 약, 처방전, 복용, 어르신, 사진, 찍어, 가져왔어",
                ),
                wake_focus_prompt=stt_cfg.get(
                    "wake_focus_prompt",
                    "오디스, 오디스야, 오디세, 오딧스, 오티스, 오지스, 보리스, 어디스, 오디서, 들려",
                ),
                wake_focus_scan_enabled=stt_cfg.get("wake_focus_scan_enabled", True),
                wake_focus_scan_min_duration_sec=float(
                    stt_cfg.get("wake_focus_scan_min_duration_sec", 2.5)
                ),
                wake_focus_tail_sec=float(stt_cfg.get("wake_focus_tail_sec", 2.8)),
                wake_focus_head_sec=float(stt_cfg.get("wake_focus_head_sec", 2.5)),
                wake_rolling_scan_enabled=stt_cfg.get("wake_rolling_scan_enabled", True),
                wake_rolling_scan_interval_ms=int(
                    stt_cfg.get("wake_rolling_scan_interval_ms", 2000)
                ),
                wake_rolling_scan_min_speech_ms=int(
                    stt_cfg.get("wake_rolling_scan_min_speech_ms", 2500)
                ),
                wake_fast_path_enabled=stt_cfg.get("wake_fast_path_enabled", True),
                wake_fast_path_max_duration_sec=float(
                    stt_cfg.get("wake_fast_path_max_duration_sec", 1.8)
                ),
                wake_stt_backend=stt_cfg.get("wake_stt_backend", "same"),
                wake_whisper_model=stt_cfg.get("wake_whisper_model", "tiny"),
                wake_whisper_compute_type=stt_cfg.get("wake_whisper_compute_type", "int8"),
                wake_whisper_device=stt_cfg.get("wake_whisper_device", "cpu"),
                discard_non_wake_background=stt_cfg.get(
                    "discard_non_wake_background", True
                ),
            )
            self.audio_pipeline: AudioPipeline | None = AudioPipeline(pipeline_cfg)
            self.vad = PipelineVAD(self.audio_pipeline)
            self.stt = PipelineSTT(self.audio_pipeline)
            logger.info("STT 모드: 실제 마이크 + faster-whisper")
        else:
            self._stt_cfg = {}
            self.audio_pipeline = None
            self.vad = StubVAD()
            self.stt = StubSTT()
            logger.info("STT 모드: Stub (시뮬레이션)")

        tts_cfg = cfg.get("tts", {})
        if tts_cfg.get("enabled", False) and tts_cfg.get("engine", "gtts") == "gtts":
            self.tts = GTTSEngine(
                speaker=self.speaker,
                lang=tts_cfg.get("lang", "ko"),
                tld=tts_cfg.get("tld", "co.kr"),
                slow=tts_cfg.get("slow", False),
                sample_rate=tts_cfg.get("sample_rate", 22050),
                channels=tts_cfg.get("channels", 1),
                on_latency_event=self._latency.mark,
            )
            logger.info(
                "TTS 모드: gTTS (lang=%s tld=%s)",
                tts_cfg.get("lang", "ko"),
                tts_cfg.get("tld", "co.kr"),
            )
        else:
            self.tts = StubTTS()
            logger.info("TTS 모드: Stub (로그 출력만)")

        quality_cfg = cfg.get("quality", {})
        self.buffer = Buffer(
            cam=self.cam,
            quality_config=QualityConfig(
                blur_threshold=quality_cfg.get("blur_threshold", 100.0),
                brightness_min=quality_cfg.get("brightness_min", 50),
                brightness_max=quality_cfg.get("brightness_max", 230),
                glare_max_ratio=quality_cfg.get("glare_max_ratio", 0.15),
            ),
            buffer_size=rtsp_cfg.get("buffer_size", 30),
            quality_fail_cooldown_sec=float(
                quality_cfg.get("quality_fail_cooldown_sec", 5.0)
            ),
        )

        capture_cfg = cfg.get("capture", {})
        self.timer = Timer(
            buffer=self.buffer,
            timer_seconds=capture_cfg.get("timer_seconds", 3),
        )

        ocr_cfg = cfg.get("ocr", {})
        self.ocr_engine = OCREngine(OCREngineConfig(
            model_path=ocr_cfg.get("model_path", ""),
            provider=ocr_cfg.get("provider", "gemini_ocr"),
            save_dir=ocr_cfg.get("save_dir", ocr_cfg.get("glmocr_save_dir", "runtime/ocr")),
            hf_device=ocr_cfg.get("hf_device", "auto"),
            hf_torch_dtype=ocr_cfg.get("hf_torch_dtype", "auto"),
            hf_prompt=ocr_cfg.get("hf_prompt", "Text Recognition:"),
            hf_max_new_tokens=ocr_cfg.get("hf_max_new_tokens", 4096),
            hf_max_image_side=ocr_cfg.get("hf_max_image_side", 1280),
            hf_extract_document=ocr_cfg.get("hf_extract_document", True),
            hf_repetition_penalty=ocr_cfg.get("hf_repetition_penalty", 1.15),
            hf_no_repeat_ngram_size=ocr_cfg.get("hf_no_repeat_ngram_size", 8),
            gemini_model=ocr_cfg.get("gemini_model", "gemini-3-flash-preview"),
            gemini_fallback_models=ocr_cfg.get(
                "gemini_fallback_models",
                ["gemini-2.5-flash", "gemini-flash-latest"],
            ),
            gemini_api_key=ocr_cfg.get("gemini_api_key", ""),
            gemini_api_key_env=ocr_cfg.get("gemini_api_key_env", "GEMINI_API_KEY"),
            gemini_prompt=ocr_cfg.get("gemini_prompt", ""),
            glmocr_mode=ocr_cfg.get("glmocr_mode", "maas"),
            glmocr_api_key_env=ocr_cfg.get("glmocr_api_key_env", "ZHIPU_API_KEY"),
            glmocr_timeout_sec=ocr_cfg.get("glmocr_timeout_sec", 600),
            glmocr_save_dir=ocr_cfg.get("glmocr_save_dir", "runtime/ocr"),
            confidence_threshold=ocr_cfg.get("confidence_threshold", 0.85),
            fuzzy_match_max_distance=ocr_cfg.get("fuzzy_match_max_distance", 2),
            medical_dict_path=ocr_cfg.get("medical_dict_path", "config/medical_terms.json"),
        ))

        wait_ux_cfg = cfg.get("wait_ux", {})
        self.wait_ux = WaitUX(
            escalation_intervals=wait_ux_cfg.get("waiting_escalation_intervals_sec", [3, 7, 15]),
            quality_guide_cooldown_sec=float(
                quality_cfg.get("quality_fail_cooldown_sec", 5.0)
            ),
        )

        self.state_machine = StateMachine()

        # --- Cloud_Server 노드 ---
        cloud_cfg = cfg.get("cloud", {})
        self._cloud_enabled = bool(cloud_cfg.get("enabled", False))
        cloud_endpoint = str(cloud_cfg.get("endpoint", "http://localhost:8000")).rstrip("/")
        cloud_speaker_id = str(cloud_cfg.get("speaker_id", "jetson_live"))
        cloud_timeout = float(cloud_cfg.get("timeout_sec", 10.0))
        cloud_ws_timeout = float(cloud_cfg.get("websocket_timeout_sec", 120.0))

        if self._cloud_enabled:
            drug_parser_path = cloud_cfg.get("drug_parser_path", "/api/ocr/analyze")
            instruction_log_path = cloud_cfg.get("instruction_log_path", "/api/stt/log")
            self.drug_parser = HttpDrugParserClient(
                DrugParserConfig(
                    endpoint=cloud_endpoint,
                    path=drug_parser_path,
                    timeout_sec=cloud_timeout,
                    retry_count=int(cloud_cfg.get("retry_count", 3)),
                    retry_backoff_sec=float(cloud_cfg.get("retry_backoff_sec", 1.0)),
                )
            )
            self.instruction_log = HttpInstructionLogClient(
                InstructionLogConfig(
                    endpoint=cloud_endpoint,
                    path=instruction_log_path,
                    timeout_sec=cloud_timeout,
                    retry_count=int(cloud_cfg.get("retry_count", 3)),
                    retry_backoff_sec=float(cloud_cfg.get("retry_backoff_sec", 1.0)),
                    speaker_id=cloud_speaker_id,
                )
            )
            self.cloud_chat = CloudChatClient(
                CloudChatConfig(
                    endpoint=cloud_endpoint,
                    websocket_path=cloud_cfg.get("websocket_path", "/ws/chat"),
                    speaker_id=cloud_speaker_id,
                    timeout_sec=cloud_ws_timeout,
                    first_response_timeout_sec=float(
                        cloud_cfg.get("websocket_first_response_timeout_sec", 8.0)
                    ),
                    on_ws_event=self._latency.mark,
                )
            )
            logger.info(
                "Cloud 모드: endpoint=%s speaker_id=%s ws=%s",
                cloud_endpoint,
                cloud_speaker_id,
                self.cloud_chat.ws_url,
            )
        else:
            self.drug_parser = StubDrugParserClient()
            self.instruction_log = StubInstructionLogClient()
            self.cloud_chat = None
            logger.info("Cloud 모드: Stub (로컬 시뮬레이션)")

        # --- 내부 큐 (엣지 연결용) ---
        self._tts_queue: asyncio.Queue[str] = asyncio.Queue()
        self._quality_fail_queue: asyncio.Queue[QualityFailReason] = asyncio.Queue()
        self._deferred_transcription_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()

        self._tasks: list[asyncio.Task[Any]] = []

    async def start(self) -> None:
        """모든 노드를 초기화하고 mermaid 엣지를 연결하여 에이전트를 시작한다."""
        logger.info("=== 로컬 에이전트 시작 ===")
        if self._latency.enabled:
            logger.info(
                "LatencyTrace 활성: %s",
                self._latency.output_path,
            )

        await self.ocr_engine.load()
        if isinstance(self.drug_parser, HttpDrugParserClient):
            await self.drug_parser.open()
        if isinstance(self.instruction_log, HttpInstructionLogClient):
            await self.instruction_log.open()
        if self.audio_pipeline is not None:
            await self.audio_pipeline.start()
        await self.vad.start()
        await self.stt.start_stream()
        await self._prewarm_tts()

        self._wire_edges()

        self._tasks = [
            asyncio.create_task(self.state_machine.run(), name="state_machine"),
            asyncio.create_task(self._wakeword_loop(), name="wakeword_loop"),
            asyncio.create_task(self._tts_consumer_loop(), name="tts_consumer"),
            asyncio.create_task(self._quality_feedback_loop(), name="quality_feedback"),
            asyncio.create_task(self._followup_transcription_loop(), name="followup_transcription"),
        ]

        logger.info("에이전트 실행 중: %d개 태스크", len(self._tasks))

    def _wire_edges(self) -> None:
        """mermaid 엣지에 따라 노드 간 큐/콜백을 연결한다."""

        # Wait_UX --> TTS (엣지 9, 11)
        self.wait_ux.set_tts_queue(self._tts_queue)

        # Timer --> TTS (엣지 6: 카운트다운)
        self.timer.set_tts_queue(self._tts_queue)

        # Buffer 품질 실패 --> Wait_UX (엣지 12 -> 품질 피드백)
        self.buffer.set_quality_fail_queue(self._quality_fail_queue)

        # StateMachine 전이 콜백 등록
        self.state_machine.on_transition(self._on_state_transition)

    async def _prewarm_tts(self) -> None:
        prepare = getattr(self.tts, "prepare_texts", None)
        if prepare is None:
            return
        await prepare(self.wait_ux.immediate_response_templates())

    async def _wakeword_loop(self) -> None:
        """엣지 1-2, 8: User --> STT --> VAD --> Wait_UX --> TTS --> Speaker

        VAD --> State1 --> STT (대화 대기 루프)
        """
        while True:
            try:
                result = await self.vad.wait_for_wakeword()
                if not result.detected:
                    continue

                if self._wake_session_active:
                    logger.info(
                        "대화 세션 진행 중 — 추가 웨이크 무시: '%s'",
                        result.keyword,
                    )
                    continue

                logger.info("Wake-word 감지: '%s'", result.keyword)
                self._wake_session_active = True
                self._drain_stale_tts_queue()

                if self.audio_pipeline is not None:
                    self.audio_pipeline.drain_wake_queue()
                    # 서버 TTS·등록 답변 전에 켜야 함. 늦게 켜면
                    # "오디스, 김영수야" 잔여 발화가 TV 배경으로 버려짐.
                    self.audio_pipeline.set_dialogue_capture_active(True)

                self._latency.begin_turn(
                    "wake_dialogue",
                    wake_keyword=result.keyword,
                )
                self._latency.mark("wake_detected", keyword=result.keyword)

                # VAD --> State1 (대화 대기 상태 진입)
                await self.state_machine.send_event(DialogueEvent.WAKE_WORD_DETECTED)

                # Wake-word is local-only. The server receives only actual user turns.
                self._pending_tts_role = "wake"
                await self.wait_ux.immediate_response()

                # State1 --> STT: 웨이크 한 번에 여러 턴(등록 → 회상) 허용
                try:
                    self._latency.mark("question_wait_start")
                    loop = asyncio.get_running_loop()
                    session_sec = float(
                        self._stt_cfg.get("dialogue_session_timeout_sec", 90.0)
                    )
                    followup_sec = float(
                        self._stt_cfg.get("followup_listen_timeout_sec", 45.0)
                    )
                    session_deadline = loop.time() + session_sec
                    turns = 0
                    listen_grace = float(
                        self._stt_cfg.get("post_tts_listen_grace_sec", 2.5)
                    )
                    if listen_grace > 0:
                        await asyncio.sleep(listen_grace)

                    if self.audio_pipeline is not None:
                        self._discard_prefilled_transcriptions()

                    while (
                        loop.time() < session_deadline
                        and turns < 6
                    ):
                        remaining = session_deadline - loop.time()
                        if (
                            self._last_tts_completed_at > 0
                            and session_deadline - self._last_tts_completed_at < followup_sec
                        ):
                            session_deadline = self._last_tts_completed_at + followup_sec
                            remaining = session_deadline - loop.time()
                        question_timeout = float(
                            self._stt_cfg.get("question_wait_timeout_sec", 35.0)
                        )
                        transcription = await self._wait_for_user_question(
                            timeout_sec=min(question_timeout, max(8.0, remaining)),
                        )
                        if transcription is None:
                            continue
                        await self._route_transcription_result(
                            transcription,
                            finalize_trace=False,
                        )
                        turns += 1
                        session_deadline = loop.time() + followup_sec
                    if turns == 0 and self._latency.enabled:
                        self._latency.end_turn(status="question_timeout")
                    elif self._latency.enabled:
                        self._latency.end_turn(status="ok", dialogue_turns=turns)
                finally:
                    if self.audio_pipeline is not None:
                        self.audio_pipeline.set_dialogue_capture_active(False)
                        self.audio_pipeline.set_wake_suppressed(False)
                    self._wake_session_active = False

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Wake-word 루프 오류")

    def _drain_stale_tts_queue(self) -> int:
        drained = 0
        while True:
            try:
                self._tts_queue.get_nowait()
                drained += 1
            except QueueEmpty:
                break
        if drained:
            logger.info("이전 TTS 큐 %d건 제거 (재요청 멘트 방지)", drained)
        return drained

    def _drain_quality_guide_tts_queue(self) -> int:
        kept: list[str] = []
        drained = 0
        while True:
            try:
                text = self._tts_queue.get_nowait()
            except QueueEmpty:
                break
            if any(marker in text for marker in QUALITY_GUIDE_TEXT_MARKERS):
                drained += 1
            else:
                kept.append(text)
        for text in kept:
            self._tts_queue.put_nowait(text)
        if drained:
            logger.info("촬영 완료 전 품질 안내 TTS 큐 %d건 제거", drained)
        return drained

    def _drain_pending_cloud_fillers(self) -> int:
        if not self._pending_cloud_filler_texts:
            return 0
        kept: list[str] = []
        drained = 0
        while True:
            try:
                text = self._tts_queue.get_nowait()
            except QueueEmpty:
                break
            if text in self._pending_cloud_filler_texts:
                drained += 1
            else:
                kept.append(text)
        for text in kept:
            self._tts_queue.put_nowait(text)
        if drained:
            logger.info("최종 답변 전 대기 filler 큐 %d건 제거", drained)
        return drained

    def _discard_prefilled_transcriptions(self) -> int:
        if self.audio_pipeline is None:
            return 0
        discarded = 0
        while True:
            try:
                self.audio_pipeline.transcription_queue.get_nowait()
                discarded += 1
            except QueueEmpty:
                break
        if discarded:
            logger.info("wake 이전 대기 STT %d건 폐기", discarded)
        return discarded

    async def _wait_for_user_question(
        self,
        timeout_sec: float = 20.0,
    ) -> TranscriptionResult | None:
        """웨이크워드 이후 실제 질문 발화를 기다린다 (빈 STT는 무시, 재요청 TTS 없음)."""
        deadline = asyncio.get_running_loop().time() + timeout_sec
        empty_count = 0
        max_empty = int(self._stt_cfg.get("max_empty_stt_skips", 20))

        while asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            try:
                result = await asyncio.wait_for(
                    self.stt.get_transcription(),
                    timeout=max(0.1, remaining),
                )
            except asyncio.TimeoutError:
                return None
            if result.text.strip():
                return result
            empty_count += 1
            if empty_count <= 3 or empty_count % 5 == 0:
                logger.info(
                    "빈 STT 무시, 질문 대기 중... (%d/%d)",
                    empty_count,
                    max_empty,
                )
            if empty_count >= max_empty:
                logger.info("빈 STT 연속 — 다음 대기 구간으로 넘어감 (TTS 재요청 없음)")
                return None
        return None

    async def _route_transcription_result(
        self,
        result: TranscriptionResult,
        *,
        finalize_trace: bool = True,
    ) -> None:
        text = _strip_wake_words(result.text.strip())
        compact = compact_korean_text(text)
        is_ocr_confirmation = (
            self._ocr_confirmation_pending
            and any(marker in compact for marker in OCR_CONFIRMATION_MARKERS)
        )
        if self._ocr_exchange_active and not _is_capture_trigger_text(text):
            logger.info("OCR 처리 중 STT 보류: '%s'", result.text[:80])
            await self._deferred_transcription_queue.put(result)
            return
        await self._handle_stt_result(result, finalize_trace=finalize_trace)

    async def _followup_transcription_loop(self) -> None:
        while True:
            try:
                if self.audio_pipeline is None:
                    await asyncio.sleep(1.0)
                    continue
                if self._wake_session_active:
                    await asyncio.sleep(0.2)
                    continue
                result = await self.audio_pipeline.transcription_queue.get()
                await self._route_transcription_result(result, finalize_trace=False)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("후속 발화 처리 루프 오류")

    async def _flush_deferred_transcriptions(self) -> None:
        while True:
            try:
                result = self._deferred_transcription_queue.get_nowait()
            except QueueEmpty:
                break
            logger.info("보류 STT 처리: '%s'", result.text[:80])
            await self._route_transcription_result(result, finalize_trace=False)

    async def _handle_stt_result(
        self,
        result: TranscriptionResult,
        *,
        finalize_trace: bool = True,
    ) -> None:
        """STT 결과를 처리한다.

        엣지 3: STT --> Instruction_Log
        엣지 5: User --"약 가져왔어"--> STT --> 촬영 트리거
        """
        text = _strip_wake_words(result.text.strip())
        if not text:
            return

        # 엣지 5: 촬영 트리거 키워드 검사
        if _is_capture_trigger_text(text):
            self._latency.mark(
                "stt_question_received",
                text_preview=text[:120],
            )
            entry = InstructionEntry(text=text, timestamp=result.timestamp)
            asyncio.create_task(self.instruction_log.send_log(entry))
            logger.info("촬영 트리거 감지: '%s'", text)
            self._latency.end_turn(status="capture_trigger", text_preview=text[:120])
            await self.state_machine.send_event(DialogueEvent.CAPTURE_TRIGGER)
            return

        if await self._handle_identity_confirmation_reply(text):
            return

        if await self._handle_ocr_confirmation_reply(text):
            return

        if is_filler_only_transcript(text):
            logger.info("짧은 추임새/잡음 STT 무시: '%s'", text)
            return

        self._latency.mark(
            "stt_question_received",
            text_preview=text[:120],
        )

        # 엣지 3: STT --> Instruction_Log
        entry = InstructionEntry(text=text, timestamp=result.timestamp)
        asyncio.create_task(self.instruction_log.send_log(entry))

        if self._is_duplicate_cloud_turn(text):
            logger.info("중복 STT 턴 무시: '%s'", text)
            return

        if self.cloud_chat is not None:
            await self._handle_cloud_dialogue(
                text,
                finalize_trace=finalize_trace,
            )

    async def _handle_identity_confirmation_reply(self, text: str) -> bool:
        if not self._identity_confirmation_pending:
            return False
        compact = compact_korean_text(text)
        if any(token in compact for token in IDENTITY_NEGATIVE_RESPONSES):
            logger.info("신원확인 부정 응답 감지: '%s'", text)
            self._identity_confirmation_pending = False
            self._last_spoken_text = ""
            self._recent_cloud_turns.clear()
            if self.cloud_chat is not None:
                asyncio.create_task(
                    self._notify_identity_event(
                        "identity_rejected",
                        text,
                        action="reset_identity",
                    )
                )
            await self._speak(
                IDENTITY_RESET_PROMPT,
                TTSPriority.HIGH,
                trace_role="identity_reset",
            )
            return True
        if any(token in compact for token in IDENTITY_AFFIRMATIVE_RESPONSES):
            logger.info("신원확인 긍정 응답 감지: '%s'", text)
            self._identity_confirmation_pending = False
            if self.cloud_chat is not None:
                asyncio.create_task(
                    self._notify_identity_event(
                        "identity_confirmed",
                        text,
                        action="confirm_identity",
                    )
                )
            await self._handle_cloud_dialogue(
                text,
                finalize_trace=False,
                event_type="identity_confirmed",
                context={"identity_action": "confirm_identity"},
            )
            return True
        return False

    async def _handle_ocr_confirmation_reply(self, text: str) -> bool:
        if not self._ocr_confirmation_pending:
            return False
        compact = compact_korean_text(text)
        if not any(marker in compact for marker in OCR_CONFIRMATION_MARKERS):
            return False
        logger.info("OCR 저장 확인 응답 감지: '%s'", text)
        self._ocr_confirmation_pending = False
        if self.cloud_chat is not None:
            await self._handle_cloud_dialogue(
                text,
                finalize_trace=False,
                event_type="stt_result",
                context={"ocr_confirmation": True},
            )
        return True

    async def _notify_identity_event(
        self,
        event_type: str,
        text: str,
        *,
        action: str,
    ) -> None:
        try:
            async for _ in self.cloud_chat.send_stt(
                text,
                event_type=event_type,
                context={"identity_action": action},
            ):
                pass
        except Exception:
            logger.exception("신원 이벤트 서버 통지 실패: %s", event_type)

    def _is_duplicate_cloud_turn(self, text: str) -> bool:
        compact = compact_korean_text(text)
        if not compact:
            return True
        now = time.monotonic()
        ttl = float(self._stt_cfg.get("duplicate_turn_ttl_sec", 4.0))
        expired = [
            key
            for key, seen_at in self._recent_cloud_turns.items()
            if now - seen_at > ttl
        ]
        for key in expired:
            self._recent_cloud_turns.pop(key, None)
        if compact in self._recent_cloud_turns:
            return True
        self._recent_cloud_turns[compact] = now
        return False

    def _sanitize_spoken_text(self, spoken: str) -> str:
        sanitized = spoken
        for old, new in PRIVACY_SAFE_SPOKEN_REPLACEMENTS:
            sanitized = sanitized.replace(old, new)
        if sanitized != spoken:
            logger.info("서버 TTS 멘트 일반화: %s -> %s", spoken[:120], sanitized[:120])
        return sanitized

    def _is_identity_confirmation_prompt(self, spoken: str) -> bool:
        return any(marker in spoken for marker in IDENTITY_CONFIRMATION_MARKERS)

    async def _speak_cloud_spoken(
        self,
        spoken: str,
        *,
        trace_role: str,
        message: dict[str, Any],
    ) -> bool:
        if not spoken or not message.get("requires_tts", True):
            return False
        spoken = self._sanitize_spoken_text(spoken)
        is_identity_prompt = self._is_identity_confirmation_prompt(spoken)
        if is_identity_prompt:
            self._identity_confirmation_pending = True
            self._identity_prompt_spoken_at = time.monotonic()
            trace_role = "identity_check"
        if not is_identity_prompt and self._should_suppress_repeated_spoken(spoken):
            logger.info("반복 서버 멘트 TTS 생략: %s", spoken[:120])
            return False
        await self._speak(
            spoken,
            TTSPriority.HIGH,
            trace_role=trace_role,
        )
        return True

    async def _handle_cloud_dialogue(
        self,
        text: str,
        *,
        finalize_trace: bool = True,
        event_type: str = "stt_result",
        context: dict[str, Any] | None = None,
        speak_error: bool = True,
    ) -> None:
        """STT --> /ws/chat --> filler/response --> TTS (엣지 13)."""
        logger.info("Cloud 대화 시작: '%s'", text)
        self._latency.mark("cloud_dialogue_start", text_preview=text[:120])
        status = "ok"
        self._pending_cloud_filler_texts.clear()
        try:
            async for message in self.cloud_chat.send_stt(
                text,
                event_type=event_type,
                context=context,
            ):
                msg_type = message.get("type", "")
                if msg_type == "filler":
                    filler = CloudChatClient.spoken_text(message)
                    if filler:
                        self._drain_pending_cloud_fillers()
                        self._pending_cloud_filler_texts.add(filler)
                        self._pending_tts_role = "filler"
                        await self._tts_queue.put(filler)
                elif msg_type == "identity_check":
                    spoken = CloudChatClient.spoken_text(message)
                    logger.info("Cloud identity_check 수신: %s", spoken[:120])
                    self._drain_pending_cloud_fillers()
                    await self.tts.stop()
                    await self._speak_cloud_spoken(
                        spoken,
                        trace_role="identity_check",
                        message=message,
                    )
                    status = "identity_check"
                elif msg_type in {"response", "reminder", "ocr_processed"}:
                    spoken = CloudChatClient.spoken_text(message)
                    self._drain_pending_cloud_fillers()
                    await self.tts.stop()
                    await self._speak_cloud_spoken(
                        spoken,
                        trace_role="response",
                        message=message,
                    )
                elif msg_type == "ocr_request":
                    spoken = CloudChatClient.spoken_text(message)
                    self._drain_pending_cloud_fillers()
                    if spoken:
                        self._pending_tts_role = "ocr_request"
                        await self._tts_queue.put(spoken)
                    await self.state_machine.send_event(DialogueEvent.CAPTURE_TRIGGER)
                    status = "ocr_request"
                elif msg_type == "error":
                    logger.warning("CloudChat error: %s", message.get("message"))
                    status = "cloud_error"
                    if speak_error:
                        self._pending_tts_role = "error"
                        await self._tts_queue.put(
                            "지금은 서버와 연결이 어렵습니다. 잠시 후 다시 말씀해 주세요."
                        )
        except Exception:
            logger.exception("Cloud 대화 처리 실패")
            status = "exception"
            if speak_error:
                self._pending_tts_role = "error"
                await self._tts_queue.put(
                    "지금은 서버와 연결이 어렵습니다. 잠시 후 다시 말씀해 주세요."
                )
        finally:
            self._latency.mark("cloud_dialogue_end", status=status)
            if finalize_trace:
                self._latency.end_turn(status=status, text_preview=text[:120])
            self._pending_cloud_filler_texts.clear()

    async def _on_state_transition(
        self,
        prev: DialogueState,
        next_state: DialogueState,
        event: DialogueEvent,
    ) -> None:
        """StateMachine 상태 전이에 따른 액션을 실행한다."""

        # 엣지 6: State3 --"촬영 실행"--> Buffer --> Timer
        if next_state == DialogueState.STATE3_CAPTURE_WAIT:
            await self.state_machine.send_event(DialogueEvent.CAPTURE_START)

        elif next_state == DialogueState.CAPTURING:
            asyncio.create_task(self._run_capture_flow())

    async def _run_capture_flow(self) -> None:
        """촬영 실행 흐름 (엣지 6, 7, 10, 11).

        State3 --> Buffer --> Timer --> OCR_Engine
                              |
                              +--> TTS --> Speaker (카운트다운)
        """
        try:
            # 엣지 12: Cam --> Capture_Mode (Buffer 활성화)
            if not self.cam.is_opened:
                try:
                    await self.cam.open()
                except ConnectionError:
                    logger.error("Cam 연결 실패: 재촬영 상태로 전이")
                    await self.wait_ux.retry_request()
                    await self.state_machine.send_event(DialogueEvent.OCR_FAIL)
                    return

            try:
                await self.cam.read_frame()
            except OSError:
                logger.warning("기존 Cam 핸들 프레임 읽기 실패 — 재연결 후 촬영")
                await self.cam.reconnect()
                await self.cam.read_frame()

            self.buffer.activate()

            # Buffer에 프레임 수집할 시간 확보
            capture_task = asyncio.create_task(self.buffer.capture_loop())
            ready = await self.buffer.wait_until_ready(min_frames=3, timeout_sec=5.0)
            if not ready:
                logger.error("촬영 전 프레임 버퍼 준비 실패: frame_count=%d", self.buffer.frame_count)
                self.buffer.deactivate()
                capture_task.cancel()
                try:
                    await capture_task
                except asyncio.CancelledError:
                    pass
                await self.cam.reconnect()
                await self.wait_ux.retry_request()
                await self.state_machine.send_event(DialogueEvent.OCR_FAIL)
                return

            for prompt in ("하나", "둘", "셋"):
                await self._speak(prompt, TTSPriority.HIGH, trace_role="countdown")
                await asyncio.sleep(0.15)

            # 엣지 6-7: Buffer --> Timer --> TTS + Timer --> OCR_Engine
            capture_result = await self.timer.run_countdown()

            # 버퍼 수집 중단
            self.buffer.deactivate()
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass

            if capture_result is None:
                # 실패: 엣지 11
                await self.wait_ux.retry_request()
                await self.state_machine.send_event(DialogueEvent.OCR_FAIL)
                return

            self._drain_quality_guide_tts_queue()
            await self.tts.stop()
            await self._play_capture_shutter()

            # 엣지 7: Timer --> OCR_Engine
            await self.state_machine.send_event(DialogueEvent.BESTSHOT_CAPTURED)
            ocr_result = await self.ocr_engine.process(capture_result.frame)

            await self._handle_ocr_result(ocr_result)

        except Exception:
            logger.exception("촬영 흐름 오류")
            await self.state_machine.send_event(DialogueEvent.OCR_FAIL)

    async def _handle_ocr_result(self, result: OCRResult) -> None:
        """OCR 결과에 따라 성공/실패/확인 요청을 분기한다.

        엣지 10: OCR_Engine --성공--> Drug_Parser --> DB
        엣지 11: OCR_Engine --실패--> Wait_UX --> TTS --> Speaker
        """
        if result.action_required == ActionRequired.PROCEED.value:
            logger.info("OCR 성공: WebSocket ocr_result로 서버 전송")
            payload = result.to_dict()
            self._log_ocr_payload(payload)
            recapture_requested = False
            if self.cloud_chat is not None:
                recapture_requested = await self._handle_cloud_ocr_result(payload)
            else:
                asyncio.create_task(self.drug_parser.send_ocr_result(payload))
            if not recapture_requested:
                await self.state_machine.send_event(DialogueEvent.OCR_SUCCESS)

        elif result.action_required == ActionRequired.NEEDS_CONFIRMATION.value:
            # 신뢰도 미달: 확인 요청
            logger.info("OCR 신뢰도 미달: 사용자 확인 요청")
            await self.wait_ux.confidence_confirm(result.text)
            await self.state_machine.send_event(DialogueEvent.OCR_NEEDS_CONFIRM)

        else:
            # 엣지 11: 실패 재요청
            logger.info("OCR 실패: 재촬영 요청")
            await self.wait_ux.retry_request()
            await self.state_machine.send_event(DialogueEvent.OCR_FAIL)

    def _log_ocr_payload(self, payload: dict[str, Any]) -> None:
        server_payload = to_server_ocr_payload(payload)
        raw_text = server_payload.get("raw_text", "")
        medications = server_payload.get("medications", [])
        out_dir = Path("runtime/ocr")
        out_dir.mkdir(parents=True, exist_ok=True)
        payload_path = out_dir / f"last_ocr_payload_{int(time.time() * 1000)}.json"
        latest_path = out_dir / "last_ocr_payload.json"
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(server_payload, f, ensure_ascii=False, indent=2)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(server_payload, f, ensure_ascii=False, indent=2)
        logger.info(
            "OCR 서버 전송 payload: path=%s latest=%s raw_text_len=%d confidence=%.3f medication_count=%d",
            payload_path,
            latest_path,
            len(raw_text),
            float(server_payload.get("confidence", 0.0)),
            len(medications),
        )
        logger.info("OCR 원문:\n%s", raw_text)
        if medications:
            logger.info("OCR 약 목록: %s", medications)

    async def _handle_cloud_ocr_result(self, payload: dict[str, Any]) -> bool:
        logger.info("Cloud OCR 결과 전송 시작: ws=%s", self.cloud_chat.ws_url)
        self._ocr_exchange_active = True
        delivered = False
        should_recapture = False
        try:
            async for message in self.cloud_chat.send_ocr_result(payload):
                msg_type = message.get("type", "")
                if msg_type in {"ocr_processed", "response"}:
                    delivered = True
                    spoken = CloudChatClient.spoken_text(message)
                    await self._speak_cloud_spoken(
                        spoken,
                        trace_role="ocr_processed",
                        message=message,
                    )
                    if self._should_recapture_after_ocr_response(spoken):
                        should_recapture = True
                    else:
                        self._ocr_confirmation_pending = True
                elif msg_type == "error":
                    logger.warning("Cloud OCR 처리 오류: %s", message.get("message"))
                    await self._tts_queue.put(
                        "약봉투는 읽었지만 서버 저장 확인 응답을 받지 못했습니다."
                    )
        except Exception:
            logger.exception("Cloud OCR 결과 전송 실패")
            await self._tts_queue.put(
                "약봉투는 읽었지만 서버로 결과를 보내지 못했습니다."
            )
        finally:
            if not delivered:
                logger.warning("Cloud OCR WebSocket 응답 없음 — HTTP /api/ocr/analyze fallback 전송")
                asyncio.create_task(self.drug_parser.send_ocr_result(payload))
            self._ocr_exchange_active = False
            if should_recapture:
                await self._start_ocr_recapture()
            await self._flush_deferred_transcriptions()
            return should_recapture

    def _should_recapture_after_ocr_response(self, spoken: str) -> bool:
        return any(marker in spoken for marker in OCR_RECAPTURE_MARKERS)

    async def _start_ocr_recapture(self) -> None:
        logger.info("OCR 서버 재촬영 요청 감지 — 자동 재촬영 시작")
        await self._speak(
            "다시 사진 찍겠습니다.",
            TTSPriority.HIGH,
            trace_role="ocr_recapture",
        )
        await self.state_machine.send_event(DialogueEvent.OCR_FAIL)

    async def _tts_consumer_loop(self) -> None:
        """TTS 큐에서 메시지를 소비하여 TTS --> Speaker로 전달한다.

        모든 Wait_UX / Timer --> TTS --> Speaker 엣지가 이 루프를 통과한다.
        """
        while True:
            try:
                text = await self._tts_queue.get()
                role = self._pending_tts_role or "queued"
                self._pending_tts_role = None
                await self._speak(
                    text,
                    TTSPriority.NORMAL,
                    trace_role=role,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("TTS consumer 오류")

    async def _speak(
        self,
        text: str,
        priority: TTSPriority = TTSPriority.NORMAL,
        *,
        trace_role: str = "generic",
    ) -> None:
        if self.audio_pipeline is not None:
            self.audio_pipeline.remember_tts_text(text)
        self._last_spoken_text = text
        self._last_spoken_at = time.monotonic()
        await self.tts.speak(text, priority, trace_role=trace_role)
        self._last_tts_completed_at = asyncio.get_running_loop().time()
        if self.audio_pipeline is not None and trace_role in {
            "response",
            "identity_check",
            "identity_reset",
        }:
            followup_sec = float(self._stt_cfg.get("followup_listen_timeout_sec", 45.0))
            self.audio_pipeline.extend_dialogue_capture_grace(followup_sec)

    async def _play_capture_shutter(self) -> None:
        logger.info("BestShot 확정 직후 실제 촬영 효과음 재생")
        try:
            await self.speaker.play_capture_shutter()
        except Exception:
            logger.exception("촬영 완료 효과음 재생 실패")

    def _should_suppress_repeated_spoken(self, spoken: str) -> bool:
        current = compact_korean_text(spoken)
        previous = compact_korean_text(self._last_spoken_text)
        if not current or current != previous:
            return False
        elapsed = time.monotonic() - self._last_spoken_at
        repeat_window = float(self._stt_cfg.get("repeat_tts_suppress_sec", 60.0))
        if elapsed > repeat_window:
            return False
        if any(marker in spoken for marker in IDENTITY_CONFIRMATION_MARKERS):
            return True
        return elapsed < float(self._stt_cfg.get("generic_repeat_tts_suppress_sec", 8.0))

    async def _quality_feedback_loop(self) -> None:
        """Buffer 품질 실패 --> Wait_UX --> TTS --> Speaker.

        실시간 프레임 품질 평가는 로그/BestShot 선별에만 쓰고,
        사용자 재요청 음성은 OCR 실패 이후에만 낸다.
        """
        while True:
            try:
                reason = await self._quality_fail_queue.get()
                logger.debug("실시간 품질 안내 음성 생략: %s", reason)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("품질 피드백 루프 오류")

    async def stop(self) -> None:
        """모든 태스크를 정리하고 에이전트를 종료한다."""
        logger.info("=== 로컬 에이전트 종료 ===")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self.state_machine.stop()
        self.buffer.deactivate()
        await self.vad.stop()
        await self.stt.stop_stream()
        if self.audio_pipeline is not None:
            await self.audio_pipeline.stop()
        await self.tts.stop()
        if isinstance(self.drug_parser, HttpDrugParserClient):
            await self.drug_parser.close()
        if isinstance(self.instruction_log, HttpInstructionLogClient):
            await self.instruction_log.close()
        await self.cam.close()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    agent = LocalAgent()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.stop()))

    await agent.start()

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
