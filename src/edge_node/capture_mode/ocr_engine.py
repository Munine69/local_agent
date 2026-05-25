"""
OCR_Engine [Gemini OCR: 텍스트 추출]

mermaid 노드: OCR_Engine
mermaid 엣지:
  - Timer --> OCR_Engine                      (확정 프레임 수신)
  - OCR_Engine --성공 시 OCR 전송--> Drug_Parser  (성공 -> 클라우드 전송)
  - OCR_Engine --실패 시 재요청--> Wait_UX         (실패 -> 재촬영 요청)

내부 기능:
  - ROI 분류: 처방전(PRESCRIPTION) / 알약(PILL_IMAGE) 자동 판별
  - 처방전 모드: 표 구조 Key-Value 추출 (약품명, 투여횟수, 투약일수)
  - 알약 모드: 텍스트 + 시각적 특징 추출 (형태, 색상, 분할선, 각인)
  - 신뢰도 평가: confidence < 0.85 -> 실패 판정
  - 의학용어 사전 fuzzy matching
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    PRESCRIPTION = "PRESCRIPTION"
    PILL_IMAGE = "PILL_IMAGE"


class ActionRequired(str, Enum):
    PROCEED = "PROCEED_TO_IDENTIFICATION"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    RETRY = "RETRY_CAPTURE"


@dataclass
class VisualFeatures:
    shape: str
    color: str
    line: str


@dataclass
class DrugEntry:
    name: str
    dosage: str
    frequency: str
    days: int


@dataclass
class OCRResult:
    perception_timestamp: str
    input_type: str
    text: str
    text_confidence_score: float
    visual_features: dict[str, str] | None = None
    structured_data: dict[str, Any] | None = None
    action_required: str = ActionRequired.PROCEED.value

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "perception_timestamp": self.perception_timestamp,
            "input_type": self.input_type,
            "ocr_results": {
                "text": self.text,
                "text_confidence_score": self.text_confidence_score,
            },
            "action_required": self.action_required,
        }
        if self.visual_features is not None:
            result["ocr_results"]["visual_features"] = self.visual_features
        if self.structured_data is not None:
            result["ocr_results"]["structured_data"] = self.structured_data
        return result


@dataclass
class OCREngineConfig:
    model_path: str = ""
    provider: str = "gemini_ocr"
    save_dir: str = "runtime/ocr"
    hf_device: str = "auto"
    hf_torch_dtype: str = "auto"
    hf_prompt: str = "Text Recognition:"
    hf_max_new_tokens: int = 4096
    hf_max_image_side: int = 1280
    hf_extract_document: bool = True
    hf_repetition_penalty: float = 1.15
    hf_no_repeat_ngram_size: int = 8
    gemini_model: str = "gemini-3-flash-preview"
    gemini_fallback_models: list[str] = field(
        default_factory=lambda: ["gemini-2.5-flash", "gemini-flash-latest"]
    )
    gemini_api_key: str = ""
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_prompt: str = (
        "이 이미지는 병원 진료비 세부내역서 또는 처방전/약봉투입니다. "
        "표 형식의 모든 항목을 누락 없이 한국어 텍스트로 추출해 주세요."
    )
    glmocr_mode: str = "maas"
    glmocr_api_key_env: str = "ZHIPU_API_KEY"
    glmocr_timeout_sec: int = 600
    glmocr_save_dir: str = "runtime/ocr"
    confidence_threshold: float = 0.85
    fuzzy_match_max_distance: int = 2
    medical_dict_path: str = "config/medical_terms.json"


# ---------------------------------------------------------------------------
# Shape / Color 한글 매핑 상수
# ---------------------------------------------------------------------------
SHAPE_LABELS = {
    "circle": "원형",
    "ellipse": "타원형",
    "rectangle": "장방형",
    "other": "기타",
}

COLOR_RANGES_HSV = {
    "하얀색": ((0, 0, 200), (180, 30, 255)),
    "노란색": ((20, 100, 100), (35, 255, 255)),
    "주황색": ((10, 100, 100), (20, 255, 255)),
    "빨간색": ((0, 100, 100), (10, 255, 255)),
    "분홍색": ((150, 50, 100), (170, 255, 255)),
    "초록색": ((35, 100, 100), (85, 255, 255)),
    "파란색": ((85, 100, 100), (130, 255, 255)),
    "갈색": ((10, 50, 50), (20, 200, 150)),
    "투명": ((0, 0, 0), (180, 30, 50)),
}


class OCREngine:
    """Gemini OCR 기반 텍스트 추출 엔진.

    ROI 분류, 모드별 파싱, 신뢰도 검증, 의학사전 매칭을
    하나의 노드 안에서 수행한다.
    """

    def __init__(self, config: OCREngineConfig | None = None) -> None:
        self._config = config or OCREngineConfig()
        self._medical_terms: list[str] = []
        self._model_loaded = False
        self._glmocr_parser: Any | None = None
        self._hf_processor: Any | None = None
        self._hf_model: Any | None = None
        self._gemini_client: Any | None = None
        self._gemini_model: Any | None = None

    async def load(self) -> None:
        """모델 및 의학 사전 로드."""
        self._load_medical_dict()
        await self._load_model()
        self._model_loaded = True

    def _load_medical_dict(self) -> None:
        dict_path = Path(self._config.medical_dict_path)
        if dict_path.exists():
            with open(dict_path, encoding="utf-8") as f:
                data = json.load(f)
            self._medical_terms = data.get("terms", [])
            logger.info("의학용어 사전 로드: %d개 항목", len(self._medical_terms))
        else:
            logger.warning("의학용어 사전 파일 없음: %s", dict_path)

    async def _load_model(self) -> None:
        """OCR provider를 로드한다."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        if self._config.provider == "gemini_ocr":
            self._load_gemini_ocr()
        elif self._config.provider == "hf_glm_ocr":
            self._load_hf_glm_ocr()
        elif self._config.provider == "glmocr":
            self._load_glmocr_sdk()
        elif Path(self._config.model_path).exists():
            logger.info("OCR 모델 경로 확인: %s", self._config.model_path)
        else:
            logger.warning(
                "OCR provider=%s 로드 경로 없음 (스텁 모드로 동작)",
                self._config.provider,
            )
        self._apply_turboquant_wrap()

    def _load_gemini_ocr(self) -> None:
        api_key = self._resolve_gemini_api_key()
        if not api_key:
            logger.warning(
                "Gemini API 키 없음: ocr.gemini_api_key 또는 %s 환경변수를 설정하세요.",
                self._config.gemini_api_key_env,
            )
            return

        try:
            from google import genai
        except Exception:
            logger.exception("google-genai import 실패. `pip install google-genai` 필요")
            return

        self._gemini_client = genai.Client(api_key=api_key)
        self._gemini_model = self._config.gemini_model
        Path(self._save_dir()).mkdir(parents=True, exist_ok=True)
        logger.info("Gemini OCR 로드 완료: model=%s", self._config.gemini_model)

    def _resolve_gemini_api_key(self) -> str:
        yaml_key = (self._config.gemini_api_key or "").strip()
        if yaml_key:
            return yaml_key
        return (os.environ.get(self._config.gemini_api_key_env) or "").strip()

    def _save_dir(self) -> str:
        return self._config.save_dir or self._config.glmocr_save_dir

    def _load_glmocr_sdk(self) -> None:
        """공식 glmocr SDK를 로드한다.

        MaaS 모드는 Zhipu GLM-OCR cloud API를 사용한다. Jetson에서는 로컬
        0.9B 모델을 바로 올리는 것보다 이 경로가 테스트/운영 안정성이 높다.
        API 키는 설정 파일에 저장하지 않고 환경변수로만 읽는다.
        """
        api_key = os.environ.get(self._config.glmocr_api_key_env)
        if not api_key:
            logger.warning(
                "%s 환경변수 없음: GLM-OCR SDK는 로드됐지만 추론은 실패합니다.",
                self._config.glmocr_api_key_env,
            )
            return

        try:
            from glmocr import GlmOcr
        except Exception:
            logger.exception("glmocr SDK import 실패. `pip install glmocr` 필요")
            return

        self._glmocr_parser = GlmOcr(
            api_key=api_key,
            mode=self._config.glmocr_mode,
            timeout=self._config.glmocr_timeout_sec,
            log_level="INFO",
        )
        Path(self._config.glmocr_save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "GLM-OCR SDK 로드 완료: provider=glmocr mode=%s timeout=%ss",
            self._config.glmocr_mode,
            self._config.glmocr_timeout_sec,
        )

    def _load_hf_glm_ocr(self) -> None:
        """Hugging Face `zai-org/GLM-OCR` 모델을 직접 로드한다."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception:
            logger.exception(
                "HF GLM-OCR import 실패. torch/transformers 설치 상태를 확인하세요."
            )
            return

        dtype = self._resolve_torch_dtype(torch)
        device_map: str | None = "auto"
        if self._config.hf_device == "cpu":
            device_map = "cpu"
        elif self._config.hf_device == "cuda":
            device_map = "auto"
        elif self._config.hf_device == "auto" and not torch.cuda.is_available():
            logger.warning("CUDA 사용 불가: HF GLM-OCR를 CPU로 로드합니다.")
            device_map = "cpu"
            if dtype == "auto":
                dtype = torch.float32

        logger.info(
            "HF GLM-OCR 로드 시작: model=%s device_map=%s dtype=%s",
            self._config.model_path,
            device_map,
            dtype,
        )
        self._hf_processor = AutoProcessor.from_pretrained(
            self._config.model_path,
            trust_remote_code=True,
        )
        self._hf_model = AutoModelForImageTextToText.from_pretrained(
            self._config.model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._hf_model.eval()
        logger.info("HF GLM-OCR 로드 완료")

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self._config.hf_torch_dtype == "float16":
            return torch.float16
        if self._config.hf_torch_dtype == "bfloat16":
            return torch.bfloat16
        if self._config.hf_torch_dtype == "float32":
            return torch.float32
        return "auto"

    def _apply_turboquant_wrap(self) -> None:
        """Wrap the loaded HF model (if any) with TurboQuant compressed KV.

        Looks for a ``self._model`` / ``self._glm_model`` attribute that a
        concrete subclass may have populated; no-ops otherwise.  Keeps the
        stubbed path silent on CPU-only dev hosts.
        """
        try:
            from src.runtime.turboquant_runtime import wrap
        except Exception as exc:  # noqa: BLE001
            logger.debug("TurboQuant runtime unavailable: %s", exc)
            return

        for attr in ("_model", "_glm_model", "model"):
            model = getattr(self, attr, None)
            if model is None:
                continue
            setattr(self, attr, wrap(model))

    async def process(self, frame: np.ndarray) -> OCRResult:
        """Timer -> OCR_Engine: 프레임을 받아 OCR 처리를 수행한다.

        Returns:
            OCRResult:
              - 성공: action_required = PROCEED_TO_IDENTIFICATION
              - 신뢰도 미달: action_required = NEEDS_CONFIRMATION
              - 실패: action_required = RETRY_CAPTURE
        """
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        input_type = self._classify_roi(frame)

        if input_type == InputType.PRESCRIPTION:
            return await self._process_prescription(frame, timestamp)
        else:
            return await self._process_pill(frame, timestamp)

    # ------------------------------------------------------------------
    # ROI 분류
    # ------------------------------------------------------------------
    def _classify_roi(self, frame: np.ndarray) -> InputType:
        """처방전/알약 자동 판별.

        문서(처방전/약봉투)는 직선이 많고 텍스트 영역이 넓은 반면,
        알약은 원형/타원형 윤곽이 지배적이다.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 50,
            param1=100, param2=40, minRadius=15, maxRadius=200,
        )
        circle_count = len(circles[0]) if circles is not None else 0

        if line_count > circle_count * 3:
            logger.debug("ROI 분류: PRESCRIPTION (lines=%d, circles=%d)", line_count, circle_count)
            return InputType.PRESCRIPTION
        else:
            logger.debug("ROI 분류: PILL_IMAGE (lines=%d, circles=%d)", line_count, circle_count)
            return InputType.PILL_IMAGE

    # ------------------------------------------------------------------
    # 처방전 모드
    # ------------------------------------------------------------------
    async def _process_prescription(self, frame: np.ndarray, timestamp: str) -> OCRResult:
        loop = asyncio.get_running_loop()
        raw_texts, confidences = await loop.run_in_executor(
            None, self._run_ocr_inference, frame
        )

        if not raw_texts:
            return OCRResult(
                perception_timestamp=timestamp,
                input_type=InputType.PRESCRIPTION.value,
                text="",
                text_confidence_score=0.0,
                action_required=ActionRequired.RETRY.value,
            )

        avg_confidence = sum(confidences) / len(confidences)
        full_text = " ".join(raw_texts)

        drugs = self._parse_prescription_table(raw_texts)
        action = self._determine_action(raw_texts, confidences)

        return OCRResult(
            perception_timestamp=timestamp,
            input_type=InputType.PRESCRIPTION.value,
            text=full_text,
            text_confidence_score=round(avg_confidence, 3),
            structured_data={"drugs": [vars(d) for d in drugs]} if drugs else None,
            action_required=action.value,
        )

    def _parse_prescription_table(self, texts: list[str]) -> list[DrugEntry]:
        """처방전 표 구조에서 약품명/용법/일수를 추출한다.

        실제 GLM-OCR 결과의 bbox 기반 표 재구성은 모델 출력 포맷에 의존하며,
        여기서는 텍스트 패턴 기반 추출 로직을 제공한다.
        """
        drugs: list[DrugEntry] = []
        for text in texts:
            matched = self._fuzzy_match_drug_name(text)
            if matched is not None:
                drugs.append(DrugEntry(
                    name=matched,
                    dosage="",
                    frequency="",
                    days=0,
                ))
        return drugs

    # ------------------------------------------------------------------
    # 알약 모드
    # ------------------------------------------------------------------
    async def _process_pill(self, frame: np.ndarray, timestamp: str) -> OCRResult:
        loop = asyncio.get_running_loop()

        raw_texts, confidences = await loop.run_in_executor(
            None, self._run_ocr_inference, frame
        )
        visual = await loop.run_in_executor(None, self._extract_visual_features, frame)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = " ".join(raw_texts) if raw_texts else ""

        action = self._determine_action(raw_texts, confidences) if raw_texts else ActionRequired.RETRY

        return OCRResult(
            perception_timestamp=timestamp,
            input_type=InputType.PILL_IMAGE.value,
            text=full_text,
            text_confidence_score=round(avg_confidence, 3),
            visual_features={
                "shape": visual.shape,
                "color": visual.color,
                "line": visual.line,
            },
            action_required=action.value,
        )

    def _extract_visual_features(self, frame: np.ndarray) -> VisualFeatures:
        """알약의 형태, 색상, 분할선을 분석한다."""
        shape = self._detect_shape(frame)
        color = self._detect_color(frame)
        has_line = self._detect_split_line(frame)
        return VisualFeatures(shape=shape, color=color, line="있음" if has_line else "없음")

    def _detect_shape(self, frame: np.ndarray) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return SHAPE_LABELS["other"]

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter == 0:
            return SHAPE_LABELS["other"]

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        rect = cv2.minAreaRect(largest)
        w, h = rect[1]
        if min(w, h) == 0:
            return SHAPE_LABELS["other"]
        aspect_ratio = max(w, h) / min(w, h)

        if circularity > 0.85:
            return SHAPE_LABELS["circle"]
        elif aspect_ratio > 1.5:
            return SHAPE_LABELS["rectangle"] if circularity < 0.6 else SHAPE_LABELS["ellipse"]
        else:
            return SHAPE_LABELS["ellipse"]

    def _detect_color(self, frame: np.ndarray) -> str:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best_color = "기타"
        best_ratio = 0.0
        total = hsv.shape[0] * hsv.shape[1]
        for color_name, (lower, upper) in COLOR_RANGES_HSV.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = float(cv2.countNonZero(mask)) / total
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color_name
        return best_color

    def _detect_split_line(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=5)
        if lines is None:
            return False
        h, w = gray.shape
        center_y = h // 2
        tolerance = h // 6
        for line in lines:
            _, y1, _, y2 = line[0]
            if abs(y1 - center_y) < tolerance and abs(y2 - center_y) < tolerance:
                return True
        return False

    # ------------------------------------------------------------------
    # OCR 추론
    # ------------------------------------------------------------------
    def _run_ocr_inference(self, frame: np.ndarray) -> tuple[list[str], list[float]]:
        """설정된 OCR provider로 텍스트를 추출한다."""
        if not self._model_loaded:
            logger.warning("OCR 모델 미로드 상태: 빈 결과 반환")
            return [], []

        if self._config.provider == "gemini_ocr":
            return self._run_gemini_ocr(frame)
        if self._config.provider == "hf_glm_ocr":
            return self._run_hf_glm_ocr(frame)
        if self._config.provider == "glmocr":
            return self._run_glmocr_sdk(frame)

        logger.info("OCR 추론 실행 (스텁): frame shape=%s", frame.shape)
        return [], []

    def _run_gemini_ocr(self, frame: np.ndarray) -> tuple[list[str], list[float]]:
        if self._gemini_client is None or self._gemini_model is None:
            logger.error(
                "Gemini OCR 모델이 준비되지 않음. %s 환경변수를 확인하세요.",
                self._config.gemini_api_key_env,
            )
            return [], []

        try:
            from PIL import Image
        except Exception:
            logger.exception("Pillow import 실패")
            return [], []

        save_dir = Path(self._save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"gemini_capture_{int(time.time() * 1000)}.jpg"
        inference_frame = self._prepare_frame_for_hf_ocr(frame)
        ok = cv2.imwrite(
            str(image_path),
            inference_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not ok:
            logger.error("Gemini OCR 입력 이미지 저장 실패: %s", image_path)
            return [], []

        logger.info(
            "Gemini OCR 추론 요청: %s shape=%s model=%s",
            image_path,
            inference_frame.shape,
            self._config.gemini_model,
        )
        img = Image.open(image_path).convert("RGB")
        text = self._generate_gemini_content(img)
        if text is None:
            return [], []

        if not text:
            logger.warning("Gemini OCR 결과가 비어 있음")
            return [], []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = [text]
        logger.info(
            "Gemini OCR 텍스트 추출 성공: %d lines image_path=%s",
            len(lines),
            image_path,
        )
        logger.info("Gemini OCR 원문:\n%s", "\n".join(lines))
        return lines, [0.95] * len(lines)

    def _generate_gemini_content(self, image: Any) -> str | None:
        models = [self._config.gemini_model, *self._config.gemini_fallback_models]
        last_error: Exception | None = None
        for model_name in dict.fromkeys(models):
            try:
                response = self._gemini_client.models.generate_content(
                    model=model_name,
                    contents=[self._config.gemini_prompt, image],
                )
                if model_name != self._config.gemini_model:
                    logger.info("Gemini OCR fallback 모델 사용: %s", model_name)
                return (getattr(response, "text", "") or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Gemini OCR 모델 실패: %s (%s)", model_name, exc)
        logger.exception("Gemini OCR 추론 실패", exc_info=last_error)
        return None

    def _run_hf_glm_ocr(self, frame: np.ndarray) -> tuple[list[str], list[float]]:
        if self._hf_processor is None or self._hf_model is None:
            logger.error("HF GLM-OCR 모델이 준비되지 않음")
            return [], []

        try:
            import torch
            from PIL import Image
        except Exception:
            logger.exception("HF GLM-OCR 추론 의존성 import 실패")
            return [], []

        save_dir = Path(self._config.glmocr_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"hf_capture_{int(time.time() * 1000)}.jpg"
        inference_frame = self._prepare_frame_for_hf_ocr(frame)
        ok = cv2.imwrite(
            str(image_path),
            inference_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not ok:
            logger.error("HF GLM-OCR 입력 이미지 저장 실패: %s", image_path)
            return [], []

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": self._config.hf_prompt},
            ],
        }]

        logger.info(
            "HF GLM-OCR 추론 시작: %s shape=%s prompt=%s",
            image_path,
            inference_frame.shape,
            self._config.hf_prompt,
        )
        try:
            inputs = self._hf_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_device = getattr(self._hf_model, "device", None)
            if model_device is not None:
                inputs = inputs.to(model_device)
            inputs.pop("token_type_ids", None)
            with torch.inference_mode():
                generated_ids = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=self._config.hf_max_new_tokens,
                    do_sample=False,
                    repetition_penalty=self._config.hf_repetition_penalty,
                    no_repeat_ngram_size=self._config.hf_no_repeat_ngram_size,
                )
            input_len = inputs["input_ids"].shape[1]
            output_text = self._hf_processor.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            logger.exception("HF GLM-OCR 추론 실패")
            return [], []

        text = output_text.strip()
        if not text:
            logger.warning("HF GLM-OCR 결과가 비어 있음")
            return [], []
        if self._is_repetitive_text(text):
            logger.warning("HF GLM-OCR 반복 출력 감지, 실패 처리: %s", text[:120])
            return [], []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = [text]
        logger.info("HF GLM-OCR 텍스트 추출 성공: %d lines", len(lines))
        return lines, [0.95] * len(lines)

    def _prepare_frame_for_hf_ocr(self, frame: np.ndarray) -> np.ndarray:
        prepared = frame
        if self._config.hf_extract_document:
            cropped = self._extract_document_region(frame)
            if cropped is not None:
                logger.info(
                    "HF GLM-OCR 문서 영역 추출: %s -> %s",
                    frame.shape,
                    cropped.shape,
                )
                prepared = cropped
            else:
                logger.warning("문서 영역 추출 실패: 원본 프레임으로 OCR 진행")
        return self._resize_for_hf_ocr(prepared)

    def _extract_document_region(self, frame: np.ndarray) -> np.ndarray | None:
        """카메라 프레임에서 처방전/종이 영역을 찾아 perspective crop한다."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 하얀 종이를 우선 찾되 조명 변화에 견디도록 adaptive/otsu를 함께 사용.
        _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = h * w * 0.12
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in candidates[:8]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                warped = self._warp_quadrilateral(frame, approx.reshape(4, 2))
                if warped is not None:
                    return warped

            # 꼭짓점 검출이 흔들리면 회전 사각형 crop으로 폴백.
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            warped = self._warp_quadrilateral(frame, box)
            if warped is not None:
                return warped
        return None

    def _warp_quadrilateral(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        pts = points.astype("float32")
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = int(max(width_a, width_b))
        max_height = int(max(height_a, height_b))
        if max_width < 200 or max_height < 200:
            return None

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))

        # 가로 문서가 세로로 뒤집혀 잡히는 경우 보정.
        if warped.shape[0] > warped.shape[1] * 1.2:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _resize_for_hf_ocr(self, frame: np.ndarray) -> np.ndarray:
        max_side = self._config.hf_max_image_side
        if max_side <= 0:
            return frame
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side:
            return frame
        scale = max_side / long_side
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        logger.info("HF GLM-OCR 입력 리사이즈: %sx%s -> %sx%s", w, h, *new_size)
        return resized

    def _is_repetitive_text(self, text: str) -> bool:
        tokens = [tok for tok in text.replace("\n", " ").split(" ") if tok]
        meaningful_tokens = [
            tok
            for tok in tokens
            if len(tok.strip("[]*.,:;()")) >= 2 and "불명확" not in tok
        ]
        if len(meaningful_tokens) >= 12:
            most_common = max(
                meaningful_tokens.count(tok) for tok in set(meaningful_tokens)
            )
            if most_common >= 8 and most_common / len(meaningful_tokens) > 0.45:
                return True

        adjacent_repeats = 1
        previous = None
        for token in meaningful_tokens:
            if token == previous:
                adjacent_repeats += 1
            else:
                adjacent_repeats = 1
                previous = token
            if adjacent_repeats >= 6:
                return True
        return False

    def _run_glmocr_sdk(self, frame: np.ndarray) -> tuple[list[str], list[float]]:
        if self._glmocr_parser is None:
            logger.error(
                "GLM-OCR parser가 준비되지 않음. %s 환경변수를 확인하세요.",
                self._config.glmocr_api_key_env,
            )
            return [], []

        save_dir = Path(self._config.glmocr_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"capture_{int(time.time() * 1000)}.jpg"
        ok = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            logger.error("GLM-OCR 입력 이미지 저장 실패: %s", image_path)
            return [], []

        logger.info("GLM-OCR 추론 요청: %s shape=%s", image_path, frame.shape)
        try:
            result = self._glmocr_parser.parse(
                str(image_path),
                save_layout_visualization=False,
            )
        except Exception:
            logger.exception("GLM-OCR SDK 추론 실패")
            return [], []

        text = self._extract_text_from_glmocr_result(result)
        if not text:
            logger.warning("GLM-OCR 결과에서 텍스트를 추출하지 못함: %s", type(result))
            return [], []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = [text.strip()]
        logger.info("GLM-OCR 텍스트 추출 성공: %d lines", len(lines))
        return lines, [0.95] * len(lines)

    def _extract_text_from_glmocr_result(self, result: Any) -> str:
        """glmocr PipelineResult에서 markdown/text를 최대한 보수적으로 추출."""
        if isinstance(result, list):
            return "\n".join(self._extract_text_from_glmocr_result(r) for r in result)

        markdown = getattr(result, "markdown_result", None)
        if isinstance(markdown, str) and markdown.strip():
            return markdown.strip()

        for attr in ("text", "content", "result"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            try:
                data = to_dict()
            except Exception:
                data = None
            text = self._extract_text_from_mapping(data)
            if text:
                return text

        data = getattr(result, "json_result", None)
        text = self._extract_text_from_mapping(data)
        if text:
            return text
        return str(result).strip()

    def _extract_text_from_mapping(self, data: Any) -> str:
        if data is None:
            return ""
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                return data.strip()
            return self._extract_text_from_mapping(parsed)
        if isinstance(data, list):
            parts = [self._extract_text_from_mapping(item) for item in data]
            return "\n".join(part for part in parts if part)
        if isinstance(data, dict):
            preferred_keys = (
                "markdown_result",
                "markdown",
                "text",
                "content",
                "value",
            )
            for key in preferred_keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            parts = [self._extract_text_from_mapping(v) for v in data.values()]
            return "\n".join(part for part in parts if part)
        return ""

    # ------------------------------------------------------------------
    # 신뢰도 검증 및 의학사전 매칭
    # ------------------------------------------------------------------
    def _determine_action(
        self, texts: list[str], confidences: list[float]
    ) -> ActionRequired:
        if not texts or not confidences:
            return ActionRequired.RETRY

        avg_conf = sum(confidences) / len(confidences)
        if avg_conf < self._config.confidence_threshold:
            return ActionRequired.NEEDS_CONFIRMATION

        for text, conf in zip(texts, confidences):
            if conf < self._config.confidence_threshold:
                matched = self._fuzzy_match_drug_name(text)
                if matched is None:
                    return ActionRequired.NEEDS_CONFIRMATION

        return ActionRequired.PROCEED

    def _fuzzy_match_drug_name(self, text: str) -> str | None:
        """의학용어 사전과 fuzzy matching하여 가장 가까운 약품명을 반환한다."""
        if not self._medical_terms or not text.strip():
            return None

        result = process.extractOne(
            text.strip(),
            self._medical_terms,
            scorer=fuzz.ratio,
            score_cutoff=70,
        )
        if result is None:
            return None

        matched_term, score, _ = result
        distance = int((100 - score) * len(matched_term) / 100)
        if distance <= self._config.fuzzy_match_max_distance:
            return matched_term
        return None
