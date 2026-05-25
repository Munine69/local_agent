"""
Drug_Parser [약물 정보 파싱: 명칭/용법/일수]

mermaid 노드: Drug_Parser
mermaid 서브그래프: Prescription_Recognition (처방전 정밀 분석 메소드)
mermaid 엣지:
  - OCR_Engine --성공 시 OCR 전송--> Drug_Parser
  - Drug_Parser --> DB

OCR 결과 JSON을 클라우드 ODISS 에이전트에 비동기 전송한다.
타임스탬프 동기화, 경량 JSON만 전송 (원본 이미지 미전송).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class DrugParserConfig:
    endpoint: str = "http://localhost:8000"
    path: str = "/api/ocr/analyze"
    timeout_sec: float = 10.0
    retry_count: int = 3
    retry_backoff_sec: float = 1.0


def to_server_ocr_payload(
    agent_payload: dict[str, Any],
    *,
    speaker_id: str | None = None,
) -> dict[str, Any]:
    """Translate ``OCRResult.to_dict()`` output into the ai-server contract.

    The ai-server ``POST /api/ocr/analyze`` endpoint expects a flat schema::

        {
          "raw_text": str,
          "medications": [{"name": str, "strength": str?, "dosage": str?,
                           "frequency": str?, "timing": str?}, ...],
          "confidence": float,
          "speaker_id": str | None
        }

    The local agent's :class:`OCRResult` uses a nested structure, so this
    function normalises it.
    """
    ocr_results = agent_payload.get("ocr_results") or {}
    text = ocr_results.get("text", agent_payload.get("text", ""))
    confidence = float(
        ocr_results.get(
            "text_confidence_score",
            agent_payload.get("text_confidence_score", 0.0),
        )
    )

    structured = ocr_results.get("structured_data") or agent_payload.get("structured_data") or {}
    raw_meds = structured.get("drugs") or []

    medications: list[dict[str, Any]] = []
    for drug in raw_meds:
        medications.append(
            {
                "name": drug.get("name", ""),
                "strength": drug.get("dosage") or None,
                "dosage": drug.get("dosage") or None,
                "frequency": drug.get("frequency") or None,
                "timing": drug.get("timing") or None,
            }
        )

    return {
        "raw_text": text,
        "medications": medications,
        "confidence": confidence,
        "speaker_id": speaker_id or agent_payload.get("speaker_id"),
    }


def _to_server_ocr_payload(agent_payload: dict[str, Any]) -> dict[str, Any]:
    return to_server_ocr_payload(agent_payload)


class DrugParserClient(ABC):
    """Drug_Parser 클라우드 통신 추상 인터페이스."""

    @abstractmethod
    async def send_ocr_result(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """OCR_Engine --성공--> Drug_Parser: OCR 결과를 클라우드에 전송한다.

        Returns:
            Drug_Parser의 파싱 응답, 또는 실패 시 None.
        """
        ...


class HttpDrugParserClient(DrugParserClient):
    """aiohttp 기반 Drug_Parser 클라우드 클라이언트."""

    def __init__(self, config: DrugParserConfig | None = None) -> None:
        self._config = config or DrugParserConfig()
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def send_ocr_result(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self._session is None:
            await self.open()
        assert self._session is not None

        url = f"{self._config.endpoint}{self._config.path}"
        backoff = self._config.retry_backoff_sec

        server_payload = _to_server_ocr_payload(payload)

        for attempt in range(1, self._config.retry_count + 1):
            try:
                async with self._session.post(url, json=server_payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info("Drug_Parser 전송 성공: %s", resp.status)
                        return data
                    logger.warning(
                        "Drug_Parser 응답 오류 (attempt %d/%d): HTTP %d",
                        attempt, self._config.retry_count, resp.status,
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Drug_Parser 전송 실패 (attempt %d/%d): %s",
                    attempt, self._config.retry_count, exc,
                )

            if attempt < self._config.retry_count:
                await asyncio.sleep(backoff)
                backoff *= 2

        logger.error("Drug_Parser 전송 최종 실패: %d회 시도 소진", self._config.retry_count)
        return None


class StubDrugParserClient(DrugParserClient):
    """테스트/개발용 Drug_Parser 스텁."""

    def __init__(self) -> None:
        self._sent_payloads: list[dict[str, Any]] = []

    async def send_ocr_result(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self._sent_payloads.append(payload)
        logger.info("StubDrugParser: payload 수신 (총 %d건)", len(self._sent_payloads))
        return {"status": "ok", "parsed": True}

    @property
    def sent_payloads(self) -> list[dict[str, Any]]:
        return self._sent_payloads
