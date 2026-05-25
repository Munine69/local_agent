"""STT transcript sanitization (prompt echo / hallucination guards)."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# initial_prompt에 넣은 단어가 TV 잡음에서 그대로 튀어나오면 웨이크가 연속 발생한다.
_PROMPT_ECHO_TOKENS = ("처방전", "복용", "가져왔어", "찍어", "사진", "약,")
_SHORT_NOISE_HALLUCINATIONS = {
    "감사합니다",
    "고맙습니다",
    "고맙습니다.",
}
_FILLER_ONLY = {
    "어",
    "음",
    "응",
    "흠",
    "흡",
    "아",
    "네",
}
_NOISE_PHRASE_COMPACT = {
    "흐음",
    "흐으음",
    "으음",
    "네어그",
    "어그",
    "음나이거",
    "나이거",
    "이거",
    "그",
}


def is_prompt_echo_transcript(text: str) -> bool:
    """Whisper가 initial_prompt를 읽어 낸 환각 문장인지 판별."""
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.count("오디스") >= 2:
        return True
    marker_hits = sum(1 for token in _PROMPT_ECHO_TOKENS if token in raw)
    return marker_hits >= 2 and "오디스" in raw


def is_short_noise_hallucination(text: str, duration_sec: float) -> bool:
    """Filter common Whisper polite-phrase hallucinations from subsecond noise."""
    raw = normalize_transcript(text)
    if duration_sec > 1.0:
        return False
    return raw in _SHORT_NOISE_HALLUCINATIONS


def normalize_transcript(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_filler_only_transcript(text: str) -> bool:
    """Return True for backchannels/noise that should not start a cloud turn."""
    raw = normalize_transcript(text)
    if re.fullmatch(r"\[[^\]]+\]", raw):
        return True
    compact = compact_korean_text(text)
    if not compact:
        return True
    return compact in _FILLER_ONLY or compact in _NOISE_PHRASE_COMPACT


def compact_korean_text(text: str) -> str:
    """Normalize text for short Korean echo comparisons."""
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", normalize_transcript(text)).lower()


def is_tts_echo_transcript(
    transcript: str,
    spoken_texts: list[str],
    *,
    min_chars: int = 6,
    ratio_threshold: float = 0.72,
) -> bool:
    """Return True when STT looks like a replay of recent local TTS."""
    compact_transcript = compact_korean_text(transcript)
    if len(compact_transcript) < min_chars:
        return False

    for spoken in spoken_texts:
        compact_spoken = compact_korean_text(spoken)
        if len(compact_spoken) < min_chars:
            continue
        if compact_transcript in compact_spoken:
            return True
        ratio = SequenceMatcher(None, compact_transcript, compact_spoken).ratio()
        if ratio >= ratio_threshold:
            return True
    return False
