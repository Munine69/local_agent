from src.edge_node.stt_sanitize import (
    is_filler_only_transcript,
    is_short_noise_hallucination,
    is_tts_echo_transcript,
)


def test_tts_echo_transcript_detects_agent_replay() -> None:
    spoken = ["마지막 대화 후 시간이 지나서 다시 확인할게요. 지금 말씀하시는 분이 김영수님 본인이 맞으신가요?"]

    assert is_tts_echo_transcript("지금 말씀하시는 분이 김영수님 본인이 맞으신가요", spoken)


def test_tts_echo_transcript_allows_short_user_confirmation() -> None:
    spoken = ["마지막 대화 후 시간이 지나서 다시 확인할게요. 지금 말씀하시는 분이 김영수님 본인이 맞으신가요?"]

    assert not is_tts_echo_transcript("어 맞아", spoken)


def test_tts_echo_transcript_allows_distinct_user_request() -> None:
    spoken = ["마지막 대화 후 시간이 지나서 다시 확인할게요. 지금 말씀하시는 분이 김영수님 본인이 맞으신가요?"]

    assert not is_tts_echo_transcript("김영수 약", spoken)


def test_short_noise_hallucination_filters_polite_phrase() -> None:
    assert is_short_noise_hallucination("고맙습니다.", 0.72)


def test_short_noise_hallucination_allows_longer_real_phrase() -> None:
    assert not is_short_noise_hallucination("고맙습니다.", 1.4)


def test_filler_only_transcript_filters_backchannels() -> None:
    assert is_filler_only_transcript("흡")
    assert is_filler_only_transcript("음")
    assert is_filler_only_transcript("어")
    assert is_filler_only_transcript("흐음")
    assert is_filler_only_transcript("네 어 그")
    assert is_filler_only_transcript("음 나 이거")
    assert is_filler_only_transcript("[숨소리]")


def test_filler_only_transcript_allows_meaningful_text() -> None:
    assert not is_filler_only_transcript("아니 나는 이재석이야")
