from src.edge_node.vad import detect_wake_in_text, strip_wake_from_text


def test_detects_common_odiss_stt_variants():
    for text in (
        "오디스",
        "오디세",
        "오딧스",
        "오티스",
        "오티즈",
        "오지스",
        "보리스",
        "보디스",
        "오 디 스",
        "오디",
        "야",
        "들려?",
        "내 말 들려?",
    ):
        assert detect_wake_in_text(text) is not None


def test_prefix_call_words_strip_without_matching_inside_words():
    assert detect_wake_in_text("야 혈압약 먹어도 돼") is not None
    assert strip_wake_from_text("야 혈압약 먹어도 돼", "오디스") == "혈압약 먹어도 돼"
    assert strip_wake_from_text("오디 혈압약 먹어도 돼", "오디스") == "혈압약 먹어도 돼"
    assert detect_wake_in_text("먹어야 하는 약 알려줘") is None
    assert detect_wake_in_text("오디오가 안 들려") is None


def test_wake_variant_remainder_is_preserved():
    assert detect_wake_in_text("오티스 혈압약 먹어도 돼") is not None
    assert strip_wake_from_text("오티스 혈압약 먹어도 돼", "오티스") == "혈압약 먹어도 돼"
    assert strip_wake_from_text("오 디 스 혈압약 먹어도 돼", "오디스") == "혈압약 먹어도 돼"
