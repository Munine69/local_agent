from src.edge_node.tts import GTTSEngine
from src.edge_node.wait_ux import WaitUX


def test_wait_ux_exposes_immediate_templates_for_tts_prewarm() -> None:
    templates = WaitUX.immediate_response_templates()

    assert templates
    assert any("오디스" in text or "말씀" in text for text in templates)


def test_gtts_engine_returns_cached_pcm_without_resynthesizing() -> None:
    engine = GTTSEngine(speaker=object())
    engine._pcm_cache["네, 오디스가 듣고 있어요!"] = b"pcm"

    assert engine._synthesize_sync("네, 오디스가 듣고 있어요!") == b"pcm"
