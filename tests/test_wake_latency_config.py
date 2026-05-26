import asyncio

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


def test_gtts_stop_cancels_inflight_synthesis_before_playback() -> None:
    class Speaker:
        def __init__(self) -> None:
            self.played: list[bytes] = []
            self.stops = 0

        async def play(self, pcm: bytes) -> None:
            self.played.append(pcm)

        async def stop(self) -> None:
            self.stops += 1

    async def run_case() -> None:
        speaker = Speaker()
        engine = GTTSEngine(speaker=speaker)
        started = asyncio.Event()
        release = asyncio.Event()

        async def synthesize(text: str) -> bytes:
            started.set()
            await release.wait()
            return b"pcm"

        engine.synthesize = synthesize  # type: ignore[method-assign]

        task = asyncio.create_task(engine.speak("확인하고 있습니다. 잠시만 기다려주세요."))
        await started.wait()
        await engine.stop()
        release.set()
        await task

        assert speaker.stops == 1
        assert speaker.played == []

    asyncio.run(run_case())
