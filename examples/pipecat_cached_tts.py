"""Caching a Pipecat TTS service, by subclassing the one you already use.

    pip install "pipecat-ai[cartesia]" tts-cache

Pipecat's `TTSService` already aggregates the model's tokens into sentences before calling
`run_tts`, which is exactly the seam a cache wants: one sentence in, one clip out. So the
whole integration is one overridden method.

Subclassing rather than wrapping is what keeps this short: the class below *is* the service
in the pipeline, so Pipecat drives its lifecycle — setup, start, stop, cancel, metrics —
and there is no inner service to relay any of that to.

Not run in CI: it needs Pipecat, a transport and vendor keys. The cache itself, and every
rule it enforces, is covered by the test suite.
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.cartesia.tts import CartesiaTTSService

from tts_cache import LocalStorage, SqliteIndex, TTSCache

#: A sentence carrying a number belongs to one caller: spoken once, never stored.
DYNAMIC = re.compile(r"\d")

#: Cached audio is handed back in 20 ms frames, the size the transport wants anyway.
FRAME_MS = 20

VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "")
MODEL = "sonic-3.6"

cache = TTSCache(
    storage=LocalStorage("/var/cache/voice/audio"),
    index=SqliteIndex("/var/cache/voice/index.db"),
    provider="cartesia",
    voice_id=VOICE_ID,
    model=MODEL,
    # PCM carries no header, so the rate belongs in the key: change the transport's rate
    # and you miss, rather than playing a clip back at the wrong speed.
    audio_format="pcm",
    max_storage_size_mb=500,
    time_to_expire=86400 * 30,
    should_cache=lambda spec: not DYNAMIC.search(spec.text),
)


class CachedTTS(CartesiaTTSService):
    """Cartesia, but a sentence it has already spoken never reaches Cartesia again.

    Swap the base class for whichever Pipecat TTS service you use — the body does not
    change, because `super().run_tts` is whatever that service already did.
    """

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        await self.start_ttfb_metrics()
        spec = {"settings": {"sample_rate": self.sample_rate, "num_channels": 1}}

        # `stream`, not `get`: a whole-blob read would make the metric below the time the
        # entire clip took to arrive, which over a bucket is exactly the latency the cache
        # is supposed to remove. This hands over the first frame as soon as it is read.
        cached = await cache.stream(
            text,
            chunk_size=self.sample_rate // 1000 * FRAME_MS * 2,
            frame_bytes=2,  # 16-bit mono: never split a sample
            **spec,
        )
        if cached is not None:
            first = True
            async for chunk in cached:
                if first:
                    await self.stop_ttfb_metrics()
                    first = False
                yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
            return

        # A miss costs exactly what it cost before: Cartesia's own call, its own frames, in
        # its own time. Metrics and usage reporting stay the base class's business.
        rendered: list[bytes] = []
        complete = False
        try:
            async for frame in super().run_tts(text, context_id):
                if isinstance(frame, TTSAudioRawFrame):
                    rendered.append(frame.audio)
                yield frame
            complete = True
        finally:
            # An interruption ends this generator early; a half-spoken sentence must never
            # be stored as if it were the whole thing.
            if complete and rendered:
                await cache.put(text, b"".join(rendered), **spec)


def build_tts() -> CachedTTS:
    """Use this exactly where `CartesiaTTSService(...)` went. The pipeline does not change."""
    return CachedTTS(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(model=MODEL, voice=VOICE_ID),
    )
