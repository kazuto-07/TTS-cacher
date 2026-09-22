"""Caching inside a LiveKit agent's ``tts_node``.

    pip install "livekit-agents[deepgram,elevenlabs,openai,silero]" tts-cache
    python examples/livekit_tts_node.py dev

``tts_node`` is the point in LiveKit's pipeline where the model's text becomes audio, so it
is where a cache belongs: the sentences an agent repeats — the greeting, the hold line,
"sorry, I didn't catch that" — are served from disk in a millisecond or two, and everything
else goes to the vendor exactly as before.

The node receives the reply as a stream of tokens, so this splits it into sentences first.
That matters more than caching whole replies: in a streaming agent the caller waits for the
*first* sentence, and the first sentence of a turn is the most repeated text in the whole
product.

The audio here is raw PCM at the TTS's own sample rate, which has no header to describe it,
so the sample rate and channel count go into the cache key along with the voice. Change
voice, model or rate and the key changes with it — an old clip can never be played in a new
voice.

Not run in CI: it needs a LiveKit project and vendor keys. The pieces that do not need
either — the key, the store, the index, eviction — are covered by the test suite.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, ModelSettings, room_io, tokenize, utils
from livekit.plugins import deepgram, elevenlabs, openai, silero

from tts_cache import LocalStorage, SqliteIndex, TTSCache

load_dotenv(".env.local")

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
TTS_MODEL = "eleven_turbo_v2_5"

#: Sentences carrying a number are specific to one caller: spoken once, never stored.
DYNAMIC = re.compile(r"\d")

#: How much cached PCM to hand over per frame. 20 ms is what the room expects anyway.
FRAME_MS = 20


def build_cache() -> TTSCache:
    """One cache for the whole worker, shared by every session it runs.

    On a machine that handles many calls this is the whole point: the greeting is rendered
    once, for the first caller of the day, and every session after that reads it from disk.
    Point `index` at Postgres instead and the same holds across every worker in the fleet.
    """
    return TTSCache(
        storage=LocalStorage("/var/cache/voice/audio"),
        index=SqliteIndex("/var/cache/voice/index.db"),
        provider="elevenlabs",
        voice_id=VOICE_ID,
        model=TTS_MODEL,
        audio_format="pcm",
        max_storage_size_mb=500,
        time_to_expire=86400 * 30,
        should_cache=lambda spec: not DYNAMIC.search(spec.text),
    )


class CachedAgent(Agent):
    """An agent that speaks from the cache whenever it can."""

    def __init__(self, cache: TTSCache, tts: elevenlabs.TTS) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant for Northwind Support. Keep replies to "
                "one or two short sentences, with no markdown or symbols."
            ),
            tts=tts,
        )
        self._cache = cache
        # Read from the TTS itself rather than hardcoded: a plugin that returns 24 kHz and
        # one that returns 48 kHz must not share a cache entry.
        self._sample_rate = tts.sample_rate
        self._num_channels = tts.num_channels
        self._frame_bytes = self._sample_rate // 1000 * FRAME_MS * 2 * self._num_channels
        self._pcm = {"sample_rate": self._sample_rate, "num_channels": self._num_channels}
        self._sentences = tokenize.basic.SentenceTokenizer()

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Speaks the reply a sentence at a time, from the cache where possible."""
        stream = self._sentences.stream()

        async def feed() -> None:
            async for chunk in text:
                stream.push_text(chunk)
            stream.end_input()

        feeding = asyncio.create_task(feed())
        try:
            async for data in stream:
                async for frame in self._say(data.token, model_settings):
                    yield frame
        finally:
            # A barge-in closes this generator mid-sentence; the feeder has to go with it.
            await utils.aio.cancel_and_wait(feeding)
            await stream.aclose()

    async def _say(
        self, sentence: str, model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """One sentence: cached bytes, or the vendor's own stream on the way through."""

        async def render() -> AsyncIterable[bytes]:
            # The default node is what would have run without this override, so a miss
            # costs exactly what it used to. One sentence per call keeps the clip
            # boundaries clean, at the price of a stream per sentence.
            async for frame in Agent.default.tts_node(self, _once(sentence), model_settings):
                yield bytes(frame.data)

        async for pcm in self._cache.stream_or_generate(
            sentence,
            generator_fn=render,
            chunk_size=self._frame_bytes,
            # A chunk that ends mid-sample would make the frame below claim a sample count
            # it does not have, and every later frame would be one byte out of phase. The
            # vendor is under no obligation to chunk on a sample boundary, so say so here.
            frame_bytes=2 * self._num_channels,
            settings=self._pcm,
        ):
            yield rtc.AudioFrame(
                pcm,
                self._sample_rate,
                self._num_channels,
                len(pcm) // (2 * self._num_channels),
            )


async def _once(sentence: str) -> AsyncIterable[str]:
    yield sentence


server = AgentServer()


@server.rtc_session(agent_name="cached-agent")
async def cached_agent(ctx: agents.JobContext) -> None:
    cache = build_cache()
    tts = elevenlabs.TTS(voice_id=VOICE_ID, model=TTS_MODEL)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-5.2-mini"),
        tts=tts,
        vad=silero.VAD.load(),
    )

    try:
        await session.start(
            room=ctx.room,
            agent=CachedAgent(cache, tts),
            room_options=room_io.RoomOptions(),
        )
        # A fixed greeting: the first caller pays for it, nobody else does.
        await session.say("Hi, you've reached Northwind Support. How can I help?")
        await session.generate_reply(instructions="Wait for the caller to speak.")
    finally:
        stats = cache.stats
        ctx.log_context_fields = {"tts_hit_rate": round(stats.hit_rate, 3)}
        await cache.aclose()


if __name__ == "__main__":
    agents.cli.run_app(server)
