"""The simplest way to cache a LiveKit agent's speech: cache what you `say()`.

    pip install "livekit-agents[deepgram,elevenlabs,openai,silero]" tts-cache
    python examples/livekit_say.py dev

No `Agent` subclass, no plugin wrapper, no override. `session.say()` accepts
`audio=AsyncIterable[rtc.AudioFrame]`, which is exactly what the cache yields — so caching
a phrase is one function.

This covers fixed phrases: the greeting, the hold line, "sorry, I didn't catch that". In a
support agent that is most of what is said, and it is all text you know in advance. To cache
*every* reply the model generates, the cache belongs in `tts_node` instead — see
`livekit_tts_node.py`, which costs sentence splitting and barge-in handling to get there.

Not run in CI: it needs a LiveKit project and vendor keys.
"""

from __future__ import annotations

from collections.abc import AsyncIterable

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, room_io
from livekit.plugins import deepgram, elevenlabs, openai, silero

from tts_cache import LocalStorage, SqliteIndex, TTSCache

load_dotenv(".env.local")

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
TTS_MODEL = "eleven_turbo_v2_5"
RATE, CHANNELS = 24_000, 1

cache = TTSCache(
    storage=LocalStorage("/var/cache/voice/audio"),
    index=SqliteIndex("/var/cache/voice/index.db"),
    provider="elevenlabs",
    voice_id=VOICE_ID,
    model=TTS_MODEL,
    # PCM has no header to say what it is, so the rate goes in the key. Change the rate
    # and you simply miss, instead of playing a clip back at the wrong speed.
    audio_format="pcm",
    settings={"sample_rate": RATE, "num_channels": CHANNELS},
)


async def say(session: AgentSession, tts: elevenlabs.TTS, text: str) -> None:
    """Says `text`, from the cache if it has been said before. This is the whole trick."""

    async def render() -> AsyncIterable[bytes]:
        stream = tts.synthesize(text)
        async for event in stream:
            yield bytes(event.frame.data)

    async def frames() -> AsyncIterable[rtc.AudioFrame]:
        async for pcm in cache.stream_or_generate(
            text,
            generator_fn=render,
            chunk_size=RATE // 1000 * 20 * 2 * CHANNELS,  # 20 ms frames
            frame_bytes=2 * CHANNELS,  # never split a 16-bit sample
        ):
            yield rtc.AudioFrame(pcm, RATE, CHANNELS, len(pcm) // (2 * CHANNELS))

    await session.say(text, audio=frames())


server = AgentServer()


@server.rtc_session(agent_name="cached-say")
async def cached_say(ctx: agents.JobContext) -> None:
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
            agent=Agent(instructions="You are a helpful voice assistant for Northwind Support."),
            room_options=room_io.RoomOptions(),
        )
        # The first caller of the day pays for this line. Nobody else does.
        await say(session, tts, "Hi, you've reached Northwind Support. How can I help?")
        await session.generate_reply(instructions="Wait for the caller to speak.")
    finally:
        ctx.log_context_fields = {"tts_hit_rate": round(cache.stats.hit_rate, 3)}
        await cache.aclose()


if __name__ == "__main__":
    agents.cli.run_app(server)
