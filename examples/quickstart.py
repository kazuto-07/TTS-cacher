"""What the cache is worth, against a vendor that behaves like a real one.

    python examples/quickstart.py

No API key: the stand-in vendor sleeps for a realistic time to first byte and returns
silence. The numbers to look at are the two latencies and the hit rate.
"""

import asyncio

from tts_cache import LocalStorage, SqliteIndex, TTSCache

#: What a hosted TTS call costs you before the first byte arrives.
VENDOR_LATENCY = 0.35

LINES = [
    "Hi, you've reached Northwind Support. How can I help?",
    "Let me pull that up for you.",
    "Sorry, I didn't catch that.",
    "Let me pull that up for you.",
    "Hi, you've reached Northwind Support. How can I help?",
    "Sorry, I didn't catch that.",
    "Is there anything else I can help with?",
    "Let me pull that up for you.",
]


class FakeVendor:
    def __init__(self) -> None:
        self.calls = 0
        self.characters = 0

    async def render(self, text: str) -> bytes:
        self.calls += 1
        self.characters += len(text)
        await asyncio.sleep(VENDOR_LATENCY)
        # Roughly one second of 64 kbps mp3 per twelve characters spoken.
        return b"\x00" * (len(text) * 700)


async def main() -> None:
    vendor = FakeVendor()
    cache = TTSCache(
        storage=LocalStorage(".tts-cache/audio"),
        index=SqliteIndex(".tts-cache/index.db"),
        provider="elevenlabs",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model="eleven_turbo_v2",
        max_storage_size_mb=100,
        time_to_expire=86400 * 30,
    )
    await cache.flush()  # so the numbers are the same on every run

    print(f"{'':2} {'source':6} {'latency':>9}  line")
    for number, line in enumerate(LINES, start=1):
        result = await cache.get_or_generate(line, generator_fn=vendor.render)
        source = "cache" if result.hit else "vendor"
        print(f"{number:>2} {source:6} {result.elapsed_ms:>7.1f}ms  {line}")

    await cache.drain()
    stats = cache.stats
    print()
    print(f"vendor calls   {vendor.calls} of {len(LINES)} lines, {vendor.characters} characters")
    print(f"hit rate       {stats.hit_rate:.0%}")
    print(f"stored         {stats.writes} clips, {stats.bytes_written / 1024:.0f} KB")
    print(f"time saved     {stats.hits * VENDOR_LATENCY:.1f}s of caller-facing silence")
    await cache.aclose()


if __name__ == "__main__":
    asyncio.run(main())
