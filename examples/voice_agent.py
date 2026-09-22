"""Per-sentence caching in a streaming agent, which is where the latency actually is.

    python examples/voice_agent.py

A voice agent speaks the model's reply sentence by sentence, so the caller waits for the
*first* sentence and nothing else. That first sentence is also the most repeated one in the
whole product — "Let me check that for you." — which is exactly what makes sentence-level
caching worth more than caching whole replies.

This example also shows ``should_cache``: a sentence with an account number in it is spoken
once and never stored.
"""

import asyncio
import re
import time

from tts_cache import TTSCache

SENTENCE = re.compile(r"[^.!?]+[.!?]?\s*")
#: Anything with a number in it is probably specific to one caller.
DYNAMIC = re.compile(r"\d")

REPLIES = [
    "Let me check that for you. Your balance is 4,281 rupees.",
    "Let me check that for you. Your last payment was 1,050 rupees.",
    "Sure thing. Let me check that for you. Anything else?",
]


class FakeVendor:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, text: str):
        self.calls += 1
        await asyncio.sleep(0.3)  # time to first byte
        for _ in range(3):
            yield b"\x00" * 4096
            await asyncio.sleep(0.02)


def sentences(reply: str) -> list[str]:
    return [s.strip() for s in SENTENCE.findall(reply) if s.strip()]


async def main() -> None:
    vendor = FakeVendor()
    cache = TTSCache(
        provider="murf",
        voice_id="en-IN-arohi",
        should_cache=lambda spec: not DYNAMIC.search(spec.text),
    )

    for turn, reply in enumerate(REPLIES, start=1):
        print(f"\nturn {turn}: {reply}")
        for sentence in sentences(reply):
            started = time.perf_counter()
            first_byte = None
            async for _chunk in cache.stream_or_generate(sentence, generator_fn=vendor.stream):
                if first_byte is None:
                    first_byte = (time.perf_counter() - started) * 1000
            cached = "cached" if first_byte is not None and first_byte < 50 else "vendor"
            print(f"  {cached:6} {first_byte:>6.1f}ms to first audio  {sentence}")

    await cache.drain()
    print(f"\nvendor calls {vendor.calls}, hit rate {cache.stats.hit_rate:.0%}")
    print(f"stored {await cache.count()} clips: the ones with numbers in them were not")
    await cache.aclose()


if __name__ == "__main__":
    asyncio.run(main())
