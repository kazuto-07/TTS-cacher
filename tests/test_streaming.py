"""What a voice pipeline is actually judged on: when the first chunk arrives.

The rest of the suite proves the bytes come back. This one proves they start coming back
early — that a hit does not wait for the whole clip to be read, that a chunk never lands
inside an audio frame, and that pacing does not delay the first chunk.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tts_cache import LocalStorage, MemoryIndex, MemoryStorage, SqliteIndex, TTSCache

CLIP = b"".join(bytes([i % 256]) for i in range(40_000))


class ChunkedStorage(MemoryStorage):
    """A store that can only produce a chunk at a time, and says when each one left."""

    name = "chunked"

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.delay = delay
        self.reads: list[float] = []

    def get(self, path):
        """The whole-blob read a streaming hit is not allowed to fall back to."""
        data = super().get(path)
        if data is not None:
            time.sleep(self.delay * (len(data) / 8192))
        return data

    def iter(self, path, chunk_size=8192):
        data = super().get(path)
        if data is None:
            return None

        def chunks():
            for at in range(0, len(data), chunk_size):
                time.sleep(self.delay)
                self.reads.append(time.perf_counter())
                yield data[at : at + chunk_size]

        return chunks()


async def collect(stream) -> tuple[list[bytes], float]:
    """Every chunk, and how long the first one took to arrive, in milliseconds."""
    started = time.perf_counter()
    first: float | None = None
    chunks = []
    async for chunk in stream:
        if first is None:
            first = (time.perf_counter() - started) * 1000
        chunks.append(chunk)
    return chunks, (first if first is not None else 0.0)


async def test_a_hit_starts_playing_before_the_whole_clip_has_been_read():
    storage = ChunkedStorage(delay=0.01)
    cache = TTSCache(
        storage=storage, index=MemoryIndex(), provider="p", voice_id="v", background_writes=False
    )
    await cache.put("hello", CLIP)

    chunks, ttfa = await collect(
        cache.stream_or_generate("hello", generator_fn=lambda: CLIP, chunk_size=8192)
    )

    assert b"".join(chunks) == CLIP
    # Five chunks at 10 ms each: the whole read is ~50 ms, so a first chunk anywhere near
    # that means the clip was assembled before the caller heard anything.
    assert ttfa < 30, f"time to first audio was {ttfa:.0f} ms — the whole clip was read first"
    assert len(storage.reads) == 5
    cache.close()


async def test_the_time_to_first_audio_is_recorded_for_hits_and_misses():
    cache = TTSCache(
        provider="p",
        voice_id="v",
        background_writes=False,
        storage=MemoryStorage(),
        index=MemoryIndex(),
    )

    async def render():
        await asyncio.sleep(0.05)
        yield CLIP

    await collect(cache.stream_or_generate("hello", generator_fn=render))
    assert cache.stats.miss_streams == 1
    assert cache.stats.miss_ttfa_ms >= 40, "a miss waits for the vendor"

    await collect(cache.stream_or_generate("hello", generator_fn=render))
    assert cache.stats.hit_streams == 1
    assert cache.stats.hit_ttfa_ms < cache.stats.miss_ttfa_ms
    assert "hit_ttfa_ms" in cache.stats.as_dict()
    cache.close()


@pytest.mark.parametrize("frame_bytes", [2, 4, 6])
async def test_no_chunk_ever_splits_an_audio_frame(frame_bytes):
    """Half a PCM frame puts every later frame out of phase, which is heard as noise."""
    cache = TTSCache(provider="p", voice_id="v", background_writes=False)

    async def render():
        # Deliberately awkward sizes that do not divide by any frame width.
        for size in (5, 7, 11, 13):
            yield b"\x01" * size

    chunks, _ = await collect(
        cache.stream_or_generate("hello", generator_fn=render, frame_bytes=frame_bytes)
    )

    assert all(len(chunk) % frame_bytes == 0 for chunk in chunks)
    joined = b"".join(chunks)
    assert len(joined) % frame_bytes == 0
    # 36 bytes in: padded up to the next whole frame, never truncated.
    assert len(joined) >= 36
    cache.close()


async def test_a_paced_stream_does_not_outrun_realtime_but_still_starts_at_once():
    cache = TTSCache(provider="p", voice_id="v", background_writes=False)
    rate = 48_000  # bytes per second: 24 kHz, 16-bit, mono

    async def render():
        yield b"\x00" * 24_000  # half a second of audio, all at once

    started = time.perf_counter()
    chunks, ttfa = await collect(
        cache.stream_or_generate(
            "hello", generator_fn=render, chunk_size=4800, bytes_per_second=rate
        )
    )
    elapsed = time.perf_counter() - started

    assert b"".join(chunks) == b"\x00" * 24_000
    # One chunk is 100 ms of audio, so anything well inside that proves the first chunk
    # was not held back to the clock. The bound is loose on purpose: a Windows timer ticks
    # every ~15 ms, and this assertion is about pacing, not about scheduler jitter.
    assert ttfa < 50, "pacing must never delay the first chunk"
    assert elapsed >= 0.4, "the rest should arrive at the speed it is spoken"
    cache.close()


async def test_a_row_whose_audio_vanished_falls_back_to_the_vendor(tmp_path):
    """Someone emptied the bucket: the stream must re-render, not fail or go silent."""
    storage = LocalStorage(tmp_path / "audio")
    cache = TTSCache(
        storage=storage,
        index=SqliteIndex(tmp_path / "index.db"),
        provider="p",
        voice_id="v",
        background_writes=False,
    )
    await cache.put("hello", CLIP)
    storage.clear()

    calls = []

    async def render():
        calls.append(1)
        yield b"fresh"

    chunks, _ = await collect(cache.stream_or_generate("hello", generator_fn=render))

    assert b"".join(chunks) == b"fresh"
    assert calls == [1], "the vendor had to be asked again"
    assert await cache.get("hello") == b"fresh"
    cache.close()


async def test_an_abandoned_stream_is_not_cached():
    """A caller barging in half way must not leave a half-spoken clip behind."""
    cache = TTSCache(provider="p", voice_id="v", background_writes=False)

    async def render():
        yield b"first-"
        yield b"second-"
        yield b"third"

    stream = cache.stream_or_generate("hello", generator_fn=render)
    assert await anext(stream) == b"first-"
    await stream.aclose()

    assert await cache.get("hello") is None
    cache.close()


def test_local_storage_streams_a_file_and_reports_a_missing_one(tmp_path):
    storage = LocalStorage(tmp_path)
    path = storage.put("abc123", CLIP, "wav")

    chunks = list(storage.iter(path, chunk_size=4096))

    assert b"".join(chunks) == CLIP
    assert len(chunks) == 10, "a streaming read must hand over more than one piece"
    assert storage.iter(str(tmp_path / "gone.wav")) is None


async def test_a_driver_without_a_streaming_read_still_works():
    """The base-class fallback slices the blob, so an old driver keeps serving hits."""

    class WholeBlobOnly(MemoryStorage):
        name = "whole-blob"
        # No iter() override at all.

    cache = TTSCache(
        storage=WholeBlobOnly(),
        index=MemoryIndex(),
        provider="p",
        voice_id="v",
        background_writes=False,
    )
    await cache.put("hello", CLIP)

    chunks, _ = await collect(
        cache.stream_or_generate("hello", generator_fn=lambda: CLIP, chunk_size=8192)
    )

    assert b"".join(chunks) == CLIP
    assert len(chunks) == 5
    cache.close()


async def test_stream_returns_none_for_something_not_cached():
    """The Pipecat shape: learn it is a miss before any audio, and keep your own path."""
    cache = TTSCache(provider="p", voice_id="v", background_writes=False)

    assert await cache.stream("never rendered") is None
    assert cache.stats.misses == 1

    await cache.put("hello", CLIP)
    stream = await cache.stream("hello", chunk_size=8192)
    assert stream is not None
    assert cache.stats.hits == 1

    chunks, _ = await collect(stream)
    assert b"".join(chunks) == CLIP
    assert cache.stats.bytes_served == len(CLIP)
    cache.close()
