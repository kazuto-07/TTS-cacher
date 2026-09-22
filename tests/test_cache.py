import asyncio
import time

import pytest

from tts_cache import LocalStorage, MemoryIndex, MemoryStorage, SqliteIndex, TTSCache
from tts_cache.errors import ConfigurationError

from .conftest import Vendor


async def test_a_miss_calls_the_vendor_and_a_hit_does_not(cache, vendor):
    first = await cache.get_or_generate("Your balance is ready.", generator_fn=vendor.render)
    second = await cache.get_or_generate("Your balance is ready.", generator_fn=vendor.render)

    assert first.hit is False
    assert second.hit is True
    assert second.audio == first.audio
    assert vendor.count == 1


async def test_the_result_unpacks_the_way_the_readme_says(cache, vendor):
    audio, hit = await cache.get_or_generate("hello", generator_fn=vendor.render)
    assert isinstance(audio, bytes) and hit is False


async def test_a_different_voice_is_a_different_clip(cache, vendor):
    await cache.get_or_generate("hello", generator_fn=vendor.render)
    result = await cache.get_or_generate("hello", generator_fn=vendor.render, voice_id="voice-2")
    assert result.hit is False
    assert vendor.count == 2


@pytest.mark.parametrize("shape", ["sync", "async", "zero-arg", "stream"])
async def test_a_generator_may_be_written_four_ways(cache, vendor, shape):
    generators = {
        "sync": vendor.render,
        "async": vendor.arender,
        "zero-arg": lambda: vendor.render("fixed"),
        "stream": vendor.stream,
    }
    result = await cache.get_or_generate("hello", generator_fn=generators[shape])
    assert result.audio
    assert (await cache.get_or_generate("hello", generator_fn=generators[shape])).hit


async def test_provider_and_voice_are_required():
    cache = TTSCache()
    with pytest.raises(ConfigurationError):
        await cache.get("hello")
    cache.close()


async def test_stats_count_what_happened(cache, vendor):
    await cache.get_or_generate("one", generator_fn=vendor.render)
    await cache.get_or_generate("one", generator_fn=vendor.render)
    await cache.get_or_generate("two", generator_fn=vendor.render)

    assert cache.stats.misses == 2
    assert cache.stats.hits == 1
    assert cache.stats.writes == 2
    assert cache.stats.hit_rate == pytest.approx(1 / 3)
    assert cache.stats.as_dict()["errors"] == 0


async def test_audio_rendered_elsewhere_can_be_put_in(cache, vendor):
    await cache.put("prerendered", b"12345")
    assert await cache.get("prerendered") == b"12345"
    assert vendor.count == 0


async def test_invalidating_forces_the_next_call_to_render(cache, vendor):
    await cache.get_or_generate("hello", generator_fn=vendor.render)
    assert await cache.invalidate("hello") is True
    assert await cache.invalidate("hello") is False

    result = await cache.get_or_generate("hello", generator_fn=vendor.render)
    assert result.hit is False
    assert vendor.count == 2


async def test_an_expired_clip_is_a_miss_and_is_cleaned_up(vendor):
    cache = TTSCache(
        provider="v", voice_id="a", time_to_expire=0.05, background_writes=False, on_error="raise"
    )
    await cache.get_or_generate("hello", generator_fn=vendor.render)
    await asyncio.sleep(0.06)

    result = await cache.get_or_generate("hello", generator_fn=vendor.render)
    assert result.hit is False
    assert vendor.count == 2
    assert await cache.count() == 1, "the stale row is replaced, not accumulated"
    cache.close()


async def test_audio_deleted_behind_the_cache_becomes_a_miss(cache, vendor, caplog):
    await cache.get_or_generate("hello", generator_fn=vendor.render)
    cache.storage.clear()

    result = await cache.get_or_generate("hello", generator_fn=vendor.render)
    assert result.hit is False
    assert vendor.count == 2


async def test_a_broken_index_does_not_break_the_pipeline(vendor):
    class BrokenIndex(MemoryIndex):
        def get(self, cache_key):
            raise RuntimeError("database is down")

    cache = TTSCache(index=BrokenIndex(), provider="v", voice_id="a", background_writes=False)
    result = await cache.get_or_generate("hello", generator_fn=vendor.render)

    assert result.audio  # the caller still gets audio
    assert result.hit is False
    assert cache.stats.errors == 1
    cache.close()


async def test_on_error_raise_surfaces_the_failure(vendor):
    class BrokenIndex(MemoryIndex):
        def get(self, cache_key):
            raise RuntimeError("database is down")

    cache = TTSCache(
        index=BrokenIndex(), provider="v", voice_id="a", background_writes=False, on_error="raise"
    )
    with pytest.raises(Exception, match="database is down"):
        await cache.get_or_generate("hello", generator_fn=vendor.render)
    cache.close()


async def test_a_write_that_fails_leaves_no_row_behind(vendor):
    class BrokenStorage(MemoryStorage):
        def put(self, cache_key, data, audio_format):
            raise RuntimeError("bucket is full")

    cache = TTSCache(storage=BrokenStorage(), provider="v", voice_id="a", background_writes=False)
    result = await cache.get_or_generate("hello", generator_fn=vendor.render)

    assert result.audio
    assert await cache.count() == 0
    assert cache.stats.errors == 1
    cache.close()


async def test_background_writes_do_not_hold_up_the_caller(vendor):
    slow = MemoryStorage()
    original = slow.put

    def slow_put(cache_key, data, audio_format):
        time.sleep(0.2)
        return original(cache_key, data, audio_format)

    slow.put = slow_put
    cache = TTSCache(storage=slow, provider="v", voice_id="a")

    started = time.perf_counter()
    result = await cache.get_or_generate("hello", generator_fn=vendor.render)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1, "the audio came back before the write finished"
    assert result.hit is False
    await cache.drain()
    assert await cache.count() == 1
    cache.close()


async def test_clips_can_be_kept_out_of_the_cache(vendor):
    cache = TTSCache(
        provider="v",
        voice_id="a",
        background_writes=False,
        max_bytes=4,
        should_cache=lambda spec: "otp" not in spec.text,
    )
    await cache.get_or_generate("a much longer sentence", generator_fn=vendor.render)
    assert await cache.count() == 0, "clips over max_bytes are served but not stored"

    await cache.get_or_generate("otp", generator_fn=lambda: b"1234")
    assert await cache.count() == 0, "should_cache said no"

    await cache.get_or_generate("ok", generator_fn=lambda: b"1234")
    assert await cache.count() == 1
    cache.close()


async def test_flush_empties_both_halves(cache, vendor):
    await cache.get_or_generate("one", generator_fn=vendor.render)
    await cache.get_or_generate("two", generator_fn=vendor.render)
    assert await cache.flush() == 2
    assert await cache.count() == 0
    assert len(cache.storage) == 0


async def test_delete_where_retires_a_voice(cache, vendor):
    await cache.get_or_generate("one", generator_fn=vendor.render)
    await cache.get_or_generate("one", generator_fn=vendor.render, voice_id="voice-2")

    assert await cache.delete_where(voice_id="voice-2") == 1
    assert await cache.count() == 1


async def test_purge_expired_removes_only_the_old(vendor):
    cache = TTSCache(provider="v", voice_id="a", background_writes=False, time_to_expire=0.05)
    await cache.get_or_generate("old", generator_fn=vendor.render)
    await asyncio.sleep(0.06)
    await cache.get_or_generate("new", generator_fn=vendor.render)

    assert await cache.purge_expired() == 1
    assert await cache.count() == 1
    cache.close()


# Streaming ---------------------------------------------------------------------------


async def test_a_streamed_miss_is_forwarded_and_then_cached(cache, vendor):
    chunks = [chunk async for chunk in cache.stream_or_generate("hi", generator_fn=vendor.stream)]
    assert chunks == [b"one-", b"two-", b"three"]

    cached = [
        chunk
        async for chunk in cache.stream_or_generate("hi", generator_fn=vendor.stream, chunk_size=4)
    ]
    assert b"".join(cached) == b"one-two-three"
    assert vendor.count == 1, "the second pass never reached the vendor"


async def test_an_abandoned_stream_is_not_cached(cache, vendor):
    stream = cache.stream_or_generate("hi", generator_fn=vendor.stream)
    assert await stream.__anext__() == b"one-"
    await stream.aclose()

    assert await cache.count() == 0, "a half-played clip must never be served as complete"


# Decorator ---------------------------------------------------------------------------


async def test_memoize_wraps_an_async_function(cache, vendor):
    @cache.memoize(voice_id="voice-1")
    async def generate_speech(text: str) -> bytes:
        return await vendor.arender(text)

    assert await generate_speech("hello") == await generate_speech("hello")
    assert vendor.count == 1
    assert generate_speech.__name__ == "generate_speech"
    assert await generate_speech.uncached("hello")
    assert vendor.count == 2, "the undecorated function still calls the vendor"


def test_memoize_wraps_a_plain_function(vendor):
    cache = TTSCache(provider="v", voice_id="a", background_writes=False)

    @cache.memoize()
    def generate_speech(text: str) -> bytes:
        return vendor.render(text)

    assert generate_speech("hello") == generate_speech("hello")
    assert generate_speech(text="hello")
    assert vendor.count == 1
    cache.close()


async def test_the_sync_api_refuses_to_block_an_event_loop(cache):
    with pytest.raises(RuntimeError, match="await the async ones"):
        cache.get_sync("hello")


def test_the_sync_api_works_outside_a_loop(vendor):
    cache = TTSCache(provider="v", voice_id="a", background_writes=False)
    result = cache.get_or_generate_sync("hello", generator_fn=vendor.render)

    assert result.hit is False
    assert cache.get_sync("hello") == result.audio
    cache.close()


# Durability --------------------------------------------------------------------------


async def test_a_cache_on_disk_survives_a_restart(tmp_path, vendor):
    def build():
        return TTSCache(
            storage=LocalStorage(tmp_path / "audio"),
            index=SqliteIndex(tmp_path / "index.db"),
            provider="v",
            voice_id="a",
            background_writes=False,
        )

    first = build()
    await first.get_or_generate("hello", generator_fn=vendor.render)
    first.close()

    second = build()
    result = await second.get_or_generate("hello", generator_fn=vendor.render)
    assert result.hit is True
    assert vendor.count == 1
    second.close()


async def test_two_caches_sharing_an_index_share_the_audio(tmp_path):
    left_vendor, right_vendor = Vendor(b"left"), Vendor(b"right")
    shared = dict(
        storage=LocalStorage(tmp_path / "audio"),
        index=SqliteIndex(tmp_path / "index.db"),
        provider="v",
        voice_id="a",
        background_writes=False,
    )
    left, right = TTSCache(**shared), TTSCache(**shared)

    await left.get_or_generate("hello", generator_fn=left_vendor.render)
    result = await right.get_or_generate("hello", generator_fn=right_vendor.render)

    assert result.hit is True
    assert result.audio.startswith(b"left")
    assert right_vendor.count == 0
    left.close()
    right.close()


# Concurrency -------------------------------------------------------------------------


async def test_the_same_sentence_asked_for_twice_at_once_is_rendered_once(vendor):
    cache = TTSCache(provider="v", voice_id="a", background_writes=False)

    async def slow(text: str) -> bytes:
        await asyncio.sleep(0.05)
        return vendor.render(text)

    results = await asyncio.gather(
        *(cache.get_or_generate("greeting", generator_fn=slow) for _ in range(5))
    )

    assert vendor.count == 1, "four callers waited on the first one's render"
    assert {bytes(r.audio) for r in results} == {results[0].audio}
    assert cache.stats.coalesced == 4
    assert cache.stats.writes == 1
    cache.close()


async def test_coalescing_can_be_turned_off(vendor):
    cache = TTSCache(provider="v", voice_id="a", background_writes=False, coalesce=False)

    async def slow(text: str) -> bytes:
        await asyncio.sleep(0.05)
        return vendor.render(text)

    await asyncio.gather(*(cache.get_or_generate("greeting", generator_fn=slow) for _ in range(3)))
    assert vendor.count == 3
    cache.close()


async def test_different_sentences_are_not_coalesced(vendor):
    cache = TTSCache(provider="v", voice_id="a", background_writes=False)

    async def slow(text: str) -> bytes:
        await asyncio.sleep(0.05)
        return vendor.render(text)

    await asyncio.gather(
        cache.get_or_generate("one", generator_fn=slow),
        cache.get_or_generate("two", generator_fn=slow),
    )
    assert vendor.count == 2
    assert cache.stats.coalesced == 0
    cache.close()


async def test_a_vendor_failure_reaches_everyone_waiting():
    cache = TTSCache(provider="v", voice_id="a", background_writes=False)
    attempts = 0

    async def failing(text: str) -> bytes:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.05)
        raise RuntimeError("vendor is down")

    results = await asyncio.gather(
        *(cache.get_or_generate("greeting", generator_fn=failing) for _ in range(3)),
        return_exceptions=True,
    )

    assert attempts == 1
    assert all(isinstance(r, RuntimeError) for r in results)
    assert await cache.count() == 0

    # The failure must not be remembered: the next call tries the vendor again.
    async def working(text: str) -> bytes:
        return b"audio"

    assert (await cache.get_or_generate("greeting", generator_fn=working)).audio == b"audio"
    cache.close()
