"""Size limits, seen from the outside: the cache stays under its ceiling on its own."""

import asyncio

import pytest

from tts_cache import TTSCache
from tts_cache.cache import MB
from tts_cache.errors import ConfigurationError


def clip(size: int) -> bytes:
    return b"x" * size


def build(policy="lru", limit_mb=1.0):
    return TTSCache(
        provider="v",
        voice_id="a",
        max_storage_size_mb=limit_mb,
        eviction_policy=policy,
        background_writes=False,
        on_error="raise",
    )


async def test_the_cache_stays_under_its_limit():
    cache = build(limit_mb=1.0)
    for i in range(20):
        await cache.get_or_generate(f"clip {i}", generator_fn=lambda: clip(100 * 1024))

    assert await cache.size_bytes() <= MB
    assert cache.stats.evictions > 0
    cache.close()


async def test_lru_keeps_what_is_still_being_played():
    cache = build(limit_mb=0.3)
    for i in range(3):
        await cache.get_or_generate(f"clip {i}", generator_fn=lambda: clip(100 * 1024))

    # "clip 0" is used again, so "clip 1" becomes the least recently used.
    await asyncio.sleep(0.01)
    assert await cache.get("clip 0") is not None
    await cache.get_or_generate("clip 3", generator_fn=lambda: clip(100 * 1024))

    assert await cache.get("clip 0") is not None
    assert await cache.get("clip 1") is None
    cache.close()


async def test_lfu_keeps_the_phrase_everyone_hears():
    cache = build(policy="lfu", limit_mb=0.3)
    await cache.get_or_generate("greeting", generator_fn=lambda: clip(100 * 1024))
    for _ in range(5):
        await cache.get("greeting")
    for i in range(2):
        await cache.get_or_generate(f"rare {i}", generator_fn=lambda: clip(100 * 1024))

    await cache.get_or_generate("another", generator_fn=lambda: clip(100 * 1024))

    assert await cache.get("greeting") is not None, "the most used clip survives"
    cache.close()


async def test_prune_takes_the_cache_down_to_a_target():
    cache = TTSCache(provider="v", voice_id="a", background_writes=False)
    for i in range(10):
        await cache.get_or_generate(f"clip {i}", generator_fn=lambda: clip(100 * 1024))

    evicted = await cache.prune(target_size_mb=0.5)
    assert evicted >= 5
    assert await cache.size_bytes() <= 0.5 * MB
    cache.close()


async def test_prune_without_a_target_is_refused():
    cache = TTSCache(provider="v", voice_id="a")
    with pytest.raises(ConfigurationError):
        await cache.prune()
    with pytest.raises(ConfigurationError):
        await cache.purge_expired()
    cache.close()


def test_an_unknown_policy_is_refused_at_construction():
    with pytest.raises(ConfigurationError, match="eviction policy"):
        TTSCache(eviction_policy="random")
