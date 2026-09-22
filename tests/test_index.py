"""Every index driver has to behave identically, so they all run the same tests."""

import time

import pytest

from tts_cache import CacheEntry, MemoryIndex, SqliteIndex
from tts_cache.index.sqlalchemy import SqlAlchemyIndex


@pytest.fixture(params=["memory", "sqlite", "sqlalchemy"])
def index(request, tmp_path):
    if request.param == "memory":
        driver = MemoryIndex()
    elif request.param == "sqlite":
        driver = SqliteIndex(tmp_path / "index.db")
    else:
        driver = SqlAlchemyIndex(f"sqlite:///{tmp_path / 'alchemy.db'}")
    yield driver
    driver.close()


def entry(key="a" * 64, *, size=100, count=1, age=0.0, idle=0.0, voice="voice-1"):
    now = time.time()
    return CacheEntry(
        cache_key=key,
        normalized_text="hello",
        provider="testvendor",
        voice_id=voice,
        model="m",
        audio_format="mp3",
        audio_path=f"mem://{key}.mp3",
        size_bytes=size,
        access_count=count,
        created_at=now - age,
        last_accessed_at=now - idle,
    )


def test_a_row_round_trips(index):
    index.put(entry())
    stored = index.get("a" * 64)

    assert stored is not None
    assert stored.normalized_text == "hello"
    assert stored.size_bytes == 100
    assert stored.access_count == 1
    assert index.get("b" * 64) is None


def test_putting_the_same_key_twice_replaces_the_row(index):
    index.put(entry())
    index.put(entry(size=250))

    assert index.count() == 1
    assert index.get("a" * 64).size_bytes == 250


def test_touching_records_a_use(index):
    index.put(entry())
    when = time.time() + 5
    index.touch("a" * 64, when)

    stored = index.get("a" * 64)
    assert stored.access_count == 2
    assert stored.last_accessed_at == pytest.approx(when, abs=0.001)


def test_touching_a_missing_key_is_harmless(index):
    index.touch("z" * 64, time.time())
    assert index.count() == 0


def test_delete_returns_the_row_it_removed(index):
    index.put(entry())
    removed = index.delete("a" * 64)

    assert removed.audio_path.endswith(".mp3")
    assert index.delete("a" * 64) is None
    assert index.count() == 0


def test_totals_add_up(index):
    index.put(entry("a" * 64, size=100))
    index.put(entry("b" * 64, size=250))

    assert index.count() == 2
    assert index.total_size() == 350
    index.clear()
    assert index.count() == 0 and index.total_size() == 0


def test_listing_filters_and_orders(index):
    index.put(entry("a" * 64, size=100, idle=10, voice="voice-1"))
    index.put(entry("b" * 64, size=300, idle=1, voice="voice-2"))

    recent = index.list()
    assert [e.cache_key[0] for e in recent] == ["b", "a"]
    assert [e.cache_key[0] for e in index.list(order="size_bytes", descending=False)] == ["a", "b"]
    assert [e.voice_id for e in index.list(voice_id="voice-2")] == ["voice-2"]
    assert index.list(provider="nobody") == []
    assert len(index.list(limit=1)) == 1


def test_listing_rejects_an_order_it_cannot_index(index):
    with pytest.raises(ValueError, match="order by"):
        index.list(order="normalized_text")


def test_expired_finds_rows_by_creation_time(index):
    index.put(entry("a" * 64, age=100))
    index.put(entry("b" * 64, age=1))

    expired = index.expired(time.time() - 50)
    assert [e.cache_key[0] for e in expired] == ["a"]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        # a was used longest ago, b was used least often, c is the oldest.
        ("lru", "a"),
        ("lfu", "b"),
        ("fifo", "c"),
    ],
)
def test_each_policy_gives_up_a_different_entry(index, policy, expected):
    index.put(entry("a" * 64, size=100, count=9, idle=60, age=10))
    index.put(entry("b" * 64, size=100, count=1, idle=10, age=5))
    index.put(entry("c" * 64, size=100, count=5, idle=30, age=99))

    candidates = index.eviction_candidates(policy, free_bytes=100)
    assert [e.cache_key[0] for e in candidates] == [expected]


def test_eviction_takes_as_many_as_it_needs(index):
    for i, key in enumerate("abcd"):
        index.put(entry(key * 64, size=100, idle=10 - i))

    candidates = index.eviction_candidates("lru", free_bytes=250)
    assert len(candidates) == 3, "three entries of 100 bytes to free 250"


def test_an_unknown_policy_is_refused(index):
    with pytest.raises(ValueError, match="eviction policy"):
        index.eviction_candidates("random", free_bytes=1)
