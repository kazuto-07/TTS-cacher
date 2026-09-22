"""Both binary stores have to behave identically, so they run the same tests."""

import pytest

from tts_cache import LocalStorage, MemoryStorage

KEY = "a" * 64


@pytest.fixture(params=["memory", "local"])
def storage(request, tmp_path):
    return MemoryStorage() if request.param == "memory" else LocalStorage(tmp_path / "audio")


def test_audio_round_trips(storage):
    path = storage.put(KEY, b"clip", "mp3")
    assert storage.get(path) == b"clip"


def test_writing_the_same_key_twice_overwrites(storage):
    storage.put(KEY, b"first", "mp3")
    path = storage.put(KEY, b"second", "mp3")
    assert storage.get(path) == b"second"


def test_a_missing_blob_reads_as_none(storage):
    path = storage.put(KEY, b"clip", "mp3")
    storage.delete(path)

    assert storage.get(path) is None
    storage.delete(path)  # deleting twice is not an error


def test_clear_removes_everything(storage):
    paths = [storage.put(key * 64, b"clip", "mp3") for key in "abc"]
    storage.clear()
    assert all(storage.get(path) is None for path in paths)


async def test_the_async_surface_reaches_the_same_bytes(storage):
    path = await storage.aput(KEY, b"clip", "wav")
    assert await storage.aget(path) == b"clip"
    await storage.adelete(path)
    assert await storage.aget(path) is None


def test_local_storage_shards_by_key_and_keeps_the_format(tmp_path):
    storage = LocalStorage(tmp_path)
    path = storage.put(KEY, b"clip", "wav")

    assert path.endswith(f"{KEY}.wav")
    assert (tmp_path / KEY[:2]).is_dir(), "keys are spread across prefix directories"


def test_local_storage_leaves_no_partial_files(tmp_path):
    storage = LocalStorage(tmp_path)
    storage.put(KEY, b"clip", "mp3")

    assert not list(tmp_path.rglob("*.part"))
