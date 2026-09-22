"""Real audio through the cache, not stand-in bytes.

Everywhere else the tests use `b"audio-bytes"`, which proves the bookkeeping but not that a
clip comes back playable. These generate actual waveforms with the standard library — a
sine tone as WAV, and raw PCM in the 20 ms frames a voice pipeline hands around — store
them, read them back and decode them again.

What this is really checking is that nothing in the path *touches* the audio: no encoding
guessed, no re-chunking, no truncation, no stray header. A TTS cache that alters a byte is
worse than no cache at all.
"""

import array
import io
import math
import wave

import pytest

from tts_cache import LocalStorage, SqliteIndex, TTSCache

RATE = 24_000
FRAME_MS = 20
FRAME_BYTES = RATE // 1000 * FRAME_MS * 2  # 20 ms of 16-bit mono


def pcm_tone(seconds: float = 0.25, hz: float = 440.0, rate: int = RATE) -> bytes:
    """A sine tone as 16-bit little-endian mono PCM — what a TTS vendor hands back."""
    samples = array.array(
        "h",
        (int(math.sin(2 * math.pi * hz * i / rate) * 12_000) for i in range(int(seconds * rate))),
    )
    # array uses the platform's byte order; the cache must not care either way, but the
    # test needs to know what it wrote.
    if array.array("h", [1]).tobytes()[0] == 0:  # big endian
        samples.byteswap()
    return samples.tobytes()


def wav_tone(seconds: float = 0.25, rate: int = RATE) -> bytes:
    """The same tone wrapped as a WAV file, header and all."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm_tone(seconds, rate=rate))
    return buffer.getvalue()


@pytest.fixture
def disk_cache(tmp_path):
    """A cache that writes real files, the way a deployment would."""
    cache = TTSCache(
        storage=LocalStorage(tmp_path / "audio"),
        index=SqliteIndex(tmp_path / "index.db"),
        provider="testvendor",
        voice_id="voice-1",
        background_writes=False,
        on_error="raise",
    )
    yield cache
    cache.close()


def test_a_wav_clip_comes_back_byte_identical_and_still_decodes(disk_cache):
    original = wav_tone()

    result = disk_cache.get_or_generate_sync(
        "Your balance is ready.", generator_fn=lambda: original, audio_format="wav"
    )
    assert result.hit is False

    served = disk_cache.get_sync("Your balance is ready.", audio_format="wav")
    assert served == original, "a cached clip must be the bytes the vendor produced"

    # And it is still a WAV, not just the right number of bytes.
    with wave.open(io.BytesIO(served), "rb") as clip:
        assert clip.getnchannels() == 1
        assert clip.getsampwidth() == 2
        assert clip.getframerate() == RATE
        assert clip.getnframes() == int(0.25 * RATE)
        assert clip.readframes(clip.getnframes()) == pcm_tone()


def test_the_stored_file_is_playable_where_the_index_says_it_is(disk_cache, tmp_path):
    """The audio_path in a row has to point at audio, not at a temp file or a fragment."""
    original = wav_tone(seconds=0.1)
    disk_cache.put_sync("hold please", original, audio_format="wav")

    entry = disk_cache.index.list()[0]
    assert entry.audio_path.endswith(".wav")
    assert entry.size_bytes == len(original)

    with open(entry.audio_path, "rb") as handle:
        on_disk = handle.read()
    # The row's size has to describe the file, or eviction budgets are fiction.
    assert len(on_disk) == entry.size_bytes

    with wave.open(io.BytesIO(on_disk), "rb") as clip:
        assert clip.getframerate() == RATE
        assert clip.getnframes() == int(0.1 * RATE)
        # A truncated file still claims a full frame count in its header, so read them.
        assert clip.readframes(clip.getnframes()) == pcm_tone(seconds=0.1)


def test_a_clip_survives_a_restart_and_still_decodes(tmp_path):
    original = wav_tone(seconds=0.1)

    def build():
        return TTSCache(
            storage=LocalStorage(tmp_path / "audio"),
            index=SqliteIndex(tmp_path / "index.db"),
            provider="testvendor",
            voice_id="voice-1",
            audio_format="wav",
            background_writes=False,
        )

    first = build()
    first.put_sync("good morning", original)
    first.close()

    second = build()
    served = second.get_sync("good morning")
    second.close()

    with wave.open(io.BytesIO(served), "rb") as clip:
        assert clip.readframes(clip.getnframes()) == pcm_tone(seconds=0.1)


async def test_raw_pcm_streams_back_in_whole_frames(disk_cache):
    """The path the LiveKit and Pipecat examples take: headerless PCM, 20 ms at a time."""
    original = pcm_tone(seconds=0.4)
    spec = {"audio_format": "pcm", "settings": {"sample_rate": RATE, "num_channels": 1}}

    async def vendor():
        # Vendors chunk on their own boundaries, not on frame boundaries.
        for start in range(0, len(original), 1_234 * 2):
            yield original[start : start + 1_234 * 2]

    forwarded = [
        chunk
        async for chunk in disk_cache.stream_or_generate(
            "one moment", generator_fn=vendor, chunk_size=FRAME_BYTES, **spec
        )
    ]
    assert b"".join(forwarded) == original, "a miss must forward exactly what the vendor sent"

    cached = [
        chunk
        async for chunk in disk_cache.stream_or_generate(
            "one moment", generator_fn=vendor, chunk_size=FRAME_BYTES, **spec
        )
    ]
    assert b"".join(cached) == original, "a hit must reproduce the waveform exactly"
    assert all(len(chunk) == FRAME_BYTES for chunk in cached[:-1]), "whole 20 ms frames"
    assert all(len(chunk) % 2 == 0 for chunk in cached), "never split a sample in half"


async def test_the_same_words_at_another_sample_rate_are_a_different_clip(disk_cache):
    """PCM has no header, so the rate lives in the key or it is lost."""
    at_24k = pcm_tone(rate=24_000)
    at_48k = pcm_tone(rate=48_000)

    await disk_cache.put("welcome", at_24k, audio_format="pcm", settings={"sample_rate": 24_000})
    served = await disk_cache.get("welcome", audio_format="pcm", settings={"sample_rate": 48_000})

    assert served is None, "a 24 kHz clip must never be served to a 48 kHz pipeline"
    assert len(at_48k) == 2 * len(at_24k)


async def test_an_interrupted_clip_is_never_stored_as_a_whole_one(disk_cache):
    """Half a sentence is worse than no cache: it would be replayed truncated forever."""
    original = pcm_tone(seconds=0.4)
    spec = {"audio_format": "pcm", "settings": {"sample_rate": RATE}}

    async def vendor():
        for start in range(0, len(original), FRAME_BYTES):
            yield original[start : start + FRAME_BYTES]

    stream = disk_cache.stream_or_generate("cut short", generator_fn=vendor, **spec)
    heard = b"".join([await stream.__anext__() for _ in range(3)])
    await stream.aclose()

    assert len(heard) == 3 * FRAME_BYTES
    assert await disk_cache.get("cut short", **spec) is None
