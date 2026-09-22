"""The cache against a real vendor: Murf, over the real network, with real speech.

    set MURF_API_KEY=...          # or put it in a .env at the project root
    pytest -m network

Everything else in the suite runs offline against invented bytes or a generated tone. This
is the one test that proves the whole thing against audio a person could actually listen
to: Murf synthesises a sentence, the cache stores it, and every later read has to give back
the same file, still decodable, without asking Murf again.

It is skipped unless `MURF_API_KEY` is set *and* you ask for it with `-m network`, so a
plain `pytest` never spends characters or waits on a network. One run makes exactly one
synthesis call — everything after the first assertion is served from the cache, which is
the point.

No HTTP dependency: `urllib` from the standard library is enough for one POST, and the
package itself still needs nothing at runtime.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

from tts_cache import LocalStorage, SqliteIndex, TTSCache

pytestmark = pytest.mark.network

#: Overridable so the test can be pointed at a stand-in that speaks Murf's response format.
MURF_URL = os.environ.get("MURF_URL", "https://api.murf.ai/v1/speech/generate")
DEFAULT_VOICE = "en-US-natalie"
SAMPLE_RATE = 24_000

#: Short, and the kind of line a support agent repeats all day — which is the whole pitch.
SENTENCE = "Let me check that for you."


def _api_key() -> str | None:
    """The key from the environment, or from a .env at the project root."""
    key = os.environ.get("MURF_API_KEY")
    if key:
        return key.strip()

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "MURF_API_KEY":
            return value.strip().strip("\"'") or None
    return None


API_KEY = _api_key()
VOICE_ID = os.environ.get("MURF_VOICE_ID", DEFAULT_VOICE)

needs_key = pytest.mark.skipif(
    API_KEY is None, reason="set MURF_API_KEY (or put it in .env) to run the live Murf test"
)


class Murf:
    """One synthesis call, counted, so the tests can prove the cache stopped making them."""

    def __init__(self, api_key: str, voice_id: str) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.calls = 0

    def render(self, text: str) -> bytes:
        """Blocking POST; the cache runs it on a thread, as it would any sync generator."""
        self.calls += 1
        payload = json.dumps(
            {
                "text": text,
                "voiceId": self.voice_id,
                "format": "WAV",
                "sampleRate": SAMPLE_RATE,
                "channelType": "MONO",
                # Base64 keeps the audio in the response, so there is no second request to
                # a CDN and Murf retains nothing.
                "encodeAsBase64": True,
            }
        ).encode()

        request = urllib.request.Request(
            MURF_URL,
            data=payload,
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            pytest.fail(f"Murf returned HTTP {e.code}: {detail}")
        except urllib.error.URLError as e:
            pytest.skip(f"Murf unreachable: {e.reason}")

        encoded = body.get("encodedAudio")
        assert encoded, f"Murf returned no audio: {json.dumps(body)[:500]}"
        return base64.b64decode(encoded)


@pytest.fixture(scope="module")
def murf() -> Murf:
    return Murf(API_KEY or "", VOICE_ID)


@pytest.fixture(scope="module")
def live_cache(tmp_path_factory):
    """A cache on disk, shared by this module so one render covers every assertion."""
    root = tmp_path_factory.mktemp("murf")
    cache = TTSCache(
        storage=LocalStorage(root / "audio"),
        index=SqliteIndex(root / "index.db"),
        provider="murf",
        voice_id=VOICE_ID,
        audio_format="wav",
        settings={"sample_rate": SAMPLE_RATE, "channel_type": "MONO"},
        background_writes=False,
        on_error="raise",
    )
    cache.root = root  # the restart test reopens the same pair
    yield cache
    cache.close()


@needs_key
async def test_murf_is_called_once_and_never_again(live_cache, murf):
    """Miss, hit, and a hit after a restart — one synthesis for all three."""
    first = await live_cache.get_or_generate(SENTENCE, generator_fn=murf.render)
    assert first.hit is False
    assert murf.calls == 1
    assert len(first.audio) > 10_000, "a spoken sentence is not a few hundred bytes"

    second = await live_cache.get_or_generate(SENTENCE, generator_fn=murf.render)
    assert second.hit is True
    assert second.audio == first.audio, "the cache must return Murf's bytes, unaltered"
    assert murf.calls == 1, "a hit must not reach the vendor"
    assert second.elapsed_ms < 250, f"a hit took {second.elapsed_ms:.0f} ms"

    # A different voice is a different clip, so it must miss — checked with `get`, which
    # cannot spend characters even if the rule were broken.
    assert await live_cache.get(SENTENCE, voice_id="en-UK-hazel") is None
    assert murf.calls == 1

    # And the clip is still there for the next process.
    reopened = TTSCache(
        storage=LocalStorage(live_cache.root / "audio"),
        index=SqliteIndex(live_cache.root / "index.db"),
        provider="murf",
        voice_id=VOICE_ID,
        audio_format="wav",
        settings={"sample_rate": SAMPLE_RATE, "channel_type": "MONO"},
        background_writes=False,
    )
    try:
        assert await reopened.get(SENTENCE) == first.audio
    finally:
        reopened.close()

    assert live_cache.stats.hits == 1
    assert live_cache.stats.writes == 1
    assert live_cache.stats.errors == 0


@needs_key
async def test_what_murf_sent_is_still_speech_after_a_round_trip(live_cache, murf):
    """Reads the clip the previous test paid for and decodes it. No new call."""
    audio = await live_cache.get(SENTENCE)
    if audio is None:
        audio = (await live_cache.get_or_generate(SENTENCE, generator_fn=murf.render)).audio

    assert audio[:4] == b"RIFF", "Murf was asked for WAV; the cache must not re-encode it"

    with wave.open(io.BytesIO(audio), "rb") as clip:
        assert clip.getnchannels() == 1
        assert clip.getsampwidth() == 2
        assert clip.getframerate() == SAMPLE_RATE
        seconds = clip.getnframes() / clip.getframerate()
        assert 0.5 < seconds < 6, f"{SENTENCE!r} should be about a second, got {seconds:.2f}s"

        frames = clip.readframes(clip.getnframes())
        assert any(frames), "the clip decoded to pure silence"

    # The row describes the file the caller got, which is what eviction budgets rely on.
    entry = live_cache.index.get(live_cache.key_for(live_cache.spec(SENTENCE)))
    assert entry is not None
    assert entry.size_bytes == len(audio)
    assert entry.audio_format == "wav"
    assert Path(entry.audio_path).read_bytes() == audio


@needs_key
def test_a_hit_is_far_faster_than_murf(live_cache, murf):
    """The number the whole package exists for, measured rather than claimed."""
    # Self-sufficient, so this test means the same thing run alone as run third. It only
    # reaches Murf if nothing above it already did.
    live_cache.get_or_generate_sync(SENTENCE, generator_fn=murf.render)

    started = time.perf_counter()
    cached = live_cache.get_sync(SENTENCE)
    hit_ms = (time.perf_counter() - started) * 1000

    assert cached is not None
    assert hit_ms < 250, f"a cached read took {hit_ms:.0f} ms"
    print(f"\nMurf calls this run: {murf.calls}; cached read: {hit_ms:.1f} ms")
