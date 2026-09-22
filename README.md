# tts-cache

**Deterministic, content-addressed caching for text-to-speech in voice AI pipelines.**

A voice agent says the same things over and over — a greeting, a hold message, "sorry, I
didn't catch that". Every one of those costs a TTS call. `tts-cache` sits between the text
and the vendor and serves audio it has already paid for.

A clip is reused only when the text and every voice setting match exactly — the key is a
SHA-256 of the text plus the whole voice configuration — so there is no way to play the
wrong sentence at a caller. No framework dependency, no vendor client: it wraps the call
you already make.

## Install

```bash
pip install tts-cache                # core: local disk + SQLite, no dependencies
pip install "tts-cache[supabase]"    # Supabase bucket and table
```

Python 3.10 or newer.

## How to run it

Wrap the function that already turns a sentence into audio. A complete program:

```python
import asyncio
import base64

from murf import AsyncMurf

from tts_cache import LocalStorage, SqliteIndex, TTSCache

murf = AsyncMurf(api_key="...")

VOICE = "en-US-natalie"

cache = TTSCache(
    storage=LocalStorage("/var/cache/voice"),
    index=SqliteIndex("/var/cache/voice/index.db"),
)


@cache.memoize(provider="murf", voice_id=VOICE)
async def generate_speech(text: str) -> bytes:
    """Runs only on a miss. The cache reads the text from the first argument."""
    response = await murf.text_to_speech.generate(
        text=text,
        voice_id=VOICE,
        format="MP3",
        # Base64 keeps the audio in the response, so there is no second request to a CDN
        # for a URL that expires.
        encode_as_base64=True,
    )
    return base64.b64decode(response.encoded_audio)


async def main() -> None:
    audio = await generate_speech("Your balance is ready.")  # vendor call

    # Writes happen on a worker thread so the caller is never blocked by them. In an agent
    # the next call is seconds later and the write has long landed; in a script this tight
    # you have to wait for it, or the second call is another miss.
    await cache.drain()

    audio = await generate_speech("Your balance is ready.")  # milliseconds, no vendor call
    print(len(audio), "bytes,", f"{cache.stats.hit_rate:.0%} hit rate")

    await cache.aclose()


asyncio.run(main())
```

The decorator keeps the original function on `generate_speech.uncached` and the cache on
`generate_speech.cache`, which is what you want in tests and when a voice needs
re-rendering.

### Or call the cache directly

`generator_fn` takes no arguments — bind the text into it — and is awaited only on a miss.
The result unpacks as `audio, hit`, and also carries `.key`, `.entry` and `.elapsed_ms`.

```python
import asyncio

from openai import AsyncOpenAI

from tts_cache import LocalStorage, SqliteIndex, TTSCache

openai = AsyncOpenAI()

cache = TTSCache(
    storage=LocalStorage("/var/cache/voice"),
    index=SqliteIndex("/var/cache/voice/index.db"),
    provider="openai",  # defaults for every call, so each call site stays short
    voice_id="alloy",
    model="gpt-4o-mini-tts",
)


async def render(text: str) -> bytes:
    response = await openai.audio.speech.create(
        model="gpt-4o-mini-tts", voice="alloy", input=text, response_format="mp3"
    )
    return response.content


async def main() -> None:
    sentence = "Your balance is ready."
    audio, hit = await cache.get_or_generate(sentence, generator_fn=lambda: render(sentence))
    print(f"{len(audio)} bytes, {'cache' if hit else 'vendor'}")
    await cache.aclose()


asyncio.run(main())
```

### Or stream it

The surface a voice pipeline wants, because it is judged on the first chunk rather than the
last. On a hit the audio streams out of the store — the first chunk leaves as soon as the
store has read that much, not after the whole clip arrives. On a miss the vendor's chunks
are forwarded as they come, so the caller starts playing at the vendor's own speed.

```python
async def speak(sentence: str, websocket) -> None:
    async def render_stream():
        """An async iterator of bytes. No arguments — close over the text."""
        async for chunk in vendor.stream(text=sentence, voice="alloy"):
            yield chunk

    async for chunk in cache.stream_or_generate(sentence, generator_fn=render_stream):
        await websocket.send_bytes(chunk)
```

For raw PCM add `frame_bytes=2 * num_channels` so a chunk never ends mid-sample, and
`bytes_per_second` if the transport should be fed at the speed the audio is spoken. Both
are shown in the LiveKit and Pipecat sections below. Time to first audio lands in
`cache.stats` as `hit_ttfa_ms` and `miss_ttfa_ms`.

### Outside async

Every method has a `_sync` twin — `get_sync`, `get_or_generate_sync`, `put_sync` — and
`@cache.memoize()` gives a plain function a plain wrapper, so synchronous code needs no
other changes. The `_sync` methods block, so inside a running event loop use the async
ones; they raise rather than quietly stalling your loop if you forget.

### Try it now, with no keys

```bash
python examples/quickstart.py     # latency of every line, against a stand-in vendor

pip install gradio
python examples/gradio_demo.py    # http://127.0.0.1:7860
```

The [Gradio demo](examples/gradio_demo.py) puts the two side by side in a browser: the same
sentence straight to the vendor on the left, through the cache on the right. Press **Speak**
twice — the left side pays every time, the right side pays once. Set `ELEVENLABS_API_KEY` or
`MURF_API_KEY` for real speech; with neither, a stand-in waits like a hosted vendor and
returns a tone so it still runs.

[`examples/voice_agent.py`](examples/voice_agent.py) is the same idea shaped like an agent
turn: split into sentences, cache, stream.

## Storage

Audio and metadata are kept separately: a lookup is one indexed row read rather than a
bucket listing.

### A local folder

```python
from tts_cache import TTSCache, LocalStorage, SqliteIndex

cache = TTSCache(
    storage=LocalStorage("/var/cache/voice"),  # sharded, atomic writes
    index=SqliteIndex("/var/cache/voice/index.db"),  # WAL, one file
)
```

That is the whole setup — both are created on first use and neither needs a dependency.

### Supabase

For several machines sharing one cache. Install with `pip install "tts-cache[supabase]"`,
create a bucket, then:

```python
from tts_cache import TTSCache, SupabaseStorage, SupabaseIndex

url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]

cache = TTSCache(
    storage=SupabaseStorage(url, key, bucket_name="voice-cache"),
    index=SupabaseIndex(url, key, table_name="tts_cache"),
)
```

The table is not created for you — run this once in the SQL editor:

```sql
CREATE TABLE tts_cache (
    cache_key        VARCHAR(64) PRIMARY KEY,   -- SHA-256 composite key
    normalized_text  TEXT NOT NULL,
    provider         VARCHAR(50) NOT NULL,
    voice_id         VARCHAR(100) NOT NULL,
    model            VARCHAR(100) NOT NULL DEFAULT '',
    audio_format     VARCHAR(10) NOT NULL DEFAULT 'mp3',
    audio_path       TEXT NOT NULL,             -- bucket URI or local path
    size_bytes       BIGINT NOT NULL,
    access_count     INT NOT NULL DEFAULT 1,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tts_cache_accessed ON tts_cache(last_accessed_at);
CREATE INDEX idx_tts_cache_created  ON tts_cache(created_at);
CREATE INDEX idx_tts_cache_voice    ON tts_cache(provider, voice_id);
```

Storage and index are independent, so a Supabase bucket with a local SQLite index — or the
other way round — is a perfectly ordinary setup.

## LiveKit Agents

Cache the phrases you repeat and hand the audio straight to `say()`. No subclass, no
plugin wrapper — `say(audio=...)` takes an `AsyncIterable[rtc.AudioFrame]`, which is what
the cache already yields:

```python
from livekit import rtc

RATE, CHANNELS = 24000, 1
cache = TTSCache(..., audio_format="pcm", settings={"sample_rate": RATE})


async def say(session, text: str) -> None:
    async def frames():
        async for pcm in cache.stream_or_generate(
            text,
            generator_fn=lambda: my_tts.synthesize(text),
            frame_bytes=2 * CHANNELS,  # never split a 16-bit sample
        ):
            yield rtc.AudioFrame(pcm, RATE, CHANNELS, len(pcm) // (2 * CHANNELS))

    await session.say(text, audio=frames())


# in your entrypoint, or an Agent.on_enter
await say(session, "Hi, you've reached Northwind Support. How can I help?")
```

That covers greetings, hold messages and confirmations — the repeated text that makes up
most of a call, and all of it text you know in advance.
[`examples/livekit_say.py`](examples/livekit_say.py) is that as a runnable agent.

To cache **every** reply instead of a fixed set, the cache belongs in `tts_node`, where the
reply becomes audio — [`examples/livekit_tts_node.py`](examples/livekit_tts_node.py), which
pays for sentence splitting and barge-in handling to get there.

LiveKit hands you raw PCM, which carries no header, so keep the rate in `settings` — change
the transport's rate and you simply miss instead of playing a clip at the wrong speed.

## Pipecat

Pipecat already aggregates tokens into sentences before calling `run_tts`, so the seam is
handed to you. Subclass the service you already use and override one method:

```python
class CachedTTS(CartesiaTTSService):  # or any Pipecat TTS service
    async def run_tts(self, text, context_id):
        cached = await cache.stream(text, provider="cartesia", voice_id=VOICE)
        if cached is not None:
            async for chunk in cached:
                yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
            return

        rendered = []
        async for frame in super().run_tts(text, context_id):
            if isinstance(frame, TTSAudioRawFrame):
                rendered.append(frame.audio)
            yield frame
        await cache.put(text, b"".join(rendered), provider="cartesia", voice_id=VOICE)
```

Then use `CachedTTS(...)` wherever the plain service went — the pipeline does not change.
`cache.stream` returns `None` before any audio when it is a miss, so your own TTFB metric
stays honest. [`examples/pipecat_cached_tts.py`](examples/pipecat_cached_tts.py) is the
runnable version, with metrics, interruption handling and a rule that keeps one-off
sentences out of the cache.

## License

MIT.
