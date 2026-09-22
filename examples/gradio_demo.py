"""Same sentence, twice, side by side: straight to the vendor, versus through the cache.

    pip install gradio
    python examples/gradio_demo.py      # http://127.0.0.1:7860

Set one of these and the audio is real speech:

    ELEVENLABS_API_KEY=...      # optionally ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID
    MURF_API_KEY=...            # optionally MURF_VOICE_ID

Either may also sit in a `.env` at the project root. With neither, the demo still runs: a
stand-in waits like a hosted vendor and returns an audible tone.

urllib is enough for one POST, so the demo needs no HTTP dependency and no vendor SDK —
which is also the point: the cache wraps whatever call you already make.
"""

import array
import asyncio
import base64
import io
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import gradio as gr

from tts_cache import LocalStorage, SqliteIndex, TTSCache

VENDOR_LATENCY = 0.6  # what a hosted TTS call costs before the first byte arrives

#: Overridable so the demo can be pointed at a mock or a proxy.
ELEVENLABS_URL = os.environ.get("ELEVENLABS_URL", "https://api.elevenlabs.io/v1/text-to-speech")
MURF_URL = os.environ.get("MURF_URL", "https://api.murf.ai/v1/speech/generate")
MURF_SAMPLE_RATE = 24_000


def _env(name: str) -> str | None:
    """A variable from the environment, or from a .env at the project root."""
    value = os.environ.get(name)
    if value:
        return value.strip()

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, _, found = line.partition("=")
        if key.strip() == name:
            return found.strip().strip("\"'") or None
    return None


def _post(url: str, payload: dict, headers: dict) -> bytes:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"{url} returned HTTP {e.code}: {detail}") from e


def render_elevenlabs(text: str) -> bytes:
    """ElevenLabs hands back the mp3 as the response body."""
    return _post(
        f"{ELEVENLABS_URL}/{VOICE_ID}",
        {"text": text, "model_id": MODEL_ID},
        {"xi-api-key": API_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"},
    )


def render_murf(text: str) -> bytes:
    """Murf answers with JSON; base64 keeps the audio in it, so there is no second request."""
    body = json.loads(
        _post(
            MURF_URL,
            {
                "text": text,
                "voiceId": VOICE_ID,
                "format": "WAV",
                "sampleRate": MURF_SAMPLE_RATE,
                "channelType": "MONO",
                "encodeAsBase64": True,
            },
            {"api-key": API_KEY, "Content-Type": "application/json"},
        )
    )
    encoded = body.get("encodedAudio")
    if not encoded:
        raise RuntimeError(f"Murf returned no audio: {json.dumps(body)[:300]}")
    return base64.b64decode(encoded)


def render_stand_in(text: str) -> bytes:
    """No key, no network: a tone, after a wait a hosted vendor would have charged you."""
    time.sleep(VENDOR_LATENCY)
    samples = array.array("h", (int(math.sin(i / 18) * 9000) for i in range(int(24000 * 0.8))))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(samples.tobytes())
    return buffer.getvalue()


# Whichever key is present wins; ElevenLabs first when both are.
if _env("ELEVENLABS_API_KEY"):
    PROVIDER, API_KEY = "elevenlabs", _env("ELEVENLABS_API_KEY")
    VOICE_ID = _env("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"  # Rachel
    MODEL_ID = _env("ELEVENLABS_MODEL_ID") or "eleven_turbo_v2_5"
    AUDIO_FORMAT, SETTINGS, render = "mp3", {}, render_elevenlabs
elif _env("MURF_API_KEY"):
    PROVIDER, API_KEY = "murf", _env("MURF_API_KEY")
    VOICE_ID = _env("MURF_VOICE_ID") or "en-US-natalie"
    MODEL_ID = ""
    AUDIO_FORMAT = "wav"
    SETTINGS = {"sample_rate": MURF_SAMPLE_RATE, "channel_type": "MONO"}
    render = render_murf
else:
    PROVIDER, API_KEY, VOICE_ID, MODEL_ID = "stand-in", "", "narrator", ""
    AUDIO_FORMAT, SETTINGS, render = "wav", {}, render_stand_in


cache = TTSCache(
    storage=LocalStorage(".tts-cache/demo/audio"),
    index=SqliteIndex(".tts-cache/demo/index.db"),
    provider=PROVIDER,
    voice_id=VOICE_ID,
    model=MODEL_ID,
    audio_format=AUDIO_FORMAT,
    # Everything that changes how it sounds belongs in the key, or a clip could be served
    # at the wrong rate after you change one of these.
    settings=SETTINGS,
    background_writes=False,  # so the very next request is already a hit
)


def _as_file(audio: bytes) -> str:
    """Gradio plays from a path, and the extension has to match what the vendor sent."""
    handle, path = tempfile.mkstemp(suffix=f".{AUDIO_FORMAT}")
    with os.fdopen(handle, "wb") as f:
        f.write(audio)
    return path


async def compare(text: str):
    text = (text or "").strip()
    if not text:
        return None, "", None, "Type a sentence first."

    try:
        # Left: no cache at all. Every request is a vendor call.
        started = time.perf_counter()
        plain = await asyncio.to_thread(render, text)
        plain_ms = (time.perf_counter() - started) * 1000

        # Right: through the cache.
        started = time.perf_counter()
        result = await cache.get_or_generate(
            text, generator_fn=lambda: asyncio.to_thread(render, text)
        )
        cached_ms = (time.perf_counter() - started) * 1000
    except Exception as e:  # a bad key should say so, not blank the page
        return None, "", None, f"**{PROVIDER} failed.** {e}"

    if result.hit:
        speedup = plain_ms / max(cached_ms, 0.001)
        label = f"### {cached_ms:.0f} ms — cache hit\nNo vendor call. {speedup:.0f} times faster."
    else:
        label = f"### {cached_ms:.0f} ms — miss, so the vendor ran\nPress **Speak** again."

    return (
        _as_file(plain),
        f"### {plain_ms:.0f} ms — {PROVIDER}\nEvery time, for the same sentence.",
        _as_file(result.audio),
        label,
    )


async def reset():
    await cache.flush()
    return None, "", None, "Cache emptied — the next press is a miss again."


with gr.Blocks(title="tts-cache") as demo:
    gr.Markdown(
        "# tts-cache\n"
        "Press **Speak** twice on the same line and watch the right side.\n\n"
        f"**Vendor:** {PROVIDER} · voice `{VOICE_ID}`"
        + ("" if API_KEY else " — set `ELEVENLABS_API_KEY` or `MURF_API_KEY` for real speech.")
    )
    text = gr.Textbox(label="What the agent says", value="Let me pull that up for you.")
    with gr.Row():
        speak = gr.Button("Speak", variant="primary")
        clear = gr.Button("Empty the cache")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Without the cache")
            plain_audio = gr.Audio(label=None)
            plain_note = gr.Markdown()
        with gr.Column():
            gr.Markdown("## With tts-cache")
            cached_audio = gr.Audio(label=None, autoplay=True)
            cached_note = gr.Markdown()

    outputs = [plain_audio, plain_note, cached_audio, cached_note]
    speak.click(compare, inputs=text, outputs=outputs)
    clear.click(reset, inputs=None, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
