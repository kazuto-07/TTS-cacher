from pathlib import Path

import pytest

from tts_cache import TTSCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config: pytest.Config) -> None:
    """Refuses to run with the wrong rootdir, instead of failing strangely later.

    Started from a parent directory, pytest never reads this project's `pyproject.toml`:
    `asyncio_mode = "auto"` is off, so every async test fails with "async def functions are
    not natively supported", and the `network` marker is unregistered. That is a confusing
    way to learn you were one `cd` away, so say it plainly here.
    """
    if config.inipath is None or config.inipath.parent != PROJECT_ROOT:
        found = config.inipath or "nothing"
        raise pytest.UsageError(
            f"pytest loaded its settings from {found}, not {PROJECT_ROOT / 'pyproject.toml'}.\n"
            f"Run it from the project directory:\n"
            f"    cd {PROJECT_ROOT}\n"
            f"    pytest\n"
            f"or point it at the config: pytest -c {PROJECT_ROOT / 'pyproject.toml'}"
        )


@pytest.fixture
def cache():
    """An in-process cache with writes that land before the call returns."""
    cache = TTSCache(
        provider="testvendor",
        voice_id="voice-1",
        model="test-model",
        background_writes=False,
        on_error="raise",
    )
    yield cache
    cache.close()


class Vendor:
    """A stand-in TTS vendor that counts how often it was actually called."""

    def __init__(self, audio: bytes = b"audio-bytes"):
        self.audio = audio
        self.calls: list[str] = []

    def render(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio + text.encode()

    async def arender(self, text: str) -> bytes:
        return self.render(text)

    async def stream(self, text: str):
        self.calls.append(text)
        for part in (b"one-", b"two-", b"three"):
            yield part

    @property
    def count(self) -> int:
        return len(self.calls)


@pytest.fixture
def vendor():
    return Vendor()
