"""The two records everything else is phrased in: a voice spec, and a cached entry."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["AudioSpec", "CacheEntry"]


@dataclass(frozen=True, slots=True)
class AudioSpec:
    """Everything that decides what the audio sounds like.

    Two specs that compare equal must be safe to serve from one another's audio, so every
    knob that changes the rendering — voice, model, rate, pitch, format — belongs here, in
    ``settings`` if the provider names it something of its own.
    """

    text: str
    provider: str
    voice_id: str
    model: str | None = None
    audio_format: str = "mp3"
    settings: Mapping[str, Any] = field(default_factory=dict)

    def with_text(self, text: str) -> AudioSpec:
        return replace(self, text=text)


@dataclass(slots=True)
class CacheEntry:
    """One row of the index: where the audio lives and how it has been used."""

    cache_key: str
    normalized_text: str
    provider: str
    voice_id: str
    model: str
    audio_format: str
    audio_path: str
    size_bytes: int
    access_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    def age(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.created_at

    def idle(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.last_accessed_at
