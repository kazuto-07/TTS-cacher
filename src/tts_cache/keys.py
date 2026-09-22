"""Canonicalisation and the composite cache key.

The promise this module makes is narrow on purpose: two specs share a key only when they
would have produced byte-identical speech. Normalisation therefore only removes things a
TTS engine cannot hear — surrounding and duplicated whitespace, and the difference between
the composed and decomposed spellings of the same character. Case, punctuation and digits
are left exactly as written, because every one of them changes how a sentence is read.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from typing import Any

from .models import AudioSpec

__all__ = ["normalize_text", "cache_key", "key_payload", "Normalizer"]

Normalizer = Callable[[str], str]

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapses whitespace and normalises Unicode, and nothing else."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _canonical(value: Any) -> Any:
    """Makes settings values comparable across JSON round trips."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and value.is_integer():
        # 1.0 and 1 are the same rate; they must not be two cache entries.
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def key_payload(spec: AudioSpec, *, normalizer: Normalizer = normalize_text) -> dict[str, Any]:
    """The exact dictionary that gets hashed. Useful for debugging a surprising miss."""
    return {
        "v": 1,
        "text": normalizer(spec.text),
        "provider": spec.provider.strip().lower(),
        "voice_id": spec.voice_id.strip(),
        "model": (spec.model or "").strip(),
        "format": spec.audio_format.strip().lower().lstrip("."),
        "settings": _canonical(dict(spec.settings)),
    }


def cache_key(spec: AudioSpec, *, normalizer: Normalizer = normalize_text) -> str:
    """SHA-256 over the canonical payload, as 64 hex characters."""
    payload = key_payload(spec, normalizer=normalizer)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
