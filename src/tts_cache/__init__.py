"""tts-cache — deterministic caching for text-to-speech in voice pipelines.

    from tts_cache import TTSCache

    cache = TTSCache(provider="elevenlabs", voice_id="21m00Tcm4TlvDq8ikWAM")
    audio, hit = await cache.get_or_generate("Your balance is ready.", generator_fn=render)

A clip is reused only when the text and every voice setting match exactly, so a hit is
always the audio the vendor would have returned.
"""

from __future__ import annotations

from .cache import CacheResult, CacheStats, TTSCache
from .errors import (
    ConfigurationError,
    MetadataError,
    MissingDependency,
    StorageError,
    TTSCacheError,
)
from .index import Index, MemoryIndex, SqliteIndex
from .keys import cache_key, key_payload, normalize_text
from .models import AudioSpec, CacheEntry
from .storage import LocalStorage, MemoryStorage, Storage

__version__ = "0.1.0"


def __getattr__(name: str):
    """The optional drivers, importable from the top level without their SDKs installed."""
    if name in {"SupabaseStorage", "S3Storage"}:
        from . import storage

        return getattr(storage, name)
    if name in {"SqlAlchemyIndex", "SupabaseIndex"}:
        from . import index

        return getattr(index, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TTSCache",
    "CacheResult",
    "CacheStats",
    "AudioSpec",
    "CacheEntry",
    "cache_key",
    "key_payload",
    "normalize_text",
    "Storage",
    "MemoryStorage",
    "LocalStorage",
    "Index",
    "MemoryIndex",
    "SqliteIndex",
    "TTSCacheError",
    "StorageError",
    "MetadataError",
    "ConfigurationError",
    "MissingDependency",
    "SupabaseStorage",
    "S3Storage",
    "SqlAlchemyIndex",
    "SupabaseIndex",
    "__version__",
]
