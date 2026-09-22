"""In-process storage. The default, and what the tests run against."""

from __future__ import annotations

import threading

from .base import Storage

__all__ = ["MemoryStorage"]


class MemoryStorage(Storage):
    """Keeps audio in a dictionary; everything is lost when the process exits.

    Useful for tests and for a single long-lived worker where the point of the cache is
    latency rather than durability.
    """

    name = "memory"

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put(self, cache_key: str, data: bytes, audio_format: str) -> str:
        path = f"mem://{cache_key}.{audio_format}"
        with self._lock:
            self._blobs[path] = bytes(data)
        return path

    def get(self, path: str) -> bytes | None:
        with self._lock:
            return self._blobs.get(path)

    def delete(self, path: str) -> None:
        with self._lock:
            self._blobs.pop(path, None)

    def clear(self) -> None:
        with self._lock:
            self._blobs.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._blobs)
