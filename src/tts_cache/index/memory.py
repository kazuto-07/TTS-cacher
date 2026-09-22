"""In-process index. The default, and what the tests run against."""

from __future__ import annotations

import threading
from dataclasses import replace

from ..models import CacheEntry
from .base import Index, sort_entries

__all__ = ["MemoryIndex"]


class MemoryIndex(Index):
    """Keeps rows in a dictionary, guarded by a lock so threads can share one cache."""

    name = "memory"

    def __init__(self) -> None:
        self._rows: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, cache_key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._rows.get(cache_key)
            # A copy, so a caller mutating the result cannot rewrite the index.
            return replace(entry) if entry else None

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._rows[entry.cache_key] = replace(entry)

    def touch(self, cache_key: str, when: float) -> None:
        with self._lock:
            entry = self._rows.get(cache_key)
            if entry is None:
                return
            entry.access_count += 1
            entry.last_accessed_at = when

    def delete(self, cache_key: str) -> CacheEntry | None:
        with self._lock:
            return self._rows.pop(cache_key, None)

    def list(
        self,
        *,
        provider: str | None = None,
        voice_id: str | None = None,
        order: str = "last_accessed_at",
        descending: bool = True,
        limit: int | None = None,
    ) -> list[CacheEntry]:
        with self._lock:
            rows = [replace(e) for e in self._rows.values()]
        if provider is not None:
            rows = [e for e in rows if e.provider == provider]
        if voice_id is not None:
            rows = [e for e in rows if e.voice_id == voice_id]
        rows = sort_entries(rows, order, descending)
        return rows[:limit] if limit is not None else rows

    def expired(self, cutoff: float) -> list[CacheEntry]:
        with self._lock:
            return [replace(e) for e in self._rows.values() if e.created_at < cutoff]

    def total_size(self) -> int:
        with self._lock:
            return sum(e.size_bytes for e in self._rows.values())

    def count(self) -> int:
        with self._lock:
            return len(self._rows)

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()
