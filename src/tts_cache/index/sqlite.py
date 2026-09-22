"""SQLite index — the default for anything that should survive a restart."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

from ..errors import MetadataError
from ..models import CacheEntry
from .base import ORDERS, EvictionPolicy, Index

__all__ = ["SqliteIndex", "SCHEMA"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS tts_cache (
    cache_key       TEXT PRIMARY KEY,
    normalized_text TEXT NOT NULL,
    provider        TEXT NOT NULL,
    voice_id        TEXT NOT NULL,
    model           TEXT NOT NULL DEFAULT '',
    audio_format    TEXT NOT NULL DEFAULT 'mp3',
    audio_path      TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 1,
    created_at      REAL NOT NULL,
    last_accessed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tts_cache_accessed ON tts_cache(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_tts_cache_created ON tts_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_tts_cache_voice ON tts_cache(provider, voice_id);
"""

_COLUMNS = (
    "cache_key, normalized_text, provider, voice_id, model, audio_format, "
    "audio_path, size_bytes, access_count, created_at, last_accessed_at"
)


class SqliteIndex(Index):
    """One table, three indexes, one file.

    The connection is shared across threads with a lock rather than opened per thread: the
    cache writes from background workers, and SQLite's own locking would otherwise turn
    those into ``database is locked`` errors under load. WAL mode keeps a slow read from
    blocking the write that follows it.
    """

    name = "sqlite"

    def __init__(self, path: str | os.PathLike[str] = ".tts-cache/index.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._lock = threading.RLock()
        try:
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            self._db.commit()
        except sqlite3.Error as e:
            raise MetadataError(f"could not open the index at {self.path}: {e}") from e

    def _query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return self._db.execute(sql, args).fetchall()
            except sqlite3.Error as e:
                raise MetadataError(f"index query failed: {e}") from e

    def _write(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            try:
                self._db.execute(sql, args)
                self._db.commit()
            except sqlite3.Error as e:
                raise MetadataError(f"index write failed: {e}") from e

    def get(self, cache_key: str) -> CacheEntry | None:
        rows = self._query(f"SELECT {_COLUMNS} FROM tts_cache WHERE cache_key = ?", (cache_key,))
        return _entry(rows[0]) if rows else None

    def put(self, entry: CacheEntry) -> None:
        self._write(
            f"INSERT OR REPLACE INTO tts_cache ({_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.cache_key,
                entry.normalized_text,
                entry.provider,
                entry.voice_id,
                entry.model,
                entry.audio_format,
                entry.audio_path,
                entry.size_bytes,
                entry.access_count,
                entry.created_at,
                entry.last_accessed_at,
            ),
        )

    def touch(self, cache_key: str, when: float) -> None:
        self._write(
            "UPDATE tts_cache SET access_count = access_count + 1, last_accessed_at = ? "
            "WHERE cache_key = ?",
            (when, cache_key),
        )

    def delete(self, cache_key: str) -> CacheEntry | None:
        with self._lock:
            entry = self.get(cache_key)
            if entry is not None:
                self._write("DELETE FROM tts_cache WHERE cache_key = ?", (cache_key,))
            return entry

    def list(
        self,
        *,
        provider: str | None = None,
        voice_id: str | None = None,
        order: str = "last_accessed_at",
        descending: bool = True,
        limit: int | None = None,
    ) -> list[CacheEntry]:
        if order not in {"last_accessed_at", "created_at", "access_count", "size_bytes"}:
            raise ValueError(f"cannot order by {order!r}")
        where, args = [], []
        if provider is not None:
            where.append("provider = ?")
            args.append(provider)
        if voice_id is not None:
            where.append("voice_id = ?")
            args.append(voice_id)
        sql = f"SELECT {_COLUMNS} FROM tts_cache"
        if where:
            sql += " WHERE " + " AND ".join(where)
        # `order` is checked against a fixed set above, so this interpolation is safe.
        sql += f" ORDER BY {order} {'DESC' if descending else 'ASC'}, created_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        return [_entry(row) for row in self._query(sql, tuple(args))]

    def expired(self, cutoff: float) -> list[CacheEntry]:
        return [
            _entry(row)
            for row in self._query(
                f"SELECT {_COLUMNS} FROM tts_cache WHERE created_at < ? ORDER BY created_at",
                (cutoff,),
            )
        ]

    def total_size(self) -> int:
        rows = self._query("SELECT COALESCE(SUM(size_bytes), 0) AS total FROM tts_cache")
        return int(rows[0]["total"])

    def count(self) -> int:
        return int(self._query("SELECT COUNT(*) AS n FROM tts_cache")[0]["n"])

    def eviction_candidates(self, policy: EvictionPolicy, free_bytes: int) -> list[CacheEntry]:
        order = ORDERS.get(policy)
        if order is None:
            raise ValueError(f"unknown eviction policy {policy!r}, expected one of {list(ORDERS)}")
        chosen: list[CacheEntry] = []
        freed = 0
        # Ranked by the database, and only read until enough has been freed: a cache of a
        # million clips must not be pulled into memory to drop three of them.
        for row in self._query(
            f"SELECT {_COLUMNS} FROM tts_cache ORDER BY {order} ASC, created_at ASC"
        ):
            if freed >= free_bytes:
                break
            entry = _entry(row)
            chosen.append(entry)
            freed += entry.size_bytes
        return chosen

    def clear(self) -> None:
        self._write("DELETE FROM tts_cache")

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _entry(row: sqlite3.Row) -> CacheEntry:
    return CacheEntry(
        cache_key=row["cache_key"],
        normalized_text=row["normalized_text"],
        provider=row["provider"],
        voice_id=row["voice_id"],
        model=row["model"],
        audio_format=row["audio_format"],
        audio_path=row["audio_path"],
        size_bytes=int(row["size_bytes"]),
        access_count=int(row["access_count"]),
        created_at=float(row["created_at"]),
        last_accessed_at=float(row["last_accessed_at"]),
    )
