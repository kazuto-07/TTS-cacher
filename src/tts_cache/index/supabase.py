"""Supabase (PostgREST) index.

Handy when the audio already lives in a Supabase bucket and adding a second database would
be silly. PostgREST has no aggregate call in the client, so ``total_size`` pages through the
size column instead; for a cache of tens of thousands of clips, prefer
:class:`~tts_cache.index.sqlalchemy.SqlAlchemyIndex` pointed at the same Postgres.

The table this driver expects (run once in the SQL editor)::

    create table tts_cache (
        cache_key text primary key,
        normalized_text text not null,
        provider text not null,
        voice_id text not null,
        model text not null default '',
        audio_format text not null default 'mp3',
        audio_path text not null,
        size_bytes bigint not null,
        access_count int not null default 1,
        created_at timestamptz not null default now(),
        last_accessed_at timestamptz not null default now()
    );
    create index idx_tts_cache_accessed on tts_cache(last_accessed_at);
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..errors import MetadataError, MissingDependency
from ..models import CacheEntry
from .base import Index

__all__ = ["SupabaseIndex"]

#: PostgREST caps a response; rows are read in pages of this size.
PAGE = 1000


class SupabaseIndex(Index):
    """One PostgREST table, reached with the Supabase client."""

    name = "supabase"

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        *,
        client: Any | None = None,
        table_name: str = "tts_cache",
    ) -> None:
        if client is None:
            try:
                from supabase import create_client
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise MissingDependency("supabase", "supabase") from e
            if not supabase_url or not supabase_key:
                raise MetadataError("supabase_url and supabase_key are required without a client")
            client = create_client(supabase_url, supabase_key)
        self._client = client
        self.table_name = table_name

    @property
    def _table(self):
        return self._client.table(self.table_name)

    def _rows(self, query) -> list[dict[str, Any]]:
        try:
            return list(query.execute().data or [])
        except Exception as e:
            raise MetadataError(f"supabase index query failed: {e}") from e

    def get(self, cache_key: str) -> CacheEntry | None:
        rows = self._rows(self._table.select("*").eq("cache_key", cache_key).limit(1))
        return _entry(rows[0]) if rows else None

    def put(self, entry: CacheEntry) -> None:
        payload = {
            "cache_key": entry.cache_key,
            "normalized_text": entry.normalized_text,
            "provider": entry.provider,
            "voice_id": entry.voice_id,
            "model": entry.model,
            "audio_format": entry.audio_format,
            "audio_path": entry.audio_path,
            "size_bytes": entry.size_bytes,
            "access_count": entry.access_count,
            "created_at": _iso(entry.created_at),
            "last_accessed_at": _iso(entry.last_accessed_at),
        }
        self._rows(self._table.upsert(payload, on_conflict="cache_key"))

    def touch(self, cache_key: str, when: float) -> None:
        # PostgREST cannot increment in place from the client, so the count is read first.
        # A lost update here costs an eviction ranking, never a wrong clip.
        current = self.get(cache_key)
        if current is None:
            return
        self._rows(
            self._table.update(
                {"access_count": current.access_count + 1, "last_accessed_at": _iso(when)}
            ).eq("cache_key", cache_key)
        )

    def delete(self, cache_key: str) -> CacheEntry | None:
        entry = self.get(cache_key)
        if entry is None:
            return None
        self._rows(self._table.delete().eq("cache_key", cache_key))
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
        entries: list[CacheEntry] = []
        start = 0
        while True:
            query = self._table.select("*")
            if provider is not None:
                query = query.eq("provider", provider)
            if voice_id is not None:
                query = query.eq("voice_id", voice_id)
            want = PAGE if limit is None else min(PAGE, limit - len(entries))
            query = query.order(order, desc=descending).range(start, start + want - 1)
            rows = self._rows(query)
            entries.extend(_entry(row) for row in rows)
            if len(rows) < want or (limit is not None and len(entries) >= limit):
                break
            start += want
        return entries

    def expired(self, cutoff: float) -> list[CacheEntry]:
        return [
            _entry(row)
            for row in self._rows(self._table.select("*").lt("created_at", _iso(cutoff)))
        ]

    def total_size(self) -> int:
        total = 0
        start = 0
        while True:
            rows = self._rows(self._table.select("size_bytes").range(start, start + PAGE - 1))
            total += sum(int(row["size_bytes"]) for row in rows)
            if len(rows) < PAGE:
                return total
            start += PAGE

    def count(self) -> int:
        try:
            response = self._table.select("cache_key", count="exact").limit(1).execute()
        except Exception as e:
            raise MetadataError(f"supabase index count failed: {e}") from e
        return int(getattr(response, "count", 0) or 0)

    def clear(self) -> None:
        # PostgREST refuses an unfiltered delete, and this filter matches every row.
        self._rows(self._table.delete().neq("cache_key", ""))


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _entry(row: dict[str, Any]) -> CacheEntry:
    return CacheEntry(
        cache_key=row["cache_key"],
        normalized_text=row["normalized_text"],
        provider=row["provider"],
        voice_id=row["voice_id"],
        model=row.get("model") or "",
        audio_format=row.get("audio_format") or "mp3",
        audio_path=row["audio_path"],
        size_bytes=int(row["size_bytes"]),
        access_count=int(row.get("access_count") or 1),
        created_at=_epoch(row["created_at"]),
        last_accessed_at=_epoch(row["last_accessed_at"]),
    )
