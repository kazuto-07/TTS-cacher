"""SQLAlchemy index — PostgreSQL, MySQL, SQLite, or anything else it drives.

This is the driver to use when several workers share one cache: the row is the source of
truth about what exists, so two machines rendering the same sentence converge on one clip.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..errors import MetadataError, MissingDependency
from ..models import CacheEntry
from .base import ORDERS, EvictionPolicy, Index

__all__ = ["SqlAlchemyIndex"]


def _sqlalchemy():
    try:
        import sqlalchemy as sa
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise MissingDependency("sqlalchemy", "sqlalchemy") from e
    return sa


def _table(sa: Any, metadata: Any, table_name: str):
    return sa.Table(
        table_name,
        metadata,
        sa.Column("cache_key", sa.String(64), primary_key=True),
        sa.Column("normalized_text", sa.Text, nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("voice_id", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False, server_default=""),
        sa.Column("audio_format", sa.String(10), nullable=False, server_default="mp3"),
        sa.Column("audio_path", sa.Text, nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("access_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Index(f"idx_{table_name}_accessed", "last_accessed_at"),
        sa.Index(f"idx_{table_name}_voice", "provider", "voice_id"),
        extend_existing=True,
    )


class SqlAlchemyIndex(Index):
    """Stores rows in any database SQLAlchemy supports.

    Pass a ``url`` (``postgresql+psycopg://…``, ``mysql+pymysql://…``, ``sqlite:///…``) or
    an ``engine`` the application already owns. ``create_table`` issues the DDL on the
    first connection; turn it off where migrations own the schema.
    """

    name = "sqlalchemy"

    def __init__(
        self,
        url: str | None = None,
        *,
        engine: Any | None = None,
        table_name: str = "tts_cache",
        create_table: bool = True,
        **engine_kwargs: Any,
    ) -> None:
        sa = _sqlalchemy()
        self._sa = sa
        if engine is None:
            if not url:
                raise MetadataError("SqlAlchemyIndex needs a url or an engine")
            engine = sa.create_engine(url, **engine_kwargs)
        self._engine = engine
        self._metadata = sa.MetaData()
        self.table = _table(sa, self._metadata, table_name)
        if create_table:
            try:
                self._metadata.create_all(self._engine, tables=[self.table])
            except Exception as e:
                raise MetadataError(f"could not create {table_name}: {e}") from e

    def _values(self, entry: CacheEntry) -> dict[str, Any]:
        return {
            "cache_key": entry.cache_key,
            "normalized_text": entry.normalized_text,
            "provider": entry.provider,
            "voice_id": entry.voice_id,
            "model": entry.model,
            "audio_format": entry.audio_format,
            "audio_path": entry.audio_path,
            "size_bytes": entry.size_bytes,
            "access_count": entry.access_count,
            "created_at": _dt(entry.created_at),
            "last_accessed_at": _dt(entry.last_accessed_at),
        }

    def get(self, cache_key: str) -> CacheEntry | None:
        sa = self._sa
        with self._engine.connect() as conn:
            row = (
                conn.execute(sa.select(self.table).where(self.table.c.cache_key == cache_key))
                .mappings()
                .first()
            )
        return _entry(row) if row else None

    def put(self, entry: CacheEntry) -> None:
        sa = self._sa
        values = self._values(entry)
        try:
            with self._engine.begin() as conn:
                # Portable upsert: the dialect-specific ON CONFLICT clauses buy nothing
                # here, because two writers racing on one key are writing the same audio.
                updated = conn.execute(
                    sa.update(self.table)
                    .where(self.table.c.cache_key == entry.cache_key)
                    .values(**values)
                ).rowcount
                if not updated:
                    conn.execute(sa.insert(self.table).values(**values))
        except Exception as e:
            if _is_duplicate(e):
                return
            raise MetadataError(f"index write failed: {e}") from e

    def touch(self, cache_key: str, when: float) -> None:
        sa = self._sa
        with self._engine.begin() as conn:
            conn.execute(
                sa.update(self.table)
                .where(self.table.c.cache_key == cache_key)
                .values(
                    access_count=self.table.c.access_count + 1,
                    last_accessed_at=_dt(when),
                )
            )

    def delete(self, cache_key: str) -> CacheEntry | None:
        sa = self._sa
        entry = self.get(cache_key)
        if entry is None:
            return None
        with self._engine.begin() as conn:
            conn.execute(sa.delete(self.table).where(self.table.c.cache_key == cache_key))
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
        sa = self._sa
        column = getattr(self.table.c, order, None)
        if column is None or order not in {
            "last_accessed_at",
            "created_at",
            "access_count",
            "size_bytes",
        }:
            raise ValueError(f"cannot order by {order!r}")
        query = sa.select(self.table)
        if provider is not None:
            query = query.where(self.table.c.provider == provider)
        if voice_id is not None:
            query = query.where(self.table.c.voice_id == voice_id)
        query = query.order_by(column.desc() if descending else column.asc())
        if limit is not None:
            query = query.limit(limit)
        with self._engine.connect() as conn:
            return [_entry(row) for row in conn.execute(query).mappings()]

    def expired(self, cutoff: float) -> list[CacheEntry]:
        sa = self._sa
        query = sa.select(self.table).where(self.table.c.created_at < _dt(cutoff))
        with self._engine.connect() as conn:
            return [_entry(row) for row in conn.execute(query).mappings()]

    def total_size(self) -> int:
        sa = self._sa
        with self._engine.connect() as conn:
            total = conn.execute(sa.select(sa.func.sum(self.table.c.size_bytes))).scalar()
        return int(total or 0)

    def count(self) -> int:
        sa = self._sa
        with self._engine.connect() as conn:
            return int(conn.execute(sa.select(sa.func.count()).select_from(self.table)).scalar())

    def eviction_candidates(self, policy: EvictionPolicy, free_bytes: int) -> list[CacheEntry]:
        order = ORDERS.get(policy)
        if order is None:
            raise ValueError(f"unknown eviction policy {policy!r}, expected one of {list(ORDERS)}")
        sa = self._sa
        column = getattr(self.table.c, order)
        chosen: list[CacheEntry] = []
        freed = 0
        with self._engine.connect() as conn:
            result = conn.execute(
                sa.select(self.table).order_by(column.asc(), self.table.c.created_at.asc())
            )
            # Streamed, so a large table is not loaded to free a few megabytes.
            for row in result.mappings():
                if freed >= free_bytes:
                    break
                entry = _entry(row)
                chosen.append(entry)
                freed += entry.size_bytes
        return chosen

    def clear(self) -> None:
        sa = self._sa
        with self._engine.begin() as conn:
            conn.execute(sa.delete(self.table))

    def close(self) -> None:
        self._engine.dispose()


def _dt(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _epoch(value: Any) -> float:
    if isinstance(value, datetime):
        # SQLite hands back naive datetimes even for timezone-aware columns.
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    return float(value)


def _entry(row: Any) -> CacheEntry:
    return CacheEntry(
        cache_key=row["cache_key"],
        normalized_text=row["normalized_text"],
        provider=row["provider"],
        voice_id=row["voice_id"],
        model=row["model"] or "",
        audio_format=row["audio_format"],
        audio_path=row["audio_path"],
        size_bytes=int(row["size_bytes"]),
        access_count=int(row["access_count"]),
        created_at=_epoch(row["created_at"]),
        last_accessed_at=_epoch(row["last_accessed_at"]),
    )


def _is_duplicate(error: Exception) -> bool:
    text = str(error).lower()
    return "unique" in text or "duplicate" in text
