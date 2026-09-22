"""Metadata indexes. The optional drivers import lazily so their SDKs stay optional."""

from __future__ import annotations

from typing import Any

from .base import ORDERS, EvictionPolicy, Index
from .memory import MemoryIndex
from .sqlite import SCHEMA, SqliteIndex

__all__ = [
    "Index",
    "EvictionPolicy",
    "ORDERS",
    "MemoryIndex",
    "SqliteIndex",
    "SCHEMA",
    "SqlAlchemyIndex",
    "SupabaseIndex",
]


def __getattr__(name: str) -> Any:
    if name == "SqlAlchemyIndex":
        from .sqlalchemy import SqlAlchemyIndex

        return SqlAlchemyIndex
    if name == "SupabaseIndex":
        from .supabase import SupabaseIndex

        return SupabaseIndex
    raise AttributeError(name)
