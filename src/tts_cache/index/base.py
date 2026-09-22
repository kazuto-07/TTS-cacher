"""The metadata side of the cache: which clips exist, how big, how often used.

Splitting metadata from the audio itself is what makes "bring your own database" work — a
lookup is one indexed row read, and the blob is only fetched once the row says it is there.
It is also what makes eviction and the CLI possible without listing a bucket.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal

from ..models import CacheEntry

__all__ = ["Index", "EvictionPolicy", "ORDERS"]

EvictionPolicy = Literal["lru", "lfu", "fifo"]

#: How each policy ranks entries when deciding what goes first.
ORDERS: dict[str, str] = {
    "lru": "last_accessed_at",
    "lfu": "access_count",
    "fifo": "created_at",
}


class Index(ABC):
    """Stores one row per cached clip.

    Drivers implement the synchronous methods; the ``a``-prefixed coroutines default to
    running them on a thread, which is right for every database client that blocks.
    """

    name = "index"

    @abstractmethod
    def get(self, cache_key: str) -> CacheEntry | None:
        """The entry for this key, without counting it as a use."""

    @abstractmethod
    def put(self, entry: CacheEntry) -> None:
        """Inserts the entry, replacing any row with the same key."""

    @abstractmethod
    def touch(self, cache_key: str, when: float) -> None:
        """Records a hit: one more access, and a new last-accessed time."""

    @abstractmethod
    def delete(self, cache_key: str) -> CacheEntry | None:
        """Removes one row and returns it, so its audio can be deleted too."""

    @abstractmethod
    def list(
        self,
        *,
        provider: str | None = None,
        voice_id: str | None = None,
        order: str = "last_accessed_at",
        descending: bool = True,
        limit: int | None = None,
    ) -> list[CacheEntry]:
        """Rows matching the filters, most interesting first."""

    @abstractmethod
    def expired(self, cutoff: float) -> list[CacheEntry]:
        """Rows created before ``cutoff``."""

    @abstractmethod
    def total_size(self) -> int:
        """Bytes of audio the index believes are stored."""

    @abstractmethod
    def count(self) -> int:
        """How many clips are cached."""

    @abstractmethod
    def clear(self) -> None:
        """Removes every row."""

    def eviction_candidates(self, policy: EvictionPolicy, free_bytes: int) -> list[CacheEntry]:
        """The cheapest rows to lose, in order, until ``free_bytes`` is covered.

        Overriding this is optional: a driver that can rank rows in the database should,
        but the default walks the ordered listing, which every driver already provides.
        """
        order = ORDERS.get(policy)
        if order is None:
            raise ValueError(f"unknown eviction policy {policy!r}, expected one of {list(ORDERS)}")

        chosen: list[CacheEntry] = []
        freed = 0
        for entry in self.list(order=order, descending=False):
            if freed >= free_bytes:
                break
            chosen.append(entry)
            freed += entry.size_bytes
        return chosen

    # Async surface ---------------------------------------------------------------

    async def aget(self, cache_key: str) -> CacheEntry | None:
        return await asyncio.to_thread(self.get, cache_key)

    async def aput(self, entry: CacheEntry) -> None:
        await asyncio.to_thread(self.put, entry)

    async def atouch(self, cache_key: str, when: float) -> None:
        await asyncio.to_thread(self.touch, cache_key, when)

    async def adelete(self, cache_key: str) -> CacheEntry | None:
        return await asyncio.to_thread(self.delete, cache_key)

    async def alist(self, **kwargs) -> list[CacheEntry]:
        return await asyncio.to_thread(lambda: self.list(**kwargs))

    async def aexpired(self, cutoff: float) -> list[CacheEntry]:
        return await asyncio.to_thread(self.expired, cutoff)

    async def atotal_size(self) -> int:
        return await asyncio.to_thread(self.total_size)

    async def acount(self) -> int:
        return await asyncio.to_thread(self.count)

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)

    async def aeviction_candidates(
        self, policy: EvictionPolicy, free_bytes: int
    ) -> list[CacheEntry]:
        return await asyncio.to_thread(self.eviction_candidates, policy, free_bytes)

    def close(self) -> None:
        """Releases the connection. Called by ``TTSCache.close()``."""


def sort_entries(entries: Iterable[CacheEntry], order: str, descending: bool) -> list[CacheEntry]:
    """Shared ordering for drivers that sort in Python."""
    if order not in {"last_accessed_at", "created_at", "access_count", "size_bytes"}:
        raise ValueError(f"cannot order by {order!r}")
    # The key is ordered second so that equal ranks fall back to insertion age, which keeps
    # eviction stable when a hundred entries share an access count of 1.
    return sorted(
        entries,
        key=lambda e: (getattr(e, order), e.created_at),
        reverse=descending,
    )
