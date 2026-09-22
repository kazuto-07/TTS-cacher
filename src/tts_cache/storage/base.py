"""The binary side of the cache: audio in, audio out, addressed by cache key."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

__all__ = ["Storage"]


class Storage(ABC):
    """Stores audio blobs.

    Drivers implement the four synchronous methods. The ``a``-prefixed coroutines are what
    the cache actually calls; the default versions push the blocking work onto a thread, so
    a driver built on a synchronous SDK (Supabase, boto3) needs no async code at all. A
    driver with a genuinely async client should override them.
    """

    #: Short name used in log lines and CLI output.
    name = "storage"

    @abstractmethod
    def put(self, cache_key: str, data: bytes, audio_format: str) -> str:
        """Writes the audio and returns the path or URI to read it back with."""

    @abstractmethod
    def get(self, path: str) -> bytes | None:
        """Reads audio back, or returns ``None`` if it is no longer there."""

    def iter(self, path: str, chunk_size: int = 8192) -> Iterator[bytes] | None:
        """Reads audio back in pieces, or returns ``None`` if it is no longer there.

        This is what decides the time to first audio of a cache hit: a driver that can
        hand over the first chunk before the last one has arrived lets the caller start
        playing then, instead of after the whole clip has been fetched. The default
        implementation cannot do that — it reads the blob and slices it — so a driver
        over a network should override this with a real streaming read.
        """
        data = self.get(path)
        if data is None:
            return None
        return iter(tuple(data[at : at + chunk_size] for at in range(0, len(data), chunk_size)))

    @abstractmethod
    def delete(self, path: str) -> None:
        """Removes one blob. Missing blobs are not an error."""

    @abstractmethod
    def clear(self) -> None:
        """Removes everything this storage owns."""

    async def aput(self, cache_key: str, data: bytes, audio_format: str) -> str:
        return await asyncio.to_thread(self.put, cache_key, data, audio_format)

    async def aget(self, path: str) -> bytes | None:
        return await asyncio.to_thread(self.get, path)

    async def aiter(self, path: str, chunk_size: int = 8192) -> AsyncIterator[bytes] | None:
        """``iter`` for the event loop: ``None`` if the blob is gone, chunks otherwise.

        Opening the stream and pulling each chunk both go to a thread, so a blocking
        driver never stalls the loop between chunks.
        """
        chunks = await asyncio.to_thread(self.iter, path, chunk_size)
        if chunks is None:
            return None
        return _drain(chunks)

    async def adelete(self, path: str) -> None:
        await asyncio.to_thread(self.delete, path)

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)

    def close(self) -> None:
        """Releases any connection the driver holds. Called by ``TTSCache.close()``."""


async def _drain(chunks: Iterator[bytes]) -> AsyncIterator[bytes]:
    """Pulls a blocking iterator from the event loop, one chunk per thread hop."""
    done = object()
    while True:
        chunk = await asyncio.to_thread(next, chunks, done)
        if chunk is done:
            return
        if chunk:
            yield chunk
