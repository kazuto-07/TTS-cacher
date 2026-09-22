"""The cache itself: look up, serve, generate, and write back out of the caller's way."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from .errors import ConfigurationError, TTSCacheError
from .index import Index, MemoryIndex
from .index.base import ORDERS, EvictionPolicy
from .keys import Normalizer, cache_key, normalize_text
from .models import AudioSpec, CacheEntry
from .storage import MemoryStorage, Storage

__all__ = ["TTSCache", "CacheResult", "CacheStats"]

log = logging.getLogger("tts_cache")

#: What a generator may hand back: bytes, an awaitable of bytes, or a stream of chunks.
Generator = Callable[..., Any]

MB = 1024 * 1024


@dataclass(slots=True)
class CacheStats:
    """Counters for one cache instance, since the process started."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    coalesced: int = 0
    errors: int = 0
    bytes_served: int = 0
    bytes_written: int = 0

    #: Streams served, and the time to first audio they added up to. A voice pipeline is
    #: judged on the first byte, not the last, so this is the number to watch.
    hit_streams: int = 0
    miss_streams: int = 0
    hit_ttfa_total_ms: float = 0.0
    miss_ttfa_total_ms: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def hit_ttfa_ms(self) -> float:
        """Mean time to first audio when the clip was already cached."""
        return self.hit_ttfa_total_ms / self.hit_streams if self.hit_streams else 0.0

    @property
    def miss_ttfa_ms(self) -> float:
        """Mean time to first audio when the vendor had to render it."""
        return self.miss_ttfa_total_ms / self.miss_streams if self.miss_streams else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "writes": self.writes,
            "evictions": self.evictions,
            "coalesced": self.coalesced,
            "errors": self.errors,
            "bytes_served": self.bytes_served,
            "bytes_written": self.bytes_written,
            "hit_ttfa_ms": round(self.hit_ttfa_ms, 2),
            "miss_ttfa_ms": round(self.miss_ttfa_ms, 2),
        }


class CacheResult:
    """Audio, and whether it came from the cache.

    Unpacks as ``audio, hit = await cache.get_or_generate(...)`` and still carries the key,
    the index row and how long the call took for anyone who wants to log them.
    """

    __slots__ = ("audio", "elapsed_ms", "entry", "hit", "key")

    def __init__(
        self,
        audio: bytes,
        hit: bool,
        key: str,
        entry: CacheEntry | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.audio = audio
        self.hit = hit
        self.key = key
        self.entry = entry
        self.elapsed_ms = elapsed_ms

    def __iter__(self):
        return iter((self.audio, self.hit))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: int):
        return (self.audio, self.hit)[item]

    def __repr__(self) -> str:
        source = "hit" if self.hit else "miss"
        return (
            f"CacheResult({source}, {len(self.audio)} bytes, "
            f"key={self.key[:12]}…, {self.elapsed_ms:.1f} ms)"
        )


class TTSCache:
    """Serves TTS audio from a store, and calls the vendor only when it has to.

    ``storage`` holds the audio, ``index`` holds one row per clip. Both default to in-process
    implementations, which makes the cache useful in a test or a notebook with no arguments
    at all; a real deployment passes a durable pair.

    Writes happen on a small thread pool after the audio has already been handed back, so a
    miss costs the vendor call and nothing else. Every failure inside the cache is logged and
    counted rather than raised, unless ``on_error="raise"``: a broken cache should make a
    pipeline slower, never broken.
    """

    def __init__(
        self,
        storage: Storage | None = None,
        index: Index | None = None,
        *,
        max_storage_size_mb: float | None = None,
        time_to_expire: float | None = None,
        eviction_policy: EvictionPolicy = "lru",
        provider: str | None = None,
        voice_id: str | None = None,
        model: str | None = None,
        audio_format: str = "mp3",
        settings: Mapping[str, Any] | None = None,
        normalizer: Normalizer = normalize_text,
        min_bytes: int = 1,
        max_bytes: int | None = None,
        should_cache: Callable[[AudioSpec], bool] | None = None,
        background_writes: bool = True,
        coalesce: bool = True,
        on_error: str = "log",
        max_workers: int = 2,
    ) -> None:
        if eviction_policy not in ORDERS:
            raise ConfigurationError(
                f"unknown eviction policy {eviction_policy!r}, expected one of {list(ORDERS)}"
            )
        if on_error not in {"log", "raise"}:
            raise ConfigurationError("on_error must be 'log' or 'raise'")

        # `is None`, not `or`: an empty store is falsy, and swapping the caller's out
        # from under them would be a cache that silently never hits.
        self.storage = MemoryStorage() if storage is None else storage
        self.index = MemoryIndex() if index is None else index
        self.max_storage_bytes = (
            int(max_storage_size_mb * MB) if max_storage_size_mb is not None else None
        )
        self.time_to_expire = time_to_expire
        self.eviction_policy: EvictionPolicy = eviction_policy
        self.defaults = AudioSpec(
            text="",
            provider=provider or "",
            voice_id=voice_id or "",
            model=model,
            audio_format=audio_format,
            settings=dict(settings or {}),
        )
        self.normalizer = normalizer
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.should_cache = should_cache
        self.background_writes = background_writes
        self.coalesce = coalesce
        self.on_error = on_error

        self.stats = CacheStats()
        self._inflight: dict[tuple[int, str], asyncio.Future] = {}
        self._stats_lock = threading.Lock()
        self._pending: set[Future] = set()
        self._pending_lock = threading.Lock()
        self._workers = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tts-cache")
        self._closed = False
        self._loop_thread: _LoopThread | None = None

    # Spec plumbing -------------------------------------------------------------------

    def spec(
        self,
        text: str,
        *,
        provider: str | None = None,
        voice_id: str | None = None,
        model: str | None = None,
        audio_format: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> AudioSpec:
        """Fills in whatever the call left out from the cache's defaults."""
        merged = dict(self.defaults.settings)
        if settings:
            merged.update(settings)
        spec = AudioSpec(
            text=text,
            provider=provider or self.defaults.provider,
            voice_id=voice_id or self.defaults.voice_id,
            model=model if model is not None else self.defaults.model,
            audio_format=audio_format or self.defaults.audio_format,
            settings=merged,
        )
        if not spec.provider or not spec.voice_id:
            raise ConfigurationError(
                "provider and voice_id are required — pass them to the call or to TTSCache(...)"
            )
        return spec

    def key_for(self, spec: AudioSpec) -> str:
        """The cache key for a spec. Two specs with this key share audio."""
        return cache_key(spec, normalizer=self.normalizer)

    # Reading -------------------------------------------------------------------------

    async def get(self, text: str, **spec_kwargs: Any) -> bytes | None:
        """Cached audio for this text, or ``None``. Counts as a hit or a miss."""
        result = await self._lookup(self.spec(text, **spec_kwargs))
        return result[0]

    async def _lookup(self, spec: AudioSpec) -> tuple[bytes | None, str, CacheEntry | None]:
        key = self.key_for(spec)
        try:
            entry = await self.index.aget(key)
            if entry is None:
                self._count(misses=1)
                return None, key, None

            if self._is_expired(entry):
                log.debug("tts-cache: %s expired", key[:12])
                await self._forget(key)
                self._count(misses=1)
                return None, key, None

            audio = await self.storage.aget(entry.audio_path)
            if audio is None:
                # The row outlived its audio — someone emptied the bucket behind us.
                log.warning("tts-cache: %s missing at %s", key[:12], entry.audio_path)
                await self._forget(key)
                self._count(misses=1)
                return None, key, None

            await self.index.atouch(key, time.time())
            self._count(hits=1, bytes_served=len(audio))
            return audio, key, entry
        except Exception as e:
            self._failed("lookup", e)
            self._count(misses=1)
            return None, key, None

    async def _lookup_entry(self, spec: AudioSpec) -> tuple[CacheEntry | None, str]:
        """The row only — no audio, so a streaming read can start on the first chunk.

        Counting is left to the caller, which knows whether the audio behind the row
        turned out to be readable.
        """
        key = self.key_for(spec)
        try:
            entry = await self.index.aget(key)
            if entry is None:
                return None, key
            if self._is_expired(entry):
                log.debug("tts-cache: %s expired", key[:12])
                await self._forget(key)
                return None, key
            return entry, key
        except Exception as e:
            self._failed("lookup", e)
            return None, key

    def _is_expired(self, entry: CacheEntry) -> bool:
        return self.time_to_expire is not None and entry.age() > self.time_to_expire

    async def get_or_generate(
        self,
        text: str,
        *,
        generator_fn: Generator,
        provider: str | None = None,
        voice_id: str | None = None,
        model: str | None = None,
        audio_format: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> CacheResult:
        """Serves the clip from the cache, or renders it with ``generator_fn`` and stores it.

        ``generator_fn`` is called only on a miss. It may be sync or async, take the text or
        take nothing, and return bytes or an iterable of chunks.
        """
        started = time.perf_counter()
        spec = self.spec(
            text,
            provider=provider,
            voice_id=voice_id,
            model=model,
            audio_format=audio_format,
            settings=settings,
        )
        audio, key, entry = await self._lookup(spec)
        if audio is not None:
            return CacheResult(audio, True, key, entry, _ms(started))

        audio = await self._generate(spec, key, generator_fn)
        return CacheResult(audio, False, key, None, _ms(started))

    async def _generate(self, spec: AudioSpec, key: str, generator_fn: Generator) -> bytes:
        """Renders a clip, once, however many callers ask for it at the same moment.

        Two callers reaching the same uncached sentence together is the normal case at the
        start of a call — several sessions hit the same greeting — and without this they
        would each pay for the same render. The second caller waits on the first one's
        result instead, and only the caller that did the work writes it back.
        """
        if not self.coalesce:
            audio = await _call_generator(generator_fn, spec.text)
            self._schedule_write(spec, key, audio)
            return audio

        loop = asyncio.get_running_loop()
        # Keyed by loop as well, because a future cannot be awaited from another one.
        waiting = (id(loop), key)
        running = self._inflight.get(waiting)
        if running is not None:
            self._count(coalesced=1)
            # Shielded: this caller going away must not cancel the render the others want.
            return await asyncio.shield(running)

        future: asyncio.Future = loop.create_future()
        self._inflight[waiting] = future
        try:
            audio = await _call_generator(generator_fn, spec.text)
        except BaseException as e:
            future.set_exception(e)
            # Marks it retrieved, so a failure nobody waited on is not also a warning.
            future.exception()
            raise
        else:
            future.set_result(audio)
            self._schedule_write(spec, key, audio)
            return audio
        finally:
            self._inflight.pop(waiting, None)

    async def stream(
        self,
        text: str,
        *,
        chunk_size: int = 8192,
        frame_bytes: int | None = None,
        bytes_per_second: float | None = None,
        **spec_kwargs: Any,
    ) -> AsyncIterator[bytes] | None:
        """Cached audio as chunks, or ``None`` if it is not cached. A hit or a miss.

        The streaming twin of ``get``, for a pipeline that wants to keep its own vendor
        path — a Pipecat service forwarding its inner service's frames, say. ``None``
        comes back before any audio does, so the caller can report its own time to first
        byte honestly instead of measuring a whole-clip read.
        """
        spec = self.spec(text, **spec_kwargs)
        started = time.perf_counter()
        stream, _ = await self._open(spec, chunk_size)
        if stream is None:
            self._count(misses=1)
            return None
        self._count(hits=1)
        return self._served(stream, started, chunk_size, frame_bytes, bytes_per_second)

    async def _open(
        self, spec: AudioSpec, chunk_size: int
    ) -> tuple[AsyncIterator[bytes] | None, str]:
        """Opens a read of the cached audio without counting it, or gives back ``None``."""
        entry, key = await self._lookup_entry(spec)
        if entry is None:
            return None, key
        try:
            stream = await self.storage.aiter(entry.audio_path, chunk_size)
        except Exception as e:
            self._failed("stream", e)
            return None, key
        if stream is None:
            # The row outlived its audio — someone emptied the bucket behind us.
            log.warning("tts-cache: %s missing at %s", key[:12], entry.audio_path)
            await self._forget(key)
            return None, key
        # An index write before the first chunk would show up as time to first audio, so
        # the touch goes to a worker and the audio goes out now.
        self._touch_later(key)
        return stream, key

    async def _served(
        self,
        stream: AsyncIterator[bytes],
        started: float,
        chunk_size: int,
        frame_bytes: int | None,
        bytes_per_second: float | None,
    ) -> AsyncIterator[bytes]:
        """Shapes a hit's chunks on the way out, and records what it cost."""
        served = 0
        async for chunk in _shape(stream, frame_bytes, bytes_per_second, started, chunk_size):
            if not served:
                self._record_ttfa(hit=True, started=started)
            served += len(chunk)
            yield chunk
        self._count(bytes_served=served)

    async def stream_or_generate(
        self,
        text: str,
        *,
        generator_fn: Generator,
        chunk_size: int = 8192,
        frame_bytes: int | None = None,
        bytes_per_second: float | None = None,
        **spec_kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Yields the clip as chunks, from the cache or from the vendor.

        This is the surface a voice pipeline should use, because it is judged on the first
        chunk rather than the last:

        *   **On a hit** the audio is streamed out of the store, so the first chunk leaves
            as soon as the store has read that much — not after the whole clip has been
            fetched. Over a bucket that is the difference between a few milliseconds and
            the download time of the entire clip.
        *   **On a miss** the vendor's chunks are forwarded as they arrive, so the caller
            starts playing at the vendor's own time to first byte. The assembled clip is
            written afterwards, and a stream the caller abandons half way is not cached.

        ``frame_bytes`` re-chunks the output onto whole audio frames — pass
        ``sample_width * num_channels`` for raw PCM, and no chunk will ever split a frame
        and misalign the rest of the stream. ``bytes_per_second`` paces the chunks at the
        speed the audio is spoken, for a transport that would otherwise be flooded by a
        hit arriving all at once; the first chunk is never delayed by it.

        Whichever path ran, the time to first audio lands in ``cache.stats``.
        """
        spec = self.spec(text, **spec_kwargs)
        started = time.perf_counter()
        stream, key = await self._open(spec, chunk_size)

        if stream is not None:
            self._count(hits=1)
            async for chunk in self._served(
                stream, started, chunk_size, frame_bytes, bytes_per_second
            ):
                yield chunk
            return

        self._count(misses=1)
        parts: list[bytes] = []
        complete = False
        first = True
        try:
            source = _iter_generator(generator_fn, spec.text)
            async for chunk in _shape(source, frame_bytes, bytes_per_second, started, chunk_size):
                if first:
                    first = False
                    self._record_ttfa(hit=False, started=started)
                parts.append(chunk)
                yield chunk
            complete = True
        finally:
            if complete:
                self._schedule_write(spec, key, b"".join(parts))

    def _record_ttfa(self, *, hit: bool, started: float) -> None:
        elapsed = (time.perf_counter() - started) * 1000
        if hit:
            self._count(hit_streams=1, hit_ttfa_total_ms=elapsed)
        else:
            self._count(miss_streams=1, miss_ttfa_total_ms=elapsed)

    def _touch_later(self, key: str) -> None:
        """Updates the LRU timestamp off the hot path; a failed touch is not a failed hit."""
        if self._closed:
            return
        # The pool refuses new work once it is shutting down; a missed touch is harmless.
        with contextlib.suppress(RuntimeError):
            self._workers.submit(_suppress, self.index.touch, key, time.time())

    # Writing -------------------------------------------------------------------------

    async def put(self, text: str, audio: bytes, **spec_kwargs: Any) -> CacheEntry | None:
        """Stores audio the caller rendered elsewhere. Waits for the write to land."""
        spec = self.spec(text, **spec_kwargs)
        return await asyncio.to_thread(self._write, spec, self.key_for(spec), audio)

    def _schedule_write(self, spec: AudioSpec, key: str, audio: bytes) -> Future | None:
        if not self._cacheable(spec, audio):
            return None
        if not self.background_writes:
            self._write(spec, key, audio)
            return None

        future = self._workers.submit(self._write, spec, key, audio)
        with self._pending_lock:
            self._pending.add(future)
        future.add_done_callback(self._forget_future)
        return future

    def _forget_future(self, future: Future) -> None:
        with self._pending_lock:
            self._pending.discard(future)

    def _cacheable(self, spec: AudioSpec, audio: bytes) -> bool:
        if self._closed:
            return False
        if audio is None or len(audio) < self.min_bytes:
            return False
        if self.max_bytes is not None and len(audio) > self.max_bytes:
            log.debug("tts-cache: %d bytes is over max_bytes, not cached", len(audio))
            return False
        return self.should_cache is None or self.should_cache(spec)

    def _write(self, spec: AudioSpec, key: str, audio: bytes) -> CacheEntry | None:
        """Runs on a worker thread: blob first, row second, then eviction."""
        if not self._cacheable(spec, audio):
            return None
        audio_format = spec.audio_format.lstrip(".").lower()
        try:
            path = self.storage.put(key, audio, audio_format)
        except Exception as e:
            self._failed("storage write", e)
            return None

        now = time.time()
        entry = CacheEntry(
            cache_key=key,
            normalized_text=self.normalizer(spec.text),
            provider=spec.provider,
            voice_id=spec.voice_id,
            model=spec.model or "",
            audio_format=audio_format,
            audio_path=path,
            size_bytes=len(audio),
            access_count=1,
            created_at=now,
            last_accessed_at=now,
        )
        try:
            # The row goes in last, so a key is never indexed before its audio is readable.
            self.index.put(entry)
        except Exception as e:
            self._failed("index write", e)
            _suppress(self.storage.delete, path)
            return None

        self._count(writes=1, bytes_written=len(audio))
        self._enforce_size_limit()
        return entry

    def _enforce_size_limit(self) -> None:
        if self.max_storage_bytes is None:
            return
        try:
            total = self.index.total_size()
            if total <= self.max_storage_bytes:
                return
            self._evict(total - self.max_storage_bytes)
        except Exception as e:
            self._failed("eviction", e)

    def _evict(self, free_bytes: int) -> int:
        """Drops the cheapest entries until ``free_bytes`` has been freed. Returns the count."""
        dropped = 0
        for candidate in self.index.eviction_candidates(self.eviction_policy, free_bytes):
            entry = self.index.delete(candidate.cache_key)
            if entry is None:
                continue
            _suppress(self.storage.delete, entry.audio_path)
            dropped += 1
        if dropped:
            self._count(evictions=dropped)
            log.info("tts-cache: evicted %d entries (%s)", dropped, self.eviction_policy)
        return dropped

    async def _forget(self, key: str) -> None:
        """Removes a row and its audio, used when an entry turns out to be unusable."""
        entry = await self.index.adelete(key)
        if entry is not None:
            await asyncio.to_thread(_suppress, self.storage.delete, entry.audio_path)

    # Decorator -----------------------------------------------------------------------

    def memoize(
        self,
        *,
        provider: str | None = None,
        voice_id: str | None = None,
        model: str | None = None,
        audio_format: str | None = None,
        settings: Mapping[str, Any] | None = None,
        text_arg: str = "text",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Wraps a TTS function so it is only called on a miss.

        The wrapped function keeps its own signature; the text is read from its first
        positional argument or from the keyword named by ``text_arg``. Async functions get
        an async wrapper, plain ones a plain wrapper.
        """

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            def spec_for(args: tuple, kwargs: dict) -> AudioSpec:
                if text_arg in kwargs:
                    text = kwargs[text_arg]
                elif args:
                    text = args[0]
                else:
                    raise TypeError(f"{fn.__name__}() needs {text_arg!r} to cache on")
                return self.spec(
                    text,
                    provider=provider,
                    voice_id=voice_id,
                    model=model,
                    audio_format=audio_format,
                    settings=settings,
                )

            if inspect.iscoroutinefunction(fn):

                async def async_wrapper(*args: Any, **kwargs: Any) -> bytes:
                    spec = spec_for(args, kwargs)
                    audio, key, _ = await self._lookup(spec)
                    if audio is not None:
                        return audio
                    return await self._generate(spec, key, lambda: fn(*args, **kwargs))

                wrapper: Callable[..., Any] = async_wrapper
            else:

                def sync_wrapper(*args: Any, **kwargs: Any) -> bytes:
                    spec = spec_for(args, kwargs)
                    audio, key, _ = self._run(self._lookup(spec))
                    if audio is not None:
                        return audio
                    audio = _as_bytes(fn(*args, **kwargs))
                    self._schedule_write(spec, key, audio)
                    return audio

                wrapper = sync_wrapper

            wrapper = _copy_metadata(fn, wrapper)
            wrapper.cache = self  # type: ignore[attr-defined]
            wrapper.uncached = fn  # type: ignore[attr-defined]
            return wrapper

        return decorate

    # Maintenance ---------------------------------------------------------------------

    async def invalidate(self, text: str, **spec_kwargs: Any) -> bool:
        """Drops one clip. Returns whether there was anything to drop."""
        key = self.key_for(self.spec(text, **spec_kwargs))
        return await self.delete_key(key)

    async def delete_key(self, key: str) -> bool:
        entry = await self.index.adelete(key)
        if entry is None:
            return False
        await asyncio.to_thread(_suppress, self.storage.delete, entry.audio_path)
        return True

    async def purge_expired(self, older_than: float | None = None) -> int:
        """Removes everything past its TTL. Returns how many entries went."""
        ttl = older_than if older_than is not None else self.time_to_expire
        if ttl is None:
            raise ConfigurationError("purge_expired needs a TTL: set time_to_expire or pass one")
        expired = await self.index.aexpired(time.time() - ttl)
        return await self._remove_all(expired)

    async def prune(self, target_size_mb: float | None = None) -> int:
        """Evicts until the cache fits in ``target_size_mb`` (or its configured maximum)."""
        target = int(target_size_mb * MB) if target_size_mb is not None else self.max_storage_bytes
        if target is None:
            raise ConfigurationError("prune needs a target: set max_storage_size_mb or pass one")
        total = await self.index.atotal_size()
        if total <= target:
            return 0
        return await asyncio.to_thread(self._evict, total - target)

    async def delete_where(
        self, *, provider: str | None = None, voice_id: str | None = None
    ) -> int:
        """Removes every clip matching the filter — a voice retired, a provider swapped."""
        if provider is None and voice_id is None:
            raise ConfigurationError("delete_where needs a provider or a voice_id")
        entries = await self.index.alist(provider=provider, voice_id=voice_id)
        return await self._remove_all(entries)

    async def flush(self) -> int:
        """Empties the index and the storage this cache owns."""
        count = await self.index.acount()
        await self.index.aclear()
        await self.storage.aclear()
        return count

    async def _remove_all(self, entries: Iterable[CacheEntry]) -> int:
        removed = 0
        for entry in entries:
            if await self.delete_key(entry.cache_key):
                removed += 1
        return removed

    async def size_bytes(self) -> int:
        return await self.index.atotal_size()

    async def count(self) -> int:
        return await self.index.acount()

    # Lifecycle -----------------------------------------------------------------------

    async def drain(self, timeout: float | None = None) -> None:
        """Waits for background writes to finish. Mostly useful in tests and shutdown."""
        await asyncio.to_thread(self.drain_sync, timeout)

    def drain_sync(self, timeout: float | None = None) -> None:
        with self._pending_lock:
            pending = set(self._pending)
        if pending:
            wait(pending, timeout=timeout)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def close(self) -> None:
        """Finishes pending writes, then releases the drivers and the worker threads."""
        if self._closed:
            return
        self.drain_sync()
        self._closed = True
        self._workers.shutdown(wait=True)
        if self._loop_thread is not None:
            self._loop_thread.stop()
            self._loop_thread = None
        _suppress(self.index.close)
        _suppress(self.storage.close)

    async def __aenter__(self) -> TTSCache:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __enter__(self) -> TTSCache:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # Synchronous mirror --------------------------------------------------------------

    def get_sync(self, text: str, **spec_kwargs: Any) -> bytes | None:
        return self._run(self.get(text, **spec_kwargs))

    def get_or_generate_sync(self, text: str, **kwargs: Any) -> CacheResult:
        return self._run(self.get_or_generate(text, **kwargs))

    def put_sync(self, text: str, audio: bytes, **spec_kwargs: Any) -> CacheEntry | None:
        return self._run(self.put(text, audio, **spec_kwargs))

    def invalidate_sync(self, text: str, **spec_kwargs: Any) -> bool:
        return self._run(self.invalidate(text, **spec_kwargs))

    def purge_expired_sync(self, older_than: float | None = None) -> int:
        return self._run(self.purge_expired(older_than))

    def prune_sync(self, target_size_mb: float | None = None) -> int:
        return self._run(self.prune(target_size_mb))

    def delete_where_sync(self, **kwargs: Any) -> int:
        return self._run(self.delete_where(**kwargs))

    def flush_sync(self) -> int:
        return self._run(self.flush())

    def _run(self, coro: Awaitable[Any]) -> Any:
        """Runs a coroutine from synchronous code, on a loop of the cache's own.

        A private loop rather than ``asyncio.run`` because these methods are also reached
        from inside a decorated synchronous function, which may itself be running on a
        worker thread of an async application. Calling one of them from the event loop
        thread of a running application would block it, so that is an error.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # Closed explicitly, so the mistake is one error and not an error plus a
            # "coroutine was never awaited" warning from the garbage collector.
            if inspect.iscoroutine(coro):
                coro.close()
            raise RuntimeError(
                "the *_sync methods block; inside an event loop await the async ones instead"
            )
        if self._loop_thread is None:
            self._loop_thread = _LoopThread()
        return self._loop_thread.run(coro)

    # Bookkeeping ---------------------------------------------------------------------

    def _count(self, **deltas: int) -> None:
        with self._stats_lock:
            for name, delta in deltas.items():
                setattr(self.stats, name, getattr(self.stats, name) + delta)

    def _failed(self, what: str, error: Exception) -> None:
        self._count(errors=1)
        if self.on_error == "raise":
            raise error if isinstance(error, TTSCacheError) else TTSCacheError(f"{what}: {error}")
        log.warning("tts-cache: %s failed: %s", what, error)

    def __repr__(self) -> str:
        limit = (
            f"{self.max_storage_bytes / MB:.0f} MB"
            if self.max_storage_bytes is not None
            else "unbounded"
        )
        return (
            f"TTSCache(storage={self.storage.name}, index={self.index.name}, "
            f"limit={limit}, policy={self.eviction_policy})"
        )


class _LoopThread:
    """An event loop on a daemon thread, so synchronous callers can await."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="tts-cache-loop", daemon=True
        )
        self.thread.start()

    def run(self, coro: Awaitable[Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()  # type: ignore[arg-type]

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


async def _call_generator(generator_fn: Generator, text: str) -> bytes:
    """Calls a vendor function that may be sync or async, and may stream."""
    result = _invoke(generator_fn, text)
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "__aiter__"):
        return b"".join([chunk async for chunk in result])
    return _as_bytes(result)


async def _iter_generator(generator_fn: Generator, text: str) -> AsyncIterator[bytes]:
    """Normalises every shape a vendor function can return into a stream of chunks."""
    result = _invoke(generator_fn, text)
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "__aiter__"):
        async for chunk in result:
            yield bytes(chunk)
    elif isinstance(result, (bytes, bytearray, memoryview)):
        yield bytes(result)
    elif hasattr(result, "__iter__"):
        for chunk in result:
            yield bytes(chunk)
    else:
        raise TypeError(f"a generator returned {type(result).__name__}, not audio")


def _invoke(generator_fn: Generator, text: str) -> Any:
    """Calls the function with the text if it takes one, and with nothing if it does not."""
    try:
        signature = inspect.signature(generator_fn)
    except (TypeError, ValueError):  # builtins and C callables
        return generator_fn()
    takes_text = any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in signature.parameters.values()
    )
    return generator_fn(text) if takes_text else generator_fn()


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, Mapping)):
        return b"".join(bytes(chunk) for chunk in value)
    raise TypeError(f"expected audio bytes, got {type(value).__name__}")


def _copy_metadata(source: Callable[..., Any], target: Callable[..., Any]) -> Callable[..., Any]:
    for attribute in ("__name__", "__qualname__", "__doc__", "__module__", "__annotations__"):
        with contextlib.suppress(AttributeError):
            setattr(target, attribute, getattr(source, attribute))
    target.__wrapped__ = source  # type: ignore[attr-defined]
    return target


def _suppress(fn: Callable[..., Any], *args: Any) -> None:
    """Best effort cleanup: a blob that will not delete must not fail the caller's turn."""
    try:
        fn(*args)
    except Exception as e:
        log.debug("tts-cache: cleanup failed: %s", e)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


async def _shape(
    source: AsyncIterator[bytes],
    frame_bytes: int | None,
    bytes_per_second: float | None,
    started: float,
    chunk_size: int,
) -> AsyncIterator[bytes]:
    """Frame-aligns and paces a stream, without ever delaying the first chunk."""
    sent = 0
    async for chunk in _align(source, frame_bytes):
        # Unpaced, a chunk is forwarded exactly as it arrived. Paced, it has to be broken
        # up first: a vendor that hands back the whole clip at once would otherwise be
        # one chunk, and there would be nothing to spread over time.
        for piece in _split(chunk, chunk_size) if bytes_per_second else (chunk,):
            if bytes_per_second and sent:
                # Where this piece sits on the clock of the audio already sent.
                ahead = sent / bytes_per_second - (time.perf_counter() - started)
                if ahead > 0:
                    await asyncio.sleep(ahead)
            sent += len(piece)
            yield piece


def _split(chunk: bytes, size: int) -> Iterable[bytes]:
    if size <= 0 or len(chunk) <= size:
        return (chunk,)
    return tuple(chunk[at : at + size] for at in range(0, len(chunk), size))


async def _align(source: AsyncIterator[bytes], frame_bytes: int | None) -> AsyncIterator[bytes]:
    """Re-chunks onto whole frames, so a chunk boundary never lands inside a sample.

    A transport handed half a PCM frame either drops it or plays every later frame one
    byte out of phase, which is heard as noise.
    """
    if not frame_bytes or frame_bytes <= 1:
        async for chunk in source:
            if chunk:
                yield chunk
        return

    pending = b""
    async for chunk in source:
        pending += chunk
        whole = len(pending) - len(pending) % frame_bytes
        if whole:
            yield pending[:whole]
            pending = pending[whole:]
    if pending:
        # A vendor that ends mid-frame is padded rather than truncated, so the clip keeps
        # its full length and the caller is not handed a fragment.
        yield pending + b"\x00" * (frame_bytes - len(pending))
