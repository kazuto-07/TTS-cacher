"""``tts-cache`` on the command line: inspect the cache, and keep it from growing forever.

Every command takes ``--storage`` and ``--index``, which also read from the environment
(``TTS_CACHE_STORAGE``, ``TTS_CACHE_INDEX``), so a cron job can be one line::

    tts-cache purge --older-than 30d
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from typing import Any

from . import __version__
from .cache import MB, TTSCache
from .errors import TTSCacheError
from .index import Index, MemoryIndex, SqliteIndex
from .models import CacheEntry
from .storage import LocalStorage, MemoryStorage, Storage

__all__ = ["main", "build_cache", "parse_duration"]

DEFAULT_STORAGE = "local:.tts-cache/audio"
DEFAULT_INDEX = "sqlite:.tts-cache/index.db"

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw]?)\s*$", re.IGNORECASE)
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "": 1}


def parse_duration(text: str) -> float:
    """``900``, ``30m``, ``14d`` — all in seconds."""
    match = _DURATION.match(text)
    if not match:
        raise argparse.ArgumentTypeError(f"not a duration: {text!r} (try 30m, 14d, 3600)")
    return float(match.group(1)) * _UNITS[match.group(2).lower()]


def _storage_from(spec: str) -> Storage:
    scheme, _, rest = spec.partition(":")
    scheme = scheme.lower()
    if scheme == "memory":
        return MemoryStorage()
    if scheme in {"local", "file"}:
        return LocalStorage(rest or ".tts-cache/audio")
    if scheme == "supabase":
        from .storage.supabase import SupabaseStorage

        return SupabaseStorage(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY"),
            bucket_name=rest or "voice-cache",
        )
    if scheme == "s3":
        from .storage.s3 import S3Storage

        bucket, _, prefix = rest.partition("/")
        return S3Storage(bucket, prefix=prefix or "tts-cache")
    raise TTSCacheError(f"unknown storage {spec!r} (try local:PATH, memory:, s3:BUCKET)")


def _index_from(spec: str) -> Index:
    scheme, _, rest = spec.partition(":")
    scheme = scheme.lower()
    if scheme == "memory":
        return MemoryIndex()
    if scheme == "sqlite":
        return SqliteIndex(rest or ".tts-cache/index.db")
    if scheme == "supabase":
        from .index.supabase import SupabaseIndex

        return SupabaseIndex(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY"),
            table_name=rest or "tts_cache",
        )
    if scheme in {"sqlalchemy", "postgresql", "postgres", "mysql", "mariadb"}:
        from .index.sqlalchemy import SqlAlchemyIndex

        # `sqlalchemy:postgresql+psycopg://…` keeps the driver's own URL intact; a bare
        # database URL is passed straight through.
        url = rest if scheme == "sqlalchemy" else spec
        return SqlAlchemyIndex(url)
    raise TTSCacheError(f"unknown index {spec!r} (try sqlite:PATH, sqlalchemy:URL)")


def build_cache(args: argparse.Namespace) -> TTSCache:
    """The cache the command operates on, wired from the flags and the environment."""
    storage = args.storage or os.environ.get("TTS_CACHE_STORAGE") or DEFAULT_STORAGE
    index = args.index or os.environ.get("TTS_CACHE_INDEX") or DEFAULT_INDEX
    return TTSCache(
        storage=_storage_from(storage),
        index=_index_from(index),
        provider="cli",
        voice_id="cli",
        on_error="raise",
    )


# Commands ----------------------------------------------------------------------------


def _cmd_stats(cache: TTSCache, args: argparse.Namespace) -> dict[str, Any]:
    entries = cache.index.list()
    by_voice: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket = by_voice.setdefault(
            f"{entry.provider}/{entry.voice_id}", {"clips": 0, "bytes": 0, "hits": 0}
        )
        bucket["clips"] += 1
        bucket["bytes"] += entry.size_bytes
        bucket["hits"] += entry.access_count - 1
    return {
        "clips": len(entries),
        "bytes": sum(e.size_bytes for e in entries),
        "served_from_cache": sum(e.access_count - 1 for e in entries),
        "voices": by_voice,
    }


def _cmd_ls(cache: TTSCache, args: argparse.Namespace) -> dict[str, Any]:
    entries = cache.index.list(
        provider=args.provider,
        voice_id=args.voice_id,
        order=args.order,
        limit=args.limit,
    )
    return {"entries": [_row(e) for e in entries]}


def _cmd_purge(cache: TTSCache, args: argparse.Namespace) -> dict[str, Any]:
    if args.older_than is None and not args.expired:
        raise TTSCacheError("purge needs --expired (with TTS_CACHE_TTL) or --older-than")
    ttl = args.older_than
    if ttl is None:
        env_ttl = os.environ.get("TTS_CACHE_TTL")
        if not env_ttl:
            raise TTSCacheError("--expired needs a TTL: set TTS_CACHE_TTL or pass --older-than")
        ttl = parse_duration(env_ttl)
    expired = cache.index.expired(time.time() - ttl)
    removed = _remove(cache, expired)
    return {"purged": removed, "older_than_seconds": ttl}


def _cmd_prune(cache: TTSCache, args: argparse.Namespace) -> dict[str, Any]:
    target = int(args.target_size_mb * MB)
    total = cache.index.total_size()
    if total <= target:
        return {"evicted": 0, "bytes": total, "target_bytes": target}
    evicted = cache._evict(total - target)
    return {"evicted": evicted, "bytes": cache.index.total_size(), "target_bytes": target}


def _cmd_delete(cache: TTSCache, args: argparse.Namespace) -> dict[str, Any]:
    if args.key:
        entry = cache.index.get(args.key)
        removed = _remove(cache, [entry] if entry else [])
        return {"deleted": removed, "key": args.key}
    if not args.provider and not args.voice_id:
        raise TTSCacheError("delete needs --key, or --provider and/or --voice-id")
    entries = cache.index.list(provider=args.provider, voice_id=args.voice_id)
    return {"deleted": _remove(cache, entries)}


def _cmd_flush(cache: TTSCache, args: argparse.Namespace) -> dict[str, Any]:
    if not args.all:
        raise TTSCacheError("flush empties everything; pass --all to confirm")
    count = cache.index.count()
    cache.index.clear()
    cache.storage.clear()
    return {"flushed": count}


def _remove(cache: TTSCache, entries: Sequence[CacheEntry]) -> int:
    removed = 0
    for entry in entries:
        row = cache.index.delete(entry.cache_key)
        if row is None:
            continue
        try:
            cache.storage.delete(row.audio_path)
        except TTSCacheError as e:
            print(f"warning: {row.cache_key[:12]}… audio not deleted: {e}", file=sys.stderr)
        removed += 1
    return removed


def _row(entry: CacheEntry) -> dict[str, Any]:
    return {
        "key": entry.cache_key,
        "text": entry.normalized_text,
        "provider": entry.provider,
        "voice_id": entry.voice_id,
        "format": entry.audio_format,
        "size_bytes": entry.size_bytes,
        "access_count": entry.access_count,
        "created_at": entry.created_at,
        "last_accessed_at": entry.last_accessed_at,
    }


# Presentation ------------------------------------------------------------------------


def _human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _ago(when: float) -> str:
    seconds = max(0.0, time.time() - when)
    for limit, unit, divisor in (
        (60, "s", 1),
        (3600, "m", 60),
        (86400, "h", 3600),
        (float("inf"), "d", 86400),
    ):
        if seconds < limit:
            return f"{seconds / divisor:.0f}{unit}"
    return f"{seconds / 86400:.0f}d"


def _print(command: str, result: dict[str, Any]) -> None:
    if command == "stats":
        print(f"{result['clips']} clips, {_human_bytes(result['bytes'])} stored")
        print(f"{result['served_from_cache']} plays served from the cache")
        for voice, row in sorted(result["voices"].items()):
            print(
                f"  {voice:<40} {row['clips']:>6} clips  "
                f"{_human_bytes(row['bytes']):>10}  {row['hits']:>6} hits"
            )
        return

    if command == "ls":
        entries = result["entries"]
        if not entries:
            print("no entries")
            return
        print(f"{'KEY':<14} {'SIZE':>9} {'HITS':>5} {'USED':>5}  TEXT")
        for row in entries:
            text = row["text"]
            if len(text) > 48:
                text = text[:47] + "…"
            print(
                f"{row['key'][:12]}… {_human_bytes(row['size_bytes']):>9} "
                f"{row['access_count'] - 1:>5} {_ago(row['last_accessed_at']):>5}  {text}"
            )
        print(f"{len(entries)} entries")
        return

    summaries = {
        "purge": lambda r: f"purged {r['purged']} expired entries",
        "prune": lambda r: (
            f"evicted {r['evicted']} entries, now {_human_bytes(r['bytes'])} "
            f"(target {_human_bytes(r['target_bytes'])})"
        ),
        "delete": lambda r: f"deleted {r['deleted']} entries",
        "flush": lambda r: f"flushed {r['flushed']} entries",
    }
    print(summaries[command](result))


# Entry point -------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-cache", description="Inspect and maintain a TTS audio cache."
    )
    parser.add_argument("--version", action="version", version=f"tts-cache {__version__}")
    parser.add_argument(
        "--storage",
        help=f"where audio lives, e.g. local:PATH, s3:BUCKET (default {DEFAULT_STORAGE})",
    )
    parser.add_argument(
        "--index",
        help=f"where rows live, e.g. sqlite:PATH, sqlalchemy:URL (default {DEFAULT_INDEX})",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("stats", help="how much is cached, and how often it is used")

    listing = subcommands.add_parser("ls", help="list cached clips")
    listing.add_argument("--provider")
    listing.add_argument("--voice-id")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument(
        "--order",
        default="last_accessed_at",
        choices=["last_accessed_at", "created_at", "access_count", "size_bytes"],
    )

    purge = subcommands.add_parser("purge", help="remove entries past their TTL")
    purge.add_argument("--expired", action="store_true", help="use the TTS_CACHE_TTL TTL")
    purge.add_argument("--older-than", type=parse_duration, metavar="DURATION")

    prune = subcommands.add_parser("prune", help="evict until the cache fits a size")
    prune.add_argument("--target-size-mb", type=float, required=True)

    delete = subcommands.add_parser("delete", help="remove entries by key or by voice")
    delete.add_argument("--key")
    delete.add_argument("--provider")
    delete.add_argument("--voice-id")

    flush = subcommands.add_parser("flush", help="empty the cache")
    flush.add_argument("--all", action="store_true", help="required: this deletes everything")

    return parser


_COMMANDS = {
    "stats": _cmd_stats,
    "ls": _cmd_ls,
    "purge": _cmd_purge,
    "prune": _cmd_prune,
    "delete": _cmd_delete,
    "flush": _cmd_flush,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Runs one command. Returns 0, 1 for a command that failed, 2 for bad wiring."""
    args = _parser().parse_args(argv)
    try:
        cache = build_cache(args)
    except TTSCacheError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        result = _COMMANDS[args.command](cache, args)
    except TTSCacheError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        cache.close()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print(args.command, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
