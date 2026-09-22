"""Filesystem storage."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from ..errors import StorageError
from .base import Storage

__all__ = ["LocalStorage"]


class LocalStorage(Storage):
    """Writes each blob to ``<root>/<first two key chars>/<key>.<format>``.

    The two-character prefix keeps directories from growing to hundreds of thousands of
    entries, which some filesystems handle badly. Writes go to a temporary file and are
    renamed into place, so a crash mid-write cannot leave a half-written clip that the
    index believes is complete.
    """

    name = "local"

    def __init__(self, root: str | os.PathLike[str] = ".tts-cache/audio") -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_key: str, audio_format: str) -> Path:
        suffix = audio_format.lstrip(".") or "bin"
        return self.root / cache_key[:2] / f"{cache_key}.{suffix}"

    def put(self, cache_key: str, data: bytes, audio_format: str) -> str:
        target = self._path(cache_key, audio_format)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp, target)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as e:
            raise StorageError(f"could not write {target}: {e}") from e
        return str(target)

    def get(self, path: str) -> bytes | None:
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise StorageError(f"could not read {path}: {e}") from e

    def iter(self, path: str, chunk_size: int = 8192) -> Iterator[bytes] | None:
        """Reads the file a chunk at a time, so the first one is ready almost at once."""
        try:
            handle = Path(path).open("rb")  # noqa: SIM115 - closed by the generator below
        except FileNotFoundError:
            return None
        except OSError as e:
            raise StorageError(f"could not read {path}: {e}") from e

        def chunks() -> Iterator[bytes]:
            with handle:
                while True:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        return
                    yield chunk

        return chunks()

    def delete(self, path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as e:
            raise StorageError(f"could not delete {path}: {e}") from e

    def clear(self) -> None:
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
