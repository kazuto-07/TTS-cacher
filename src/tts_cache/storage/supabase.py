"""Supabase Storage bucket driver."""

from __future__ import annotations

from ..errors import MissingDependency, StorageError
from .base import Storage

__all__ = ["SupabaseStorage"]


class SupabaseStorage(Storage):
    """Stores clips as objects in a Supabase bucket.

    The Supabase client is synchronous, so the base class's thread offload is what keeps an
    async pipeline from blocking on uploads. Pass an existing ``client`` if the application
    already has one; otherwise give the URL and a key. A service-role key is needed unless
    the bucket's row level security allows the anon key to write.
    """

    name = "supabase"

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        bucket_name: str = "voice-cache",
        *,
        client: object | None = None,
        prefix: str = "",
        upsert: bool = True,
    ) -> None:
        if client is None:
            try:
                from supabase import create_client
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise MissingDependency("supabase", "supabase") from e
            if not supabase_url or not supabase_key:
                raise StorageError("supabase_url and supabase_key are required without a client")
            client = create_client(supabase_url, supabase_key)

        self._client = client
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.upsert = upsert

    @property
    def _bucket(self):
        return self._client.storage.from_(self.bucket_name)

    def _object_name(self, path: str) -> str:
        """Accepts either a bare object name or the ``supabase://bucket/name`` form."""
        if path.startswith("supabase://"):
            _, _, rest = path.partition("supabase://")
            bucket, _, name = rest.partition("/")
            if bucket != self.bucket_name:
                raise StorageError(f"path belongs to bucket {bucket!r}, not {self.bucket_name!r}")
            return name
        return path

    def put(self, cache_key: str, data: bytes, audio_format: str) -> str:
        name = _object_path(self.prefix, cache_key, audio_format)
        options = {
            "content-type": content_type(audio_format),
            # supabase-py passes these through as HTTP headers, so they must be strings.
            "upsert": "true" if self.upsert else "false",
        }
        try:
            self._bucket.upload(name, bytes(data), options)
        except Exception as e:  # the client raises its own error types
            raise StorageError(f"supabase upload failed for {name}: {e}") from e
        return f"supabase://{self.bucket_name}/{name}"

    def get(self, path: str) -> bytes | None:
        name = self._object_name(path)
        try:
            return bytes(self._bucket.download(name))
        except Exception as e:
            # A missing object is an ordinary cache miss, not a failure worth raising.
            if is_not_found(e):
                return None
            raise StorageError(f"supabase download failed for {name}: {e}") from e

    def delete(self, path: str) -> None:
        name = self._object_name(path)
        try:
            self._bucket.remove([name])
        except Exception as e:
            if is_not_found(e):
                return
            raise StorageError(f"supabase delete failed for {name}: {e}") from e

    def clear(self) -> None:
        root = self.prefix or ""
        try:
            for folder in self._bucket.list(root):
                folder_name = folder["name"] if isinstance(folder, dict) else folder.name
                inner = f"{root}/{folder_name}" if root else folder_name
                names = [
                    f"{inner}/{item['name'] if isinstance(item, dict) else item.name}"
                    for item in self._bucket.list(inner)
                ]
                if names:
                    self._bucket.remove(names)
        except Exception as e:
            raise StorageError(f"supabase clear failed: {e}") from e


def _object_path(prefix: str, cache_key: str, audio_format: str) -> str:
    suffix = audio_format.lstrip(".") or "bin"
    name = f"{cache_key[:2]}/{cache_key}.{suffix}"
    return f"{prefix}/{name}" if prefix else name


def content_type(audio_format: str) -> str:
    """The media type to store a clip under, so a bucket serves it playable."""
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "webm": "audio/webm",
        "pcm": "application/octet-stream",
        "raw": "application/octet-stream",
    }.get(audio_format.lstrip(".").lower(), "application/octet-stream")


def is_not_found(error: Exception) -> bool:
    """Whether a client error means "no such object" — an ordinary miss, not a failure."""
    text = str(error).lower()
    return "not found" in text or "404" in text or "does not exist" in text
