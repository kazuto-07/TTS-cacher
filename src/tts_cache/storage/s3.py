"""S3 and S3-compatible bucket driver (AWS, Cloudflare R2, MinIO, Backblaze B2)."""

from __future__ import annotations

from collections.abc import Iterator

from ..errors import MissingDependency, StorageError
from .base import Storage
from .supabase import _object_path, content_type, is_not_found

__all__ = ["S3Storage"]


class S3Storage(Storage):
    """Stores clips as objects in a bucket reachable with the S3 API.

    ``client_kwargs`` go straight to ``boto3.client("s3", ...)``, which is where an
    ``endpoint_url`` for a non-AWS provider belongs.
    """

    name = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "tts-cache",
        client: object | None = None,
        **client_kwargs: object,
    ) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise MissingDependency("boto3", "s3") from e
            client = boto3.client("s3", **client_kwargs)
        self._client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _object_name(self, path: str) -> str:
        if path.startswith("s3://"):
            _, _, rest = path.partition("s3://")
            bucket, _, name = rest.partition("/")
            if bucket != self.bucket:
                raise StorageError(f"path belongs to bucket {bucket!r}, not {self.bucket!r}")
            return name
        return path

    def put(self, cache_key: str, data: bytes, audio_format: str) -> str:
        name = _object_path(self.prefix, cache_key, audio_format)
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=name,
                Body=bytes(data),
                ContentType=content_type(audio_format),
            )
        except Exception as e:
            raise StorageError(f"s3 put failed for {name}: {e}") from e
        return f"s3://{self.bucket}/{name}"

    def get(self, path: str) -> bytes | None:
        name = self._object_name(path)
        try:
            return self._client.get_object(Bucket=self.bucket, Key=name)["Body"].read()
        except Exception as e:
            if is_not_found(e) or type(e).__name__ == "NoSuchKey":
                return None
            raise StorageError(f"s3 get failed for {name}: {e}") from e

    def iter(self, path: str, chunk_size: int = 8192) -> Iterator[bytes] | None:
        """Streams the object body, so a hit starts playing before the object has landed.

        The bucket is the far side of a network: reading the whole object first would make
        the time to first audio of a cache hit the download time of the entire clip.
        """
        name = self._object_name(path)
        try:
            body = self._client.get_object(Bucket=self.bucket, Key=name)["Body"]
        except Exception as e:
            if is_not_found(e) or type(e).__name__ == "NoSuchKey":
                return None
            raise StorageError(f"s3 get failed for {name}: {e}") from e

        def chunks() -> Iterator[bytes]:
            try:
                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        return
                    yield chunk
            finally:
                body.close()

        return chunks()

    def delete(self, path: str) -> None:
        name = self._object_name(path)
        try:
            self._client.delete_object(Bucket=self.bucket, Key=name)
        except Exception as e:
            raise StorageError(f"s3 delete failed for {name}: {e}") from e

    def clear(self) -> None:
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                names = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                if names:
                    self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": names})
        except Exception as e:
            raise StorageError(f"s3 clear failed: {e}") from e
