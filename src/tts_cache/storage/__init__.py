"""Binary stores. The optional drivers import lazily so their SDKs stay optional."""

from __future__ import annotations

from typing import Any

from .base import Storage
from .local import LocalStorage
from .memory import MemoryStorage

__all__ = ["Storage", "LocalStorage", "MemoryStorage", "SupabaseStorage", "S3Storage"]


def __getattr__(name: str) -> Any:
    if name == "SupabaseStorage":
        from .supabase import SupabaseStorage

        return SupabaseStorage
    if name == "S3Storage":
        from .s3 import S3Storage

        return S3Storage
    raise AttributeError(name)
