"""Exceptions raised by tts-cache."""


class TTSCacheError(Exception):
    """Base class for every error this package raises."""


class StorageError(TTSCacheError):
    """A binary store (filesystem, bucket, memory) failed."""


class IndexError_(TTSCacheError):
    """A metadata index (SQLite, SQLAlchemy, Supabase) failed."""


# `IndexError` is a builtin, so the class above carries a trailing underscore and is
# exported under a name that cannot be confused with it.
MetadataError = IndexError_


class ConfigurationError(TTSCacheError):
    """The cache was wired up with options that cannot work together."""


class MissingDependency(ConfigurationError):
    """An optional driver was used without its extra installed."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"this driver needs the {package!r} package: pip install 'tts-cache[{extra}]'"
        )
