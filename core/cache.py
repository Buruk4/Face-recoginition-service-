"""Small cache-management helpers (standard library only)."""

from __future__ import annotations

from config import CACHE_PATH


def clear_cache() -> bool:
    """Delete the embedding cache.  Returns ``True`` if a file was removed."""

    removed = False

    for path in (CACHE_PATH, CACHE_PATH.with_suffix(".tmp.npz")):
        try:
            if path.is_file():
                path.unlink()
                removed = True
        except OSError:
            pass

    return removed
