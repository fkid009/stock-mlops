"""Shared dependencies for API routers."""

from functools import lru_cache

from src.data import DatabaseManager


@lru_cache
def get_db() -> DatabaseManager:
    """Get cached DatabaseManager instance."""
    return DatabaseManager()
