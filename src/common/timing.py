"""Timing utilities for performance measurement."""

import functools
import time
from typing import Callable, Any

from .logger import get_logger

logger = get_logger(__name__)


def timed(func: Callable) -> Callable:
    """Decorator to measure and log function execution time.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise

    return wrapper


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("data loading"):
            load_data()
    """

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            logger.info(f"{self.name} completed in {self.elapsed:.2f}s")
        else:
            logger.error(f"{self.name} failed after {self.elapsed:.2f}s")
