"""Logging configuration for the stock prediction pipeline."""

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .config import get_config


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


@lru_cache
def get_logger(name: str = "stock_mlops") -> logging.Logger:
    """Get a cached logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    config = get_config()
    level = logging.DEBUG if config.debug else logging.INFO

    log_file = None
    if config.app_env == "production":
        log_file = config.project_root / "logs" / f"{name}.log"

    return setup_logger(name, level=level, log_file=log_file)
