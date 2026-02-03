"""Utility functions for pipeline operations."""

from datetime import datetime
from typing import Optional
import pandas as pd


def calculate_accuracy(df: pd.DataFrame) -> Optional[float]:
    """Calculate prediction accuracy from DataFrame.

    Args:
        df: DataFrame with 'prediction' and 'actual' columns.

    Returns:
        Accuracy as float, or None if no valid data.
    """
    if df.empty:
        return None

    valid = df[df["actual"].notna()]
    if len(valid) == 0:
        return None

    correct = (valid["prediction"] == valid["actual"]).sum()
    return correct / len(valid)


def get_date_range(date: datetime) -> tuple[datetime, datetime]:
    """Get start and end datetime for a given date.

    Args:
        date: Date to get range for.

    Returns:
        Tuple of (start_datetime, end_datetime).
    """
    start = datetime.combine(date.date(), datetime.min.time())
    end = datetime.combine(date.date(), datetime.max.time())
    return start, end


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Value to return if division is not possible.

    Returns:
        Result of division or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator
