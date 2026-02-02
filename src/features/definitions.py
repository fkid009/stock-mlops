"""Feature definitions for stock prediction."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""

    name: str
    description: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    category: str = "primary"  # primary or secondary


def compute_return_1d(df: pd.DataFrame) -> pd.Series:
    """Compute 1-day return."""
    return df["close"].pct_change(1)


def compute_return_5d(df: pd.DataFrame) -> pd.Series:
    """Compute 5-day return."""
    return df["close"].pct_change(5)


def compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute volume relative to 20-day average."""
    vol_mean = df["volume"].rolling(window=20).mean()
    # Replace 0 with NaN to avoid division by zero
    vol_mean = vol_mean.replace(0, np.nan)
    return df["volume"] / vol_mean


def compute_high_low_range(df: pd.DataFrame) -> pd.Series:
    """Compute daily price range relative to open."""
    # Replace 0 with NaN to avoid division by zero
    open_price = df["open"].replace(0, np.nan)
    return (df["high"] - df["low"]) / open_price


def compute_ma_5_20_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute 5-day MA / 20-day MA ratio."""
    ma_5 = df["close"].rolling(window=5).mean()
    ma_20 = df["close"].rolling(window=20).mean()
    # Replace 0 with NaN to avoid division by zero
    ma_20 = ma_20.replace(0, np.nan)
    return ma_5 / ma_20


def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
    """Compute 14-day RSI."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Handle division by zero: when avg_loss is 0, RSI should be 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss is 0 (all gains), RSI = 100
    rsi = rsi.fillna(100)
    # When avg_gain is 0 and avg_loss > 0, RSI = 0 (handled by formula)

    return rsi


def compute_volatility_20(df: pd.DataFrame) -> pd.Series:
    """Compute 20-day rolling volatility of returns."""
    returns = df["close"].pct_change()
    return returns.rolling(window=20).std()


def compute_target(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
    """Compute target variable (next day direction).

    Args:
        df: DataFrame with 'close' column
        threshold: Threshold for considering as neutral (excluded)

    Returns:
        Series with 1 (up), 0 (down), or NaN (neutral)
    """
    next_return = df["close"].shift(-1) / df["close"] - 1

    target = pd.Series(index=df.index, dtype=float)
    target[next_return > threshold] = 1
    target[next_return < -threshold] = 0
    # Neutral (within threshold) remains NaN

    return target


# Feature definitions registry
FEATURE_DEFINITIONS: dict[str, FeatureDefinition] = {
    "return_1d": FeatureDefinition(
        name="return_1d",
        description="1-day return (yesterday vs today)",
        compute_fn=compute_return_1d,
        category="primary",
    ),
    "return_5d": FeatureDefinition(
        name="return_5d",
        description="5-day return",
        compute_fn=compute_return_5d,
        category="primary",
    ),
    "volume_ratio": FeatureDefinition(
        name="volume_ratio",
        description="Volume relative to 20-day average",
        compute_fn=compute_volume_ratio,
        category="primary",
    ),
    "high_low_range": FeatureDefinition(
        name="high_low_range",
        description="Daily price range relative to open",
        compute_fn=compute_high_low_range,
        category="primary",
    ),
    "ma_5_20_ratio": FeatureDefinition(
        name="ma_5_20_ratio",
        description="5-day MA / 20-day MA ratio",
        compute_fn=compute_ma_5_20_ratio,
        category="secondary",
    ),
    "rsi_14": FeatureDefinition(
        name="rsi_14",
        description="14-day RSI",
        compute_fn=compute_rsi_14,
        category="secondary",
    ),
    "volatility_20": FeatureDefinition(
        name="volatility_20",
        description="20-day rolling volatility",
        compute_fn=compute_volatility_20,
        category="secondary",
    ),
}

# Feature names list
PRIMARY_FEATURES = [
    name for name, defn in FEATURE_DEFINITIONS.items() if defn.category == "primary"
]

SECONDARY_FEATURES = [
    name for name, defn in FEATURE_DEFINITIONS.items() if defn.category == "secondary"
]

ALL_FEATURES = list(FEATURE_DEFINITIONS.keys())
