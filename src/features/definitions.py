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


def _compute_rsi_raw(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute raw RSI preserving warmup NaN."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss is 0 and past warmup (avg_gain is valid), RSI = 100
    past_warmup = avg_gain.notna()
    rsi = rsi.where(~(past_warmup & (avg_loss == 0)), 100)

    return rsi


def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
    """Compute 14-day RSI."""
    return _compute_rsi_raw(df, window=14)


def compute_volatility_20(df: pd.DataFrame) -> pd.Series:
    """Compute 20-day rolling volatility of returns."""
    returns = df["close"].pct_change()
    return returns.rolling(window=20).std()


def compute_bollinger_position(df: pd.DataFrame) -> pd.Series:
    """Compute position within Bollinger Bands (0=lower, 1=upper)."""
    close = df["close"]
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    band_width = 4 * std_20
    # Replace 0 with NaN to avoid division by zero
    band_width = band_width.replace(0, np.nan)
    return (close - (ma_20 - 2 * std_20)) / band_width


def compute_momentum_10d(df: pd.DataFrame) -> pd.Series:
    """Compute 10-day return (mid-term momentum)."""
    return df["close"].pct_change(10)


def compute_macd_signal(df: pd.DataFrame) -> pd.Series:
    """Compute MACD histogram normalized by close price."""
    close = df["close"]
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    # Normalize by close to make it scale-independent
    return (macd - signal) / close


def compute_volume_price_divergence(df: pd.DataFrame) -> pd.Series:
    """Compute divergence between price direction and volume trend."""
    price_change = df["close"].pct_change(5)
    vol_ma_5 = df["volume"].rolling(5).mean()
    vol_ma_20 = df["volume"].rolling(20).mean()
    # Replace 0 with NaN to avoid division by zero
    vol_ma_20 = vol_ma_20.replace(0, np.nan)
    vol_trend = (vol_ma_5 / vol_ma_20) - 1
    return np.sign(price_change) * (-vol_trend)


def compute_volatility_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute 5-day / 20-day volatility ratio (regime indicator)."""
    returns = df["close"].pct_change()
    vol_5 = returns.rolling(5).std()
    vol_20 = returns.rolling(20).std()
    # Replace 0 with NaN to avoid division by zero
    vol_20 = vol_20.replace(0, np.nan)
    return vol_5 / vol_20


def compute_rsi_divergence(df: pd.DataFrame) -> pd.Series:
    """Compute 5-day change in RSI (momentum acceleration)."""
    rsi = _compute_rsi_raw(df)
    return rsi - rsi.shift(5)


def compute_gap_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute overnight gap (today's open / yesterday's close - 1)."""
    return (df["open"] / df["close"].shift(1)) - 1


def compute_close_location_value(df: pd.DataFrame) -> pd.Series:
    """Compute close position within daily range (0=low, 1=high)."""
    range_size = df["high"] - df["low"]
    # Replace 0 with NaN to avoid division by zero (when high == low)
    range_size = range_size.replace(0, np.nan)
    return (df["close"] - df["low"]) / range_size


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
    "bollinger_position": FeatureDefinition(
        name="bollinger_position",
        description="Position within Bollinger Bands (0=lower, 1=upper)",
        compute_fn=compute_bollinger_position,
        category="primary",
    ),
    "momentum_10d": FeatureDefinition(
        name="momentum_10d",
        description="10-day return (mid-term momentum)",
        compute_fn=compute_momentum_10d,
        category="primary",
    ),
    "macd_signal": FeatureDefinition(
        name="macd_signal",
        description="MACD histogram normalized by close price",
        compute_fn=compute_macd_signal,
        category="secondary",
    ),
    "volume_price_divergence": FeatureDefinition(
        name="volume_price_divergence",
        description="Divergence between price direction and volume trend",
        compute_fn=compute_volume_price_divergence,
        category="primary",
    ),
    "volatility_ratio": FeatureDefinition(
        name="volatility_ratio",
        description="5-day / 20-day volatility ratio (regime indicator)",
        compute_fn=compute_volatility_ratio,
        category="primary",
    ),
    "rsi_divergence": FeatureDefinition(
        name="rsi_divergence",
        description="5-day change in RSI (momentum acceleration)",
        compute_fn=compute_rsi_divergence,
        category="secondary",
    ),
    "gap_ratio": FeatureDefinition(
        name="gap_ratio",
        description="Overnight gap (today's open / yesterday's close - 1)",
        compute_fn=compute_gap_ratio,
        category="primary",
    ),
    "close_location_value": FeatureDefinition(
        name="close_location_value",
        description="Close position within daily range (0=low, 1=high)",
        compute_fn=compute_close_location_value,
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
