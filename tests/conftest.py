"""Pytest fixtures for stock-mlops tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 100

    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = {
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
        "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n_days),
    }

    df = pd.DataFrame(data)
    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    # Add symbol column for cache tests
    df["symbol"] = "TEST"

    return df


@pytest.fixture
def sample_features_data(sample_ohlcv_data) -> pd.DataFrame:
    """Generate sample data with computed features."""
    from src.features import FeatureEngineer

    engineer = FeatureEngineer()
    df = engineer.compute_features(sample_ohlcv_data)
    df = engineer.compute_target(df)
    df = engineer.handle_missing_values(df)

    return df


@pytest.fixture
def sample_train_data(sample_features_data) -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample training data (X, y)."""
    from src.features import ALL_FEATURES

    df = sample_features_data.dropna(subset=ALL_FEATURES + ["target"])
    X = df[ALL_FEATURES]
    y = df["target"]

    return X, y
