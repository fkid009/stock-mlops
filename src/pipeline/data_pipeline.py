"""Common data pipeline functions for DAGs."""

from typing import Optional
import pandas as pd

from src.common import get_config, get_logger
from src.data import StockDataCollector, DataCache
from src.features import FeatureEngineer

logger = get_logger(__name__)


def collect_and_cache_data(
    symbols: Optional[list[str]] = None,
    days: int = 30,
    use_latest: bool = True,
) -> list[pd.DataFrame]:
    """Collect stock data and cache it.

    Args:
        symbols: List of stock symbols. If None, uses config.symbols_list.
        days: Number of days of data to fetch.
        use_latest: If True, uses fetch_latest for incremental update.

    Returns:
        List of DataFrames with collected data.
    """
    config = get_config()
    symbols = symbols or config.symbols_list

    collector = StockDataCollector(symbols)
    cache = DataCache()

    results = []
    for symbol in symbols:
        try:
            if use_latest:
                df = collector.fetch_latest(symbol, days=days)
            else:
                df = collector.fetch(symbol, days=days)

            if not df.empty:
                cache.save(symbol, df)
                results.append(df)
                logger.info(f"Collected {len(df)} rows for {symbol}")
            else:
                logger.warning(f"No data collected for {symbol}")
        except Exception as e:
            logger.error(f"Failed to collect data for {symbol}: {e}")
            continue

    return results


def load_and_prepare_features(
    symbols: Optional[list[str]] = None,
    include_target: bool = True,
) -> list[pd.DataFrame]:
    """Load cached data and compute features.

    Args:
        symbols: List of stock symbols. If None, uses config.symbols_list.
        include_target: Whether to compute target variable.

    Returns:
        List of DataFrames with computed features.
    """
    config = get_config()
    symbols = symbols or config.symbols_list

    cache = DataCache()
    engineer = FeatureEngineer()

    results = []
    for symbol in symbols:
        try:
            df = cache.load(symbol)
            if df.empty:
                logger.warning(f"No cached data for {symbol}")
                continue

            df = engineer.compute_features(df)
            if include_target:
                df = engineer.compute_target(df)
            df = engineer.handle_missing_values(df)
            df["symbol"] = symbol
            results.append(df)
        except Exception as e:
            logger.error(f"Failed to prepare features for {symbol}: {e}")
            continue

    return results


def batch_prepare_dataset(
    symbols: Optional[list[str]] = None,
    scaler_path: Optional[str] = None,
    fit_scaler: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare dataset for all symbols with optional scaling.

    Args:
        symbols: List of stock symbols. If None, uses config.symbols_list.
        scaler_path: Path to load/save scaler. If None, uses default.
        fit_scaler: Whether to fit a new scaler or load existing.

    Returns:
        Tuple of (X, y) DataFrames.
    """
    config = get_config()
    symbols = symbols or config.symbols_list
    scaler_path = scaler_path or str(config.data_dir / "scaler.joblib")

    cache = DataCache()
    engineer = FeatureEngineer()

    if not fit_scaler:
        engineer.load_scaler(scaler_path)

    all_X = []
    all_y = []

    for symbol in symbols:
        try:
            df = cache.load(symbol)
            if df.empty:
                continue
            X, y = engineer.prepare_dataset(df, fit=False)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            logger.warning(f"Failed to prepare dataset for {symbol}: {e}")
            continue

    if not all_X:
        raise ValueError("No data available for dataset preparation")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    return X, y
