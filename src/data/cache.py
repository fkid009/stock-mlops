"""Data caching with Parquet files."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common import get_config, get_logger

logger = get_logger(__name__)


class DataCache:
    """Cache stock data as Parquet files."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files
        """
        config = get_config()
        self.cache_dir = cache_dir or config.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.cache_dir / f"{symbol.lower()}.parquet"

    def save(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save data to cache.

        Args:
            symbol: Stock ticker symbol
            df: DataFrame to cache

        Returns:
            True if successful
        """
        if df.empty:
            logger.warning(f"Cannot cache empty DataFrame for {symbol}")
            return False

        cache_path = self._get_cache_path(symbol)

        try:
            # Load existing cache and merge
            if cache_path.exists():
                existing = pd.read_parquet(cache_path)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["date", "symbol"], keep="last")
                df = df.sort_values("date")

            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(df)} rows for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error caching {symbol}: {e}")
            return False

    def load(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load data from cache.

        Args:
            symbol: Stock ticker symbol
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Cached DataFrame or empty DataFrame if not found
        """
        cache_path = self._get_cache_path(symbol)

        if not cache_path.exists():
            logger.info(f"No cache found for {symbol}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(cache_path)

            # Convert date column to datetime if needed
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

                if start_date:
                    df = df[df["date"] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df["date"] <= pd.to_datetime(end_date)]

            logger.info(f"Loaded {len(df)} rows from cache for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}")
            return pd.DataFrame()

    def exists(self, symbol: str) -> bool:
        """Check if cache exists for a symbol."""
        return self._get_cache_path(symbol).exists()

    def get_date_range(self, symbol: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range of cached data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (min_date, max_date) or (None, None) if no cache
        """
        df = self.load(symbol)

        if df.empty or "date" not in df.columns:
            return None, None

        # Handle potential NaT (Not a Time) values
        min_date = df["date"].min()
        max_date = df["date"].max()

        # Convert pandas Timestamp to datetime, handle NaT
        if pd.isna(min_date) or pd.isna(max_date):
            return None, None

        return min_date.to_pydatetime(), max_date.to_pydatetime()

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear cache for a symbol or all symbols.

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol:
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for {symbol}")
        else:
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info("Cleared all cache")
