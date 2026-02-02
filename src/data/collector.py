"""Stock data collection from various sources."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.common import get_logger

logger = get_logger(__name__)


class StockDataCollector:
    """Collect stock OHLCV data from Yahoo Finance."""

    def __init__(self, symbols: Optional[list[str]] = None):
        """Initialize the collector.

        Args:
            symbols: List of stock symbols to collect
        """
        self.symbols = symbols or []

    def fetch(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 120,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            days: Number of days to fetch if start_date is not provided

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Keep only OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]]
            df["symbol"] = symbol
            df.index.name = "date"
            df = df.reset_index()

            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        symbols: Optional[list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 120,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of stock symbols (uses self.symbols if not provided)
            start_date: Start date for data collection
            end_date: End date for data collection
            days: Number of days to fetch if start_date is not provided

        Returns:
            DataFrame with OHLCV data for all symbols
        """
        symbols = symbols or self.symbols

        if not symbols:
            logger.warning("No symbols provided")
            return pd.DataFrame()

        dfs = []
        for symbol in symbols:
            df = self.fetch(symbol, start_date, end_date, days)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def fetch_latest(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Fetch the latest data for a symbol.

        Args:
            symbol: Stock ticker symbol
            days: Number of recent days to fetch

        Returns:
            DataFrame with recent OHLCV data
        """
        return self.fetch(symbol, days=days)
