"""Feature engineering pipeline."""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common import get_logger
from .definitions import (
    FEATURE_DEFINITIONS,
    ALL_FEATURES,
    compute_target,
)

logger = get_logger(__name__)


class FeatureEngineer:
    """Feature engineering for stock prediction."""

    def __init__(
        self,
        features: Optional[list[str]] = None,
        target_threshold: float = 0.001,
        max_consecutive_missing: int = 5,
    ):
        """Initialize feature engineer.

        Args:
            features: List of feature names to compute (default: all)
            target_threshold: Threshold for neutral classification
            max_consecutive_missing: Max consecutive missing values allowed
        """
        self.features = features or ALL_FEATURES
        self.target_threshold = target_threshold
        self.max_consecutive_missing = max_consecutive_missing
        self.scaler: Optional[StandardScaler] = None
        self._is_fitted = False

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for a DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with computed features
        """
        df = df.copy()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by date if available
        if "date" in df.columns:
            df = df.sort_values("date")

        # Compute each feature
        for feature_name in self.features:
            if feature_name not in FEATURE_DEFINITIONS:
                logger.warning(f"Unknown feature: {feature_name}")
                continue

            defn = FEATURE_DEFINITIONS[feature_name]
            df[feature_name] = defn.compute_fn(df)
            logger.debug(f"Computed feature: {feature_name}")

        return df

    def compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute target variable.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()
        df["target"] = compute_target(df, threshold=self.target_threshold)
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        # Forward fill for features
        feature_cols = [col for col in self.features if col in df.columns]
        df[feature_cols] = df[feature_cols].ffill()

        # Check for excessive consecutive missing values
        for col in feature_cols:
            consecutive_missing = (
                df[col]
                .isna()
                .astype(int)
                .groupby((~df[col].isna()).cumsum())
                .cumsum()
            )
            if consecutive_missing.max() > self.max_consecutive_missing:
                logger.warning(
                    f"Column {col} has more than {self.max_consecutive_missing} "
                    "consecutive missing values"
                )

        return df

    def fit_scaler(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit the scaler on training data.

        Args:
            df: Training DataFrame with features

        Returns:
            Self for chaining
        """
        feature_cols = [col for col in self.features if col in df.columns]

        if not feature_cols:
            raise ValueError("No feature columns found in DataFrame")

        # Drop rows with NaN in features
        valid_data = df[feature_cols].dropna()

        self.scaler = StandardScaler()
        self.scaler.fit(valid_data)
        self._is_fitted = True

        logger.info(f"Fitted scaler on {len(valid_data)} samples")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with scaled features
        """
        if not self._is_fitted or self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")

        df = df.copy()
        feature_cols = [col for col in self.features if col in df.columns]

        # Only transform rows without NaN
        valid_mask = ~df[feature_cols].isna().any(axis=1)
        df.loc[valid_mask, feature_cols] = self.scaler.transform(
            df.loc[valid_mask, feature_cols]
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform in one step.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with scaled features
        """
        self.fit_scaler(df)
        return self.transform(df)

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare complete dataset for training or prediction.

        Args:
            df: Raw DataFrame with OHLCV data
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Compute features
        df = self.compute_features(df)

        # Compute target
        df = self.compute_target(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Scale features
        if fit:
            df = self.fit_transform(df)
        else:
            df = self.transform(df)

        # Get feature columns and target
        feature_cols = [col for col in self.features if col in df.columns]
        X = df[feature_cols]
        y = df["target"]

        # Remove rows with NaN in features or target
        valid_mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"Prepared dataset: {len(X)} samples, {len(feature_cols)} features")

        return X, y

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self.features.copy()

    def save_scaler(self, path: str) -> None:
        """Save fitted scaler to file.

        Args:
            path: Path to save scaler
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted")

        import joblib
        joblib.dump(self.scaler, path)
        logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path: str) -> "FeatureEngineer":
        """Load scaler from file.

        Args:
            path: Path to load scaler from

        Returns:
            Self for chaining
        """
        import joblib
        self.scaler = joblib.load(path)
        self._is_fitted = True
        logger.info(f"Loaded scaler from {path}")
        return self
