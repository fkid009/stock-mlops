"""Tests for feature engineering module."""

import pandas as pd
import pytest

from src.features import FeatureEngineer, ALL_FEATURES
from src.features.definitions import (
    compute_return_1d,
    compute_return_5d,
    compute_volume_ratio,
    compute_rsi_14,
    compute_target,
)


class TestFeatureDefinitions:
    """Tests for individual feature computation functions."""

    def test_compute_return_1d(self, sample_ohlcv_data):
        """Test 1-day return computation."""
        result = compute_return_1d(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        assert result.isna().sum() == 1  # First row should be NaN
        assert result.iloc[1:].notna().all()

    def test_compute_return_5d(self, sample_ohlcv_data):
        """Test 5-day return computation."""
        result = compute_return_5d(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        assert result.isna().sum() == 5  # First 5 rows should be NaN

    def test_compute_volume_ratio(self, sample_ohlcv_data):
        """Test volume ratio computation."""
        result = compute_volume_ratio(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        # After 20 days, we should have valid ratios
        assert result.iloc[20:].notna().all()
        # Ratio should be positive
        assert (result.iloc[20:] > 0).all()

    def test_compute_rsi_14(self, sample_ohlcv_data):
        """Test RSI computation."""
        result = compute_rsi_14(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        # RSI should be between 0 and 100
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_compute_target(self, sample_ohlcv_data):
        """Test target variable computation."""
        result = compute_target(sample_ohlcv_data, threshold=0.001)

        assert len(result) == len(sample_ohlcv_data)
        # Last row should be NaN (no next day)
        assert pd.isna(result.iloc[-1])
        # Values should be 0, 1, or NaN
        valid_values = result.dropna()
        assert set(valid_values.unique()).issubset({0, 1})


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_init_default_features(self):
        """Test initialization with default features."""
        engineer = FeatureEngineer()

        assert engineer.features == ALL_FEATURES
        assert engineer.scaler is None
        assert not engineer._is_fitted

    def test_init_custom_features(self):
        """Test initialization with custom features."""
        custom_features = ["return_1d", "rsi_14"]
        engineer = FeatureEngineer(features=custom_features)

        assert engineer.features == custom_features

    def test_compute_features(self, sample_ohlcv_data):
        """Test feature computation."""
        engineer = FeatureEngineer()
        result = engineer.compute_features(sample_ohlcv_data)

        # Check all features are computed
        for feature in ALL_FEATURES:
            assert feature in result.columns

    def test_compute_features_missing_columns(self):
        """Test feature computation with missing required columns."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.compute_features(df)

    def test_fit_scaler(self, sample_features_data):
        """Test scaler fitting."""
        engineer = FeatureEngineer()
        engineer.fit_scaler(sample_features_data)

        assert engineer._is_fitted
        assert engineer.scaler is not None

    def test_transform_without_fit(self, sample_features_data):
        """Test transform raises error without fitting."""
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="Scaler not fitted"):
            engineer.transform(sample_features_data)

    def test_fit_transform(self, sample_features_data):
        """Test fit_transform."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_features_data)

        assert engineer._is_fitted
        # Check features are scaled (mean ~0, std ~1 for valid data)
        for feature in ALL_FEATURES:
            if feature in result.columns:
                valid_data = result[feature].dropna()
                if len(valid_data) > 10:
                    assert abs(valid_data.mean()) < 0.5
                    assert 0.5 < valid_data.std() < 2.0

    def test_prepare_dataset(self, sample_ohlcv_data):
        """Test complete dataset preparation."""
        engineer = FeatureEngineer()
        X, y = engineer.prepare_dataset(sample_ohlcv_data, fit=True)

        # Check X has all features
        assert list(X.columns) == ALL_FEATURES
        # Check no NaN values
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0
        # Check X and y have same length
        assert len(X) == len(y)

    def test_save_load_scaler(self, sample_features_data, tmp_path):
        """Test scaler save and load."""
        engineer = FeatureEngineer()
        engineer.fit_scaler(sample_features_data)

        scaler_path = str(tmp_path / "scaler.joblib")
        engineer.save_scaler(scaler_path)

        # Load into new engineer
        new_engineer = FeatureEngineer()
        new_engineer.load_scaler(scaler_path)

        assert new_engineer._is_fitted
        assert new_engineer.scaler is not None
