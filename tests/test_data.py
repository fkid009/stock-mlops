"""Tests for data module."""

import pandas as pd
import pytest

from src.data import DataCache, DatabaseManager


class TestDataCache:
    """Tests for DataCache class."""

    def test_init(self, tmp_path):
        """Test initialization."""
        cache = DataCache(cache_dir=tmp_path)

        assert cache.cache_dir == tmp_path
        assert tmp_path.exists()

    def test_save_and_load(self, sample_ohlcv_data, tmp_path):
        """Test saving and loading data."""
        cache = DataCache(cache_dir=tmp_path)

        # Save
        cache.save("AAPL", sample_ohlcv_data)

        # Verify file exists (cache uses lowercase)
        cache_file = tmp_path / "aapl.parquet"
        assert cache_file.exists()

        # Load
        loaded = cache.load("AAPL")

        assert len(loaded) == len(sample_ohlcv_data)
        assert list(loaded.columns) == list(sample_ohlcv_data.columns)

    def test_load_nonexistent(self, tmp_path):
        """Test loading nonexistent symbol returns empty DataFrame."""
        cache = DataCache(cache_dir=tmp_path)

        result = cache.load("NONEXISTENT")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_save_incremental(self, sample_ohlcv_data, tmp_path):
        """Test incremental save (append without duplicates)."""
        cache = DataCache(cache_dir=tmp_path)

        # Save initial data
        initial_data = sample_ohlcv_data.iloc[:50]
        cache.save("AAPL", initial_data)

        # Save overlapping + new data
        new_data = sample_ohlcv_data.iloc[40:]
        cache.save("AAPL", new_data)

        # Load and check no duplicates
        loaded = cache.load("AAPL")

        # Should have all unique dates
        assert len(loaded) == len(sample_ohlcv_data)

    def test_get_date_range(self, sample_ohlcv_data, tmp_path):
        """Test getting date range from cache."""
        cache = DataCache(cache_dir=tmp_path)
        cache.save("AAPL", sample_ohlcv_data)

        start, end = cache.get_date_range("AAPL")

        assert start is not None
        assert end is not None
        assert start <= end


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager with temp database."""
        db_path = tmp_path / "test.db"
        return DatabaseManager(database_url=f"sqlite:///{db_path}")

    def test_init_creates_tables(self, db_manager):
        """Test that initialization creates required tables."""
        # Tables should be created on init
        assert db_manager is not None

    def test_save_and_get_prediction(self, db_manager):
        """Test saving and retrieving predictions."""
        from datetime import datetime

        # Save prediction
        db_manager.save_prediction(
            symbol="AAPL",
            date=datetime(2024, 1, 15),
            prediction=1,
            probability=0.75,
            model_name="test_model",
        )

        # Get predictions
        predictions = db_manager.get_predictions(symbol="AAPL")

        assert len(predictions) == 1
        assert predictions.iloc[0]["symbol"] == "AAPL"
        assert predictions.iloc[0]["prediction"] == 1
        assert predictions.iloc[0]["probability"] == 0.75

    def test_update_actual(self, db_manager):
        """Test updating actual values for predictions."""
        from datetime import datetime

        pred_date = datetime(2024, 1, 15)

        # Save prediction
        db_manager.save_prediction(
            symbol="AAPL",
            date=pred_date,
            prediction=1,
            probability=0.75,
            model_name="test_model",
        )

        # Update actual
        db_manager.update_actual("AAPL", pred_date, actual=1)

        # Verify
        predictions = db_manager.get_predictions(symbol="AAPL")
        assert predictions.iloc[0]["actual"] == 1

    def test_save_daily_metric(self, db_manager):
        """Test saving daily metrics."""
        from datetime import datetime

        db_manager.save_daily_metric(
            date=datetime(2024, 1, 15),
            accuracy=0.85,
            total=10,
            correct=8,
            model_name="test_model",
        )

        metrics = db_manager.get_metrics()

        assert len(metrics) == 1
        assert metrics.iloc[0]["accuracy"] == 0.85

    def test_save_model_record(self, db_manager):
        """Test saving model records."""
        db_manager.save_model_record(
            model_name="logistic_weekly",
            model_type="logistic_l1",
            accuracy=0.85,
            score=0.88,
            is_best=True,
        )

        best = db_manager.get_best_model()

        assert best is not None
        assert best["model_type"] == "logistic_l1"
        assert best["accuracy"] == 0.85

    def test_get_predictions_with_filters(self, db_manager):
        """Test prediction retrieval with filters."""
        from datetime import datetime

        # Save multiple predictions
        for i in range(5):
            db_manager.save_prediction(
                symbol="AAPL" if i < 3 else "GOOGL",
                date=datetime(2024, 1, 10 + i),
                prediction=1,
                probability=0.7,
                model_name="test_model",
            )

        # Filter by symbol
        aapl_preds = db_manager.get_predictions(symbol="AAPL")
        assert len(aapl_preds) == 3

        # Filter by date range
        range_preds = db_manager.get_predictions(
            start_date=datetime(2024, 1, 12),
            end_date=datetime(2024, 1, 14),
        )
        assert len(range_preds) == 3

    def test_duplicate_prediction_updates(self, db_manager):
        """Test that duplicate predictions update instead of insert."""
        from datetime import datetime

        pred_date = datetime(2024, 1, 15)

        # Save prediction
        db_manager.save_prediction(
            symbol="AAPL",
            date=pred_date,
            prediction=1,
            probability=0.75,
            model_name="test_model",
        )

        # Save same prediction with different values
        db_manager.save_prediction(
            symbol="AAPL",
            date=pred_date,
            prediction=0,
            probability=0.65,
            model_name="test_model",
        )

        # Should only have one record
        predictions = db_manager.get_predictions(symbol="AAPL")
        assert len(predictions) == 1
        assert predictions.iloc[0]["prediction"] == 0
        assert predictions.iloc[0]["probability"] == 0.65
