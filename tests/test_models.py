"""Tests for model training module."""

import numpy as np
import pytest

from src.models import ModelTrainer


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_init(self):
        """Test initialization."""
        trainer = ModelTrainer()

        assert trainer.models == {}
        assert trainer.results == {}
        assert "logistic_l1" in trainer.MODEL_CLASSES
        assert "lightgbm" in trainer.MODEL_CLASSES
        assert "svm" in trainer.MODEL_CLASSES

    def test_create_model_logistic(self):
        """Test logistic regression model creation."""
        trainer = ModelTrainer()
        model = trainer._create_model("logistic_l1")

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_lightgbm(self):
        """Test LightGBM model creation."""
        trainer = ModelTrainer()
        model = trainer._create_model("lightgbm")

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_svm(self):
        """Test SVM model creation."""
        trainer = ModelTrainer()
        model = trainer._create_model("svm")

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_unknown(self):
        """Test unknown model type raises error."""
        trainer = ModelTrainer()

        with pytest.raises(ValueError, match="Unknown model type"):
            trainer._create_model("unknown_model")

    def test_train_single_model(self, sample_train_data):
        """Test training a single model."""
        X, y = sample_train_data
        trainer = ModelTrainer()

        model = trainer.train(X, y, "logistic_l1")

        assert model is not None
        assert "logistic_l1" in trainer.models

    def test_train_all_models(self, sample_train_data):
        """Test training all models."""
        X, y = sample_train_data
        trainer = ModelTrainer()

        models = trainer.train_all(X, y, tune=False)

        assert len(models) == 3
        assert "logistic_l1" in models
        assert "lightgbm" in models
        assert "svm" in models

    def test_evaluate(self, sample_train_data):
        """Test model evaluation."""
        X, y = sample_train_data
        trainer = ModelTrainer()

        # Train first
        trainer.train(X, y, "logistic_l1")

        # Split for evaluation
        split_idx = int(len(X) * 0.8)
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

        results = trainer.evaluate(X_val, y_val)

        assert "logistic_l1" in results
        assert "accuracy" in results["logistic_l1"]
        assert "f1" in results["logistic_l1"]
        assert "precision" in results["logistic_l1"]
        assert "recall" in results["logistic_l1"]
        assert 0 <= results["logistic_l1"]["accuracy"] <= 1

    def test_select_best_model(self, sample_train_data):
        """Test best model selection."""
        X, y = sample_train_data
        trainer = ModelTrainer()

        # Train all models
        trainer.train_all(X, y, tune=False)

        # Evaluate
        split_idx = int(len(X) * 0.8)
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
        trainer.evaluate(X_val, y_val)

        # Select best
        best_type, best_model, best_score = trainer.select_best_model()

        assert best_type in trainer.models
        assert best_model is not None
        assert 0 <= best_score <= 1

    def test_save_load_model(self, sample_train_data, tmp_path):
        """Test model save and load."""
        X, y = sample_train_data
        trainer = ModelTrainer()

        # Train
        trainer.train(X, y, "logistic_l1")

        # Save
        model_path = str(tmp_path / "model.joblib")
        trainer.save_model("logistic_l1", model_path)

        # Load into new trainer
        new_trainer = ModelTrainer()
        new_trainer.load_model("logistic_l1", model_path)

        assert "logistic_l1" in new_trainer.models

        # Verify predictions are same
        pred1 = trainer.models["logistic_l1"].predict(X)
        pred2 = new_trainer.models["logistic_l1"].predict(X)
        np.testing.assert_array_equal(pred1, pred2)

    def test_get_model_params(self, sample_train_data):
        """Test getting model parameters."""
        X, y = sample_train_data
        trainer = ModelTrainer()

        trainer.train(X, y, "logistic_l1")
        params = trainer.get_model_params("logistic_l1")

        assert isinstance(params, dict)
        assert "penalty" in params or "C" in params

    def test_get_model_not_trained(self):
        """Test getting untrained model raises error."""
        trainer = ModelTrainer()

        with pytest.raises(ValueError, match="Model not trained"):
            trainer.get_model("logistic_l1")
