"""Model training for stock prediction."""

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

from src.common import get_logger, get_models_config, timed

logger = get_logger(__name__)


class ModelTrainer:
    """Train classification models for stock prediction."""

    # Model class mapping
    MODEL_CLASSES = {
        "logistic_l1": LogisticRegression,
        "lightgbm": lgb.LGBMClassifier,
        "svm": SVC,
    }

    def __init__(self):
        """Initialize the trainer."""
        self.config = get_models_config()
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict] = {}

    def _get_model_config(self, model_type: str) -> dict:
        """Get configuration for a model type."""
        if model_type not in self.config["models"]:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.config["models"][model_type]

    def _create_model(self, model_type: str, params: Optional[dict] = None) -> Any:
        """Create a model instance.

        Args:
            model_type: Type of model (logistic_l1, lightgbm, svm)
            params: Optional parameters override

        Returns:
            Model instance
        """
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")

        model_config = self._get_model_config(model_type)
        model_params = {**model_config["params"], **(params or {})}

        model_class = self.MODEL_CLASSES[model_type]
        return model_class(**model_params)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        params: Optional[dict] = None,
    ) -> Any:
        """Train a single model.

        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to train
            params: Optional parameters override

        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")

        model = self._create_model(model_type, params)
        model.fit(X_train, y_train)

        self.models[model_type] = model
        logger.info(f"Trained {model_type} model on {len(X_train)} samples")

        return model

    @timed
    def train_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        cv: int = 3,
    ) -> tuple[Any, dict]:
        """Train a model with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to train
            cv: Number of cross-validation folds

        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info(f"Training {model_type} with hyperparameter tuning...")

        model_config = self._get_model_config(model_type)
        base_model = self._create_model(model_type)
        param_grid = model_config.get("tuning_grid", {})

        if not param_grid:
            logger.warning(f"No tuning grid for {model_type}, training with defaults")
            return self.train(X_train, y_train, model_type), model_config["params"]

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        self.models[model_type] = best_model
        logger.info(f"Best params for {model_type}: {best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return best_model, best_params

    @timed
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune: bool = False,
    ) -> dict[str, Any]:
        """Train all configured models.

        Args:
            X_train: Training features
            y_train: Training target
            tune: Whether to perform hyperparameter tuning

        Returns:
            Dictionary of trained models
        """
        model_types = list(self.config["models"].keys())

        for model_type in model_types:
            try:
                if tune:
                    model, _ = self.train_with_tuning(X_train, y_train, model_type)
                else:
                    model = self.train(X_train, y_train, model_type)
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue

        return self.models

    def evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: Optional[str] = None,
    ) -> dict[str, dict]:
        """Evaluate trained models on validation data.

        Args:
            X_val: Validation features
            y_val: Validation target
            model_type: Specific model to evaluate (or all if None)

        Returns:
            Dictionary of evaluation results
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        models_to_eval = {model_type: self.models[model_type]} if model_type else self.models

        results = {}
        for name, model in models_to_eval.items():
            y_pred = model.predict(X_val)

            results[name] = {
                "accuracy": accuracy_score(y_val, y_pred),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "n_samples": len(y_val),
            }
            logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}")

        self.results = results
        return results

    def select_best_model(
        self,
        results: Optional[dict[str, dict]] = None,
        accuracy_weight: float = 0.7,
        stability_weight: float = 0.3,
    ) -> tuple[str, Any, float]:
        """Select the best model based on combined score.

        Args:
            results: Evaluation results (uses self.results if None)
            accuracy_weight: Weight for accuracy in score
            stability_weight: Weight for stability (1 - std)

        Returns:
            Tuple of (model_type, model, score)
        """
        results = results or self.results

        if not results:
            raise ValueError("No evaluation results available")

        # Calculate scores
        scores = {}
        for model_type, metrics in results.items():
            accuracy = metrics["accuracy"]
            # For stability, use 1 if we don't have std (single evaluation)
            stability = 1.0 - metrics.get("accuracy_std", 0.0)

            score = accuracy_weight * accuracy + stability_weight * stability
            scores[model_type] = score
            logger.info(f"{model_type} combined score: {score:.4f}")

        # Select best
        best_type = max(scores, key=scores.get)
        best_model = self.models[best_type]
        best_score = scores[best_type]

        logger.info(f"Selected best model: {best_type} (score: {best_score:.4f})")

        return best_type, best_model, best_score

    def get_model(self, model_type: str) -> Any:
        """Get a trained model by type.

        Args:
            model_type: Type of model

        Returns:
            Trained model
        """
        if model_type not in self.models:
            raise ValueError(f"Model not trained: {model_type}")
        return self.models[model_type]

    def get_model_params(self, model_type: str) -> dict:
        """Get parameters of a trained model.

        Args:
            model_type: Type of model

        Returns:
            Model parameters
        """
        model = self.get_model(model_type)
        return model.get_params()

    def save_model(self, model_type: str, path: str) -> None:
        """Save a trained model to file.

        Args:
            model_type: Type of model to save
            path: Path to save model
        """
        import joblib

        model = self.get_model(model_type)
        joblib.dump(model, path)
        logger.info(f"Saved {model_type} model to {path}")

    def load_model(self, model_type: str, path: str) -> Any:
        """Load a model from file.

        Args:
            model_type: Type of model
            path: Path to load from

        Returns:
            Loaded model
        """
        import joblib

        model = joblib.load(path)
        self.models[model_type] = model
        logger.info(f"Loaded {model_type} model from {path}")
        return model
