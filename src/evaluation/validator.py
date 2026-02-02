"""Model validation for stock prediction."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.common import get_logger, get_models_config

logger = get_logger(__name__)


class ModelValidator:
    """Validate models using various strategies."""

    def __init__(
        self,
        validation_size: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        """Initialize validator.

        Args:
            validation_size: Number of days for validation (default from config)
            window_size: Number of days for training window (default from config)
        """
        config = get_models_config()
        training_config = config.get("training", {})

        self.validation_size = (
            validation_size
            if validation_size is not None
            else training_config.get("validation_size", 20)
        )
        self.window_size = (
            window_size
            if window_size is not None
            else training_config.get("window_size", 120)
        )

    def time_series_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data for time series validation.

        Uses the last `validation_size` samples for validation.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if len(X) <= self.validation_size:
            raise ValueError(
                f"Not enough data: {len(X)} samples, need > {self.validation_size}"
            )

        split_idx = len(X) - self.validation_size

        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]

        logger.info(
            f"Split data: {len(X_train)} train, {len(X_val)} validation"
        )

        return X_train, X_val, y_train, y_val

    def sliding_window_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 3,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Split data using sliding window for multiple validation folds.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of splits

        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        splits = []
        total_len = len(X)
        fold_size = self.validation_size

        for i in range(n_splits):
            val_end = total_len - (i * fold_size)
            val_start = val_end - fold_size
            train_end = val_start
            train_start = max(0, train_end - self.window_size)

            if train_start >= train_end or val_start >= val_end:
                break

            X_train = X.iloc[train_start:train_end]
            X_val = X.iloc[val_start:val_end]
            y_train = y.iloc[train_start:train_end]
            y_val = y.iloc[val_start:val_end]

            splits.append((X_train, X_val, y_train, y_val))

        logger.info(f"Created {len(splits)} sliding window splits")
        return splits

    def evaluate_model(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Evaluate a model on validation data.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target

        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_val)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "n_samples": len(y_val),
            "n_positive": int(y_val.sum()),
            "n_negative": int(len(y_val) - y_val.sum()),
        }

        logger.info(
            f"Evaluation: accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, n={metrics['n_samples']}"
        )

        return metrics

    def evaluate_with_sliding_window(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 3,
    ) -> dict:
        """Evaluate model using sliding window cross-validation.

        Args:
            model: Trained model (or model class that can be cloned)
            X: Full feature DataFrame
            y: Full target Series
            n_splits: Number of splits

        Returns:
            Dictionary of aggregated metrics
        """
        from sklearn.base import clone

        splits = self.sliding_window_split(X, y, n_splits)
        all_metrics = []

        for i, (X_train, X_val, y_train, y_val) in enumerate(splits):
            # Clone and retrain model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            metrics = self.evaluate_model(model_clone, X_val, y_val)
            all_metrics.append(metrics)
            logger.debug(f"Fold {i+1}: accuracy={metrics['accuracy']:.4f}")

        # Aggregate metrics
        aggregated = {
            "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "accuracy_std": np.std([m["accuracy"] for m in all_metrics]),
            "f1": np.mean([m["f1"] for m in all_metrics]),
            "f1_std": np.std([m["f1"] for m in all_metrics]),
            "precision": np.mean([m["precision"] for m in all_metrics]),
            "recall": np.mean([m["recall"] for m in all_metrics]),
            "n_folds": len(all_metrics),
            "total_samples": sum(m["n_samples"] for m in all_metrics),
        }

        logger.info(
            f"Sliding window evaluation: "
            f"accuracy={aggregated['accuracy']:.4f} ± {aggregated['accuracy_std']:.4f}"
        )

        return aggregated

    def get_confusion_matrix(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> np.ndarray:
        """Get confusion matrix for predictions.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target

        Returns:
            Confusion matrix array
        """
        y_pred = model.predict(X_val)
        return confusion_matrix(y_val, y_pred)

    def get_classification_report(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> str:
        """Get classification report.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target

        Returns:
            Classification report string
        """
        y_pred = model.predict(X_val)
        return classification_report(
            y_val, y_pred,
            target_names=["Down", "Up"],
            zero_division=0,
        )

    def compare_with_baseline(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Compare model performance with simple baselines.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target

        Returns:
            Comparison results
        """
        y_pred = model.predict(X_val)
        model_accuracy = accuracy_score(y_val, y_pred)

        # Baseline 1: Always predict majority class
        majority_class = y_val.mode().iloc[0]
        majority_accuracy = (y_val == majority_class).mean()

        # Baseline 2: Random prediction
        random_accuracy = 0.5

        # Baseline 3: Predict based on class distribution
        class_ratio = y_val.mean()
        weighted_accuracy = max(class_ratio, 1 - class_ratio)

        return {
            "model_accuracy": model_accuracy,
            "majority_baseline": majority_accuracy,
            "random_baseline": random_accuracy,
            "weighted_baseline": weighted_accuracy,
            "vs_majority": model_accuracy - majority_accuracy,
            "vs_random": model_accuracy - random_accuracy,
        }
