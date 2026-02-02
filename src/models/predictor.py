"""Model prediction for stock direction."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.common import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """Make predictions using trained models."""

    def __init__(self, model: Any, model_type: str):
        """Initialize the predictor.

        Args:
            model: Trained model instance
            model_type: Type of model (logistic_l1, lightgbm, svm)
        """
        self.model = model
        self.model_type = model_type

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        predictions = self.model.predict(X)
        logger.debug(f"Made {len(predictions)} predictions")
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities for class 1
        """
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            # Return probability of class 1
            return probas[:, 1] if probas.ndim > 1 else probas
        else:
            logger.warning(f"{self.model_type} does not support predict_proba")
            return self.predict(X).astype(float)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """Make predictions with confidence scores.

        Args:
            X: Feature DataFrame
            confidence_threshold: Minimum confidence for prediction

        Returns:
            DataFrame with prediction, probability, and confidence flag
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        # Confidence is distance from 0.5 (scaled)
        confidence = np.abs(probabilities - 0.5) * 2

        result = pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities,
            "confidence": confidence,
            "high_confidence": confidence >= confidence_threshold,
        }, index=X.index)

        high_conf_count = result["high_confidence"].sum()
        logger.info(
            f"Predictions: {len(result)}, "
            f"High confidence: {high_conf_count} ({high_conf_count/len(result)*100:.1f}%)"
        )

        return result

    def batch_predict(
        self,
        X: pd.DataFrame,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """Make predictions in batches.

        Args:
            X: Feature DataFrame
            batch_size: Size of each batch

        Returns:
            DataFrame with predictions and probabilities
        """
        all_predictions = []
        all_probabilities = []

        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i + batch_size]
            predictions = self.predict(batch)
            probabilities = self.predict_proba(batch)

            # Convert numpy arrays to lists for extend
            all_predictions.extend(predictions.tolist())
            all_probabilities.extend(probabilities.tolist())

        result = pd.DataFrame({
            "prediction": all_predictions,
            "probability": all_probabilities,
        }, index=X.index)

        return result

    @classmethod
    def from_file(cls, path: str, model_type: str) -> "ModelPredictor":
        """Load a predictor from a saved model file.

        Args:
            path: Path to model file
            model_type: Type of model

        Returns:
            ModelPredictor instance
        """
        import joblib

        model = joblib.load(path)
        logger.info(f"Loaded {model_type} model from {path}")
        return cls(model, model_type)

    @classmethod
    def from_mlflow(cls, run_id: str, model_type: str) -> "ModelPredictor":
        """Load a predictor from MLflow.

        Args:
            run_id: MLflow run ID
            model_type: Type of model

        Returns:
            ModelPredictor instance
        """
        import mlflow

        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded {model_type} model from MLflow run {run_id}")
        return cls(model, model_type)
