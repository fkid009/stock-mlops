"""MLflow model registry integration."""

import json
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.common import get_config, get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Manage models with MLflow tracking and registry."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize the registry.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
        """
        config = get_config()

        self.tracking_uri = tracking_uri or config.mlflow_tracking_uri
        self.experiment_name = experiment_name or config.mlflow_experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

        # Create or get experiment
        # Use mlflow-artifacts:/ scheme to route artifacts through the tracking server
        artifact_location = f"mlflow-artifacts:/{self.experiment_name}"

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=artifact_location,
            )
            logger.info(f"Created experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using experiment: {self.experiment_name}")

        mlflow.set_experiment(self.experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.

        Args:
            run_name: Optional run name
            tags: Optional tags

        Returns:
            Active run context
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_model(
        self,
        model: Any,
        model_type: str,
        metrics: dict,
        params: Optional[dict] = None,
        tags: Optional[dict] = None,
    ) -> str:
        """Log a trained model to MLflow.

        Args:
            model: Trained model instance
            model_type: Type of model
            metrics: Evaluation metrics
            params: Model parameters
            tags: Additional tags

        Returns:
            Run ID
        """
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            if params:
                mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            tags = tags or {}
            tags["model_type"] = model_type
            mlflow.set_tags(tags)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            run_id = run.info.run_id
            logger.info(f"Logged {model_type} model with run_id: {run_id}")

            return run_id

    def log_training_run(
        self,
        models: dict[str, Any],
        results: dict[str, dict],
        best_model_type: str,
        feature_names: list[str],
    ) -> dict[str, str]:
        """Log a complete training run with multiple models.

        Args:
            models: Dictionary of trained models
            results: Evaluation results for each model
            best_model_type: Type of the best model
            feature_names: List of feature names used

        Returns:
            Dictionary mapping model_type to run_id
        """
        run_ids = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_type, model in models.items():
            metrics = results.get(model_type, {})
            is_best = model_type == best_model_type

            tags = {
                "model_type": model_type,
                "is_best": str(is_best),
                "training_timestamp": timestamp,
            }

            run_name = f"{model_type}_{timestamp}"

            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters
                params = model.get_params()
                # Filter out non-serializable params
                serializable_params = {
                    k: v for k, v in params.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
                mlflow.log_params(serializable_params)

                # Log metrics
                mlflow.log_metrics({
                    "accuracy": metrics.get("accuracy", 0),
                    "f1": metrics.get("f1", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                })

                # Log feature names
                mlflow.log_dict({"features": feature_names}, "features.json")

                # Set tags
                mlflow.set_tags(tags)

                # Log model
                mlflow.sklearn.log_model(model, "model")

                run_ids[model_type] = run.info.run_id

        logger.info(f"Logged {len(run_ids)} models. Best: {best_model_type}")
        return run_ids

    def load_model(self, run_id: str) -> Any:
        """Load a model from MLflow.

        Args:
            run_id: MLflow run ID

        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from run {run_id}")
        return model

    def get_best_run(
        self,
        metric: str = "accuracy",
        order: str = "DESC",
    ) -> Optional[dict]:
        """Get the best run based on a metric.

        Args:
            metric: Metric to sort by
            order: Sort order (ASC or DESC)

        Returns:
            Best run info or None
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="tags.is_best = 'True'",
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "model_type": run.data.tags.get("model_type"),
                "accuracy": run.data.metrics.get("accuracy"),
                "timestamp": run.info.start_time,
            }
        return None

    def get_recent_runs(
        self,
        limit: int = 10,
        model_type: Optional[str] = None,
    ) -> list[dict]:
        """Get recent training runs.

        Args:
            limit: Maximum number of runs
            model_type: Filter by model type

        Returns:
            List of run info dicts
        """
        filter_string = ""
        if model_type:
            filter_string = f"tags.model_type = '{model_type}'"

        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=limit,
        )

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "model_type": run.data.tags.get("model_type"),
                "accuracy": run.data.metrics.get("accuracy"),
                "is_best": run.data.tags.get("is_best") == "True",
                "timestamp": run.info.start_time,
            }
            for run in runs
        ]

    def compare_models(self, run_ids: list[str]) -> pd.DataFrame:
        """Compare multiple model runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with comparison
        """
        data = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            data.append({
                "run_id": run_id,
                "model_type": run.data.tags.get("model_type"),
                "accuracy": run.data.metrics.get("accuracy"),
                "f1": run.data.metrics.get("f1"),
                "precision": run.data.metrics.get("precision"),
                "recall": run.data.metrics.get("recall"),
            })

        return pd.DataFrame(data)
