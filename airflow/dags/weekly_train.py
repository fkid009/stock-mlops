"""Weekly model training DAG.

Runs every Sunday to train all models and select the best one.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 2, 1),  # Recent start date
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def collect_data(**context):
    """Collect stock data for training."""
    from src.common import get_pipeline_config
    from src.pipeline import collect_and_cache_data

    pipeline_config = get_pipeline_config()
    training_days = pipeline_config.get("data", {}).get("weekly_training_days", 750)

    results = collect_and_cache_data(days=training_days, use_latest=False)

    context["ti"].xcom_push(key="data_collected", value=True)
    return f"Collected data for {len(results)} symbols"


def prepare_features(**context):
    """Prepare features for training."""
    import pandas as pd
    from src.common import get_config
    from src.features import FeatureEngineer
    from src.pipeline import load_and_prepare_features

    config = get_config()
    engineer = FeatureEngineer()

    # Load and prepare features for all symbols
    all_dfs = load_and_prepare_features(include_target=True)

    if not all_dfs:
        raise ValueError("No data available for feature preparation")

    # Combine all data and fit scaler
    combined_df = pd.concat(all_dfs, ignore_index=True)
    feature_cols = engineer.get_feature_names()

    # Remove rows with NaN in features or target
    valid_mask = ~combined_df[feature_cols].isna().any(axis=1) & ~combined_df["target"].isna()
    combined_df = combined_df[valid_mask]

    # Fit and save scaler
    engineer.fit_scaler(combined_df)
    scaler_path = config.data_dir / "scaler.joblib"
    engineer.save_scaler(str(scaler_path))

    context["ti"].xcom_push(key="scaler_path", value=str(scaler_path))
    context["ti"].xcom_push(key="total_samples", value=len(combined_df))
    return f"Prepared features for {len(all_dfs)} symbols, {len(combined_df)} total samples"


def train_models(**context):
    """Train all configured models."""
    from src.common import Timer
    from src.models import ModelTrainer
    from src.evaluation import ModelValidator
    from src.evaluation.drift import DriftDetector
    from src.pipeline import batch_prepare_dataset

    # Check if tuning is needed
    drift_detector = DriftDetector()
    should_tune = drift_detector.should_trigger_tuning()

    # Load scaler path and prepare dataset
    scaler_path = context["ti"].xcom_pull(key="scaler_path")
    if not scaler_path:
        raise ValueError("Scaler path not found in xcom. Run prepare_features first.")

    with Timer("dataset preparation"):
        X, y = batch_prepare_dataset(scaler_path=scaler_path)

    # Split data
    validator = ModelValidator()
    X_train, X_val, y_train, y_val = validator.time_series_split(X, y)

    # Train models
    trainer = ModelTrainer()
    with Timer(f"training all models (tune={should_tune})"):
        trainer.train_all(X_train, y_train, tune=should_tune)

    # Evaluate models
    results = trainer.evaluate(X_val, y_val)

    # Select best model
    best_type, best_model, best_score = trainer.select_best_model(results)

    # Save results to XCom
    context["ti"].xcom_push(key="best_model_type", value=best_type)
    context["ti"].xcom_push(key="best_score", value=best_score)
    context["ti"].xcom_push(key="results", value=results)
    context["ti"].xcom_push(key="tuned", value=should_tune)

    return f"Best model: {best_type} with score {best_score:.4f}"


def register_models(**context):
    """Register trained models with MLflow."""
    import json
    from src.common import get_config
    from src.data import DatabaseManager
    from src.features import ALL_FEATURES
    from src.models import ModelTrainer, ModelRegistry
    from src.pipeline import batch_prepare_dataset

    config = get_config()

    best_type = context["ti"].xcom_pull(key="best_model_type")
    best_score = context["ti"].xcom_pull(key="best_score")
    results = context["ti"].xcom_pull(key="results")
    scaler_path = context["ti"].xcom_pull(key="scaler_path")

    # Prepare dataset using common pipeline
    X, y = batch_prepare_dataset(scaler_path=scaler_path)

    trainer = ModelTrainer()
    trainer.train_all(X, y, tune=False)

    # Register with MLflow
    registry = ModelRegistry()
    run_ids = registry.log_training_run(
        models=trainer.models,
        results=results,
        best_model_type=best_type,
        feature_names=ALL_FEATURES,
    )

    # Save to database with consistent score calculation
    db = DatabaseManager()
    accuracy_weight = 0.7
    stability_weight = 0.3

    for model_type, metrics in results.items():
        accuracy = metrics["accuracy"]
        accuracy_std = metrics.get("accuracy_std", 0.0)
        stability = 1.0 - accuracy_std
        score = accuracy_weight * accuracy + stability_weight * stability

        db.save_model_record(
            model_name=f"{model_type}_weekly",
            model_type=model_type,
            accuracy=accuracy,
            accuracy_std=accuracy_std,
            score=score,
            mlflow_run_id=run_ids.get(model_type),
            hyperparameters=json.dumps(trainer.get_model_params(model_type)),
            is_best=(model_type == best_type),
        )

    # Save best model locally
    model_path = config.data_dir / "best_model.joblib"
    trainer.save_model(best_type, str(model_path))

    return f"Registered {len(run_ids)} models. Best: {best_type}"


with DAG(
    "weekly_model_training",
    default_args=default_args,
    description="Weekly training of stock prediction models",
    schedule_interval="0 22 * * 0",  # Sunday 17:00 EST (market closed)
    catchup=False,
    tags=["stock", "ml", "training"],
) as dag:

    collect_task = PythonOperator(
        task_id="collect_data",
        python_callable=collect_data,
    )

    prepare_task = PythonOperator(
        task_id="prepare_features",
        python_callable=prepare_features,
    )

    train_task = PythonOperator(
        task_id="train_models",
        python_callable=train_models,
    )

    register_task = PythonOperator(
        task_id="register_models",
        python_callable=register_models,
    )

    collect_task >> prepare_task >> train_task >> register_task
