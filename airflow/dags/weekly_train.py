"""Weekly model training DAG.

Runs every Sunday to train all models and select the best one.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def collect_data(**context):
    """Collect stock data for training."""
    from src.common import get_config
    from src.data import StockDataCollector, DataCache

    config = get_config()
    collector = StockDataCollector(config.symbols_list)
    cache = DataCache()

    all_data = []
    for symbol in config.symbols_list:
        df = collector.fetch(symbol, days=150)  # Extra days for feature computation
        if not df.empty:
            cache.save(symbol, df)
            all_data.append(df)

    context["ti"].xcom_push(key="data_collected", value=True)
    return f"Collected data for {len(all_data)} symbols"


def prepare_features(**context):
    """Prepare features for training."""
    import pandas as pd
    from src.common import get_config
    from src.data import DataCache
    from src.features import FeatureEngineer

    config = get_config()
    cache = DataCache()
    engineer = FeatureEngineer()

    # Step 1: Compute features for all symbols (without scaling)
    all_dfs = []
    for symbol in config.symbols_list:
        df = cache.load(symbol)
        if df.empty:
            continue

        df = engineer.compute_features(df)
        df = engineer.compute_target(df)
        df = engineer.handle_missing_values(df)
        df["symbol"] = symbol
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data available for feature preparation")

    # Step 2: Combine all data and fit scaler on entire dataset
    combined_df = pd.concat(all_dfs, ignore_index=True)
    feature_cols = engineer.get_feature_names()

    # Remove rows with NaN in features or target
    valid_mask = ~combined_df[feature_cols].isna().any(axis=1) & ~combined_df["target"].isna()
    combined_df = combined_df[valid_mask]

    # Fit scaler on all data
    engineer.fit_scaler(combined_df)

    # Save scaler
    scaler_path = config.data_dir / "scaler.joblib"
    engineer.save_scaler(str(scaler_path))

    context["ti"].xcom_push(key="scaler_path", value=str(scaler_path))
    context["ti"].xcom_push(key="total_samples", value=len(combined_df))
    return f"Prepared features for {len(all_dfs)} symbols, {len(combined_df)} total samples"


def train_models(**context):
    """Train all configured models."""
    import pandas as pd
    from src.common import get_config
    from src.data import DataCache
    from src.features import FeatureEngineer
    from src.models import ModelTrainer
    from src.evaluation import ModelValidator
    from src.evaluation.drift import DriftDetector

    config = get_config()
    cache = DataCache()

    # Check if tuning is needed
    drift_detector = DriftDetector()
    should_tune = drift_detector.should_trigger_tuning()

    # Reload data and prepare features
    engineer = FeatureEngineer()
    scaler_path = context["ti"].xcom_pull(key="scaler_path")
    if not scaler_path:
        raise ValueError("Scaler path not found in xcom. Run prepare_features first.")
    engineer.load_scaler(scaler_path)

    all_X = []
    all_y = []

    for symbol in config.symbols_list:
        df = cache.load(symbol)
        if df.empty:
            continue
        X, y = engineer.prepare_dataset(df, fit=False)
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No data available for training")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    # Split data
    validator = ModelValidator()
    X_train, X_val, y_train, y_val = validator.time_series_split(X, y)

    # Train models
    trainer = ModelTrainer()
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
    import pandas as pd
    from src.common import get_config
    from src.data import DataCache, DatabaseManager
    from src.features import FeatureEngineer, ALL_FEATURES
    from src.models import ModelTrainer, ModelRegistry

    config = get_config()

    best_type = context["ti"].xcom_pull(key="best_model_type")
    best_score = context["ti"].xcom_pull(key="best_score")
    results = context["ti"].xcom_pull(key="results")

    # Load scaler from previous step
    cache = DataCache()
    engineer = FeatureEngineer()
    scaler_path = context["ti"].xcom_pull(key="scaler_path")
    if scaler_path:
        engineer.load_scaler(scaler_path)

    trainer = ModelTrainer()

    all_X = []
    all_y = []

    for symbol in config.symbols_list:
        df = cache.load(symbol)
        if df.empty:
            continue
        # Use fit=False to reuse the existing scaler
        X, y = engineer.prepare_dataset(df, fit=False)
        all_X.append(X)
        all_y.append(y)

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

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
    schedule_interval="0 0 * * 0",  # Every Sunday at midnight
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
