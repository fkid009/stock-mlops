"""Daily prediction DAG.

Runs every weekday after market close to make predictions for the next day.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 2, 1),  # Recent start date
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def collect_latest_data(**context):
    """Collect latest stock data."""
    from src.common import get_pipeline_config
    from src.pipeline import collect_and_cache_data

    pipeline_config = get_pipeline_config()
    prediction_days = pipeline_config.get("data", {}).get("daily_prediction_days", 30)

    results = collect_and_cache_data(days=prediction_days, use_latest=True)

    context["ti"].xcom_push(key="data_date", value=datetime.now().isoformat())
    return f"Collected latest data for {len(results)} symbols"


def update_actuals(**context):
    """Update actual values for previous predictions."""
    import pandas as pd
    from src.common import get_config, get_logger
    from src.data import DataCache, DatabaseManager

    logger = get_logger(__name__)
    config = get_config()
    cache = DataCache()
    db = DatabaseManager()

    updated_count = 0

    for symbol in config.symbols_list:
        df = cache.load(symbol)
        if df.empty or len(df) < 2:
            continue

        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Calculate actual direction for each day (comparing with next day)
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # Calculate price change
            if current_row["close"] == 0:
                continue

            change = (next_row["close"] - current_row["close"]) / current_row["close"]

            # Determine actual direction (threshold: ±0.1%)
            if abs(change) > 0.001:
                actual = 1 if change > 0 else 0
                pred_date = current_row["date"].to_pydatetime()

                # Remove timezone info for database consistency
                if pred_date.tzinfo is not None:
                    pred_date = pred_date.replace(tzinfo=None)

                try:
                    db.update_actual(symbol, pred_date, actual)
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to update actual for {symbol} on {pred_date}: {e}")

    logger.info(f"Updated {updated_count} actual values")
    return f"Updated {updated_count} actual values"


def calculate_daily_accuracy(**context):
    """Calculate and save daily accuracy."""
    from datetime import datetime
    import pandas as pd
    from src.data import DatabaseManager
    from src.pipeline.utils import calculate_accuracy

    db = DatabaseManager()

    # Get predictions with actuals from recent days
    predictions = db.get_predictions(limit=500)

    if predictions.empty:
        return "No predictions to evaluate"

    # Filter predictions that have actual values
    valid_preds = predictions[predictions["actual"].notna()]

    if valid_preds.empty:
        return "No predictions with actual values"

    # Group by date and calculate accuracy
    valid_preds["date"] = pd.to_datetime(valid_preds["date"]).dt.date

    for date, group in valid_preds.groupby("date"):
        accuracy = calculate_accuracy(group)
        if accuracy is None:
            continue

        correct = (group["prediction"] == group["actual"]).sum()
        total = len(group)

        db.save_daily_metric(
            date=datetime.combine(date, datetime.min.time()),
            accuracy=accuracy,
            total=total,
            correct=correct,
            model_name=group["model_name"].iloc[0],
        )

    return f"Calculated accuracy for {len(valid_preds['date'].unique())} days"


def train_daily_model(**context):
    """Train model with latest data using fixed hyperparameters from weekly training."""
    import json
    from src.common import get_config, get_logger
    from src.data import DatabaseManager
    from src.models import ModelTrainer
    from src.pipeline import batch_prepare_dataset

    logger = get_logger(__name__)
    config = get_config()
    db = DatabaseManager()

    # Get best model config from weekly training
    best_model = db.get_best_model()
    if not best_model:
        raise ValueError("No best model found. Run weekly training first.")

    model_type = best_model["model_type"]
    hyperparameters = (
        json.loads(best_model["hyperparameters"])
        if best_model.get("hyperparameters")
        else {}
    )

    logger.info(f"Training {model_type} with hyperparameters from weekly training")

    # Prepare dataset with scaler
    scaler_path = config.data_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError("Scaler not found. Run weekly training first.")

    X, y = batch_prepare_dataset(scaler_path=str(scaler_path))

    # Train model with fixed hyperparameters
    trainer = ModelTrainer()
    trainer.train(X, y, model_type, params=hyperparameters)

    # Save model
    model_path = config.data_dir / "best_model.joblib"
    trainer.save_model(model_type, str(model_path))

    # Update model record in DB (mark as new best with updated trained_at)
    db.save_model_record(
        model_name=f"{model_type}_daily",
        model_type=model_type,
        accuracy=best_model["accuracy"],
        score=best_model["score"],
        hyperparameters=json.dumps(hyperparameters),
        is_best=True,
    )

    context["ti"].xcom_push(key="model_type", value=model_type)
    logger.info(f"Trained and saved {model_type} model with latest data")
    return f"Trained {model_type} model with latest data"


def make_predictions(**context):
    """Make predictions for the next trading day."""
    import joblib
    import pandas as pd
    from src.common import get_config, get_logger
    from src.data import DataCache, DatabaseManager
    from src.features import FeatureEngineer
    from src.models import ModelPredictor

    logger = get_logger(__name__)
    config = get_config()
    cache = DataCache()
    db = DatabaseManager()

    # Load model with error handling
    model_path = config.data_dir / "best_model.joblib"
    model_type = context["ti"].xcom_pull(key="model_type")

    if model_type is None:
        logger.warning("model_type not found in xcom, using 'unknown'")
        model_type = "unknown"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(str(model_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    predictor = ModelPredictor(model, model_type)

    # Load scaler - required for accurate predictions
    engineer = FeatureEngineer()
    scaler_path = config.data_dir / "scaler.joblib"
    if scaler_path.exists():
        engineer.load_scaler(str(scaler_path))
        logger.info("Scaler loaded successfully")
    else:
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Run weekly training first to create the scaler."
        )

    predictions_made = 0
    prediction_errors = 0

    for symbol in config.symbols_list:
        try:
            df = cache.load(symbol)
            if df.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue

            # Ensure date column is datetime and get the last date
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Use the last data date as prediction date (not datetime.now())
            # This ensures consistency with update_actuals matching
            prediction_date = df["date"].iloc[-1].to_pydatetime()

            # Remove timezone info for database consistency
            if prediction_date.tzinfo is not None:
                prediction_date = prediction_date.replace(tzinfo=None)

            # Prepare features for the latest data point
            df = engineer.compute_features(df)
            df = engineer.handle_missing_values(df)

            feature_cols = engineer.get_feature_names()
            latest = df[feature_cols].iloc[[-1]]

            if latest.isna().any().any():
                logger.warning(f"Missing features for {symbol}, skipping")
                continue

            # Scale features
            if engineer._is_fitted:
                latest = engineer.transform(latest.reset_index(drop=True))

            # Make prediction
            result = predictor.predict_with_confidence(latest)

            # Save to database (save_prediction handles duplicates by updating)
            db.save_prediction(
                symbol=symbol,
                date=prediction_date,
                prediction=int(result["prediction"].iloc[0]),
                probability=float(result["probability"].iloc[0]),
                model_name=f"{model_type}_daily",
            )

            predictions_made += 1
            logger.info(f"Prediction for {symbol} on {prediction_date.date()}: {result['prediction'].iloc[0]}")

        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            prediction_errors += 1
            continue

    context["ti"].xcom_push(key="predictions_made", value=predictions_made)
    context["ti"].xcom_push(key="prediction_errors", value=prediction_errors)

    return f"Made {predictions_made} predictions ({prediction_errors} errors)"


with DAG(
    "daily_prediction",
    default_args=default_args,
    description="Daily stock movement predictions",
    schedule_interval="0 22 * * 1-5",  # Weekdays 17:00 EST (after market close)
    catchup=False,
    tags=["stock", "ml", "prediction"],
) as dag:

    collect_task = PythonOperator(
        task_id="collect_latest_data",
        python_callable=collect_latest_data,
    )

    update_actuals_task = PythonOperator(
        task_id="update_actuals",
        python_callable=update_actuals,
    )

    accuracy_task = PythonOperator(
        task_id="calculate_daily_accuracy",
        python_callable=calculate_daily_accuracy,
    )

    train_model_task = PythonOperator(
        task_id="train_daily_model",
        python_callable=train_daily_model,
    )

    predict_task = PythonOperator(
        task_id="make_predictions",
        python_callable=make_predictions,
    )

    collect_task >> update_actuals_task >> accuracy_task
    collect_task >> train_model_task >> predict_task
