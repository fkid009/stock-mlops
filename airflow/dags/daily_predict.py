"""Daily prediction DAG.

Runs every weekday after market close to make predictions for the next day.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def collect_latest_data(**context):
    """Collect latest stock data."""
    from src.pipeline import collect_and_cache_data

    results = collect_and_cache_data(days=30, use_latest=True)

    context["ti"].xcom_push(key="data_date", value=datetime.now().isoformat())
    return f"Collected latest data for {len(results)} symbols"


def update_actuals(**context):
    """Update actual values for previous predictions."""
    from datetime import datetime, timedelta
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


def load_best_model(**context):
    """Load the best model for predictions."""
    from src.common import get_config
    from src.data import DatabaseManager
    from src.models import ModelPredictor

    config = get_config()
    db = DatabaseManager()

    # Get best model info
    best_model = db.get_best_model()

    if best_model and best_model.get("mlflow_run_id"):
        predictor = ModelPredictor.from_mlflow(
            best_model["mlflow_run_id"],
            best_model["model_type"],
        )
        context["ti"].xcom_push(key="model_source", value="mlflow")
        context["ti"].xcom_push(key="model_type", value=best_model["model_type"])
    else:
        # Fall back to local model
        model_path = config.data_dir / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("No trained model found")

        predictor = ModelPredictor.from_file(str(model_path), "unknown")
        context["ti"].xcom_push(key="model_source", value="local")
        context["ti"].xcom_push(key="model_type", value="unknown")

    return "Model loaded successfully"


def make_predictions(**context):
    """Make predictions for the next trading day."""
    from datetime import datetime
    import joblib
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
    prediction_date = datetime.now()

    for symbol in config.symbols_list:
        try:
            df = cache.load(symbol)
            if df.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue

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

            # Save to database
            db.save_prediction(
                symbol=symbol,
                date=prediction_date,
                prediction=int(result["prediction"].iloc[0]),
                probability=float(result["probability"].iloc[0]),
                model_name=f"{model_type}_daily",
            )

            predictions_made += 1

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
    schedule_interval="0 0 * * 1-5",  # Weekdays at 9 AM KST (UTC 00:00)
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

    load_model_task = PythonOperator(
        task_id="load_best_model",
        python_callable=load_best_model,
    )

    predict_task = PythonOperator(
        task_id="make_predictions",
        python_callable=make_predictions,
    )

    collect_task >> update_actuals_task >> accuracy_task
    collect_task >> load_model_task >> predict_task
