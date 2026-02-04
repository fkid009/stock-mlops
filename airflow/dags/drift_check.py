"""Drift check DAG.

Runs daily after predictions to check for performance drift.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 2, 1),  # Recent start date
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_performance_drift(**context):
    """Check for performance drift."""
    from src.evaluation.drift import DriftDetector

    detector = DriftDetector()
    result = detector.check_performance_drift()

    context["ti"].xcom_push(key="drift_result", value=result)
    context["ti"].xcom_push(key="drift_detected", value=result.get("drift_detected", False))

    if result.get("drift_detected"):
        return f"Drift detected! Drop: {result.get('accuracy_drop', 0):.4f}"
    else:
        return "No drift detected"


def decide_action(**context):
    """Decide next action based on drift detection."""
    drift_detected = context["ti"].xcom_pull(key="drift_detected")

    if drift_detected:
        return "trigger_alert"
    else:
        return "no_action"


def trigger_alert(**context):
    """Trigger alert for drift detection."""
    from src.common import get_logger

    logger = get_logger("drift_alert")

    drift_result = context["ti"].xcom_pull(key="drift_result")

    def format_value(value, fmt=".4f"):
        """Format value safely, handling None."""
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except (TypeError, ValueError):
            return str(value)

    message = (
        f"Performance drift detected!\n"
        f"Baseline accuracy: {format_value(drift_result.get('baseline_accuracy'))}\n"
        f"Recent accuracy: {format_value(drift_result.get('recent_accuracy'))}\n"
        f"Accuracy drop: {format_value(drift_result.get('accuracy_drop'))}\n"
        f"Threshold: {format_value(drift_result.get('threshold'))}\n"
        f"\nRecommendation: Trigger hyperparameter tuning in next weekly training."
    )

    logger.warning(message)

    # Here you could add:
    # - Email notification
    # - Slack notification
    # - Database flag update

    return "Alert triggered"


def log_drift_status(**context):
    """Log drift check status to database."""
    import json
    from src.common import get_logger

    drift_result = context["ti"].xcom_pull(key="drift_result")
    logger = get_logger("drift_check")
    logger.info(f"Drift check completed: {json.dumps(drift_result, default=str)}")

    return "Status logged"


def generate_drift_report(**context):
    """Generate comprehensive drift report."""
    from src.common import get_config
    from src.data import DataCache
    from src.features import FeatureEngineer
    from src.evaluation.drift import DriftDetector

    config = get_config()
    cache = DataCache()
    detector = DriftDetector()

    # Get feature data for feature drift check
    engineer = FeatureEngineer()
    scaler_path = config.data_dir / "scaler.joblib"
    if scaler_path.exists():
        engineer.load_scaler(str(scaler_path))

    # Collect recent features
    all_features = []
    for symbol in config.symbols_list:
        df = cache.load(symbol)
        if df.empty:
            continue
        df = engineer.compute_features(df)
        feature_cols = engineer.get_feature_names()
        all_features.append(df[feature_cols])

    if not all_features:
        return "No feature data available for report"

    import pandas as pd
    features_df = pd.concat(all_features, ignore_index=True)

    # Split into baseline and current
    split_idx = int(len(features_df) * 0.7)
    baseline_features = features_df.iloc[:split_idx]
    current_features = features_df.iloc[split_idx:]

    # Generate report
    report = detector.get_drift_report(
        baseline_features=baseline_features,
        current_features=current_features,
    )

    context["ti"].xcom_push(key="drift_report", value=report)

    from src.common import get_logger
    logger = get_logger("drift_report")
    logger.info(f"Drift report generated: {report.get('recommended_action', 'unknown')}")

    return f"Report generated. Recommendation: {report.get('recommended_action', 'unknown')}"


with DAG(
    "drift_check",
    default_args=default_args,
    description="Daily performance drift check",
    schedule_interval="0 23 * * 1-5",  # Weekdays 18:00 EST (after daily prediction)
    catchup=False,
    tags=["stock", "ml", "monitoring"],
) as dag:

    check_drift_task = PythonOperator(
        task_id="check_performance_drift",
        python_callable=check_performance_drift,
    )

    decide_task = BranchPythonOperator(
        task_id="decide_action",
        python_callable=decide_action,
    )

    alert_task = PythonOperator(
        task_id="trigger_alert",
        python_callable=trigger_alert,
    )

    no_action_task = EmptyOperator(
        task_id="no_action",
    )

    log_task = PythonOperator(
        task_id="log_drift_status",
        python_callable=log_drift_status,
        trigger_rule="none_failed_min_one_success",
    )

    report_task = PythonOperator(
        task_id="generate_drift_report",
        python_callable=generate_drift_report,
    )

    check_drift_task >> decide_task
    decide_task >> [alert_task, no_action_task]
    [alert_task, no_action_task] >> log_task >> report_task
