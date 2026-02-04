"""Metrics API router."""

from datetime import datetime, date, time, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

from src.data import DatabaseManager
from src.pipeline.utils import calculate_accuracy
from api.dependencies import get_db

router = APIRouter()


class DailyMetricResponse(BaseModel):
    """Daily metric response."""

    date: datetime
    accuracy: float
    total_predictions: int
    correct_predictions: int
    model_name: str


class MetricsListResponse(BaseModel):
    """List of metrics response."""

    metrics: list[dict]
    total: int
    avg_accuracy: Optional[float]


class DriftStatusResponse(BaseModel):
    """Drift status response."""

    drift_detected: bool
    baseline_accuracy: Optional[float]
    recent_accuracy: Optional[float]
    accuracy_drop: Optional[float]
    threshold: float
    recommendation: str


@router.get("", response_model=MetricsListResponse)
async def get_metrics(
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    days: int = Query(30, ge=1, le=365, description="Number of days if no date range"),
    db: DatabaseManager = Depends(get_db),
):
    """Get daily accuracy metrics.

    Returns daily accuracy metrics for the specified date range.
    """
    if start_date and end_date:
        start_dt = datetime.combine(start_date, time.min)
        end_dt = datetime.combine(end_date, time(23, 59, 59, 999999))
    else:
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        start_dt = end_dt - timedelta(days=days)

    df = db.get_metrics(start_date=start_dt, end_date=end_dt)

    # Convert to JSON-serializable format
    metrics = []
    for _, row in df.iterrows():
        metrics.append({
            "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
            "accuracy": float(row["accuracy"]),
            "total_predictions": int(row["total_predictions"]),
            "correct_predictions": int(row["correct_predictions"]),
            "model_name": str(row["model_name"]),
        })

    avg_accuracy = float(df["accuracy"].mean()) if not df.empty else None

    return MetricsListResponse(
        metrics=metrics,
        total=len(metrics),
        avg_accuracy=avg_accuracy,
    )


@router.get("/summary")
async def get_metrics_summary(
    days: int = Query(30, ge=7, le=365, description="Number of days for summary"),
    db: DatabaseManager = Depends(get_db),
):
    """Get metrics summary statistics."""
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=days)

    df = db.get_metrics(start_date=start_dt, end_date=end_dt)

    if df.empty:
        return {
            "period_days": days,
            "data_available": False,
            "message": "No metrics data available for the specified period",
        }

    # Calculate summary statistics (convert numpy types to Python types)
    total_preds = int(df["total_predictions"].sum())
    correct_preds = int(df["correct_predictions"].sum())

    return {
        "period_days": days,
        "data_available": True,
        "total_days_with_data": len(df),
        "accuracy": {
            "mean": float(df["accuracy"].mean()),
            "std": float(df["accuracy"].std()) if len(df) > 1 else 0.0,
            "min": float(df["accuracy"].min()),
            "max": float(df["accuracy"].max()),
            "median": float(df["accuracy"].median()),
        },
        "predictions": {
            "total": total_preds,
            "correct": correct_preds,
            "daily_avg": float(df["total_predictions"].mean()),
        },
        "overall_accuracy": correct_preds / total_preds if total_preds > 0 else None,
    }


@router.get("/trend")
async def get_accuracy_trend(
    days: int = Query(30, ge=7, le=90, description="Number of days"),
    window: int = Query(5, ge=3, le=20, description="Moving average window"),
    db: DatabaseManager = Depends(get_db),
):
    """Get accuracy trend with moving average."""
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=days + window)

    df = db.get_metrics(start_date=start_dt, end_date=end_dt)

    if df.empty:
        return {
            "trend": [],
            "message": "No metrics data available",
        }

    df = df.sort_values("date")
    df["accuracy_ma"] = df["accuracy"].rolling(window=window, min_periods=1).mean()

    # Keep only the requested days
    df = df.tail(days)

    trend = []
    for _, row in df.iterrows():
        trend.append({
            "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
            "accuracy": float(row["accuracy"]),
            "accuracy_ma": float(row["accuracy_ma"]),
        })

    return {
        "trend": trend,
        "window": window,
        "days": days,
    }


@router.get("/drift", response_model=DriftStatusResponse)
async def get_drift_status():
    """Get current drift detection status."""
    from src.evaluation.drift import DriftDetector

    detector = DriftDetector()
    result = detector.check_performance_drift()

    # Determine recommendation
    if result.get("drift_detected"):
        recommendation = "Consider triggering model retraining with hyperparameter tuning"
    elif result.get("reason") == "insufficient_data":
        recommendation = "Collect more data before drift detection is possible"
    else:
        recommendation = "No action needed"

    return DriftStatusResponse(
        drift_detected=result.get("drift_detected", False),
        baseline_accuracy=result.get("baseline_accuracy"),
        recent_accuracy=result.get("recent_accuracy"),
        accuracy_drop=result.get("accuracy_drop"),
        threshold=result.get("threshold", 0.05),
        recommendation=recommendation,
    )


@router.get("/by-symbol/{symbol}")
async def get_metrics_by_symbol(
    symbol: str,
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    db: DatabaseManager = Depends(get_db),
):
    """Get accuracy metrics for a specific symbol."""

    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=days)

    # Get predictions for this symbol
    predictions_df = db.get_predictions(
        symbol=symbol.upper(),
        start_date=start_dt,
        end_date=end_dt,
        limit=500,
    )

    if predictions_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for symbol: {symbol}",
        )

    # Filter predictions with actual values
    valid = predictions_df[predictions_df["actual"].notna()]

    if valid.empty:
        return {
            "symbol": symbol.upper(),
            "period_days": days,
            "total_predictions": len(predictions_df),
            "evaluated_predictions": 0,
            "accuracy": None,
            "message": "No predictions with actual values yet",
        }

    accuracy = calculate_accuracy(valid)
    correct = (valid["prediction"] == valid["actual"]).sum()

    return {
        "symbol": symbol.upper(),
        "period_days": days,
        "total_predictions": len(predictions_df),
        "evaluated_predictions": len(valid),
        "correct_predictions": int(correct),
        "accuracy": accuracy,
    }
