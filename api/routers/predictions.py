"""Predictions API router."""

from datetime import datetime, date, time
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

from src.data import DatabaseManager
from src.pipeline.utils import calculate_accuracy
from api.dependencies import get_db

router = APIRouter()


class PredictionResponse(BaseModel):
    """Prediction response model."""

    id: int
    symbol: str
    date: datetime
    prediction: int
    probability: Optional[float]
    actual: Optional[int]
    model_name: str
    created_at: datetime


class PredictionListResponse(BaseModel):
    """List of predictions response."""

    predictions: list[dict]
    total: int


class FeatureResponse(BaseModel):
    """Feature values response."""

    symbol: str
    date: datetime
    features: dict


@router.get("", response_model=PredictionListResponse)
async def get_predictions(
    symbol: Optional[str] = Query(None, description="Filter by stock symbol"),
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records"),
    db: DatabaseManager = Depends(get_db),
):
    """Get recent predictions.

    Returns a list of predictions optionally filtered by symbol and date range.
    """
    start_dt = datetime.combine(start_date, time.min) if start_date else None
    end_dt = datetime.combine(end_date, time(23, 59, 59, 999999)) if end_date else None

    df = db.get_predictions(
        symbol=symbol,
        start_date=start_dt,
        end_date=end_dt,
        limit=limit,
    )

    predictions = df.to_dict(orient="records")

    return PredictionListResponse(
        predictions=predictions,
        total=len(predictions),
    )


@router.get("/latest")
async def get_latest_predictions(
    limit: int = Query(10, ge=1, le=100, description="Number of predictions"),
    db: DatabaseManager = Depends(get_db),
):
    """Get the most recent predictions."""
    df = db.get_predictions(limit=limit)

    return {
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
    }


@router.get("/by-date/{prediction_date}")
async def get_predictions_by_date(
    prediction_date: date,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    db: DatabaseManager = Depends(get_db),
):
    """Get predictions for a specific date."""
    start_dt = datetime.combine(prediction_date, time.min)
    end_dt = datetime.combine(prediction_date, time(23, 59, 59, 999999))

    df = db.get_predictions(
        symbol=symbol,
        start_date=start_dt,
        end_date=end_dt,
        limit=100,
    )

    if df.empty:
        return {
            "date": prediction_date.isoformat(),
            "predictions": [],
            "count": 0,
        }

    # Calculate accuracy if actuals are available
    accuracy = calculate_accuracy(df)

    return {
        "date": prediction_date.isoformat(),
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
        "accuracy": accuracy,
    }


@router.get("/symbol/{symbol}")
async def get_predictions_by_symbol(
    symbol: str,
    limit: int = Query(30, ge=1, le=200, description="Maximum number of records"),
    db: DatabaseManager = Depends(get_db),
):
    """Get predictions for a specific symbol."""
    df = db.get_predictions(symbol=symbol.upper(), limit=limit)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No predictions found for {symbol}")

    # Calculate overall accuracy
    accuracy = calculate_accuracy(df)

    return {
        "symbol": symbol.upper(),
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
        "accuracy": accuracy,
    }


@router.get("/features/{feature_date}")
async def get_features_by_date(
    feature_date: date,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
):
    """Get feature values for a specific date."""
    from src.common import get_config
    from src.data import DataCache
    from src.features import FeatureEngineer

    config = get_config()
    cache = DataCache()
    engineer = FeatureEngineer()

    symbols = [symbol.upper()] if symbol else config.symbols_list
    results = []

    for sym in symbols:
        df = cache.load(sym)
        if df.empty:
            continue

        df = engineer.compute_features(df)
        df["date"] = df["date"].dt.date

        date_row = df[df["date"] == feature_date]
        if date_row.empty:
            continue

        feature_cols = engineer.get_feature_names()
        features = date_row[feature_cols].iloc[0].to_dict()

        results.append({
            "symbol": sym,
            "date": feature_date.isoformat(),
            "features": features,
        })

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for date {feature_date}",
        )

    return {"data": results, "count": len(results)}
