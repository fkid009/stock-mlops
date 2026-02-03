"""Models API router."""

from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

from src.data import DatabaseManager
from api.dependencies import get_db

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information response."""

    model_name: str
    model_type: str
    accuracy: float
    score: float
    is_best: bool
    trained_at: str
    mlflow_run_id: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of models response."""

    models: list[dict]
    total: int
    best_model: Optional[dict]


@router.get("", response_model=ModelListResponse)
async def get_models(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of records"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    db: DatabaseManager = Depends(get_db),
):
    """Get list of trained models.

    Returns recent model training records, optionally filtered by type.
    """
    df = db.get_model_records(limit=limit)

    if model_type:
        df = df[df["model_type"] == model_type]

    models = df.to_dict(orient="records")
    best_model = db.get_best_model()

    return ModelListResponse(
        models=models,
        total=len(models),
        best_model=best_model,
    )


@router.get("/best")
async def get_best_model(db: DatabaseManager = Depends(get_db)):
    """Get the current best model."""
    best = db.get_best_model()

    if not best:
        raise HTTPException(status_code=404, detail="No best model found")

    return best


@router.get("/compare")
async def compare_models(
    model_types: str = Query(
        "logistic_l1,lightgbm,svm",
        description="Comma-separated model types to compare",
    ),
    db: DatabaseManager = Depends(get_db),
):
    """Compare performance across model types."""

    df = db.get_model_records(limit=100)

    if df.empty:
        return {"comparison": [], "message": "No models found"}

    types = [t.strip() for t in model_types.split(",")]
    df = df[df["model_type"].isin(types)]

    # Aggregate by model type
    comparison = []
    for model_type in types:
        type_df = df[df["model_type"] == model_type]
        if type_df.empty:
            continue

        comparison.append({
            "model_type": model_type,
            "avg_accuracy": type_df["accuracy"].mean(),
            "max_accuracy": type_df["accuracy"].max(),
            "min_accuracy": type_df["accuracy"].min(),
            "avg_score": type_df["score"].mean(),
            "training_count": len(type_df),
            "times_selected_best": type_df["is_best"].sum(),
        })

    return {
        "comparison": comparison,
        "model_types": types,
    }


@router.get("/history/{model_type}")
async def get_model_history(
    model_type: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum number of records"),
    db: DatabaseManager = Depends(get_db),
):
    """Get training history for a specific model type."""
    df = db.get_model_records(limit=limit)
    df = df[df["model_type"] == model_type]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for model type: {model_type}",
        )

    return {
        "model_type": model_type,
        "history": df.to_dict(orient="records"),
        "count": len(df),
    }


@router.get("/mlflow")
async def get_mlflow_runs(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of runs"),
):
    """Get recent MLflow runs."""
    try:
        from src.models import ModelRegistry

        registry = ModelRegistry()
        runs = registry.get_recent_runs(limit=limit)

        return {
            "runs": runs,
            "count": len(runs),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to MLflow: {str(e)}",
        )


@router.get("/mlflow/best")
async def get_mlflow_best_run():
    """Get the best run from MLflow."""
    try:
        from src.models import ModelRegistry

        registry = ModelRegistry()
        best_run = registry.get_best_run()

        if not best_run:
            raise HTTPException(status_code=404, detail="No best run found in MLflow")

        return best_run
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to MLflow: {str(e)}",
        )
