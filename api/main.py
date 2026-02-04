"""FastAPI application for stock prediction API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common import get_config, get_logger
from src.data import DatabaseManager
from api.routers import predictions_router, models_router, metrics_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Stock Prediction API...")
    config = get_config()

    # Initialize database
    db = DatabaseManager()
    app.state.db = db
    app.state.config = config

    logger.info(f"API running in {config.app_env} mode")
    yield

    # Shutdown
    logger.info("Shutting down Stock Prediction API...")


app = FastAPI(
    title="Stock Prediction API",
    description="API for stock movement predictions and model management",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
allowed_origins = [
    "http://localhost:3000",
    "https://stock.lhb99.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include routers
app.include_router(predictions_router, prefix="/api/predictions", tags=["predictions"])
app.include_router(models_router, prefix="/api/models", tags=["models"])
app.include_router(metrics_router, prefix="/api/metrics", tags=["metrics"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Stock Prediction API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
    )
