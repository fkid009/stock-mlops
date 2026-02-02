"""SQLite database management for predictions and metrics."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.common import get_config, get_logger

logger = get_logger(__name__)
Base = declarative_base()


class Prediction(Base):
    """Prediction record."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=True)
    actual = Column(Integer, nullable=True)  # 0, 1, or NULL if unknown
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyMetric(Base):
    """Daily accuracy metrics."""

    __tablename__ = "daily_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    accuracy = Column(Float, nullable=False)
    total_predictions = Column(Integer, nullable=False)
    correct_predictions = Column(Integer, nullable=False)
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRecord(Base):
    """Model training record."""

    __tablename__ = "model_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # logistic_l1, lightgbm, svm
    mlflow_run_id = Column(String(100), nullable=True)
    accuracy = Column(Float, nullable=False)
    accuracy_std = Column(Float, nullable=True)
    score = Column(Float, nullable=False)  # Combined score
    is_best = Column(Integer, default=0)  # 1 if current best model
    trained_at = Column(DateTime, default=datetime.utcnow)
    hyperparameters = Column(String(1000), nullable=True)  # JSON string


class DatabaseManager:
    """Manage SQLite database operations."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection.

        Args:
            database_url: SQLite connection URL
        """
        config = get_config()
        self.database_url = database_url or config.database_url

        # Ensure data directory exists
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {self.database_url}")

    def save_prediction(
        self,
        symbol: str,
        date: datetime,
        prediction: int,
        model_name: str,
        probability: Optional[float] = None,
    ) -> None:
        """Save a prediction to database.

        Args:
            symbol: Stock ticker symbol
            date: Prediction date
            prediction: Predicted direction (0 or 1)
            model_name: Name of the model used
            probability: Prediction probability
        """
        with self.Session() as session:
            # Check for existing prediction on the same day
            date_start = datetime(date.year, date.month, date.day)
            date_end = datetime(date.year, date.month, date.day, 23, 59, 59)

            existing = (
                session.query(Prediction)
                .filter(
                    Prediction.symbol == symbol,
                    Prediction.date >= date_start,
                    Prediction.date <= date_end,
                )
                .first()
            )

            if existing:
                # Update existing prediction
                existing.prediction = prediction
                existing.probability = probability
                existing.model_name = model_name
                existing.date = date
                logger.debug(f"Updated prediction for {symbol} on {date.date()}")
            else:
                # Create new prediction
                record = Prediction(
                    symbol=symbol,
                    date=date,
                    prediction=prediction,
                    probability=probability,
                    model_name=model_name,
                )
                session.add(record)
                logger.debug(f"Saved prediction for {symbol} on {date.date()}")

            session.commit()

    def update_actual(self, symbol: str, date: datetime, actual: int) -> None:
        """Update the actual value for a prediction.

        Args:
            symbol: Stock ticker symbol
            date: Prediction date
            actual: Actual direction (0 or 1)
        """
        with self.Session() as session:
            session.execute(
                text(
                    "UPDATE predictions SET actual = :actual "
                    "WHERE symbol = :symbol AND date = :date"
                ),
                {"actual": actual, "symbol": symbol, "date": date},
            )
            session.commit()
            logger.debug(f"Updated actual for {symbol} on {date.date()}")

    def save_daily_metric(
        self,
        date: datetime,
        accuracy: float,
        total: int,
        correct: int,
        model_name: str,
    ) -> None:
        """Save daily accuracy metric.

        Args:
            date: Metric date
            accuracy: Accuracy score
            total: Total predictions
            correct: Correct predictions
            model_name: Model name
        """
        with self.Session() as session:
            # Update or insert
            existing = session.query(DailyMetric).filter_by(date=date).first()
            if existing:
                existing.accuracy = accuracy
                existing.total_predictions = total
                existing.correct_predictions = correct
                existing.model_name = model_name
            else:
                record = DailyMetric(
                    date=date,
                    accuracy=accuracy,
                    total_predictions=total,
                    correct_predictions=correct,
                    model_name=model_name,
                )
                session.add(record)
            session.commit()
            logger.info(f"Saved daily metric for {date.date()}: accuracy={accuracy:.4f}")

    def save_model_record(
        self,
        model_name: str,
        model_type: str,
        accuracy: float,
        score: float,
        accuracy_std: Optional[float] = None,
        mlflow_run_id: Optional[str] = None,
        hyperparameters: Optional[str] = None,
        is_best: bool = False,
    ) -> None:
        """Save model training record.

        Args:
            model_name: Full model name
            model_type: Model type (logistic_l1, lightgbm, svm)
            accuracy: Validation accuracy
            score: Combined selection score
            accuracy_std: Accuracy standard deviation
            mlflow_run_id: MLflow run ID
            hyperparameters: JSON string of hyperparameters
            is_best: Whether this is the current best model
        """
        with self.Session() as session:
            # If this is best, unset previous best
            if is_best:
                session.execute(text("UPDATE model_records SET is_best = 0"))

            record = ModelRecord(
                model_name=model_name,
                model_type=model_type,
                accuracy=accuracy,
                accuracy_std=accuracy_std,
                score=score,
                mlflow_run_id=mlflow_run_id,
                hyperparameters=hyperparameters,
                is_best=1 if is_best else 0,
            )
            session.add(record)
            session.commit()
            logger.info(f"Saved model record: {model_name}, score={score:.4f}")

    def get_predictions(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get predictions from database.

        Args:
            symbol: Filter by symbol
            start_date: Filter start date
            end_date: Filter end date
            limit: Maximum number of records

        Returns:
            DataFrame with predictions
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = {}

        if symbol:
            query += " AND symbol = :symbol"
            params["symbol"] = symbol
        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY date DESC LIMIT :limit"
        params["limit"] = limit

        return pd.read_sql(text(query), self.engine, params=params)

    def get_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get daily metrics from database.

        Args:
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            DataFrame with daily metrics
        """
        query = "SELECT * FROM daily_metrics WHERE 1=1"
        params = {}

        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY date DESC"

        return pd.read_sql(text(query), self.engine, params=params)

    def get_best_model(self) -> Optional[dict]:
        """Get the current best model record.

        Returns:
            Dict with model info or None
        """
        with self.Session() as session:
            record = session.query(ModelRecord).filter_by(is_best=1).first()
            if record:
                return {
                    "model_name": record.model_name,
                    "model_type": record.model_type,
                    "accuracy": record.accuracy,
                    "score": record.score,
                    "mlflow_run_id": record.mlflow_run_id,
                    "trained_at": record.trained_at,
                }
        return None

    def get_model_records(self, limit: int = 20) -> pd.DataFrame:
        """Get recent model training records.

        Args:
            limit: Maximum number of records

        Returns:
            DataFrame with model records
        """
        query = "SELECT * FROM model_records ORDER BY trained_at DESC LIMIT :limit"
        return pd.read_sql(text(query), self.engine, params={"limit": limit})
