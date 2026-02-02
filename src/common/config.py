"""Configuration management for the stock prediction pipeline."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Application
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)

    # Database
    database_url: str = Field(default="sqlite:///./data/stock.db")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="stock-prediction")

    # Stock symbols
    stock_symbols: str = Field(default="AAPL,GOOGL,MSFT")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def symbols_list(self) -> list[str]:
        """Return stock symbols as a list."""
        return [s.strip() for s in self.stock_symbols.split(",")]

    @property
    def configs_dir(self) -> Path:
        """Return configs directory path."""
        return self.project_root / "configs"

    @property
    def data_dir(self) -> Path:
        """Return data directory path."""
        data_path = self.project_root / "data"
        data_path.mkdir(exist_ok=True)
        return data_path


def load_yaml_config(config_name: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_name: Name of the config file (without .yaml extension)

    Returns:
        Dictionary containing the configuration
    """
    config = get_config()
    config_path = config.configs_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_config() -> Config:
    """Get cached configuration instance."""
    return Config()


def get_features_config() -> dict[str, Any]:
    """Load features configuration."""
    return load_yaml_config("features")


def get_models_config() -> dict[str, Any]:
    """Load models configuration."""
    return load_yaml_config("models")


def get_pipeline_config() -> dict[str, Any]:
    """Load pipeline configuration."""
    return load_yaml_config("pipeline")
