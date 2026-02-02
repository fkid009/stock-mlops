# Airflow Dockerfile
FROM apache/airflow:2.8.1-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install only the ML dependencies (not the full package to avoid SQLAlchemy conflicts)
RUN pip install --no-cache-dir \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "scikit-learn>=1.3.0" \
    "lightgbm>=4.0.0" \
    "yfinance>=0.2.0" \
    "mlflow==2.9.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "pyyaml>=6.0.0" \
    "joblib>=1.3.0" \
    "scipy>=1.11.0"

# Source code will be mounted via volumes
# PYTHONPATH will be set via docker-compose environment
