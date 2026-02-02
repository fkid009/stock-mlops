# FastAPI Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly (not as package)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "scikit-learn>=1.3.0" \
    "lightgbm>=4.0.0" \
    "yfinance>=0.2.0" \
    "mlflow>=2.9.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "sqlalchemy>=2.0.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "pyyaml>=6.0.0" \
    "httpx>=0.25.0" \
    "joblib>=1.3.0" \
    "scipy>=1.11.0"

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# Set PYTHONPATH to find src module
ENV PYTHONPATH=/app

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
