# Airflow Dockerfile
FROM apache/airflow:2.7.3-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements and install
COPY pyproject.toml /tmp/
RUN pip install --no-cache-dir /tmp/

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"

# Copy source code will be done via volume mount
