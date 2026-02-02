# Airflow Dockerfile
FROM apache/airflow:2.8.1-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements and install
COPY pyproject.toml README.md /tmp/
COPY src/ /tmp/src/
RUN pip install --no-cache-dir -e /tmp/

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow:/tmp"

# Copy source code will be done via volume mount
