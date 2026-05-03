# Dockerfile for ML Engine
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory and model
COPY . .

# Container for ML processing doesn't necessarily need a long-running entrypoint 
# unless it's a worker. For now, we keep it as a base image.
CMD ["python", "src/trainer.py"]
