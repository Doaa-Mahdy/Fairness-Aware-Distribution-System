FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models database in plots

# Ensure data and models are present (will be overridden by volume mounts in production)
# For local testing: COPY models/ models/
# For local testing: COPY charity_recipients_dataset.csv .

# Default command: run serverless handler
CMD ["python", "handler.py"]
