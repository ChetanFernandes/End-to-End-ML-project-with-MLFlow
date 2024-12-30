
FROM python:3.10-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the Git repository
RUN git clone https://github.com/ChetanFernandes/End-to-End-ML-project-with-MLFlow /app

# Copy dependencies first (for caching layers)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with S3 support
RUN pip install dvc[s3]  # Modify `[s3]` to match your remote type, if needed.


# Copy the rest of the application code
COPY . .

# Expose the application port (uncomment if needed)
#EXPOSE 8000

# Start the application
CMD ["gunicorn", "-w", "6","--timeout", "900", "--bind", "0.0.0.0:8000", "app:app"]

