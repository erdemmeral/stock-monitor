FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and Git LFS
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    curl \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and pull LFS objects
COPY . /app
RUN git lfs pull

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "-m", "app.monitoring.real_time_monitor"]