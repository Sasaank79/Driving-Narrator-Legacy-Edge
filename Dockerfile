# Driving Narrator - Docker Image
# For reproducible inference environment

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: run inference on a video
# Usage: docker run -v /path/to/video:/app/input.mp4 driving-narrator
ENTRYPOINT ["python", "scripts/deploy.py"]
CMD ["--video", "input.mp4", "--model", "models/best_int8_openvino_model/"]
