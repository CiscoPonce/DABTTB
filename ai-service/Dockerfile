# TTBall_4 AI Service Dockerfile
# Multi-stage build for AI/ML workloads with CUDA support

# Stage 1: Base Python image with system dependencies
FROM python:3.10-slim AS base

# Set environment variables for optimal AI/ML performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=random \
    # GPU optimization (will work when CUDA is available)
    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_CACHE_DISABLE=0 \
    # Memory optimization
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base AS python-deps

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Application
FROM base AS app

# Copy virtual environment from previous stage
COPY --from=python-deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directories
RUN mkdir -p /app/models /app/uploads /app/results /app/cache /app/logs \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Install PyTorch with CUDA support first (this is key for CUDA compatibility)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create requirements.txt for remaining dependencies
RUN echo "# TTBall_4 AI Service Dependencies" > requirements.txt && \
    echo "fastapi==0.115.14" >> requirements.txt && \
    echo "uvicorn==0.34.0" >> requirements.txt && \
    echo "pydantic==2.11.7" >> requirements.txt && \
    echo "pydantic-settings==2.7.0" >> requirements.txt && \
    echo "ultralytics==8.3.160" >> requirements.txt && \
    echo "opencv-python==4.11.0.86" >> requirements.txt && \
    echo "numpy==1.26.4" >> requirements.txt && \
    echo "Pillow==11.0.0" >> requirements.txt && \
    echo "psutil==7.0.0" >> requirements.txt && \
    echo "transformers==4.47.1" >> requirements.txt && \
    echo "accelerate==1.2.1" >> requirements.txt && \
    echo "bitsandbytes==0.45.0" >> requirements.txt && \
    echo "aiofiles==24.1.0" >> requirements.txt && \
    echo "python-multipart==0.0.20" >> requirements.txt && \
    echo "websockets==14.1" >> requirements.txt && \
    echo "duckdb==1.3.1" >> requirements.txt && \
    echo "requests==2.32.3" >> requirements.txt && \
    echo "httpx==0.28.1" >> requirements.txt

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest pytest-asyncio

# Create startup script
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo 'echo "Starting TTBall_4 AI Service..."' >> /app/start.sh && \
    echo 'echo "Python version: $(python --version)"' >> /app/start.sh && \
    echo 'echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"' >> /app/start.sh && \
    echo 'echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"' >> /app/start.sh && \
    echo 'echo "GPU count: $(python -c "import torch; print(torch.cuda.device_count())")"' >> /app/start.sh && \
    echo 'echo "Starting FastAPI server..."' >> /app/start.sh && \
    echo 'exec uvicorn simple_main:app --host 0.0.0.0 --port 8005 --workers 1' >> /app/start.sh && \
    chmod +x /app/start.sh

# Set environment variables for Python path and application configuration
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

# Switch to app user
USER appuser

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Set default command
CMD ["/app/start.sh"] 