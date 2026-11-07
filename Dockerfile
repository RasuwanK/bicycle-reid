FROM python:3.11-slim

LABEL maintainer="Bicycle Re-ID Team" \
      description="Cross-Camera Bicycle & Motorcycle Re-Identification System"

# Configure environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    TORCH_HOME=/home/appuser/.cache/torch \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics

# Install system dependencies for OpenCV and video decoding/processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user and group (UID 1000 matching standard host user)
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Upgrade pip and install Python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        ultralytics \
        opencv-python \
        numpy \
        torch \
        torchvision \
        pyyaml \
        streamlit \
        pandas \
        pytest

# Copy application source code, configs, demo data, and entrypoint
COPY configs /app/configs
COPY src /app/src
COPY tests /app/tests
COPY yolov8n.pt /app/yolov8n.pt
COPY README.md /app/README.md
COPY .streamlit /app/.streamlit
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Ensure script is executable and install package in editable mode
RUN chmod +x /app/docker-entrypoint.sh && \
    pip install --no-cache-dir -e .

# Pre-populate model caches (torchvision ResNet50 & MobileNetV3) for offline operation
RUN mkdir -p /home/appuser/.cache/torch/hub/checkpoints \
             /home/appuser/.config/Ultralytics \
             /app/data && \
    python3 -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.DEFAULT); m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)" && \
    chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose Streamlit web application port
EXPOSE 8501

# Health check to ensure web server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]

CMD ["streamlit", "run", "src/bicycle_reid/app/streamlit_app.py"]
