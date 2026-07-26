# ── CropGuard AI — FastAPI Backend ─────────────────────────────────────────
FROM python:3.10-slim

# System deps (OpenCV headless needs libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/  ./src/
COPY api/  ./api/
COPY models/ ./models/

# Ensure data dirs exist
RUN mkdir -p data/train data/test data/retrain models

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
