# CropGuard AI — Plant Disease Detection for African Agriculture

**African Leadership University · Machine Learning Module · Summative Assignment**  
**Student:** Ajak Bul Zachariah Chol  
**GitHub:** [SLICKMAN-TYRUS/cropguard-ai](https://github.com/SLICKMAN-TYRUS/cropguard-ai)

---

## Video Demo

[![YouTube Demo](https://img.shields.io/badge/▶%20Watch%20Demo-YouTube-red?logo=youtube)](https://youtu.be/biDDqvrF2F8)
&nbsp;&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SLICKMAN-TYRUS/cropguard-ai/blob/main/notebook/crop_disease_detector.ipynb)

> Replace `PENDING` in the YouTube badge URL with your video ID after recording.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Cloud Deployment](#cloud-deployment)
- [API Reference](#api-reference)
- [Retraining Pipeline](#retraining-pipeline)
- [Load Test Results](#load-test-results)
- [Notebook](#notebook)

---

## Overview

CropGuard AI is an end-to-end MLOps pipeline that classifies tomato leaf images
into three categories using a fine-tuned MobileNetV2 deep learning model:

| Class | Pathogen | Severity |
|-------|----------|----------|
| Early Blight | *Alternaria solani* | Medium |
| Late Blight | *Phytophthora infestans* | High |
| Healthy | — | None |

Crop disease is responsible for an estimated $220 billion in annual agricultural
losses globally. This project enables smallholder farmers and field agents in
Sub-Saharan Africa to get a real-time plant health diagnosis from a single leaf
photo — no specialist knowledge required.

### Dataset

| Property | Value |
|----------|-------|
| Source | PlantVillage (Kaggle) |
| Images used | ~1,800 (600 per class) |
| Input resolution | 224 × 224 × 3 RGB |
| Split | 70% train / 15% val / 15% test |

### Model Architecture

```
Input (224 × 224 × 3)
  └─ MobileNetV2 backbone — ImageNet weights
       Phase 1: backbone frozen, head trained
       Phase 2: top 30% of backbone unfrozen, fine-tuned at lr=1e-5
  └─ GlobalAveragePooling2D
  └─ Dense(256, ReLU) → BatchNorm → Dropout(0.4)
  └─ Dense(128, ReLU) → Dropout(0.3)
  └─ Dense(3, Softmax)
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | TensorFlow 2.18, Keras, MobileNetV2 |
| API | FastAPI, Uvicorn, SQLite |
| UI | Streamlit, Plotly |
| Containers | Docker, Docker Compose |
| Load testing | Locust 2.46 |
| Cloud | Google Cloud Run |
| Training environment | Google Colab (T4 GPU) |

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.0% |
| Macro AUC-ROC | 0.9913 |
| Early Blight F1 | 0.917 |
| Late Blight F1 | 0.897 |
| Healthy F1 | 0.977 |
| Avg inference latency | ~115 ms (CPU) |
| Model file size | ~14 MB (.h5) |

**Optimization techniques applied:**

- Pre-trained base — MobileNetV2 ImageNet weights
- Regularization — Dropout (0.4, 0.3) and BatchNormalization
- Optimizer — Adam, lr=1e-3 (Phase 1) → lr=1e-5 (Phase 2)
- Early stopping — patience=5, restore_best_weights=True
- Learning rate decay — ReduceLROnPlateau, factor=0.5, patience=3
- Data augmentation — random flip, brightness, contrast, saturation, hue jitter
- Two-phase training — frozen backbone first, then backbone fine-tuning

---

## Repository Structure

```
cropguard-ai/
├── notebook/
│   └── crop_disease_detector.ipynb    # 40-cell training notebook
│
├── src/
│   ├── preprocessing.py               # tf.data pipeline and augmentation
│   ├── model.py                       # MobileNetV2 architecture, training,
│   │                                  # retrain_from_existing()
│   ├── prediction.py                  # Singleton model cache, inference
│   └── database.py                    # SQLite: uploads, sessions, predictions
│
├── api/
│   ├── main.py                        # FastAPI — 10 endpoints
│   └── requirements.txt
│
├── ui/
│   ├── app.py                         # Streamlit dashboard — 5 tabs
│   └── requirements.txt
│
├── locust/
│   └── locustfile.py                  # Load testing — 3 user profiles
│
├── models/
│   ├── crop_disease_model.h5          # Trained model weights
│   ├── training_history.json          # Per-epoch accuracy and loss
│   └── eval_metrics.json             # F1, AUC-ROC, confusion matrix
│
├── results/
│   ├── LOAD_TEST_RESULTS.md
│   └── locust_results_100users_stats.csv
│
├── data/
│   ├── train/                         # Training images (by class subfolder)
│   ├── test/                          # Test images
│   └── retrain/                       # Uploaded images queued for retraining
│
├── Dockerfile                         # API container
├── Dockerfile.ui                      # Streamlit container
├── docker-compose.yml                 # Multi-service orchestration
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- Git
- 4 GB RAM minimum (8 GB recommended for training)
- Docker and Docker Compose (for containerised deployment)

---

### Option A — Local (without Docker)

```bash
# Clone
git clone https://github.com/SLICKMAN-TYRUS/cropguard-ai.git
cd cropguard-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model via Colab (recommended — free GPU)
# Download crop_disease_model.h5 from Colab and place it in models/

# Start the API (Terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start the UI (Terminal 2)
streamlit run ui/app.py
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

---

### Option B — Docker Compose

```bash
git clone https://github.com/SLICKMAN-TYRUS/cropguard-ai.git
cd cropguard-ai

# Place models/crop_disease_model.h5 before starting

# Start API + UI
docker compose up --build

# Scale API for load testing
docker compose up --scale api=3 --build
```

- API: http://localhost:8000
- UI: http://localhost:8501

---

### Option C — Load Testing

```bash
# Start Locust web UI (stack must be running)
docker compose --profile loadtest up

# Open http://localhost:8089
# Set host = http://localhost:8000, users = 100, spawn rate = 10

# Or headless
locust -f locust/locustfile.py \
  --host=http://localhost:8000 \
  --users=100 --spawn-rate=10 \
  --run-time=60s --headless \
  --csv=results/my_test
```

---

## Cloud Deployment

### Google Cloud Run

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cropguard-api .

# Deploy API
gcloud run deploy cropguard-api \
  --image gcr.io/YOUR_PROJECT_ID/cropguard-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 5 \
  --allow-unauthenticated

# Deploy UI via Streamlit Cloud
# Connect repo at https://share.streamlit.io
# Set secret: API_URL = https://your-api-url.run.app
```

---

## API Reference

Base URL: `http://localhost:8000` (local) or your Cloud Run URL

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Uptime, model status, prediction count |
| GET | `/model-info` | Architecture, accuracy, epochs trained |
| GET | `/training-history` | Per-epoch accuracy and loss |
| GET | `/metrics` | Precision, recall, F1, AUC-ROC, confusion matrix |
| GET | `/dataset-stats` | Image counts per split, SQLite upload records |
| GET | `/retrain-status` | Live retraining progress |
| GET | `/retrain-sessions` | Full history of retrain sessions |
| POST | `/predict` | Single image → class, confidence, disease info |
| POST | `/batch-predict` | Multiple images → list of predictions |
| POST | `/upload` | Upload images to disk and SQLite |
| POST | `/retrain` | Trigger background fine-tuning |

**Predict a leaf image:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@tomato_leaf.jpg"
```

**Response:**

```json
{
  "class": "Tomato___Early_blight",
  "confidence": 0.9173,
  "latency_ms": 112.4,
  "probabilities": {
    "Tomato___Early_blight": 0.9173,
    "Tomato___Late_blight": 0.0614,
    "Tomato___healthy": 0.0213
  },
  "disease_info": {
    "description": "Caused by Alternaria solani. Brown spots with concentric rings.",
    "treatment": "Apply copper-based fungicides. Remove infected leaves.",
    "severity": "Medium"
  }
}
```

**Upload images for retraining:**

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@leaf1.jpg" \
  -F "files=@leaf2.jpg" \
  -F "class_name=Tomato___Late_blight"
```

**Trigger retraining:**

```bash
curl -X POST "http://localhost:8000/retrain?epochs=10&triggered_by=curl"
```

---

## Retraining Pipeline

The system supports continuous learning. Upload new images from the UI or API
and retrain without any downtime.

**Step 1 — Upload**  
Images are saved to `data/retrain/<class_name>/` and each upload is recorded
in SQLite with filename, class, file size, and timestamp.

**Step 2 — Preprocessing**  
The uploaded images are resized to 224×224, normalised to [0, 1], and augmented
with random flip, brightness, contrast, and hue jitter.

**Step 3 — Fine-tune**  
The existing `crop_disease_model.h5` is loaded as the pre-trained base. The top
layers are unfrozen and fine-tuned on the new data. After training, the model is
saved and hot-reloaded — all subsequent `/predict` calls immediately use the
updated weights without a server restart.

Retrain session history (start time, end time, val_accuracy, session ID) is
stored in SQLite and visible in the dashboard under Retrain Session History.

---

## Load Test Results

Tested with Locust 2.46 · 3 user profiles · synthetic 224×224 JPEG payloads

| Containers | Users | RPS | p95 `/predict` | Fail Rate |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 50 | 65.7 | 310 ms | 0.00% |
| 1 | 100 | 215.7 | 265 ms | 0.10% |
| 2 | 100 | 218.5 | 195 ms | 0.02% |
| 3 | 200 | 435.7 | 210 ms | 0.03% |

Throughput scales near-linearly with container count. Two containers reduce p95
latency by 35% compared to one at the same load. Three containers sustain 435
requests per second at 200 concurrent users with a 0.03% failure rate.

Full results: [`results/LOAD_TEST_RESULTS.md`](results/LOAD_TEST_RESULTS.md)

---

## Notebook

`notebook/crop_disease_detector.ipynb` — 40 cells, self-contained, runnable on
Google Colab with a free T4 GPU in approximately 8 minutes.

| Section | Content |
|---------|---------|
| 1–2 | Environment setup, constants, class configuration |
| 3 | Data acquisition from Kaggle, class filtering |
| 4 | EDA — class distribution, sample images, RGB analysis, texture variance |
| 5 | Preprocessing — train/val/test split, tf.data pipeline, augmentation |
| 6 | Model architecture — MobileNetV2 + custom head |
| 7–8 | Two-phase training — Phase 1 (frozen) and Phase 2 (fine-tuned) |
| 9 | Evaluation — Accuracy, Loss, Precision, Recall, F1, AUC-ROC, Confusion Matrix |
| 10–11 | Save artefacts, prediction demo on test images |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SLICKMAN-TYRUS/cropguard-ai/blob/main/notebook/crop_disease_detector.ipynb)

---

## License

MIT © 2025 Ajak Bul Zacharia Chol — African Leadership University