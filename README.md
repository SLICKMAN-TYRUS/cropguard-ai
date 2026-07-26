# 🌿 CropGuard AI — Plant Disease Detection for African Agriculture

> **End-to-End ML Pipeline | African Leadership University**  
> Summative Assignment — Machine Learning Module  
> **Student:** Ajak Bul Zacharia Chol

---

## 📹 Video Demo

[![CropGuard AI Demo](https://img.shields.io/badge/YouTube-Demo%20Video-red?logo=youtube)](https://www.youtube.com/watch?v=PENDING)

> 📌 Replace `PENDING` with your actual YouTube video ID after recording.

---

## 🌐 Live URLs

| Service       | URL                                              |
|---------------|--------------------------------------------------|
| FastAPI Docs  | `https://cropguard-api-XXXX.run.app/docs`        |
| Streamlit UI  | `https://cropguard-ui.streamlit.app`             |
| Health Check  | `https://cropguard-api-XXXX.run.app/health`      |

> 📌 Replace URLs above after deploying to your cloud platform.

---

## 📖 Project Description

CropGuard AI is an end-to-end machine learning pipeline that classifies tomato plant
leaf images into three categories — **Early Blight**, **Late Blight**, and **Healthy** —
using a fine-tuned **MobileNetV2** deep learning model.

The project directly addresses food insecurity in Sub-Saharan Africa by enabling
smallholder farmers and field agents to get real-time plant health diagnostics from a
simple photo, without requiring specialist knowledge.

### Dataset
- **Source**: [PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) via TensorFlow Datasets
- **Subset**: 3 tomato classes (~3,500 images after filtering)
- **Input**: RGB leaf images resized to 224 × 224 px

### Model Architecture
```
Input (224×224×3)
  └─ MobileNetV2 backbone (ImageNet weights, frozen → fine-tuned)
       └─ GlobalAveragePooling2D
            └─ Dense(256, ReLU) → BatchNorm → Dropout(0.4)
                 └─ Dense(128, ReLU) → Dropout(0.3)
                      └─ Dense(3, Softmax)
```

### Key Results

| Metric          | Value  |
|-----------------|--------|
| Test Accuracy   | ~94 %  |
| Macro AUC-ROC   | ~0.98  |
| Model size      | ~14 MB |
| Avg latency     | ~115 ms|

---

## 🗂️ Repository Structure

```
cropguard-ai/
├── README.md
├── requirements.txt
├── Dockerfile               ← API container
├── Dockerfile.ui            ← UI container
├── docker-compose.yml       ← Multi-container orchestration
│
├── notebook/
│   └── crop_disease_detector.ipynb   ← Full training notebook (38 cells)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     ← Data pipeline & augmentation
│   ├── model.py             ← MobileNetV2 architecture & training
│   └── prediction.py        ← Inference with singleton model cache
│
├── api/
│   ├── main.py              ← FastAPI backend (10 endpoints)
│   └── requirements.txt
│
├── ui/
│   ├── app.py               ← Streamlit dashboard (5 tabs)
│   └── requirements.txt
│
├── locust/
│   └── locustfile.py        ← Load testing (3 user profiles)
│
├── data/
│   ├── train/               ← Training images (by class subfolder)
│   └── test/                ← Test images (by class subfolder)
│
├── models/
│   ├── crop_disease_model.h5
│   ├── training_history.json
│   └── eval_metrics.json
│
├── scripts/
│   └── generate_notebook.py ← Notebook generator script
│
└── results/
    ├── LOAD_TEST_RESULTS.md
    └── locust_results_100users_stats.csv
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for containerised deployment)
- 4 GB RAM minimum (8 GB recommended for training)

---

### Option A — Run Locally (without Docker)

```bash
# 1. Clone the repository
git clone https://github.com/SLICKMAN-TYRUS/cropguard-ai.git
cd cropguard-ai

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (runs the notebook or directly)
#    Option 1: Jupyter notebook
jupyter notebook notebook/crop_disease_detector.ipynb

#    Option 2: Run training script directly (coming soon)
#    python scripts/train.py

# 5. Start the FastAPI backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start the Streamlit UI (new terminal)
streamlit run ui/app.py
```

Open:
- API docs: http://localhost:8000/docs
- UI dashboard: http://localhost:8501

---

### Option B — Docker Compose (Recommended)

```bash
# 1. Clone and enter repo
git clone https://github.com/SLICKMAN-TYRUS/cropguard-ai.git
cd cropguard-ai

# 2. Place your trained model in models/
#    (or train it via the notebook first)

# 3. Build and start all services
docker compose up --build

# 4. Scale API to multiple replicas (for load testing)
docker compose up --scale api=3 --build
```

Services:
- API:  http://localhost:8000
- UI:   http://localhost:8501
- Docs: http://localhost:8000/docs

---

### Option C — Load Testing with Locust

```bash
# With Docker (recommended)
docker compose --profile loadtest up --build

# Open Locust UI → http://localhost:8089
# Set: Host = http://localhost:8000, Users = 100, Spawn rate = 10

# Or headless
locust -f locust/locustfile.py \
  --host=http://localhost:8000 \
  --users=100 --spawn-rate=10 \
  --run-time=60s --headless \
  --csv=results/my_test
```

---

## ☁️ Cloud Deployment (Google Cloud Run)

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push API image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cropguard-api .

# 3. Deploy to Cloud Run
gcloud run deploy cropguard-api \
  --image gcr.io/YOUR_PROJECT_ID/cropguard-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 5 \
  --allow-unauthenticated

# 4. Deploy UI (Streamlit Cloud or second Cloud Run service)
#    Set environment variable: API_URL=https://YOUR_API_URL.run.app
```

---

## 🔌 API Reference

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| GET    | `/health`          | Server uptime, model status, request count|
| GET    | `/model-info`      | Architecture, accuracy, classes          |
| GET    | `/training-history`| Epoch-level accuracy and loss curves     |
| GET    | `/metrics`         | Precision, recall, F1, AUC-ROC, CM      |
| GET    | `/dataset-stats`   | Image counts per class per split         |
| GET    | `/retrain-status`  | Live retraining progress                 |
| POST   | `/predict`         | Single image → class + confidence        |
| POST   | `/batch-predict`   | Multiple images → list of predictions    |
| POST   | `/upload`          | Bulk upload images for retraining        |
| POST   | `/retrain`         | Trigger background retraining            |

**Quick test:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/leaf.jpg"
```

---

## 📊 Load Test Results Summary

| Containers | Users | RPS    | p95 /predict | Fail Rate |
|------------|-------|--------|--------------|-----------|
| 1          | 50    | 65.7   | 310 ms       | 0.0%      |
| 1          | 100   | 215.7  | 265 ms       | 0.1%      |
| 2          | 100   | 218.5  | 195 ms       | 0.02%     |
| 3          | 200   | 435.7  | 210 ms       | 0.03%     |

Full results: [`results/LOAD_TEST_RESULTS.md`](results/LOAD_TEST_RESULTS.md)

---

## 🔄 Retraining Workflow

```
Upload new images (UI / POST /upload)
        ↓
Images saved to data/retrain/<class_name>/
        ↓
Press "Trigger Retraining" (UI / POST /retrain)
        ↓
Background thread: build_generators → train_model → save_model → evaluate_model
        ↓
Poll GET /retrain-status for progress
        ↓
Model hot-reloaded → all subsequent /predict calls use new weights
```

---

## 🧪 Running the Notebook

The notebook (`notebook/crop_disease_detector.ipynb`) is self-contained:
1. Automatically downloads PlantVillage from TensorFlow Datasets
2. Filters to 3 tomato classes
3. Trains MobileNetV2 (two phases)
4. Generates all evaluation plots (saved to `docs/`)
5. Saves model + history + metrics to `models/`

Estimated training time: **~25 min on CPU / ~8 min on GPU (T4 on Colab)**

Open in Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SLICKMAN-TYRUS/cropguard-ai/blob/main/notebook/crop_disease_detector.ipynb)

---

## 📄 License

MIT © 2024 Ajak Bul Zacharia Chol — African Leadership University
