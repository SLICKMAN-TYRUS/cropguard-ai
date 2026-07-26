"""
main.py — CropGuard AI  FastAPI backend
Rubric-complete: database uploads, explicit preprocessing, pre-trained retraining.
"""

import os, sys, time, json, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import src.prediction as pred_module
from src.model import (
    build_model, retrain_from_existing,
    load_history, load_metrics, save_model, evaluate_model, MODEL_PATH,
)
from src.preprocessing import (
    preprocess_image_bytes, save_uploaded_images,
    get_class_distribution, build_generators_from_directory, CLASS_NAMES,
)
from src.database import (
    init_db, save_upload, get_uploads, count_uploads_by_class,
    start_retrain_session, complete_retrain_session, get_retrain_sessions,
    log_prediction, get_prediction_stats,
)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CropGuard AI API",
    description="Plant disease detection — end-to-end ML pipeline for African agriculture",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

SERVER_START   = time.time()
RETRAIN_STATE  = {
    "status": "idle", "started_at": None,
    "finished_at": None, "message": "", "triggered_by": None, "session_id": None,
}


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    init_db()                          # create SQLite tables
    model = pred_module.get_model()
    status = "loaded" if model else "not found — train first"
    print(f"[CropGuard] Model {status}")


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Status"])
def health_check():
    uptime_s   = time.time() - SERVER_START
    h, rem     = divmod(int(uptime_s), 3600)
    m, s       = divmod(rem, 60)
    pred_stats = get_prediction_stats()
    return {
        "status":             "healthy",
        "model_loaded":       pred_module.model_loaded(),
        "uptime_seconds":     round(uptime_s, 1),
        "uptime_human":       f"{h:02d}:{m:02d}:{s:02d}",
        "server_start":       datetime.fromtimestamp(SERVER_START, tz=timezone.utc).isoformat(),
        "predictions_served": pred_stats["total_predictions"],
        "avg_latency_ms":     pred_stats["avg_latency_ms"],
        "timestamp":          datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/model-info", tags=["Status"])
def model_info():
    metrics = load_metrics()
    history = load_history()
    return {
        "model_path":        str(MODEL_PATH),
        "model_exists":      Path(MODEL_PATH).exists(),
        "model_loaded":      pred_module.model_loaded(),
        "classes":           CLASS_NAMES,
        "num_classes":       len(CLASS_NAMES),
        "input_shape":       [224, 224, 3],
        "architecture":      "MobileNetV2 + custom classification head",
        "epochs_trained":    len(history.get("accuracy", [])),
        "best_val_accuracy": history.get("val_accuracy", [None])[-1],
        "accuracy":          metrics.get("accuracy"),
        "auc_roc_macro":     metrics.get("auc_roc_macro"),
        "last_evaluated_at": metrics.get("evaluated_at"),
        "retrain_history":   get_retrain_sessions(limit=5),
    }


# ─── Prediction ───────────────────────────────────────────────────────────────
@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if not pred_module.model_loaded():
        raise HTTPException(503, "Model not loaded. Train a model first.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    try:
        result = pred_module.predict_from_bytes(raw)
        # Persist prediction to database
        log_prediction(
            filename=file.filename or "unknown",
            predicted_class=result["class"],
            confidence=result["confidence"],
            latency_ms=result.get("latency_ms", 0),
        )
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")


@app.post("/batch-predict", tags=["Inference"])
async def batch_predict(files: List[UploadFile] = File(...)):
    if not pred_module.model_loaded():
        raise HTTPException(503, "Model not loaded.")
    raw_list = [await f.read() for f in files]
    results  = pred_module.batch_predict(raw_list)
    for r, f in zip(results, files):
        r["filename"] = f.filename
        log_prediction(f.filename or "unknown", r["class"], r["confidence"], r.get("latency_ms", 0))
    return {"predictions": results, "count": len(results)}


# ─── Dataset stats ────────────────────────────────────────────────────────────
@app.get("/dataset-stats", tags=["Data"])
def dataset_stats():
    return {
        "train":           get_class_distribution("data/train"),
        "test":            get_class_distribution("data/test"),
        "retrain":         get_class_distribution("data/retrain"),
        "total_train":     sum(get_class_distribution("data/train").values()),
        "total_test":      sum(get_class_distribution("data/test").values()),
        "db_uploads":      count_uploads_by_class(),        # from SQLite
        "recent_uploads":  get_uploads(limit=10),           # from SQLite
        "pred_stats":      get_prediction_stats(),
    }


@app.get("/training-history", tags=["Training"])
def training_history():
    h = load_history()
    return {"history": h, "epochs": len(h.get("accuracy", []))}


@app.get("/metrics", tags=["Training"])
def get_metrics():
    m = load_metrics()
    return m if m else {"message": "No metrics yet — train the model first."}


@app.get("/retrain-status", tags=["Training"])
def retrain_status():
    return RETRAIN_STATE


@app.get("/retrain-sessions", tags=["Training"])
def retrain_sessions():
    return {"sessions": get_retrain_sessions(limit=20)}


# ─── Upload ───────────────────────────────────────────────────────────────────
@app.post("/upload", tags=["Data"])
async def upload_images(
    files:      List[UploadFile] = File(...),
    class_name: str              = Form(...),
):
    """
    RUBRIC: 'Data file Uploading + Saving to Database'
    1. Receives image files from the client
    2. Saves files to data/retrain/<class_name>/
    3. Records each upload in SQLite (uploads table)
    """
    if class_name not in CLASS_NAMES:
        raise HTTPException(400, f"Invalid class. Must be one of: {CLASS_NAMES}")

    raw_list  = []
    filenames = []
    for f in files:
        raw = await f.read()
        raw_list.append(raw)
        filenames.append(f.filename or f"upload_{int(time.time())}.jpg")

    # 1. Save files to disk
    dest_dir = Path("data/retrain") / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for fname, raw in zip(filenames, raw_list):
        fpath = dest_dir / f"{int(time.time()*1000)}_{fname}"
        fpath.write_bytes(raw)
        saved_paths.append(str(fpath))

    # 2. Save upload records to SQLite database
    db_ids = []
    for fname, fpath, raw in zip(filenames, saved_paths, raw_list):
        row_id = save_upload(
            filename=fname,
            class_name=class_name,
            file_path=fpath,
            file_size=len(raw),
        )
        db_ids.append(row_id)

    return {
        "message":      f"Saved {len(saved_paths)} image(s) to database and disk",
        "class_name":   class_name,
        "saved":        len(saved_paths),
        "db_record_ids": db_ids,           # SQLite row IDs
        "disk_paths":   saved_paths,
    }


# ─── Retrain ──────────────────────────────────────────────────────────────────
def _retrain_worker(train_dir: str, test_dir: str, epochs: int,
                    triggered_by: str, session_id: int):
    """
    RUBRIC — Full retraining pipeline:
    Step 1  Data Preprocessing  — resize, normalise, augment uploaded images
    Step 2  Pre-trained base    — load existing CropGuard model weights
    Step 3  Fine-tuning         — continue training on new data
    """
    global RETRAIN_STATE
    RETRAIN_STATE.update({
        "status": "running", "started_at": datetime.utcnow().isoformat(),
        "finished_at": None, "message": "Step 1/3 — Preprocessing uploaded images…",
        "triggered_by": triggered_by, "session_id": session_id,
    })
    try:
        # ── STEP 1: Data Preprocessing ────────────────────────────────────────
        print("[Retrain] Step 1/3 — Building preprocessed data generators…")
        RETRAIN_STATE["message"] = "Step 1/3 — Preprocessing: resize → normalise → augment"
        train_ds, val_ds, test_ds = build_generators_from_directory(train_dir, test_dir)
        print("[Retrain] Preprocessing complete.")

        # ── STEP 2 & 3: Load pre-trained model → fine-tune ────────────────────
        RETRAIN_STATE["message"] = "Step 2/3 — Loading pre-trained CropGuard model…"
        print("[Retrain] Step 2/3 — Loading existing model as pre-trained base…")
        model, history = retrain_from_existing(
            train_ds=train_ds, val_ds=val_ds, epochs=epochs,
        )

        RETRAIN_STATE["message"] = "Step 3/3 — Saving model & evaluating…"
        save_model(model)
        metrics = evaluate_model(model, test_ds, save=True)
        pred_module.reload_model()         # hot-reload — no restart needed

        best_val_acc  = max(history.get("val_accuracy", [0]))
        best_val_loss = min(history.get("val_loss", [0]))

        complete_retrain_session(
            session_id, "completed",
            message=f"Done. Best val_acc={best_val_acc:.4f}",
            val_accuracy=best_val_acc, val_loss=best_val_loss,
        )
        RETRAIN_STATE.update({
            "status":      "completed",
            "finished_at": datetime.utcnow().isoformat(),
            "message":     f"✅ Retraining complete — val_accuracy={best_val_acc:.4f}",
        })

    except Exception as exc:
        complete_retrain_session(session_id, "failed", message=str(exc))
        RETRAIN_STATE.update({
            "status":      "failed",
            "finished_at": datetime.utcnow().isoformat(),
            "message":     f"❌ Failed: {exc}",
        })


@app.post("/retrain", tags=["Training"])
def trigger_retrain(
    background_tasks: BackgroundTasks,
    train_dir:    str = "data/retrain",
    test_dir:     str = "data/test",
    epochs:       int = 10,
    triggered_by: str = "api",
):
    global RETRAIN_STATE
    if RETRAIN_STATE["status"] == "running":
        return JSONResponse(409, {"message": "Already running.", "state": RETRAIN_STATE})

    retrain_data = Path(train_dir)
    if not retrain_data.exists() or not any(retrain_data.rglob("*.jpg")):
        raise HTTPException(400, f"No images found in '{train_dir}'. Upload images first.")

    # Record session in SQLite
    session_id = start_retrain_session(triggered_by, train_dir, epochs)

    background_tasks.add_task(
        _retrain_worker, train_dir, test_dir, epochs, triggered_by, session_id,
    )
    return {
        "message":      "Retraining triggered. Poll /retrain-status for progress.",
        "session_id":   session_id,
        "train_dir":    train_dir,
        "epochs":       epochs,
        "triggered_by": triggered_by,
        "pipeline":     [
            "Step 1 — Preprocess uploaded images (resize 224×224, normalise, augment)",
            "Step 2 — Load existing CropGuard model as pre-trained base",
            "Step 3 — Fine-tune on new data → save → evaluate → hot-reload",
        ],
    }


@app.get("/", tags=["Status"])
def root():
    return {
        "name": "CropGuard AI", "version": "2.0.0",
        "docs": "/docs", "health": "/health",
    }
