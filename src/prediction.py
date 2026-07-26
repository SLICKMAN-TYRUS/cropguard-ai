"""
prediction.py — CropGuard AI
Singleton model loader + inference functions for single & batch image predictions.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from src.preprocessing import preprocess_image_bytes, preprocess_image_path
from src.model import load_model, MODEL_PATH

# ─── Config ───────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Tomato___Early_blight": {
        "description": "Caused by Alternaria solani fungus. Appears as brown spots with concentric rings.",
        "treatment":   "Apply copper-based fungicides. Remove infected leaves. Improve air circulation.",
        "severity":    "Medium",
    },
    "Tomato___Late_blight": {
        "description": "Caused by Phytophthora infestans. Dark water-soaked lesions on leaves and stems.",
        "treatment":   "Apply preventive fungicides (mancozeb, chlorothalonil). Destroy infected plants.",
        "severity":    "High",
    },
    "Tomato___healthy": {
        "description": "Plant shows no signs of disease.",
        "treatment":   "Continue regular care and monitoring.",
        "severity":    "None",
    },
}

# ─── Singleton Cache ──────────────────────────────────────────────────────────
_model: Optional[tf.keras.Model] = None
_model_load_time: Optional[float] = None


def get_model(path: str = str(MODEL_PATH)) -> Optional[tf.keras.Model]:
    """Load model once and cache in module-level variable."""
    global _model, _model_load_time
    if _model is None:
        _model = load_model(path)
        _model_load_time = time.time()
    return _model


def reload_model(path: str = str(MODEL_PATH)) -> Optional[tf.keras.Model]:
    """Force reload model (called after retraining)."""
    global _model, _model_load_time
    _model = None
    return get_model(path)


def model_loaded() -> bool:
    return _model is not None


def model_load_time() -> Optional[float]:
    return _model_load_time


# ─── Inference ────────────────────────────────────────────────────────────────
def _predict_array(arr: np.ndarray) -> Dict:
    """Core inference on a preprocessed (1, H, W, 3) array."""
    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded. Check that models/crop_disease_model.h5 exists.")

    t0    = time.time()
    probs = model.predict(arr, verbose=0)[0]
    latency_ms = (time.time() - t0) * 1000

    idx = int(np.argmax(probs))
    cls = CLASS_NAMES[idx]

    return {
        "class":            cls,
        "class_index":      idx,
        "confidence":       float(probs[idx]),
        "probabilities":    {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "disease_info":     DISEASE_INFO.get(cls, {}),
        "latency_ms":       round(latency_ms, 2),
    }


def predict_from_bytes(image_bytes: bytes) -> Dict:
    """Predict class from raw image bytes (used by API)."""
    arr = preprocess_image_bytes(image_bytes)
    return _predict_array(arr)


def predict_from_path(image_path: str) -> Dict:
    """Predict class from an image file path."""
    arr = preprocess_image_path(image_path)
    return _predict_array(arr)


def batch_predict(image_bytes_list: List[bytes]) -> List[Dict]:
    """
    Predict on a list of image byte strings.
    Batches them into a single model.predict() call for efficiency.
    """
    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded.")

    arrays = np.vstack([preprocess_image_bytes(b) for b in image_bytes_list])

    t0     = time.time()
    preds  = model.predict(arrays, verbose=0)
    total_ms = (time.time() - t0) * 1000
    per_ms   = total_ms / max(len(image_bytes_list), 1)

    results = []
    for i, probs in enumerate(preds):
        idx = int(np.argmax(probs))
        cls = CLASS_NAMES[idx]
        results.append({
            "index":         i,
            "class":         cls,
            "class_index":   idx,
            "confidence":    float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "disease_info":  DISEASE_INFO.get(cls, {}),
            "latency_ms":    round(per_ms, 2),
        })
    return results


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_class_names() -> List[str]:
    return CLASS_NAMES


def get_disease_info(class_name: str) -> Dict:
    return DISEASE_INFO.get(class_name, {})
