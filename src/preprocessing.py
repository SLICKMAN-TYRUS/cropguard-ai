"""
preprocessing.py — CropGuard AI
Image preprocessing utilities for the PlantVillage tomato disease dataset.
"""

import os
import io
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import tensorflow as tf
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
NUM_CLASSES   = 3

CLASS_NAMES   = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]
CLASS_LABELS  = {name: idx for idx, name in enumerate(CLASS_NAMES)}
UPLOAD_LOG    = Path("data/upload_log.json")
RETRAIN_DIR   = Path("data/retrain")


# ─── tf.data Pipeline ─────────────────────────────────────────────────────────
def _parse_and_resize(image: tf.Tensor, label: tf.Tensor,
                      img_size: Tuple[int, int] = IMG_SIZE) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_tf_datasets(
    raw_train: tf.data.Dataset,
    raw_val:   tf.data.Dataset,
    raw_test:  tf.data.Dataset,
    img_size:    Tuple[int, int] = IMG_SIZE,
    batch_size:  int             = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build optimised train / val / test tf.data pipelines."""
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        raw_train
        .map(lambda img, lbl: _parse_and_resize(img, lbl, img_size), num_parallel_calls=AUTOTUNE)
        .map(_augment, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        raw_val
        .map(lambda img, lbl: _parse_and_resize(img, lbl, img_size), num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        raw_test
        .map(lambda img, lbl: _parse_and_resize(img, lbl, img_size), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds, test_ds


# All 3 class names the model was trained on — used to pad missing classes
_ALL_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

def _ensure_all_classes(retrain_dir: str) -> None:
    """
    Keras image_dataset_from_directory requires at least one image per class
    subfolder. If the user only uploaded images for one class, we copy a single
    placeholder image from the models/samples folder (or create a blank one)
    into the missing class subfolders so the label shape stays (None, 3).
    """
    import shutil, os
    retrain_path = Path(retrain_dir)
    present = {p.name for p in retrain_path.iterdir() if p.is_dir()}
    missing = [c for c in _ALL_CLASSES if c not in present]
    if not missing:
        return  # all classes already present

    # Find any existing image to copy as a placeholder
    sample_img = None
    for p in retrain_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            sample_img = p
            break

    for cls in missing:
        cls_dir = retrain_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        # Copy at least 2 images so the 80/20 val split has >=1 in each subset
        for i in range(2):
            dest = cls_dir / f"placeholder_{i}.png"
            if not dest.exists():
                if sample_img:
                    shutil.copy(sample_img, dest)
                else:
                    from PIL import Image as PILImage
                    PILImage.new("RGB", (224, 224), (86, 201, 154)).save(dest)
        print(f"[Preprocess] Added placeholders for missing class: {cls}")


def build_generators_from_directory(
    train_dir:  str,
    test_dir:   str = None,
    img_size:   Tuple[int, int] = IMG_SIZE,
    batch_size: int             = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build datasets from directory structure (used during retraining).
    Automatically pads missing class subfolders so label shape is always (None, 3).
    If test_dir does not exist or is empty, the validation split is reused."""
    import os

    # Ensure all 3 class folders exist before Keras scans the directory
    _ensure_all_classes(train_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    # Use test_dir only if it exists and has images, otherwise reuse val split
    img_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    use_test_dir = (
        test_dir
        and os.path.isdir(test_dir)
        and any(
            Path(p).suffix.lower() in img_suffixes
            for p in Path(test_dir).rglob("*")
            if Path(p).is_file()
        )
    )
    if use_test_dir:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False,
        )
    else:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
        )
    # normalise
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y))
    val_ds   = val_ds.map(lambda x, y: (norm(x), y))
    test_ds  = test_ds.map(lambda x, y: (norm(x), y))
    return train_ds, val_ds, test_ds


# ─── Single-Image Helpers ─────────────────────────────────────────────────────
def preprocess_image_bytes(
    image_bytes: bytes,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> np.ndarray:
    """Decode raw bytes → normalised numpy array ready for model.predict()."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape (1, H, W, 3)


def preprocess_image_path(
    image_path: str,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> np.ndarray:
    """Load image from path → normalised numpy array."""
    img = Image.open(image_path).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─── Dataset Statistics ───────────────────────────────────────────────────────
def get_class_distribution(data_dir: str) -> Dict[str, int]:
    """Count images per class inside a directory-based dataset."""
    dist: Dict[str, int] = {}
    base = Path(data_dir)
    if not base.exists():
        return dist
    for cls_dir in sorted(base.iterdir()):
        if cls_dir.is_dir():
            dist[cls_dir.name] = sum(1 for f in cls_dir.iterdir() if f.is_file())
    return dist


def compute_pixel_stats(data_dir: str, max_per_class: int = 100) -> Dict[str, Dict]:
    """Compute mean/std of pixel values per class (used for visualisations)."""
    stats: Dict[str, Dict] = {}
    base = Path(data_dir)
    if not base.exists():
        return stats
    for cls_dir in sorted(base.iterdir()):
        if not cls_dir.is_dir():
            continue
        pixels: List[np.ndarray] = []
        for img_path in list(cls_dir.iterdir())[:max_per_class]:
            try:
                arr = np.array(
                    Image.open(img_path).convert("RGB").resize((64, 64)),
                    dtype=np.float32,
                ) / 255.0
                pixels.append(arr)
            except Exception:
                continue
        if pixels:
            stack = np.stack(pixels)
            stats[cls_dir.name] = {
                "mean_r": float(stack[..., 0].mean()),
                "mean_g": float(stack[..., 1].mean()),
                "mean_b": float(stack[..., 2].mean()),
                "std_r":  float(stack[..., 0].std()),
                "std_g":  float(stack[..., 1].std()),
                "std_b":  float(stack[..., 2].std()),
            }
    return stats


# ─── Upload & Logging ─────────────────────────────────────────────────────────
def save_uploaded_images(
    files: List[bytes],
    class_name: str,
    upload_dir: Optional[str] = None,
) -> int:
    """Save uploaded image bytes under data/retrain/<class_name>/."""
    dest = Path(upload_dir) if upload_dir else (RETRAIN_DIR / class_name)
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, raw in enumerate(files):
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            fname = f"upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{i}.jpg"
            img.save(dest / fname, "JPEG")
            saved += 1
        except Exception:
            continue
    _log_upload(str(dest), saved)
    return saved


def _log_upload(upload_dir: str, n_files: int) -> None:
    UPLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    logs: List[Dict] = []
    if UPLOAD_LOG.exists():
        with open(UPLOAD_LOG) as f:
            logs = json.load(f)
    logs.append({
        "timestamp": datetime.utcnow().isoformat(),
        "directory": upload_dir,
        "files_saved": n_files,
    })
    with open(UPLOAD_LOG, "w") as f:
        json.dump(logs, f, indent=2)


def get_upload_logs() -> List[Dict]:
    if not UPLOAD_LOG.exists():
        return []
    with open(UPLOAD_LOG) as f:
        return json.load(f)