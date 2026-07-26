"""
locustfile.py — CropGuard AI
Simulates a flood of prediction requests to measure latency and throughput
across different numbers of Docker containers / replicas.

Run:
    locust -f locustfile.py --host=http://localhost:8000
    # Or headless:
    locust -f locustfile.py --host=http://localhost:8000 \
           --users=100 --spawn-rate=10 --run-time=60s --headless \
           --csv=results/locust_results
"""

import os
import random
import io
from locust import HttpUser, task, between, events
from PIL import Image
import numpy as np


# ─── Generate synthetic leaf-like test images in memory ───────────────────────
def _make_test_image(label: str = "healthy") -> bytes:
    """
    Produce a random 224×224 RGB image whose colour distribution
    loosely mimics the three tomato classes. Used so tests don't need
    real images on disk.
    """
    img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    if label == "healthy":
        # green-dominant
        img_arr[:, :, 0] = np.random.randint(40, 100, (224, 224))
        img_arr[:, :, 1] = np.random.randint(100, 180, (224, 224))
        img_arr[:, :, 2] = np.random.randint(30, 80, (224, 224))
    elif label == "early_blight":
        # brown tones
        img_arr[:, :, 0] = np.random.randint(100, 180, (224, 224))
        img_arr[:, :, 1] = np.random.randint(60, 120, (224, 224))
        img_arr[:, :, 2] = np.random.randint(20, 60, (224, 224))
    else:  # late_blight
        # dark water-soaked
        img_arr[:, :, 0] = np.random.randint(20, 80, (224, 224))
        img_arr[:, :, 1] = np.random.randint(20, 70, (224, 224))
        img_arr[:, :, 2] = np.random.randint(20, 60, (224, 224))

    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


# Pre-generate a small pool of test images so we're not generating on every request
_IMAGE_POOL = [
    _make_test_image(label)
    for label in ["healthy", "early_blight", "late_blight"] * 10  # 30 images
]


# ─── User Profiles ────────────────────────────────────────────────────────────
class FarmerUser(HttpUser):
    """
    Simulates a field agent uploading single leaf images for prediction.
    Heavy on /predict, light polling of /health.
    """
    wait_time = between(0.5, 2.0)   # seconds between tasks
    weight    = 70                   # 70 % of virtual users

    @task(10)
    def predict_single(self):
        img_bytes = random.choice(_IMAGE_POOL)
        with self.client.post(
            "/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
            catch_response=True,
            name="/predict (single)",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "class" not in data:
                    resp.failure("Response missing 'class' field")
            elif resp.status_code == 503:
                resp.failure("Model not loaded (503)")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(2)
    def health_check(self):
        with self.client.get("/health", catch_response=True, name="/health") as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(1)
    def model_info(self):
        self.client.get("/model-info", name="/model-info")


class AgriAnalystUser(HttpUser):
    """
    Simulates an analyst sending batches of images for bulk classification.
    Heavy on /batch-predict and dataset stats.
    """
    wait_time = between(1.0, 4.0)
    weight    = 20                   # 20 % of virtual users

    @task(5)
    def batch_predict(self):
        batch_size = random.randint(2, 6)
        files = [
            ("files", (f"leaf_{i}.jpg", random.choice(_IMAGE_POOL), "image/jpeg"))
            for i in range(batch_size)
        ]
        with self.client.post(
            "/batch-predict",
            files=files,
            catch_response=True,
            name="/batch-predict",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if len(data.get("predictions", [])) != batch_size:
                    resp.failure("Batch size mismatch in response")
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def dataset_stats(self):
        self.client.get("/dataset-stats", name="/dataset-stats")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="/metrics")


class MonitorUser(HttpUser):
    """
    Simulates a DevOps/monitoring probe — rapid lightweight health checks.
    """
    wait_time = between(0.2, 0.8)
    weight    = 10                   # 10 % of virtual users

    @task(1)
    def health(self):
        self.client.get("/health", name="/health (monitor)")

    @task(1)
    def retrain_status(self):
        self.client.get("/retrain-status", name="/retrain-status")


# ─── Event hooks (CSV logging) ─────────────────────────────────────────────────
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    os.makedirs("results", exist_ok=True)
    print("\n[Locust] Load test starting — results will be written to results/")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats
    print("\n[Locust] ─── Load Test Summary ───")
    for name, entry in stats.entries.items():
        print(
            f"  {name[1]:30s} | "
            f"Reqs: {entry.num_requests:5d} | "
            f"Fails: {entry.num_failures:4d} | "
            f"Avg: {entry.avg_response_time:6.1f} ms | "
            f"p50: {entry.get_response_time_percentile(0.50):6.1f} ms | "
            f"p95: {entry.get_response_time_percentile(0.95):6.1f} ms | "
            f"p99: {entry.get_response_time_percentile(0.99):6.1f} ms | "
            f"RPS: {entry.current_rps:5.1f}"
        )
    print("─────────────────────────────────────────────────────────────────────")
