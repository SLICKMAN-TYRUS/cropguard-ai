"""
database.py — CropGuard AI
SQLite persistence layer for upload records and retraining sessions.
Satisfies rubric requirement: "Data file Uploading + Saving to Database".
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

_DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = _DB_DIR / "cropguard.db"


# ─── Schema ───────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    class_name  TEXT    NOT NULL,
    file_path   TEXT    NOT NULL,
    file_size   INTEGER,
    uploaded_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS retrain_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_by    TEXT,
    train_dir       TEXT,
    epochs          INTEGER,
    status          TEXT    NOT NULL,   -- running | completed | failed
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    val_accuracy    REAL,
    val_loss        REAL,
    message         TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT,
    predicted_class TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    latency_ms      REAL,
    predicted_at    TEXT    NOT NULL
);
"""


# ─── Connection helper ────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    """Create tables if they don't exist yet. Call once at startup."""
    with _conn() as con:
        con.executescript(_SCHEMA)
    print(f"[DB] Initialised SQLite database at {DB_PATH}")


# ─── Upload records ───────────────────────────────────────────────────────────
def save_upload(
    filename:   str,
    class_name: str,
    file_path:  str,
    file_size:  int = 0,
) -> int:
    """Insert one upload record. Returns the new row id."""
    sql = """
        INSERT INTO uploads (filename, class_name, file_path, file_size, uploaded_at)
        VALUES (?, ?, ?, ?, ?)
    """
    with _conn() as con:
        cur = con.execute(
            sql,
            (filename, class_name, file_path, file_size, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def get_uploads(
    class_name: Optional[str] = None,
    limit:      int           = 50,
) -> List[Dict]:
    """Return upload records, optionally filtered by class."""
    sql  = "SELECT * FROM uploads"
    args: tuple = ()
    if class_name:
        sql  += " WHERE class_name = ?"
        args  = (class_name,)
    sql += f" ORDER BY uploaded_at DESC LIMIT {limit}"
    with _conn() as con:
        rows = con.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def get_unprocessed_uploads(train_dir: str) -> List[Dict]:
    """Return uploads that have not yet been used in a retrain session."""
    # We consider all uploads added after the last completed retrain as "unprocessed"
    sql = """
        SELECT * FROM uploads
        WHERE uploaded_at > COALESCE(
            (SELECT MAX(started_at) FROM retrain_sessions WHERE status = 'completed'),
            '1970-01-01'
        )
        ORDER BY uploaded_at ASC
    """
    with _conn() as con:
        rows = con.execute(sql).fetchall()
    return [dict(r) for r in rows]


def count_uploads_by_class() -> Dict[str, int]:
    sql = "SELECT class_name, COUNT(*) AS cnt FROM uploads GROUP BY class_name"
    with _conn() as con:
        rows = con.execute(sql).fetchall()
    return {r["class_name"]: r["cnt"] for r in rows}


# ─── Retrain sessions ─────────────────────────────────────────────────────────
def start_retrain_session(
    triggered_by: str,
    train_dir:    str,
    epochs:       int,
) -> int:
    """Insert a retrain session record with status='running'. Returns session id."""
    sql = """
        INSERT INTO retrain_sessions
            (triggered_by, train_dir, epochs, status, started_at)
        VALUES (?, ?, ?, 'running', ?)
    """
    with _conn() as con:
        cur = con.execute(sql, (triggered_by, train_dir, epochs, datetime.utcnow().isoformat()))
        return cur.lastrowid


def complete_retrain_session(
    session_id:   int,
    status:       str,
    message:      str       = "",
    val_accuracy: float     = None,
    val_loss:     float     = None,
) -> None:
    sql = """
        UPDATE retrain_sessions
        SET status=?, finished_at=?, message=?, val_accuracy=?, val_loss=?
        WHERE id=?
    """
    with _conn() as con:
        con.execute(
            sql,
            (status, datetime.utcnow().isoformat(), message, val_accuracy, val_loss, session_id),
        )


def get_retrain_sessions(limit: int = 10) -> List[Dict]:
    sql = "SELECT * FROM retrain_sessions ORDER BY started_at DESC LIMIT ?"
    with _conn() as con:
        rows = con.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ─── Prediction logging ───────────────────────────────────────────────────────
def log_prediction(
    filename:        str,
    predicted_class: str,
    confidence:      float,
    latency_ms:      float = 0.0,
) -> None:
    sql = """
        INSERT INTO predictions (filename, predicted_class, confidence, latency_ms, predicted_at)
        VALUES (?, ?, ?, ?, ?)
    """
    with _conn() as con:
        con.execute(
            sql,
            (filename, predicted_class, confidence, latency_ms, datetime.utcnow().isoformat()),
        )


def get_prediction_stats() -> Dict:
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        by_class = con.execute(
            "SELECT predicted_class, COUNT(*) AS cnt, AVG(confidence) AS avg_conf "
            "FROM predictions GROUP BY predicted_class"
        ).fetchall()
        avg_latency = con.execute("SELECT AVG(latency_ms) FROM predictions").fetchone()[0]
    return {
        "total_predictions": total,
        "avg_latency_ms":    round(avg_latency or 0, 2),
        "by_class":          [dict(r) for r in by_class],
    }