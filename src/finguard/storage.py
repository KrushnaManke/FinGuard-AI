"""Small SQLite audit store for demo predictions and analyst decisions."""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path


class AuditStore:
    def __init__(self, path: str | Path = "data/processed/finguard.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    transaction_id TEXT NOT NULL,
                    fraud_score REAL NOT NULL,
                    anomaly_score REAL NOT NULL,
                    model_version TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    transaction_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS investigations (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    transaction_id TEXT NOT NULL,
                    data_classification TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )"""
            )

    def _connect(self):
        return sqlite3.connect(self.path)

    def log_prediction(self, transaction_id, fraud_score, anomaly_score, model_version):
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO predictions (transaction_id, fraud_score, anomaly_score, model_version)
                   VALUES (?, ?, ?, ?)""",
                (transaction_id, fraud_score, anomaly_score, model_version),
            )
            return cursor.lastrowid

    def save_review(self, transaction_id, reviewer, status, notes=""):
        if status not in {"open", "confirmed_fraud", "legitimate", "escalated"}:
            raise ValueError("Unsupported analyst review status.")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reviews (transaction_id, reviewer, status, notes) VALUES (?, ?, ?, ?)",
                (transaction_id, reviewer, status, notes),
            )
            return cursor.lastrowid

    def save_investigation(self, transaction_id, report):
        lineage = report.get("data_lineage", {})
        classification = (
            "BENCHMARK_WITH_SYNTHETIC_CONTEXT"
            if lineage.get("synthetic_context_attached")
            else "BENCHMARK_ONLY"
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO investigations (transaction_id, data_classification, report_json)
                   VALUES (?, ?, ?)""",
                (transaction_id, classification, json.dumps(report, sort_keys=True)),
            )
            return cursor.lastrowid

    def dashboard_summary(self):
        with self._connect() as connection:
            return {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("predictions", "reviews", "investigations")
            }

    def recent_records(self, table, limit=20):
        if table not in {"predictions", "reviews", "investigations"}:
            raise ValueError("Unsupported audit table.")
        if not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100.")
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
