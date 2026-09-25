"""
db.py — Supabase client and all database helpers.
Credits: pattern adapted from Supabase Python SDK docs.
"""
from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime
from typing import Optional

from src.config import SUPABASE_URL, SUPABASE_KEY

# ---------------------------------------------------------------------------
# Client initialisation (lazy — only fails if credentials are wrong at call
# time, so unit-tests that mock this module still import cleanly)
# ---------------------------------------------------------------------------
_client = None
_PREDICTION_TABLES = ("prediction_history", "predictions")


def _get_client():
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env. "
                "Copy .env.example → .env and fill in your credentials."
            )
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ---------------------------------------------------------------------------
# patients
# ---------------------------------------------------------------------------
def insert_patient(patient_data: dict) -> Optional[str]:
    """Insert a patient record and return the new UUID."""
    try:
        client = _get_client()
        print(f"[DB] insert_patient -> payload={patient_data}")
        result = client.table("patients").insert(patient_data).execute()
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        print(f"[DB] insert_patient error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
def _table_for_prediction_op():
    client = _get_client()
    for table_name in _PREDICTION_TABLES:
        try:
            client.table(table_name).select("*").limit(1).execute()
            return table_name
        except Exception:
            continue
    return "prediction_history"


def insert_prediction(
    patient_id: Optional[str],
    model_name: str,
    risk_label: str,
    confidence: float,
    probabilities: dict,
    shap_values: dict,
    llm_summary: str,
    input_data: Optional[dict] = None,
) -> Optional[str]:
    """Persist a prediction result and return its UUID."""
    try:
        client = _get_client()
        table_name = _table_for_prediction_op()
        payload = {
            "patient_id": patient_id,
            "model_name": model_name,
            "risk_label": risk_label,
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities,
            "input_data": input_data or {},
            "prediction_result": risk_label,
            "shap_values": shap_values,
            "llm_summary": llm_summary,
        }
        print(f"[DB] insert_prediction -> table={table_name}, payload={payload}")
        try:
            result = client.table(table_name).insert(payload).execute()
        except Exception:
            # Keep existing deployments working until the additive migration is applied.
            legacy_payload = {key: value for key, value in payload.items()
                              if key not in {"input_data", "prediction_result"}}
            result = client.table(table_name).insert(legacy_payload).execute()
        if result.data:
            return result.data[0]["id"]
        print(f"[DB] insert_prediction returned no data: {result}")
        return None
    except Exception as e:
        print(f"[DB] insert_prediction error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def fetch_predictions(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` predictions."""
    try:
        client = _get_client()
        table_name = _table_for_prediction_op()
        result = (
            client.table(table_name)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        records = []
        for row in result.data or []:
            probabilities = row.get("probabilities") or {}
            records.append({
                **row,
                "input_data": row.get("input_data") or {},
                "prediction_result": row.get("prediction_result") or row.get("risk_label"),
                "confidence_scores": probabilities,
            })
        return records
    except Exception as e:
        print(f"[DB] fetch_predictions error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# model_runs
# ---------------------------------------------------------------------------
def log_model_run(
    model_name: str,
    metrics: dict,
    hyperparams: dict,
    cv_folds: int,
    train_rows: int,
    test_rows: int,
    is_best: bool = False,
) -> Optional[str]:
    """Log a model training run with metrics to model_runs table."""
    try:
        client = _get_client()
        # Clear previous best if we're setting a new one
        if is_best:
            client.table("model_runs").update({"is_best": False}).eq(
                "is_best", True
            ).execute()

        payload = {
            "model_name": model_name,
            "accuracy": round(float(metrics.get("accuracy", 0)), 4),
            "precision_macro": round(float(metrics.get("precision_macro", 0)), 4),
            "recall_macro": round(float(metrics.get("recall_macro", 0)), 4),
            "f1_macro": round(float(metrics.get("f1_macro", 0)), 4),
            "roc_auc": round(float(metrics.get("roc_auc", 0)), 4),
            "hyperparams": hyperparams,
            "cv_folds": cv_folds,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "is_best": is_best,
        }
        result = client.table("model_runs").insert(payload).execute()
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        print(f"[DB] log_model_run error: {e}")
        return None


def fetch_model_runs(limit: int = 20) -> list[dict]:
    """Return the most recent model run records."""
    try:
        client = _get_client()
        result = (
            client.table("model_runs")
            .select("*")
            .order("trained_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[DB] fetch_model_runs error: {e}")
        return []


def fetch_best_model_run() -> Optional[dict]:
    """Return the current best model run record."""
    try:
        client = _get_client()
        result = (
            client.table("model_runs")
            .select("*")
            .eq("is_best", True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"[DB] fetch_best_model_run error: {e}")
        return None


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def upsert_dataset_meta(name: str, source_url: str, rows: int, features: int):
    """Record dataset metadata in the datasets table."""
    try:
        client = _get_client()
        client.table("datasets").insert(
            {"name": name, "source_url": source_url, "rows": rows, "features": features}
        ).execute()
    except Exception as e:
        print(f"[DB] upsert_dataset_meta error: {e}")
