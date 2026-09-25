"""Prediction, history, and model metadata endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_PATH, PREPROCESSOR_PATH
from src.db import fetch_model_runs, insert_prediction
from src.explanations import build_explanation
from src.inference import explain_prediction, load_artifacts, predict_patient
from src.schema import PatientRecord


router = APIRouter()
_artifacts: tuple[Any, Any, dict] | None = None


class PredictionResponse(BaseModel):
    risk_label: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    contributing_factors: list[dict[str, float | str]] = Field(default_factory=list)
    explanation: str
    prediction_id: str | None = None


def _artifacts_available() -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (Path(MODEL_PATH), Path(PREPROCESSOR_PATH))
    )


def _load_prediction_artifacts():
    global _artifacts
    if _artifacts is None:
        if not _artifacts_available():
            raise HTTPException(
                status_code=503,
                detail="Model artifacts are unavailable. Train and save a model first.",
            )
        try:
            _artifacts = load_artifacts()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model artifacts could not be loaded.") from exc
    return _artifacts


@router.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientRecord) -> PredictionResponse:
    model, preprocessor, metadata = _load_prediction_artifacts()
    result = predict_patient(patient, model, preprocessor)
    transformed = preprocessor.transform(pd.DataFrame([patient.to_feature_dict()]))
    factors = explain_prediction(model, transformed, preprocessor.feature_names_)
    explanation = build_explanation(result["risk_label"], factors)
    prediction_id = insert_prediction(
        patient_id=None,
        model_name=metadata["model_name"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        shap_values={item["feature"]: item["impact"] for item in factors},
        llm_summary=explanation,
        input_data=patient.to_input_dict(),
    )
    return PredictionResponse(
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_name=metadata["model_name"],
        contributing_factors=factors,
        explanation=explanation,
        prediction_id=prediction_id,
    )


@router.get("/model-info")
def model_info() -> dict:
    if not _artifacts_available():
        return {"available": False, "model_name": None, "metrics": None}
    try:
        _, preprocessor, metadata = _load_prediction_artifacts()
    except HTTPException:
        return {"available": False, "model_name": None, "metrics": None}
    runs = fetch_model_runs(limit=20)
    matching = [run for run in runs if run.get("model_name") == metadata["model_name"]]
    return {
        "available": True,
        "model_name": metadata["model_name"],
        "features": preprocessor.feature_names_,
        "metrics": matching[0] if matching else None,
    }