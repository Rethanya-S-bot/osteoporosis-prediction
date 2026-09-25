"""Persisted model inference and optional SHAP explanations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, MODEL_PATH, PREPROCESSOR_PATH
from src.preprocessor import OsteoporosisPreprocessor
from src.schema import PatientRecord


def save_artifacts(
    model: Any,
    preprocessor: OsteoporosisPreprocessor,
    model_name: str,
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
) -> None:
    """Save the selected model and metadata used by inference."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "model_name": model_name, "classes": CLASS_NAMES},
        model_path,
    )
    preprocessor.save(preprocessor_path)


def load_artifacts(
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
) -> tuple[Any, OsteoporosisPreprocessor, dict[str, Any]]:
    """Load the persisted model, preprocessing pipeline, and metadata."""
    bundle = joblib.load(model_path)
    preprocessor = OsteoporosisPreprocessor.load(preprocessor_path)
    return bundle["model"], preprocessor, bundle


def predict_patient(
    patient: PatientRecord | dict[str, Any],
    model: Any,
    preprocessor: OsteoporosisPreprocessor,
) -> dict[str, Any]:
    """Return class probabilities and the selected risk label for one patient."""
    record = patient if isinstance(patient, PatientRecord) else PatientRecord.model_validate(patient)
    frame = pd.DataFrame([record.to_feature_dict()])
    transformed = preprocessor.transform(frame)
    probabilities = model.predict_proba(transformed)[0]
    class_names = preprocessor.label_classes_ or CLASS_NAMES
    probability_map = {
        class_names[index]: float(probabilities[index])
        for index in range(min(len(class_names), len(probabilities)))
    }
    label = max(probability_map, key=probability_map.get)
    return {
        "risk_label": label,
        "confidence": probability_map[label],
        "probabilities": probability_map,
        "model_name": getattr(model, "model_name_", None),
    }


def _fallback_explanations(
    model: Any,
    transformed_patient: pd.DataFrame,
    feature_names: list[str],
    max_features: int = 8,
) -> list[dict[str, float | str]]:
    """Deterministic fallback when SHAP is unavailable or errors out."""
    row = transformed_patient.iloc[0].astype(float)
    if hasattr(model, "feature_importances_"):
        weights = np.abs(np.asarray(model.feature_importances_, dtype=float))
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        weights = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        weights = np.ones(len(feature_names), dtype=float)

    if len(weights) < len(feature_names):
        weights = np.pad(weights, (0, len(feature_names) - len(weights)), constant_values=1.0)
    elif len(weights) > len(feature_names):
        weights = weights[: len(feature_names)]

    scores = np.abs(row.to_numpy(dtype=float)) * weights
    ranked = sorted(
        zip(feature_names, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:max_features]
    return [{"feature": name, "impact": float(value)} for name, value in ranked]


def explain_prediction(
    model: Any,
    transformed_patient: pd.DataFrame,
    feature_names: list[str],
    max_features: int = 8,
) -> list[dict[str, float | str]]:
    """Return the largest absolute SHAP contributions for one transformed row."""
    try:
        import shap
    except ImportError:
        return _fallback_explanations(model, transformed_patient, feature_names, max_features)

    try:
        explainer = shap.Explainer(model, transformed_patient)
        values = explainer(transformed_patient).values
        if values.ndim == 3:
            values = values[:, :, -1]
        row = values[0]
        ranked = sorted(
            zip(feature_names, row), key=lambda item: abs(float(item[1])), reverse=True
        )[:max_features]
        return [
            {"feature": name, "impact": float(value)}
            for name, value in ranked
        ]
    except Exception as exc:
        print(f"[SHAP] explanation failed: {type(exc).__name__}: {exc}")
        return _fallback_explanations(model, transformed_patient, feature_names, max_features)