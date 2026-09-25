from fastapi.testclient import TestClient

from api.main import app
from src.explanations import build_explanation


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_reports_missing_artifacts(monkeypatch, tmp_path):
    import api.routers.prediction as prediction_router

    monkeypatch.setattr(prediction_router, "MODEL_PATH", tmp_path / "missing-model.joblib")
    monkeypatch.setattr(prediction_router, "PREPROCESSOR_PATH", tmp_path / "missing-preprocessor.joblib")
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_predict_validates_required_patient_fields():
    response = client.post("/predict", json={"Gender": "Female"})

    assert response.status_code == 422


def test_explanation_fallback_is_plain_language():
    explanation = build_explanation(
        "Osteopenia", [{"feature": "Age", "impact": 0.4}]
    )

    assert "Osteopenia" in explanation
    assert "not a diagnosis" in explanation