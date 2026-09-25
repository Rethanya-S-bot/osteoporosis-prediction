import numpy as np
import pandas as pd

from src.inference import explain_prediction


class DummyModel:
    feature_importances_ = np.array([0.6, 0.25, 0.1, 0.05])


def test_explain_prediction_uses_deterministic_fallback_without_shap():
    transformed = pd.DataFrame(
        {
            "Age": [62],
            "BMI": [27.5],
            "Gender_Male": [1],
            "Gender_Female": [0],
        }
    )

    factors = explain_prediction(
        DummyModel(),
        transformed,
        ["Age", "BMI", "Gender_Male", "Gender_Female"],
        max_features=3,
    )

    assert factors
    assert set(item["feature"] for item in factors).issubset({"Age", "BMI", "Gender_Male", "Gender_Female"})
    assert factors[0]["impact"] >= 0
