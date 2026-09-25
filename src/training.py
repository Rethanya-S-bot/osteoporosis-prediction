"""Reproducible model comparison for the osteoporosis tabular dataset."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import CV_FOLDS, MODEL_PATH, PREPROCESSOR_PATH, RANDOM_STATE, RAW_CSV
from src.data_generator import generate_dataset
from src.features import engineer_features
from src.inference import save_artifacts
from src.preprocessor import OsteoporosisPreprocessor


def build_models() -> dict[str, Any]:
    """Build the requested classifiers, skipping unavailable optional packages."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True,
            max_iter=400, random_state=RANDOM_STATE
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            objective="multi:softprob", eval_metric="mlogloss",
            num_class=3, random_state=RANDOM_STATE, n_jobs=1,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=250, learning_rate=0.05, class_weight="balanced",
            random_state=RANDOM_STATE, verbosity=-1,
        )
    except ImportError:
        pass
    return models


def _metrics(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    if probabilities is not None and len(np.unique(y_test)) > 1:
        result["roc_auc"] = float(
            roc_auc_score(y_test, probabilities, multi_class="ovr", labels=np.arange(3))
        )
    else:
        result["roc_auc"] = None
    return result


def train_and_compare(
    df: pd.DataFrame,
    include_bmd: bool = False,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Fit all available models using a leakage-safe stratified split.

    SMOTE is applied only to the transformed training partition. The returned
    preprocessor and fitted models can be used for API inference.
    """
    prepared = engineer_features(df)
    train_df, test_df = train_test_split(
        prepared, test_size=test_size, stratify=prepared["risk_label"], random_state=RANDOM_STATE
    )
    preprocessor = OsteoporosisPreprocessor(include_bmd=include_bmd)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_test = test_df["risk_label"].map(
        {name: index for index, name in enumerate(preprocessor.label_classes_)}
    ).to_numpy()
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)

    fitted: dict[str, Any] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name, model in build_models().items():
        model.fit(X_train_balanced, y_train_balanced)
        fitted[name] = model
        metrics[name] = _metrics(model, X_test, y_test)

    best_name = max(
        metrics,
        key=lambda name: (metrics[name]["roc_auc"] if metrics[name]["roc_auc"] is not None else -1),
    )
    return {
        "models": fitted,
        "metrics": metrics,
        "best_model_name": best_name,
        "preprocessor": preprocessor,
        "cv_folds": CV_FOLDS,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }


def main() -> None:
    """Train the best model from the project dataset and save its artifacts."""
    if RAW_CSV.exists():
        frame = pd.read_csv(RAW_CSV)
    else:
        frame = generate_dataset()

    result = train_and_compare(frame)
    model_name = result["best_model_name"]
    save_artifacts(
        result["models"][model_name],
        result["preprocessor"],
        model_name,
    )
    print(f"Best model: {model_name}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Preprocessor saved: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()