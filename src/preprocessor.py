"""Leakage-safe tabular preprocessing for osteoporosis risk models."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    LABEL_ENCODER_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    SMOTE_THRESHOLD,
    TARGET_3CLASS,
)
from src.features import engineer_features
from src.schema import normalize_patient_frame


TARGET_COLUMNS = {TARGET_3CLASS, "Risk_Label", "Osteoporosis"}
NUMERIC_CANDIDATES = [
    "Age", "BMI", "BMD_T_Score", "age_bmi_interaction", "calcium_vitd_score"
]
CATEGORICAL_CANDIDATES = [
    "Gender", "Calcium Intake", "Vitamin D Intake", "Smoking",
    "Alcohol Consumption", "Family History", "Hormonal Changes",
    "Physical Activity", "Prior Fractures", "Medications",
    "Medical Conditions", "Race/Ethnicity",
]


class OsteoporosisPreprocessor:
    """Fit preprocessing once on training data and reuse it unchanged."""

    def __init__(self, include_bmd: bool = False):
        self.include_bmd = include_bmd
        self.transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []
        self.input_feature_names_: list[str] = []
        self.label_classes_: list[str] = []
        self._fitted = False

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = normalize_patient_frame(df)
        result = result.drop(columns=list(TARGET_COLUMNS), errors="ignore")
        if not self.include_bmd:
            result = result.drop(columns=["BMD_T_Score"], errors="ignore")
        return result

    def fit(self, df: pd.DataFrame) -> "OsteoporosisPreprocessor":
        features = self._features(engineer_features(df))
        numeric = [c for c in NUMERIC_CANDIDATES if c in features.columns]
        categorical = [c for c in CATEGORICAL_CANDIDATES if c in features.columns]
        unknown = [c for c in features.columns if c not in numeric + categorical]
        if unknown:
            raise ValueError(f"Unsupported feature columns: {unknown}")
        self.input_feature_names_ = list(features.columns)
        self.transformer = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), numeric),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), categorical),
            ],
            remainder="drop",
        )
        self.transformer.fit(features)
        self.feature_names_ = list(self.transformer.get_feature_names_out())
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.transformer is None:
            raise RuntimeError("Call .fit() before .transform().")
        features = self._features(engineer_features(df)).reindex(
            columns=self.input_feature_names_
        )
        values = self.transformer.transform(features)
        return pd.DataFrame(values, columns=self.feature_names_, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        self.fit(df)
        transformed = self.transform(df)
        labels = engineer_features(df).get(TARGET_3CLASS)
        if labels is None:
            raise ValueError("Training data must contain Risk_Label or observed BMD_T_Score.")
        self.label_classes_ = ["Normal", "Osteopenia", "Osteoporosis"]
        unknown = set(labels.dropna().astype(str)) - set(self.label_classes_)
        if unknown:
            raise ValueError(f"Unsupported risk labels: {sorted(unknown)}")
        encoded = labels.map({name: index for index, name in enumerate(self.label_classes_)})
        if encoded.isna().any():
            raise ValueError("Training labels cannot be missing.")
        return transformed, encoded.astype(int).to_numpy()

    def apply_smote(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        counts = np.bincount(y)
        if len(counts) < 2 or counts.min() < 2 or counts.min() / counts.sum() >= SMOTE_THRESHOLD:
            return X, y
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise RuntimeError(
                "imbalanced-learn is required when SMOTE is enabled; "
                "install dependencies from requirements.txt."
            ) from exc
        neighbors = max(1, min(5, int(counts.min()) - 1))
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=neighbors)
        values, labels = sampler.fit_resample(X, y)
        return pd.DataFrame(values, columns=X.columns), labels

    def save(self, path: Path = PREPROCESSOR_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        joblib.dump(self.label_classes_, LABEL_ENCODER_PATH)

    @classmethod
    def load(cls, path: Path = PREPROCESSOR_PATH) -> "OsteoporosisPreprocessor":
        return joblib.load(path)


def load_and_preprocess(csv_path: Path, include_bmd: bool = False):
    df = pd.read_csv(csv_path)
    preprocessor = OsteoporosisPreprocessor(include_bmd=include_bmd)
    X, y = preprocessor.fit_transform(df)
    X, y = preprocessor.apply_smote(X, y)
    return X, y, preprocessor