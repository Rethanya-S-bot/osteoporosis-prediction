"""
features.py — Feature engineering on top of the raw DataFrame.

Adds observed-BMD labels and interaction features. It never fabricates a BMD
measurement: measured T-scores are labels or optional inputs, not replacements
for missing clinical measurements.
"""
import numpy as np
import pandas as pd
from src.config import TARGET_3CLASS
from src.schema import normalize_patient_frame


def classify_risk(t_score: pd.Series) -> pd.Series:
    """WHO T-score classification → Normal / Osteopenia / Osteoporosis."""
    return pd.cut(
        t_score,
        bins=[-np.inf, -2.5, -1.0, np.inf],
        labels=["Osteoporosis", "Osteopenia", "Normal"],
    ).astype(str)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point — enriches df with derived features.
    Call before OsteoporosisPreprocessor.fit_transform().
    """
    df = normalize_patient_frame(df)

    if TARGET_3CLASS not in df.columns and "Risk_Label" in df.columns:
        df[TARGET_3CLASS] = df["Risk_Label"]
    if TARGET_3CLASS not in df.columns and "BMD_T_Score" in df.columns:
        observed_score = pd.to_numeric(df["BMD_T_Score"], errors="coerce")
        if observed_score.notna().all():
            df[TARGET_3CLASS] = classify_risk(observed_score)

    # Interaction features
    if "BMI" in df.columns:
        df["age_bmi_interaction"] = df["Age"] * df["BMI"]
    calcium_score = (
        df["Calcium Intake"].map({"Low": 0, "Adequate": 1, "High": 2}).fillna(1)
        if "Calcium Intake" in df.columns
        else pd.Series(1, index=df.index)
    )
    vitamin_d_score = (
        df["Vitamin D Intake"].map({"Deficient": 0, "Sufficient": 1, "High": 2}).fillna(1)
        if "Vitamin D Intake" in df.columns
        else pd.Series(1, index=df.index)
    )
    df["calcium_vitd_score"] = calcium_score + vitamin_d_score

    return df
