"""
data_generator.py
-----------------
Generates a realistic synthetic osteoporosis dataset (2 000 patients)
matching the Kaggle "Osteoporosis Risk Prediction" schema.

Run directly:
    python src/data_generator.py

Or import:
    from src.data_generator import generate_dataset
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW_DIR, RANDOM_STATE

np.random.seed(RANDOM_STATE)


def generate_dataset(n: int = 2000, save: bool = True) -> pd.DataFrame:
    """Generate n synthetic patient records and optionally save to CSV."""

    ages = np.random.randint(30, 85, n)
    genders = np.random.choice(["Female", "Male"], n, p=[0.62, 0.38])

    # BMI: slightly higher for older patients
    bmi = np.clip(np.random.normal(26 + ages * 0.02, 4.5, n), 16, 42)

    # Calcium intake: low / adequate / high
    calcium = np.random.choice(["Low", "Adequate", "High"], n, p=[0.30, 0.55, 0.15])

    # Vitamin D: deficient / sufficient / high
    vitamin_d = np.random.choice(["Deficient", "Sufficient", "High"], n, p=[0.35, 0.50, 0.15])

    # Lifestyle
    physical_activity = np.random.choice(["Sedentary", "Active", "Very Active"], n, p=[0.40, 0.40, 0.20])
    smoking = np.random.choice(["Yes", "No"], n, p=[0.22, 0.78])
    alcohol = np.random.choice(["None", "Moderate", "High"], n, p=[0.45, 0.40, 0.15])

    # Medical history
    family_history = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
    prior_fractures = np.random.choice(["Yes", "No"], n, p=[0.20, 0.80])
    hormonal_changes = np.random.choice(["Normal", "Postmenopausal", "Deficiency"], n, p=[0.50, 0.30, 0.20])
    race_ethnicity = np.random.choice(
        ["Caucasian", "Asian", "African American", "Hispanic"],
        n, p=[0.45, 0.25, 0.18, 0.12]
    )
    medical_conditions = np.random.choice(
        ["None", "Rheumatoid Arthritis", "Hyperthyroidism", "Diabetes", "CKD"],
        n, p=[0.55, 0.12, 0.10, 0.13, 0.10]
    )
    medications = np.random.choice(
        ["None", "Corticosteroids", "Anticonvulsants", "Antidepressants", "Bisphosphonates"],
        n, p=[0.55, 0.15, 0.10, 0.10, 0.10]
    )

    # --- Derive BMD T-score (synthetic, clinically motivated) ---
    t_score = np.zeros(n)
    t_score -= (ages - 30) * 0.025
    t_score -= (genders == "Female").astype(float) * 0.4
    t_score += (bmi - 26) * 0.05
    t_score -= (calcium == "Low").astype(float) * 0.3
    t_score -= (vitamin_d == "Deficient").astype(float) * 0.35
    t_score -= (physical_activity == "Sedentary").astype(float) * 0.3
    t_score += (physical_activity == "Very Active").astype(float) * 0.2
    t_score -= (smoking == "Yes").astype(float) * 0.25
    t_score -= (alcohol == "High").astype(float) * 0.2
    t_score -= (family_history == "Yes").astype(float) * 0.3
    t_score -= (prior_fractures == "Yes").astype(float) * 0.4
    t_score -= (hormonal_changes == "Postmenopausal").astype(float) * 0.5
    t_score -= (hormonal_changes == "Deficiency").astype(float) * 0.35
    t_score -= (medical_conditions == "Rheumatoid Arthritis").astype(float) * 0.25
    t_score -= (medical_conditions == "Hyperthyroidism").astype(float) * 0.2
    t_score -= (medical_conditions == "CKD").astype(float) * 0.3
    t_score -= (medications == "Corticosteroids").astype(float) * 0.3
    t_score += np.random.normal(0, 0.3, n)   # noise

    # --- WHO classification ---
    conditions = [t_score >= -1.0, (t_score < -1.0) & (t_score >= -2.5), t_score < -2.5]
    risk_label  = np.select(
        conditions, ["Normal", "Osteopenia", "Osteoporosis"], default="Normal"
    )

    # Binary Osteoporosis label (for compatibility with Kaggle CSV)
    osteoporosis_binary = (risk_label == "Osteoporosis").astype(int)

    df = pd.DataFrame({
        "Age": ages,
        "Gender": genders,
        "Hormonal Changes": hormonal_changes,
        "Family History": family_history,
        "Race/Ethnicity": race_ethnicity,
        "Body Weight": np.round(bmi, 1),
        "Calcium Intake": calcium,
        "Vitamin D Intake": vitamin_d,
        "Physical Activity": physical_activity,
        "Smoking": smoking,
        "Alcohol Consumption": alcohol,
        "Medical Conditions": medical_conditions,
        "Medications": medications,
        "Prior Fractures": prior_fractures,
        "BMD_T_Score": np.round(t_score, 2),
        "Risk_Label": risk_label,           # 3-class label
        "Osteoporosis": osteoporosis_binary # binary label
    })

    if save:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_RAW_DIR / "osteoporosis.csv"
        df.to_csv(out, index=False)
        print(f"[✓] Synthetic dataset saved → {out}  ({len(df)} rows)")
        print(df["Risk_Label"].value_counts())

    return df


if __name__ == "__main__":
    generate_dataset()
