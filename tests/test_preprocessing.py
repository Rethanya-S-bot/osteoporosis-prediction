import pandas as pd

from src.features import engineer_features
from src.preprocessor import OsteoporosisPreprocessor
from src.schema import PatientRecord


def _training_frame():
    return pd.DataFrame([
        {
            "Age": 60,
            "Gender": "Female",
            "Body Weight": 24.0,
            "Calcium Intake": "Low",
            "Vitamin D Intake": "Deficient",
            "Smoking": "No",
            "Alcohol Consumption": "None",
            "Family History": "Yes",
            "Hormonal Changes": "Postmenopausal",
            "Physical Activity": "Sedentary",
            "Prior Fractures": "No",
            "Medications": "None",
            "Medical Conditions": "None",
            "Risk_Label": "Osteopenia",
        },
        {
            "Age": 45,
            "Gender": "Male",
            "Body Weight": 29.0,
            "Calcium Intake": "Adequate",
            "Vitamin D Intake": "Sufficient",
            "Smoking": "No",
            "Alcohol Consumption": "Moderate",
            "Family History": "No",
            "Hormonal Changes": "Normal",
            "Physical Activity": "Active",
            "Prior Fractures": "No",
            "Medications": "None",
            "Medical Conditions": "None",
            "Risk_Label": "Normal",
        },
    ])


def test_patient_contract_normalizes_legacy_bmi_name():
    patient = PatientRecord(Age=55, Gender="Female", **{"Body Weight": 25.0})

    assert patient.to_feature_dict()["BMI"] == 25.0
    assert "Body Weight" not in patient.to_feature_dict()


def test_screening_preprocessor_transforms_unlabeled_rows_without_bmd():
    training = _training_frame()
    preprocessor = OsteoporosisPreprocessor()
    X_train, y_train = preprocessor.fit_transform(training)
    X_inference = preprocessor.transform(training.drop(columns=["Risk_Label"]).iloc[[0]])

    assert X_train.shape[1] == X_inference.shape[1]
    assert y_train.tolist() == [1, 0]
    assert not any("BMD_T_Score" in name for name in preprocessor.feature_names_)


def test_observed_bmd_can_derive_who_label_without_synthesis():
    frame = pd.DataFrame({"Age": [40, 60, 75], "Gender": ["Male"] * 3, "BMD_T_Score": [0.0, -1.5, -3.0]})

    engineered = engineer_features(frame)

    assert engineered["risk_label"].tolist() == ["Normal", "Osteopenia", "Osteoporosis"]