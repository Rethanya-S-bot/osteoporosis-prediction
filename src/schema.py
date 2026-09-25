"""Canonical patient input and label contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, confloat, conint


RiskLabel = Literal["Normal", "Osteopenia", "Osteoporosis"]


class PatientRecord(BaseModel):
    """Validated patient features accepted by training and prediction APIs."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    age: conint(ge=1, le=120) = Field(alias="Age")
    gender: Literal["Male", "Female", "Other"] = Field(alias="Gender")
    height_cm: confloat(gt=50, lt=250) | None = Field(default=None, alias="Height")
    weight_kg: confloat(gt=10, lt=300) | None = Field(default=None, alias="Weight")
    bmi: confloat(gt=10, lt=80) | None = Field(default=None, alias="BMI")
    body_weight_legacy: confloat(gt=10, lt=80) | None = Field(
        default=None, alias="Body Weight"
    )
    calcium_intake: Literal["Low", "Adequate", "High"] | None = Field(
        default=None, alias="Calcium Intake"
    )
    vitamin_d_intake: Literal["Deficient", "Sufficient", "High"] | None = Field(
        default=None, alias="Vitamin D Intake"
    )
    smoking: Literal["Yes", "No"] | None = Field(default=None, alias="Smoking")
    alcohol_consumption: Literal["None", "Moderate", "High"] | None = Field(
        default=None, alias="Alcohol Consumption"
    )
    family_history: Literal["Yes", "No"] | None = Field(
        default=None, alias="Family History"
    )
    hormonal_changes: Literal["Normal", "Postmenopausal", "Deficiency"] | None = Field(
        default=None, alias="Hormonal Changes"
    )
    physical_activity: Literal["Sedentary", "Active", "Very Active"] | None = Field(
        default=None, alias="Physical Activity"
    )
    prior_fractures: Literal["Yes", "No"] | None = Field(
        default=None, alias="Prior Fractures"
    )
    medications: str | None = Field(default=None, alias="Medications")
    medical_conditions: str | None = Field(default=None, alias="Medical Conditions")
    race_ethnicity: str | None = Field(default=None, alias="Race/Ethnicity")
    bmd_t_score: confloat(ge=-10, le=5) | None = Field(
        default=None, alias="BMD_T_Score"
    )

    def to_feature_dict(self) -> dict:
        values = self.model_dump(by_alias=True, exclude_none=True)
        values.pop("Height", None)
        values.pop("Weight", None)
        if "BMI" not in values and "Body Weight" in values:
            values["BMI"] = values.pop("Body Weight")
        else:
            values.pop("Body Weight", None)
        return values

    def to_input_dict(self) -> dict:
        """Return all submitted patient fields for prediction history."""
        values = self.model_dump(by_alias=True, exclude_none=True)
        if "BMI" not in values and "Height" in values and "Weight" in values:
            height_m = values["Height"] / 100
            values["BMI"] = values["Weight"] / (height_m * height_m)
        values.pop("Body Weight", None)
        return values


def normalize_patient_frame(df):
    """Normalize the legacy BMI column without inventing clinical values."""
    result = df.copy()
    if "BMI" not in result.columns and "Body Weight" in result.columns:
        result = result.rename(columns={"Body Weight": "BMI"})
    return result