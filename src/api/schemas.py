"""
Request / response helpers for the assessment API.
Maps user-facing JSON keys to the model's 42-feature column names.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# API-friendly keys → exact keys expected by agents + XGBoost pipeline
_FEATURE_KEY_MAP: dict[str, str] = {
    "age": " Age (yrs)",
    "bmi": "BMI",
    "cycle_ri": "Cycle(R/I)",
    "cycle_length_days": "Cycle length(days)",
    "lh": "LH(mIU/mL)",
    "fsh": "FSH(mIU/mL)",
    "tsh": "TSH (mIU/L)",
    "hair_growth": "hair growth(Y/N)",
    "skin_darkening": "Skin darkening (Y/N)",
    "pimples": "Pimples(Y/N)",
    "weight_gain": "Weight gain(Y/N)",
    "follicle_l": "Follicle No. (L)",
    "follicle_r": "Follicle No. (R)",
}


class PatientAssessmentRequest(BaseModel):
    """
    Subset of features for the screening form. Omitted fields are imputed at inference.

    You may also send any model feature name directly (e.g. ``\"BMI\"`` or ``\" Age (yrs)\"``);
    those are merged after the mapped fields.
    """

    model_config = ConfigDict(extra="allow")

    age: float | None = Field(default=None, ge=5, le=90, description="Age (years)")
    bmi: float | None = Field(default=None, ge=10, le=70, description="Body mass index")
    cycle_ri: int | None = Field(
        default=None,
        ge=1,
        le=2,
        description="Menstrual pattern: 1 = regular, 2 = irregular",
    )
    cycle_length_days: float | None = Field(default=None, ge=10, le=120)
    lh: float | None = Field(default=None, description="LH (mIU/mL)")
    fsh: float | None = Field(default=None, description="FSH (mIU/mL)")
    tsh: float | None = Field(default=None, description="TSH (mIU/L)")
    hair_growth: int | None = Field(default=None, ge=0, le=1)
    skin_darkening: int | None = Field(default=None, ge=0, le=1)
    pimples: int | None = Field(default=None, ge=0, le=1)
    weight_gain: int | None = Field(default=None, ge=0, le=1)
    follicle_l: int | None = Field(default=None, ge=0, le=40)
    follicle_r: int | None = Field(default=None, ge=0, le=40)


def patient_dict_from_request(body: PatientAssessmentRequest) -> dict[str, Any]:
    """Map API fields to model column names; pass through any extra feature keys."""
    data = body.model_dump(exclude_none=True)
    patient: dict[str, Any] = {}
    for api_key, model_key in _FEATURE_KEY_MAP.items():
        if api_key in data:
            patient[model_key] = data[api_key]

    mapped_api = set(_FEATURE_KEY_MAP.keys())
    for key, val in data.items():
        if key in mapped_api:
            continue
        patient[key] = val

    return patient
