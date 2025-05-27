from typing import Literal

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    gender: Literal["Male", "Female", "Other"] = Field(
        title="Gender", description="Gender of the patient"
    )
    age: int = Field(gt=0, title="Age", description="Age of the patient in years")
    hypertension: bool = Field(
        default=False, title="Hypertension", description="Whether the patient has hypertension"
    )
    heart_disease: bool = Field(
        default=False, title="Heart Disease", description="Whether the patient has heart disease"
    )
    ever_married: bool = Field(
        default=False, title="Ever Married", description="Whether the patient has ever been married"
    )
    work_type: Literal["Private", "Self-employed", "Govt_job", "Children", "Never_worked"] = Field(
        title="Work Type", description="Type of work the patient is engaged in"
    )
    residence_type: Literal["Urban", "Rural"] = Field(
        title="Residence Type", description="Type of residence where the patient lives"
    )
    avg_glucose_level: float = Field(
        gt=0, title="Average Glucose Level", description="Average glucose level of the patient"
    )
    bmi: float = Field(gt=0, title="BMI", description="Body Mass Index of the patient")
    smoking_status: Literal["never smoked", "formerly smoked", "smokes", "unknown"] = Field(
        title="Smoking Status", description="Smoking status of the patient"
    )


class StrokePrediction(BaseModel):
    stroke: bool = Field(title="Stroke", description="Stroke prediction result")
    confidence: float = Field(
        gt=0, le=1, title="Confidence", description="Confidence level of the stroke prediction"
    )
