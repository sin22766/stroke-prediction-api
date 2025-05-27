import os
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from sklearn.pipeline import Pipeline

from app.models.predict import PatientData, StrokePrediction
from app.util import ONNXRunner, load_preprocessor
from stroke_prediction.config import PROJ_ROOT

app = FastAPI()
header_scheme = APIKeyHeader(name="x-key")

preprocessor = load_preprocessor(PROJ_ROOT / "models" / "preprocessor.pkl")
preprocessor.set_output(transform="pandas")

COLUMNS_ORDER = [
    "ever_married",
    "work_type_Never_worked",
    "work_type_Private",
    "work_type_Self-employed",
    "work_type_children",
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes",
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

runner = ONNXRunner(
    model_path=PROJ_ROOT / "models" / "model.onnx",
)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/health")
def health():
    errors = []

    # Check model
    try:
        _ = runner.session  # Access a property to confirm it's loaded
    except Exception as e:
        errors.append(f"Model error: {str(e)}")

    # Check preprocessor
    if not preprocessor:
        errors.append("Preprocessor is not loaded")

    if not isinstance(preprocessor, Pipeline):
        errors.append("Preprocessor is not a valid Pipeline instance")
    
    if errors:
        raise HTTPException(status_code=500, detail={"status": "fail", "errors": errors})

    return {"status": "ok"}


@app.post("/predict")
def predict(inputs: List[PatientData], key: str = Depends(header_scheme)) -> List[StrokePrediction]:
    """
    Predicts the likelihood of a stroke based on patient data.
    Requires an API key for access.

    Parameters
    ----------
    inputs : List[PatientData]
        The patient data containing various health metrics and demographics.
    key : str
        The API key for authentication, provided in the request header `x-key`.

    Returns
    -------
    StrokePrediction
        The prediction result indicating whether a stroke is likely and the confidence level of the prediction.
    """

    if key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

    input_df = pd.DataFrame([input.model_dump() for input in inputs])
    input_df: pd.DataFrame = preprocessor.transform(input_df)

    input_df = input_df[COLUMNS_ORDER]

    input_df = input_df.astype("float32")
    probs = runner.predict_proba(input_df)

    predictions = probs.idxmax(axis=1)
    confidence = probs.max(axis=1)

    return [
        StrokePrediction(stroke=bool(prediction), confidence=float(confidence_value))
        for prediction, confidence_value in zip(predictions, confidence)
    ]
