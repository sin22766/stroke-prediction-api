from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader

from app.models.predict import PatientData, StrokePrediction

app = FastAPI()
header_scheme = APIKeyHeader(name="x-key")


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(input: PatientData, key: str = Depends(header_scheme)) -> StrokePrediction:
    """
    Predicts the likelihood of a stroke based on patient data.
    Requires an API key for access.

    Parameters
    ----------
    input : PatientData
        The patient data containing various health metrics and demographics.
    key : str
        The API key for authentication, provided in the request header `x-key`.

    Returns
    -------
    StrokePrediction
        The prediction result indicating whether a stroke is likely and the confidence level of the prediction.
    """

    if key != "expected_api_key":
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

    return StrokePrediction(
        stroke=False,  # Placeholder for actual prediction logic
        confidence=0.0,  # Placeholder for actual confidence score
    )
