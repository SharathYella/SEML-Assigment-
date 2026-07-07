from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

from src.predict import DefaultPredictor

# Resolve model path relative to the package root (one level up from src)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "loanguard_model.joblib"

app = FastAPI(title="LoanGuard Digital Lending API", version="1.0")
predictor: Optional[DefaultPredictor] = None

class LoanInput(BaseModel):
    credit_score: int = Field(..., ge=300, le=850)
    annual_income: int = Field(..., ge=0)
    loan_amount: int = Field(..., ge=100)
    dti_ratio: float = Field(..., ge=0.0, le=1.0)
    emp_length: int = Field(..., ge=0)

@app.on_event("startup")
def load_model():
    """Load the persisted model artifact at application startup.

    The app will fail to start if the model artifact is missing, which is desirable
    for fail-fast deployments where the model must be present.
    """
    global predictor
    if not MODEL_PATH.exists():
        raise RuntimeError("Model missing. Complete bootstrap compilation initialization first.")
    predictor = DefaultPredictor(MODEL_PATH)

@app.post("/predict")
def predict_default(loan: LoanInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model core uninitialized")

    try:
        decision = predictor.predict(**loan.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return {"input": loan.model_dump(), "decision": decision}

@app.get("/metrics")
def get_metrics():
    if predictor is None:
        return {}
    return predictor.metrics
