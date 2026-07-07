from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

from src.predict import DefaultPredictor

# Resolve MODEL_PATH relative to the package root
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "loanguard_model.joblib"

app = FastAPI(title="LoanGuard Digital Lending API", version="1.0")
predictor: Optional[DefaultPredictor] = None

class LoanInput(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, description="FICO-like credit score")
    annual_income: int = Field(..., ge=0, description="Verified gross annual income in USD")
    loan_amount: int = Field(..., ge=100, description="Requested principal amount in USD")
    dti_ratio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-income ratio (0.0-1.0)")
    emp_length: int = Field(..., ge=0, description="Employment length in years")

@app.on_event("startup")
def load_model():
    global predictor
    if not MODEL_PATH.exists():
        # Fail-fast in many production apps; here we raise an exception so FastAPI startup fails clearly
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Please run setup to generate model.")
    predictor = DefaultPredictor(MODEL_PATH)

@app.post("/predict")
def predict_default(loan: LoanInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model core uninitialized")
    try:
        decision = predictor.predict(
            credit_score=loan.credit_score,
            annual_income=loan.annual_income,
            loan_amount=loan.loan_amount,
            dti_ratio=loan.dti_ratio,
            emp_length=loan.emp_length
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
    return {"input": loan.model_dump(), "decision": decision}

@app.get("/metrics")
def get_metrics():
    if predictor is None:
        return {}
    return predictor.metrics
