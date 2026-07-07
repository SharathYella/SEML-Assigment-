import joblib
import pandas as pd
from pathlib import Path
from src.pipeline import LoanDataPipeline

class DefaultPredictor:
    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        artifact = joblib.load(model_path)
        self.model = artifact.get("model")
        self.scaler = artifact.get("scaler")
        self.metrics = artifact.get("metrics", {}) or {}

        if self.model is None or self.scaler is None:
            raise ValueError("Model artifact is missing required components ('model' and 'scaler').")

    def predict(self, credit_score: int, annual_income: int, loan_amount: int, dti_ratio: float, emp_length: int):
        # Build DataFrame row with same feature order used during training
        row = pd.DataFrame(
            [[credit_score, annual_income, loan_amount, dti_ratio, emp_length]],
            columns=LoanDataPipeline.FEATURES
        )
        # Transform using persisted scaler (do not fit)
        try:
            scaled = self.scaler.transform(row)
        except Exception as exc:
            raise RuntimeError(f"Failed to scale input: {exc}")

        # Calculate default probability (probability of positive class)
        try:
            prob = float(self.model.predict_proba(scaled)[0, 1])
        except Exception as exc:
            raise RuntimeError(f"Model prediction failed: {exc}")

        is_default = int(prob >= 0.5)
        return {
            "default_prediction": is_default,
            "default_probability": round(prob, 4),
            "risk_status": "High Risk (Review/Reject Adverse Status)" if is_default else "Low Risk (Auto-Approve)"
        }
