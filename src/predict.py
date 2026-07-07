import joblib
from pathlib import Path
import pandas as pd
from src.pipeline import LoanDataPipeline
from typing import Any, Dict


class DefaultPredictor:
    """Loads a persisted artifact (model + scaler + metrics) and exposes a predict API.

    The artifact is expected to be a joblib file containing keys: 'model', 'scaler', and
    optionally 'metrics'. The predictor is stateless and thread-safe for inference.
    """

    def __init__(self, model_path: Any):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        try:
            artifact = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model artifact from {model_path}: {exc}")

        self.model = artifact.get("model")
        self.scaler = artifact.get("scaler")
        self.metrics = artifact.get("metrics", {}) or {}

        if self.model is None or self.scaler is None:
            raise ValueError("Model artifact is missing required components ('model' and 'scaler').")

    def predict(
        self,
        credit_score: int,
        annual_income: int,
        loan_amount: int,
        dti_ratio: float,
        emp_length: int,
    ) -> Dict[str, Any]:
        """Predict default probability for a single applicant profile.

        Returns a dictionary with:
        - default_prediction: binary 0/1
        - default_probability: float (rounded to 4 decimals)
        - risk_status: human readable risk tier
        """
        # Build a one-row DataFrame consistent with training feature order
        row = pd.DataFrame(
            [[credit_score, annual_income, loan_amount, dti_ratio, emp_length]],
            columns=LoanDataPipeline.FEATURES,
        )

        # Scale using the persisted scaler (do not fit)
        try:
            scaled = self.scaler.transform(row)
        except Exception as exc:
            raise RuntimeError(f"Failed to scale input features: {exc}")

        # Predict probability for the positive class (default)
        try:
            prob = float(self.model.predict_proba(scaled)[0, 1])
        except Exception as exc:
            raise RuntimeError(f"Model failed to produce probabilities: {exc}")

        is_default = int(prob >= 0.5)

        return {
            "default_prediction": is_default,
            "default_probability": round(prob, 4),
            "risk_status": "High Risk (Review/Reject Adverse Status)" if is_default else "Low Risk (Auto-Approve)",
        }
