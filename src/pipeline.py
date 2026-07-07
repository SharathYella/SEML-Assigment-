import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class LoanDataPipeline:
    FEATURES = ["credit_score", "annual_income", "loan_amount", "dti_ratio", "emp_length"]

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.scaler = StandardScaler()

    def extract(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Data CSV not found at {self.csv_path}")
        return pd.read_csv(self.csv_path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop_duplicates(subset=["applicant_id"]).copy()
        # Fill missing numeric feature values with median
        out[self.FEATURES] = out[self.FEATURES].fillna(out[self.FEATURES].median())
        # Ensure 'default' column exists and is integer
        if "default" not in out.columns:
            raise KeyError("Input data must contain a 'default' column")
        out["default"] = out["default"].astype(int)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Fit scaler and transform features
        out[self.FEATURES] = self.scaler.fit_transform(out[self.FEATURES])
        return out

    def load(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        prepared = self.transform(self.clean(self.extract()))
        X = prepared[self.FEATURES]
        y = prepared["default"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    @staticmethod
    def generate_sample(path, n: int = 1000, seed: int = 42):
        rng = np.random.default_rng(seed)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Generate synthetic features
        credit_score = rng.integers(300, 851, size=n)
        income = rng.integers(30000, 200001, size=n)
        loan_amt = rng.integers(1000, 50001, size=n)
        dti = rng.uniform(0.1, 0.6, size=n)
        emp_length = rng.integers(0, 21, size=n)

        # Logit formulation for default probability
        logit = (
            5.0
            - 0.015 * credit_score
            - 0.00001 * income
            + 0.00005 * loan_amt
            + 4.0 * dti
            - 0.1 * emp_length
        )
        prob = 1 / (1 + np.exp(-logit))
        default = (rng.random(n) < prob).astype(int)

        df = pd.DataFrame({
            "applicant_id": [f"APP{i:05d}" for i in range(n)],
            "credit_score": credit_score,
            "annual_income": income,
            "loan_amount": loan_amt,
            "dti_ratio": np.round(dti, 2),
            "emp_length": emp_length,
            "default": default,
        })
        df.to_csv(path, index=False)
