import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.pipeline import LoanDataPipeline
from typing import Dict


def train_model(data_path, model_path) -> Dict[str, float]:
    """
    Train a LogisticRegression model using the provided dataset and persist the artifact.

    The pipeline will extract, clean, and transform the data (fitting the scaler).
    The trained model, the fitted scaler, and evaluation metrics are serialized to `model_path`.

    Returns a dictionary with accuracy, precision, recall, and f1_score (all floats).
    """
    pipeline = LoanDataPipeline(data_path)
    X_train, X_test, y_train, y_test = pipeline.load()

    # Stratified target weight optimization to enforce balanced learning
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")

    # Fit model
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Persist the artifact (model + fitted scaler + metrics)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "scaler": pipeline.scaler, "metrics": metrics}
    joblib.dump(artifact, model_path)

    return metrics
