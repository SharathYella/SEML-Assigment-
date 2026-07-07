import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.pipeline import LoanDataPipeline

def train_model(data_path, model_path):
    """
    Train a logistic regression model on the provided dataset and persist the artifact.
    Returns a metrics dictionary.
    """
    pipeline = LoanDataPipeline(data_path)
    X_train, X_test, y_train, y_test = pipeline.load()

    # Instantiate and fit model with balanced class weights to account for class imbalance
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # Predictions and metrics on test set
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }

    # Ensure model directory exists
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist artifact containing model, scaler and metrics
    artifact = {"model": model, "scaler": pipeline.scaler, "metrics": metrics}
    joblib.dump(artifact, model_path)

    return metrics
