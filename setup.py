from pathlib import Path
import sys

from src.pipeline import LoanDataPipeline
from src.train import train_model

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_FILE = DATA_DIR / "loan_data.csv"
MODEL_FILE = MODELS_DIR / "loanguard_model.joblib"

def main():
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic loan applications registry datastore (1000 records)...")
    try:
        LoanDataPipeline.generate_sample(DATA_FILE, n=1000)
        print(f"[SUCCESS] Saved 1000 operational records to: {DATA_FILE}")
    except Exception as exc:
        print(f"[ERROR] Failed to generate sample data: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nExecuting complete offline ML Ingestion, Cleansing and Training Pipeline...")
    try:
        metrics = train_model(DATA_FILE, MODEL_FILE)
        print(f"[SUCCESS] Persisted production model artifact to: {MODEL_FILE}")
    except Exception as exc:
        print(f"[ERROR] Training pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print formatted metrics
    print("\n--- Realized Validation Model Metrics Against Targets ---")
    try:
        print(f"Accuracy : {metrics.get('accuracy', 0.0):.4f}")
        print(f"Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"Recall   : {metrics.get('recall', 0.0):.4f}")
        print(f"F1-score : {metrics.get('f1_score', 0.0):.4f}")
    except Exception:
        print("Warning: Unable to display metrics; metrics object malformed.", file=sys.stderr)

if __name__ == "__main__":
    main()
