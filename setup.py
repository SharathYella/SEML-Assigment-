from pathlib import Path
from src.pipeline import LoanDataPipeline
from src.train import train_model

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "loan_data.csv"
MODEL = ROOT / "models" / "loanguard_model.joblib"

def main():
    print("Generating synthetic loan applications registry datastore...")
    LoanDataPipeline.generate_sample(DATA, n=1000)
    print(f"[SUCCESS] Saved 1000 operational records directly into: {DATA}")

    print("Executing complete offline ML Ingestion, Cleansing and Scaling Pipeline...")
    metrics = train_model(DATA, MODEL)
    print(f"[SUCCESS] Persisted comprehensive production model dictionary to: {MODEL}")

    print("\n--- Realized Validation Model Metrics Against Targets ---")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")

if __name__ == "__main__":
    main()
