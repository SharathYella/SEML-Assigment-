# LoanGuard — Digital Lending Risk Architecture

LoanGuard is a small, production-oriented Fintech reference application demonstrating a Pipeline Pattern (Ingest → Clean → Scale → Fit → Serialize) and a Microservices-style inference loop (FastAPI prediction gateway + Streamlit frontend). The project produces a logistic credit-risk model from synthetic data and exposes an interactive underwriting portal alongside a REST API for programmatic scoring.

---

## Getting Started

### 1. Prerequisites & Installation
- Python 3.10+ is recommended.
- Clone this repository and install runtime dependencies:

```bash
# Install the required dependencies
pip install -r requirements.txt
```

The dependencies are pinned to minimum major versions to ensure reproducible behavior:
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- fastapi>=0.104.0
- uvicorn>=0.24.0
- joblib>=1.3.0
- streamlit>=1.28.0

### 2. Bootstrapping the System (Data Generation & Training)
Before running the services, generate the synthetic dataset and compile the machine learning model. The `setup.py` script executes the complete pipeline:

```bash
python setup.py
```

What this does:
- Generates 1,000 synthetic historical loan applications and writes them to `data/loan_data.csv`.
- Runs the ingestion → cleansing → scaling → train → serialize pipeline.
- Trains a balanced `LogisticRegression` model and evaluates validation metrics (Accuracy, Precision, Recall, F1-score).
- Serializes the compiled production artifact to `models/loanguard_model.joblib`.

---

## 🛠️ Running the Services

You can run the API microservice and the visual dashboard simultaneously or independently.

### Option A: Launching the REST API Gateway (FastAPI)
Start the high-performance inference engine:

```bash
uvicorn src.api:app --reload --port 8000
```

Interactive documentation (Swagger UI): http://127.0.0.1:8000/docs

Endpoints:
- `GET /metrics` — Returns evaluation metrics produced by the active production model.
- `POST /predict` — Evaluates a unique customer payload for default risk.

Example API payload (POST /predict):

```json
{
  "credit_score": 720,
  "annual_income": 85000,
  "loan_amount": 25000,
  "dti_ratio": 0.28,
  "emp_length": 6
}
```

The API performs input validation using Pydantic and returns the input together with a decision object that contains:
- `default_prediction` (0 or 1)
- `default_probability` (float, 0.0–1.0)
- `risk_status` (human readable text)

### Option B: Launching the Underwriting Client Portal (Streamlit)
Start the human-in-the-loop underwriting dashboard:

```bash
streamlit run streamlit_app.py
```

Default local URL: http://localhost:8501

Features:
- Interactive inputs (sliders and number fields) for credit score, income, loan amount, DTI, and employment length.
- Sidebar displays model metrics read directly from the persisted joblib artifact.
- One-click evaluation returns an auto-approval or review/reject recommendation based on the modeled probability.

---

## 📈 Model Features Evaluated
The model consumes five primary financial attributes:

- Credit Score Bureau Index: Range 300 - 850  
- Verified Gross Annual Income ($): Annual income in USD  
- Requested Principal Loan Amount ($): Loan amount requested  
- Calculated Debt-To-Income (DTI) Ratio: Range 0.0 - 1.0  
- Employment Length (Years): Years on current job

The synthetic dataset generation uses a seeded random generator and a custom logistic formulation to simulate realistic default probability behavior.

---

## Project Structure

```
SEML-Assignment-1/
├── requirements.txt
├── setup.py
├── streamlit_app.py
└── src/
    ├── __init__.py
    ├── api.py          # FastAPI prediction gateway
    ├── pipeline.py     # Pipeline pattern (extract -> clean -> transform -> load)
    ├── predict.py      # Predictor wrapper loading serialized artifact
    └── train.py        # Offline training & artifact serialization
```

Artifacts:
- `data/loan_data.csv` — Synthetic historical loan dataset (generated when `setup.py` runs).
- `models/loanguard_model.joblib` — Serialized artifact containing the trained model, scaler, and evaluation metrics.

---

## Notes & Recommendations
- The API is stateless and expects the model artifact to be present in `models/loanguard_model.joblib`. Run `python setup.py` first to bootstrap.
- For production deployment, containerize the API and dashboard (separate containers) and serve the API behind a load balancer/ingress. Consider secure storage for model artifacts (S3, GCS) and CI pipelines to validate and publish model versions.
- Consider monitoring model inputs and prediction distributions in production to detect data drift.

---

## Contributing
Contributions are welcome. Suggested improvements:
- Add unit tests for the pipeline and predictor.
- Add Dockerfiles and a `docker-compose` manifest for local multi-service orchestration.
- Add CI (GitHub Actions) for linting, type checks, and test runs.

---

## License
This project is provided as a reference architecture. Adopt a license appropriate for your organization or intended distribution.
