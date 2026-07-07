import os
from pathlib import Path

import streamlit as st
from src.predict import DefaultPredictor

MODEL_FILENAME = "loanguard_model.joblib"


def find_model_path() -> Path:
    """Search for the model artifact in several sensible locations.

    Order of resolution:
    1. LOANGUARD_MODEL_PATH environment variable (explicit override)
    2. <repo-root>/models/loanguard_model.joblib when running from repo root
    3. Current working directory ./models/loanguard_model.joblib
    4. One level up from this file (useful when file is inside project root)
    5. Two levels up (useful in containerized mounts like /mount/src/<repo>/)
    """
    # 1) Environment override
    env_path = os.getenv("LOANGUARD_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Candidate locations
    candidates = []
    # a) models/ relative to this file (streamlit_app.py is repo root)
    candidates.append(Path(__file__).parent.resolve() / "models" / MODEL_FILENAME)
    # b) models/ in current working directory (supports different invocation contexts)
    candidates.append(Path.cwd() / "models" / MODEL_FILENAME)
    # c) one level up from this file
    candidates.append(Path(__file__).resolve().parents[1] / "models" / MODEL_FILENAME)
    # d) two levels up (common in container mounts where app is under /mount/src/<repo>/src)
    candidates.append(Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME)

    for c in candidates:
        if c.exists():
            return c

    # If not found, return the primary candidate (repo-root) for error messaging
    return candidates[0]


@st.cache_resource
def load_predictor(model_path: Path):
    if not model_path.exists():
        # Provide a helpful message rather than raising an exception so the UI shows guidance
        st.error(
            "Production model artifact not found at: {p}\n\n".format(p=model_path)
            + "Please run `python setup.py` from the repository root to generate the synthetic dataset "
            + "and train the model, which will create models/loanguard_model.joblib.\n\n"
            + "Or set the environment variable LOANGUARD_MODEL_PATH to the absolute path of the artifact."
        )
        st.stop()

    try:
        return DefaultPredictor(model_path)
    except Exception as exc:
        st.error(f"Failed to initialize predictor: {exc}")
        st.stop()


def main():
    st.set_page_config(page_title="LoanGuard Underwriting Portal", layout="wide")
    st.title("LoanGuard - Digital Lending Risk Predictor")

    model_path = find_model_path()
    predictor = load_predictor(model_path)

    with st.sidebar:
        st.header("Model Performance Metrics")
        metrics = predictor.metrics or {}
        if not metrics:
            st.info("No metrics available in model artifact.")
        for k, v in metrics.items():
            # Expecting metrics as floats in [0,1]
            try:
                st.metric(k.replace("_", " ").title(), f"{v:.2%}")
            except Exception:
                st.metric(k.replace("_", " ").title(), str(v))

    st.markdown("### Underwriting Desktop Interface Terminal Module")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score Bureau Index", 300, 850, 650)
        annual_income = st.number_input(
            "Verified Gross Annual Income ($)", min_value=0, max_value=5_000_000, value=60_000, step=1000
        )
        emp_length = st.slider("Employment Length (Years)", 0, 40, 5)

    with col2:
        loan_amount = st.number_input(
            "Requested Principal Loan Amount ($)", min_value=100, max_value=1_000_000, value=15_000, step=100
        )
        dti_ratio = st.slider("Calculated Debt-To-Income (DTI) Ratio", 0.0, 1.0, 0.3, step=0.01)

    if st.button("Evaluate Application Risk Profiles", type="primary"):
        try:
            result = predictor.predict(
                credit_score=int(credit_score),
                annual_income=int(annual_income),
                loan_amount=int(loan_amount),
                dti_ratio=float(dti_ratio),
                emp_length=int(emp_length),
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.divider()
            prob = result.get("default_probability", 0.0)
            status = result.get("risk_status", "Unknown")
            if result.get("default_prediction") == 1:
                st.error(f"⚠️ {status} • Calculated Default Probability Score: {prob:.2%}")
            else:
                st.success(f"✅ {status} • Calculated Default Probability Score: {prob:.2%}")


if __name__ == "__main__":
    main()
