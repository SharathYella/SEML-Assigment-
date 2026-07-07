import streamlit as st
from pathlib import Path
from src.predict import DefaultPredictor

MODEL_PATH = Path(__file__).parent.resolve() / "models" / "loanguard_model.joblib"

@st.cache_resource
def load_predictor():
    if not MODEL_PATH.exists():
        st.error("Production model artifact not found at: {}".format(MODEL_PATH))
        st.stop()
    try:
        return DefaultPredictor(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to initialize predictor: {exc}")
        st.stop()

def main():
    st.set_page_config(page_title="LoanGuard Underwriting Portal", layout="wide")
    st.title("LoanGuard - Digital Lending Risk Predictor")

    predictor = load_predictor()

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
        annual_income = st.number_input("Verified Gross Annual Income ($)", min_value=0, max_value=5_000_000, value=60_000, step=1000)
        emp_length = st.slider("Employment Length (Years)", 0, 40, 5)
        
    with col2:
        loan_amount = st.number_input("Requested Principal Loan Amount ($)", min_value=100, max_value=1_000_000, value=15_000, step=100)
        dti_ratio = st.slider("Calculated Debt-To-Income (DTI) Ratio", 0.0, 1.0, 0.3, step=0.01)
        
    if st.button("Evaluate Application Risk Profiles", type="primary"):
        try:
            result = predictor.predict(
                credit_score=int(credit_score),
                annual_income=int(annual_income),
                loan_amount=int(loan_amount),
                dti_ratio=float(dti_ratio),
                emp_length=int(emp_length)
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
