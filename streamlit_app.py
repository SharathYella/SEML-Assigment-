import os
import sys
import subprocess
from pathlib import Path

import streamlit as st
from src.predict import DefaultPredictor

MODEL_FILENAME = "loanguard_model.joblib"


def find_model_path() -> Path | None:
    """Search for the model artifact in several sensible locations.

    Strategy (in order):
    1. LOANGUARD_MODEL_PATH environment variable (explicit override)
    2. models/ relative to this file (repo root)
    3. models/ in the current working directory
    4. one and two levels up from this file
    5. scan common mount points (/mount, /mnt, /workspace, /home) for the artifact

    Returns the Path if found, otherwise None.
    """
    # 1) Environment override
    env_path = os.getenv("LOANGUARD_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path(__file__).parent.resolve() / "models" / MODEL_FILENAME,  # repo-root
        Path.cwd() / "models" / MODEL_FILENAME,  # current working dir
        Path(__file__).resolve().parents[1] / "models" / MODEL_FILENAME,  # one level up
        Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME,  # two levels up
    ]

    for c in candidates:
        if c.exists():
            return c

    # 5) Opportunistic scan of common mount roots (fast break on first found)
    search_roots = [Path("/mount"), Path("/mnt"), Path("/workspace"), Path("/home")]
    for root in search_roots:
        if not root.exists():
            continue
        try:
            # rglob can be expensive; break early on first match
            for p in root.rglob(MODEL_FILENAME):
                if p.exists():
                    return p
        except PermissionError:
            # skip locations we can't access
            continue
        except Exception:
            continue

    return None


@st.cache_resource
def load_predictor(model_path: str | Path):
    """Load and return a DefaultPredictor for the given model path.

    This function is cached by Streamlit so subsequent calls with the same
    model_path are inexpensive.
    """
    model_path = Path(model_path)
    return DefaultPredictor(model_path)


def try_generate_model(setup_path: Path) -> tuple[bool, str]:
    """Attempt to run the repository bootstrapper (setup.py) to generate the model.

    Returns (success, output).
    """
    if not setup_path.exists():
        return False, f"setup.py not found at {setup_path}"

    try:
        # Run setup.py with the current python interpreter
        res = subprocess.run([sys.executable, str(setup_path)], capture_output=True, text=True, check=False)
        out = res.stdout + "\n" + res.stderr
        return (res.returncode == 0, out)
    except Exception as exc:
        return False, str(exc)


def main():
    st.set_page_config(page_title="LoanGuard Underwriting Portal", layout="wide")
    st.title("LoanGuard 🛡️ - Digital Lending Risk Predictor")

    model_path = find_model_path()

    if model_path is None:
        st.warning(
            "Production model artifact not found.\n\n"
            "Please run `python setup.py` from the repository root to generate the synthetic dataset "
            "and train the model, which will create models/loanguard_model.joblib.\n\n"
            "Alternatively, set the environment variable LOANGUARD_MODEL_PATH to the absolute path of the artifact."
        )

        setup_py = Path(__file__).parent / "setup.py"
        if st.button("Generate model now (runs setup.py)"):
            with st.spinner("Running setup.py — this can take a minute..."):
                success, output = try_generate_model(setup_py)
            if success:
                st.success("setup.py completed successfully — attempting to load the model now.")
                # re-resolve model path
                model_path = find_model_path()
                if model_path is None:
                    st.error("Model still not found after running setup.py. See output below.")
                    st.code(output)
                    st.stop()
            else:
                st.error("Model generation failed. See setup.py output for details.")
                st.code(output)
                st.stop()
        else:
            st.info("You can either run `python setup.py` in your shell or click the button above to run it here.")
            st.stop()

    # At this point we should have a model_path
    try:
        predictor = load_predictor(str(model_path))
    except Exception as exc:
        st.error(f"Failed to initialize predictor from {model_path}: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Model Performance Metrics")
        metrics = predictor.metrics or {}
        if not metrics:
            st.info("No metrics available in model artifact.")
        for k, v in metrics.items():
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
