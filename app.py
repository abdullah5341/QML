"""Interactive demo: one patient, three models (classical, quantum kernel, VQC).

    streamlit run app.py

Loads the pipelines saved by run_experiment.py. Research demo only.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from qmlbc import config, data

N_SLIDERS = 6

st.set_page_config(page_title="Hybrid QML breast cancer classifier", layout="wide")


@st.cache_resource
def load_models():
    names = {
        "Logistic Regression (30 features)": "logistic_regression.joblib",
        "QSVM (quantum kernel, 4 qubits)": "qsvm.joblib",
        "VQC (variational circuit, 4 qubits)": "vqc.joblib",
    }
    return {name: joblib.load(config.MODEL_DIR / f) for name, f in names.items()}


@st.cache_data
def load_data():
    X, y = data.load_dataset()
    _, X_test, _, y_test = data.split(X, y)
    top = data.target_correlations(X, y).index[:N_SLIDERS].tolist()
    return X, X_test, y_test, top


st.title("Hybrid quantum-classical breast cancer classifier")
st.caption(
    "Research demonstration on the Wisconsin Diagnostic dataset. "
    "Not a medical device and not to be used for diagnosis."
)

if not (config.MODEL_DIR / "vqc.joblib").exists():
    st.error("No trained models found. Run `python run_experiment.py` first.")
    st.stop()

models = load_models()
X, X_test, y_test, top_features = load_data()

with st.sidebar:
    st.header("Patient")
    idx = st.selectbox("Start from a held-out test case", X_test.index,
                       format_func=lambda i: f"Case {i}")
    sample = X_test.loc[idx].copy()
    st.write(f"Recorded diagnosis: **{data.CLASS_NAMES[y_test.loc[idx]]}**")
    st.subheader("Adjust the most predictive features")
    for feature in top_features:
        lo, hi = float(X[feature].min()), float(X[feature].max())
        sample[feature] = st.slider(feature, lo, hi, float(sample[feature]))

x = pd.DataFrame([sample])

st.subheader("Predictions")
for col, (name, model) in zip(st.columns(len(models)), models.items()):
    label = data.CLASS_NAMES[int(model.predict(x)[0])]
    with col:
        st.markdown(f"**{name}**")
        st.metric("Prediction", label)
        if hasattr(model, "predict_proba"):
            st.progress(float(model.predict_proba(x)[0, 1]), text="P(malignant)")
        else:
            st.write(f"SVM margin: {float(model.decision_function(x)[0]):+.2f} "
                     "(positive = malignant)")

st.subheader("What the quantum models see")
vqc = models["VQC (variational circuit, 4 qubits)"]
angles = vqc[:-1].transform(x)[0]
left, right = st.columns(2)
with left:
    st.write("The 30 features are standardised, projected onto 4 principal components "
             "and rescaled to rotation angles in [0, π], one per qubit:")
    st.dataframe(pd.DataFrame({
        "Qubit": range(config.N_QUBITS),
        "RY angle (rad)": np.round(angles, 3),
    }), hide_index=True)
    z = float(vqc["vqc"]._circuit()(vqc["vqc"].weights_, angles))
    st.write(f"Circuit read-out ⟨Z₀⟩ = **{z:+.3f}**")
with right:
    circuit_png = config.FIG_DIR / "06_vqc_circuit.png"
    if circuit_png.exists():
        st.image(str(circuit_png), caption="Variational quantum classifier")

with st.expander("All 30 feature values for this patient"):
    st.dataframe(x.T.rename(columns={x.index[0]: "value"}))
