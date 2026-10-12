"""Interactive demo: one patient, classical and quantum models, explanation and noise.

    streamlit run app.py

Loads the pipelines saved by run_experiment.py and run_extensions.py. Research demo only.
"""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from qmlbc import config, data, explain, noise

N_SLIDERS = 6
MALIGNANT, BENIGN = "#d03b3b", "#2a78d6"
MODEL_FILES = ["logistic_regression.joblib", "qsvm.joblib", "vqc_ensemble.joblib"]

st.set_page_config(page_title="Hybrid QML breast cancer classifier", page_icon="🔬",
                   layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
.verdict {border-radius: 12px; padding: 1.1rem 1.3rem; border: 1px solid rgba(128,128,128,.25);
          border-left-width: 8px; margin-bottom: .6rem;}
.verdict .label {font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; opacity: .7;}
.verdict .value {font-size: 2rem; font-weight: 700; line-height: 1.25;}
.verdict .sub {font-size: .95rem; opacity: .85;}
.card {border-radius: 10px; padding: .9rem 1rem; border: 1px solid rgba(128,128,128,.25);
       height: 100%;}
.card .name {font-weight: 600;} .card .kind {font-size: .8rem; opacity: .65; margin-bottom: .5rem;}
.card .pred {font-size: 1.25rem; font-weight: 700;}
.pill {display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .4rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return [joblib.load(config.MODEL_DIR / f) for f in MODEL_FILES]


@st.cache_data
def load_data():
    X, y = data.load_dataset()
    X_train, X_test, _, y_test = data.split(X, y)
    top = data.target_correlations(X, y).index[:N_SLIDERS].tolist()
    return X, X_train.mean(), X_test, y_test, top


@st.cache_data
def load_csv(name, **kwargs):
    path = config.RESULTS_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def label(is_malignant):
    return "Malignant" if is_malignant else "Benign"


def colour(is_malignant):
    return MALIGNANT if is_malignant else BENIGN


def figure(name, caption):
    path = config.FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)


st.title("Hybrid quantum-classical breast cancer classifier")
st.caption("Research demonstration on the Breast Cancer Wisconsin (Diagnostic) dataset. "
           "Not a medical device and not to be used for diagnosis.")

missing = [f for f in MODEL_FILES if not (config.MODEL_DIR / f).exists()]
if missing:
    st.error("Trained models not found: " + ", ".join(missing) +
             ". Run `python run_experiment.py` and then `python run_extensions.py`.")
    st.stop()

logreg, qsvm, ensemble = load_models()
X, baseline, X_test, y_test, top_features = load_data()
calibrated = float(ensemble["vqc"].threshold)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Patient")
    idx = st.selectbox(
        "Held-out test case", X_test.index,
        format_func=lambda i: f"Case {i} · recorded {label(y_test.loc[i]).lower()}",
    )
    sample = X_test.loc[idx].copy()
    st.subheader("Adjust measurements")
    st.caption("The six measurements most correlated with malignancy. "
               "The other 24 keep this patient's recorded values.")
    for feature in top_features:
        sample[feature] = st.slider(
            feature, float(X[feature].min()), float(X[feature].max()),
            float(sample[feature]), key=f"{idx}-{feature}",
        )
    edited = not np.allclose(sample.values, X_test.loc[idx].values)
    st.divider()
    screening = st.toggle(
        "Screening mode", value=True,
        help=f"Uses the threshold calibrated for {config.TARGET_SENSITIVITY:.0%} sensitivity "
             f"({calibrated:.2f}) instead of 0.5: fewer missed cancers, more false alarms.",
    )

x = pd.DataFrame([sample])
threshold = calibrated if screening else 0.5
p_vqc = float(ensemble.predict_proba(x)[0, 1])
p_lr = float(logreg.predict_proba(x)[0, 1])
qsvm_margin = float(qsvm.decision_function(x)[0])
calls = {"VQC ensemble": p_vqc >= threshold, "Logistic regression": p_lr >= 0.5,
         "QSVM": qsvm_margin > 0}

tab_dx, tab_why, tab_quantum, tab_study = st.tabs(
    ["Diagnosis", "Why this prediction", "Quantum circuit and noise", "Study results"])

# --------------------------------------------------------------- diagnosis
with tab_dx:
    is_mal = calls["VQC ensemble"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"<div class='verdict' style='border-left-color:{colour(is_mal)}'>"
            f"<div class='label'>Quantum model (ensemble of 5 circuits, 4 qubits)</div>"
            f"<div class='value'>{label(is_mal)}</div>"
            f"<div class='sub'>Estimated probability of malignancy {p_vqc:.1%} · "
            f"decision threshold {threshold:.2f}</div></div>", unsafe_allow_html=True)
        st.progress(p_vqc, text="Probability of malignancy")
    with right:
        recorded = label(y_test.loc[idx])
        if edited:
            st.info(f"Measurements edited. The original case {idx} was recorded as "
                    f"**{recorded.lower()}**.")
        elif label(is_mal) == recorded:
            st.success(f"Matches the recorded diagnosis: **{recorded.lower()}**.")
        else:
            st.warning(f"Differs from the recorded diagnosis: **{recorded.lower()}**.")
        n_mal = sum(calls.values())
        st.write("All three models agree." if n_mal in (0, 3) else
                 f"The models disagree: {n_mal} of 3 call this case malignant.")

    st.subheader("Model by model")
    cards = [
        ("VQC ensemble", "Quantum · 4 principal components", f"P(malignant) {p_vqc:.1%}"),
        ("QSVM", "Quantum kernel · 4 principal components",
         f"SVM margin {qsvm_margin:+.2f} (positive = malignant)"),
        ("Logistic regression", "Classical · all 30 measurements", f"P(malignant) {p_lr:.1%}"),
    ]
    for col, (name, kind, detail) in zip(st.columns(3), cards):
        m = calls[name]
        col.markdown(
            f"<div class='card'><div class='name'>{name}</div><div class='kind'>{kind}</div>"
            f"<div class='pred'><span class='pill' style='background:{colour(m)}'></span>"
            f"{label(m)}</div><div class='sub'>{detail}</div></div>", unsafe_allow_html=True)

    with st.expander("All 30 measurements for this patient"):
        st.dataframe(pd.DataFrame({"This patient": sample, "Training average": baseline}),
                     width="stretch")

# ------------------------------------------------------------- explanation
with tab_why:
    attr = pd.Series(
        explain.integrated_gradients(ensemble, x.values, baseline.values, config.IG_STEPS)[0],
        index=X.columns)
    top = attr.reindex(attr.abs().sort_values(ascending=False).index).head(10)
    chart_df = pd.DataFrame({
        "Measurement": top.index, "Contribution": top.values,
        "Direction": np.where(top.values > 0, "Towards malignant", "Towards benign"),
        "This patient": sample[top.index].values, "Training average": baseline[top.index].values,
    })
    st.subheader("Which measurements drove the quantum model's prediction")
    st.caption("Contribution of each measurement to the model's score, compared with the "
               "average training patient (integrated gradients, computed with the "
               "parameter-shift rule). Bars to the right push towards malignant.")
    st.altair_chart(
        alt.Chart(chart_df).mark_bar(cornerRadius=3, height=18).encode(
            x=alt.X("Contribution:Q", title="Contribution to the score (logit units)"),
            y=alt.Y("Measurement:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=220)),
            color=alt.Color("Direction:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["Towards malignant", "Towards benign"],
                                            range=[MALIGNANT, BENIGN])),
            tooltip=["Measurement", alt.Tooltip("Contribution:Q", format="+.3f"),
                     alt.Tooltip("This patient:Q", format=".4g"),
                     alt.Tooltip("Training average:Q", format=".4g")],
        ).properties(height=340),
        width="stretch")
    grouped = explain.by_quantity(attr.values, list(X.columns)).sort_values(key=np.abs,
                                                                           ascending=False)
    lead = grouped.index[0]
    st.write(f"Grouped by cell-nucleus property, the strongest influence is **{lead}** "
             f"({'towards malignant' if grouped.iloc[0] > 0 else 'towards benign'}). "
             f"The ten bars above account for "
             f"{top.abs().sum() / attr.abs().sum():.0%} of the total attribution.")

# ------------------------------------------------------- circuit and noise
with tab_quantum:
    angles = ensemble[:-1].transform(x)
    members = ensemble["vqc"].members_
    left, right = st.columns([2, 3])
    with left:
        st.subheader("What the circuit receives")
        st.write("The 30 measurements are standardised, projected onto 4 principal components "
                 "and rescaled to rotation angles in [0, π], one per qubit.")
        st.dataframe(pd.DataFrame({"Qubit": range(config.N_QUBITS),
                                   "RY angle (rad)": np.round(angles[0], 3)}),
                     hide_index=True, width="stretch")
    with right:
        figure("06_vqc_circuit.png", "The variational quantum classifier")

    st.subheader("Try it on a noisy quantum computer")
    st.caption("The app simulates a perfect machine. Real hardware makes gate errors and "
               "reads each answer from a limited number of repeated runs (shots).")
    c1, c2 = st.columns(2)
    p_two = c1.select_slider("CNOT gate error rate", config.NOISE_CNOT_ERRORS, value=0.02)
    shots = c2.select_slider("Shots per circuit", config.NOISE_SHOTS, value=1000)
    z_clean = np.array([noise.noisy_expectation(m, angles, 0.0)[0] for m in members])
    z_noisy = np.array([noise.noisy_expectation(m, angles, p_two)[0] for m in members])
    rng = np.random.default_rng(int(idx))
    z_shots = noise.sample_shots(z_noisy, shots, rng)
    scale = np.array([m.scale_ for m in members])
    bias = np.array([m.bias_ for m in members])

    def prob(z):
        return float(1 / (1 + np.exp(-np.mean(scale * z + bias))))

    m1, m2, m3 = st.columns(3)
    m1.metric("Ideal simulator", f"{prob(z_clean):.1%}", help="P(malignant), no noise")
    m2.metric("With gate errors", f"{prob(z_noisy):.1%}",
              delta=f"{prob(z_noisy) - prob(z_clean):+.1%}", delta_color="off")
    m3.metric(f"Gate errors + {shots} shots", f"{prob(z_shots):.1%}",
              delta=f"{prob(z_shots) - prob(z_clean):+.1%}", delta_color="off")
    retained = np.mean(np.abs(z_noisy)) / max(np.mean(np.abs(z_clean)), 1e-9)
    st.write(f"Average circuit read-out ⟨Z₀⟩: **{z_clean.mean():+.3f}** ideal, "
             f"**{z_noisy.mean():+.3f}** with gate errors "
             f"({retained:.0%} of the signal strength survives). Noise pulls the read-out "
             "towards zero, so the read-out has to be recalibrated on a real device; "
             "see the study results.")

# ------------------------------------------------------------------ study
with tab_study:
    st.subheader("Held-out test set, 114 patients")
    metrics = load_csv("metrics.csv", index_col=0)
    if metrics is not None:
        cols = ["Family", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
        st.dataframe(metrics[cols].style.format(precision=3), width="stretch")
    figure("11_model_comparison.png", "Quantum and classical models on the same test patients")

    st.subheader("Reliability: ensemble and calibrated threshold")
    operating = load_csv("operating_points.csv", index_col=0)
    if operating is not None:
        st.dataframe(operating.drop(columns=["group", "ROC AUC"]).style.format(precision=3),
                     width="stretch")
    figure("13_operating_points.png", "Sensitivity and specificity at each operating point")

    st.subheader("Robustness to hardware noise")
    a, b = st.columns(2)
    with a:
        figure("14_gate_noise.png", "Gate errors: accuracy and ranking quality")
    with b:
        figure("15_shot_noise.png", "Measurement shots needed per patient")

    st.subheader("Explainability")
    a, b = st.columns(2)
    with a:
        figure("16_global_attribution.png", "What the quantum and classical models rely on")
    with b:
        figure("17_patient_attribution.png", "Explanation for one malignant test case")
    st.caption("Full analysis: paper/paper.md in the repository.")
