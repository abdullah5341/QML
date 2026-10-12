# QML

Hybrid Quantum-Classical Machine Learning Approach for Breast Cancer Classification Using Variational Quantum Classifier

A reproducible study comparing a 4-qubit Variational Quantum Classifier (VQC) and a quantum-kernel SVM with classical models on the Breast Cancer Wisconsin (Diagnostic) dataset, plus an interactive demo app.

**Full write-up: [paper/paper.md](paper/paper.md)**

## Results

Held-out test set, 114 patients. Positive class = malignant, so recall is clinical sensitivity.

| Model | Input | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 30 features | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| SVM (RBF) | 30 features | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| Random Forest | 30 features | 0.974 | 1.000 | 0.929 | 0.963 | 0.994 |
| XGBoost | 30 features | 0.965 | 1.000 | 0.905 | 0.950 | 0.993 |
| Logistic Regression | 4 PCA components | 0.974 | 1.000 | 0.929 | 0.963 | 0.998 |
| SVM (RBF) | 4 PCA components | 0.947 | 0.974 | 0.881 | 0.925 | 0.997 |
| Random Forest | 4 PCA components | 0.956 | 0.974 | 0.905 | 0.938 | 0.995 |
| XGBoost | 4 PCA components | 0.939 | 0.973 | 0.857 | 0.911 | 0.994 |
| QSVM (quantum kernel) | 4 qubits | 0.947 | 0.974 | 0.881 | 0.925 | 0.997 |
| **VQC** (mean ± s.d., 5 seeds) | 4 qubits | 0.956 ± 0.018 | 0.976 ± 0.025 | 0.905 ± 0.067 | 0.937 ± 0.030 | 0.997 ± 0.001 |

![Model comparison](results/figures/11_model_comparison.png)

The quantum models are competitive with classical models but not better, and the differences are within the uncertainty of a 114-sample test set. No quantum advantage is claimed. See the paper for the analysis, including why the VQC's sensitivity depends on its random initialisation and why a one-layer circuit fails.

## The quantum model

![VQC circuit](results/figures/06_vqc_circuit.png)

30 features → standardise → PCA to 4 components → rescale to angles in [0, π] → RY angle encoding on 4 qubits → 3 layers of RX-RY-RZ rotations and a CNOT ring → measure ⟨Z₀⟩ → classical sigmoid read-out. 38 trainable parameters, trained with Adam on the binary cross-entropy, simulated with PennyLane.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python run_experiment.py          # full study, about 8 minutes on a laptop CPU
python run_experiment.py --quick  # skips VQC cross-validation and the ablation, about 2 minutes
```

This regenerates everything in `results/` and the trained models in `models/`.

### Demo app

```bash
streamlit run app.py
```

Pick a held-out patient, adjust the most predictive measurements, and see the predictions of logistic regression, the QSVM and the VQC side by side, along with the rotation angles sent to each qubit. Research demonstration only, not for clinical use.

## Layout

```
qmlbc/
  config.py      every setting that shapes a result (seeds, qubits, layers, epochs)
  data.py        loading, split, preprocessing steps
  classical.py   Logistic Regression, SVM, Random Forest, XGBoost
  vqc.py         variational quantum classifier
  qkernel.py     quantum-kernel SVM
  metrics.py     accuracy, precision, recall, F1, ROC AUC
  plots.py       all figures
run_experiment.py   runs the whole study
app.py              Streamlit demo
paper/paper.md      the research paper
results/            metrics.csv, vqc_seeds.csv, vqc_ablation.csv, summary.json, figures/
models/             fitted pipelines used by the app
```

## Notes on method

- scikit-learn labels malignant as 0; this project flips it to 1 so the disease is the positive class.
- Scaling, PCA and angle rescaling are pipeline steps, fitted on training data only (also inside cross-validation), so nothing leaks from the test set.
- Classical models are run on all 30 features and on the same 4 PCA components as the quantum models, so the comparison is like for like.
- The VQC is reported as the mean over 5 random initialisations.
