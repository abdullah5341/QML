"""Experiment configuration. Every number that shapes a result lives here."""
from pathlib import Path

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Quantum models
N_QUBITS = 4          # = number of principal components fed to the circuit
N_LAYERS = 3          # variational layers of the headline VQC (fixed a priori)
VQC_EPOCHS = 40
VQC_BATCH_SIZE = 32
VQC_LR = 0.05
VQC_SEEDS = [0, 1, 2, 3, 4]   # independent weight initialisations

# Ablation (run on a validation split carved out of the training set)
ABLATION_LAYERS = [1, 2, 3, 4]
ABLATION_SEEDS = [0, 1, 2]
ABLATION_EPOCHS = 30
ABLATION_VAL_SIZE = 0.25

# Quantum-kernel SVM grid (selected by cross-validation on the training set)
QSVM_GRID = {
    "qsvm__bandwidth": [0.05, 0.1, 0.25, 0.5, 1.0],
    "qsvm__C": [0.1, 1.0, 10.0],
}

# Extensions (run_extensions.py)
TARGET_SENSITIVITY = 0.98     # operating point the decision threshold is calibrated to
NOISE_CNOT_ERRORS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
NOISE_SHOTS = [10, 30, 100, 300, 1000, 3000]
NOISE_SHOT_LEVELS = [0.0, 0.02, 0.05]   # CNOT error rates for the shot study
NOISE_SHOT_REPEATS = 200
IG_STEPS = 64

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"   # fitted pipelines, loaded by app.py
