"""Classical baselines, on the full feature set and on the PCA-reduced one."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config
from .data import reduction_steps


def _estimators():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=config.SEED),
        "SVM (RBF)": SVC(kernel="rbf", random_state=config.SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=config.SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=3,
            eval_metric="logloss", random_state=config.SEED,
        ),
    }


def full_feature_models():
    """Baselines on all 30 standardised features: the best a classical model can do."""
    return {
        name: Pipeline([("scale", StandardScaler()), ("clf", est)])
        for name, est in _estimators().items()
    }


def reduced_feature_models(n_components):
    """Baselines on the same principal components the quantum models receive.

    This is the like-for-like comparison: any gap to the quantum models here is
    due to the model, not to the information thrown away by PCA.
    """
    return {
        name: Pipeline(reduction_steps(n_components) + [("clf", est)])
        for name, est in _estimators().items()
    }
