"""Central configuration: paths, column names, feature groups, hyperparameter search spaces."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DATA_FILES = {
    "cleveland": DATA_DIR / "processed.cleveland.data",
    "hungarian": DATA_DIR / "processed.hungarian.data",
    "switzerland": DATA_DIR / "processed.switzerland.data",
    "va": DATA_DIR / "processed.va.data",
}

BEST_MODEL_PATH = MODELS_DIR / "best_pipeline.joblib"

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

TARGET_RAW = "num"
TARGET = "risk_score"

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
DISCRETE_NUMERIC = ["ca"]  # median-imputed, passed as numeric

ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + DISCRETE_NUMERIC

# ---------------------------------------------------------------------------
# Risk bands
# ---------------------------------------------------------------------------
RISK_BAND_BINS = [0.0, 0.25, 0.50, 0.75, 1.001]  # 1.001 so 1.0 falls in last bin
RISK_BAND_LABELS = ["Low", "Moderate", "High", "Very High"]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------
XGB_PARAM_GRID = {
    "regressor__n_estimators": [100, 300, 500],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__subsample": [0.7, 0.8, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 1.0],
    "regressor__reg_alpha": [0, 0.01, 0.1, 1],
    "regressor__reg_lambda": [0.1, 1, 5, 10],
}

LGBM_PARAM_GRID = {
    "regressor__n_estimators": [100, 300, 500],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__subsample": [0.7, 0.8, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 1.0],
    "regressor__reg_alpha": [0, 0.01, 0.1, 1],
    "regressor__reg_lambda": [0.1, 1, 5, 10],
}

GBR_PARAM_GRID = {
    "regressor__n_estimators": [100, 300, 500],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__subsample": [0.7, 0.8, 1.0],
}

RANDOM_SEARCH_ITER = 30
