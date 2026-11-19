import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-cache")

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_CLINVAR_FILE,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

MODELS_PATH = MODELS_DIR / "logistic_regression_model.pkl"
PERMUTATION_IMPORTANCE_PATH = METRICS_DIR / "permutation_importance.csv"
PERMUTATION_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "permutation_importance.png"

