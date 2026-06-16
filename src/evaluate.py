import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-cache")

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_CLINVAR_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
PROBABILITY_METRICS_PATH = REPORTS_DIR / "probability_metrics.txt"


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed dataset and recreate the training test split."""
    df = pd.read_csv(PROCESSED_CLINVAR_FILE)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_test, y_test


def save_classification_report(y_test: pd.Series, predictions: pd.Series) -> None:
    """Save precision, recall, and F1 metrics to a text file."""
    report = classification_report(y_test, predictions)

    CLASSIFICATION_REPORT_PATH.write_text(report)
    print(report)
    print(f"Saved classification report to: {CLASSIFICATION_REPORT_PATH}")


def save_confusion_matrix(y_test: pd.Series, predictions: pd.Series) -> None:
    """Save the confusion matrix as a PNG figure."""
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Pathogenic"],
        yticklabels=["Benign", "Pathogenic"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")


def print_probability_metrics(y_test: pd.Series, probabilities: pd.Series) -> None:
    """Print and save threshold-independent probability metrics."""
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    metrics = (
        f"ROC-AUC: {roc_auc:.3f}\n"
        f"PR-AUC: {pr_auc:.3f}\n"
    )

    PROBABILITY_METRICS_PATH.write_text(metrics)
    print(metrics, end="")
    print(f"Saved probability metrics to: {PROBABILITY_METRICS_PATH}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    save_classification_report(y_test, predictions)
    save_confusion_matrix(y_test, predictions)
    print_probability_metrics(y_test, probabilities)


if __name__ == "__main__":
    main()
