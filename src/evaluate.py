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
    METRICS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
CLASSIFICATION_REPORT_PATH = METRICS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
PROBABILITY_METRICS_PATH = METRICS_DIR / "probability_metrics.txt"
COEFFICIENTS_PATH = METRICS_DIR / "logistic_regression_coefficients.csv"
COEFFICIENTS_FIGURE_PATH = FIGURES_DIR / "logistic_regression_coefficients.png"


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

def save_logistic_regression_coefficients(model) -> None:
    """Save Logistic Regression coefficients as a CSV file and bar plot."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    coefficients_df.to_csv(COEFFICIENTS_PATH, index=False)

    top_coefficients = coefficients_df.head(15).sort_values("coefficient")

    plt.figure(figsize=(9, 6))
    colors = [
        "tab:red" if value > 0 else "tab:blue"
        for value in top_coefficients["coefficient"]
    ]

    plt.barh(
        top_coefficients["feature"],
        top_coefficients["coefficient"],
        color=colors,
    )
    plt.xlabel("Coefficient")
    plt.title("Top Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig(COEFFICIENTS_FIGURE_PATH, dpi=150)
    plt.close()

    print(f"Saved Logistic Regression coefficients to: {COEFFICIENTS_PATH}")
    print(f"Saved coefficient plot to: {COEFFICIENTS_FIGURE_PATH}")

def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    save_classification_report(y_test, predictions)
    save_confusion_matrix(y_test, predictions)
    print_probability_metrics(y_test, probabilities)
    save_logistic_regression_coefficients(model)


if __name__ == "__main__":
    main()
