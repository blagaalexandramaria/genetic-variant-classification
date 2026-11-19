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
    roc_curve,
    precision_recall_curve,
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


MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
]


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


def save_classification_report(
    model_name: str,
    y_test: pd.Series,
    predictions,
) -> None:
    """Save a classification report for each model."""
    report = classification_report(
        y_test,
        predictions,
        target_names=["Benign", "Pathogenic"],
    )

    report_path = METRICS_DIR / f"{model_name}_classification_report.txt"
    report_path.write_text(report)

    print(f"\n--- {model_name} Classification Report ---")
    print(report)


def save_confusion_matrix(
    model_name: str,
    y_test: pd.Series,
    predictions,
) -> None:
    """Save the confusion matrix for each model."""
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

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
    plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")

    plt.tight_layout()

    figure_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved confusion matrix to: {figure_path}")


def save_probability_metrics(
    model_name: str,
    y_test: pd.Series,
    probabilities,
) -> None:
    """Save ROC-AUC and PR-AUC for each model."""
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    metrics = (
        f"ROC-AUC: {roc_auc:.4f}\n"
        f"PR-AUC: {pr_auc:.4f}\n"
    )

    metrics_path = METRICS_DIR / f"{model_name}_probability_metrics.txt"
    metrics_path.write_text(metrics)

    print(metrics)


def save_logistic_regression_coefficients(model) -> None:
    """Save Logistic Regression coefficients and their visualization."""
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

    coefficients_path = METRICS_DIR / "logistic_regression_coefficients.csv"
    coefficients_df.to_csv(coefficients_path, index=False)

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

    figure_path = FIGURES_DIR / "logistic_regression_coefficients.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved Logistic Regression coefficients to: {coefficients_path}")
    print(f"Saved coefficient plot to: {figure_path}")


def save_random_forest_importance(model) -> None:
    """Save Random Forest feature importance and its visualization."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    importance_path = METRICS_DIR / "random_forest_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    top_features = importance_df.head(15).sort_values("importance")

    plt.figure(figsize=(9, 6))

    plt.barh(
        top_features["feature"],
        top_features["importance"],
        color="steelblue",
    )

    plt.xlabel("Feature Importance")
    plt.title("Top Random Forest Features")
    plt.tight_layout()

    figure_path = FIGURES_DIR / "random_forest_feature_importance.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved Random Forest feature importance to: {importance_path}")
    print(f"Saved feature importance plot to: {figure_path}")


def save_roc_curves(y_test: pd.Series, model_probabilities: dict) -> None:
    """Compare ROC curves across models."""
    plt.figure(figsize=(7, 6))

    for model_name, probabilities in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name.replace('_', ' ').title()} (AUC={auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURES_DIR / "roc_curve_comparison.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved ROC curves to: {figure_path}")


def save_precision_recall_curves(
    y_test: pd.Series,
    model_probabilities: dict,
) -> None:
    """Compare Precision-Recall curves across models."""
    plt.figure(figsize=(7, 6))

    for model_name, probabilities in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities,
        )

        ap = average_precision_score(y_test, probabilities)

        plt.plot(
            recall,
            precision,
            label=f"{model_name.replace('_', ' ').title()} (AP={ap:.3f})",
        )

    plt.axhline(
        y=y_test.mean(),
        color="black",
        linestyle="--",
        label="Baseline",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURES_DIR / "precision_recall_comparison.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved Precision-Recall curves to: {figure_path}")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data()

    model_probabilities = {}

    for model_name in MODEL_NAMES:
        print(f"\nEvaluating {model_name}...")

        model_path = MODELS_DIR / f"{model_name}_model.pkl"
        model = joblib.load(model_path)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        model_probabilities[model_name] = probabilities

        save_classification_report(
            model_name,
            y_test,
            predictions,
        )

        save_confusion_matrix(
            model_name,
            y_test,
            predictions,
        )

        save_probability_metrics(
            model_name,
            y_test,
            probabilities,
        )

        if model_name == "logistic_regression":
            save_logistic_regression_coefficients(model)

        elif model_name == "random_forest":
            save_random_forest_importance(model)

    save_roc_curves(y_test, model_probabilities)
    save_precision_recall_curves(y_test, model_probabilities)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()