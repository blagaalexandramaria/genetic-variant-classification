import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from metrics import evaluate_classification_model

from config import (
    PROCESSED_CLINVAR_FILE,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)


FEATURE_SETS = {
    "structural_only": {
        "categorical": [
            "mutation_type",
        ],
        "numeric": [
            "ref_length",
            "alt_length",
            "length_difference",
        ],
    },
    "structural_plus_genomic": {
        "categorical": [
            "Chromosome",
            "GeneSymbol",
            "mutation_type",
        ],
        "numeric": [
            "ref_length",
            "alt_length",
            "length_difference",
        ],
    },
    "structural_plus_origin": {
        "categorical": [
            "mutation_type",
            "OriginSimple",
        ],
        "numeric": [
            "ref_length",
            "alt_length",
            "length_difference",
        ],
    },
    "full_features": {
        "categorical": [
            "Chromosome",
            "GeneSymbol",
            "mutation_type",
            "OriginSimple",
        ],
        "numeric": [
            "ref_length",
            "alt_length",
            "length_difference",
        ],
    },
}


def build_model(
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Build a Logistic Regression pipeline for a given feature set."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    df = pd.read_csv(PROCESSED_CLINVAR_FILE)

    y = df[TARGET_COLUMN]
    results = []

    for feature_set_name, feature_config in FEATURE_SETS.items():
        print(f"Running ablation: {feature_set_name}")

        categorical_features = feature_config["categorical"]
        numeric_features = feature_config["numeric"]

        selected_features = categorical_features + numeric_features

        X = df[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        model = build_model(
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )

        model.fit(X_train, y_train)

        metrics = evaluate_classification_model(
            model,
            X_test,
            y_test,
        )

        metrics["feature_set"] = feature_set_name
        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "feature_set",
            "accuracy",
            "precision_pathogenic",
            "recall_pathogenic",
            "f1_pathogenic",
            "roc_auc",
            "pr_auc",
        ]
    ]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = METRICS_DIR / "ablation_study.csv"
    results_df.to_csv(output_path, index=False)

    print("\nAblation study results:")
    print(results_df.to_string(index=False))

    print(f"\nSaved ablation study to: {output_path}")


if __name__ == "__main__":
    main()