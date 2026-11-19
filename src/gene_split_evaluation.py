import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    PROCESSED_CLINVAR_FILE,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)

from metrics import evaluate_classification_model


CATEGORICAL_FEATURES = [
    "Chromosome",
    "mutation_type",
    "OriginSimple",
]

NUMERIC_FEATURES = [
    "ref_length",
    "alt_length",
    "length_difference",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

METRIC_COLUMNS = [
    "accuracy",
    "precision_pathogenic",
    "recall_pathogenic",
    "f1_pathogenic",
    "roc_auc",
    "pr_auc",
]


def build_model() -> Pipeline:
    # Build a Logistic Regression pipeline without GeneSymbol.

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                "passthrough",
                NUMERIC_FEATURES,
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


def evaluate_split(split_name: str,X_train: pd.DataFrame, X_test: pd.DataFrame,y_train: pd.Series,y_test: pd.Series,) -> dict:
    # Train and evaluate the model for a given data split.
    print(f"\nEvaluating: {split_name}")

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate_classification_model(model,X_test,y_test)

    metrics["split"] = split_name
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)
    metrics["train_pathogenic_ratio"] = y_train.mean()
    metrics["test_pathogenic_ratio"] = y_test.mean()

    return metrics


def main() -> None:
    df = pd.read_csv(PROCESSED_CLINVAR_FILE)

    # Keep only variants with a known gene symbol.
    df = df.dropna(subset=["GeneSymbol"]).copy()
    df = df[df["GeneSymbol"].astype(str).str.strip() != ""].copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    groups = df["GeneSymbol"]

    results = []

    # Random split used as a baseline.
    X_train_random, X_test_random, y_train_random, y_test_random = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    random_results = evaluate_split(
        "random_split",
        X_train_random,
        X_test_random,
        y_train_random,
        y_test_random,
    )

    random_results["iteration"] = 0
    results.append(random_results)

    # Repeat the gene-based split five times.
    splitter = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    for iteration, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        X_train_gene = X.iloc[train_idx]
        X_test_gene = X.iloc[test_idx]

        y_train_gene = y.iloc[train_idx]
        y_test_gene = y.iloc[test_idx]

        train_genes = set(groups.iloc[train_idx])
        test_genes = set(groups.iloc[test_idx])

        overlapping_genes = train_genes.intersection(test_genes)

        print(f"\n--- Gene Split Validation {iteration} ---")
        print(f"Training genes: {len(train_genes)}")
        print(f"Testing genes: {len(test_genes)}")
        print(f"Overlapping genes: {len(overlapping_genes)}")

        assert len(overlapping_genes) == 0, (
            "Gene leakage detected between training and testing."
        )

        gene_results = evaluate_split(
            f"gene_based_split_{iteration}",
            X_train_gene,
            X_test_gene,
            y_train_gene,
            y_test_gene,
        )

        gene_results["iteration"] = iteration
        gene_results["train_genes"] = len(train_genes)
        gene_results["test_genes"] = len(test_genes)

        results.append(gene_results)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "split",
            "iteration",
            "train_samples",
            "test_samples",
            "train_pathogenic_ratio",
            "test_pathogenic_ratio",
            *METRIC_COLUMNS,
            "train_genes",
            "test_genes",
        ]
    ]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the individual results.
    output_path = METRICS_DIR / "gene_split_comparison.csv"
    results_df.to_csv(output_path, index=False)

    # Calculate mean and standard deviation for gene-based splits.
    gene_results_df = results_df[results_df["iteration"] > 0]

    summary_df = gene_results_df[METRIC_COLUMNS].agg(["mean", "std"])

    summary_path = METRICS_DIR / "gene_split_summary.csv"
    summary_df.to_csv(summary_path)

    print("\n--- Gene Split Comparison ---")
    print(results_df.to_string(index=False))

    print("\n--- Gene Split Summary (5 iterations) ---")
    print(summary_df.to_string())

    print(f"\nSaved individual results to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()