import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from metrics import evaluate_classification_model


from config import (
    PROCESSED_CLINVAR_FILE,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

def build_preprocessor() -> ColumnTransformer:
    # Build preprocessing for categorical and numerical features.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

def build_models() -> dict[str, Pipeline]:
    # Build all models used for comparison.
    return {
        "dummy": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DummyClassifier(strategy="most_frequent")
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

def main() -> None:
    df = pd.read_csv(PROCESSED_CLINVAR_FILE)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    models = build_models()
    results = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_classification_model(model, X_test, y_test)
        metrics["model"] = model_name
        results.append(metrics)

        model_path = MODELS_DIR / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)

        print(f"Saved model to: {model_path}")

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "model",
            "accuracy",
            "precision_pathogenic",
            "recall_pathogenic",
            "f1_pathogenic",
            "roc_auc",
            "pr_auc",
        ]
    ]

    comparison_path = METRICS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\nModel comparison:")
    print(results_df)

    print(f"\nSaved model comparison to: {comparison_path}")


if __name__ == "__main__":
    main()
