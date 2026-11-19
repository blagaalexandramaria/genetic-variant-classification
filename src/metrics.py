import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

def evaluate_classification_model(model: Pipeline,X_test: pd.DataFrame,y_test: pd.Series,) -> dict[str, float]:
    # Evaluate a binary classification model with class-specific and probability metrics.
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions
    
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_pathogenic": precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall_pathogenic": recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "f1_pathogenic": f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
