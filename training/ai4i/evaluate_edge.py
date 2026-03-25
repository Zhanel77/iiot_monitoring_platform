import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from preprocess import build_edge_dataset


BASE_DIR = Path(__file__).resolve().parent / "training"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVAL_DIR = BASE_DIR / "evaluation" / "edge"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "edge_model.pkl"
METRICS_PATH = EVAL_DIR / "metrics.json"
PREDICTIONS_PATH = EVAL_DIR / "predictions.csv"


def evaluate_edge() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Edge model not found: {MODEL_PATH}")

    X, y = build_edge_dataset()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_type": "edge",
        "model_name": type(model).__name__,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "features": list(X_test.columns),
        "test_size": int(len(X_test)),
    }

    print("Edge classification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\nEdge metrics:")
    print(json.dumps(metrics, indent=2))

    predictions_df = X_test.reset_index(drop=True).copy()
    predictions_df["true_label"] = y_test.reset_index(drop=True)
    predictions_df["predicted_label"] = y_pred
    predictions_df["predicted_probability"] = y_prob
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved edge evaluation metrics to: {METRICS_PATH}")
    print(f"Saved edge evaluation predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    evaluate_edge()