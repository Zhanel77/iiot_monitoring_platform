import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from preprocess import build_cloud_dataset


BASE_DIR = Path(__file__).resolve().parent / "training"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVAL_DIR = BASE_DIR / "evaluation" / "cloud"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "cloud_model.json"
METRICS_PATH = EVAL_DIR / "metrics.json"
PREDICTIONS_PATH = EVAL_DIR / "predictions.csv"

SAFE_FEATURE_MAP = {
    "Air temperature [K]": "air_temperature_k",
    "temp_diff": "temp_diff",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "power_kw": "power_kw",
    "Tool wear [min]": "tool_wear_min",
}


def evaluate_cloud() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Cloud model not found: {MODEL_PATH}")

    X, y = build_cloud_dataset()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_test_safe = X_test.rename(columns=SAFE_FEATURE_MAP)

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test_safe)
    y_prob = model.predict_proba(X_test_safe)[:, 1]

    metrics = {
        "model_type": "cloud",
        "model_name": type(model).__name__,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "features": list(X_test_safe.columns),
        "test_size": int(len(X_test_safe)),
    }

    print("Cloud classification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\nCloud metrics:")
    print(json.dumps(metrics, indent=2))

    predictions_df = X_test_safe.reset_index(drop=True).copy()
    predictions_df["true_label"] = y_test.reset_index(drop=True)
    predictions_df["predicted_label"] = y_pred
    predictions_df["predicted_probability"] = y_prob
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved cloud evaluation metrics to: {METRICS_PATH}")
    print(f"Saved cloud evaluation predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    evaluate_cloud()