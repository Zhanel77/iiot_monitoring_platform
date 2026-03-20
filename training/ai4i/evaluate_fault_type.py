import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from preprocess import build_fault_type_dataset, COMMON_FEATURES


BASE_DIR = Path(__file__).resolve().parent / "training"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVAL_DIR = BASE_DIR / "evaluation" / "fault_type"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "fault_type_model.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "fault_type_label_encoder.pkl"
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


def evaluate_fault_type():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Fault type model not found: {MODEL_PATH}")

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")

    X, y = build_fault_type_dataset(only_failures=True)

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    y_encoded = label_encoder.transform(y)

    _, X_test, _, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    X_test_safe = X_test.rename(columns=SAFE_FEATURE_MAP)

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test_safe)
    y_prob = model.predict_proba(X_test_safe)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    class_report = classification_report(
        y_test_labels,
        y_pred_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_type": "fault_type_multiclass",
        "model_name": type(model).__name__,
        "classes": list(label_encoder.classes_),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted")), 6),
        "confusion_matrix": confusion_matrix(
            y_test_labels,
            y_pred_labels,
            labels=list(label_encoder.classes_)
        ).tolist(),
        "features": list(COMMON_FEATURES),
        "test_size": int(len(X_test)),
        "per_class": {},
    }

    for cls in label_encoder.classes_:
        if cls in class_report:
            metrics["per_class"][cls] = {
                "precision": round(float(class_report[cls]["precision"]), 6),
                "recall": round(float(class_report[cls]["recall"]), 6),
                "f1_score": round(float(class_report[cls]["f1-score"]), 6),
                "support": int(class_report[cls]["support"]),
            }

    print("Fault type classification report:")
    print(classification_report(y_test_labels, y_pred_labels, digits=4, zero_division=0))

    print("\nFault type metrics:")
    print(json.dumps(metrics, indent=2))

    proba_df = pd.DataFrame(
        y_prob,
        columns=[f"prob_{cls}" for cls in label_encoder.classes_]
    )

    predictions_df = X_test.reset_index(drop=True).copy()
    predictions_df["true_fault_type"] = y_test_labels
    predictions_df["predicted_fault_type"] = y_pred_labels
    predictions_df = pd.concat([predictions_df, proba_df], axis=1)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved fault type evaluation metrics to: {METRICS_PATH}")
    print(f"Saved fault type evaluation predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    evaluate_fault_type()