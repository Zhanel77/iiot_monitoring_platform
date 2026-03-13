from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# XGBoost optional
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics


def main():
    # Project root: iiot_platform2/
    project_root = Path(__file__).resolve().parents[2]

    raw_data_path = project_root / "data" / "raw" / "ai4i2020.csv"
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "ml" / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_data_path}")

    print(f"Loading dataset from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    print("\nInitial dataset shape:", df.shape)
    print("Initial columns:", list(df.columns))

    # 1. Drop identifier columns
    columns_to_drop = ["UDI", "Product ID"]
    existing_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_to_drop)

    # 2. Encode Type
    if "Type" in df.columns:
        df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    # 3. Rename columns to safe names for XGBoost and cleaner pipeline
    df = df.rename(columns={
        "Type": "type",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
        "TWF": "twf",
        "HDF": "hdf",
        "PWF": "pwf",
        "OSF": "osf",
        "RNF": "rnf",
    })

    # 4. Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"\nMissing values found: {missing_before}. Dropping missing rows.")
        df = df.dropna()

    print("\nDataset shape after preprocessing:", df.shape)
    print("Processed columns:", list(df.columns))

    # 5. Save cleaned dataset
    processed_path = processed_dir / "ai4i_clean.csv"
    df.to_csv(processed_path, index=False)
    print(f"\nCleaned dataset saved to: {processed_path}")

    # 6. Define target and features
    # IMPORTANT: Do not use twf, hdf, pwf, osf, rnf as features
    # because they are failure-type flags and create data leakage.
    target_col = "machine_failure"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    feature_cols = [
        "type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]

    missing_feature_cols = [col for col in feature_cols if col not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing required feature columns: {missing_feature_cols}")

    X = df[feature_cols]
    y = df[target_col]

    print("\nTarget distribution (counts):")
    print(y.value_counts())

    print("\nTarget distribution (ratio):")
    print(y.value_counts(normalize=True))

    # 7. Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain shape:", x_train.shape)
    print("Test shape:", x_test.shape)

    # 8. Define models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        print("\nXGBoost is not installed. Skipping XGBoost.")
        print("Install it with: pip install xgboost")

    # 9. Train and evaluate
    results = {}
    best_model_name = None
    best_model = None
    best_f1 = -1.0

    print("\nTraining models...\n")

    for model_name, model in models.items():
        print(f"Training: {model_name}")
        model.fit(x_train, y_train)

        metrics = evaluate_model(model, x_test, y_test)

        results[model_name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
        }

        print(f"\nResults for {model_name}:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-score : {metrics['f1_score']:.4f}")
        if metrics["roc_auc"] is not None:
            print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

        print(metrics["report"])

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was successfully trained.")

    # 10. Save best model
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    # 11. Save feature columns for inference
    feature_columns_path = models_dir / "feature_columns.json"
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # 12. Save all model results
    results_path = models_dir / "model_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 13. Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "best_f1_score": best_f1,
        "dataset_path": str(raw_data_path),
        "processed_dataset_path": str(processed_path),
        "target_column": target_col,
        "feature_columns": feature_cols,
        "notes": "Failure-type flags (twf, hdf, pwf, osf, rnf) were excluded to avoid data leakage.",
    }

    metadata_path = models_dir / "best_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Training completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Saved best model to: {model_path}")
    print(f"Saved feature columns to: {feature_columns_path}")
    print(f"Saved results to: {results_path}")
    print(f"Saved metadata to: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()