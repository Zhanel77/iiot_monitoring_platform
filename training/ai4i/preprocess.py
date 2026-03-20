from pathlib import Path
import json

import numpy as np
import pandas as pd


TARGET_COLUMN = "Machine failure"

RAW_REQUIRED_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET_COLUMN,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

COMMON_FEATURES = [
    "Air temperature [K]",
    "temp_diff",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "power_kw",
    "Tool wear [min]",
]

EDGE_FEATURES = COMMON_FEATURES
CLOUD_FEATURES = COMMON_FEATURES


COLUMNS_TO_DROP = [
    "UDI",
    "Product ID",
    "Type",
]

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = BASE_DIR / "data/processed/predictive_maintenance_processed.csv"
RAW_DATA_PATH = BASE_DIR / "data/raw/ai4i2020.csv"

def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()
    df = df.dropna(subset=RAW_REQUIRED_COLUMNS)

    existing_cols = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=existing_cols, errors="ignore")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    df["power_kw"] = (
        2 * np.pi * df["Torque [Nm]"] * df["Rotational speed [rpm]"] / 60.0 / 1000.0
    )

    return df


def preprocess_and_save_data(csv_path: str | Path) -> None:
    df = load_raw_data(csv_path)
    df = clean_data(df)
    df = add_engineered_features(df)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        df[col] = df[col].astype(int)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    save_feature_list(COMMON_FEATURES, PROCESSED_DATA_PATH.parent / "common_features.json")


def get_processed_dataframe() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. "
            "Run preprocess_and_save_data() first."
        )

    return pd.read_csv(PROCESSED_DATA_PATH)


def build_edge_dataset(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if csv_path is not None:
        df = load_raw_data(csv_path)
        df = clean_data(df)
        df = add_engineered_features(df)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    else:
        df = get_processed_dataframe()

    X = df[EDGE_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y


def build_cloud_dataset(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if csv_path is not None:
        df = load_raw_data(csv_path)
        df = clean_data(df)
        df = add_engineered_features(df)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    else:
        df = get_processed_dataframe()

    X = df[CLOUD_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y


def add_fault_type_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def detect_fault_type(row):
        if row["TWF"] == 1:
            return "TWF"
        elif row["HDF"] == 1:
            return "HDF"
        elif row["PWF"] == 1:
            return "PWF"
        elif row["OSF"] == 1:
            return "OSF"
        elif row["RNF"] == 1:
            return "RNF"
        else:
            return "NO_FAILURE"

    df["fault_type"] = df.apply(detect_fault_type, axis=1)
    return df


def build_fault_type_dataset(only_failures: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    df = get_processed_dataframe()
    df = add_fault_type_column(df)

    if only_failures:
        df = df[df[TARGET_COLUMN] == 1].copy()

    X = df[COMMON_FEATURES].copy()
    y = df["fault_type"].copy()

    return X, y


def save_feature_list(features: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"Feature list saved to: {output_path}")


if __name__ == "__main__":
    raw_data_path = "training/data/raw/ai4i2020.csv"

    preprocess_and_save_data(raw_data_path)

    X_edge, y_edge = build_edge_dataset()
    print(f"\nEdge features shape: {X_edge.shape}")
    print(f"Edge target shape: {y_edge.shape}")

    X_cloud, y_cloud = build_cloud_dataset()
    print(f"\nCloud features shape: {X_cloud.shape}")
    print(f"Cloud target shape: {y_cloud.shape}")

    X_fault, y_fault = build_fault_type_dataset()
    print(f"\nFault type features shape: {X_fault.shape}")
    print(f"Fault type target shape: {y_fault.shape}")