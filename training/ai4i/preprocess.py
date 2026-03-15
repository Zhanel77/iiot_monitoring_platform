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
]

EDGE_FEATURES = [
    "Air temperature [K]",
    "temp_diff",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "power_kw",
    "Tool wear [min]",
]

COLUMNS_TO_DROP = [
    "UDI",
    "Product ID",
    "Type",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

PROCESSED_DATA_PATH = Path("training/data/processed/predictive_maintenance_processed.csv")


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

    # remove duplicates
    df = df.drop_duplicates()

    # remove rows with missing values in required columns
    df = df.dropna(subset=RAW_REQUIRED_COLUMNS)

    # drop unused columns
    existing_cols = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=existing_cols, errors="ignore")

    return df


def add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # temperature difference
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # mechanical power in kW
    df["power_kw"] = (
        2 * np.pi * df["Torque [Nm]"] * df["Rotational speed [rpm]"] / 60.0 / 1000.0
    )

    return df


def preprocess_and_save_data(csv_path: str | Path) -> None:
    """
    Load raw data, preprocess it, and save to training/data/processed/
    """
    df = load_raw_data(csv_path)
    df = clean_data(df)
    df = add_edge_features(df)
    
    # Ensure target column is integer
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    
    # Create processed data directory if it doesn't exist
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full processed dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Also save feature list separately
    save_feature_list(PROCESSED_DATA_PATH.parent / "edge_features.json")


def get_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data and split into features and target for edge deployment
    """
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. "
            "Run preprocess_and_save_data() first."
        )
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    X = df[EDGE_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    
    return X, y


def save_feature_list(output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(EDGE_FEATURES, f, ensure_ascii=False, indent=2)
        print(f"Feature list saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Replace with your actual raw data path
    raw_data_path = "training/data/raw/ai4i2020.csv"
    
    # Preprocess and save data
    preprocess_and_save_data(raw_data_path)
    
    # Load processed data for edge deployment
    X, y = get_processed_data()
    print(f"\nEdge features shape: {X.shape}")
    print(f"Target shape: {y.shape}")