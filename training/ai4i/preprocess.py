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
NORMALIZATION_CONFIG_PATH = BASE_DIR / "data/processed/normalization_config.json"
COMMON_FEATURES_PATH = BASE_DIR / "data/processed/common_features.json"


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


def _safe_minmax_scale(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    denom = max_val - min_val
    if abs(denom) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    scaled = (series - min_val) / denom
    return scaled.clip(0.0, 1.0)


def _safe_ratio_scale(series: pd.Series, max_val: float) -> pd.Series:
    if abs(max_val) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    scaled = series / max_val
    return scaled.clip(0.0, 1.0)


def build_normalization_config_from_df(df: pd.DataFrame) -> dict:
    """
    Fallback normalization config built from training data.
    We use robust percentile-based ranges instead of raw min/max
    to reduce sensitivity to outliers.
    """
    return {
        "Air temperature [K]": {
            "mode": "minmax",
            "min": float(df["Air temperature [K]"].quantile(0.05)),
            "max": float(df["Air temperature [K]"].quantile(0.95)),
        },
        "temp_diff": {
            "mode": "minmax",
            "min": float(df["temp_diff"].quantile(0.05)),
            "max": float(df["temp_diff"].quantile(0.95)),
        },
        "Rotational speed [rpm]": {
            "mode": "ratio",
            "max": float(df["Rotational speed [rpm]"].quantile(0.95)),
        },
        "Torque [Nm]": {
            "mode": "ratio",
            "max": float(df["Torque [Nm]"].quantile(0.95)),
        },
        "power_kw": {
            "mode": "ratio",
            "max": float(df["power_kw"].quantile(0.95)),
        },
        "Tool wear [min]": {
            "mode": "ratio",
            "max": float(df["Tool wear [min]"].quantile(0.95)),
        },
    }


def normalize_features(
    df: pd.DataFrame,
    normalization_config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    if normalization_config is None:
        normalization_config = build_normalization_config_from_df(df)

    for feature in COMMON_FEATURES:
        if feature not in normalization_config:
            raise ValueError(f"Missing normalization config for feature: {feature}")

        cfg = normalization_config[feature]
        mode = cfg.get("mode")

        if mode == "minmax":
            df[feature] = _safe_minmax_scale(
                df[feature],
                float(cfg["min"]),
                float(cfg["max"]),
            )
        elif mode == "ratio":
            df[feature] = _safe_ratio_scale(
                df[feature],
                float(cfg["max"]),
            )
        else:
            raise ValueError(
                f"Unsupported normalization mode '{mode}' for feature '{feature}'"
            )

    return df, normalization_config


def save_json(data: dict | list, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(input_path: str | Path) -> dict:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_feature_list(features: list[str], output_path: str | Path) -> None:
    save_json(features, output_path)
    print(f"Feature list saved to: {output_path}")


def preprocess_and_save_data(
    csv_path: str | Path,
    normalization_config: dict | None = None,
) -> None:
    df = load_raw_data(csv_path)
    df = clean_data(df)
    df = add_engineered_features(df)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        df[col] = df[col].astype(int)

    df, used_config = normalize_features(df, normalization_config=normalization_config)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    save_feature_list(COMMON_FEATURES, COMMON_FEATURES_PATH)
    save_json(used_config, NORMALIZATION_CONFIG_PATH)

    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Normalization config saved to: {NORMALIZATION_CONFIG_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def get_processed_dataframe() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. "
            f"Run preprocess_and_save_data() first."
        )

    return pd.read_csv(PROCESSED_DATA_PATH)


def _prepare_dataset_from_raw(
    csv_path: str | Path,
    normalization_config: dict | None = None,
) -> pd.DataFrame:
    df = load_raw_data(csv_path)
    df = clean_data(df)
    df = add_engineered_features(df)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        df[col] = df[col].astype(int)

    if normalization_config is None and NORMALIZATION_CONFIG_PATH.exists():
        normalization_config = load_json(NORMALIZATION_CONFIG_PATH)

    df, _ = normalize_features(df, normalization_config=normalization_config)
    return df


def build_edge_dataset(
    csv_path: str | Path | None = None,
    normalization_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if csv_path is not None:
        df = _prepare_dataset_from_raw(
            csv_path=csv_path,
            normalization_config=normalization_config,
        )
    else:
        df = get_processed_dataframe()

    X = df[EDGE_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def build_cloud_dataset(
    csv_path: str | Path | None = None,
    normalization_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if csv_path is not None:
        df = _prepare_dataset_from_raw(
            csv_path=csv_path,
            normalization_config=normalization_config,
        )
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
        if row["HDF"] == 1:
            return "HDF"
        if row["PWF"] == 1:
            return "PWF"
        if row["OSF"] == 1:
            return "OSF"
        if row["RNF"] == 1:
            return "RNF"
        return "NO_FAILURE"

    df["fault_type"] = df.apply(detect_fault_type, axis=1)
    return df


def build_fault_type_dataset(
    only_failures: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    df = get_processed_dataframe()
    df = add_fault_type_column(df)

    if only_failures:
        df = df[df[TARGET_COLUMN] == 1].copy()
        df = df[df["fault_type"] != "NO_FAILURE"].copy()

    X = df[COMMON_FEATURES].copy()
    y = df["fault_type"].copy()
    return X, y
  


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