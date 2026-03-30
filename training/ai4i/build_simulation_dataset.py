from pathlib import Path
import numpy as np
import pandas as pd

from preprocess import get_processed_dataframe, COMMON_FEATURES, TARGET_COLUMN


BASE_DIR = Path(__file__).resolve().parents[2]
SIMULATOR_DATA_DIR = BASE_DIR / "simulator"
SIMULATION_DATASET_PATH = SIMULATOR_DATA_DIR / "simulation_input.csv"

def build_simulation_dataset(
    output_path: str | Path = SIMULATION_DATASET_PATH,
    add_machine_ids: bool = True,
    num_machines: int = 5,
):
    RAW_DATA_PATH = BASE_DIR / "training" / "data" / "raw" / "ai4i2020.csv"

    df = pd.read_csv(RAW_DATA_PATH)


    COLUMNS_TO_DROP = ["UDI", "Product ID", "Type"]
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    simulation_df = df.copy()


    if add_machine_ids:
        rng = np.random.default_rng(42)
        simulation_df.insert(
            0,
            "machine_id",
            rng.integers(1, num_machines + 1, size=len(simulation_df)),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    simulation_df.to_csv(output_path, index=False)

    print(f"Simulation dataset saved to: {output_path}")
    print(f"Shape: {simulation_df.shape}")
    print(f"Columns: {list(simulation_df.columns)}")

if __name__ == "__main__":
    build_simulation_dataset()