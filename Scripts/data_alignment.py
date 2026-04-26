import os

import pandas as pd
from config_utils import load_config

def align_data(config_path: str = "config/benchmark_config.yaml"):
    print("LOG: Running data alignment pipeline")
    config = load_config(config_path)
    data_path = config["paths"]["data_path"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Remove index-like export column if present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Class" not in df.columns and "Class_Encoded" not in df.columns:
        raise ValueError("Dataset must include 'Class' or 'Class_Encoded' target column.")

    # Keep both label and encoded target for downstream scripts.
    if "Class_Encoded" not in df.columns and "Class" in df.columns:
        class_labels = sorted(df["Class"].astype(str).unique().tolist())
        mapping = {label: idx for idx, label in enumerate(class_labels)}
        df["Class_Encoded"] = df["Class"].map(mapping)

    df.to_csv(data_path, index=False)
    print(f"LOG: Dataset aligned and saved to: {data_path}")
    print(f"LOG: Rows={len(df)} Columns={len(df.columns)}")


if __name__ == "__main__":
    align_data()