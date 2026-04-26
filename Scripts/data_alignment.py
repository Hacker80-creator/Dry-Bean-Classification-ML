import os

import pandas as pd
from config_utils import load_config

def align_data(config_path: str = "config/benchmark_config.yaml"):
    print("LOG: Running data alignment pipeline")
    config = load_config(config_path)
    
    # Read from original dataset
    source_path = "Data_sets/Dry_Beans_Dataset.csv"
    output_path = "Data_sets/train_dataset.csv"

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source dataset not found at {source_path}")

    df = pd.read_csv(source_path)

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

    df.to_csv(output_path, index=False)
    print(f"LOG: Dataset aligned and saved to: {output_path}")
    print(f"LOG: Rows={len(df)} Columns={len(df.columns)}")


if __name__ == "__main__":
    align_data()