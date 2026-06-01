import os

import numpy as np
import pandas as pd

from config_utils import load_config


def align_data(config_path: str = "config/benchmark_config.yaml"):
    print("LOG: Running data alignment pipeline")
    load_config(config_path)

    source_path = "Data_sets/Dry_Beans_Dataset.csv"
    output_path = "Data_sets/train_dataset.csv"

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source dataset not found at {source_path}")

    df = pd.read_csv(source_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Class" not in df.columns and "Class_Encoded" not in df.columns:
        raise ValueError("Dataset must include 'Class' or 'Class_Encoded' target column.")

    print("LOG: Engineering features...")
    df["Area_Perimeter_Ratio"] = df["Area"] / df["Perimeter"]
    df["Log_Area"] = np.log1p(df["Area"])
    df["Shape_Complexity"] = df["Perimeter"] ** 2 / df["Area"]
    df["Axis_Difference"] = df["MajorAxisLength"] - df["MinorAxisLength"]
    df["Eccentricity_Roundness"] = df["Eccentricity"] * df["roundness"]
    print("LOG: Added 5 engineered features")

    if "Class_Encoded" not in df.columns and "Class" in df.columns:
        class_labels = sorted(df["Class"].astype(str).unique().tolist())
        mapping = {label: idx for idx, label in enumerate(class_labels)}
        df["Class_Encoded"] = df["Class"].map(mapping)
        print(f"LOG: Encoded {len(class_labels)} classes: {class_labels}")

    df.to_csv(output_path, index=False)
    print(f"LOG: Dataset aligned and saved to: {output_path}")
    print(f"LOG: Rows={len(df)}  Columns={len(df.columns)}")
    feature_cols = [c for c in df.columns if c not in ["Class", "Class_Encoded"]]
    print(f"LOG: Feature columns: {feature_cols}")


if __name__ == "__main__":
    align_data()
