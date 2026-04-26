import json
import os

import joblib
import pandas as pd
from config_utils import load_config


def get_feature_columns(data_path: str, meta_path: str):
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if "feature_columns" in meta:
                return meta["feature_columns"]

    data = pd.read_csv(data_path)
    drop_cols = ["Class", "Class_Encoded", "Unnamed: 0"]
    return [c for c in data.columns if c not in drop_cols]


def main():
    config = load_config()
    model_path = config["paths"]["model_path"]
    meta_path = config["paths"]["metadata_path"]
    data_path = config["paths"]["data_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Run python Scripts/bean_classifier.py first."
        )

    model = joblib.load(model_path)
    features = get_feature_columns(data_path=data_path, meta_path=meta_path)
    print(f"--- Bean Type Predictor ({len(features)} Features Required) ---")

    user_values = []
    for col in features:
        val = float(input(f"Enter {col}: "))
        user_values.append(val)

    input_df = pd.DataFrame([user_values], columns=features)
    prediction = model.predict(input_df)

    print("\n" + "=" * 30)
    print(f"RESULT: This bean is a '{prediction[0]}'")
    print("=" * 30)


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")