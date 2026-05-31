"""
Generate SHAP-based feature importance explanations for the best model.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config_utils import load_config


def explain_model(config_path: str = "config/benchmark_config.yaml"):
    config = load_config(config_path)
    data_path = config["paths"]["data_path"]
    model_dir = config["paths"]["model_dir"]
    report_dir = config["paths"]["report_dir"]

    model_path = os.path.join(model_dir, "best_model.joblib")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run Scripts/benchmark_models.py first.")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(data_path)
    feature_cols = meta["feature_columns"]
    X = df[feature_cols]

    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    print(f"Computing SHAP values for {sample_size} samples...")

    model_name = meta["best_model_name"]
    os.makedirs(report_dir, exist_ok=True)

    if "random_forest" in model_name or "decision_tree" in model_name:
        inner_model = model.named_steps.get("model", model)
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(X_sample)
    else:
        X_bg = X.sample(n=min(50, len(X)), random_state=42)
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.KernelExplainer(predict_fn, X_bg)
        shap_values = explainer.shap_values(X_sample.iloc[:50], nsamples=50)

    plt.figure(figsize=(10, 8))
    if isinstance(shap_values, list):
        avg_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        avg_shap = np.abs(shap_values).mean(axis=0)

    feat_importance = pd.Series(avg_shap, index=feature_cols).sort_values(ascending=True)
    feat_importance.plot(kind="barh", color="#2E86AB", edgecolor="black")
    plt.xlabel("Mean |SHAP value|", fontweight="bold")
    plt.title(f"Feature Importance (SHAP) - {model_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(report_dir, "shap_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[SUCCESS] SHAP importance plot saved to: {plot_path}")

    importance_dict = feat_importance.sort_values(ascending=False).to_dict()
    json_path = os.path.join(report_dir, "shap_importance.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(importance_dict, f, indent=2)

    print("\nTop 10 Most Important Features:")
    for feat, val in list(feat_importance.sort_values(ascending=False).items())[:10]:
        print(f"  {feat:<30} SHAP={val:.4f}")


if __name__ == "__main__":
    explain_model()
