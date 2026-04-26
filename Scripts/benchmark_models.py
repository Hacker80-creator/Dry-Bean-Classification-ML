import json
import os
import argparse
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from config_utils import DEFAULT_CONFIG_PATH, load_config


def benchmark(config_path: str = DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    data_path = config["paths"]["data_path"]
    model_dir = config["paths"]["model_dir"]
    report_dir = config["paths"]["report_dir"]
    random_state = int(config["training"]["random_state"])
    cv_splits = int(config["training"]["cv_splits"])
    test_size = float(config["training"]["test_size"])
    enabled_models = config["models"]["enabled"]

    if not os.path.exists(data_path):
        raise FileNotFoundError("Run Scripts/data_alignment.py first to create dataset.")

    df = pd.read_csv(data_path)
    target_col = "Class" if "Class" in df.columns else "Class_Encoded"
    drop_cols = [target_col, "Class", "Class_Encoded", "Unnamed: 0"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    model_registry = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "knn": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
        ),
        "decision_tree": Pipeline(
            [("model", DecisionTreeClassifier(random_state=random_state))]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300, random_state=random_state, n_jobs=1
                    ),
                )
            ]
        ),
        "svm": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf"))]),
        "gaussian_nb": Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
    }
    models = {name: model_registry[name] for name in enabled_models if name in model_registry}
    if not models:
        raise ValueError("No valid models enabled in config.")

    rows = []
    best_name = None
    best_model = None
    best_holdout_acc = -1.0

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        holdout_acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        rows.append(
            {
                "model_name": name,
                "cv_accuracy_mean": round(float(cv_scores.mean()), 6),
                "cv_accuracy_std": round(float(cv_scores.std()), 6),
                "holdout_accuracy": round(float(holdout_acc), 6),
                "holdout_macro_f1": round(float(macro_f1), 6),
            }
        )

        if holdout_acc > best_holdout_acc:
            best_holdout_acc = holdout_acc
            best_name = name
            best_model = model

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    result_path = os.path.join(report_dir, "benchmark_results.csv")
    best_metrics_path = os.path.join(report_dir, "best_model_metrics.json")
    model_path = os.path.join(model_dir, "best_model.joblib")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    result_df = pd.DataFrame(rows).sort_values("holdout_accuracy", ascending=False)
    result_df.to_csv(result_path, index=False)

    best_row = result_df.iloc[0].to_dict()
    best_metrics = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "target_column": str(target_col),
        "best_model_name": best_name,
        "best_model_metrics": best_row,
        "feature_columns": feature_cols,
    }

    with open(best_metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    joblib.dump(best_model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print("Benchmark complete.")
    print(f"Best model: {best_name}")
    print(f"Holdout accuracy: {best_holdout_acc:.4f}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model benchmark from YAML config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    args = parser.parse_args()
    benchmark(config_path=args.config)
