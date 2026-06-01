import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config_utils import DEFAULT_CONFIG_PATH, load_config

PARAM_GRIDS = {
    "svm": {
        "model__C": [1, 10, 100],
        "model__gamma": ["scale", 0.01],
    },
    "random_forest": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 20],
        "model__max_features": ["sqrt"],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
    },
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "decision_tree": {
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
    },
}


def benchmark(config_path: str = DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    data_path = config["paths"]["data_path"]
    model_dir = config["paths"]["model_dir"]
    report_dir = config["paths"]["report_dir"]
    random_state = int(config["training"]["random_state"])
    cv_splits = int(config["training"]["cv_splits"])
    test_size = float(config["training"]["test_size"])
    n_jobs = int(config["training"].get("n_jobs", 1))
    if os.name == "nt":
        n_jobs = 1
    enable_tuning = bool(config["training"].get("enable_tuning", True))
    tuning_top_n = int(config["training"].get("tuning_top_n", 3))
    enabled_models = config["models"]["enabled"]

    if not os.path.exists(data_path):
        raise FileNotFoundError("Run Scripts/data_alignment.py first to create dataset.")

    df = pd.read_csv(data_path)
    target_col = "Class" if "Class" in df.columns else "Class_Encoded"
    drop_cols = [target_col, "Class", "Class_Encoded", "Unnamed: 0"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]

    if "Class" in df.columns:
        class_names = sorted(df["Class"].unique().tolist())
    else:
        class_names = [str(c) for c in sorted(y.unique())]

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
        "svm": Pipeline(
            [("scaler", StandardScaler()), ("model", SVC(kernel="rbf", probability=True))]
        ),
        "gaussian_nb": Pipeline(
            [("scaler", StandardScaler()), ("model", GaussianNB())]
        ),
    }
    models = {name: model_registry[name] for name in enabled_models if name in model_registry}
    if not models:
        raise ValueError("No valid models enabled in config.")

    rows = []
    best_name = None
    best_model = None
    best_holdout_acc = -1.0

    print("\n" + "=" * 80)
    print("PHASE 1: INITIAL BENCHMARK (default hyperparameters)")
    print("=" * 80)
    sys.stdout.flush()

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
                "tuned": False,
            }
        )
        print(
            f"  {name:<25} CV={cv_scores.mean():.4f}  "
            f"Holdout={holdout_acc:.4f}  F1={macro_f1:.4f}"
        )
        sys.stdout.flush()

        if holdout_acc > best_holdout_acc:
            best_holdout_acc = holdout_acc
            best_name = name
            best_model = model

    print("\n" + "=" * 80)
    print("PHASE 2: HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 80)

    tuned_results = {}
    phase1_df = pd.DataFrame(rows).sort_values("holdout_accuracy", ascending=False)
    models_to_tune = phase1_df.head(tuning_top_n)["model_name"].tolist()

    if not enable_tuning:
        print("  Tuning disabled in config (enable_tuning: false). Skipping.")
        models_to_tune = []

    for name in models_to_tune:
        if name not in models or name not in PARAM_GRIDS:
            print(f"  {name:<25} No param grid defined, skipping.")
            continue

        model = models[name]
        print(f"\n  Tuning {name}...")
        grid = GridSearchCV(
            model, PARAM_GRIDS[name], cv=cv, scoring="accuracy", n_jobs=n_jobs, verbose=0
        )
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        holdout_acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        tuned_results[name] = {
            "best_params": grid.best_params_,
            "cv_best_score": grid.best_score_,
            "holdout_accuracy": holdout_acc,
            "macro_f1": macro_f1,
            "estimator": grid.best_estimator_,
        }
        rows.append(
            {
                "model_name": f"{name}_tuned",
                "cv_accuracy_mean": round(float(grid.best_score_), 6),
                "cv_accuracy_std": 0.0,
                "holdout_accuracy": round(float(holdout_acc), 6),
                "holdout_macro_f1": round(float(macro_f1), 6),
                "tuned": True,
            }
        )
        print(f"    Best params:  {grid.best_params_}")
        print(f"    CV accuracy:  {grid.best_score_:.4f}")
        print(f"    Holdout acc:  {holdout_acc:.4f}")
        print(f"    Macro F1:     {macro_f1:.4f}")

        if holdout_acc > best_holdout_acc:
            best_holdout_acc = holdout_acc
            best_name = f"{name}_tuned"
            best_model = grid.best_estimator_

    print("\n" + "=" * 80)
    print("PHASE 3: FINAL EVALUATION")
    print("=" * 80)

    y_pred = best_model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\nBest Model: {best_name}")
    print(f"\nClassification Report:\n{report_text}")

    cm = confusion_matrix(y_test, y_pred)
    cm_dict = {"matrix": cm.tolist(), "labels": class_names}

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    result_path = os.path.join(report_dir, "benchmark_results.csv")
    best_metrics_path = os.path.join(report_dir, "best_model_metrics.json")
    confusion_path = os.path.join(report_dir, "confusion_matrix.json")
    class_report_path = os.path.join(report_dir, "classification_report.json")
    model_path = os.path.join(model_dir, "best_model.joblib")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    result_df = pd.DataFrame(rows).sort_values("holdout_accuracy", ascending=False)
    result_df.to_csv(result_path, index=False)

    best_row = result_df.iloc[0].to_dict()
    base_model_name = best_name.replace("_tuned", "") if best_name else ""
    best_metrics = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "target_column": str(target_col),
        "best_model_name": best_name,
        "best_model_metrics": {k: v for k, v in best_row.items() if k != "tuned"},
        "feature_columns": feature_cols,
        "class_names": class_names,
        "tuned_params": tuned_results.get(base_model_name, {}).get("best_params", {}),
    }

    with open(best_metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)
    with open(class_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    joblib.dump(best_model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\n[SUCCESS] Model saved to: {model_path}")
    print(f"[SUCCESS] Results saved to: {result_path}")
    print(f"[SUCCESS] Confusion matrix saved to: {confusion_path}")
    print(f"[SUCCESS] Classification report saved to: {class_report_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model benchmark from YAML config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    args = parser.parse_args()
    benchmark(config_path=args.config)
