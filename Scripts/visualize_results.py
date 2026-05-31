import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, learning_curve

from config_utils import load_config


def create_visualizations(config_path: str = "config/benchmark_config.yaml"):
    config = load_config(config_path)
    report_dir = config["paths"]["report_dir"]
    chart_output_path = config["paths"]["chart_output_path"]

    result_path = os.path.join(report_dir, "benchmark_results.csv")
    confusion_path = os.path.join(report_dir, "confusion_matrix.json")

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Benchmark results not found at {result_path}")

    df = pd.read_csv(result_path)

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    has_confusion = os.path.exists(confusion_path)
    if has_confusion:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(
        "Model Benchmark Results - Dry Bean Classification",
        fontsize=16,
        fontweight="bold",
    )

    df_sorted = df.sort_values("holdout_accuracy", ascending=False)

    ax1 = axes[0, 0]
    colors = ["#2ecc71" if "_tuned" in str(n) else "#2E86AB" for n in df_sorted["model_name"]]
    bars = ax1.bar(
        range(len(df_sorted)),
        df_sorted["holdout_accuracy"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted["model_name"], rotation=45, ha="right")
    ax1.set_ylabel("Holdout Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Holdout Accuracy by Model", fontsize=12, fontweight="bold")
    y_min = max(0.80, df_sorted["holdout_accuracy"].min() - 0.03)
    ax1.set_ylim(y_min, min(1.0, df_sorted["holdout_accuracy"].max() + 0.02))
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax2 = axes[0, 1]
    ax2.scatter(
        df["cv_accuracy_mean"],
        df["holdout_accuracy"],
        s=200,
        c=df["holdout_accuracy"],
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.set_xlabel("CV Accuracy (Mean)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Holdout Accuracy", fontsize=11, fontweight="bold")
    ax2.set_title("CV vs Holdout Accuracy", fontsize=12, fontweight="bold")
    lo = min(df["cv_accuracy_mean"].min(), df["holdout_accuracy"].min()) - 0.02
    hi = max(df["cv_accuracy_mean"].max(), df["holdout_accuracy"].max()) + 0.02
    ax2.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
    ax2.legend()
    for _, row in df.iterrows():
        ax2.annotate(
            row["model_name"],
            (row["cv_accuracy_mean"], row["holdout_accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
        )

    ax3 = axes[0, 2] if has_confusion else axes[1, 0]
    bars = ax3.bar(
        range(len(df_sorted)),
        df_sorted["holdout_macro_f1"],
        color="#A23B72",
        edgecolor="black",
        linewidth=1.5,
    )
    ax3.set_xticks(range(len(df_sorted)))
    ax3.set_xticklabels(df_sorted["model_name"], rotation=45, ha="right")
    ax3.set_ylabel("Macro F1 Score", fontsize=11, fontweight="bold")
    ax3.set_title("Macro F1 Score by Model", fontsize=12, fontweight="bold")
    f1_min = max(0.80, df_sorted["holdout_macro_f1"].min() - 0.03)
    ax3.set_ylim(f1_min, min(1.0, df_sorted["holdout_macro_f1"].max() + 0.02))
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    if has_confusion:
        ax4 = axes[1, 0]
        with open(confusion_path, "r", encoding="utf-8") as f:
            cm_data = json.load(f)
        cm = np.array(cm_data["matrix"])
        labels = cm_data["labels"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax4,
        )
        ax4.set_xlabel("Predicted", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Actual", fontsize=11, fontweight="bold")
        ax4.set_title("Confusion Matrix (Best Model)", fontsize=12, fontweight="bold")
        ax4.tick_params(axis="x", rotation=45)
        ax4.tick_params(axis="y", rotation=0)

        ax5 = axes[1, 1]
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        ax5.barh(labels, per_class_acc, color="#16a085", edgecolor="black", linewidth=1.2)
        ax5.set_xlabel("Accuracy", fontsize=11, fontweight="bold")
        ax5.set_title("Per-Class Accuracy", fontsize=12, fontweight="bold")
        ax5.set_xlim(0.7, 1.05)
        for bar, acc in zip(ax5.patches, per_class_acc):
            ax5.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2.0,
                f"{acc:.2%}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax6 = axes[1, 2]
    else:
        ax6 = axes[1, 1]

    ax6.axis("tight")
    ax6.axis("off")
    table_data = []
    for _, row in df_sorted.iterrows():
        table_data.append(
            [
                row["model_name"],
                f"{row['cv_accuracy_mean']:.4f}",
                f"{row['cv_accuracy_std']:.4f}",
                f"{row['holdout_accuracy']:.4f}",
                f"{row['holdout_macro_f1']:.4f}",
            ]
        )
    table = ax6.table(
        cellText=table_data,
        colLabels=["Model", "CV Mean", "CV Std", "Holdout Acc", "Macro F1"],
        cellLoc="center",
        loc="center",
        colColours=["#2E86AB"] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    for i in range(5):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")
    for j in range(5):
        table[(1, j)].set_facecolor("#90EE90")
        table[(1, j)].set_text_props(weight="bold")
    ax6.set_title("Performance Metrics Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=300, bbox_inches="tight")
    print(f"[SUCCESS] Visualization saved to: {chart_output_path}")

    return chart_output_path


def plot_learning_curves(config_path: str = "config/benchmark_config.yaml"):
    config = load_config(config_path)
    data_path = config["paths"]["data_path"]
    report_dir = config["paths"]["report_dir"]
    random_state = int(config["training"]["random_state"])
    cv_splits = int(config["training"]["cv_splits"])
    n_jobs = int(config["training"].get("n_jobs", 1))
    if os.name == "nt":
        n_jobs = 1

    df = pd.read_csv(data_path)
    target_col = "Class" if "Class" in df.columns else "Class_Encoded"
    drop_cols = [target_col, "Class", "Class_Encoded", "Unnamed: 0"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df[target_col].values

    model = joblib.load(os.path.join(config["paths"]["model_dir"], "best_model.joblib"))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
        n_jobs=n_jobs,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue"
    )
    plt.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="green"
    )
    plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training")
    plt.plot(train_sizes, val_mean, "o-", color="green", label="Validation")
    plt.xlabel("Training Set Size", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title("Learning Curves (Best Model)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(report_dir, "learning_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[SUCCESS] Learning curves saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_visualizations()
    plot_learning_curves()
