import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config_utils import load_config

def create_visualizations(config_path: str = "config/benchmark_config.yaml"):
    config = load_config(config_path)
    report_dir = config["paths"]["report_dir"]
    chart_output_path = config["paths"]["chart_output_path"]
    
    result_path = os.path.join(report_dir, "benchmark_results.csv")
    
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Benchmark results not found at {result_path}")
    
    df = pd.read_csv(result_path)
    
    # Set style for professional look
    sns.set_style("whitegrid")
    # Use a font that is available in most Linux containers.
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Benchmark Results - Dry Bean Classification', fontsize=16, fontweight='bold')
    
    # Sort by holdout accuracy for better visualization
    df_sorted = df.sort_values('holdout_accuracy', ascending=False)
    
    # Plot 1: Holdout Accuracy Comparison (Bar Chart)
    ax1 = axes[0, 0]
    bars = ax1.bar(df_sorted['model_name'], df_sorted['holdout_accuracy'], 
                   color='#2E86AB', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Holdout Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Holdout Accuracy by Model', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.85, 0.96)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: CV Accuracy vs Holdout Accuracy (Scatter)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['cv_accuracy_mean'], df['holdout_accuracy'], 
                         s=200, c=df['holdout_accuracy'], cmap='RdYlGn', 
                         edgecolors='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Cross-Validation Accuracy (Mean)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Holdout Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('CV vs Holdout Accuracy', fontsize=12, fontweight='bold')
    ax2.plot([0.88, 0.95], [0.88, 0.95], 'k--', alpha=0.5, label='y=x')
    ax2.legend()
    
    # Add model labels to scatter points
    for i, row in df.iterrows():
        ax2.annotate(row['model_name'], 
                    (row['cv_accuracy_mean'], row['holdout_accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
    
    # Plot 3: Macro F1 Score Comparison (Bar Chart)
    ax3 = axes[1, 0]
    bars = ax3.bar(df_sorted['model_name'], df_sorted['holdout_macro_f1'],
                   color='#A23B72', edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Macro F1 Score', fontsize=11, fontweight='bold')
    ax3.set_title('Macro F1 Score by Model', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.88, 0.96)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Comprehensive Metrics Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in df_sorted.iterrows():
        table_data.append([
            row['model_name'],
            f"{row['cv_accuracy_mean']:.4f}",
            f"{row['cv_accuracy_std']:.4f}",
            f"{row['holdout_accuracy']:.4f}",
            f"{row['holdout_macro_f1']:.4f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'CV Mean', 'CV Std', 'Holdout Acc', 'Macro F1'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#2E86AB'] * 5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Highlight header row
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best model row
    best_idx = df_sorted.index[0]
    for j in range(5):
        table[(best_idx + 1, j)].set_facecolor('#90EE90')
        table[(best_idx + 1, j)].set_text_props(weight='bold')
    
    ax4.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Visualization saved to: {chart_output_path}")
    
    return chart_output_path

if __name__ == "__main__":
    create_visualizations()
