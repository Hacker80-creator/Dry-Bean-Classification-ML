# Dry Bean Morphological Classification - Model Benchmarking

### Project Overview
This repository implements an end-to-end machine learning pipeline designed to classify seven distinct varieties of dry beans utilizing high-dimensional morphological and geometric data. The project features a comprehensive model benchmarking framework that evaluates multiple algorithms to identify the optimal classifier for automated agricultural sorting.

### Technical Performance
* **Best Model Accuracy:** 94.2% (Support Vector Machine with RBF Kernel)
* **Feature Space:** 21-Dimensional (Morphological & Geometric)
* **Dataset Size:** 2,500 samples with 7 bean classes
* **Evaluation Method:** 5-fold Stratified Cross-Validation + Holdout Test Set

---

## Phase2 Benchmark Results

The benchmark evaluated 6 different machine learning algorithms to identify the best performing model:

| Model | CV Accuracy (Mean) | CV Accuracy (Std) | Holdout Accuracy | Macro F1 Score |
|-------|-------------------|-------------------|------------------|----------------|
| **SVM** | 92.75% | 1.26% | **94.20%** | 0.9498 |
| Logistic Regression | 92.35% | 0.82% | 94.00% | 0.9475 |
| KNN | 91.50% | 1.39% | 94.00% | 0.9471 |
| Random Forest | 91.80% | 0.76% | 93.40% | 0.9404 |
| Gaussian NB | 90.75% | 0.94% | 91.40% | 0.9192 |
| Decision Tree | 89.80% | 1.62% | 91.00% | 0.9192 |

**Winner:** Support Vector Machine (SVM) with RBF kernel achieved the highest holdout accuracy of 94.2%, representing a 4.2% improvement over the baseline KNN model.

---

### System Architecture
The project utilizes a **modular, configuration-driven architecture** with strict separation of concerns:

* **`config/benchmark_config.yaml`**: Centralized configuration for paths, training parameters, and model selection
* **`Scripts/data_alignment.py`**: Data preprocessing pipeline that ensures dataset consistency
* **`Scripts/benchmark_models.py`**: Model benchmarking engine with cross-validation and automated evaluation
* **`Scripts/visualize_results.py`**: Visualization module for generating performance charts
* **`Scripts/config_utils.py`**: Configuration loading utilities
* **`models/best_model.joblib`**: Serialized best-performing model (SVM)
* **`models/model_metadata.json`**: Model metadata and feature information
* **`reports/benchmark_results.csv`**: Complete benchmark results for all models
* **`reports/best_model_metrics.json`**: Detailed metrics for the best model

---

### Engineering Principles
1. **Configuration-Driven Design:** All hyperparameters and paths externalized to YAML for easy experimentation
2. **Model Persistence:** Leverages `joblib` for efficient object serialization
3. **Feature Normalization:** StandardScaler applied to ensure feature parity across algorithms
4. **Reproducibility:** Fixed random states and standardized `requirements.txt` for consistent environments
5. **Modular Pipeline:** Clear separation between data preparation, training, evaluation, and visualization

---

## Installation and Deployment

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```powershell
# Step 1: Data alignment
python Scripts\data_alignment.py

# Step 2: Run model benchmark
python Scripts\benchmark_models.py

# Step 3: Generate visualizations
python Scripts\visualize_results.py
```

### 3. View Results
```powershell
# View benchmark comparison
notepad reports\benchmark_results.csv

# View best model details
notepad reports\best_model_metrics.json
```

---

## Configuration

The benchmark behavior is controlled via `config/benchmark_config.yaml`:

```yaml
paths:
  data_path: Data_sets/train_dataset.csv
  model_dir: models
  report_dir: reports

training:
  random_state: 50
  cv_splits: 5
  test_size: 0.2

models:
  enabled:
    - logistic_regression
    - knn
    - decision_tree
    - random_forest
    - svm
    - gaussian_nb
```

To customize the benchmark:
- Add/remove models from the `enabled` list
- Adjust training parameters (random_state, cv_splits, test_size)
- Modify paths for data, models, and reports

---

## Performance Visualization

![Benchmark Results](performance_chart.png)

The visualization includes:
- Holdout accuracy comparison across all models
- Cross-validation vs holdout accuracy scatter plot
- Macro F1 score comparison
- Comprehensive metrics table with best model highlighted

---

## Key Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Best Model** | **SVM (RBF Kernel)** |
| **Best Holdout Accuracy** | **94.20%** |
| **Best Macro F1 Score** | **0.9498** |
| **Dataset Size** | 2,500 samples |
| **Feature Count** | 21 morphological features |
| **Number of Classes** | 7 bean varieties |
| **Cross-Validation** | 5-fold Stratified |

---

## Project Structure

```
Karunadu Project/
├── config/
│   └── benchmark_config.yaml      # Configuration file
├── Data_sets/
│   ├── Dry_Beans_Dataset.csv      # Original dataset
│   └── train_dataset.csv           # Processed dataset
├── models/
│   ├── best_model.joblib          # Trained SVM model
│   └── model_metadata.json         # Model metadata
├── reports/
│   ├── benchmark_results.csv      # All model results
│   └── best_model_metrics.json    # Best model details
├── Scripts/
│   ├── benchmark_models.py         # Benchmarking engine
│   ├── data_alignment.py          # Data preprocessing
│   ├── visualize_results.py       # Visualization generator
│   └── config_utils.py            # Config utilities
├── requirements.txt                # Python dependencies
└── performance_chart.png          # Results visualization
```

---

## License
This project is released under the MIT License. Feel free to use, modify, and distribute as per the license terms.
