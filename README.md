# Dry Bean Morphological Classification - Model Benchmarking

### Project Overview
This repository implements an end-to-end machine learning pipeline designed to classify seven distinct varieties of dry beans utilizing high-dimensional morphological and geometric data. The project features a comprehensive model benchmarking framework that evaluates multiple algorithms to identify the optimal classifier for automated agricultural sorting.

### Technical Performance
* **Best Model Accuracy:** 93.65% (Support Vector Machine with RBF Kernel, tuned)
* **Feature Space:** 21-Dimensional (Morphological & Geometric + 5 Engineered Features)
* **Dataset Size:** 13,611 samples with 7 bean classes
* **Evaluation Method:** 5-fold Stratified Cross-Validation + Holdout Test Set
* **Macro F1 Score:** 0.9464

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
* **`Scripts/explain_model.py`**: SHAP-based model explainability
* **`Scripts/config_utils.py`**: Configuration loading utilities
* **`app.py`**: Flask REST API for model inference
* **`models/best_model.joblib`**: Serialized best-performing model (SVM)
* **`models/model_metadata.json`**: Model metadata and feature information
* **`reports/benchmark_results.csv`**: Complete benchmark results for all models
* **`reports/best_model_metrics.json`**: Detailed metrics for the best model
* **`reports/confusion_matrix.json`**: Confusion matrix for best model
* **`reports/classification_report.json`**: Per-class precision/recall/F1 scores
* **`reports/shap_importance.png`**: SHAP feature importance visualization
* **`reports/learning_curves.png`**: Learning curves for bias/variance analysis

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

# Step 4: Model explainability (optional)
python Scripts\explain_model.py

# Step 5: Start Flask API (optional)
python app.py
```

### 3. View Results
```powershell
# View benchmark comparison
notepad reports\benchmark_results.csv

# View best model details
notepad reports\best_model_metrics.json

# View confusion matrix
notepad reports\confusion_matrix.json

# View classification report
notepad reports\classification_report.json

# View SHAP importance visualization
start reports\shap_importance.png

# View learning curves
start reports\learning_curves.png
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

### Main Performance Dashboard
The benchmark chart (`performance_chart.png`) shows all key metrics in one comprehensive view:

![Performance Chart](performance_chart.png)

The visualization includes:
- Holdout accuracy comparison across all models
- Cross-validation vs holdout accuracy scatter plot
- Macro F1 score comparison
- Confusion matrix heatmap (best model)
- Per-class accuracy bar chart
- Comprehensive metrics table with best model highlighted

### Additional Visualizations

#### Learning Curves
Bias/variance analysis showing training vs validation accuracy across dataset sizes:

![Learning Curves](reports/learning_curves.png)

#### SHAP Feature Importance
Feature importance visualization showing which features drive predictions:

![SHAP Importance](reports/shap_importance.png)

---

## Key Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Best Model** | **SVM (RBF Kernel, tuned)** |
| **Best Holdout Accuracy** | **93.65%** |
| **Best Macro F1 Score** | **0.9464** |
| **Dataset Size** | 13,611 samples |
| **Feature Count** | 21 morphological features + 5 engineered features |
| **Number of Classes** | 7 bean varieties |
| **Cross-Validation** | 5-fold Stratified |
| **Hyperparameter Tuning** | GridSearchCV on top 3 models |

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
│   ├── best_model_metrics.json    # Best model details
│   ├── confusion_matrix.json      # Confusion matrix
│   ├── classification_report.json # Per-class metrics
│   ├── shap_importance.png       # SHAP visualization
│   └── learning_curves.png        # Learning curves
├── Scripts/
│   ├── benchmark_models.py         # Benchmarking engine
│   ├── data_alignment.py          # Data preprocessing
│   ├── visualize_results.py       # Visualization generator
│   ├── explain_model.py           # SHAP explainability
│   └── config_utils.py            # Config utilities
├── notebooks/
│   └── EDA.ipynb                  # Exploratory data analysis
├── app.py                          # Flask REST API
├── requirements.txt                # Python dependencies
└── performance_chart.png          # Results visualization
```

---

## Phase3: CI/CD with Docker and Jenkins

### Overview
Phase3 adds production-grade CI/CD capabilities using Dockerized execution and Jenkins orchestration. The pipeline now runs fully without external artifact repositories and archives outputs to Jenkins and VM storage.

### Components

#### 1. Docker containerization
- **`Dockerfile`**: Builds runtime image with project scripts, config, and base dataset.
- **`docker-compose.yml`**: Local development/test orchestration.
- **`.dockerignore`**: Reduces Docker build context.

#### 2. Jenkins declarative pipeline with `vars/` modularization
- **`Jenkinsfile`**: Stage orchestration.
- **`vars/docker.groovy`**: Docker helpers (`buildImage`, `runCommand`, `removeImage`).
- **`vars/pipeline.groovy`**: Pipeline stage implementations.

#### 3. Artifact strategy (current)
- Jenkins `archiveArtifacts` is used for build outputs.
- VM/local output directory (`/tmp/bean-classification-output`) stores copied artifacts.
- No JFrog dependency in current Phase3 flow.

### Final Jenkins pipeline stages
1. **Checkout** - pull branch source.
2. **Build Docker Image** - build `bean-classification:${BUILD_NUMBER}`.
3. **Run Data Alignment** - generate `train_dataset.csv` and copy it into Jenkins workspace in the same container lifecycle.
4. **Run Model Benchmarking** - train/evaluate models and persist model/report outputs.
5. **Generate Visualizations** - produce `performance_chart.png`.
6. **Archive Artifacts to VM** - copy models/reports/chart/config to output directory and archive in Jenkins.
7. **Cleanup** - remove build image.

### Important implementation notes
- Data alignment uses the container's built-in source dataset and copies generated `train_dataset.csv` to mounted workspace path.
- `vars/docker.groovy` executes commands using `sh -c` inside container so compound commands run in-container.
- Visualization font is set to a container-safe default (`DejaVu Sans`) to avoid font warnings in Linux/Jenkins containers.

### Jenkins setup (current)
1. Install required plugins:
   - Docker Pipeline
   - Credentials Binding (for SCM credentials or other Jenkins credentials you use)
2. Create pipeline job:
   - **Pipeline script from SCM**
   - repository URL
   - branch: `usr/Jagadev/Enhancement`
   - script path: `Jenkinsfile`
3. Ensure Jenkins agent can access Docker daemon.
4. Run **Build Now** and monitor stages.

### Jenkins run evidence (Build #54)

Latest validated run completed with:
- **Status**: `Finished: SUCCESS`
- **Branch/Commit**: `usr/Jagadev/Enhancement` / `df48c52`
- **Image tag**: `bean-classification:54`
- **Jenkins UI**: Last Successful Build artifact panel confirms archived outputs.

Stage completion observed in console output:
- Checkout
- Build Docker Image
- Run Data Alignment
- Run Model Benchmarking
- Generate Visualizations
- Archive Artifacts to VM
- Cleanup

Benchmark highlights from the same run:
- **Best model**: `svm_tuned`
- **Holdout Accuracy**: `0.9365` (93.65%)
- **Macro F1 Score**: `0.9464`
- **CV Accuracy**: `0.9303`
- **Best params**: `{'model__C': 100, 'model__gamma': 'scale'}`

Archived artifacts visible in Jenkins:
- `benchmark_config.yaml`
- `best_model.joblib`
- `model_metadata.json`
- `performance_chart.png`
- `benchmark_results.csv`
- `best_model_metrics.json`
- `confusion_matrix.json`
- `classification_report.json`
- `learning_curves.png`

### Phase3 project structure

```
Karunadu Project/
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
├── .dockerignore
├── pipeline-main.groovy
├── vars/
│   ├── docker.groovy
│   └── pipeline.groovy
├── config/
│   └── benchmark_config.yaml
├── Data_sets/
├── models/
├── reports/
├── Scripts/
└── requirements.txt
```

### Benefits
- **Reproducibility**: consistent runtime through Docker image build.
- **Automation**: end-to-end CI execution in Jenkins.
- **Traceability**: build-numbered Docker image and archived artifacts per run.
- **Operational simplicity**: no external artifact repository dependency for current scope.

---

## Project Improvements

### Hyperparameter Tuning
Models are tuned using GridSearchCV with 5-fold Stratified CV. Best parameters are saved in `models/model_metadata.json`.

### Model Explainability
SHAP (SHapley Additive exPlanations) provides feature importance analysis. Run:

```powershell
python Scripts\explain_model.py
```

### API Endpoint

```powershell
python app.py
```

```powershell
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [28395, 610.29, ...]}"
```

### EDA Notebook
See `notebooks/EDA.ipynb` for exploratory data analysis including:
- Class distribution analysis
- Feature distributions and skewness
- Correlation heatmap
- Outlier detection
- Feature-target relationships

### Extended Pipeline (optional steps)

```powershell
python Scripts\data_alignment.py
python Scripts\benchmark_models.py
python Scripts\visualize_results.py
python Scripts\explain_model.py
python app.py
```

---

## License
This project is released under the MIT License. Feel free to use, modify, and distribute as per the license terms.
