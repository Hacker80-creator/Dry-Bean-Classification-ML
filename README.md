# Dry Bean Morphological Classification via K-Nearest Neighbors

### Project Overview
This repository implements an end-to-end machine learning pipeline designed to classify seven distinct varieties of dry beans utilizing high-dimensional morphological and geometric data. By leveraging 16 distinct features, the system achieves a high degree of precision in automated agricultural sorting.

### Technical Performance
* **Model Accuracy:** 90.0% (Generalization Accuracy)
* **Algorithm:** K-Nearest Neighbors (KNN)
* **Feature Space:** 16-Dimensional (Morphological & Geometric)
* **Data Transformation:** Standardized via `StandardScaler` to ensure Euclidean distance parity across heterogeneous feature scales.

---

### System Architecture
The project utilizes a **decoupled architecture**, strictly separating the training pipeline from the inference engine. This modularity ensures the system is ready for containerization or integration into a larger microservices ecosystem.

* **`bean_classifier.py`**: The Training Pipeline. Handles data ingestion, feature scaling, model fitting, and automated evaluation.
* **`predict.py`**: The Inference Engine. A CLI-based utility that loads serialized artifacts to perform real-time classification on new data points.
* **`evaluate_model.py`**: Evaluation module used to generate performance metrics and visualizations.
* **`knn_classifier_model.pkl`**: Serialized model artifact representing the trained state of the classifier.
* **`scaler.pkl`**: Serialized transformation parameters required to maintain data consistency during inference.

---

### Engineering Principles
1. **Model Persistence:** Leverages `joblib` for efficient object serialization, enabling rapid deployment without the overhead of retraining.
2. **Feature Normalization:** Essential for distance-based algorithms, the pipeline ensures that high-magnitude features (e.g., Area) do not disproportionately influence the classification.
3. **Reproducibility:** The project includes a standardized `requirements.txt` to ensure consistent execution environments.

---

## Installation and Deployment

1. **Environment Isolation**: Initialize a clean virtual environment:
   ```powershell
   python -m venv env
   .\env\Scripts\activate
   pip install -r requirements.txt

Model Training: To retrain the model and update artifacts:
python bean_classifier.py

Real-time Inference: To run a prediction on new data:
python predict.py

Model Evaluation
The following Confusion Matrix Heatmap visualizes the model's performance across all 7 bean classes. The strong diagonal trend confirms high precision, with minimal overlap occurring only between morphologically similar varieties.

License
This project is released under the MIT License. Feel free to use, modify, and distribute as per the license terms.

![Model Performance Chart](performance_chart.png)
