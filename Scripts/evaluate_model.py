import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd
import os
from config_utils import load_config

config = load_config()
DATA_PATH = config["paths"]["data_path"]
MODEL_PATH = config["paths"]["model_path"]
OUTPUT_PATH = config["paths"]["chart_output_path"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run python Scripts/benchmark_models.py first.")

data = pd.read_csv(DATA_PATH)
drop_cols = ["Class", "Class_Encoded", "Unnamed: 0"]
feature_cols = [c for c in data.columns if c not in drop_cols]
target_col = "Class" if "Class" in data.columns else "Class_Encoded"

model = joblib.load(MODEL_PATH)
X = data[feature_cols]
y_true = data[target_col]
y_pred = model.predict(X)

# Get class labels from the data
class_labels = sorted(y_true.unique())

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.title('Model Performance: Confusion Matrix')
plt.savefig(OUTPUT_PATH)
print(f"Chart saved to {OUTPUT_PATH}")