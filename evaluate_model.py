import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd

# Load data and model
data = pd.read_csv('train_dataset.csv')
model = joblib.load('knn_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Generate predictions
X = scaler.transform(data.drop(columns=['Class']))
y_true = data['Class']
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Model Performance: Confusion Matrix')
plt.savefig('performance_chart.png')
print("Chart saved! Now upload performance_chart.png to GitHub.")