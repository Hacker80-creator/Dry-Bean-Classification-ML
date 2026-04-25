import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    print("LOG: Initializing Model Training Pipeline")
    
    data_path = 'Data_sets/train_dataset.csv'
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'bean_knn_model.pkl')

    if not os.path.exists(data_path):
        print(f"CRITICAL: Data source not found. Run Scripts/data_alignment.py first.")
        return

    # 1. Load Processed Data
    df = pd.read_csv(data_path)
    
    # 2. Feature and Target Extraction
    y = df['Class_Encoded']
    # Removing non-predictive columns
    drop_cols = ['Class', 'Class_Encoded']
    if 'Unnamed: 0' in df.columns:
        drop_cols.append('Unnamed: 0')
    X = df.drop(columns=drop_cols)

    # 3. Data Partitioning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training (k=5 Baseline)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    # 6. Detailed Performance Metrics
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"OVERALL ACCURACY: {accuracy:.4f}")
    print("-" * 30)
    print("DETAILED CLASSIFICATION REPORT:")
    # This shows precision/recall for every bean category
    print(classification_report(y_test, y_pred))
    print("-" * 30)

    # 7. Serialization
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, model_path)
    print(f"LOG: Model artifact successfully saved to: {model_path}")

if __name__ == "__main__":
    train_model()