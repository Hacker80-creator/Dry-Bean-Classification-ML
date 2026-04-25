import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training accuracy', marker='o')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation accuracy', marker='o')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title('Model Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_conf_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # 1. Load and Split
    print(" Initializing Training Pipeline...")
    data = pd.read_csv('Data_sets/train_dataset.csv')
    X = data.drop(columns=['Class']) 
    y = data['Class']
    class_names = sorted(y.unique()) # Dynamically get bean names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    # 2. Scaling (The "Ruler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Training (The "Brain")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # 4. Save Artifacts
    joblib.dump(knn, 'knn_classifier_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Model and Scaler saved successfully.")

    # 5. Evaluation
    y_pred = knn.predict(X_test_scaled)
    print(f' Final Accuracy: {accuracy_score(y_test, y_pred):.2f}')

    # 6. Visuals
    plot_conf_matrix(y_test, y_pred, class_names)
    plot_learning_curve(knn, X_train_scaled, y_train)

if __name__ == "__main__":
    main()