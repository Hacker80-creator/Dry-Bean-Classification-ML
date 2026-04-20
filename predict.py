import joblib
import numpy as np
import pandas as pd

# 1. Load the "Brain" and "Ruler"
model = joblib.load('knn_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Get the actual feature names from the dataset
data = pd.read_csv('train_dataset.csv')
features = data.drop(columns=['Class']).columns.tolist()

print(f"--- Bean Type Predictor ({len(features)} Features Required) ---")

try:
    user_values = []
    for col in features:
        val = float(input(f"Enter {col}: "))
        user_values.append(val)

    # 3. Create a DataFrame so the Scaler doesn't give a "Warning"
    input_df = pd.DataFrame([user_values], columns=features)

    # 4. Scale and Predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)

    print("\n" + "="*30)
    print(f"RESULT: This bean is a '{prediction[0]}'")
    print("="*30)

except ValueError as e:
    print(f"\n Error: {e}")
except Exception as e:
    print(f"\n Unexpected Error: {e}")