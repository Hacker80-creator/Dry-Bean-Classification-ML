"""
Flask API for Dry Bean Classification predictions.

Run: python app.py
Test:
  curl http://localhost:5000/health
  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"features": [...]}'
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_DIR = "models"
model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURE_COLS = metadata["feature_columns"]
CLASS_NAMES = metadata.get("class_names", [])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": metadata["best_model_name"]})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    features = np.array(data["features"]).reshape(1, -1)
    if features.shape[1] != len(FEATURE_COLS):
        return jsonify(
            {
                "error": f"Expected {len(FEATURE_COLS)} features, got {features.shape[1]}",
            }
        ), 400

    prediction = model.predict(features)[0]

    if CLASS_NAMES:
        if isinstance(prediction, (int, np.integer)):
            class_name = CLASS_NAMES[prediction]
        else:
            class_name = str(prediction)
    else:
        class_name = str(prediction)

    result = {
        "prediction": class_name,
        "model": metadata["best_model_name"],
        "features_received": len(FEATURE_COLS),
    }

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(features)[0]
        result["probabilities"] = {
            (CLASS_NAMES[i] if CLASS_NAMES else str(i)): round(float(p), 4)
            for i, p in enumerate(probas)
        }

    return jsonify(result)


@app.route("/info", methods=["GET"])
def info():
    return jsonify(
        {
            "model": metadata["best_model_name"],
            "features": FEATURE_COLS,
            "classes": CLASS_NAMES,
            "metrics": metadata.get("best_model_metrics", {}),
        }
    )


if __name__ == "__main__":
    print(f"Loaded model: {metadata['best_model_name']}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Classes: {CLASS_NAMES}")
    app.run(host="0.0.0.0", port=5000, debug=True)
