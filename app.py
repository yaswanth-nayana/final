"""
app.py  -  Flask REST API for Obesity Level Prediction
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ── CORS: allow ALL origins so any frontend can call this API ──────
CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type"],
    "methods": ["GET", "POST", "OPTIONS"]
}})


# ── Ensure CORS headers on EVERY response (belt + braces) ─────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Load model artifacts ───────────────────────────────────────────
ARTIFACTS_PATH = "model_artifacts.pkl"

if not os.path.exists(ARTIFACTS_PATH):
    raise FileNotFoundError(
        "model_artifacts.pkl not found!\n"
        "Run:  python train_model.py   first."
    )

with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)

model          = artifacts["model"]
feature_cols   = artifacts["feature_cols"]
class_names    = artifacts["class_names"]
label_encoders = artifacts["label_encoders"]
ordinal_maps   = artifacts["ordinal_maps"]
binary_cols    = artifacts["binary_cols"]
onehot_cols    = artifacts["onehot_cols"]
mtrans_values  = artifacts["mtrans_values"]

print(f"Model loaded. Classes: {class_names}")

COLOR_MAP = {
    "Insufficient_Weight": "#3B82F6",
    "Normal_Weight":        "#22C55E",
    "Overweight_Level_I":   "#EAB308",
    "Overweight_Level_II":  "#F97316",
    "Obesity_Type_I":       "#EF4444",
    "Obesity_Type_II":      "#DC2626",
    "Obesity_Type_III":     "#991B1B",
}

DISPLAY_NAMES = {
    "Insufficient_Weight": "Insufficient Weight",
    "Normal_Weight":        "Normal Weight",
    "Overweight_Level_I":   "Overweight Level I",
    "Overweight_Level_II":  "Overweight Level II",
    "Obesity_Type_I":       "Obesity Type I",
    "Obesity_Type_II":      "Obesity Type II",
    "Obesity_Type_III":     "Obesity Type III",
}


def preprocess(data):
    row = {}
    numeric = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in numeric:
        row[col] = float(data[col])
    for col in binary_cols:
        le = label_encoders[col]
        row[col] = int(le.transform([data[col]])[0])
    for col, mapping in ordinal_maps.items():
        row[col] = mapping.get(data[col], 0)
    mtrans_val = data.get("MTRANS", "Public_Transportation")
    for v in mtrans_values:
        row[f"MTRANS_{v}"] = 1 if mtrans_val == v else 0
    return pd.DataFrame([row], columns=feature_cols)


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>", methods=["GET"])
def static_files(path):
    return send_from_directory(".", path)


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok", "classes": class_names})


@app.route("/meta", methods=["GET", "OPTIONS"])
def meta():
    return jsonify({
        "mtrans": mtrans_values,
        "caec":   list(ordinal_maps["CAEC"].keys()),
        "calc":   list(ordinal_maps["CALC"].keys()),
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data      = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        df_input   = preprocess(data)
        pred_idx   = int(model.predict(df_input)[0])
        proba      = model.predict_proba(df_input)[0]
        pred_class = class_names[pred_idx]
        confidence = round(float(proba[pred_idx]) * 100, 2)
        bmi        = round(float(data["Weight"]) / (float(data["Height"]) ** 2), 2)
        all_probs  = {class_names[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)}

        return jsonify({
            "prediction":        pred_class,
            "display_name":      DISPLAY_NAMES.get(pred_class, pred_class),
            "confidence":        confidence,
            "bmi":               bmi,
            "color":             COLOR_MAP.get(pred_class, "#5eead4"),
            "all_probabilities": all_probs,
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # host="0.0.0.0" lets the frontend reach it from any address
    app.run(debug=True, host="0.0.0.0", port=5000)
