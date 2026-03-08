"""
train_model.py
Run this ONCE to train the XGBoost model on train.csv and save it to disk.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# ── 1. Load dataset ────────────────────────────────────────────────
df = pd.read_csv("train.csv")
df.drop(columns=["id"], inplace=True, errors="ignore")

print(f"Dataset shape: {df.shape}")
print(f"Target classes: {df['NObeyesdad'].unique()}")

# ── 2. Encode categorical features ────────────────────────────────
BINARY_COLS   = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
ORDINAL_COLS  = {
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
}
ONEHOT_COLS   = ["MTRANS"]

# Binary encode
label_encoders = {}
for col in BINARY_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Ordinal encode
for col, mapping in ORDINAL_COLS.items():
    df[col] = df[col].map(mapping)

# One-hot encode MTRANS
df = pd.get_dummies(df, columns=ONEHOT_COLS)

# Encode target
target_le = LabelEncoder()
df["NObeyesdad"] = target_le.fit_transform(df["NObeyesdad"])
class_names = list(target_le.classes_)
print(f"Classes after encoding: {class_names}")

# ── 3. Split ───────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != "NObeyesdad"]
X = df[feature_cols]
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ── 4. Train XGBoost ───────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── 5. Evaluate ────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 6. Save everything ─────────────────────────────────────────────
artifacts = {
    "model":          model,
    "feature_cols":   feature_cols,
    "class_names":    class_names,
    "label_encoders": label_encoders,
    "ordinal_maps":   ORDINAL_COLS,
    "binary_cols":    BINARY_COLS,
    "onehot_cols":    ONEHOT_COLS,
    "mtrans_values":  [c.replace("MTRANS_", "") for c in feature_cols if c.startswith("MTRANS_")]
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\n✅ Model saved to model_artifacts.pkl")
print(f"   Features used: {feature_cols}")
