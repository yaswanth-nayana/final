# ObesityIQ — XGBoost Obesity Predictor

## 📁 Folder Structure

```
obesity-predictor/
├── backend/
│   ├── train_model.py       ← Step 1: Train XGBoost on train.csv
│   ├── app.py               ← Step 2: Flask REST API
│   └── requirements.txt     ← Python dependencies
├── frontend/
│   ├── index.html           ← UI
│   ├── style.css            ← Dark theme styles
│   └── app.js               ← API + rendering logic
└── README.md
```

## ⚡ Quick Start

### Step 1 — Backend Setup

```bash
cd obesity-predictor/backend

# Create virtual env
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy your dataset here
cp /path/to/train.csv .

# Train the model (one-time)
python train_model.py
# → Creates model_artifacts.pkl

# Start the API server
python app.py
# → Running on http://localhost:5000
```

### Step 2 — Frontend

```bash
cd obesity-predictor/frontend
python -m http.server 8080
# → Open http://localhost:8080
```

## 🔌 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | /health | Server status |
| GET | /meta | Returns dropdown options |
| POST | /predict | Returns prediction JSON |

### POST /predict — Request body

```json
{
  "Gender": "Male",
  "Age": 24.4,
  "Height": 1.70,
  "Weight": 81.6,
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "FCVC": 2.0,
  "NCP": 3.0,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2.0,
  "SCC": "no",
  "FAF": 1.0,
  "TUE": 1.0,
  "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}
```

### Response

```json
{
  "prediction": "Overweight_Level_II",
  "display_name": "Overweight Level II",
  "confidence": 87.4,
  "bmi": 28.2,
  "color": "#F97316",
  "all_probabilities": { ... }
}
```

## 🎯 Target Classes

| Class | Description |
|---|---|
| Insufficient_Weight | BMI < 18.5 |
| Normal_Weight | BMI 18.5–24.9 |
| Overweight_Level_I | BMI 25–27.4 |
| Overweight_Level_II | BMI 27.5–29.9 |
| Obesity_Type_I | BMI 30–34.9 |
| Obesity_Type_II | BMI 35–39.9 |
| Obesity_Type_III | BMI ≥ 40 |
