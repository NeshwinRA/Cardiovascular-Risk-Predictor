# Cardiovascular Risk Prediction System — Master Project Prompt

> This is the complete prompt to build the project step by step.  
> Use this prompt (or sections of it) when working with an AI coding assistant or building manually in VS Code.  
> Follow the phases in order. Do not skip phases.

---

## PROJECT OVERVIEW

Build a web-based Cardiovascular Risk Prediction System using the Framingham Heart Study dataset.  
The system predicts a patient's 10-year risk of coronary heart disease (CHD) using machine learning.  
It is split into a training script, a Flask backend, and a multi-page HTML/CSS/JS frontend.

---

## FOLDER STRUCTURE

```
cardio-project/
│
├── data/
│   └── framingham.csv
│
├── models/
│   ├── best_model.pkl       ← saved after train.py runs
│   ├── scaler.pkl           ← saved after train.py runs
│   └── model_info.json      ← saved after train.py runs
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── templates/
│   ├── index.html           ← patient input form
│   ├── result.html          ← prediction result
│   └── recommendations.html ← health recommendations
│
├── app.py                   ← Flask backend
├── train.py                 ← model training script
└── requirements.txt
```

---

## PHASE 1 — PROJECT SETUP

**Goal:** Get the environment ready before writing any code.

### Steps:
1. Create the folder structure exactly as shown above.
2. Place `framingham.csv` inside the `/data` folder.
3. Create `requirements.txt` with these packages:
   ```
   flask
   pandas
   numpy
   scikit-learn
   imbalanced-learn
   joblib
   matplotlib
   ```
4. Open a terminal in VS Code. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
5. Verify the CSV loads without errors:
   ```python
   import pandas as pd
   df = pd.read_csv("data/framingham.csv")
   print(df.shape)
   print(df.head())
   ```

---

## PHASE 2 — DATA PREPROCESSING (train.py)

**Goal:** Clean and prepare the Framingham dataset for training.

### Steps inside train.py:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib, json

# 1. Load data
df = pd.read_csv("data/framingham.csv")

# 2. Handle missing values — fill with column mean
df.fillna(df.mean(), inplace=True)

# 3. Define features and target
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# 4. Split 80/20 BEFORE applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Apply SMOTE on training data only
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 6. Scale features — fit on train, transform both
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Save scaler
joblib.dump(scaler, "models/scaler.pkl")
```

---

## PHASE 3 — MODEL TRAINING & EVALUATION (train.py continued)

**Goal:** Train both models, evaluate them, save the best one.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
print("=== Logistic Regression ===")
print(classification_report(y_test, lr_preds))
print(f"ROC-AUC: {lr_auc:.4f}")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print("=== Random Forest ===")
print(classification_report(y_test, rf_preds))
print(f"ROC-AUC: {rf_auc:.4f}")

# Select best model
if rf_auc >= lr_auc:
    best_model = rf
    best_name = "Random Forest"
    best_auc = rf_auc
else:
    best_model = lr
    best_name = "Logistic Regression"
    best_auc = lr_auc

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

# Save model info
model_info = {
    "selected_model": best_name,
    "roc_auc": best_auc,
    "lr_auc": lr_auc,
    "rf_auc": rf_auc
}
with open("models/model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)

print(f"\nBest Model: {best_name} saved to /models/best_model.pkl")
```

**Run training:**
```bash
python train.py
```
Confirm that `/models/best_model.pkl`, `/models/scaler.pkl`, and `/models/model_info.json` are created.

---

## PHASE 4 — FLASK BACKEND (app.py)

**Goal:** Build the backend that connects the model to the web interface.

```python
from flask import Flask, render_template, request, redirect, url_for, session
import joblib, json, numpy as np

app = Flask(__name__)
app.secret_key = "cardio_secret_key"

# Load model and scaler at startup
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURE_COLUMNS = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

# ─── Input Validation ───────────────────────────────────────────────────────
def validate_input(data):
    errors = []
    ranges = {
        "age": (20, 80), "sysBP": (80, 250), "diaBP": (50, 150),
        "totChol": (100, 400), "BMI": (10, 60),
        "glucose": (50, 400), "heartRate": (40, 150)
    }
    for field, (low, high) in ranges.items():
        val = data.get(field)
        if val is None or val == "":
            errors.append(f"{field} is required.")
        elif not (low <= float(val) <= high):
            errors.append(f"{field} must be between {low} and {high}.")
    return errors

# ─── Rule-based XAI ─────────────────────────────────────────────────────────
def get_risk_factors(data):
    factors = []
    if float(data.get("totChol", 0)) > 240:
        factors.append("High Cholesterol")
    if float(data.get("sysBP", 0)) > 140:
        factors.append("High Blood Pressure")
    if float(data.get("BMI", 0)) > 30:
        factors.append("Obesity (High BMI)")
    if float(data.get("glucose", 0)) > 126:
        factors.append("High Glucose Level")
    if int(data.get("currentSmoker", 0)) == 1:
        factors.append("Active Smoker")
    if int(data.get("diabetes", 0)) == 1:
        factors.append("Diabetes")
    return factors

# ─── Disease Association ─────────────────────────────────────────────────────
def get_associated_diseases(factors):
    diseases = []
    if "High Blood Pressure" in factors:
        diseases.append("Hypertension")
    if "High Cholesterol" in factors:
        diseases.append("Atherosclerosis")
    if "High Glucose Level" in factors or "Diabetes" in factors:
        diseases.append("Diabetic Heart Disease")
    if "Active Smoker" in factors:
        diseases.append("Coronary Heart Disease")
    if "Obesity (High BMI)" in factors:
        diseases.append("Metabolic Syndrome")
    return diseases if diseases else ["No specific condition associated"]

# ─── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    errors = validate_input(data)
    if errors:
        return render_template("index.html", errors=errors, form_data=data)

    input_values = [float(data[col]) for col in FEATURE_COLUMNS]
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    risk_percent = round(probability * 100, 2)

    if risk_percent < 20:
        risk_level = "Low"
    elif risk_percent <= 50:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    risk_factors = get_risk_factors(data)
    diseases = get_associated_diseases(risk_factors)

    session["risk_percent"] = risk_percent
    session["risk_level"] = risk_level
    session["risk_factors"] = risk_factors
    session["diseases"] = diseases

    return render_template("result.html",
        risk_percent=risk_percent,
        risk_level=risk_level,
        risk_factors=risk_factors,
        diseases=diseases
    )

@app.route("/recommendations")
def recommendations():
    risk_percent = session.get("risk_percent", 0)
    risk_level = session.get("risk_level", "Unknown")
    risk_factors = session.get("risk_factors", [])
    diseases = session.get("diseases", [])
    return render_template("recommendations.html",
        risk_percent=risk_percent,
        risk_level=risk_level,
        risk_factors=risk_factors,
        diseases=diseases
    )

if __name__ == "__main__":
    app.run(debug=True)
```

---

## PHASE 5 — FRONTEND: INPUT PAGE (templates/index.html)

**Goal:** A clean professional patient data entry form.

- Create an HTML form with `method="POST"` and `action="/predict"`
- Include all 15 input fields:
  - `male` (0 or 1, dropdown)
  - `age` (number, 20–80)
  - `education` (1–4, dropdown)
  - `currentSmoker` (0 or 1, dropdown)
  - `cigsPerDay` (number)
  - `BPMeds` (0 or 1, dropdown)
  - `prevalentStroke` (0 or 1, dropdown)
  - `prevalentHyp` (0 or 1, dropdown)
  - `diabetes` (0 or 1, dropdown)
  - `totChol` (number, 100–400)
  - `sysBP` (number, 80–250)
  - `diaBP` (number, 50–150)
  - `BMI` (number, 10–60)
  - `heartRate` (number, 40–150)
  - `glucose` (number, 50–400)
- A "Predict Risk" submit button
- If the backend returns `errors`, display them at the top of the form in red
- If the backend returns `form_data`, pre-fill the form fields so the user doesn't retype everything

**Styling (style.css):**
- White card layout centered on a light grey background
- Blue/teal (`#1a6fb5` or similar) accent color for headings and button
- Clear label + input pairs with adequate spacing
- Red border on invalid fields
- Responsive (works on desktop at minimum)

**Client-side validation (main.js):**
- On form submit, check all required fields are filled
- Validate numeric ranges (same as backend)
- Show inline error messages next to each field
- Prevent form submission if errors exist
- On valid submit, show a loading spinner on the button

---

## PHASE 6 — FRONTEND: RESULT PAGE (templates/result.html)

**Goal:** Display prediction result with color coding and risk details.

- Large centered display of risk percentage (e.g., **78%**)
- Risk level label: Low / Moderate / High
- Color-coded banner:
  - Green (`#2e7d32`) for Low Risk (< 20%)
  - Yellow/Orange (`#f57c00`) for Moderate Risk (20%–50%)
  - Red (`#c62828`) for High Risk (> 50%)
- List of **flagged XAI risk factors** (passed from Flask)
- List of **associated conditions** (passed from Flask)
- If `risk_percent > 70`, show emergency warning banner:
  ```
  ⚠ High Cardiovascular Risk Detected
  Immediate medical consultation is strongly recommended.
  ```
- Button: **"View Health Recommendations"** → links to `/recommendations`
- Disclaimer at the bottom:
  > "This is an AI-generated cardiovascular risk assessment and not a final medical diagnosis. Please consult a licensed doctor for professional medical evaluation."

**Use Jinja2 conditionals** to set the color class:
```html
{% if risk_level == "Low" %}
  <div class="banner green">...</div>
{% elif risk_level == "Moderate" %}
  <div class="banner yellow">...</div>
{% else %}
  <div class="banner red">...</div>
{% endif %}
```

---

## PHASE 7 — FRONTEND: RECOMMENDATIONS PAGE (templates/recommendations.html)

**Goal:** Personalized health guidance based on the prediction.

### Sections to include:

**1. Risk Summary Header**
- Show risk level and percentage at the top

**2. Disease Information Cards**
For each disease in `diseases` list, show a card with:
- Disease name
- What it is and how it affects the body
- Prevention methods

Hardcode the disease information as a Python dict in `app.py` or as a JS object in the template:
```python
DISEASE_INFO = {
    "Hypertension": {
        "description": "High blood pressure that strains the heart and arteries.",
        "effects": "Can damage arteries, lead to stroke, kidney failure, or heart attack.",
        "prevention": "Reduce sodium intake, exercise regularly, take prescribed medication."
    },
    "Atherosclerosis": { ... },
    "Diabetic Heart Disease": { ... },
    "Coronary Heart Disease": { ... },
    "Metabolic Syndrome": { ... }
}
```
Pass `DISEASE_INFO` to the template from the `/recommendations` route.

**3. Healthy Heart Maintenance Plan** (show if risk_level == "Low")
- Exercise at least 30 minutes/day, 5 days/week
- Follow a heart-healthy diet (low sodium, low saturated fat)
- Avoid smoking and excessive alcohol
- Manage stress through meditation or hobbies
- Get annual cardiovascular health checkups

**4. Emergency Recommendation** (show if risk_percent > 70)
- Red alert box: "Please consult a cardiologist or visit a hospital immediately."

**5. Disclaimer**
- Same disclaimer as result.html

**6. "Start Over" Button**
- Links back to `/` (index.html)

---

## PHASE 8 — TESTING & FINAL POLISH

### Testing Checklist:

1. **Run train.py** → confirm 3 files appear in `/models`
2. **Run app.py** → confirm Flask starts with no errors
3. **Open browser** → go to `http://127.0.0.1:5000`
4. **Submit valid data** → confirm result page shows correctly
5. **Submit empty form** → confirm validation errors appear
6. **Submit out-of-range values** → confirm range errors appear
7. **Test low-risk profile:**
   - Age: 25, Non-smoker, Normal BP, Normal Cholesterol, Normal Glucose, Normal BMI
   - Expect: Green banner, Low Risk, Maintenance Plan shown
8. **Test high-risk profile:**
   - Age: 65, Smoker, High BP, High Cholesterol, High Glucose, High BMI, Diabetes
   - Expect: Red banner, High Risk, Emergency Warning shown
9. **Check all 3 pages** for broken links, missing styles, console errors

---

## QUICK REFERENCE — KEY VALUES

| Parameter | Valid Range |
|-----------|-------------|
| Age | 20 – 80 |
| Systolic BP | 80 – 250 |
| Diastolic BP | 50 – 150 |
| Total Cholesterol | 100 – 400 |
| BMI | 10 – 60 |
| Glucose | 50 – 400 |
| Heart Rate | 40 – 150 |

| Risk Level | Range |
|------------|-------|
| Low | < 20% |
| Moderate | 20% – 50% |
| High | > 50% |
| Emergency | > 70% |

---

## HOW TO RUN THE PROJECT

```bash
# Step 1: Activate virtual environment
venv\Scripts\activate

# Step 2: Train the model (run once)
python train.py

# Step 3: Start the Flask app
python app.py

# Step 4: Open browser
http://127.0.0.1:5000
```

---

*End of Master Project Prompt*
