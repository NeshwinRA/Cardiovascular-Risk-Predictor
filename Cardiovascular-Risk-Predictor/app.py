import json
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "cardio_secret_key_2026"

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURES = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

RANGE_VALIDATION = {
    "age": (20, 80),
    "male": (0, 1),
    "currentSmoker": (0, 1),
    "cigsPerDay": (0, 60),
    "BPMeds": (0, 1),
    "prevalentStroke": (0, 1),
    "prevalentHyp": (0, 1),
    "diabetes": (0, 1),
    "totChol": (100, 400),
    "sysBP": (80, 250),
    "diaBP": (50, 150),
    "BMI": (10, 60),
    "heartRate": (40, 150),
    "glucose": (50, 400),
}


def parse_value(name, value):
    if name in {"male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"}:
        return int(value)
    if name in {"age", "cigsPerDay", "totChol", "sysBP", "diaBP", "heartRate", "glucose"}:
        return int(value)
    return float(value)


def validate_inputs(form_data):
    errors = []
    values = {}
    for field in FEATURES:
        raw = form_data.get(field)
        if raw is None or raw == "":
            errors.append(f"{field} is required.")
            continue
        try:
            parsed = parse_value(field, raw)
        except ValueError:
            errors.append(f"{field} must be a valid number.")
            continue
        min_val, max_val = RANGE_VALIDATION[field]
        if parsed < min_val or parsed > max_val:
            errors.append(f"{field} must be between {min_val} and {max_val}.")
        values[field] = parsed
    return values, errors


def risk_level_label(risk_percentage):
    if risk_percentage > 50:
        return "High Risk"
    if risk_percentage >= 20:
        return "Moderate Risk"
    return "Low Risk"


def xai_flags(values):
    flags = []
    if values["totChol"] > 240:
        flags.append("High Cholesterol")
    if values["sysBP"] > 140 or values["diaBP"] > 90:
        flags.append("High Blood Pressure")
    if values["BMI"] >= 30:
        flags.append("Obesity")
    if values["glucose"] > 126:
        flags.append("High Glucose")
    if values["currentSmoker"] == 1:
        flags.append("Active Smoker")
    if values["diabetes"] == 1:
        flags.append("Diabetes")
    if not flags:
        flags.append("No major rule-based flags detected.")
    return flags


def disease_associations(flags):
    associations = []
    if "High Blood Pressure" in flags:
        associations.append("Hypertension")
    if "High Cholesterol" in flags:
        associations.append("Atherosclerosis")
    if "High Glucose" in flags or "Diabetes" in flags:
        associations.append("Diabetic Heart Disease")
    if "Active Smoker" in flags:
        associations.append("Coronary Heart Disease")
    if not associations:
        associations.append("No additional disease associations detected.")
    return associations


def build_recommendations(risk_label, flags):
    recommendations = [
        "Maintain regular checkups with a healthcare provider.",
        "Follow a balanced diet rich in vegetables, lean protein, and whole grains.",
        "Stay physically active at least 150 minutes per week.",
        "Avoid smoking and reduce exposure to secondhand smoke.",
    ]
    if risk_label == "High Risk":
        recommendations.append("Seek professional medical advice promptly due to elevated cardiovascular risk.")
    if "High Blood Pressure" in flags:
        recommendations.append("Monitor blood pressure regularly and follow hypertension management plans.")
    if "High Cholesterol" in flags:
        recommendations.append("Reduce saturated fats and trans fats; choose heart-healthy fats.")
    if "Active Smoker" in flags:
        recommendations.append("Consider smoking cessation support and avoid tobacco products.")
    if "Diabetes" in flags or "High Glucose" in flags:
        recommendations.append("Manage blood sugar with diet, exercise, and medical guidance.")
    if "Obesity" in flags:
        recommendations.append("Adopt sustainable weight management strategies and regular exercise.")
    return recommendations


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate_inputs(request.form)
    if errors:
        return render_template("index.html", errors=errors, form=request.form)

    feature_array = np.array([[values[field] for field in FEATURES]], dtype=float)
    scaled = scaler.transform(feature_array)
    try:
        probability = float(model.predict_proba(scaled)[0, 1])
    except AttributeError:
        probability = float(model.predict(scaled)[0])
    risk_percentage = int(round(probability * 100))
    level = risk_level_label(risk_percentage)
    flags = xai_flags(values)
    associations = disease_associations(flags)
    recommendations = build_recommendations(level, flags)

    result_data = {
        "risk_percentage": risk_percentage,
        "risk_level": level,
        "flags": flags,
        "associations": associations,
        "recommendations": recommendations,
        "values": values,
    }
    session["result_data"] = result_data
    return render_template("result.html", **result_data)


@app.route("/recommendations")
def recommendations():
    result_data = session.get("result_data")
    if not result_data:
        return redirect(url_for("home"))
    return render_template(
        "recommendations.html",
        risk_percentage=result_data["risk_percentage"],
        risk_level=result_data["risk_level"],
        flags=result_data["flags"],
        associations=result_data["associations"],
        recommendations=result_data["recommendations"],
    )


@app.route("/reset")
def reset():
    session.pop("result_data", None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run()
