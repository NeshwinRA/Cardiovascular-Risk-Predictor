import joblib
import numpy as np

# Load model and scaler
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

# Sample data
sample_values = {
    "male": 1,
    "age": 50,
    "currentSmoker": 0,
    "cigsPerDay": 0,
    "BPMeds": 0,
    "prevalentStroke": 0,
    "prevalentHyp": 0,
    "diabetes": 0,
    "totChol": 200,
    "sysBP": 120,
    "diaBP": 80,
    "BMI": 25.0,
    "heartRate": 70,
    "glucose": 90,
}

# Build feature array
feature_array = np.array([[sample_values[field] for field in FEATURES]], dtype=float)
print("Feature array shape:", feature_array.shape)
print("Feature array:", feature_array)

# Scale
scaled = scaler.transform(feature_array)
print("Scaled shape:", scaled.shape)
print("Scaled:", scaled)

# Predict
try:
    probability = float(model.predict_proba(scaled)[0, 1])
    print("Probability:", probability)
    risk_percentage = int(round(probability * 100))
    print("Risk percentage:", risk_percentage)
except Exception as e:
    print("Error in prediction:", e)