# src/deploy.py
import joblib
import numpy as np

def predict_new(temperature, pressure, vibration):
    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load("model/trained_model.pkl")

    X_new = scaler.transform([[temperature, pressure, vibration]])

    prediction = model.predict(X_new)[0]
    return "Failure" if prediction == 1 else "Normal"

# Example:
print(predict_new(85, 36, 7))  # Should return "Failure"
print(predict_new(65, 29, 4))  # Should return "Normal"
