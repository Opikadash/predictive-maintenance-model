# (Streamlit UI)

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/trained_model.pkl")

st.title("Predict Equipment Failure")
st.write("Enter sensor values:")

temperature = st.number_input("Temperature", 0.0, 100.0, 70.0)
pressure = st.number_input("Pressure", 0.0, 50.0, 30.0)
vibration = st.number_input("Vibration", 0.0, 10.0, 5.0)

if st.button("Predict"):
    X_new = scaler.transform([[temperature, pressure, vibration]])
    prediction = model.predict(X_new)[0]
    result = "Failure" if prediction == 1 else "Normal"

    st.success(f"The equipment is predicted to be: {result}")
