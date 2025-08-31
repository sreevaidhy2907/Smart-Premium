from __future__ import annotations
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="SmartPremium – Insurance Premium Predictor", page_icon="💰")
st.title("SmartPremium – Insurance Premium Predictor")


model = joblib.load("artifacts/premium_model.joblib")

col1, col2 = st.columns(2)
age = col1.number_input("Age", 18, 100, 30)
income = col2.number_input("Annual Income", 10_000, 10_000_000, 500_000)
health = col1.slider("Health Score", 1, 100, 70)
claims = col2.number_input("Previous Claims", 0, 50, 0)
vehicle_age = col1.number_input("Vehicle Age (years)", 0, 30, 5)
credit = col2.number_input("Credit Score", 300, 850, 650)
duration = col1.number_input("Insurance Duration (years)", 0, 50, 5)
dependents = col2.number_input("Number of Dependents", 0, 10, 0)


policy_type = col1.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
gender = col2.selectbox("Gender", ["Male", "Female"])
location = col1.selectbox("Location", ["Urban", "Suburban", "Rural"])


if st.button("Predict Premium"):
    X = pd.DataFrame([
    {
    "Age": age,
    "Annual Income": income,
    "Health Score": health,
    "Previous Claims": claims,
    "Vehicle Age": vehicle_age,
    "Credit Score": credit,
    "Insurance Duration": duration,
    "Number of Dependents": dependents,
    "Policy Type": policy_type,
    "Gender": gender,
    "Location": location,
    }
    ])
pred = float(model.predict(X)[0])
st.success(f"Predicted Premium Amount: ₹ {pred:,.2f}")