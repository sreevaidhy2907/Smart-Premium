# app/streamlit_app.py
import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# --- Fix Python path so we can import from src ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import build_features  # now works

# --- Load model ---
MODEL_PATH = "artifacts/premium_model_stacked.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at {MODEL_PATH}. Please run `python -m src.train` first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💰 Insurance Premium Prediction App")

st.markdown("Enter customer details below to estimate the insurance premium:")

# Collect inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=1000, max_value=1000000, value=50000)
health = st.slider("Health Score", min_value=1, max_value=10, value=5)
credit = st.slider("Credit Score", min_value=300, max_value=850, value=650)
claims = st.number_input("Previous Claims", min_value=0, max_value=50, value=1)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)

policy_type = st.selectbox("Policy Type", ["Type A", "Type B", "Type C"])
property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
occupation = st.selectbox("Occupation", ["Engineer", "Doctor", "Teacher", "Other"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])

# Predict button
if st.button("🔮 Predict Premium"):
    # Build dataframe
    data = pd.DataFrame([{
        "Age": age,
        "Annual Income": income,
        "Health Score": health,
        "Credit Score": credit,
        "Previous Claims": claims,
        "Vehicle Age": vehicle_age,
        "Policy Type": policy_type,
        "Property Type": property_type,
        "Occupation": occupation,
        "Location": location
    }])

    # Feature engineering
    preprocessor, X_test, _, _ = build_features(data)

    # Make prediction (model was trained on log scale)
    pred_log = model.predict(X_test)
    pred = float(np.clip(np.expm1(pred_log[0]), 100, 1300))  # ✅ fixed
    st.success(f"💵 Estimated Premium: **${pred:,.2f}**")
