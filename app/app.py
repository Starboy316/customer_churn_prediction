import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- Setup Paths ---
base_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(base_path, '..', 'model', 'churn_rf_model.pkl'))
features_path = os.path.abspath(os.path.join(base_path, '..', 'model', 'feature_columns.pkl'))

# --- Check if files exist ---
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

if not os.path.exists(features_path):
    st.error(f"Feature columns file not found at {features_path}")
    st.stop()

# --- Load Model and Feature Columns ---
model = joblib.load(model_path)
feature_names = joblib.load(features_path)

# --- Streamlit UI ---
st.title("📞 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict whether they are likely to churn.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Manual encoding
input_dict = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
}

# Create input DataFrame
input_df = pd.DataFrame([input_dict])

# Fill missing columns
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[feature_names]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("❌ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
