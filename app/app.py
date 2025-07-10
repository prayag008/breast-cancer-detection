import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Breast Cancer Detector", layout="centered")
st.title("🔬 Breast Cancer Detection (Logistic Regression)")

# Load trained model
model = joblib.load("models/logistic_model.pkl")

# Optional: Load feature names if needed
FEATURE_NAMES = [
    "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
    "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", 
    "Bland Chromatin", "Normal Nucleoli", "Mitoses"
]

st.subheader("📥 Input the features for prediction")

user_input = []
for feature in FEATURE_NAMES:
    val = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
    user_input.append(val)

if st.button("🧠 Predict"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]
    label = "Malignant" if prediction == 1 else "Benign"
    st.success(f"🔎 Prediction: **{label}**")
