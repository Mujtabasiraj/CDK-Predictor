import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="CKD Prediction", layout="wide")
st.image("kidney.jpeg", width=100)

# Load model and medians
model = pickle.load(open("xgb_ckd_model", "rb"))
median_values = pickle.load(open("median_values.pkl", "rb"))

selected_features = ['hemo', 'dm', 'sg', 'sc', 'htn', 'al', 'rc', 'bgr', 'sod', 'age']

# Descriptions and ranges
feature_descriptions = {
    "age": ("Age", "Age of the patient (15 - 90 years)", 15, 90),
    "sg": ("Specific Gravity", "Urine specific gravity (1.005 - 1.025)", 1.005, 1.025),
    "al": ("Albumin", "Albumin level in urine (0 - 5)", 0, 5),
    "bgr": ("Blood Glucose", "Blood glucose level (70 - 200 mg/dL)", 70, 200),
    "sc": ("Serum Creatinine", "Kidney function (0.4 - 12 mg/dL)", 0.4, 12),
    "sod": ("Sodium", "Sodium in blood (135 - 145 mEq/L)", 135, 145),
    "hemo": ("Hemoglobin", "Hemoglobin (10 - 17 g/dL)", 10, 17),
    "rc": ("Red Blood Cell Count", "RBC count (2.5 - 6.5 million/mm³)", 2.5, 6.5),
    "htn": ("Hypertension", "0 = No, 1 = Yes", 0, 1),
    "dm": ("Diabetes Mellitus", "0 = No, 1 = Yes", 0, 1)
}

# Title and inputs
st.title("🩺 Chronic Kidney Disease (CKD) Predictor")
st.markdown("Enter your details below to check your CKD risk.")

st.subheader("🔢 Patient Data Entry")
cols = st.columns(4)
input_data = {}

for i, feature in enumerate(selected_features):
    col = cols[i % 4]
    label, tooltip, _, _ = feature_descriptions[feature]
    input_data[feature] = col.number_input(f"{label}", help=tooltip, step=0.1, format="%.2f")

# Predict button
if st.button("🔍 Predict"):
    invalid_features = []
    full_features = model.feature_names_in_

    # Validation: check for out-of-range
    for feature in selected_features:
        val = input_data[feature]
        _, _, min_val, max_val = feature_descriptions[feature]
        if not (min_val <= val <= max_val):
            invalid_features.append((feature_descriptions[feature][0], min_val, max_val))

    if invalid_features:
        st.warning("⚠️ Please correct the following fields before prediction:")
        for name, min_val, max_val in invalid_features:
            st.error(f"❗ {name} is out of range ({min_val} - {max_val})")
    else:
        # Fill missing features with medians
        for feat in full_features:
            if feat not in input_data:
                input_data[feat] = median_values[feat]

        input_df = pd.DataFrame([input_data])[full_features]
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            if prediction_proba >= 0.85:
                st.error(f"🛑 **High Risk of CKD!** (Confidence: {prediction_proba:.2%})")
            else:
                st.warning(f"⚠️ **Possible CKD.** (Confidence: {prediction_proba:.2%})")
        else:
            if prediction_proba < 0.5:
                st.success(f"✅ **No CKD Detected.** (Confidence: {(1 - prediction_proba):.2%})")
            else:
                st.warning(f"⚠️ **Uncertain Result.** (Confidence: {(1 - prediction_proba):.2%})")

# Reference table
st.markdown("### 📋 Recommended Input Ranges (for reference)")
st.dataframe(pd.DataFrame({
    "Feature": [feature_descriptions[f][0] for f in selected_features],
    "Valid Range / Values": [f"{feature_descriptions[f][2]} - {feature_descriptions[f][3]}" for f in selected_features]
}))

# About section
st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Model**: Trained using XGBoost on real patient data  
    - **Accuracy**: 98.75%  
    - **Inputs**: 10 most important medical indicators  
    - **Developer**: MD Inam and Syed Mujtaba (AI & Data Science)  
    - **Purpose**: Help early detection and awareness of Chronic Kidney Disease (CKD)  
    """)
