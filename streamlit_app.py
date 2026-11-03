import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Genetic Disorder Classifier", layout="centered")

st.title("🧬 Genetic Disorder Risk Classification System")
st.write("Predict the risk of genetic disorders based on patient clinical parameters.")

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# User Input Form
st.header("Enter Patient Details")
age = st.number_input("Age", min_value=1, max_value=100, value=25)
bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
gene_score = st.slider("Gene Mutation Score", 0.0, 1.0, 0.5)
family_hist = st.selectbox("Family History of Genetic Disorder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict Risk"):
    features = np.array([[age, bp, gene_score, family_hist]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk Detected (Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk Detected (Probability: {(1-prob)*100:.2f}%)")
