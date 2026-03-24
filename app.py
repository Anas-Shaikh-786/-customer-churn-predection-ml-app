import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

# -------------------------------------------------------
# 1️⃣ Define binary_mapper (must exist before loading pipeline)
# -------------------------------------------------------
def binary_mapper(X):
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    return X

# -------------------------------------------------------
# 2️⃣ Load saved pipeline
# -------------------------------------------------------
pipeline = joblib.load("churn_rf_pipeline_streamlit.pkl")

# -------------------------------------------------------
# 3️⃣ Streamlit page setup
# -------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="💡", layout="wide")

st.title("💡 Customer Churn Prediction App")
st.markdown("Predict whether a telecom customer will churn based on their service and billing information.")

st.sidebar.header("📋 Customer Information")

# -------------------------------------------------------
# 4️⃣ Sidebar Inputs
# -------------------------------------------------------
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0)

# -------------------------------------------------------
# 5️⃣ Prepare Input DataFrame
# -------------------------------------------------------
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# -------------------------------------------------------
# 6️⃣ Prediction Section
# -------------------------------------------------------
st.subheader("🔍 Prediction Result")

if st.button("Predict Churn"):
    try:
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        col1, col2 = st.columns([2, 3])

        # Display textual result
        with col1:
            if prediction == 1:
                st.error(f"⚠️ **Customer is likely to churn.**")
                st.write(f"**Churn Probability:** {probability:.2f}")
            else:
                st.success(f"✅ **Customer is not likely to churn.**")
                st.write(f"**Churn Probability:** {probability:.2f}")

        # Display a simple probability gauge chart
        with col2:
            fig, ax = plt.subplots(figsize=(4, 1.2))
            ax.barh(["Churn Probability"], [probability], color="red" if prediction == 1 else "green")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("📊 Churn Probability Gauge")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("👈 Fill in customer details in the sidebar and click **Predict Churn**.")
