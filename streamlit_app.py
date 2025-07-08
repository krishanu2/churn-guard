# ✅ File: streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import os

# Load trained model
model = joblib.load("model.joblib")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Predictor")
st.markdown("""
Welcome to the Telco Churn Predictor App. Please enter the customer's details below, and we will predict whether the customer is likely to churn.
""")

st.header("📝 Customer Information")

# Helper for sidebar descriptions
def help_text(field):
    return f"Enter customer's {field.lower()}"

# Input fields
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"], help=help_text("Gender"))
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], help="1 if Senior Citizen, 0 otherwise")
        Partner = st.selectbox("Partner", ["Yes", "No"], help=help_text("Partner"))
        Dependents = st.selectbox("Dependents", ["Yes", "No"], help=help_text("Dependents"))
        tenure = st.number_input("Tenure (in months)", min_value=0, help=help_text("Tenure"))
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"], help=help_text("Phone Service"))
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], help=help_text("Multiple Lines"))
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help=help_text("Internet Service"))
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], help=help_text("Online Security"))
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], help=help_text("Online Backup"))

    with col2:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], help=help_text("Device Protection"))
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], help=help_text("Tech Support"))
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], help=help_text("Streaming TV"))
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], help=help_text("Streaming Movies"))
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help=help_text("Contract"))
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], help=help_text("Paperless Billing"))
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], help=help_text("Payment Method"))
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, help=help_text("Monthly Charges"))
        TotalCharges = st.number_input("Total Charges", min_value=0.0, help=help_text("Total Charges"))

    name = st.text_input("Your Name (for the PDF report)")
    email = st.text_input("Your Email (optional, not yet used)")

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Collect data into a DataFrame
    user_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    # Make prediction
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]

    status = "✅ Not Likely to Churn" if prediction == 0 else "⚠️ Likely to Churn"
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Status:** {status}")
    st.markdown(f"**Churn Probability:** {prob:.2%}")

    # Generate PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Customer Churn Prediction Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {'Likely to Churn' if prediction == 1 else 'Not Likely to Churn'}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {prob:.2%}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Input Details:", ln=True)
    for key, value in user_input.iloc[0].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    # Save PDF
    pdf_path = os.path.join(os.getcwd(), f"churn_report_{name.replace(' ', '_')}.pdf")
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name=f"churn_report_{name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
