import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load('churn_model_pipeline.pkl')

st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="wide")

st.title("Customer Churn Prediction 🔮")
st.markdown("Enter customer details below to predict their churn risk.")

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Account Information")
    tenure = st.number_input("Account Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.01)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=780.0)

with col2:
    st.subheader("Demographics & Services")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    partner = st.selectbox("Has Partner", ['No', 'Yes'])
    dependents = st.selectbox("Has Dependents", ['No', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    # Add other service-related fields as needed for the model
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])


# --- Prediction Logic ---
if st.button("Analyze Churn Risk", type="primary"):
    # Create a DataFrame from the inputs
    # The order and names of columns must match the training data
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': ['No'], # Default value, can be an input field
        'OnlineBackup': ['No'], # Default value
        'DeviceProtection': ['No'], # Default value
        'TechSupport': ['No'], # Default value
        'StreamingTV': ['No'], # Default value
        'StreamingMovies': ['No'], # Default value
        'Contract': [contract],
        'PaperlessBilling': ['Yes'], # Default value
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Predict churn probability
    churn_probability = model.predict_proba(input_data)[:, 1][0]
    churn_prediction = model.predict(input_data)[0]

    # --- Display Result ---
    st.header("Prediction Result")
    if churn_prediction == 1:
        st.error(f"High Churn Risk 🚨")
        st.write(f"This customer has a **{churn_probability:.2%}** probability of churning.")
        st.progress(churn_probability)
    else:
        st.success(f"Low Churn Risk ✅")
        st.write(f"This customer has a **{churn_probability:.2%}** probability of churning.")
        st.progress(churn_probability)