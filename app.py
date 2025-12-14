import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_churn_pipeline.pkl")

st.title("📉 Customer Churn Prediction")

age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure", 0, 100)
usage = st.number_input("Usage Frequency", 0, 100)
support = st.number_input("Support Calls", 0, 50)
delay = st.number_input("Payment Delay", 0, 100)
spend = st.number_input("Total Spend", 0.0)
last_interaction = st.number_input("Last Interaction", 0, 100)

gender = st.selectbox("Gender", ["Male", "Female"])
sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

input_df = pd.DataFrame([{
    "Age": age,
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support,
    "Payment Delay": delay,
    "Total Spend": spend,
    "Last Interaction": last_interaction,
    "Gender": gender,
    "Subscription Type": sub_type,
    "Contract Length": contract
}])

if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ High churn risk ({prob:.2%})")
        st.markdown("""
        **Business Recommendation:**
        - Prioritize customer retention
        - Offer loyalty discounts
        - Proactive customer support
        """)
    else:
        st.success(f"✅ Low churn risk ({prob:.2%})")
