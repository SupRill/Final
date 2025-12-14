import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("knn_churn_pipeline.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
div[data-testid="stMetric"] {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 About")
    st.write("""
    Customer churn prediction app using
    **K-Nearest Neighbors (KNN)**.
    """)

    st.header("⚙️ Model Info")
    st.write("""
    - Encoding: One-Hot Encoder  
    - Scaling: StandardScaler  
    - Algorithm: KNN  
    """)

st.title("📉 Customer Churn Prediction")
st.caption("Interactive churn risk analysis for customer retention")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")
    age = st.number_input("Age", 18, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure", 0, 100)
    contract = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )

with col2:
    st.subheader("📊 Usage & Payment")
    usage = st.number_input("Usage Frequency", 0, 100)
    support = st.number_input("Support Calls", 0, 50)
    delay = st.number_input("Payment Delay", 0, 100)
    spend = st.number_input("Total Spend", 0.0)
    last_interaction = st.number_input("Last Interaction", 0, 100)
    sub_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )
