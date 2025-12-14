import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

model = joblib.load("knn_churn_pipeline.pkl")

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
    st.header("📌 About App")
    st.write("""
    This application predicts **customer churn risk**
    using **K-Nearest Neighbors (KNN)**.
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

    age = st.slider("Age", 18, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (months)", 0, 120, 12)
    contract = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )

with col2:
    st.subheader("📊 Usage & Payment")

    usage = st.slider("Usage Frequency", 0, 100, 20)
    support = st.slider("Support Calls", 0, 20, 1)
    delay = st.slider("Payment Delay (days)", 0, 60, 5)
    spend = st.slider("Total Spend", 0, 20000, 5000)
    last_interaction = st.slider("Last Interaction (days ago)", 0, 90, 7)
    sub_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

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

st.divider()

predict_clicked = st.button(
    "🔍 Predict Churn Risk",
    use_container_width=True
)

if predict_clicked:
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Churn Probability", f"{prob:.1%}")

    with colB:
        status = "High Risk" if pred == 1 else "Low Risk"
        st.metric("Churn Status", status)

    if pred == 1:
        st.error("🚨 Customer is at HIGH risk of churn")
        st.markdown("""
        ### 🎯 Recommended Actions
        - Offer loyalty discounts
        - Proactive customer support
        - Personalized engagement
        """)
    else:
        st.success("✅ Customer has LOW churn risk")
        st.markdown("""
        ### 💡 Recommended Actions
        - Maintain service quality
        - Upselling opportunities
        - Loyalty programs
        """)

