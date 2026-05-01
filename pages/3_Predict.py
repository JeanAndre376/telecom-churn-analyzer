import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predict", layout="wide")
st.title("🔮 Predict Customer Churn Risk")

scaler  = joblib.load('models/scaler.pkl')
pca_13  = joblib.load('models/pca_13.pkl')
kmeans  = joblib.load('models/kmeans.pkl')
rf_c1   = joblib.load('models/rf_cluster1.pkl')
cols    = joblib.load('models/feature_names.pkl')

st.markdown("Enter a customer's details to predict their cluster and churn probability.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account Info")
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly        = st.slider("Monthly Charges ($)", 18, 120, 65)
    total          = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly * tenure))

with col2:
    st.subheader("Services")
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup  = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Demographics")
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner", ["Yes", "No"])
    dependents     = st.selectbox("Dependents", ["Yes", "No"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment        = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict Churn Risk", type="primary"):
    # Build input row matching training columns
    row = dict.fromkeys(cols, 0)

    # Numeric
    row['tenure']         = tenure
    row['MonthlyCharges'] = monthly
    row['TotalCharges']   = total
    row['charges_ratio']  = monthly / (total + 1)
    row['high_value']     = 1 if monthly > 64.76 else 0
    row['SeniorCitizen']  = 1 if senior == "Yes" else 0
    row['Partner']        = 1 if partner == "Yes" else 0
    row['Dependents']     = 1 if dependents == "Yes" else 0
    row['PaperlessBilling'] = 1 if paperless == "Yes" else 0

    # Tenure group
    if tenure <= 12:   tg = 'New'
    elif tenure <= 24: tg = 'Developing'
    elif tenure <= 48: tg = 'Mature'
    else:              tg = 'Loyal'
    if f'tenure_group_{tg}' in row: row[f'tenure_group_{tg}'] = 1

    # Contract
    contract_map = {
        "Month-to-month": "Contract_Month-to-month",
        "One year": "Contract_One year",
        "Two year": "Contract_Two year"
    }
    if contract_map[contract] in row: row[contract_map[contract]] = 1

    # Internet
    inet_map = {
        "DSL": "InternetService_DSL",
        "Fiber optic": "InternetService_Fiber optic",
        "No": "InternetService_No"
    }
    if inet_map[internet] in row: row[inet_map[internet]] = 1

    # Tech support
    ts_map = {"Yes": "TechSupport_Yes", "No": "TechSupport_No",
               "No internet service": "TechSupport_No internet service"}
    if ts_map[tech_support] in row: row[ts_map[tech_support]] = 1

    # Online security
    os_map = {"Yes": "OnlineSecurity_Yes", "No": "OnlineSecurity_No",
               "No internet service": "OnlineSecurity_No internet service"}
    if os_map[online_sec] in row: row[os_map[online_sec]] = 1

    # Payment
    pay_map = {
        "Electronic check": "PaymentMethod_Electronic check",
        "Mailed check": "PaymentMethod_Mailed check",
        "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
        "Credit card (automatic)": "PaymentMethod_Credit card (automatic)"
    }
    if pay_map[payment] in row: row[pay_map[payment]] = 1

    # Scale and predict cluster
    X_input = pd.DataFrame([row])[cols]
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio']
    X_input[scale_cols] = scaler.transform(X_input[scale_cols])
    X_pca = pca_13.transform(X_input)
    cluster = int(kmeans.predict(X_pca)[0])

    cluster_names = {0: 'VIP Loyalists', 1: 'At-Risk New Customers', 2: 'Low-spend Safe'}
    cluster_colors = {0: 'green', 1: 'red', 2: 'blue'}
    churn_rates = {0: '16%', 1: '45%', 2: '7%'}

    st.markdown("---")
    st.subheader("Prediction Result")
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Assigned Cluster", f"Cluster {cluster} — {cluster_names[cluster]}")
        st.metric("Typical Churn Rate for this Segment", churn_rates[cluster])

    with col_b:
        if cluster == 1:
            prob = rf_c1.predict_proba(X_input)[0][1]
            st.metric("Churn Probability (RF model)", f"{prob:.1%}")
            if prob > 0.6:
                st.error("High churn risk — recommend immediate intervention")
            elif prob > 0.4:
                st.warning("Medium churn risk — monitor closely")
            else:
                st.success("Lower risk within at-risk segment")
        else:
            st.info(f"Using segment baseline: {churn_rates[cluster]} churn rate")

    # Recommendation
    recs = {
        0: "This is a VIP customer. Protect with loyalty rewards. Avoid disruptive upsells.",
        1: "At-risk customer! Offer TechSupport add-on and contract upgrade within 30 days.",
        2: "Stable phone-only customer. Great candidate for internet service upsell campaign."
    }
    st.markdown(f"**Recommendation:** {recs[cluster]}")