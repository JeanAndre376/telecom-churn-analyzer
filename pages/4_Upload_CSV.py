import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Upload CSV", layout="wide")
st.title("📁 Batch Score Customers")

scaler = joblib.load('models/scaler.pkl')
pca_13 = joblib.load('models/pca_13.pkl')
kmeans = joblib.load('models/kmeans.pkl')
cols   = joblib.load('models/feature_names.pkl')

st.markdown("Upload your Telco Customer Churn CSV to score all customers at once.")
st.markdown("**Accepted files:** WA_Fn-UseC_-Telco-Customer-Churn.csv or Telco_Customer_Churn.csv")

uploaded = st.file_uploader("Upload Telco Customer Churn CSV file", type="csv")

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.write(f"Loaded {len(df_raw):,} customers")

    # Minimal preprocessing
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    df_raw.dropna(inplace=True)
    df_raw['SeniorCitizen'] = df_raw['SeniorCitizen'].astype(int)

    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
    for c in binary_cols:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].map({'Yes':1,'No':0,'Male':1,'Female':0}).fillna(df_raw[c])

    def tenure_group(t):
        if t <= 12:   return 'New'
        elif t <= 24: return 'Developing'
        elif t <= 48: return 'Mature'
        else:         return 'Loyal'

    df_raw['tenure_group']  = df_raw['tenure'].apply(tenure_group)
    df_raw['charges_ratio'] = df_raw['MonthlyCharges'] / (df_raw['TotalCharges'] + 1)
    df_raw['high_value']    = (df_raw['MonthlyCharges'] > df_raw['MonthlyCharges'].median()).astype(int)

    multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod','tenure_group']
    df_enc = pd.get_dummies(df_raw, columns=multi_cols, drop_first=False)
    df_enc.drop(columns=['customerID'], errors='ignore', inplace=True)

    # Align columns
    for c in cols:
        if c not in df_enc.columns:
            df_enc[c] = 0
    X = df_enc[cols].copy()
    scale_cols = ['tenure','MonthlyCharges','TotalCharges','charges_ratio']
    X[scale_cols] = scaler.transform(X[scale_cols])
    X_pca = pca_13.transform(X)
    df_raw['cluster'] = kmeans.predict(X_pca)
    df_raw['segment'] = df_raw['cluster'].map({
        0: 'VIP Loyalists', 1: 'At-Risk New', 2: 'Low-spend Safe'})

    # Summary
    st.markdown("---")
    st.subheader("Segment Distribution")
    summary = df_raw.groupby('segment').agg(
        Count=('cluster','count'),
        Churn_Rate=('Churn', lambda x: f"{pd.to_numeric(x, errors='coerce').mean():.1%}")
    ).reset_index()
    st.dataframe(summary)

    st.subheader("Full Scored Dataset")
    st.dataframe(df_raw[['tenure','MonthlyCharges','TotalCharges','cluster','segment']].head(50))

    csv = df_raw.to_csv(index=False).encode('utf-8')
    st.download_button("Download scored CSV", csv,
                       "customers_scored.csv", "text/csv")