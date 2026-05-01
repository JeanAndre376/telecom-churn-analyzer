import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Cluster Explorer", page_icon="🔵", layout="wide")
st.title("🔵 Cluster Explorer")

df     = pd.read_csv('models/df_clustered.csv')
pca_2d = joblib.load('models/pca_2d.pkl')
scaler = joblib.load('models/scaler.pkl')
cols   = joblib.load('models/feature_names.pkl')

scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_ratio']
X = df[cols].copy()
X[scale_cols] = scaler.transform(X[scale_cols])
X_2d = pca_2d.transform(X)

scatter_df = pd.DataFrame({
    'PC1': X_2d[:, 0],
    'PC2': X_2d[:, 1],
    'Segment': df['cluster'].map({
        0: 'VIP Loyalists',
        1: 'At-Risk New',
        2: 'Low-spend Safe'
    }),
    'Churn': df['Churn'].map({0: 'No', 1: 'Yes'}),
    'Tenure': df['tenure'],
    'Monthly ($)': df['MonthlyCharges']
})

st.subheader("PCA 2D — Customer Segments")
fig = px.scatter(
    scatter_df, x='PC1', y='PC2', color='Segment',
    color_discrete_map={
        'VIP Loyalists':   '#1D9E75',
        'At-Risk New':     '#D85A30',
        'Low-spend Safe':  '#378ADD'
    },
    hover_data=['Tenure', 'Monthly ($)', 'Churn'],
    opacity=0.5
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Segment Profiles")

col0, col1, col2 = st.columns(3)
for col, cid, name, color in zip(
    [col0, col1, col2],
    [0, 1, 2],
    ['🟢 VIP Loyalists', '🔴 At-Risk New', '🔵 Low-spend Safe'],
    ['#E1F5EE', '#FAECE7', '#E6F1FB']
):
    mask = df['cluster'] == cid
    with col:
        st.markdown(f"#### {name}")
        st.metric("Customers", f"{mask.sum():,}")
        st.metric("Churn Rate", f"{df[mask]['Churn'].mean():.1%}")
        st.metric("Avg Tenure", f"{df[mask]['tenure'].mean():.0f} months")
        st.metric("Avg Monthly", f"${df[mask]['MonthlyCharges'].mean():.0f}")
        st.metric("Avg Total Spend", f"${df[mask]['TotalCharges'].mean():,.0f}")

st.markdown("---")
st.subheader("Churn Rate by Segment")
segment_churn = df.groupby('cluster')['Churn'].mean().reset_index()
segment_churn['Segment'] = segment_churn['cluster'].map({
    0: 'VIP Loyalists', 1: 'At-Risk New', 2: 'Low-spend Safe'})
segment_churn['Churn Rate'] = segment_churn['Churn'] * 100
fig2 = px.bar(segment_churn, x='Segment', y='Churn Rate',
              color='Segment',
              color_discrete_map={
                  'VIP Loyalists':  '#1D9E75',
                  'At-Risk New':    '#D85A30',
                  'Low-spend Safe': '#378ADD'
              }, text='Churn Rate')
fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig2.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
st.plotly_chart(fig2, use_container_width=True)