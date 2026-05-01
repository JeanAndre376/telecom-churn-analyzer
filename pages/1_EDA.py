import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

df = pd.read_csv('models/df_clustered.csv')

st.markdown(f"**Dataset:** {len(df):,} customers · {df.shape[1]} features")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].map({0: 'No Churn', 1: 'Churn'}).value_counts().reset_index()
    churn_counts.columns = ['Status', 'Count']
    fig1 = px.bar(churn_counts, x='Status', y='Count',
                  color='Status',
                  color_discrete_map={'No Churn': '#1D9E75', 'Churn': '#D85A30'})
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Monthly Charges vs Tenure")
    fig2 = px.scatter(df, x='tenure', y='MonthlyCharges',
                      color=df['Churn'].map({0: 'No Churn', 1: 'Churn'}),
                      color_discrete_map={'No Churn': '#1D9E75', 'Churn': '#D85A30'},
                      opacity=0.4,
                      labels={'color': 'Churn'})
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Tenure Distribution by Churn")
fig3 = px.histogram(df, x='tenure',
                    color=df['Churn'].map({0: 'No Churn', 1: 'Churn'}),
                    nbins=30, barmode='overlay', opacity=0.7,
                    color_discrete_map={'No Churn': '#1D9E75', 'Churn': '#D85A30'},
                    labels={'color': 'Churn'})
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Key Statistics by Churn Status")
stats = df.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean().round(2)
stats.index = ['No Churn', 'Churn']
st.dataframe(stats, use_container_width=True)