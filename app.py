import streamlit as st

st.set_page_config(
    page_title="Telecom Churn Analyzer",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Customer Churn Analyzer")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", "7,032")
    st.metric("Overall Churn Rate", "26.5%")
with col2:
    st.metric("Clusters Found", "3")
    st.metric("Best Model", "K-Means k=3")
with col3:
    st.metric("Silhouette Score", "0.302")
    st.metric("Cluster 1 F1 Score", "0.662")

st.markdown("---")
st.markdown("""
### 🎯 Project Overview
This app presents the results of an unsupervised machine learning analysis
of telecom customer churn using the IBM Telco dataset (7,032 customers).

**Navigate using the sidebar:**
- 📊 **EDA** — dataset exploration and churn distributions
- 🔵 **Cluster Explorer** — the 3 customer segments and their profiles
- 🔮 **Predict** — enter a customer's details and get their churn risk
- 📁 **Upload CSV** — score a batch of customers at once

---
### 👥 Three Customer Segments Discovered
| Segment | Size | Churn Rate | Avg Tenure | Avg Monthly |
|---|---|---|---|---|
| 🟢 Cluster 0 — VIP Loyalists | 2,487 (35.4%) | 16% | 56 months | \$88 |
| 🔴 Cluster 1 — At-Risk New | 3,025 (43.0%) | 45% | 14 months | \$67 |
| 🔵 Cluster 2 — Low-spend Safe | 1,520 (21.6%) | 7% | 31 months | \$21 |

---
### 🏆 Key Finding
A dedicated Random Forest classifier for **Cluster 1 (At-Risk New)**
achieved an F1 score of **0.662** — an **18.6% improvement** over the
global baseline of 0.558. Clustering added measurable predictive value.
""")

st.sidebar.success("Select a page above to get started.")