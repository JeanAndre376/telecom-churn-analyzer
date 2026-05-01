# Telecom Customer Churn Analyzer
### End-to-end Unsupervised ML project with Streamlit deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
(https://telecom-churn-analyzer-eywtuf35hy25opabxkndr3.streamlit.app/)


---

## Live Demo
Visit the live app: [(https://telecom-churn-analyzer-eywtuf35hy25opabxkndr3.streamlit.app/)](https://telecom-churn-analyzer-eywtuf35hy25opabxkndr3.streamlit.app/)]

---

## Project Overview
This project applies unsupervised machine learning to identify
natural customer segments in telecom churn data and derive
actionable retention strategies.

**Dataset:** IBM Telco Customer Churn (Kaggle)
**Customers:** 7,032
**Features:** 21 (demographics, services, billing)
**Target:** Churn (Yes/No) — 26.5% overall churn rate

---

## Three Customer Segments Discovered

| Segment | Size | Churn Rate | Avg Tenure | Avg Monthly |
|---|---|---|---|---|
| Cluster 0 — VIP Loyalists | 2,487 (35.4%) | 16% | 56 months | $88 |
| Cluster 1 — At-Risk New | 3,025 (43.0%) | 45% | 14 months | $67 |
| Cluster 2 — Low-spend Safe | 1,520 (21.6%) | 7% | 31 months | $21 |

---

## Key Finding
A dedicated Random Forest classifier for Cluster 1 (At-Risk New)
achieved F1 = 0.662 — an 18.6% improvement over the global
baseline of 0.558. Clustering added measurable predictive value.

---

## Pipeline

```
Raw Data
   -> EDA & Visualization
   -> Cleaning & Feature Engineering
   -> PCA (13 components, 80% variance)
   -> K-Means | DBSCAN | Hierarchical (3 model comparison)
   -> Cluster Profiling & Business Insights
   -> Per-cluster Random Forest (optional supervised)
   -> Streamlit Deployment
```

---

## Model Comparison

| Model | Silhouette | Davies-Bouldin |
|---|---|---|
| K-Means k=3 (winner) | 0.302 | 1.345 |
| K-Means k=2 | 0.294 | 1.200 |
| K-Means k=4 | 0.282 | 1.359 |
| Hierarchical k=4 | 0.259 | 1.312 |
| DBSCAN | 0.001 | 1.373 |

---

## App Features

| Page | Description |
|---|---|
| Overview | Project summary and key metrics |
| EDA | Churn distributions and feature analysis |
| Cluster Explorer | Interactive PCA scatter + segment profiles |
| Predict | Enter customer details, get churn probability |
| Upload CSV | Batch score any Telco CSV file |

---

| Page | Description |
|---|---|
| Overview | Project summary and key metrics |
| EDA | Churn distributions and feature analysis |
| Cluster Explorer | Interactive PCA scatter + segment profiles |
| Predict | Enter customer details, get churn probability |
| Upload CSV | Batch score any Telco CSV file |

---

## Tech Stack

- Python 3.14
- Scikit-learn (KMeans, DBSCAN, PCA, RandomForest)
- Streamlit 1.57.0
- Plotly (interactive charts)
- Pandas / NumPy
- Joblib (model serialization)
- Google Colab (development environment)

---

## Run Locally

```bash
git clone https://github.com/JeanAndre376/telecom-churn-analyzer.git
cd telecom-churn-analyzer
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
telecom_churn_app/
   app.py                    # Main Streamlit page
   requirements.txt          # Dependencies
   README.md                 # This file
   models/
      scaler.pkl             # StandardScaler
      pca_13.pkl             # PCA 13 components
      pca_2d.pkl             # PCA 2D visualization
      kmeans.pkl             # K-Means k=3
      rf_cluster1.pkl        # Random Forest Cluster 1
      feature_names.pkl      # Column names
      df_clustered.csv       # Processed dataset
   pages/
      1_EDA.py
      2_Cluster_Explorer.py
      3_Predict.py
      4_Upload_CSV.py
```

---

## Author
Built by Jean Fred A. Williama — Panama, Latin America
Open to remote ML/Data Science opportunities worldwide.

LinkedIn: [www.linkedin.com/in/jean-fred-a-williama-36905315]
Email: [jeanfred4@gmail.com]
