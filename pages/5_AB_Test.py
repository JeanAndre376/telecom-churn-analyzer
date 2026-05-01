import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2_contingency, norm, chi2
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="A/B Test", page_icon="🧪", layout="wide")
st.title("🧪 A/B Test — Retention Intervention Simulator")
st.markdown("Simulate the TechSupport + contract upgrade offer on Cluster 1 customers.")
st.markdown("---")

# ── Sidebar controls ─────────────────────────────────────
st.sidebar.header("Experiment Parameters")

n_customers      = st.sidebar.slider("Cluster 1 customers",
                                      500, 5000, 3025, step=25)
baseline_churn   = st.sidebar.slider("Control churn rate (%)",
                                      20, 70, 45) / 100
treatment_churn  = st.sidebar.slider("Treatment churn rate (%)",
                                      10, 60, 29) / 100
offer_cost       = st.sidebar.slider("Offer cost per customer ($)",
                                      5, 50, 15)
monthly_charges  = st.sidebar.slider("Avg monthly charges ($)",
                                      20, 120, 67)
alpha            = st.sidebar.selectbox("Significance level (α)",
                                         [0.01, 0.05, 0.10], index=1)
np.random.seed(st.sidebar.number_input("Random seed", value=42, step=1))

st.sidebar.markdown("---")
st.sidebar.markdown("**Your actual results:**")
st.sidebar.markdown("Control: 45.6% | Treatment: 29.4%")
st.sidebar.markdown("χ²=84.90 | p≈0 | ROI=774%")

# ── Run simulation ────────────────────────────────────────
n_control   = n_customers // 2
n_treatment = n_customers - n_control

control_churned   = np.random.binomial(1, baseline_churn,  n_control)
treatment_churned = np.random.binomial(1, treatment_churn, n_treatment)
offer_accepted    = np.random.binomial(1, 0.65, n_treatment).astype(bool)

p_ctrl = control_churned.mean()
p_trt  = treatment_churned.mean()
diff   = p_ctrl - p_trt

# Chi-square test
contingency = np.array([
    [n_control   - control_churned.sum(),   control_churned.sum()],
    [n_treatment - treatment_churned.sum(), treatment_churned.sum()]
])
chi2_stat, p_value, dof, _ = chi2_contingency(contingency, correction=False)

# Confidence interval
se       = np.sqrt((p_ctrl*(1-p_ctrl)/n_control) +
                   (p_trt*(1-p_trt)/n_treatment))
z_crit   = norm.ppf(0.975)
ci_lower = diff - z_crit * se
ci_upper = diff + z_crit * se

# Business impact
customers_saved  = int(diff * n_treatment)
revenue_saved    = customers_saved * monthly_charges * 12
total_offer_cost = n_treatment * offer_cost
net_benefit      = revenue_saved - total_offer_cost
roi              = (net_benefit / total_offer_cost) * 100
breakeven_month  = total_offer_cost / (customers_saved * monthly_charges) \
                   if customers_saved > 0 else 99

significant = p_value < alpha

# ── Top metrics ───────────────────────────────────────────
st.subheader("Experiment Results")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Control churn",   f"{p_ctrl:.1%}")
col2.metric("Treatment churn", f"{p_trt:.1%}",
            delta=f"-{diff:.1%}", delta_color="inverse")
col3.metric("P-value",         f"{p_value:.6f}",
            delta="significant ✅" if significant else "not significant ❌",
            delta_color="normal" if significant else "inverse")
col4.metric("ROI",             f"{roi:.0f}%")
col5.metric("Break-even",      f"Month {breakeven_month:.1f}")

st.markdown("---")

# ── Hypothesis result banner ──────────────────────────────
if significant:
    st.success(f"""
    ✅ **REJECT H0** — The intervention IS statistically significant!
    χ²={chi2_stat:.2f} | p={p_value:.6f} | 95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]
    **Recommendation: Roll out the offer to all Cluster 1 customers.**
    """)
else:
    st.error(f"""
    ❌ **FAIL TO REJECT H0** — Result is NOT statistically significant.
    χ²={chi2_stat:.2f} | p={p_value:.4f} | Need more data or larger effect.
    **Recommendation: Extend the experiment or revise the offer.**
    """)

st.markdown("---")

# ── Charts ────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Churn rate comparison")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=['Control (no offer)', 'Treatment (with offer)'],
        y=[p_ctrl*100, p_trt*100],
        marker_color=['#D85A30', '#1D9E75'],
        text=[f'{p_ctrl:.1%}', f'{p_trt:.1%}'],
        textposition='outside',
        textfont=dict(size=14, color='white')
    ))
    fig1.add_hline(y=baseline_churn*100, line_dash='dash',
                   line_color='gray',
                   annotation_text=f'Baseline {baseline_churn:.0%}')
    fig1.update_layout(
        yaxis_title='Churn Rate (%)',
        yaxis_range=[0, 70],
        showlegend=False,
        height=350
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Chi-square distribution")
    x_range  = np.linspace(0, 20, 500)
    y_dist   = chi2.pdf(x_range, df=1)
    crit_val = chi2.ppf(1 - alpha, df=1)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x_range, y=y_dist,
        fill='tozeroy', fillcolor='rgba(29,158,117,0.15)',
        line=dict(color='#1D9E75', width=2),
        name='Chi-square distribution'
    ))
    # Rejection region
    x_rej = x_range[x_range >= crit_val]
    y_rej = chi2.pdf(x_rej, df=1)
    fig2.add_trace(go.Scatter(
        x=x_rej, y=y_rej,
        fill='tozeroy', fillcolor='rgba(216,90,48,0.4)',
        line=dict(color='#D85A30'), name=f'Rejection region (α={alpha})'
    ))
    # Your chi2 value
    fig2.add_vline(x=min(chi2_stat, 20),
                   line_dash='dash', line_color='#534AB7',
                   annotation_text=f'χ²={chi2_stat:.1f}',
                   annotation_position='top right')
    fig2.update_layout(
        xaxis_title='Chi-square statistic',
        yaxis_title='Probability density',
        height=350, legend=dict(font=dict(size=10))
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 charts ──────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Confidence interval")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=['Control', 'Treatment'],
        y=[p_ctrl*100, p_trt*100],
        error_y=dict(
            type='constant',
            value=z_crit * se * 100,
            visible=True,
            color='#1D9E75',
            thickness=2,
            width=10
        ),
        mode='markers',
        marker=dict(size=14, color=['#D85A30', '#1D9E75']),
    ))
    fig3.update_layout(
        yaxis_title='Churn Rate (%)',
        yaxis_range=[0, 70],
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("Cumulative revenue saved")
    months       = np.arange(1, 13)
    monthly_rev  = customers_saved * monthly_charges
    cumulative   = monthly_rev * months

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=months, y=cumulative/1000,
        fill='tozeroy', fillcolor='rgba(29,158,117,0.15)',
        line=dict(color='#1D9E75', width=2.5),
        mode='lines+markers',
        name='Cumulative revenue saved'
    ))
    fig4.add_hline(y=total_offer_cost/1000,
                   line_dash='dash', line_color='#D85A30',
                   annotation_text=f'Offer cost ${total_offer_cost/1000:.0f}K')
    if breakeven_month <= 12:
        fig4.add_vline(x=breakeven_month,
                       line_dash='dot', line_color='#534AB7',
                       annotation_text=f'Break-even month {breakeven_month:.1f}')
    fig4.update_layout(
        xaxis_title='Month',
        yaxis_title='Amount ($K)',
        height=350
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Business impact table ─────────────────────────────────
st.subheader("Business Impact Summary")
col_e, col_f = st.columns(2)

with col_e:
    impact_data = {
        'Metric': [
            'Customers in experiment',
            'Customers saved from churning',
            'Annual revenue protected',
            'Total offer cost',
            'Net annual benefit',
            'ROI',
            'Break-even month'
        ],
        'Value': [
            f'{n_customers:,}',
            f'{customers_saved:,}',
            f'${revenue_saved:,.0f}',
            f'${total_offer_cost:,.0f}',
            f'${net_benefit:,.0f}',
            f'{roi:.0f}%',
            f'Month {breakeven_month:.1f}'
        ]
    }
    st.dataframe(pd.DataFrame(impact_data), use_container_width=True,
                 hide_index=True)

with col_f:
    stats_data = {
        'Statistical Metric': [
            'Chi-square statistic',
            'Degrees of freedom',
            'P-value',
            'Significance level (α)',
            'Result',
            '95% Confidence interval',
            'Effect size (Cohen\'s h)'
        ],
        'Value': [
            f'{chi2_stat:.4f}',
            f'{dof}',
            f'{p_value:.8f}',
            f'{alpha}',
            'REJECT H0 ✅' if significant else 'FAIL TO REJECT H0 ❌',
            f'[{ci_lower:.1%}, {ci_upper:.1%}]',
            f'{abs(p_ctrl - p_trt) / np.sqrt((p_ctrl+p_trt)/2*(1-(p_ctrl+p_trt)/2)):.4f}'
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True,
                 hide_index=True)

st.markdown("---")

# ── Sensitivity analysis ──────────────────────────────────
st.subheader("Sensitivity analysis — what if the effect is smaller?")
scenarios = {
    'Conservative (CI lower)': ci_lower,
    'Observed result':          diff,
    'Optimistic (CI upper)':    ci_upper,
}
sens_rows = []
for name, red in scenarios.items():
    saved = int(red * n_treatment)
    rev   = saved * monthly_charges * 12
    net   = rev - total_offer_cost
    r     = (net / total_offer_cost) * 100
    sens_rows.append({
        'Scenario':           name,
        'Churn reduction':    f'{red:.1%}',
        'Customers saved':    f'{saved:,}',
        'Revenue saved':      f'${rev:,.0f}',
        'Net benefit':        f'${net:,.0f}',
        'ROI':                f'{r:.0f}%'
    })

st.dataframe(pd.DataFrame(sens_rows), use_container_width=True,
             hide_index=True)

st.markdown("---")
st.caption("""
A/B Test Simulator — Telecom Churn Analyzer |
Adjust parameters in the sidebar to explore different scenarios.
Statistical test: Chi-square test of proportions (scipy.stats).
""")