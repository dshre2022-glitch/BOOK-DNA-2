"""
Book DNA Analytics Dashboard  —  app.py  (Home / entry point)
Run with:  streamlit run app.py
"""

# ── IMPORTS ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    load_data, get_clean_df,
    CLUSTER_COLORS, PALETTE,
    AGE_LABELS, CITY_LABELS, PAY_LABELS, PRODUCT_COLS, PRODUCT_NAMES,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book DNA Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]          { background: #1a1612; }
[data-testid="stSidebar"] *        { color: #faf8f4 !important; }
[data-testid="stSidebar"] a:hover  { color: #c8922a !important; }
.kpi-wrap  { background:#faf8f4; border:1px solid #ddd8ce; border-radius:12px;
             padding:18px 20px; text-align:center; height:110px;
             display:flex; flex-direction:column; justify-content:center; }
.kpi-val   { font-size:1.9rem; font-weight:700; color:#1a1612; line-height:1.1; }
.kpi-lbl   { font-size:0.72rem; color:#7a736b; text-transform:uppercase;
             letter-spacing:.07em; margin-top:5px; }
.kpi-delta { font-size:0.80rem; font-weight:600; color:#1a6b5a; margin-top:3px; }
.seg-row   { border-left:4px solid {color}; background:#faf8f4;
             border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:6px; }
.insight   { background:#f3f0ea; border-left:4px solid #c8922a;
             border-radius:0 8px 8px 0; padding:13px 17px;
             font-size:.88rem; color:#4a4540; line-height:1.7; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Book DNA")
    st.markdown("*Founder Analytics Dashboard*")
    st.divider()
    st.markdown("**Pages**")
    st.page_link("app.py",                         label="🏠  Home")
    st.page_link("pages/1_Descriptive.py",          label="📊  Descriptive Analysis")
    st.page_link("pages/2_Clustering.py",           label="🔵  Clustering & Personas")
    st.page_link("pages/3_ARM.py",                  label="🔗  Association Rules")
    st.page_link("pages/4_Predictive.py",           label="🔮  Predictive Models")
    st.page_link("pages/5_Prescriptive_Upload.py",  label="🎯  Prescriptive & Upload")
    st.divider()
    up = st.file_uploader("📂 Upload survey CSV", type=["csv"], key="sidebar_upload")
    if up:
        st.session_state["uploaded_file"] = up
        st.success("File loaded — all pages updated.")
    st.divider()
    st.caption("Book DNA v1.0  ·  Built with Streamlit")

# ── DATA ──────────────────────────────────────────────────────────────
uploaded = st.session_state.get("uploaded_file", None)
df       = load_data(uploaded)
clean    = get_clean_df(df)

# ── TITLE ─────────────────────────────────────────────────────────────
st.markdown("# 📚 Book DNA — Founder Analytics Dashboard")
st.markdown(
    "**Descriptive · Diagnostic · Predictive · Prescriptive** "
    "analysis for data-driven decision making"
)
st.markdown("---")

# ── KPI ROW ───────────────────────────────────────────────────────────
will_buy_pct = clean["will_buy"].mean() * 100
avg_spend    = clean["max_single_spend"].mean()
avg_nps      = clean["nps_proxy"].mean()
promoters    = (clean["nps_proxy"] >= 9).mean() * 100
top_seg      = clean["dna_segment"].value_counts().idxmax()
psm_med      = clean["psm_bargain"].median()
high_intent  = int((clean["purchase_intent"] <= 2).sum())
churn_risk_n = int(((clean["switching_tendency"] >= 3) &
                    (clean["nps_proxy"] <= 5)).sum())

cols = st.columns(7)
kpis = [
    (f"{len(df):,}",          "Total respondents",     f"{len(clean):,} clean"),
    (f"{will_buy_pct:.1f}%",  "Will buy",              f"{high_intent} high-intent"),
    (f"₹{avg_spend:,.0f}",   "Avg max spend",         "per purchase"),
    (f"{avg_nps:.1f}/10",    "NPS proxy avg",         f"{promoters:.0f}% promoters"),
    (top_seg.split()[0],      "Largest persona",       f"{clean['dna_segment'].value_counts().max()} people"),
    (f"₹{psm_med:,.0f}",     "Optimal box price",     "PSM median bargain"),
    (f"{churn_risk_n}",       "Churn-risk users",      "switching≥3 & NPS≤5"),
]
for col, (val, lbl, delta) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-wrap">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── SEGMENT OVERVIEW ─────────────────────────────────────────────────
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Respondent distribution by DNA segment")
    seg_df = clean["dna_segment"].value_counts().reset_index()
    seg_df.columns = ["Segment", "Count"]
    seg_df["Pct"]  = (seg_df["Count"] / len(clean) * 100).round(1)
    bar_colors     = [CLUSTER_COLORS.get(s, "#888") for s in seg_df["Segment"]]

    fig_seg = go.Figure(go.Bar(
        x=seg_df["Segment"], y=seg_df["Count"],
        marker_color=bar_colors,
        text=seg_df["Pct"].astype(str) + "%",
        textposition="outside",
    ))
    fig_seg.update_layout(
        height=320, margin=dict(t=20, b=10),
        xaxis_title=None, yaxis_title="Respondents",
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    st.plotly_chart(fig_seg, use_container_width=True)

with right:
    st.subheader("Segment quick stats")
    for _, row in seg_df.iterrows():
        seg   = row["Segment"]
        sub   = clean[clean["dna_segment"] == seg]
        buy   = sub["will_buy"].mean() * 100
        spend = sub["max_single_spend"].mean()
        color = CLUSTER_COLORS.get(seg, "#888")
        st.markdown(f"""
        <div style="border-left:4px solid {color};background:#faf8f4;
             border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:6px;">
            <strong style="color:{color}">{seg}</strong>
            &nbsp;·&nbsp; n={row['Count']} ({row['Pct']}%)
            &nbsp;·&nbsp; Buy intent: <strong>{buy:.0f}%</strong>
            &nbsp;·&nbsp; Avg spend: <strong>₹{spend:,.0f}</strong>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── PURCHASE INTENT BY SEGMENT ────────────────────────────────────────
st.subheader("Purchase intent rate by segment")
buy_seg = (clean.groupby("dna_segment")["will_buy"]
           .mean().reset_index()
           .rename(columns={"will_buy":"Buy Rate"}))
buy_seg["Buy %"] = (buy_seg["Buy Rate"] * 100).round(1)
buy_seg = buy_seg.sort_values("Buy %", ascending=False)

fig_buy = go.Figure(go.Bar(
    x=buy_seg["dna_segment"], y=buy_seg["Buy %"],
    marker_color=[CLUSTER_COLORS.get(s,"#888") for s in buy_seg["dna_segment"]],
    text=buy_seg["Buy %"].astype(str) + "%",
    textposition="outside",
))
fig_buy.update_layout(
    height=300, margin=dict(t=10, b=10),
    xaxis_title=None, yaxis_title="% Will Buy",
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#eee"),
)
st.plotly_chart(fig_buy, use_container_width=True)

st.markdown("---")

# ── TOP INSIGHTS ──────────────────────────────────────────────────────
st.subheader("📌 Key founder insights")
ic1, ic2, ic3 = st.columns(3)

with ic1:
    prod_demand = [(n, clean[c].mean()*100)
                   for c, n in zip(PRODUCT_COLS, PRODUCT_NAMES)
                   if c in clean.columns]
    prod_demand.sort(key=lambda x: -x[1])
    lines = "<br>".join([f"• {n}: {v:.0f}%" for n, v in prod_demand[:5]])
    st.markdown(f"""<div class="insight">
    <strong>🛍 Product demand rank</strong><br>{lines}<br>
    <em>Lead your launch with candles + journals.</em>
    </div>""", unsafe_allow_html=True)

with ic2:
    city_buy = clean.groupby("city_tier")["will_buy"].mean() * 100
    best_k   = city_buy.idxmax()
    best_c   = CITY_LABELS.get(best_k, str(best_k))
    t12_pct  = (clean["city_tier"] <= 2).mean() * 100
    st.markdown(f"""<div class="insight">
    <strong>📍 Geographic priority</strong><br>
    • Highest buy intent: <strong>{best_c} ({city_buy[best_k]:.0f}%)</strong><br>
    • Metro + Tier 1 = {t12_pct:.0f}% of high-intent respondents<br>
    • Tier 2 blocked mainly by delivery friction<br>
    <em>Launch Metro + Tier 1. Unlock Tier 2 with free shipping.</em>
    </div>""", unsafe_allow_html=True)

with ic3:
    upi_pct    = (clean["payment_method"] == 1).mean() * 100
    credit_pct = (clean["payment_method"] == 3).mean() * 100
    st.markdown(f"""<div class="insight">
    <strong>💳 Payment & pricing</strong><br>
    • UPI dominant: <strong>{upi_pct:.0f}%</strong><br>
    • Credit card users: <strong>{credit_pct:.0f}%</strong> (highest spenders)<br>
    • PSM optimal box price: ₹{int(psm_med)}<br>
    <em>Price MVP box ₹299–₹349. Premium tier ₹599.</em>
    </div>""", unsafe_allow_html=True)

st.caption(
    "Use the sidebar to navigate to Descriptive Analysis, Clustering, "
    "Association Rules, Predictive Models, and Prescriptive Strategy pages."
)
