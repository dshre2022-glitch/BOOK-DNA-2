"""
Page 3 — Association Rule Mining
Apriori · Support · Confidence · Lift · Scatter · Bar · Heatmap · Business rules
"""

# ── IMPORTS ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    load_data, get_clean_df, run_arm,
    CLUSTER_COLORS, PRODUCT_COLS, PRODUCT_NAMES,
)

# ── CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="ARM · Book DNA", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#1a1612; }
[data-testid="stSidebar"] * { color:#faf8f4 !important; }
.rule-card { background:#faf8f4; border:1px solid #ddd8ce;
             border-radius:10px; padding:12px 16px; margin-bottom:6px; }
.insight   { background:#f3f0ea; border-left:4px solid #c8922a;
             border-radius:0 8px 8px 0; padding:13px 17px;
             font-size:.88rem; color:#4a4540; line-height:1.7; margin:8px 0; }
</style>""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────
uploaded = st.session_state.get("uploaded_file", None)
df       = load_data(uploaded)
clean    = get_clean_df(df)

st.title("🔗 Association Rule Mining")
st.caption("Apriori algorithm — discovers which products, behaviours, and barriers are chosen together.")
st.markdown("---")

# ── CONTROLS ──────────────────────────────────────────────────────────
st.subheader("⚙️ Algorithm parameters")
pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    min_sup  = st.slider("Min Support",    0.04, 0.30, 0.08, 0.01,
                          help="Fraction of respondents showing this itemset")
with pc2:
    min_conf = st.slider("Min Confidence", 0.20, 0.90, 0.40, 0.05,
                          help="P(consequent | antecedent)")
with pc3:
    min_lift = st.slider("Min Lift",       1.0,  5.0,  1.2,  0.1,
                          help="How much more likely than chance (>1 = positive)")
with pc4:
    seg_arm = st.selectbox("Segment:",
                            ["All segments"] + sorted(clean["dna_segment"].unique().tolist()),
                            key="arm_seg")

arm_data = clean if seg_arm == "All segments" else clean[clean["dna_segment"] == seg_arm]

with st.spinner("Mining association rules..."):
    try:
        rules = run_arm(arm_data, min_sup, min_conf, min_lift)
    except Exception as e:
        st.error(f"ARM error: {e}")
        rules = pd.DataFrame()

if rules.empty:
    st.warning("No rules found. Try lowering Support or Confidence thresholds.")
    st.stop()

st.success(f"Found **{len(rules)}** association rules for: *{seg_arm}*")
st.markdown("---")

# ── KPI CARDS ─────────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
k1.metric("Rules found",      len(rules))
k2.metric("Avg confidence",   f"{rules['confidence'].mean():.3f}")
k3.metric("Avg lift",         f"{rules['lift'].mean():.3f}")
k4.metric("Max lift",         f"{rules['lift'].max():.3f}")
st.markdown("---")

# ── 1. SUPPORT × CONFIDENCE SCATTER (bubble = lift) ──────────────────
st.subheader("1. Support × Confidence scatter  (bubble size = Lift)")
fig_sc = go.Figure(go.Scatter(
    x=rules["support"],
    y=rules["confidence"],
    mode="markers",
    marker=dict(
        size=rules["lift"] * 12,
        color=rules["lift"],
        colorscale="Purples",
        showscale=True,
        colorbar=dict(title="Lift", thickness=14),
        opacity=0.72,
        line=dict(color="white", width=0.6),
    ),
    text=rules["antecedents"] + " → " + rules["consequents"],
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Support: %{x:.4f}<br>"
        "Confidence: %{y:.4f}<br>"
        "Lift: %{marker.color:.4f}<extra></extra>"
    ),
))
fig_sc.update_layout(
    height=440, plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(title="Support",    showgrid=True, gridcolor="#eee"),
    yaxis=dict(title="Confidence", showgrid=True, gridcolor="#eee"),
    margin=dict(t=20,b=20),
)
st.plotly_chart(fig_sc, use_container_width=True)

# ── 2. TOP 15 RULES BY LIFT (horizontal bar) ──────────────────────────
st.markdown("---")
st.subheader("2. Top 15 rules ranked by Lift")
top15 = rules.head(15).copy()
top15["rule"]  = top15["antecedents"] + "  →  " + top15["consequents"]
top15["label"] = (top15["lift"].round(2).astype(str)
                  + "  |  conf=" + top15["confidence"].round(2).astype(str)
                  + "  |  sup=" + top15["support"].round(3).astype(str))
top15 = top15.sort_values("lift", ascending=True)

fig_bar = go.Figure(go.Bar(
    y=top15["rule"], x=top15["lift"],
    orientation="h",
    marker=dict(color=top15["lift"], colorscale="Purples", showscale=True,
                colorbar=dict(title="Lift", thickness=14)),
    text=top15["label"], textposition="outside",
))
fig_bar.update_layout(
    height=max(380, len(top15)*30+60),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(title="Lift", showgrid=True, gridcolor="#eee"),
    yaxis_title=None,
    margin=dict(t=10,b=10,l=10,r=180),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── 3. CONFIDENCE vs LIFT scatter ─────────────────────────────────────
st.markdown("---")
st.subheader("3. Confidence vs Lift — all rules")
fig_cl = go.Figure(go.Scatter(
    x=rules["confidence"], y=rules["lift"],
    mode="markers",
    marker=dict(
        size=rules["support"] * 120,
        color=rules["support"],
        colorscale="Teal",
        showscale=True,
        colorbar=dict(title="Support", thickness=14),
        opacity=0.68,
        line=dict(color="white", width=0.5),
    ),
    text=rules["antecedents"] + " → " + rules["consequents"],
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Confidence: %{x:.4f}<br>"
        "Lift: %{y:.4f}<extra></extra>"
    ),
))
fig_cl.update_layout(
    height=400, plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(title="Confidence", showgrid=True, gridcolor="#eee"),
    yaxis=dict(title="Lift",       showgrid=True, gridcolor="#eee"),
    margin=dict(t=20,b=20),
)
st.plotly_chart(fig_cl, use_container_width=True)

# ── 4. SORTABLE RULES TABLE ───────────────────────────────────────────
st.markdown("---")
st.subheader("4. Full rules table")
sort_col = st.selectbox("Sort by:", ["lift","confidence","support"], key="arm_sort")
disp = (rules.sort_values(sort_col, ascending=False)
        .reset_index(drop=True))
disp.index = disp.index + 1

st.dataframe(
    disp.style
        .background_gradient(subset=["lift"],       cmap="Purples")
        .background_gradient(subset=["confidence"], cmap="Blues")
        .background_gradient(subset=["support"],    cmap="Greens")
        .format({"support":"{:.4f}","confidence":"{:.4f}","lift":"{:.4f}"}),
    use_container_width=True, height=400,
)

# ── 5. PLAIN-ENGLISH BUSINESS ACTIONS ────────────────────────────────
st.markdown("---")
st.subheader("5. What these rules mean for your business")
for _, row in rules.head(5).iterrows():
    ant  = row["antecedents"]
    con  = row["consequents"]
    lift = row["lift"]
    conf = row["confidence"]
    sup  = row["support"]
    st.markdown(f"""
    <div class="rule-card">
        <strong>Rule:</strong> <code>{ant}</code> &rarr; <code>{con}</code><br>
        <strong>Lift: {lift:.2f}</strong> &nbsp;·&nbsp;
        Confidence: {conf:.2f} &nbsp;·&nbsp;
        Support: {sup:.3f}<br>
        <strong>Action:</strong> Customers who show <em>{ant}</em> are
        <strong>{lift:.1f}×</strong> more likely to also show <em>{con}</em>.
        Bundle these together or use one as an upsell trigger for the other.
        Rule covers <strong>{sup*100:.1f}%</strong> of your respondents.
    </div>""", unsafe_allow_html=True)

# ── 6. PRODUCT CO-INTEREST HEATMAP ───────────────────────────────────
st.markdown("---")
st.subheader("6. Product co-interest confidence heatmap")
st.caption("P(column | row) — given a customer wants the row product, probability they also want the column product.")

pp = [c for c in PRODUCT_COLS if c in arm_data.columns]
pn = [PRODUCT_NAMES[PRODUCT_COLS.index(c)] for c in pp]
n  = len(pp)
matrix = np.zeros((n, n))
for i, ci in enumerate(pp):
    for j, cj in enumerate(pp):
        if i != j:
            mask = arm_data[ci] == 1
            matrix[i][j] = arm_data.loc[mask, cj].mean() if mask.sum() > 0 else 0.0

fig_hm = go.Figure(go.Heatmap(
    z=matrix, x=pn, y=pn,
    colorscale="Purples", zmin=0, zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in matrix],
    texttemplate="%{text}",
    colorbar=dict(title="P(col|row)", thickness=14),
))
fig_hm.update_layout(
    height=420, margin=dict(t=20,b=10),
    xaxis_title=None, yaxis_title=None,
    plot_bgcolor="white", paper_bgcolor="white",
)
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("""<div class="insight"><strong>📌 ARM business strategy:</strong>
Rules with <strong>Lift > 2.5</strong> are your highest-value bundle pairs.
Rules connecting barrier flags to discount preferences are your personalised offer engine:
e.g., {barrier_delivery} → {prefer free_shipping} tells you exactly which incentive unlocks a city-tier.
Rules linking Instagram shopping with high sharing are your organic ambassador pipeline.
Every top rule is a campaign brief waiting to be executed.</div>""",
unsafe_allow_html=True)
