"""
Page 1 — Descriptive Analysis
Demographics · Van Westendorp PSM · Product heatmap · Reading habits
"""

# ── IMPORTS ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    load_data, get_clean_df, psm_chart,
    AGE_LABELS, GENDER_LABELS, CITY_LABELS, OCC_LABELS,
    CLUSTER_COLORS, PALETTE,
    PRODUCT_COLS, PRODUCT_NAMES,
    RH_LABELS, FORMAT_LABELS, MOOD_LABELS,
)

# ── CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Descriptive · Book DNA", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"]    { background:#1a1612; }
[data-testid="stSidebar"] *  { color:#faf8f4 !important; }
.insight { background:#f3f0ea; border-left:4px solid #c8922a;
           border-radius:0 8px 8px 0; padding:13px 17px;
           font-size:.88rem; color:#4a4540; line-height:1.7; margin:8px 0; }
</style>""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────
uploaded = st.session_state.get("uploaded_file", None)
df       = load_data(uploaded)
clean    = get_clean_df(df)

st.title("📊 Descriptive Analysis")
st.caption(f"{len(clean):,} clean respondents · {len(df):,} total")
st.markdown("---")

# ── FILTERS ───────────────────────────────────────────────────────────
with st.expander("🔧 Filters", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        seg_sel  = st.multiselect("Segment", sorted(clean["dna_segment"].unique()),
                                   default=sorted(clean["dna_segment"].unique()))
    with c2:
        city_sel = st.multiselect("City tier", sorted(clean["city_tier"].unique()),
                                   default=sorted(clean["city_tier"].unique()),
                                   format_func=lambda x: CITY_LABELS.get(x, x))
    with c3:
        age_sel  = st.multiselect("Age group", sorted(clean["age_group"].unique()),
                                   default=sorted(clean["age_group"].unique()),
                                   format_func=lambda x: AGE_LABELS.get(x, x))

filt = clean[
    clean["dna_segment"].isin(seg_sel) &
    clean["city_tier"].isin(city_sel) &
    clean["age_group"].isin(age_sel)
].copy()
st.caption(f"Filtered: {len(filt):,} respondents")

# helper bar chart
def make_bar(col, label_map, title, color="#5C3D8F"):
    cnt = filt[col].value_counts().reset_index()
    cnt.columns = ["code","n"]
    cnt["label"] = cnt["code"].map(label_map).fillna(cnt["code"].astype(str))
    cnt = cnt.sort_values("code")
    fig = go.Figure(go.Bar(
        x=cnt["label"], y=cnt["n"],
        marker_color=color,
        text=cnt["n"], textposition="outside",
    ))
    fig.update_layout(
        title=title, height=290, margin=dict(t=42,b=8),
        xaxis=dict(title=None, tickangle=-30),
        yaxis=dict(title=None, showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

# ── 1. DEMOGRAPHICS ───────────────────────────────────────────────────
st.subheader("1. Demographics")
d1,d2,d3,d4 = st.columns(4)
d1.plotly_chart(make_bar("age_group",  AGE_LABELS,    "Age groups",  "#5C3D8F"), use_container_width=True)
d2.plotly_chart(make_bar("gender",     GENDER_LABELS, "Gender",      "#C8922A"), use_container_width=True)
d3.plotly_chart(make_bar("city_tier",  CITY_LABELS,   "City tier",   "#1A6B5A"), use_container_width=True)
d4.plotly_chart(make_bar("occupation", OCC_LABELS,    "Occupation",  "#D45379"), use_container_width=True)

# income
inc_map = {2500:"<₹5K",10000:"₹5–15K",22500:"₹15–30K",
           45000:"₹30–60K",80000:"₹60K–1L",120000:">₹1L"}
inc_cnt = filt["monthly_income_midpoint"].value_counts().sort_index().reset_index()
inc_cnt.columns = ["code","n"]
inc_cnt["label"] = inc_cnt["code"].map(inc_map).fillna(inc_cnt["code"].astype(str))
fig_inc = go.Figure(go.Bar(
    x=inc_cnt["label"], y=inc_cnt["n"],
    marker_color="#378ADD", text=inc_cnt["n"], textposition="outside",
))
fig_inc.update_layout(
    title="Monthly income distribution (₹)", height=270, margin=dict(t=42,b=8),
    xaxis_title=None, yaxis=dict(showgrid=True, gridcolor="#eee"),
    plot_bgcolor="white", paper_bgcolor="white",
)
st.plotly_chart(fig_inc, use_container_width=True)

st.markdown("""<div class="insight"><strong>📌 Demographic insight:</strong>
Respondents skew 18–28 with UPI as dominant payment method.
Metro + Tier 1 cities drive the highest purchase intent.
Focus initial distribution and influencer spend in these geographies.</div>""",
unsafe_allow_html=True)

# ── 2. VAN WESTENDORP PSM ─────────────────────────────────────────────
st.markdown("---")
st.subheader("2. Van Westendorp Price Sensitivity Meter")
st.caption("Four cumulative curves reveal the psychologically safe subscription price range.")

pa, pb = st.columns([3,1])
with pa:
    seg_opts = ["All segments"] + sorted(clean["dna_segment"].unique().tolist())
    psm_seg  = st.selectbox("PSM view:", seg_opts, key="psm_sel")
with pb:
    psm_data = filt if psm_seg == "All segments" else filt[filt["dna_segment"] == psm_seg]
    st.metric("Respondents", len(psm_data.dropna(subset=["psm_bargain"])))

fig_psm, pmc, opp, pme = psm_chart(psm_data, None)
if fig_psm is not None:
    st.plotly_chart(fig_psm, use_container_width=True)
    m1,m2,m3 = st.columns(3)
    m1.metric("Point of Marginal Cheapness (floor)", f"₹{pmc}", "Do not price below this")
    m2.metric("Optimal Price Point",                 f"₹{opp}", "Minimum overall rejection")
    m3.metric("Point of Marginal Expensiveness",     f"₹{pme}", "Upper price ceiling")
    st.markdown(f"""<div class="insight"><strong>📌 Pricing action:</strong>
    Price your MVP subscription box at <strong>₹{opp}</strong> (Optimal Price Point).
    Acceptable range is ₹{pmc} → ₹{pme}.
    Premium tier should sit at ₹{pme - 50} to stay inside the ceiling.</div>""",
    unsafe_allow_html=True)
else:
    st.info("Not enough data for PSM. Lower filters or select a larger segment.")

# ── 3. PRODUCT INTEREST HEATMAP ───────────────────────────────────────
st.markdown("---")
st.subheader("3. Product interest heatmap by segment")
prod_present = [c for c in PRODUCT_COLS if c in filt.columns]
name_present = [PRODUCT_NAMES[PRODUCT_COLS.index(c)] for c in prod_present]
segs         = sorted(filt["dna_segment"].unique())
z_vals       = [[filt[filt["dna_segment"]==s][c].mean()*100 for c in prod_present]
                for s in segs]
fig_heat = go.Figure(go.Heatmap(
    z=z_vals, x=name_present, y=segs,
    colorscale="Purples",
    text=[[f"{v:.0f}%" for v in row] for row in z_vals],
    texttemplate="%{text}",
    colorbar=dict(title="% interested"),
))
fig_heat.update_layout(
    height=340, margin=dict(t=20,b=10),
    xaxis_title=None, yaxis_title=None,
    plot_bgcolor="white", paper_bgcolor="white",
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── 4. GENRE POPULARITY ───────────────────────────────────────────────
st.markdown("---")
st.subheader("4. Genre popularity")
genre_cols  = [c for c in filt.columns if c.startswith("genre_")]
genre_names = [c.replace("genre_","").replace("_"," ").title() for c in genre_cols]
genre_pcts  = [filt[c].mean()*100 for c in genre_cols]
gdf = pd.DataFrame({"Genre":genre_names,"Pct":genre_pcts}).sort_values("Pct",ascending=True)
fig_genre = go.Figure(go.Bar(
    y=gdf["Genre"], x=gdf["Pct"], orientation="h",
    marker_color="#C8922A",
    text=[f"{v:.1f}%" for v in gdf["Pct"]], textposition="outside",
))
fig_genre.update_layout(
    height=360, margin=dict(t=10,b=10,r=60),
    xaxis=dict(title="% respondents", range=[0,92], showgrid=True, gridcolor="#eee"),
    yaxis_title=None, plot_bgcolor="white", paper_bgcolor="white",
)
st.plotly_chart(fig_genre, use_container_width=True)

# ── 5. READING HABITS ────────────────────────────────────────────────
st.markdown("---")
st.subheader("5. Reading habits")
h1, h2, h3 = st.columns(3)

with h1:
    bpm = filt["books_per_month"].value_counts().sort_index().reset_index()
    bpm.columns = ["bpm","n"]
    fig_bpm = go.Figure(go.Bar(
        x=bpm["bpm"].astype(str), y=bpm["n"],
        marker_color="#5DCAA5", text=bpm["n"], textposition="outside",
    ))
    fig_bpm.update_layout(
        title="Books per month", height=280, margin=dict(t=42,b=8),
        xaxis_title=None,
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_bpm, use_container_width=True)

with h2:
    time_map = {"reads_morning":"Morning","reads_commute":"Commute",
                "reads_afternoon":"Afternoon","reads_evening":"Evening",
                "reads_latenight":"Late night","reads_weekend":"Weekend"}
    t_labels = [v for k,v in time_map.items() if k in filt.columns]
    t_pcts   = [filt[k].mean()*100 for k in time_map if k in filt.columns]
    fig_time = go.Figure(go.Bar(
        x=t_labels, y=t_pcts, marker_color="#378ADD",
        text=[f"{v:.0f}%" for v in t_pcts], textposition="outside",
    ))
    fig_time.update_layout(
        title="When do they read?", height=280, margin=dict(t=42,b=8),
        xaxis_title=None,
        yaxis=dict(range=[0,88], showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_time, use_container_width=True)

with h3:
    rhs = filt["reading_habit_status"].value_counts().reset_index()
    rhs.columns = ["code","n"]
    rhs["label"] = rhs["code"].map(RH_LABELS).fillna(rhs["code"].astype(str))
    fig_rhs = go.Figure(go.Pie(
        labels=rhs["label"], values=rhs["n"],
        hole=0.42, marker_colors=PALETTE[:len(rhs)],
    ))
    fig_rhs.update_layout(
        title="Reading habit status", height=280,
        margin=dict(t=42,l=10,r=10,b=10),
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_rhs, use_container_width=True)

st.markdown("""<div class="insight"><strong>📌 Reading habits insight:</strong>
Late-night (after 10 PM) is the single biggest reading window.
Schedule Instagram posts and push notifications between 9–11 PM IST.
The "Want to return to reading" group is your fastest re-activation segment —
they have the intent but lack the trigger. Book DNA is that trigger.</div>""",
unsafe_allow_html=True)
