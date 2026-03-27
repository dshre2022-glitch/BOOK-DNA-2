"""
Page 5 — Prescriptive Analysis + New Data Upload & Predict
Discount engine · Bundle recommender · Churn flags · Focus customer
CSV upload → DNA segment + buy probability + spend prediction + recommended offer
"""

# ── IMPORTS ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    load_data, get_clean_df,
    train_classifiers, train_regressors, train_kmeans, get_cluster_segment_map,
    CLUSTER_COLORS, PRESCRIPTIVE,
    PRODUCT_COLS, PRODUCT_NAMES,
    CLF_FEATURES, REG_FEATURES, CLUSTER_FEATURES,
    PAY_LABELS, CITY_LABELS, DISCOUNT_LABELS,
)

# ── CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prescriptive · Book DNA", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#1a1612; }
[data-testid="stSidebar"] * { color:#faf8f4 !important; }
.pcard  { background:#faf8f4; border:1px solid #ddd8ce; border-radius:12px;
          padding:16px 18px; margin-bottom:8px; }
.insight{ background:#f3f0ea; border-left:4px solid #c8922a;
          border-radius:0 8px 8px 0; padding:13px 17px;
          font-size:.88rem; color:#4a4540; line-height:1.7; margin:8px 0; }
.upload-zone { background:#f9f7f3; border:2px dashed #ddd8ce; border-radius:12px;
               padding:30px; text-align:center; color:#7a736b; }
</style>""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────
uploaded = st.session_state.get("uploaded_file", None)
df       = load_data(uploaded)
clean    = get_clean_df(df)

st.title("🎯 Prescriptive Analysis & New Customer Prediction")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Prescriptive Strategy", "📍 Focus Customer", "📤 Upload & Predict"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PRESCRIPTIVE STRATEGY
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Segment-by-segment prescriptive recommendations")

    # ── PRIORITY MATRIX TABLE ─────────────────────────────────────────
    st.markdown("#### Marketing priority matrix")
    rows_p = []
    for seg in CLUSTER_COLORS:
        sub = clean[clean["dna_segment"] == seg]
        p   = PRESCRIPTIVE[seg]
        rows_p.append({
            "Segment":    seg,
            "Size":       len(sub),
            "Buy %":      f"{sub['will_buy'].mean()*100:.0f}%",
            "Avg spend":  f"₹{sub['max_single_spend'].mean():,.0f}",
            "Avg NPS":    f"{sub['nps_proxy'].mean():.1f}",
            "Churn risk": f"{sub['switching_tendency'].mean():.1f}/4",
            "Priority":   p["priority"],
            "LTV band":   p["ltv_band"],
        })
    st.dataframe(pd.DataFrame(rows_p).set_index("Segment"),
                 use_container_width=True)

    st.markdown("---")

    # ── SEGMENT ACTION CARDS ──────────────────────────────────────────
    st.markdown("#### Prescriptive action cards")
    for seg, color in CLUSTER_COLORS.items():
        p = PRESCRIPTIVE[seg]
        with st.expander(f"{p['priority']}  **{seg}**",
                         expanded=(seg == "Midnight Escapist")):
            cx1, cx2, cx3 = st.columns(3)
            with cx1:
                st.markdown(f"""<div class="pcard" style="border-top:3px solid {color}">
                <strong style="color:{color}">💸 Offer</strong><br>{p['offer']}<br><br>
                <strong style="color:{color}">💰 LTV band</strong><br>{p['ltv_band']}
                </div>""", unsafe_allow_html=True)
            with cx2:
                st.markdown(f"""<div class="pcard" style="border-top:3px solid {color}">
                <strong style="color:{color}">📦 Bundle</strong><br>{p['bundle']}<br><br>
                <strong style="color:{color}">⚠️ Churn risk</strong><br>{p['churn_risk']}
                </div>""", unsafe_allow_html=True)
            with cx3:
                st.markdown(f"""<div class="pcard" style="border-top:3px solid {color}">
                <strong style="color:{color}">📢 Channel</strong><br>{p['channel']}<br><br>
                <strong style="color:{color}">⏰ Best timing</strong><br>{p['timing']}
                </div>""", unsafe_allow_html=True)

    # ── DISCOUNT PREFERENCE BAR ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Discount preference by segment")
    segs_d = sorted(clean["dna_segment"].unique())
    disc_data = {}
    for seg in segs_d:
        sub = clean[clean["dna_segment"] == seg]
        vc  = sub["discount_preference"].value_counts(normalize=True)
        disc_data[seg] = {DISCOUNT_LABELS.get(k, str(k)): round(v*100,1)
                          for k,v in vc.items()}
    disc_df = pd.DataFrame(disc_data).fillna(0).T
    fig_disc = px.bar(
        disc_df.reset_index(), x="index", y=disc_df.columns.tolist(),
        barmode="group", height=360,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"index":"Segment","value":"% preference","variable":"Offer type"},
    )
    fig_disc.update_layout(
        xaxis_title=None, yaxis_title="% preference",
        plot_bgcolor="white", paper_bgcolor="white",
        legend_title="Offer type",
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        margin=dict(t=10,b=10),
    )
    st.plotly_chart(fig_disc, use_container_width=True)

    # ── CHURN RISK FLAGS ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### High churn-risk customers (switching≥3 AND NPS≤5 AND will_buy=1)")
    churn = clean[
        (clean["switching_tendency"] >= 3) &
        (clean["nps_proxy"]          <= 5) &
        (clean["will_buy"]           == 1)
    ]
    st.metric("High churn-risk would-be buyers",
              len(churn),
              f"{len(churn)/max(len(clean),1)*100:.1f}% of clean dataset")

    if len(churn) > 0:
        churn_segs = churn["dna_segment"].value_counts().reset_index()
        churn_segs.columns = ["Segment","Count"]
        fig_ch = go.Figure(go.Bar(
            x=churn_segs["Segment"], y=churn_segs["Count"],
            marker_color=[CLUSTER_COLORS.get(s,"#888") for s in churn_segs["Segment"]],
            text=churn_segs["Count"], textposition="outside",
        ))
        fig_ch.update_layout(
            height=300, xaxis_title=None,
            yaxis=dict(title="Count", showgrid=True, gridcolor="#eee"),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10,b=10),
        )
        st.plotly_chart(fig_ch, use_container_width=True)
        st.markdown("""<div class="insight"><strong>📌 Churn prevention:</strong>
        Flag these customers before they subscribe.
        Offer a locking incentive: annual plan at 30% saving, a loyalty-point welcome bonus,
        or a money-back guarantee.
        Do NOT onboard high-risk customers on a standard monthly plan without a retention hook.</div>""",
        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — FOCUS CUSTOMER
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📍 The Focus Customer — highest-value acquisition target")
    st.caption("Intersection of: high purchase intent + high spend + low churn risk + high sharing.")

    focus = clean[
        (clean["purchase_intent"]    <= 2) &
        (clean["max_single_spend"]   >= 750) &
        (clean["switching_tendency"] <= 2) &
        (clean["social_sharing_level"] <= 2) &
        (clean["nps_proxy"]          >= 7)
    ].copy()

    st.metric("Focus customers",
              len(focus),
              f"{len(focus)/max(len(clean),1)*100:.1f}% of dataset · highest ROI pool")

    if len(focus) > 0:
        fa2, fb2 = st.columns(2)

        with fa2:
            st.markdown("#### Who they are")
            top_seg_f = focus["dna_segment"].value_counts().idxmax()
            st.markdown(f"""
| Attribute | Focus customer profile |
|---|---|
| **Top segment** | {top_seg_f} |
| **Avg NPS proxy** | {focus['nps_proxy'].mean():.1f}/10 |
| **Avg stress score** | {focus['stress_score'].mean():.1f}/16 |
| **Avg predicted spend** | ₹{focus['max_single_spend'].mean():,.0f} |
| **Social sharing** | {focus['social_sharing_level'].mean():.1f}/4 |
| **Payment mode** | {PAY_LABELS.get(int(focus['payment_method'].mode()[0]),'UPI')} |
| **City (mode)** | {CITY_LABELS.get(int(focus['city_tier'].mode()[0]),'—')} |
            """)

        with fb2:
            fc_segs = focus["dna_segment"].value_counts().reset_index()
            fc_segs.columns = ["Segment","Count"]
            fig_fc = go.Figure(go.Bar(
                x=fc_segs["Segment"], y=fc_segs["Count"],
                marker_color=[CLUSTER_COLORS.get(s,"#888") for s in fc_segs["Segment"]],
                text=fc_segs["Count"], textposition="outside",
            ))
            fig_fc.update_layout(
                title="Focus customers by segment",
                height=300, xaxis_title=None,
                yaxis=dict(showgrid=True, gridcolor="#eee"),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=50,b=10),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        # Product interests
        st.markdown("#### What they want to buy")
        pp_fc = [(PRODUCT_NAMES[PRODUCT_COLS.index(c)], focus[c].mean()*100)
                 for c in PRODUCT_COLS if c in focus.columns]
        pp_fc.sort(key=lambda x: -x[1])
        fig_pfoc = go.Figure(go.Bar(
            x=[x[0] for x in pp_fc], y=[x[1] for x in pp_fc],
            marker_color="#5C3D8F",
            text=[f"{x[1]:.0f}%" for x in pp_fc], textposition="outside",
        ))
        fig_pfoc.update_layout(
            height=290, xaxis_title=None,
            yaxis=dict(title="% interested", range=[0,100],
                       showgrid=True, gridcolor="#eee"),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10,b=10),
        )
        st.plotly_chart(fig_pfoc, use_container_width=True)

        st.markdown("""<div class="insight"><strong>📌 Focus customer playbook:</strong>
        Put 60–70% of Year 1 paid marketing budget against this profile.
        They buy willingly, share organically, and don't churn.
        Acquire via Instagram Reels (late-night) and BookTok.
        Lead with candle + journal bundle at ₹599.
        Upsell to subscription within 30 days of first purchase.</div>""",
        unsafe_allow_html=True)
    else:
        st.info("No focus customers found — try relaxing the criteria.")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — UPLOAD & PREDICT
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📤 Upload new survey data — predict DNA segment, buy intent & spend")
    st.caption(
        "Upload a CSV with the same column structure as the training survey. "
        "The pipeline assigns DNA segments, predicts purchase probability, estimates spend, "
        "and recommends an offer for each respondent."
    )

    # ── TEMPLATE DOWNLOAD ─────────────────────────────────────────────
    sample_cols = (
        ["respondent_id"] +
        [c for c in CLF_FEATURES if c in clean.columns] +
        [c for c in REG_FEATURES  if c in clean.columns
         and c not in CLF_FEATURES]
    )
    sample_cols = list(dict.fromkeys(sample_cols))
    template_df = clean[sample_cols].head(3).copy()
    st.download_button(
        label="⬇️ Download template CSV (3 sample rows — use as column guide)",
        data=template_df.to_csv(index=False),
        file_name="book_dna_upload_template.csv",
        mime="text/csv",
    )
    st.markdown("---")

    new_file = st.file_uploader("Upload your new survey CSV", type=["csv"], key="new_csv")

    if new_file is not None:
        new_df = pd.read_csv(new_file)
        st.success(f"Loaded {len(new_df):,} new respondents · {new_df.shape[1]} columns")
        st.dataframe(new_df.head(3), use_container_width=True)

        with st.spinner("Running prediction pipeline..."):
            # Load trained models
            clf_r2, clf_f2, _, rf_clf2, _ = train_classifiers(clean)
            reg_r2, reg_f2, _, rf_reg2, _ = train_regressors(clean)
            km2, km_sc2, km_lbl2, km_f2   = train_kmeans(clean, k=5)
            cmap2 = get_cluster_segment_map(clean, km_lbl2)

            out = new_df.copy()

            # ── CLUSTER ───────────────────────────────────────────────
            clus_cols = [c for c in km_f2 if c in out.columns]
            if len(clus_cols) >= 5:
                Xc = out[clus_cols].fillna(0)
                for mc in km_f2:
                    if mc not in Xc.columns:
                        Xc[mc] = 0
                Xc   = Xc[km_f2]
                Xc_s = km_sc2.transform(Xc)
                lbl_new = km2.predict(Xc_s)
                out["predicted_dna_segment"] = [cmap2.get(l, f"Cluster {l}")
                                                for l in lbl_new]
            else:
                out["predicted_dna_segment"] = "Insufficient features"

            # ── BUY INTENT ────────────────────────────────────────────
            clf_c = [c for c in clf_f2 if c in out.columns]
            if len(clf_c) >= 5:
                Xcl = out[clf_c].fillna(0)
                for mc in clf_f2:
                    if mc not in Xcl.columns:
                        Xcl[mc] = 0
                Xcl  = Xcl[clf_f2]
                prob = rf_clf2.predict_proba(Xcl)[:, 1]
                out["buy_probability"] = np.round(prob, 4)
                out["buy_prediction"]  = (prob >= 0.5).astype(int)
                out["priority_lead"]   = (prob >= 0.65).astype(bool)
            else:
                out["buy_probability"] = np.nan
                out["buy_prediction"]  = np.nan
                out["priority_lead"]   = False

            # ── SPEND ─────────────────────────────────────────────────
            reg_c = [c for c in reg_f2 if c in out.columns]
            if len(reg_c) >= 4:
                Xrg = out[reg_c].fillna(0)
                for mc in reg_f2:
                    if mc not in Xrg.columns:
                        Xrg[mc] = 0
                Xrg  = Xrg[reg_f2]
                out["predicted_spend"] = np.round(rf_reg2.predict(Xrg), 0).astype(int)
            else:
                out["predicted_spend"] = np.nan

            # ── OFFER + BUNDLE ────────────────────────────────────────
            def _offer(row):
                s = str(row.get("predicted_dna_segment",""))
                return PRESCRIPTIVE.get(s, {}).get("offer",  "Standard intro offer")
            def _bundle(row):
                s = str(row.get("predicted_dna_segment",""))
                return PRESCRIPTIVE.get(s, {}).get("bundle", "Curated book bundle")

            out["recommended_offer"]  = out.apply(_offer,  axis=1)
            out["recommended_bundle"] = out.apply(_bundle, axis=1)

        st.markdown("---")
        st.subheader("Prediction results")

        # KPI summary
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("New respondents", len(out))
        if out["buy_probability"].notna().any():
            s2.metric("Predicted will buy",    f"{out['buy_prediction'].mean()*100:.1f}%")
            s3.metric("Priority leads (≥65%)", int(out["priority_lead"].sum()))
        if out["predicted_spend"].notna().any():
            s4.metric("Avg predicted spend",   f"₹{out['predicted_spend'].mean():,.0f}")

        # Segment bar
        if "predicted_dna_segment" in out.columns:
            sn = out["predicted_dna_segment"].value_counts().reset_index()
            sn.columns = ["Segment","Count"]
            fig_sn = go.Figure(go.Bar(
                x=sn["Segment"], y=sn["Count"],
                marker_color=[CLUSTER_COLORS.get(s,"#888") for s in sn["Segment"]],
                text=sn["Count"], textposition="outside",
            ))
            fig_sn.update_layout(
                title="Predicted DNA segments — new respondents",
                height=300, xaxis_title=None,
                yaxis=dict(showgrid=True, gridcolor="#eee"),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=50,b=10),
            )
            st.plotly_chart(fig_sn, use_container_width=True)

        # Results preview
        disp_cols = (
            (["respondent_id"] if "respondent_id" in out.columns else []) +
            [c for c in ["predicted_dna_segment","buy_probability","buy_prediction",
                         "priority_lead","predicted_spend",
                         "recommended_offer","recommended_bundle"]
             if c in out.columns]
        )
        st.dataframe(out[disp_cols].head(25), use_container_width=True)

        # Download
        st.download_button(
            label="⬇️ Download enriched predictions CSV",
            data=out.to_csv(index=False),
            file_name="book_dna_predictions_enriched.csv",
            mime="text/csv",
        )

        st.markdown("""<div class="insight"><strong>📌 How to use the enriched file:</strong>
        Share with your marketing team — each row is a personalised campaign brief.<br>
        Filter <code>priority_lead = True</code> for highest-ROI outreach.<br>
        Use <code>recommended_offer</code> to personalise each communication.<br>
        Use <code>predicted_dna_segment</code> to assign respondents to the correct content funnel.</div>""",
        unsafe_allow_html=True)

    else:
        st.markdown("""<div class="upload-zone">
        <strong>Drop your new survey CSV here or click Browse files above.</strong><br><br>
        Download the template above to ensure column compatibility.<br>
        The pipeline will auto-assign DNA segments, predict buy probability,
        estimate spend, and recommend a personalised offer for each row.
        </div>""", unsafe_allow_html=True)
