"""
Page 4 — Predictive Models
Classification : Accuracy · Precision · Recall · F1-Score · ROC-AUC
               · Confusion Matrix · ROC Curve · Feature Importance · Decision Tree rules
Regression     : R² · RMSE · Predicted vs Actual · Residuals · Ridge coefficients
Live predict   : Enter a profile → instant prediction
"""

# ── IMPORTS ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import export_text

from utils import (
    load_data, get_clean_df,
    train_classifiers, train_format_classifier, train_regressors,
    CLF_FEATURES, REG_FEATURES,
    CLUSTER_COLORS, FORMAT_CLASS, PAY_LABELS, CITY_LABELS,
)

# ── CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictive · Book DNA", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#1a1612; }
[data-testid="stSidebar"] * { color:#faf8f4 !important; }
.mcard { background:#faf8f4; border:1px solid #ddd8ce; border-radius:10px;
         padding:14px 16px; text-align:center; }
.mval  { font-size:1.65rem; font-weight:700; color:#1a1612; }
.mlbl  { font-size:.72rem; color:#7a736b; text-transform:uppercase;
         letter-spacing:.07em; margin-top:5px; }
.insight { background:#f3f0ea; border-left:4px solid #c8922a;
           border-radius:0 8px 8px 0; padding:13px 17px;
           font-size:.88rem; color:#4a4540; line-height:1.7; margin:8px 0; }
</style>""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────
uploaded = st.session_state.get("uploaded_file", None)
df       = load_data(uploaded)
clean    = get_clean_df(df)

st.title("🔮 Predictive Models")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Buy-Intent Classification",
    "📖 Format Classification",
    "💰 Spend Regression",
    "🔬 Live Prediction",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — BUY INTENT CLASSIFICATION
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Will the customer buy? — Binary classification")
    st.caption("Target: `will_buy` (1=Yes, 0=No). Three models trained and compared.")

    with st.spinner("Training classifiers..."):
        clf_res, clf_feats, clf_scaler, rf_model, dt_model = train_classifiers(clean)

    model_sel = st.selectbox("Inspect model:", list(clf_res.keys()), key="clf_sel")
    res = clf_res[model_sel]

    # ── METRIC CARDS ──────────────────────────────────────────────────
    st.markdown("#### Performance metrics")
    m1,m2,m3,m4,m5 = st.columns(5)
    for col, lbl, val in [
        (m1, "Accuracy",  f"{res['accuracy']*100:.2f}%"),
        (m2, "Precision", f"{res['precision']*100:.2f}%"),
        (m3, "Recall",    f"{res['recall']*100:.2f}%"),
        (m4, "F1-Score",  f"{res['f1']*100:.2f}%"),
        (m5, "ROC-AUC",   f"{res['roc_auc']:.4f}"),
    ]:
        with col:
            st.markdown(f"""<div class="mcard">
            <div class="mval">{val}</div>
            <div class="mlbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── CONFUSION MATRIX + ROC CURVE ──────────────────────────────────
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion matrix")
        cm   = res["cm"]
        lbls = ["Won't buy (0)", "Will buy (1)"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=lbls, y=lbls,
            colorscale=[[0,"#EEEDFE"],[1,"#5C3D8F"]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont=dict(size=20),
            showscale=False,
        ))
        fig_cm.update_layout(
            height=330, margin=dict(t=20,b=20),
            xaxis_title="Predicted", yaxis_title="Actual",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        tn,fp,fn,tp = cm.ravel()
        st.caption(f"TP={tp} · FP={fp} · FN={fn} · TN={tn}")

    with col_roc:
        st.markdown("#### ROC curve — all models")
        roc_colors = {
            "Random Forest":       "#5C3D8F",
            "Logistic Regression": "#C8922A",
            "Decision Tree":       "#1A6B5A",
        }
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dash", color="#ccc", width=1.5),
            name="Random (AUC=0.50)",
        ))
        for name, r in clf_res.items():
            fig_roc.add_trace(go.Scatter(
                x=r["fpr"], y=r["tpr"], mode="lines",
                line=dict(color=roc_colors.get(name,"#888"),
                           width=3 if name == model_sel else 1.6),
                name=f"{name}  (AUC={r['roc_auc']:.3f})",
            ))
        fig_roc.update_layout(
            height=330, margin=dict(t=20,b=20),
            xaxis=dict(title="False Positive Rate", range=[0,1],
                       showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="True Positive Rate", range=[0,1],
                       showgrid=True, gridcolor="#eee"),
            legend=dict(x=0.32, y=0.08, font=dict(size=10)),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Feature importance — Random Forest")
    fi_df = pd.DataFrame({
        "Feature":    clf_feats,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=True).tail(20)

    fig_fi = go.Figure(go.Bar(
        y=fi_df["Feature"], x=fi_df["Importance"],
        orientation="h",
        marker=dict(color=fi_df["Importance"],
                    colorscale="Purples", showscale=False),
        text=[f"{v:.4f}" for v in fi_df["Importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        height=530, margin=dict(t=10,b=10,l=10,r=90),
        xaxis=dict(title="Importance score",
                   showgrid=True, gridcolor="#eee"),
        yaxis_title=None,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── DECISION TREE TEXT RULES ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Decision tree rules (max depth 5)")
    feats_dt = [c for c in CLF_FEATURES if c in clean.columns]
    tree_txt = export_text(dt_model, feature_names=feats_dt, max_depth=5)
    if len(tree_txt) > 4000:
        tree_txt = tree_txt[:4000] + "\n... [truncated for readability]"
    with st.expander("View decision tree"):
        st.code(tree_txt)

    # ── MODEL COMPARISON TABLE ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Model comparison")
    comp_rows = []
    for name, r in clf_res.items():
        comp_rows.append({
            "Model":     name,
            "Accuracy":  f"{r['accuracy']*100:.2f}%",
            "Precision": f"{r['precision']*100:.2f}%",
            "Recall":    f"{r['recall']*100:.2f}%",
            "F1-Score":  f"{r['f1']*100:.2f}%",
            "ROC-AUC":   f"{r['roc_auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(comp_rows).set_index("Model"),
                 use_container_width=True)

    st.markdown("""<div class="insight"><strong>📌 Classification insight:</strong>
    Top predictors of purchase intent: <strong>nps_proxy</strong>, <strong>stress_score</strong>,
    <strong>personalization_importance</strong>, <strong>switching_tendency</strong>.<br>
    Customers who would recommend a reading brand AND are highly stressed are your most reliable buyers.
    Target this intersection first with your paid acquisition spend.</div>""",
    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — FORMAT PREFERENCE (MULTI-CLASS)
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Reading format preference — 3-class classification")
    st.caption("Classes: Physical / Mixed / Digital. Weighted metrics used for multi-class evaluation.")

    with st.spinner("Training format classifier..."):
        fmt_model, fmt_feats, fmt_m = train_format_classifier(clean)

    f1,f2,f3,f4 = st.columns(4)
    for col, lbl, val in [
        (f1, "Accuracy",          f"{fmt_m['accuracy']*100:.2f}%"),
        (f2, "Precision (wtd.)",  f"{fmt_m['precision']*100:.2f}%"),
        (f3, "Recall (wtd.)",     f"{fmt_m['recall']*100:.2f}%"),
        (f4, "F1-Score (wtd.)",   f"{fmt_m['f1']*100:.2f}%"),
    ]:
        with col:
            st.markdown(f"""<div class="mcard">
            <div class="mval">{val}</div>
            <div class="mlbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    fa, fb = st.columns(2)

    with fa:
        st.markdown("#### Confusion matrix")
        cm2  = fmt_m["cm"]
        cls2 = fmt_m["classes"]
        fig_cm2 = go.Figure(go.Heatmap(
            z=cm2, x=cls2, y=cls2,
            colorscale=[[0,"#E1F5EE"],[1,"#0F6E56"]],
            text=cm2, texttemplate="<b>%{text}</b>",
            textfont=dict(size=18), showscale=False,
        ))
        fig_cm2.update_layout(
            height=320, margin=dict(t=20,b=20),
            xaxis_title="Predicted", yaxis_title="Actual",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_cm2, use_container_width=True)

    with fb:
        st.markdown("#### Format distribution")
        fd = clean["format_class"].map(FORMAT_CLASS).value_counts().reset_index()
        fd.columns = ["Format","Count"]
        fig_fd = go.Figure(go.Pie(
            labels=fd["Format"], values=fd["Count"],
            hole=0.42,
            marker_colors=["#5C3D8F","#C8922A","#1A6B5A"],
        ))
        fig_fd.update_layout(
            height=320, margin=dict(t=20,b=20),
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig_fd, use_container_width=True)

    st.markdown("""<div class="insight"><strong>📌 Format insight:</strong>
    Physical-preferring customers → push print book kits and physical bundles.<br>
    Digital-first → e-book affiliate links and Kindle/Storytel partnerships.<br>
    Mixed-format (the biggest group) → ideal subscription box target.</div>""",
    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — REGRESSION
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Predicting maximum single spend (₹) — Regression")
    st.caption("Target: `max_single_spend`. Random Forest Regressor vs Ridge Regression.")

    with st.spinner("Training regression models..."):
        reg_res, reg_feats, reg_scaler, rf_reg, reg_coef = train_regressors(clean)

    reg_sel = st.selectbox("Model:", list(reg_res.keys()), key="reg_sel")
    rr      = reg_res[reg_sel]

    r1, r2 = st.columns(2)
    r1.markdown(f"""<div class="mcard">
    <div class="mval">{rr['r2']:.4f}</div>
    <div class="mlbl">R² Score</div></div>""", unsafe_allow_html=True)
    r2.markdown(f"""<div class="mcard">
    <div class="mval">₹{rr['rmse']:,.0f}</div>
    <div class="mlbl">RMSE</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    ra, rb = st.columns(2)

    with ra:
        st.markdown("#### Predicted vs Actual")
        maxv = float(max(rr["y_test"].max(), rr["y_pred"].max()))
        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(
            x=rr["y_test"], y=rr["y_pred"], mode="markers",
            marker=dict(color="#5C3D8F", size=5, opacity=0.48),
            name="Predictions",
        ))
        fig_pa.add_trace(go.Scatter(
            x=[0,maxv], y=[0,maxv], mode="lines",
            line=dict(dash="dash", color="#C8922A", width=2),
            name="Perfect fit",
        ))
        fig_pa.update_layout(
            height=380, margin=dict(t=20,b=20),
            xaxis=dict(title="Actual (₹)",    showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="Predicted (₹)", showgrid=True, gridcolor="#eee"),
            legend=dict(x=0.02,y=0.95),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_pa, use_container_width=True)

    with rb:
        st.markdown("#### Residual distribution")
        resid = rr["y_test"] - rr["y_pred"]
        fig_res = go.Figure(go.Histogram(
            x=resid, nbinsx=40,
            marker_color="#5C3D8F", opacity=0.78,
        ))
        fig_res.add_vline(x=0, line_dash="dash", line_color="#C8922A",
                           line_width=2, annotation_text="Zero residual")
        fig_res.update_layout(
            height=380, margin=dict(t=20,b=20),
            xaxis=dict(title="Residual (Actual − Predicted)",
                       showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="Count", showgrid=True, gridcolor="#eee"),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # ── RF FEATURE IMPORTANCE ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Feature importance — Random Forest Regressor")
    rfi_df = pd.DataFrame({
        "Feature":    reg_feats,
        "Importance": rf_reg.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_rfi = go.Figure(go.Bar(
        y=rfi_df["Feature"], x=rfi_df["Importance"],
        orientation="h",
        marker=dict(color=rfi_df["Importance"],
                    colorscale="Teal", showscale=False),
        text=[f"{v:.4f}" for v in rfi_df["Importance"]],
        textposition="outside",
    ))
    fig_rfi.update_layout(
        height=400, margin=dict(t=10,b=10,l=10,r=90),
        xaxis=dict(title="Importance", showgrid=True, gridcolor="#eee"),
        yaxis_title=None,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_rfi, use_container_width=True)

    # ── RIDGE COEFFICIENTS ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Ridge regression — coefficient plot")
    coef_df = pd.DataFrame({
        "Feature":     list(reg_coef.keys()),
        "Coefficient": list(reg_coef.values()),
    }).sort_values("Coefficient", key=abs, ascending=True)

    bar_colors = ["#1A6B5A" if v >= 0 else "#D45379"
                  for v in coef_df["Coefficient"]]
    fig_coef = go.Figure(go.Bar(
        y=coef_df["Feature"], x=coef_df["Coefficient"],
        orientation="h", marker_color=bar_colors,
        text=[f"{v:+.1f}" for v in coef_df["Coefficient"]],
        textposition="outside",
    ))
    fig_coef.add_vline(x=0, line_color="#ccc", line_width=1)
    fig_coef.update_layout(
        height=400, margin=dict(t=10,b=10,l=10,r=80),
        xaxis=dict(title="Coefficient (₹ impact)",
                   showgrid=True, gridcolor="#eee"),
        yaxis_title=None,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    # ── SPEND BY SEGMENT ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Predicted spend by DNA segment")
    rf_feats_p = [c for c in reg_feats if c in clean.columns]
    X_all = clean[rf_feats_p].fillna(0)
    pred_all = rf_reg.predict(X_all)
    tmp = clean.copy()
    tmp["predicted_spend"] = pred_all
    seg_spend = (tmp.groupby("dna_segment")["predicted_spend"]
                 .mean().sort_values(ascending=False).reset_index())
    fig_ss = go.Figure(go.Bar(
        x=seg_spend["dna_segment"], y=seg_spend["predicted_spend"],
        marker_color=[CLUSTER_COLORS.get(s,"#888") for s in seg_spend["dna_segment"]],
        text=["₹"+f"{v:,.0f}" for v in seg_spend["predicted_spend"]],
        textposition="outside",
    ))
    fig_ss.update_layout(
        height=310, xaxis_title=None,
        yaxis=dict(title="Avg predicted spend (₹)",
                   showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10,b=10),
    )
    st.plotly_chart(fig_ss, use_container_width=True)

    st.markdown("""<div class="insight"><strong>📌 Regression insight:</strong>
    Monthly income and lifestyle spend are the strongest predictors.<br>
    NPS proxy is a surprisingly powerful predictor — loyal advocates spend more.<br>
    Credit card users have significantly higher predicted spend and are ideal
    targets for premium product launches.</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Live prediction — enter a customer profile")
    st.caption("Fill in a respondent profile to get instant purchase intent probability and spend estimate.")

    with st.spinner("Loading trained models..."):
        clf_res2, clf_f2, _, rf2, _ = train_classifiers(clean)
        reg_res2, reg_f2, _, rf_r2, _ = train_regressors(clean)

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        inp_age    = st.select_slider("Age group", [2,3,4,5,6,7],
                                       format_func=lambda x:{2:"13-17",3:"18-22",
                                           4:"23-28",5:"29-35",6:"36-45",7:"46+"}[x], value=3)
        inp_city   = st.select_slider("City tier", [1,2,3,4,5],
                                       format_func=lambda x:{1:"Metro",2:"Tier 1",
                                           3:"Tier 2",4:"Tier 3",5:"Rural"}[x])
        inp_income = st.select_slider("Monthly income (₹)",
                                       [2500,10000,22500,45000,80000,120000],
                                       format_func=lambda x:f"₹{x:,}")
    with lc2:
        inp_stress = st.slider("Stress score",       0, 16, 8)
        inp_open   = st.slider("Openness (1–5)",     1,  5, 3)
        inp_nps    = st.slider("NPS proxy (0–10)",   0, 10, 6)
    with lc3:
        inp_books  = st.select_slider("Books/month", [0,1,2.5,5,8])
        inp_switch = st.slider("Switching tendency (1–4)", 1, 4, 2)
        inp_perso  = st.slider("Personalization importance (1–5)", 1, 5, 4)

    if st.button("🔮 Predict now", type="primary"):
        # Build sample dict aligned to CLF_FEATURES
        base = {f: 0 for f in clf_f2}
        overrides = {
            "age_group":               inp_age,
            "city_tier":               inp_city,
            "monthly_income_midpoint": inp_income,
            "stress_score":            inp_stress,
            "openness_score":          inp_open,
            "nps_proxy":               inp_nps,
            "books_per_month":         inp_books,
            "switching_tendency":      inp_switch,
            "personalization_importance": inp_perso,
            "conscientiousness_score": 3,
            "neuroticism_score":       max(1, inp_stress // 3),
            "agreeableness_score":     3,
            "lifestyle_spend":         inp_income * 0.08,
            "purchase_intent":         2,
            "platform_interest":       2,
            "barrier_price":           int(inp_income < 15000),
            "barrier_trust":           int(inp_nps < 5),
            "barrier_delivery":        int(inp_city >= 3),
            "subscription_interest":   2,
            "impulse_buying":          2,
            "discovery_social":        int(inp_open >= 4),
            "shops_instagram":         int(inp_open >= 4),
        }
        base.update(overrides)
        X_clf_s = pd.DataFrame([base])[clf_f2].fillna(0)
        prob = float(rf2.predict_proba(X_clf_s)[0][1])

        # Regression sample
        base_r = {f: 0 for f in reg_f2}
        base_r.update({k:v for k,v in overrides.items() if k in reg_f2})
        X_reg_s = pd.DataFrame([base_r])[reg_f2].fillna(0)
        pred_spend = float(rf_r2.predict(X_reg_s)[0])

        # Display
        pr1, pr2, pr3 = st.columns(3)
        buy_color = "#1A6B5A" if prob >= 0.5 else "#D45379"
        buy_label = "Will buy ✓" if prob >= 0.5 else "Won't buy ✗"
        pr1.markdown(f"""<div class="mcard" style="border-top:4px solid {buy_color}">
        <div class="mval" style="color:{buy_color}">{buy_label}</div>
        <div class="mlbl">Purchase intent</div></div>""", unsafe_allow_html=True)
        pr2.markdown(f"""<div class="mcard">
        <div class="mval">{prob*100:.1f}%</div>
        <div class="mlbl">Buy probability</div></div>""", unsafe_allow_html=True)
        pr3.markdown(f"""<div class="mcard">
        <div class="mval">₹{pred_spend:,.0f}</div>
        <div class="mlbl">Predicted max spend</div></div>""", unsafe_allow_html=True)

        # Estimated DNA segment
        if inp_stress >= 10:
            est = "Midnight Escapist"
        elif inp_income >= 45000 and inp_open <= 3:
            est = "Productivity Achiever"
        elif inp_open >= 4:
            est = "Curious Explorer"
        else:
            est = "Emotional Reader"
        ec = CLUSTER_COLORS.get(est, "#888")
        st.markdown(f"""
        <div style="margin-top:12px;border-left:4px solid {ec};background:#faf8f4;
             border-radius:0 8px 8px 0;padding:12px 16px;">
            <strong>Estimated DNA segment:</strong>
            <span style="color:{ec};font-weight:600;margin-left:8px">{est}</span>
        </div>""", unsafe_allow_html=True)
