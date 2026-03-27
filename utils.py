"""
utils.py  —  Book DNA Analytics Dashboard
All shared helpers: data loading, preprocessing, label maps, cached model training.
Every import is explicit — no missing packages.
"""

# ── STANDARD LIBRARY ──────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── THIRD-PARTY ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_squared_error, r2_score,
)

# ── COLOUR PALETTE ────────────────────────────────────────────────────
CLUSTER_COLORS = {
    "Midnight Escapist":     "#5C3D8F",
    "Productivity Achiever": "#C8922A",
    "Emotional Reader":      "#D45379",
    "Curious Explorer":      "#1A6B5A",
    "Non-Reader":            "#888780",
}
PALETTE = list(CLUSTER_COLORS.values())

# ── LABEL MAPS ────────────────────────────────────────────────────────
AGE_LABELS      = {2:"13-17",3:"18-22",4:"23-28",5:"29-35",6:"36-45",7:"46+"}
GENDER_LABELS   = {1:"Male",2:"Female",3:"Non-binary",4:"Prefer not say"}
CITY_LABELS     = {1:"Metro",2:"Tier 1",3:"Tier 2",4:"Tier 3",5:"Rural"}
OCC_LABELS      = {1:"School student",2:"College student",3:"Working professional",
                   4:"Freelancer",5:"Homemaker",6:"Other"}
FORMAT_LABELS   = {1:"Physical only",2:"Mostly physical",3:"Both equal",
                   4:"Mostly digital",5:"Digital only"}
FORMAT_CLASS    = {1:"Physical",2:"Mixed",3:"Digital"}
MOOD_LABELS     = {1:"Stressed",2:"Motivated",3:"Reflective",
                   4:"Curious",5:"Calm",6:"Numb"}
LS_LABELS       = {1:"Exam pressure",2:"Finding myself",3:"Building career",
                   4:"Relationship",5:"Parenting",6:"Feeling stuck",7:"Content"}
RH_LABELS       = {1:"Active reader",2:"Want to return",3:"Building habit",
                   4:"Occasional",5:"Not my thing"}
RI_LABELS       = {1:"To escape",2:"Self-improvement",3:"Feel understood",
                   4:"Curiosity",5:"Habit",6:"Don't read"}
TRUST_LABELS    = {1:"Influencer",2:"Reviews",3:"Free trial",
                   4:"Media coverage",5:"Friend rec.",6:"Money-back"}
DISCOUNT_LABELS = {1:"Flat % off",2:"Bundle deal",3:"Free shipping",
                   4:"Loyalty pts",5:"Festival sale",6:"First-buyer offer"}
PAY_LABELS      = {1:"UPI",2:"Debit card",3:"Credit card",4:"COD",5:"BNPL"}

PRODUCT_COLS  = ["interest_books","interest_journal","interest_bookmark",
                 "interest_candle","interest_tote","interest_apparel",
                 "interest_decor","interest_annotation","interest_pin"]
PRODUCT_NAMES = ["Book bundles","Journals","Bookmarks","Candles",
                 "Tote bags","Apparel","Shelf décor","Annotation kits","Enamel pins"]

# ── FEATURE COLUMN LISTS ─────────────────────────────────────────────
CLUSTER_FEATURES = [
    "openness_score","conscientiousness_score","extraversion_score",
    "agreeableness_score","neuroticism_score","stress_score",
    "books_per_month","social_sharing_level","lifestyle_spend",
    "genre_fantasy","genre_selfhelp","genre_literary","genre_romance",
    "genre_thriller","genre_biography","genre_business",
    "reading_motivation","reader_identity","personalization_importance",
    "nps_proxy","monthly_income_midpoint",
]

CLF_FEATURES = [
    "age_group","city_tier","monthly_income_midpoint","stress_score",
    "openness_score","conscientiousness_score","neuroticism_score",
    "agreeableness_score","books_per_month","social_sharing_level",
    "nps_proxy","switching_tendency","personalization_importance",
    "barrier_price","barrier_trust","barrier_delivery",
    "subscription_interest","lifestyle_spend","impulse_buying",
    "discovery_social","shops_instagram","platform_interest",
]

REG_FEATURES = [
    "monthly_income_midpoint","lifestyle_spend","stress_score","nps_proxy",
    "conscientiousness_score","openness_score","city_tier","payment_method",
    "purchase_intent","books_per_month","personalization_importance",
    "subscription_interest","impulse_buying","barrier_price",
]

ARM_COLS = (PRODUCT_COLS +
            ["barrier_price","barrier_quality","barrier_trust",
             "barrier_delivery","barrier_privacy","barrier_platform",
             "shops_instagram","shops_amazon","shops_offline"])

# ── DATA LOADING ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv("book_dna_data.csv")
    df = df.copy()
    if "psm_bargain" in df.columns:
        df["psm_bargain"] = df["psm_bargain"].fillna(df["psm_bargain"].median())
    return df

@st.cache_data(show_spinner=False)
def get_clean_df(_df):
    if "data_quality_flag" in _df.columns:
        clean = _df[_df["data_quality_flag"] == "clean"].copy()
    else:
        clean = _df.copy()
    if "psm_bargain" in clean.columns:
        clean["psm_bargain"] = clean["psm_bargain"].fillna(clean["psm_bargain"].median())
    return clean

# ── CLUSTERING ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_kmeans(_df, k=5):
    feats = [c for c in CLUSTER_FEATURES if c in _df.columns]
    X = _df[feats].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    return km, scaler, labels, feats

@st.cache_data(show_spinner=False)
def elbow_silhouette(_df, k_max=10):
    feats = [c for c in CLUSTER_FEATURES if c in _df.columns]
    X = _df[feats].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    inertias, silhouettes = [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, lbl))
    return list(range(2, k_max + 1)), inertias, silhouettes

@st.cache_data(show_spinner=False)
def compute_pca(_df, _labels):
    feats = [c for c in CLUSTER_FEATURES if c in _df.columns]
    X = _df[feats].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    out = pd.DataFrame(coords, columns=["PC1","PC2"])
    out["cluster"] = _labels
    if "dna_segment" in _df.columns:
        out["segment"] = _df["dna_segment"].values
    else:
        out["segment"] = _labels.astype(str)
    return out, pca.explained_variance_ratio_

def get_cluster_segment_map(clean_df, labels):
    """Map KMeans cluster index → dominant dna_segment label."""
    tmp = clean_df.copy()
    tmp["_km"] = labels
    mapping = {}
    for c in np.unique(labels):
        sub = tmp[tmp["_km"] == c]
        if "dna_segment" in sub.columns and len(sub) > 0:
            mapping[c] = sub["dna_segment"].value_counts().idxmax()
        else:
            mapping[c] = f"Cluster {c}"
    return mapping

# ── CLASSIFICATION ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_classifiers(_df):
    feats = [c for c in CLF_FEATURES if c in _df.columns]
    X = _df[feats].fillna(0)
    y = _df["will_buy"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "Random Forest":      (RandomForestClassifier(
                                   n_estimators=150, random_state=42,
                                   class_weight="balanced"), X_tr, X_te),
        "Logistic Regression":(LogisticRegression(
                                   max_iter=500, random_state=42,
                                   class_weight="balanced"), X_tr_s, X_te_s),
        "Decision Tree":      (DecisionTreeClassifier(
                                   max_depth=5, random_state=42,
                                   class_weight="balanced"), X_tr, X_te),
    }

    results = {}
    rf_model = dt_model = None
    for name, (model, Xtr_, Xte_) in models.items():
        model.fit(Xtr_, y_tr)
        yp    = model.predict(Xte_)
        yprob = model.predict_proba(Xte_)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, yprob)
        results[name] = {
            "model":     model,
            "y_test":    y_te,
            "y_pred":    yp,
            "y_prob":    yprob,
            "accuracy":  round(accuracy_score(y_te, yp), 4),
            "precision": round(precision_score(y_te, yp, zero_division=0), 4),
            "recall":    round(recall_score(y_te, yp, zero_division=0), 4),
            "f1":        round(f1_score(y_te, yp, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_te, yprob), 4),
            "fpr": fpr, "tpr": tpr,
            "cm": confusion_matrix(y_te, yp),
        }
        if name == "Random Forest":
            rf_model = model
        if name == "Decision Tree":
            dt_model = model

    return results, feats, scaler, rf_model, dt_model

@st.cache_resource(show_spinner=False)
def train_format_classifier(_df):
    feats = [c for c in CLF_FEATURES if c in _df.columns]
    X = _df[feats].fillna(0)
    y = _df["format_class"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    classes = [FORMAT_CLASS.get(c, str(c)) for c in sorted(y.unique())]
    return model, feats, {
        "accuracy":  round(accuracy_score(y_te, yp), 4),
        "precision": round(precision_score(y_te, yp, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_te, yp, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_te, yp, average="weighted", zero_division=0), 4),
        "cm": confusion_matrix(y_te, yp),
        "classes": classes,
    }

# ── REGRESSION ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_regressors(_df):
    feats = [c for c in REG_FEATURES if c in _df.columns]
    X = _df[feats].fillna(0)
    y = _df["max_single_spend"].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf    = RandomForestRegressor(n_estimators=150, random_state=42)
    ridge = Ridge(alpha=1.0)
    rf.fit(X_tr, y_tr)
    ridge.fit(X_tr_s, y_tr)

    results = {}
    for name, model, Xte_ in [("Random Forest", rf, X_te),
                                ("Ridge Regression", ridge, X_te_s)]:
        yp   = model.predict(Xte_)
        rmse = float(np.sqrt(mean_squared_error(y_te, yp)))
        results[name] = {
            "model":  model,
            "y_test": y_te.values,
            "y_pred": yp,
            "r2":     round(r2_score(y_te, yp), 4),
            "rmse":   round(rmse, 2),
        }

    coef = dict(zip(feats, ridge.coef_))
    return results, feats, scaler, rf, coef

# ── ASSOCIATION RULE MINING ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_arm(_df, min_support=0.08, min_confidence=0.40, min_lift=1.0):
    from mlxtend.frequent_patterns import apriori, association_rules

    cols = [c for c in ARM_COLS if c in _df.columns]
    basket = _df[cols].copy().astype(bool)

    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence",
                               min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()

    rules = rules[rules["lift"] >= min_lift].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    out = rules[["antecedents","consequents","support","confidence","lift"]].copy()
    for col in ["support","confidence","lift"]:
        out[col] = out[col].round(4)
    return out.sort_values("lift", ascending=False).reset_index(drop=True)

# ── VAN WESTENDORP CHART ─────────────────────────────────────────────
def psm_chart(df, segment=None):
    import plotly.graph_objects as go
    sub = df if segment is None else df[df["dna_segment"] == segment]
    sub = sub.dropna(subset=["psm_too_cheap","psm_bargain",
                              "psm_expensive_ok","psm_too_expensive"])
    if len(sub) < 20:
        return None, None, None, None

    prices = np.linspace(50, 3000, 300)

    def above(col): return np.array([np.mean(sub[col].values >= p) for p in prices]) * 100
    def below(col): return np.array([np.mean(sub[col].values <= p) for p in prices]) * 100

    tc   = above("psm_too_cheap")
    barg = below("psm_bargain")
    exok = above("psm_expensive_ok")
    toex = above("psm_too_expensive")

    fig = go.Figure()
    for name, y, color in [
        ("Too cheap (quality doubt)",  tc,   "#E24B4A"),
        ("Bargain / great value",      barg, "#1D9E75"),
        ("Expensive but acceptable",   exok, "#EF9F27"),
        ("Too expensive",              toex, "#5C3D8F"),
    ]:
        fig.add_trace(go.Scatter(
            x=prices, y=y, name=name, mode="lines",
            line=dict(color=color, width=2.5)
        ))

    pmc = int(prices[np.argmin(np.abs(tc  - barg))])
    pme = int(prices[np.argmin(np.abs(exok - toex))])
    opp = int(prices[np.argmin(tc + toex)])

    for xv, lbl, col in [(pmc,"PMC","#1D9E75"),(opp,"OPP","#000000"),(pme,"PME","#5C3D8F")]:
        fig.add_vline(x=xv, line_dash="dash", line_color=col, line_width=1.8,
                      annotation_text=f"{lbl}: ₹{xv}",
                      annotation_position="top right")

    title = f"Van Westendorp PSM — {'All respondents' if segment is None else segment}"
    fig.update_layout(
        title=title,
        xaxis_title="Monthly price (₹)",
        yaxis_title="Cumulative %",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=30),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", range=[0, 105])
    return fig, pmc, opp, pme

# ── PRESCRIPTIVE PLAYBOOK ────────────────────────────────────────────
PRESCRIPTIVE = {
    "Midnight Escapist": {
        "priority":   "🔴 Primary target",
        "offer":      "Free shipping + 10% first-order discount",
        "bundle":     "Scented candle + dark journal + 1 fantasy book",
        "channel":    "Instagram Reels · Late-night BookTok",
        "timing":     "Post 9 PM — this segment reads late",
        "churn_risk": "Medium — high stress → impulse buyer but easily disappointed",
        "ltv_band":   "₹6,000 – ₹12,000 / year",
    },
    "Productivity Achiever": {
        "priority":   "🟠 High priority",
        "offer":      "Annual plan at 25% saving — locks in low-churn segment",
        "bundle":     "Self-help book + Annotation kit + Notion template",
        "channel":    "LinkedIn · YouTube productivity · Google Search ads",
        "timing":     "Morning 6–9 AM · Commute window",
        "churn_risk": "Low — high conscientiousness = plan adherence",
        "ltv_band":   "₹10,000 – ₹20,000 / year",
    },
    "Emotional Reader": {
        "priority":   "🟡 Secondary",
        "offer":      "Buy 2 get 1 free bundle — triggers gifting instinct",
        "bundle":     "Literary fiction + Romance novel + Illustrated bookmarks",
        "channel":    "Instagram Stories · WhatsApp gifting posts",
        "timing":     "Evening 7–10 PM · Weekend afternoons",
        "churn_risk": "Low-medium — high agreeableness = brand loyalty once bonded",
        "ltv_band":   "₹6,000 – ₹10,000 / year",
    },
    "Curious Explorer": {
        "priority":   "🟢 Growth engine",
        "offer":      "Free DNA quiz card (zero cost) → first-purchase 15% off",
        "bundle":     "Surprise 3-genre mystery box + Enamel pin collectible",
        "channel":    "Instagram · Meme pages · Discord book servers",
        "timing":     "Any time — impulse buyers, channel-agnostic",
        "churn_risk": "High — always seeking the next thing",
        "ltv_band":   "₹4,000 – ₹8,000 / year (high referral value K>1.3)",
    },
    "Non-Reader": {
        "priority":   "⚪ Deprioritize",
        "offer":      "Free DNA quiz only — no paid product push in Phase 1",
        "bundle":     "N/A — re-evaluate after 90 days of quiz data",
        "channel":    "Organic only — not worth paid spend",
        "timing":     "N/A",
        "churn_risk": "Very high",
        "ltv_band":   "< ₹1,000 / year",
    },
}
