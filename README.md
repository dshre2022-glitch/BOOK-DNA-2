# 📚 Book DNA — Customer Analytics Dashboard

A multi-page **Streamlit** analytics dashboard for data-driven decision making around the Book DNA personality-driven reading lifestyle brand.

## Features

| Page | Type | What it does |
|---|---|---|
| 🏠 Home | Overview | KPI cards · segment bar · quick insights |
| 📊 Descriptive | Descriptive | Demographics · Van Westendorp PSM · product heatmap · reading habits |
| 🔵 Clustering | Diagnostic | K-Means · Elbow chart · Silhouette scores · PCA scatter · OCEAN radar · persona cards |
| 🔗 ARM | Diagnostic | Apriori · Support / Confidence / Lift scatter · confidence–lift scatter · sortable table · co-interest heatmap |
| 🔮 Predictive | Predictive | Classification (Accuracy, Precision, Recall, F1, ROC-AUC, CM, Feature Importance, DT rules) · Regression (R², RMSE, residuals, coefficients) · Live predict widget |
| 🎯 Prescriptive | Prescriptive | Discount engine · Bundle recommender · Churn flags · Focus customer definition · CSV upload & predict |

## Algorithms

- **K-Means Clustering** — Elbow Chart + Silhouette Score validation, PCA 2-D visualisation, OCEAN radar per cluster
- **Classification** — Random Forest, Logistic Regression, Decision Tree with Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Feature Importance, Decision Tree text rules
- **Regression** — Random Forest Regressor + Ridge Regression with R², RMSE, predicted-vs-actual scatter, residual histogram, coefficient plot
- **Association Rule Mining** — Apriori with Support, Confidence, Lift — scatter, ranked bar, confidence-vs-lift, co-interest heatmap
- **Van Westendorp PSM** — PMC, OPP, PME markers, per-segment view

## Quick start (local)

```bash
git clone https://github.com/yourusername/book-dna-dashboard.git
cd book-dna-dashboard
pip install -r requirements.txt
python generate_data.py        # run once — creates book_dna_data.csv
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push **all files** to a GitHub repo (flat root — no sub-folder except `pages/`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → set **Main file path**: `app.py`
4. Click **Deploy** — dependencies install automatically from `requirements.txt`

> **Note:** `book_dna_data.csv` must be committed to the repo.  
> Run `python generate_data.py` locally and `git add book_dna_data.csv` before pushing.

## Repo structure (flat — upload all to GitHub root)

```
app.py                        ← Streamlit entry point
generate_data.py              ← Run once to create the CSV
utils.py                      ← Shared helpers & model training
requirements.txt              ← Pinned dependencies
book_dna_data.csv             ← Generated dataset (2,000 rows)
README.md
pages/
  1_Descriptive.py
  2_Clustering.py
  3_ARM.py
  4_Predictive.py
  5_Prescriptive_Upload.py
```

## New data upload

Navigate to **Prescriptive & Upload → Upload & Predict**.  
Upload a CSV with the same columns as the template (download button provided).  
The pipeline automatically:
- Assigns a predicted DNA segment (K-Means)
- Predicts purchase-intent probability (Random Forest Classifier)
- Estimates maximum single spend (Random Forest Regressor)
- Recommends a personalised offer and bundle per respondent
- Flags priority leads (probability ≥ 65%)

Download the enriched CSV and use it directly in your marketing CRM.

---
Built with [Streamlit](https://streamlit.io) · [scikit-learn](https://scikit-learn.org) · [mlxtend](http://rasbt.github.io/mlxtend/) · [Plotly](https://plotly.com)
