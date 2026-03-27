"""
Book DNA — Synthetic Dataset Generator
Run once before launching the app: python generate_data.py
Produces: book_dna_data.csv (2000 rows x 93 columns)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

ARCHETYPES = {
    "Midnight Escapist":     0.30,
    "Productivity Achiever": 0.22,
    "Emotional Reader":      0.18,
    "Curious Explorer":      0.18,
    "Non-Reader":            0.12,
}
segments = np.random.choice(list(ARCHETYPES.keys()), size=N, p=list(ARCHETYPES.values()))
rows = []

for i, seg in enumerate(segments):
    r = {"respondent_id": f"BDNA{i+1:04d}", "dna_segment": seg}

    # ── DEMOGRAPHICS ────────────────────────────────────────────────────
    age_p = {"Midnight Escapist":[0.18,0.38,0.22,0.12,0.06,0.04],
              "Productivity Achiever":[0.08,0.22,0.30,0.24,0.11,0.05],
              "Emotional Reader":[0.10,0.28,0.26,0.20,0.11,0.05],
              "Curious Explorer":[0.20,0.42,0.22,0.10,0.04,0.02],
              "Non-Reader":[0.10,0.20,0.24,0.22,0.14,0.10]}
    r["age_group"] = int(np.random.choice([2,3,4,5,6,7], p=age_p[seg]))

    city_p = {"Midnight Escapist":[0.22,0.25,0.28,0.18,0.07],
               "Productivity Achiever":[0.38,0.30,0.20,0.09,0.03],
               "Emotional Reader":[0.28,0.28,0.24,0.14,0.06],
               "Curious Explorer":[0.30,0.26,0.24,0.15,0.05],
               "Non-Reader":[0.20,0.22,0.28,0.20,0.10]}
    r["city_tier"] = int(np.random.choice([1,2,3,4,5], p=city_p[seg]))

    r["gender"] = int(np.random.choice([1,2,3,4], p=[0.43,0.52,0.03,0.02]))

    occ_p = {"Midnight Escapist":[0.20,0.42,0.20,0.08,0.05,0.05],
              "Productivity Achiever":[0.05,0.25,0.48,0.14,0.04,0.04],
              "Emotional Reader":[0.12,0.38,0.28,0.08,0.10,0.04],
              "Curious Explorer":[0.18,0.48,0.18,0.08,0.04,0.04],
              "Non-Reader":[0.15,0.28,0.32,0.10,0.10,0.05]}
    r["occupation"] = int(np.random.choice([1,2,3,4,5,6], p=occ_p[seg]))

    inc_opts = [2500,10000,22500,45000,80000,120000]
    inc_p = {"Midnight Escapist":[0.28,0.32,0.22,0.11,0.05,0.02],
              "Productivity Achiever":[0.05,0.15,0.28,0.30,0.16,0.06],
              "Emotional Reader":[0.12,0.28,0.30,0.18,0.09,0.03],
              "Curious Explorer":[0.20,0.38,0.24,0.12,0.05,0.01],
              "Non-Reader":[0.20,0.30,0.26,0.14,0.07,0.03]}
    r["monthly_income_midpoint"] = int(np.random.choice(inc_opts, p=inc_p[seg]))

    # ── OCEAN PERSONALITY ───────────────────────────────────────────────
    ocean_mu = {"Midnight Escapist":[2.8,2.6,2.5,3.8,4.2],
                "Productivity Achiever":[3.4,4.5,3.2,3.0,2.4],
                "Emotional Reader":[3.5,3.2,2.8,4.6,3.2],
                "Curious Explorer":[4.6,2.9,4.0,3.4,2.8],
                "Non-Reader":[2.5,2.8,3.0,3.0,2.5]}
    for idx, trait in enumerate(["openness","conscientiousness","extraversion",
                                   "agreeableness","neuroticism"]):
        r[f"{trait}_score"] = int(np.clip(round(np.random.normal(ocean_mu[seg][idx], 0.7)), 1, 5))

    # ── PSS-4 STRESS ────────────────────────────────────────────────────
    stress_mu = {"Midnight Escapist":10.5,"Productivity Achiever":6.0,
                 "Emotional Reader":7.8,"Curious Explorer":5.5,"Non-Reader":5.0}
    r["stress_score"] = int(np.clip(round(np.random.normal(stress_mu[seg], 2.2)), 0, 16))
    for k in ["pss_control","pss_overwhelm","pss_confidence_r","pss_irritation_r"]:
        r[k] = int(np.clip(round(r["stress_score"]/4 + np.random.normal(0,0.5)), 0, 4))

    # ── LIFE STAGE & MOOD ───────────────────────────────────────────────
    ls_p = {"Midnight Escapist":[0.35,0.20,0.15,0.10,0.05,0.10,0.05],
             "Productivity Achiever":[0.10,0.12,0.42,0.08,0.08,0.05,0.15],
             "Emotional Reader":[0.08,0.20,0.18,0.22,0.10,0.08,0.14],
             "Curious Explorer":[0.22,0.38,0.18,0.08,0.04,0.04,0.06],
             "Non-Reader":[0.12,0.15,0.25,0.10,0.12,0.08,0.18]}
    r["life_stage"] = int(np.random.choice([1,2,3,4,5,6,7], p=ls_p[seg]))

    mood_p = {"Midnight Escapist":[0.40,0.12,0.22,0.10,0.08,0.08],
               "Productivity Achiever":[0.08,0.48,0.10,0.16,0.12,0.06],
               "Emotional Reader":[0.15,0.10,0.35,0.16,0.14,0.10],
               "Curious Explorer":[0.08,0.18,0.12,0.40,0.14,0.08],
               "Non-Reader":[0.18,0.12,0.16,0.14,0.22,0.18]}
    r["current_mood"] = int(np.random.choice([1,2,3,4,5,6], p=mood_p[seg]))

    ri_p = {"Midnight Escapist":[0.50,0.10,0.12,0.14,0.08,0.06],
             "Productivity Achiever":[0.06,0.50,0.06,0.14,0.16,0.08],
             "Emotional Reader":[0.12,0.08,0.48,0.10,0.12,0.10],
             "Curious Explorer":[0.08,0.10,0.10,0.48,0.14,0.10],
             "Non-Reader":[0.06,0.06,0.08,0.12,0.12,0.56]}
    r["reader_identity"] = int(np.random.choice([1,2,3,4,5,6], p=ri_p[seg]))

    pi_mu = {"Midnight Escapist":4.2,"Productivity Achiever":4.0,
              "Emotional Reader":3.8,"Curious Explorer":4.4,"Non-Reader":2.5}
    r["personalization_importance"] = int(np.clip(round(np.random.normal(pi_mu[seg], 0.8)), 1, 5))

    # ── READING BEHAVIOUR ───────────────────────────────────────────────
    bpm_p = {"Midnight Escapist":[0.05,0.20,0.38,0.25,0.12],
              "Productivity Achiever":[0.04,0.18,0.35,0.28,0.15],
              "Emotional Reader":[0.06,0.22,0.40,0.22,0.10],
              "Curious Explorer":[0.08,0.25,0.38,0.20,0.09],
              "Non-Reader":[0.50,0.28,0.15,0.05,0.02]}
    r["books_per_month"] = float(np.random.choice([0,1,2.5,5,8], p=bpm_p[seg]))

    time_probs = {
        "reads_morning":  {"Midnight Escapist":0.15,"Productivity Achiever":0.55,
                           "Emotional Reader":0.30,"Curious Explorer":0.35,"Non-Reader":0.15},
        "reads_commute":  {"Midnight Escapist":0.30,"Productivity Achiever":0.40,
                           "Emotional Reader":0.28,"Curious Explorer":0.32,"Non-Reader":0.12},
        "reads_afternoon":{"Midnight Escapist":0.20,"Productivity Achiever":0.25,
                           "Emotional Reader":0.25,"Curious Explorer":0.28,"Non-Reader":0.10},
        "reads_evening":  {"Midnight Escapist":0.45,"Productivity Achiever":0.38,
                           "Emotional Reader":0.45,"Curious Explorer":0.42,"Non-Reader":0.18},
        "reads_latenight":{"Midnight Escapist":0.72,"Productivity Achiever":0.22,
                           "Emotional Reader":0.38,"Curious Explorer":0.35,"Non-Reader":0.08},
        "reads_weekend":  {"Midnight Escapist":0.40,"Productivity Achiever":0.30,
                           "Emotional Reader":0.38,"Curious Explorer":0.35,"Non-Reader":0.20},
    }
    for col, probs in time_probs.items():
        r[col] = int(np.random.random() < probs[seg])

    rm_p = {"Midnight Escapist":[0.55,0.08,0.18,0.06,0.08,0.05],
             "Productivity Achiever":[0.06,0.55,0.10,0.18,0.06,0.05],
             "Emotional Reader":[0.18,0.10,0.38,0.08,0.18,0.08],
             "Curious Explorer":[0.10,0.22,0.30,0.10,0.18,0.10],
             "Non-Reader":[0.10,0.08,0.14,0.16,0.08,0.44]}
    r["reading_motivation"] = int(np.random.choice([1,2,3,4,5,6], p=rm_p[seg]))

    # Genres
    genre_p = {
        "genre_fantasy":  {"Midnight Escapist":0.75,"Productivity Achiever":0.18,
                           "Emotional Reader":0.28,"Curious Explorer":0.50,"Non-Reader":0.10},
        "genre_selfhelp": {"Midnight Escapist":0.22,"Productivity Achiever":0.80,
                           "Emotional Reader":0.28,"Curious Explorer":0.38,"Non-Reader":0.12},
        "genre_literary": {"Midnight Escapist":0.30,"Productivity Achiever":0.20,
                           "Emotional Reader":0.70,"Curious Explorer":0.40,"Non-Reader":0.08},
        "genre_romance":  {"Midnight Escapist":0.35,"Productivity Achiever":0.12,
                           "Emotional Reader":0.62,"Curious Explorer":0.22,"Non-Reader":0.08},
        "genre_thriller": {"Midnight Escapist":0.48,"Productivity Achiever":0.30,
                           "Emotional Reader":0.32,"Curious Explorer":0.38,"Non-Reader":0.10},
        "genre_biography":{"Midnight Escapist":0.15,"Productivity Achiever":0.55,
                           "Emotional Reader":0.28,"Curious Explorer":0.30,"Non-Reader":0.08},
        "genre_business": {"Midnight Escapist":0.08,"Productivity Achiever":0.68,
                           "Emotional Reader":0.10,"Curious Explorer":0.22,"Non-Reader":0.06},
        "genre_regional": {"Midnight Escapist":0.22,"Productivity Achiever":0.20,
                           "Emotional Reader":0.30,"Curious Explorer":0.25,"Non-Reader":0.14},
        "genre_comics":   {"Midnight Escapist":0.28,"Productivity Achiever":0.12,
                           "Emotional Reader":0.15,"Curious Explorer":0.48,"Non-Reader":0.12},
        "genre_poetry":   {"Midnight Escapist":0.20,"Productivity Achiever":0.10,
                           "Emotional Reader":0.42,"Curious Explorer":0.28,"Non-Reader":0.05},
    }
    for g, probs in genre_p.items():
        r[g] = int(np.random.random() < probs[seg])

    # Discovery channels
    disc_p = {
        "discovery_social":     {"Midnight Escapist":0.60,"Productivity Achiever":0.35,
                                  "Emotional Reader":0.48,"Curious Explorer":0.78,"Non-Reader":0.25},
        "discovery_friends":    {"Midnight Escapist":0.45,"Productivity Achiever":0.38,
                                  "Emotional Reader":0.52,"Curious Explorer":0.42,"Non-Reader":0.28},
        "discovery_bestsellers":{"Midnight Escapist":0.22,"Productivity Achiever":0.45,
                                  "Emotional Reader":0.28,"Curious Explorer":0.30,"Non-Reader":0.18},
        "discovery_apps":       {"Midnight Escapist":0.30,"Productivity Achiever":0.38,
                                  "Emotional Reader":0.28,"Curious Explorer":0.42,"Non-Reader":0.12},
        "discovery_bookstore":  {"Midnight Escapist":0.25,"Productivity Achiever":0.22,
                                  "Emotional Reader":0.32,"Curious Explorer":0.28,"Non-Reader":0.12},
        "discovery_youtube":    {"Midnight Escapist":0.35,"Productivity Achiever":0.28,
                                  "Emotional Reader":0.25,"Curious Explorer":0.45,"Non-Reader":0.15},
        "discovery_syllabus":   {"Midnight Escapist":0.18,"Productivity Achiever":0.20,
                                  "Emotional Reader":0.15,"Curious Explorer":0.22,"Non-Reader":0.18},
    }
    for d, probs in disc_p.items():
        r[d] = int(np.random.random() < probs[seg])

    ssl_mu = {"Midnight Escapist":2.8,"Productivity Achiever":2.5,
               "Emotional Reader":2.6,"Curious Explorer":1.8,"Non-Reader":3.5}
    r["social_sharing_level"] = int(np.clip(round(np.random.normal(ssl_mu[seg], 0.8)), 1, 4))

    fp_p = {"Midnight Escapist":[0.48,0.28,0.14,0.07,0.03],
             "Productivity Achiever":[0.22,0.26,0.22,0.20,0.10],
             "Emotional Reader":[0.40,0.30,0.18,0.08,0.04],
             "Curious Explorer":[0.28,0.24,0.22,0.16,0.10],
             "Non-Reader":[0.20,0.22,0.28,0.20,0.10]}
    r["format_preference"] = int(np.random.choice([1,2,3,4,5], p=fp_p[seg]))

    # ── PRODUCTS ────────────────────────────────────────────────────────
    prod_p = {
        "interest_books":     {"Midnight Escapist":0.78,"Productivity Achiever":0.70,
                               "Emotional Reader":0.72,"Curious Explorer":0.68,"Non-Reader":0.20},
        "interest_journal":   {"Midnight Escapist":0.65,"Productivity Achiever":0.55,
                               "Emotional Reader":0.60,"Curious Explorer":0.50,"Non-Reader":0.15},
        "interest_bookmark":  {"Midnight Escapist":0.58,"Productivity Achiever":0.42,
                               "Emotional Reader":0.52,"Curious Explorer":0.55,"Non-Reader":0.12},
        "interest_candle":    {"Midnight Escapist":0.72,"Productivity Achiever":0.28,
                               "Emotional Reader":0.60,"Curious Explorer":0.38,"Non-Reader":0.10},
        "interest_tote":      {"Midnight Escapist":0.50,"Productivity Achiever":0.38,
                               "Emotional Reader":0.45,"Curious Explorer":0.60,"Non-Reader":0.12},
        "interest_apparel":   {"Midnight Escapist":0.42,"Productivity Achiever":0.30,
                               "Emotional Reader":0.35,"Curious Explorer":0.65,"Non-Reader":0.08},
        "interest_decor":     {"Midnight Escapist":0.60,"Productivity Achiever":0.25,
                               "Emotional Reader":0.48,"Curious Explorer":0.40,"Non-Reader":0.08},
        "interest_annotation":{"Midnight Escapist":0.28,"Productivity Achiever":0.65,
                               "Emotional Reader":0.32,"Curious Explorer":0.38,"Non-Reader":0.05},
        "interest_pin":       {"Midnight Escapist":0.35,"Productivity Achiever":0.20,
                               "Emotional Reader":0.28,"Curious Explorer":0.55,"Non-Reader":0.06},
    }
    for p, probs in prod_p.items():
        r[p] = int(np.random.random() < probs[seg])

    sub_mu = {"Midnight Escapist":1.8,"Productivity Achiever":2.2,
               "Emotional Reader":2.0,"Curious Explorer":2.0,"Non-Reader":3.5}
    r["subscription_interest"] = int(np.clip(round(np.random.normal(sub_mu[seg], 0.8)), 1, 4))
    r["eco_importance"]  = int(np.clip(round(np.random.normal(2.5, 0.9)), 1, 4))
    gift_mu = {"Midnight Escapist":2.2,"Productivity Achiever":2.0,
                "Emotional Reader":1.8,"Curious Explorer":2.2,"Non-Reader":3.0}
    r["gifting_behaviour"] = int(np.clip(round(np.random.normal(gift_mu[seg], 0.8)), 1, 4))

    # Barriers
    bar_p = {
        "barrier_price":   {"Midnight Escapist":0.45,"Productivity Achiever":0.22,
                            "Emotional Reader":0.35,"Curious Explorer":0.38,"Non-Reader":0.40},
        "barrier_quality": {"Midnight Escapist":0.32,"Productivity Achiever":0.25,
                            "Emotional Reader":0.28,"Curious Explorer":0.22,"Non-Reader":0.30},
        "barrier_trust":   {"Midnight Escapist":0.28,"Productivity Achiever":0.18,
                            "Emotional Reader":0.24,"Curious Explorer":0.20,"Non-Reader":0.35},
        "barrier_delivery":{"Midnight Escapist":0.22,"Productivity Achiever":0.12,
                            "Emotional Reader":0.18,"Curious Explorer":0.15,"Non-Reader":0.28},
        "barrier_privacy": {"Midnight Escapist":0.15,"Productivity Achiever":0.20,
                            "Emotional Reader":0.12,"Curious Explorer":0.18,"Non-Reader":0.15},
        "barrier_platform":{"Midnight Escapist":0.20,"Productivity Achiever":0.25,
                            "Emotional Reader":0.18,"Curious Explorer":0.15,"Non-Reader":0.30},
    }
    for b, probs in bar_p.items():
        r[b] = int(np.random.random() < probs[seg])

    td_p = {"Midnight Escapist":[0.30,0.28,0.18,0.08,0.12,0.04],
             "Productivity Achiever":[0.15,0.25,0.20,0.18,0.12,0.10],
             "Emotional Reader":[0.20,0.22,0.18,0.10,0.22,0.08],
             "Curious Explorer":[0.38,0.30,0.12,0.06,0.10,0.04],
             "Non-Reader":[0.14,0.18,0.25,0.12,0.18,0.13]}
    r["trust_driver"] = int(np.random.choice([1,2,3,4,5,6], p=td_p[seg]))

    dp_p = {"Midnight Escapist":[0.22,0.35,0.20,0.12,0.08,0.03],
             "Productivity Achiever":[0.18,0.20,0.15,0.28,0.12,0.07],
             "Emotional Reader":[0.20,0.28,0.18,0.16,0.12,0.06],
             "Curious Explorer":[0.25,0.28,0.18,0.10,0.12,0.07],
             "Non-Reader":[0.28,0.22,0.20,0.10,0.14,0.06]}
    r["discount_preference"] = int(np.random.choice([1,2,3,4,5,6], p=dp_p[seg]))

    # ── VAN WESTENDORP PSM ──────────────────────────────────────────────
    base_b = {"Midnight Escapist":320,"Productivity Achiever":550,
               "Emotional Reader":380,"Curious Explorer":280,"Non-Reader":180}[seg]
    inc_boost = r["monthly_income_midpoint"] / 1000 * 8
    psm_b   = max(50,  int(np.random.normal(base_b + inc_boost, 80)))
    psm_tc  = max(30,  int(np.random.normal(psm_b * 0.45, 60)))
    psm_eo  = max(psm_b+50, int(np.random.normal(psm_b * 1.65, 100)))
    psm_te  = max(psm_eo+100, int(np.random.normal(psm_b * 2.40, 150)))
    r["psm_too_cheap"]    = psm_tc
    r["psm_bargain"]      = psm_b
    r["psm_expensive_ok"] = psm_eo
    r["psm_too_expensive"]= psm_te

    # ── SPENDING ────────────────────────────────────────────────────────
    bs_p = {"Midnight Escapist":[0.12,0.22,0.35,0.20,0.08,0.03],
             "Productivity Achiever":[0.05,0.15,0.28,0.30,0.15,0.07],
             "Emotional Reader":[0.10,0.20,0.35,0.22,0.10,0.03],
             "Curious Explorer":[0.15,0.25,0.32,0.18,0.08,0.02],
             "Non-Reader":[0.52,0.28,0.12,0.05,0.02,0.01]}
    r["monthly_book_spend"] = int(np.random.choice([0,100,350,750,1500,2500], p=bs_p[seg]))

    ls_p2 = {"Midnight Escapist":[0.10,0.30,0.35,0.18,0.07],
              "Productivity Achiever":[0.05,0.18,0.30,0.30,0.17],
              "Emotional Reader":[0.12,0.28,0.35,0.18,0.07],
              "Curious Explorer":[0.15,0.30,0.32,0.16,0.07],
              "Non-Reader":[0.38,0.35,0.18,0.07,0.02]}
    r["lifestyle_spend"] = int(np.random.choice([250,1000,2250,4500,7500], p=ls_p2[seg]))

    mss_p = {"Midnight Escapist":[0.25,0.30,0.25,0.12,0.08],
              "Productivity Achiever":[0.10,0.22,0.32,0.24,0.12],
              "Emotional Reader":[0.20,0.30,0.28,0.16,0.06],
              "Curious Explorer":[0.28,0.32,0.24,0.12,0.04],
              "Non-Reader":[0.55,0.28,0.12,0.04,0.01]}
    r["max_single_spend"] = int(np.random.choice([150,350,750,1500,2500], p=mss_p[seg]))

    # ── DIGITAL BEHAVIOUR ───────────────────────────────────────────────
    shop_p = {
        "shops_amazon":   {"Midnight Escapist":0.65,"Productivity Achiever":0.70,
                           "Emotional Reader":0.62,"Curious Explorer":0.58,"Non-Reader":0.55},
        "shops_flipkart": {"Midnight Escapist":0.45,"Productivity Achiever":0.42,
                           "Emotional Reader":0.40,"Curious Explorer":0.38,"Non-Reader":0.35},
        "shops_d2c":      {"Midnight Escapist":0.28,"Productivity Achiever":0.38,
                           "Emotional Reader":0.30,"Curious Explorer":0.32,"Non-Reader":0.10},
        "shops_instagram":{"Midnight Escapist":0.38,"Productivity Achiever":0.22,
                           "Emotional Reader":0.32,"Curious Explorer":0.62,"Non-Reader":0.12},
        "shops_whatsapp": {"Midnight Escapist":0.20,"Productivity Achiever":0.15,
                           "Emotional Reader":0.18,"Curious Explorer":0.25,"Non-Reader":0.15},
        "shops_qcommerce":{"Midnight Escapist":0.22,"Productivity Achiever":0.28,
                           "Emotional Reader":0.18,"Curious Explorer":0.20,"Non-Reader":0.12},
        "shops_offline":  {"Midnight Escapist":0.28,"Productivity Achiever":0.22,
                           "Emotional Reader":0.30,"Curious Explorer":0.22,"Non-Reader":0.25},
    }
    for s, probs in shop_p.items():
        r[s] = int(np.random.random() < probs[seg])

    pm_p = {"Midnight Escapist":[0.60,0.18,0.06,0.14,0.02],
             "Productivity Achiever":[0.48,0.18,0.20,0.08,0.06],
             "Emotional Reader":[0.58,0.18,0.08,0.12,0.04],
             "Curious Explorer":[0.62,0.14,0.06,0.12,0.06],
             "Non-Reader":[0.55,0.20,0.05,0.18,0.02]}
    r["payment_method"] = int(np.random.choice([1,2,3,4,5], p=pm_p[seg]))

    ib_mu = {"Midnight Escapist":2.2,"Productivity Achiever":3.0,
              "Emotional Reader":2.5,"Curious Explorer":1.8,"Non-Reader":3.2}
    r["impulse_buying"] = int(np.clip(round(np.random.normal(ib_mu[seg], 0.8)), 1, 4))

    # ── LOYALTY ─────────────────────────────────────────────────────────
    nps_mu = {"Midnight Escapist":7.2,"Productivity Achiever":6.8,
               "Emotional Reader":7.0,"Curious Explorer":7.8,"Non-Reader":3.5}
    r["nps_proxy"] = int(np.clip(round(np.random.normal(nps_mu[seg], 1.8)), 0, 10))

    sw_p = {"Midnight Escapist":[0.35,0.30,0.22,0.13],
             "Productivity Achiever":[0.40,0.30,0.20,0.10],
             "Emotional Reader":[0.38,0.30,0.22,0.10],
             "Curious Explorer":[0.28,0.28,0.28,0.16],
             "Non-Reader":[0.20,0.25,0.30,0.25]}
    r["switching_tendency"] = int(np.random.choice([1,2,3,4], p=sw_p[seg]))

    # ── INTENT ──────────────────────────────────────────────────────────
    plat_p = {"Midnight Escapist":[0.35,0.38,0.15,0.08,0.04],
               "Productivity Achiever":[0.28,0.35,0.22,0.10,0.05],
               "Emotional Reader":[0.30,0.36,0.20,0.10,0.04],
               "Curious Explorer":[0.30,0.35,0.20,0.10,0.05],
               "Non-Reader":[0.04,0.08,0.20,0.35,0.33]}
    r["platform_interest"] = int(np.random.choice([1,2,3,4,5], p=plat_p[seg]))

    share_p = {"Midnight Escapist":[0.30,0.32,0.20,0.12,0.06],
                "Productivity Achiever":[0.20,0.28,0.25,0.18,0.09],
                "Emotional Reader":[0.22,0.28,0.25,0.16,0.09],
                "Curious Explorer":[0.40,0.30,0.16,0.10,0.04],
                "Non-Reader":[0.05,0.10,0.20,0.32,0.33]}
    r["share_intent"] = int(np.random.choice([1,2,3,4,5], p=share_p[seg]))

    rhs_p = {"Midnight Escapist":[0.40,0.25,0.15,0.12,0.08],
              "Productivity Achiever":[0.45,0.22,0.14,0.12,0.07],
              "Emotional Reader":[0.38,0.28,0.16,0.12,0.06],
              "Curious Explorer":[0.38,0.20,0.22,0.14,0.06],
              "Non-Reader":[0.05,0.20,0.22,0.25,0.28]}
    r["reading_habit_status"] = int(np.random.choice([1,2,3,4,5], p=rhs_p[seg]))

    pur_p = {"Midnight Escapist":[0.35,0.38,0.15,0.08,0.04],
              "Productivity Achiever":[0.28,0.34,0.22,0.12,0.04],
              "Emotional Reader":[0.30,0.36,0.20,0.10,0.04],
              "Curious Explorer":[0.28,0.35,0.22,0.10,0.05],
              "Non-Reader":[0.03,0.07,0.18,0.36,0.36]}
    r["purchase_intent"] = int(np.random.choice([1,2,3,4,5], p=pur_p[seg]))

    # ── DERIVED TARGETS ─────────────────────────────────────────────────
    r["will_buy"]     = int(r["purchase_intent"] <= 2)
    r["format_class"] = (1 if r["format_preference"] <= 2 else
                         2 if r["format_preference"] == 3 else 3)
    rows.append(r)

df = pd.DataFrame(rows)

# ── INJECT NOISE (~8%) ──────────────────────────────────────────────────
noise_idx = np.random.choice(df.index, size=int(N*0.08), replace=False)
likert_cols = ["openness_score","conscientiousness_score","extraversion_score",
               "agreeableness_score","neuroticism_score","personalization_importance",
               "social_sharing_level","subscription_interest","eco_importance",
               "gifting_behaviour","impulse_buying","switching_tendency"]
for idx in noise_idx:
    df.at[idx, np.random.choice(likert_cols)] = np.random.choice([1,2,3,4,5])

# ── PSM INVERSIONS (~5%) ────────────────────────────────────────────────
inv_idx = np.random.choice(df.index, size=int(N*0.05), replace=False)
for idx in inv_idx:
    df.at[idx,"psm_too_cheap"] = int(df.at[idx,"psm_bargain"] + np.random.randint(50,200))

# ── MISSING VALUES (~3%) ────────────────────────────────────────────────
miss_idx = np.random.choice(df.index, size=int(N*0.03), replace=False)
for idx in miss_idx:
    df.at[idx,"psm_bargain"] = np.nan

# ── OUTLIER HIGH SPENDERS (~2%) ─────────────────────────────────────────
high_idx = np.random.choice(df.index, size=int(N*0.02), replace=False)
for idx in high_idx:
    df.at[idx,"max_single_spend"]    = int(np.random.choice([2500,3500,5000]))
    df.at[idx,"psm_too_expensive"]   = int(np.random.randint(5000,8000))
    df.at[idx,"monthly_income_midpoint"] = 120000

# ── DATA QUALITY FLAG ────────────────────────────────────────────────────
df["data_quality_flag"] = "clean"
df.loc[inv_idx,  "data_quality_flag"] = "psm_inversion"
df.loc[miss_idx, "data_quality_flag"] = "missing_psm"
df.loc[high_idx, "data_quality_flag"] = "outlier_spend"

print(f"Generated: {df.shape[0]} rows x {df.shape[1]} columns")
print(df["dna_segment"].value_counts().to_string())
print(f"Will-buy rate: {df['will_buy'].mean():.1%}")
df.to_csv("book_dna_data.csv", index=False)
print("Saved: book_dna_data.csv")
