"""
Big Data-Driven Apartment Rent Price Prediction — Streamlit Web App
====================================================================
Front-end demo for the PySpark Big Data Analytics project.

Loads model_bundle.pkl (exported by Rent_Prediction_PySpark.ipynb, Phase 7) and
serves live rent predictions plus data insights and model comparison — no Spark
needed at serve time.

Run locally:  streamlit run app.py
Run in Colab: use the launch cells at the bottom of the notebook.
"""
import math
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Apartment Rent Predictor · PySpark",
                   page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
  .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
  h1, h2, h3 {color: #21295C;}
  .hero {background: linear-gradient(120deg,#21295C 0%,#065A82 60%,#1C7293 100%);
         padding: 1.4rem 1.6rem; border-radius: 14px; color: #fff; margin-bottom: 1.1rem;}
  .hero h1 {color:#fff; margin:0; font-size:1.7rem;}
  .hero p  {color:#CADCFC; margin:.35rem 0 0 0; font-size:.95rem;}
  .metric-card {background:#EEF4F7; border:1px solid #D9E2E8; border-radius:12px;
                padding:1rem 1.1rem; text-align:center;}
  .metric-card .lbl {color:#5A6B7B; font-size:.8rem; font-weight:600; text-transform:uppercase;}
  .metric-card .val {color:#065A82; font-size:1.5rem; font-weight:800; margin-top:.2rem;}
  .big-price {background:linear-gradient(120deg,#028090,#00A896); color:#fff;
              border-radius:14px; padding:1.2rem 1.4rem; text-align:center;}
  .big-price .p {font-size:2.6rem; font-weight:800; margin:.1rem 0;}
  .stTabs [data-baseweb="tab-list"] {gap: 4px;}
  .stTabs [data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- load model
@st.cache_resource
def load_bundle(path="model_bundle.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    B = load_bundle()
except Exception as e:
    st.error(f"Could not load **model_bundle.pkl** — run the notebook's Phase 7 first. ({e})")
    st.stop()

MODEL          = B["model"]
FEATURE_COLS   = B["feature_cols"]
CAT_MAPS       = B["cat_index_maps"]
TOP_LOCALITIES = B["top_localities"]
MED_AREA       = B["median_floor_area"]
PREMIUM        = B["premium_amenities"]
METRICS        = B.get("metrics", {})
IMPORTANCES    = B.get("feature_importances", {})
EDA            = B.get("eda", {})
CUR            = B.get("currency", "GHS")

NICE = {  # human-friendly feature names for charts
    "has_ac":"Air Conditioning","has_wifi":"WiFi","has_wardrobe":"Wardrobe","has_hotwater":"Hot Water",
    "has_tv":"TV","has_microwave":"Microwave","has_fridge":"Fridge","has_balcony":"Balcony",
    "has_security":"Security","has_pool":"Pool","amenity_count":"Amenity Count","luxury_score":"Luxury Score",
    "is_new":"Is New","furnished":"Furnished","has_parking":"Parking","bedrooms":"Bedrooms",
    "bathrooms":"Bathrooms","floor_area_capped":"Floor Area","log_floor_area":"Floor Area (log)",
    "rooms_total":"Total Rooms","bed_bath_ratio":"Bed/Bath Ratio","dist_accra":"Distance from Accra",
    "category_idx":"Property Type","region_idx":"Region","locality_top_idx":"Locality",
}

# ----------------------------------------------------------------- feature builder
def haversine(lat, lng, c=(5.56, -0.20)):
    R = 6371.0
    la1, lo1 = math.radians(c[0]), math.radians(c[1])
    la2, lo2 = math.radians(lat), math.radians(lng)
    d = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R*math.asin(min(1, math.sqrt(d)))

def build_row(inp):
    a = inp["amenities"].lower()
    flags = {
        "has_ac":        int("air condition" in a or " ac" in a or "a/c" in a),
        "has_wifi":      int("wifi" in a or "internet" in a or "wi-fi" in a),
        "has_wardrobe":  int("wardrobe" in a or "closet" in a),
        "has_hotwater":  int("hot water" in a or "water heater" in a),
        "has_tv":        int(" tv" in a or "television" in a),
        "has_microwave": int("microwave" in a),
        "has_fridge":    int("fridge" in a or "refrigerator" in a),
        "has_balcony":   int("balcony" in a),
        "has_security":  int("security" in a or "cctv" in a or "gated" in a),
        "has_pool":      int("pool" in a or "swimming" in a),
    }
    loc_top = inp["locality"] if inp["locality"] in TOP_LOCALITIES else "Other"
    row = dict(flags)
    row["amenity_count"] = len([x for x in inp["amenities"].split(",") if x.strip()])
    row["luxury_score"]  = sum(flags[k] for k in PREMIUM)
    row["is_new"]        = int(inp["condition"] == "New")
    row["furnished"]     = int(inp["furnished"])
    row["has_parking"]   = int(inp["parking"])
    row["bedrooms"]      = inp["bedrooms"]
    row["bathrooms"]     = inp["bathrooms"]
    row["floor_area_capped"] = inp["floor_area"]
    row["log_floor_area"]    = math.log1p(inp["floor_area"])
    row["rooms_total"]       = inp["bedrooms"] + inp["bathrooms"]
    row["bed_bath_ratio"]    = inp["bedrooms"] / (inp["bathrooms"] + 1.0)
    row["dist_accra"]        = haversine(inp["lat"], inp["lng"])
    row["category_idx"]      = CAT_MAPS["category"].get(inp["category"], len(CAT_MAPS["category"]))
    row["region_idx"]        = CAT_MAPS["region"].get(inp["region"], len(CAT_MAPS["region"]))
    row["locality_top_idx"]  = CAT_MAPS["locality_top"].get(loc_top, len(CAT_MAPS["locality_top"]))
    return pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

# ----------------------------------------------------------------- hero header
st.markdown(f"""
<div class="hero">
  <h1>🏙️ Big Data-Driven Apartment Rent Price Prediction</h1>
  <p>Apache Spark (PySpark) MLlib pipeline · 25 engineered features · gradient-boosted serving model — Review demo</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("📋 Property details")
    categories = sorted(CAT_MAPS["category"].keys())
    regions    = sorted(CAT_MAPS["region"].keys())
    localities = sorted([l for l in TOP_LOCALITIES if l != "Other"])

    category = st.selectbox("Property type", categories,
                            index=categories.index("Flats") if "Flats" in categories else 0)
    region   = st.selectbox("Region", regions,
                            index=regions.index("Greater Accra") if "Greater Accra" in regions else 0)
    locality = st.selectbox("Locality / town", localities + ["Other (not listed)"])
    if locality == "Other (not listed)":
        locality = "Other"

    c1, c2 = st.columns(2)
    bedrooms   = c1.number_input("Bedrooms", 1, 15, 3)
    bathrooms  = c2.number_input("Bathrooms", 1, 15, 2)
    floor_area = st.slider("Floor area (sqm)", 20, 800, 120, 5)
    condition  = st.radio("Condition", ["New", "Used"], horizontal=True)
    fc1, fc2 = st.columns(2)
    furnished = fc1.checkbox("Furnished", value=True)
    parking   = fc2.checkbox("Parking", value=True)

    st.markdown("**Amenities**")
    am_opts = ["Air Conditioning", "WiFi", "Wardrobe", "Hot Water", "TV",
               "Microwave", "Fridge", "Balcony", "Security", "Swimming Pool"]
    chosen = st.multiselect("Select amenities", am_opts,
                            default=["Air Conditioning", "WiFi", "Balcony", "Security"])

    with st.expander("📍 Location (lat/lng)"):
        lat = st.number_input("Latitude", value=5.60, format="%.4f")
        lng = st.number_input("Longitude", value=-0.18, format="%.4f")

    go = st.button("💰 Predict rent", type="primary")

# ----------------------------------------------------------------- tabs
tab_pred, tab_eda, tab_models, tab_about = st.tabs(
    ["🏠 Predict", "📊 Data Insights", "🤖 Model Comparison", "ℹ️ About"])

# =========================================================== TAB 1: PREDICT
with tab_pred:
    if go:
        inp = dict(category=category, region=region, locality=locality,
                   bedrooms=bedrooms, bathrooms=bathrooms, floor_area=floor_area,
                   condition=condition, furnished=furnished, parking=parking,
                   amenities=", ".join(chosen), lat=lat, lng=lng)
        X = build_row(inp)
        pred = float(np.expm1(MODEL.predict(X)[0]))
        lo, hi = pred*0.85, pred*1.15

        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"""
            <div class="big-price">
              <div style="font-size:.9rem;opacity:.9;">Estimated monthly rent</div>
              <div class="p">{CUR} {pred:,.0f}</div>
              <div style="font-size:.9rem;opacity:.9;">Typical range {CUR} {lo:,.0f} – {hi:,.0f} / month</div>
            </div>
            """, unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="lbl">Per year</div><div class="val">{CUR} {pred*12:,.0f}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="lbl">Per sqm</div><div class="val">{CUR} {pred/max(floor_area,1):,.0f}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="lbl">Dist. Accra</div><div class="val">{haversine(lat,lng):.0f} km</div></div>', unsafe_allow_html=True)

            if IMPORTANCES:
                st.markdown("##### 🔑 Top factors driving the estimate")
                s = pd.Series(IMPORTANCES).sort_values(ascending=False).head(8)
                s.index = [NICE.get(i, i) for i in s.index]
                st.bar_chart(s, color="#028090", horizontal=True)
        with right:
            st.markdown("##### 📝 Your inputs")
            st.dataframe(pd.DataFrame({
                "Field": ["Property type","Region","Locality","Bedrooms","Bathrooms",
                          "Floor area","Condition","Furnished","Parking","Amenities"],
                "Value": [category, region, locality, bedrooms, bathrooms,
                          f"{floor_area} sqm", condition, "Yes" if furnished else "No",
                          "Yes" if parking else "No", ", ".join(chosen) or "None"],
            }), hide_index=True, use_container_width=True)
            st.caption("Prediction served by an exported gradient-boosted model using the "
                       "identical 25 features engineered in the PySpark pipeline.")
    else:
        st.info("👈 Set the property details in the sidebar, then click **Predict rent**.")
        st.markdown("This tab estimates a fair monthly rent from apartment characteristics. "
                    "Explore the other tabs for the data insights and the model comparison behind it.")

# =========================================================== TAB 2: DATA INSIGHTS
with tab_eda:
    if not EDA:
        st.info("EDA summary not found in the bundle. Re-export from the notebook (Phase 7) to enable this tab.")
    else:
        st.subheader("Exploratory Data Analysis")
        k1,k2,k3,k4 = st.columns(4)
        k1.markdown(f'<div class="metric-card"><div class="lbl">Records</div><div class="val">{EDA["n_records"]:,}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="metric-card"><div class="lbl">Median rent</div><div class="val">{CUR} {EDA["price_median"]:,.0f}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="metric-card"><div class="lbl">Mean rent</div><div class="val">{CUR} {EDA["price_mean"]:,.0f}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="metric-card"><div class="lbl">Range</div><div class="val" style="font-size:1.05rem;">{CUR} {EDA["price_min"]:,.0f}–{EDA["price_max"]:,.0f}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Price distribution** (right-skewed → we model log-price)")
            if EDA.get("price_hist"):
                edges = EDA["price_hist_edges"]
                centers = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
                st.bar_chart(pd.DataFrame({"count": EDA["price_hist"]}, index=[f"{c:,.0f}" for c in centers]),
                             color="#065A82")
                st.caption(f"Skew: raw price {EDA['skew_price']:.2f} → log-price {EDA['skew_logprice']:.2f}")
        with c2:
            st.markdown("**Median rent by property type**")
            st.bar_chart(pd.Series(EDA["by_category"]), color="#8e44ad", horizontal=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Median rent by number of bedrooms**")
            st.bar_chart(pd.Series(EDA["by_bedrooms"]), color="#16a085")
        with c4:
            st.markdown("**Top localities by median rent**")
            st.bar_chart(pd.Series(EDA["top_localities"]), color="#d35400", horizontal=True)

        if EDA.get("furnished_premium"):
            fp = EDA["furnished_premium"]
            st.markdown("**Furnished vs unfurnished (median rent)**")
            st.bar_chart(pd.Series(fp), color="#27ae60")

# =========================================================== TAB 3: MODEL COMPARISON
with tab_models:
    if not METRICS:
        st.info("Model metrics not found in the bundle.")
    else:
        st.subheader("Model Comparison (held-out test set)")
        mdf = pd.DataFrame(METRICS).T.sort_values("R2", ascending=False)
        best = mdf.index[0]
        c1, c2 = st.columns([2,3])
        with c1:
            st.markdown(f'<div class="metric-card"><div class="lbl">Best model</div><div class="val" style="font-size:1.15rem;">{best}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card" style="margin-top:.6rem;"><div class="lbl">Best R²</div><div class="val">{mdf.loc[best,"R2"]:.3f}</div></div>', unsafe_allow_html=True)
            st.caption("R² = variance explained (higher is better). "
                       "RMSE / MAE are errors on log-price (lower is better).")
        with c2:
            st.markdown("**R² by model**")
            st.bar_chart(mdf["R2"], color="#065A82", horizontal=True)
        st.markdown("**Full metrics**")
        st.dataframe(mdf.style.format("{:.4f}").background_gradient(cmap="Greens", subset=["R2"]),
                     use_container_width=True)
        st.caption("The distributed Spark pipeline handles ingestion, cleaning, feature engineering and "
                   "tuning; XGBoost & LightGBM are added on the same features for comparison.")

# =========================================================== TAB 4: ABOUT
with tab_about:
    st.subheader("About this project")
    st.markdown("""
**Big Data-Driven Apartment Rent Price Prediction Using PySpark** is a Big Data Analytics
project that predicts apartment rental prices using a scalable, distributed pipeline built on
**Apache Spark (PySpark)**.

**Pipeline (7 phases in the notebook):**
1. Data ingestion & cleaning → Parquet
2. Exploratory Data Analysis
3. Feature engineering — **25 features** via Spark UDFs & MLlib transformers
4. Train & compare 6 models (LR, Decision Tree, Random Forest, GBT, XGBoost, LightGBM)
5. Hyperparameter tuning with 5-fold CrossValidator
6. Explainability (feature importance + SHAP) and error analysis
7. Deployment — export the model that powers **this app**

**Why Big Data / PySpark?** Distributed, in-memory processing that scales from thousands to
millions of listings, with MLlib for scalable machine learning.

**How this app works:** it loads a lightweight model bundle exported from the notebook and
rebuilds the same 25 features from your inputs — no Spark needed to serve, so it runs anywhere.
    """)
    with st.expander("🔧 Technical details"):
        st.write(f"- Serving model: **{type(MODEL).__name__}**")
        st.write(f"- Features: **{len(FEATURE_COLS)}**")
        st.write(f"- Records analysed: **{EDA.get('n_records','—'):,}**" if EDA else "- Records analysed: —")
        st.write(f"- Currency: **{CUR}**")
    st.caption("Review demo · Course: Big Data Analytics (BITE411L)")
