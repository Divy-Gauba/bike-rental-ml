# 🚲 Hourly Bike Sharing Demand Forecasting Study

An end-to-end Machine Learning and Data Science study predicting hourly bike rental demand using the Capital Bikeshare system dataset. The final solution deploys an **XGBoost Regressor** enriched with **K-Means Demand Regimes** and **cyclical time encodings**, achieving **$R^2 = 0.9522$** and **RMSE = 38.90 bikes/hour**.

---

## 📌 Executive Summary

| Metric / Dimension | Detail |
| :--- | :--- |
| **Dataset Size** | 17,379 hourly observations across 2011–2012 (100% complete, 0 nulls) |
| **Primary Target** | `cnt` (Total hourly rentals = `casual` + `registered`) |
| **Champion Model** | **XGBoost Regressor** (Trained on $\log(1 + cnt)$ with shrinkage & subsampling) |
| **Champion Performance** | **$R^2 = 0.9522$**, **RMSE = 38.90**, **MAE = 23.23 bikes/hr** |
| **Median Prediction Error**| **13.4 bikes/hour** (70.4% within $\pm 25$ bikes) |
| **Serialized Artifact** | `bike_demand_xgboost_bundle.pkl` (~706 KB, fully self-contained) |
| **Interactive UI** | Streamlit application (`app.py`) |

---

## 🔬 Project Phases & Scientific Architecture

### Phase 1: Exploratory Data Analysis (EDA)
* **Data Volume & Integrity:** 17,379 observations, 17 features. Zero missing records and zero duplicates.
* **Target Breakdown & Leakage Hazard:** Identified that `casual + registered == cnt` strictly holds across 100% of rows. To eliminate fatal target leakage, `casual` and `registered` are strictly removed prior to model training.
* **Diurnal Dynamics:** Demand exhibits distinct bimodal peaks during commuter rush hours (**8:00 AM** and **5:00–6:00 PM**) on working days, while non-working days exhibit a broad midday leisure curve.
* **Environmental Impact:** Severe precipitation/snow drops demand by over 63% compared to clear conditions.

### Phase 2: Feature Engineering
1. **Calendar & Commute Indicators:**
   * `rush_hour`: Identifies morning (7–9 AM) and evening (4–7 PM) spikes on working days.
   * `is_weekend`: Identifies leisure days.
   * `day`: Day of month extracted from `dteday`.
2. **Cyclical Periodic Encodings:**
   * `hr_sin` & `hr_cos` (period 24) and `mnth_sin` & `mnth_cos` (period 12) preserve angular continuity (e.g. 23:00 is contiguous to 00:00).
3. **Weather Comfort Proxies:**
   * `temp_diff`: Gap between perceived and actual temperature (`atemp - temp`).
   * `temp_hum_interaction`: Discomfort/heat index proxy (`temp * hum`).
   * `bad_weather`: Precipitation indicator (`weathersit >= 3`).
4. **Multicollinearity Elimination:**
   * Dropped `atemp` due to extreme collinearity with `temp` ($r = 0.99$).
   * Dropped `instant` (index) and raw `dteday`.
5. **Target Transformation:**
   * Applied $\log(1 + cnt)$, reducing target skewness from **+1.28** to **-0.82**, stabilizing residual variance and naturally preventing negative count predictions.

### Phase 3 & 4: Model Benchmarking

Evaluated strictly on true rental count scale ($\text{exp}(\hat{y}) - 1$):

| Rank | Model Family | RMSE | MAE | $R^2$ Score | Key Note |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **XGBoost Regressor** | **38.90** | **23.23** | **0.9522** | **Champion:** Lowest error & tightest variance |
| 🥈 | **Random Forest Regressor** | 41.01 | 23.77 | 0.9469 | Strong bagged baseline |
| 🥉 | **Hist Gradient Boosting** | 41.18 | 24.87 | 0.9465 | Fast scikit-learn GBDT |
| 4 | **Ridge Regression ($L_2$)** | 127.20 | 74.33 | 0.4890 | Linear baseline; misses non-linear thresholds |
| 5 | **Dummy Regressor (Mean)** | 198.83 | 137.44 | -0.2484 | Sanity benchmark |

### Phase 5: Diagnostic Residual & Error Analysis
* **Bias:** Mean residual is **$-0.17$ bikes**, proving the model has zero systematic directional bias.
* **Symmetry:** Errors are normally distributed with tight concentration around zero.
* **Outlier Behavior:** The largest error residuals occur exclusively during 900+ surge events where extreme tail demand is conservatively pulled inward by regularization.

### Phase 6: Production Pipeline & Serialization
* Serialized `bike_demand_xgboost_bundle.pkl` containing the trained estimator, the K-Means regime mapping, and the training feature schema.
* Packaged `predict_bike_demand()` for real-time one-step inference from raw inputs.

---

## 🚀 Running the Interactive Streamlit App

A live interactive dashboard is provided in `app.py`.

```bash
# Run the interactive Streamlit dashboard
streamlit run app.py
```

The app lets operators and stakeholders dynamically adjust hour, day type, temperature, and weather to visualize immediate predicted demand and operational regime alerts.

---

## 📂 Project Directory Structure

```text
├── Bike_Rental_EDA.ipynb             # Master Data Science & ML Notebook (Phases 1-7)
├── app.py                            # Streamlit Interactive Web Application
├── bike_demand_xgboost_bundle.pkl    # Serialized Champion Model Bundle
├── hour.csv                          # Raw Hourly Bike Dataset (17,379 rows)
├── figures/
│   └── model_diagnostics.png         # Diagnostic charts (Scatter, Residuals, Curves)
└── README.md                         # Executive Summary Report
```
