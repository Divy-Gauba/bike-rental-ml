# Project Explanation: Hourly Bike Sharing Demand Forecasting

This document explains the full project end to end: the business problem, the data, the exploratory findings, the feature engineering, the models, the evaluation, and the interactive application. It is written for a reader who wants to understand both what the project does and why each design choice was made.

For the audit of known issues, see [`FLAGS_AND_LIMITATIONS.md`](FLAGS_AND_LIMITATIONS.md).

---

## 1. What the project does

The project predicts the number of bike rentals in a given hour for a bike-sharing system, using calendar, weather, and time-of-day information. It is a supervised regression problem built on the Capital Bikeshare hourly dataset.

The final deliverable has three parts:

1. A trained **XGBoost regressor** that forecasts hourly demand.
2. A serialized **model bundle** that can be loaded without retraining.
3. An interactive **Streamlit dashboard** for scenario analysis, forecast visualization, prediction explanation, and sensitivity testing.

The model's headline holdout performance is:

| Metric | Value |
| :--- | ---: |
| R-squared | 0.9522 |
| RMSE | 38.90 bikes/hour |
| MAE | 23.23 bikes/hour |
| Median absolute error | 13.4 bikes/hour |
| Predictions within +/- 25 bikes | 70.4% |

These metrics come from the project's original 80/20 random holdout. Their limitations are discussed in F4 in the flags document.

---

## 2. Dataset

| Property | Value |
| :--- | :--- |
| Source | UCI Machine Learning Repository, Capital Bikeshare |
| File | `hour.csv` |
| Rows | 17,379 hourly observations |
| Period | 1 January 2011 to 31 December 2012 |
| Original columns | 17 |
| Missing values | 0 |
| Duplicate rows | 0 |

### Column overview

| Column | Meaning |
| :--- | :--- |
| `instant` | Row index (dropped) |
| `dteday` | Date (used to derive calendar features, then dropped) |
| `season` | 1 = spring, 2 = summer, 3 = fall, 4 = winter |
| `yr` | 0 = 2011, 1 = 2012 |
| `mnth` | Month, 1-12 |
| `hr` | Hour, 0-23 |
| `holiday` | 1 if the day is a public holiday |
| `weekday` | 0 = Sunday, 1 = Monday, ..., 6 = Saturday (UCI encoding) |
| `workingday` | 1 if neither weekend nor holiday |
| `weathersit` | 1 = clear, 2 = mist/cloudy, 3 = light rain/snow, 4 = heavy rain/storm |
| `temp` | Normalized temperature, Celsius / 41 |
| `atemp` | Normalized perceived temperature, Celsius / 50 |
| `hum` | Normalized humidity, percent / 100 |
| `windspeed` | Normalized wind speed, km/h / 67 |
| `casual` | Rentals by casual users |
| `registered` | Rentals by registered users |
| `cnt` | Total rentals (target) |

### Target and leakage

The target is `cnt`. A critical property is that `casual + registered == cnt` holds for every row. Keeping either component as a feature would leak the target, so both are removed before modeling. This is the single most important data-cleaning decision in the project.

---

## 3. Exploratory data analysis: what was learned

### 3.1 Daily cycle

Demand is strongly bimodal on working days:

- A morning commute peak around 08:00.
- An evening commute peak around 17:00-18:00.
- Very low demand between 00:00 and 05:00.

On weekends and holidays the pattern changes to a single broad midday plateau, peaking around 12:00-15:00. This is why day type and hour interactions matter.

### 3.2 Weather effects

- Clear conditions produce the highest demand.
- Mist and cloud reduce demand slightly.
- Rain and snow reduce it substantially.
- The README reports that severe precipitation or snow lowers demand by more than 60% relative to clear conditions.

### 3.3 Temperature and comfort

Demand rises with temperature up to a comfortable range around the mid-20s Celsius, then flattens or declines at extremes. Humidity has a milder negative effect, mostly at very high levels. Wind speed has a small negative effect.

### 3.4 Growth between 2011 and 2012

The system grew year over year, so `yr` carries real signal. In the model, 2012 demand is substantially higher than 2011 for otherwise similar conditions.

### 3.5 Multicollinearity

`temp` and `atemp` correlate at roughly 0.99. Keeping both adds no information and can destabilize linear models, so `atemp` is dropped. A derived `temp_diff = atemp - temp` is kept as a comfort proxy in training.

---

## 4. Feature engineering

The model uses 22 features. All are derived from raw inputs at prediction time.

### 4.1 Calendar and commute features

| Feature | Construction | Purpose |
| :--- | :--- | :--- |
| `season` | From month: Jan-Mar spring, Apr-Jun summer, Jul-Sep fall, Oct-Dec winter | Captures seasonal demand level |
| `yr` | 0 for 2011, 1 for 2012 | Captures system growth |
| `mnth` | Month number | Seasonal detail |
| `hr` | Hour of day | Strongest demand driver |
| `holiday` | Binary | Distinguishes holidays from normal days |
| `weekday` | UCI encoding, 0 = Sunday | Day-of-week effects |
| `workingday` | 1 if not weekend and not holiday | Commute vs leisure behavior |
| `day` | Day of month | Minor within-month variation |
| `is_weekend` | 1 for Saturday or Sunday | Explicit leisure-day flag |
| `rush_hour` | 1 on working days during hours 7-9 and 16-19 | Captures commute spikes |

### 4.2 Cyclical time encodings

Raw hour and month are circular: 23:00 is adjacent to 00:00, and December is adjacent to January. Sine and cosine pairs preserve that continuity:

```text
hr_sin     = sin(2 * pi * hr / 24)
hr_cos     = cos(2 * pi * hr / 24)
mnth_sin   = sin(2 * pi * (mnth - 1) / 12)
mnth_cos   = cos(2 * pi * (mnth - 1) / 12)
```

### 4.3 Weather and comfort proxies

| Feature | Construction | Purpose |
| :--- | :--- | :--- |
| `temp` | Celsius / 41 | Thermal comfort |
| `hum` | Percent / 100 | Humidity discomfort |
| `windspeed` | km/h / 67 | Wind discomfort |
| `temp_diff` | `atemp - temp` during training | Perceived vs actual temperature gap |
| `temp_hum_interaction` | `temp * hum` | Heat/discomfort interaction |
| `bad_weather` | 1 if `weathersit >= 3` | Precipitation indicator |
| `weathersit` | 1-4 category | Weather severity |

### 4.4 K-Means demand regimes

A 3-cluster K-Means model is fit on the average hourly profile using `casual` and `registered` rentals. The 24 hours are grouped into three behavioral regimes, then the cluster label is attached to every row through its hour.

| Cluster | Hours | Mean `cnt` | Interpretation |
| :--- | :--- | ---: | :--- |
| 0 | 8, 17, 18 | 415.4 | Peak hours |
| 1 | 0-6, 22, 23 | 49.7 | Night, low demand |
| 2 | 7, 9-16, 19-21 | 236.2 | Daytime and shoulder hours |

The label `demand_regime` is included as a feature. It has high measured importance (0.1574), but it is a deterministic function of hour, so it compresses existing information rather than adding a new signal. See F5 in the flags document.

### 4.5 Target transformation

The target is transformed as:

```text
y = log(1 + cnt)
```

This reduces skewness from about +1.28 to about -0.82 and prevents negative predictions. Predictions are mapped back with:

```text
cnt_hat = max(exp(y_hat) - 1, 0)
```

---

## 5. Modeling

### 5.1 Candidate models

Five model families were benchmarked on the count scale after back-transformation:

| Rank | Model | RMSE | MAE | R-squared |
| :---: | :--- | ---: | ---: | ---: |
| 1 | **XGBoost Regressor** | **38.90** | **23.23** | **0.9522** |
| 2 | Random Forest Regressor | 41.01 | 23.77 | 0.9469 |
| 3 | Hist Gradient Boosting | 41.18 | 24.87 | 0.9465 |
| 4 | Ridge Regression (L2) | 127.20 | 74.33 | 0.4890 |
| 5 | Dummy Regressor (mean) | 198.83 | 137.44 | -0.2484 |

The result confirms the expected pattern: tree ensembles capture nonlinear interactions between hour, day type, weather, and temperature, while a linear model underfits the daily cycle.

### 5.2 Champion configuration

```python
XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
```

Key choices:

- **Shallow-to-moderate depth (6):** enough to model hour-by-weather interactions without overfitting.
- **Learning rate 0.08 with 150 trees:** a balance between bias and variance.
- **Row and column subsampling (0.8):** regularization and decorrelation across trees.
- **Log target:** stabilizes variance and enforces non-negativity after inversion.

### 5.3 Feature importance

Top XGBoost importances from the serialized model:

| Feature | Importance |
| :--- | ---: |
| `hr` | 0.2147 |
| `demand_regime` | 0.1574 |
| `hr_sin` | 0.1019 |
| `hr_cos` | 0.0970 |
| `rush_hour` | 0.0928 |
| `is_weekend` | 0.0701 |
| `workingday` | 0.0578 |
| `yr` | 0.0428 |
| `temp` | 0.0416 |
| `season` | 0.0375 |

Time-of-day signals dominate, followed by day-type behavior, system growth, and weather. This matches the EDA and domain intuition.

### 5.4 Residual diagnostics

- Mean residual: -0.17 bikes, indicating no meaningful systematic bias.
- Median absolute error: 13.4 bikes/hour.
- 70.4% of predictions within +/- 25 bikes.
- Largest errors occur during extreme demand surges above 900 rentals, where regularization pulls predictions toward the center.

---

## 6. The interactive application

The dashboard is implemented in `app.py` with Streamlit. It loads `bike_demand_xgboost_bundle.pkl` once and reuses it across reruns.

### 6.1 Scenario inputs

The sidebar provides:

- Date, constrained to 2011-01-01 through 2012-12-31.
- Hour of day, 0-23.
- Public holiday toggle.
- Weather condition, 1-4.
- Temperature in Celsius.
- Humidity in percent.
- Wind speed in km/h.

### 6.2 Automatic calendar derivation

To prevent inconsistent scenarios, the app derives the following from the selected date and holiday flag:

- Month and day of month.
- Season from month.
- Weekday in the UCI encoding used during training.
- Weekend flag.
- Working-day flag.

This removes the class of invalid inputs described in F2 and fixes the weekday encoding issue in F1.

### 6.3 Main dashboard

- Four KPI cards: predicted hourly demand, projected daily total, expected peak hour, and operational demand level.
- An operational note generated from the predicted volume: low, moderate, high, or very high.
- A 24-hour forecast chart with the selected hour highlighted.
- An optional weekend or working-day comparison curve that holds weather and month fixed.
- A downloadable CSV forecast and a per-hour table with each hour's share of the daily total.

### 6.4 Prediction drivers

This tab shows the top 10 XGBoost feature contributions for the selected scenario, computed with `pred_contribs=True`. Positive values push the prediction higher in log space; negative values pull it lower. This makes the model's reasoning visible instead of presenting a single opaque number.

### 6.5 Sensitivity tab

Two partial-dependence style views:

- **Temperature response:** predicted demand from 0 to 41 Celsius at the selected hour, with the current temperature marked.
- **Weather response:** predicted demand across the four weather categories, with the selected condition highlighted.

These views let a user see the shape and strength of each effect in the current context.

### 6.6 Running the app

```bash
streamlit run app.py
```

The app opens in a browser. No training step is required because the bundle is already serialized.

---

## 7. How a prediction is produced

For a given scenario, the app performs these steps:

1. Convert the date into month, day, season, UCI weekday, weekend flag, and year index.
2. Convert temperature, humidity, and wind into the same normalized scales used in training.
3. Compute categorical and cyclical features: `rush_hour`, `hr_sin`, `hr_cos`, `mnth_sin`, `mnth_cos`.
4. Compute weather proxies: `temp_hum_interaction`, `bad_weather`, and `temp_diff`.
5. Look up `demand_regime` from the serialized hour-to-cluster mapping.
6. Assemble the 22 features in the exact training order.
7. Call `model.predict` to obtain log demand.
8. Convert back with `max(exp(y_hat) - 1, 0)` and round to a whole number of bikes.

The 24-hour chart repeats steps 1-8 for each hour from 0 to 23.

---

## 8. File structure

```text
ML Bike Rental/
|-- Bike_Rental_EDA.ipynb             # EDA, feature engineering, benchmarks, diagnostics
|-- app.py                            # Interactive Streamlit dashboard
|-- bike_demand_xgboost_bundle.pkl    # Serialized model, regime mapping, feature schema
|-- hour.csv                          # Raw hourly dataset
|-- figures/                          # EDA and diagnostic figures
|   |-- fig1_target_distribution.png
|   |-- fig2_hourly_by_daytype.png
|   |-- fig3_temp_vs_cnt.png
|   |-- fig4_correlation_matrix.png
|   `-- model_diagnostics.png
|-- assets/                           # Animated SVG assets used by the README
|   |-- workflow.svg
|   |-- structure.svg
|   |-- model_leaderboard.svg
|   |-- demand_curve.svg
|   |-- feature_importance.svg
|   `-- generate_assets.py
|-- README.md                         # Executive summary
|-- PROJECT_EXPLANATION.md            # This document
`-- FLAGS_AND_LIMITATIONS.md          # Audit findings and recommendations
```

---

## 9. Key design decisions and why

| Decision | Rationale |
| :--- | :--- |
| Drop `casual` and `registered` | They sum to the target and would cause leakage. |
| Drop `atemp` | Near-perfect collinearity with `temp` (r = 0.99). |
| Log-transform the target | Reduces skew, stabilizes variance, prevents negative predictions. |
| Add cyclical encodings | Preserves continuity at day and year boundaries. |
| Add `rush_hour` | Encodes the commute interaction between hour and working day. |
| Use XGBoost | Best accuracy, handles nonlinear interactions, compact and fast. |
| Serialize a bundle | Deployment without retraining; consistent feature schema. |
| Derive calendar fields in the app | Prevents contradictory user inputs and encoding mistakes. |
| Show feature contributions | Makes model behavior transparent for each scenario. |
| Add sensitivity curves | Communicates effect shapes and supports what-if analysis. |

---

## 10. Assumptions and limitations

- The model is valid only for the 2011-2012 period and the Capital Bikeshare system.
- Reported metrics come from a random holdout and are likely optimistic relative to future forecasting. See F4.
- `demand_regime` is hour-only and derived from the full dataset. See F5.
- `temp_diff` is set to zero at inference. See F6.
- Storm weather has only three training examples, so its estimate equals light rain/snow. See F7.
- Forecasts are point estimates with no uncertainty intervals. See F11.
- The 24-hour total is a sum of independent point forecasts and should be treated as an expected value.
- The app intentionally blocks dates outside the training window rather than extrapolating.

---

## 11. Glossary

| Term | Meaning |
| :--- | :--- |
| **Target / target leakage** | The value being predicted; leakage is when information unavailable at prediction time is used as a feature. |
| **RMSE** | Root mean squared error; penalizes large errors more strongly than MAE. |
| **MAE** | Mean absolute error; average absolute difference in bikes per hour. |
| **R-squared** | Share of variance explained; 1.0 is perfect, 0 is no better than the mean. |
| **Cyclical encoding** | Sine/cosine transformation that preserves circular continuity of time. |
| **K-Means** | Unsupervised clustering that groups hours by their demand profile. |
| **XGBoost** | Gradient-boosted decision tree ensemble. |
| **Log target** | Training on `log(1 + count)` to stabilize variance and avoid negative predictions. |
| **Partial dependence** | How a prediction changes as one input varies while others are held fixed. |
| **Feature contribution** | The additive effect of one feature on a single prediction in log space. |
