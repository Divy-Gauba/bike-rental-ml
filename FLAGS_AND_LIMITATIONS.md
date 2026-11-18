# Model and Application Audit: Flags and Limitations

**Project:** Hourly Bike Sharing Demand Forecasting (Capital Bikeshare, 2011-2012)
**Audit date:** 2026-11-19
**Artifacts reviewed:** `Bike_Rental_EDA.ipynb`, `app.py`, `bike_demand_xgboost_bundle.pkl`, `hour.csv`
**Environment:** Python 3.13.17, xgboost 3.4.1, scikit-learn 1.9.1, streamlit 1.67.0, pandas 3.1.1, numpy 2.5.4

---

## 1. How the audit was performed

1. Loaded the serialized bundle and inspected its keys, feature schema, and K-Means hour mapping.
2. Rebuilt the inference feature vector exactly as the app does and ran controlled sweeps over hour, day type, holiday, weather, temperature, humidity, wind, weekday encoding, and date/year.
3. Compared predictions against conditional means from the raw dataset for matching slices.
4. Checked the notebook's training split, target transformation, feature construction, metric computation, and serialization.
5. Exercised the rebuilt Streamlit app programmatically with `streamlit.testing.v1.AppTest` across 10+ scenarios (night, rush hour, weekend, holiday, storm, temperature extremes, date changes, comparison toggle).

All numerical examples below are reproducible with the dataset and bundle in this repository.

---

## 2. Severity legend

| Level | Meaning |
| :--- | :--- |
| **High** | Produces incorrect or misleading results in normal use, or materially overstates model quality. |
| **Medium** | Produces unrealistic outputs for valid-looking inputs, or creates training-serving skew. |
| **Low** | Cosmetic, documentation, or edge-case issue with limited practical impact. |
| **Info** | Verified behavior that is correct and should be preserved. |

---

## 3. Summary of findings

| ID | Severity | Area | Finding | Status |
| :--- | :--- | :--- | :--- | :--- |
| F1 | **High** | Inference | Weekday feature used Python encoding (0 = Monday) while the model was trained on UCI encoding (0 = Sunday). | Fixed in `app.py` |
| F2 | **High** | Inference | Impossible calendar/weather combinations were accepted: working day on weekends, holiday plus working day, season inconsistent with month. | Fixed in `app.py` |
| F3 | **Medium** | Inference | `yr = year - 2011` was unbounded; dates after 2012 silently reused the 2012 response. | Fixed in `app.py` |
| F4 | **High** | Validation | Random train/test split on hourly time-series data causes temporal leakage and optimistic metrics. | Open (notebook) |
| F5 | **Medium** | Validation | `demand_regime` was computed on the full dataset before splitting and is a deterministic function of `hr`. | Open (notebook) |
| F6 | **Low-Medium** | Inference | `temp_diff` is hardcoded to 0 at inference but was `atemp - temp` in training. | Open (documented) |
| F7 | **Low** | Model | `weathersit = 4` (storm) has only 3 training rows, so predictions collapse to the `weathersit = 3` response. | Documented in UI |
| F8 | **Low** | Presentation | Hour-only K-Means labels misrepresent weekends (8:00 Saturday was shown as "Commuter peak"). | Mitigated in `app.py` |
| F9 | **Low** | Metrics | Model minimizes squared error on `log1p(cnt)` but is evaluated after `expm1` on the count scale. | Open (notebook) |
| F10 | **Low** | Reproducibility | Pickle bundle contains no versions, training metadata, feature ranges, or split information. | Open |
| F11 | **Low** | Forecasting | 24-hour totals are sums of independent point estimates with no uncertainty intervals. | Open |
| F12 | **Low** | Frontend | `use_container_width=True` is deprecated in Streamlit 1.67. | Fixed in `app.py` |
| F13 | **Info** | Leakage | `casual` and `registered` are correctly removed before training. | Verified |
| F14 | **Low** | Validation | Metrics come from one random 80/20 holdout with no cross-validation or variance estimate. | Open |

---

## 4. Detailed findings

### F1. Weekday encoding mismatch between training and inference (High, fixed)

**Evidence**

- `hour.csv` uses the UCI convention: `weekday` 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
- The original `app.py` used `dteday.weekday()` from pandas, where 0 = Monday and 6 = Sunday, and fed that value directly into the `weekday` feature.
- Example: 2012-06-15 is a Friday. The old app sent `weekday = 4` (Thursday in UCI encoding) instead of `5` (Friday).
- Measured effect at 2012-06-15, 08:00, clear, 25 C, 55% humidity, 12 km/h wind:
  - `weekday = 4` (old behavior): **737 bikes/hour**
  - `weekday = 5` (correct UCI Friday): **726 bikes/hour**

**Impact**

The error is systematic for every prediction. It is limited in magnitude because the model leans primarily on `hr`, `rush_hour`, `is_weekend`, and `workingday`, and `weekday` importance is only 0.0061. It nevertheless makes the `weekday` input factually wrong and interacts with any date-based analysis.

**Fix applied**

The app now converts with `weekday_uci = (date.weekday() + 1) % 7`. The old comment claiming "0 = Monday, 6 = Sunday" was removed. `is_weekend` is derived from the UCI value `0 or 6`.

**Recommendation**

Add an explicit encoding map or helper function and unit tests asserting known dates (for example, 2012-06-15 is Friday and must map to 5).

---

### F2. Impossible input combinations accepted (High, fixed)

**Evidence from `hour.csv`**

| Combination | Rows in dataset |
| :--- | :---: |
| `holiday = 1` and `workingday = 1` | 0 |
| Saturday or Sunday with `workingday = 1` | 0 |
| Month of June with `season = 1` (spring) | 0 |

The original app exposed `Working Day?`, `Holiday?`, and `Season` as independent controls. It therefore allowed all of the above.

**Observed impact**

| Scenario | Old app | Correct behavior | New app |
| :--- | ---: | ---: | ---: |
| Saturday 2012-06-16, 08:00, working day forced to "Yes" | 668 | Saturday is a non-working day | 198 |
| 2012-06-15, 08:00, holiday = Yes + working day = Yes | 742 | Impossible combination | 249 (holiday, non-working) |
| June with `season = Spring` | 627 | June is summer in this dataset | 726 (derived) |
| June with `season = Winter` | 735 | June is summer in this dataset | 726 (derived) |

The Saturday case alone changes the estimate by a factor of 3.4. The old app could therefore produce a "commuter peak" forecast for a weekend morning.

**Fix applied**

- `season` is derived from month: Jan-Mar = spring, Apr-Jun = summer, Jul-Sep = fall, Oct-Dec = winter (the modal mapping in the dataset).
- `workingday` is derived as `not weekend and not holiday`.
- `is_weekend` is derived from the date.
- Only `holiday` remains user-selectable, which cannot create contradictions.

**Recommendation**

Keep derivation in one shared helper used by both training and serving. If manual overrides are ever required, validate against the training-domain constraints and surface an explicit warning.

---

### F3. Unbounded year extrapolation (Medium, fixed)

**Evidence**

- The original inference code used `yr = max(date.year - 2011, 0)` with no upper bound.
- Training contains only `yr = 0` (2011) and `yr = 1` (2012).
- For years 2013 and later the model returns the same saturated value because tree splits cannot extrapolate:
  - `yr = 1` (2012): 726
  - `yr = 2`, `3`, `6`, `10`, `15`, `19`: all **726**
- Dates before 2011 were clamped to `yr = 0`, silently reusing 2011.

**Impact**

A user selecting today's date (2026) would receive a forecast that is effectively "June 2012" with no warning. The value is bounded, not nonsensical, but it is not a valid 2026 forecast.

**Fix applied**

The date picker is constrained to 2011-01-01 through 2012-12-31 and the sidebar states this explicitly.

**Recommendation**

Treat this model as a fixed-period study. For live forecasting, retrain with recent data and add explicit horizon and drift monitoring.

---

### F4. Random split on time-series data (High, open)

**Evidence**

```python
X_train, X_test, ... = train_test_split(
    X, y_log, y_actual, test_size=0.20, random_state=42
)
```

`shuffle` defaults to `True`, and the rows are hourly observations. Adjacent hours of the same day, and the same hour across nearby days, appear in both train and test.

**Impact**

- Temporal autocorrelation lets the model see near-duplicate contexts during training.
- Reported `R2 = 0.9522`, `RMSE = 38.90`, and `MAE = 23.23` are likely optimistic relative to a genuine future forecast.
- The high apparent accuracy should not be presented as evidence of out-of-period generalization.

**Recommendation**

Use a chronological holdout, for example train on 2011 and test on 2012, or the last 20% of rows by timestamp. Optionally add blocked time-series cross-validation and report mean and standard deviation across folds. Recompute the leaderboard under the new protocol.

---

### F5. Target-derived and redundant `demand_regime` feature (Medium, open)

**Evidence**

```python
hourly_profiles = df.groupby('hr')[['casual', 'registered', 'cnt']].mean()
kmeans.fit_predict(hourly_profiles[['casual', 'registered']])
X_train['demand_regime'] = X_train['hr'].map(...)
X_test['demand_regime'] = X_test['hr'].map(...)
```

- The mapping is learned from the full dataset, including test rows, before the split.
- The resulting feature is a deterministic lookup from `hr` to one of three labels; it contains no information beyond the hour.
- XGBoost importance for `demand_regime` is **0.1574**, second only to `hr` at 0.2147. The hour-derived group (`hr`, `demand_regime`, `hr_sin`, `hr_cos`, `rush_hour`) together accounts for roughly 0.6 of total importance.

**Impact**

Small optimistic bias from using test-period aggregates, plus misleading feature-importance interpretation. The feature compresses the daily cycle rather than adding a new signal.

**Recommendation**

Fit the K-Means mapping on the training partition only. Report ablations with and without `demand_regime`; if the gain is negligible, remove it and simplify the schema.

---

### F6. `temp_diff` hardcoded at inference (Low-Medium, open)

**Evidence**

- Training: `temp_diff = atemp - temp`, with realistic values roughly in `[-0.06, +0.24]`.
- Notebook production function and app: `temp_diff = 0.0` for every request.
- Sensitivity at 08:00, clear, 25 C:

| `temp_diff` | Prediction |
| :---: | ---: |
| -0.06 | 702 |
| 0.00 | 726 |
| +0.05 | 750 |
| +0.10 | 663 |
| +0.20 | 663 |

- Feature importance is 0.0017, so the effect is small, but the training-serving skew is real.

**Recommendation**

Either estimate `atemp` from `temp` with a fitted relationship, or drop `temp_diff` entirely and retrain. Do not feed a constant for a feature that varies during training.

---

### F7. `weathersit = 4` collapses to `weathersit = 3` (Low, documented)

**Evidence**

- Dataset counts: `weathersit = 1`: 11,413, `2`: 4,544, `3`: 1,419, `4`: **3**.
- Predictions at 08:00, 25 C, clear baseline: `weathersit = 3` gives **473**, `weathersit = 4` also gives **473**.
- The three storm rows occur at 01:00 with 23 rentals, 16:00 with 36, and 18:00 with 164.

**Impact**

The "Heavy Rain / Storm" option does not produce a distinct estimate. This is a data limitation, not an implementation error.

**Recommendation**

Merge categories 3 and 4 during training, or label the UI option as "Light rain / snow or storm". The rebuilt app includes a caption explaining the behavior.

---

### F8. Hour-only regime labels misstate weekends (Low, mitigated)

**Evidence**

- `hourly_regime_mapping` assigns a cluster purely from `hr`.
- Hour 8 always maps to cluster 0 (peak), including Saturday and Sunday.
- The original KPI displayed "Commuter Rush Peak" for Saturday 08:00 even though a weekend morning is not a commuter peak.

**Mitigation applied**

- The headline KPI now reports an operational demand level derived from the predicted volume: Low (< 200), Moderate (200-399), High (400-599), Very high (>= 600).
- The hour cluster is shown only as secondary context with neutral names: Peak-hour profile, Night profile, Daytime profile.

**Recommendation**

If regime semantics matter operationally, cluster on `(hr, workingday)` profiles rather than `hr` alone.

---

### F9. Log-scale training versus count-scale evaluation (Low, open)

**Evidence**

- The model minimizes squared error on `log1p(cnt)`.
- Metrics are computed after `np.expm1`, on the original count scale.
- The naive inverse transform is biased low under Jensen's inequality. The measured mean residual of -0.17 bikes suggests the bias is small in practice, but the objective and the reported metric are not aligned.

**Recommendation**

Consider training with a Poisson or Tweedie objective, which models counts directly, or apply a bias correction to the back-transformed mean. Report both log-scale and count-scale metrics.

---

### F10. Bundle lacks metadata (Low, open)

**Evidence**

The pickle contains only:

```python
{
  "model": XGBRegressor,
  "hourly_regime_mapping": dict,
  "feature_columns": list
}
```

There is no record of library versions, training date, data hash, feature ranges, split strategy, target transform, or the metrics associated with the artifact.

**Recommendation**

Add a `metadata` entry with versions, training window, feature ranges, transform description, metrics, and a schema version. Validate incoming feature frames against the stored schema at load time.

---

### F11. Point forecasts presented as daily totals (Low, open)

**Evidence**

- The 24-hour projection sums 24 independent point predictions.
- There are no prediction intervals, and hourly errors are correlated but treated as independent.

**Impact**

The daily total looks more precise than it is. It should be labeled as an expected value.

**Recommendation**

Add residual-based or quantile-regression intervals, and label daily totals as "expected" rather than exact. The rebuilt app already uses planning-guidance language.

---

### F12. Deprecated Streamlit parameter (Low, fixed)

**Evidence**

The original app used `st.altair_chart(..., use_container_width=True)`. Streamlit 1.67 deprecates `use_container_width` in favor of `width="stretch"`.

**Fix applied**

All charts now use `width="stretch"`.

---

### F13. Target leakage correctly avoided (Info)

`casual` and `registered` sum exactly to `cnt` and were removed before model training. This is the correct handling and prevents a fatal leakage problem. Verified in the notebook and in the serialized feature list.

---

### F14. Single holdout, no cross-validation (Low, open)

**Evidence**

One 80/20 split with `random_state=42`. No repeated splits, no cross-validation, no confidence intervals on the reported metrics.

**Recommendation**

Report cross-validated metrics with standard deviations, and use the same folds when comparing model families.

---

## 5. Prediction sanity checks

Scenario unless stated: **2012-06-15 (Friday), 08:00, working day, clear weather, 25 C, 55% humidity, 12 km/h wind.**
All checks were run against the corrected inference logic.

| Scenario | Prediction (bikes/hour) | Reference / assessment |
| :--- | ---: | :--- |
| Baseline working day, 08:00 | 726 | Actual June-2012 working-day clear 08:00 mean: 654.5 (n = 13). Difference about 11%, acceptable for a 13-row slice. |
| Working day, 17:00 | 784 | Actual June-2012 working-day clear 17:00 mean: 784.6. Near-exact match. |
| Working day, 00:00 | 86 | Night baseline. Dataset overall 00:00 mean is 53.9; higher here because of June, 2012, and clear weather. Reasonable. |
| Saturday, 08:00 | 198 | Actual June-2012 weekend clear 08:00 mean: 177.8 (n = 9). Reasonable. |
| Public holiday, 08:00 | 249 | Actual holiday 08:00 mean: 170.5 across all months. Direction is correct: no commute peak. |
| Mist | 705 | Clear 726; small reduction. Consistent with weather importance. |
| Light rain / snow | 473 | Clear 726, a 35% reduction. Direction and magnitude plausible. |
| Heavy rain / storm | 473 | Identical to rain/snow (see F7). |
| Temperature 0 C | 412 | Reduced demand. |
| Temperature 25 C | 726 | Comfortable baseline. |
| Temperature 35 C | 603 | Demand decline at heat extreme. |
| Temperature 41 C | 545 | Further decline. Inverted-U shape is physically plausible. |
| Humidity 0% | 608 | Lower than baseline. |
| Humidity 55% | 726 | Baseline. |
| Humidity 100% | 546 | Reduced demand. |
| Wind 0 km/h | 727 | Baseline. |
| Wind 67 km/h | 586 | About 19% reduction. Plausible. |
| Working-day curve range | 7 to 784 | Non-negative, bounded, bimodal at 08:00 and 17:00-18:00. |
| Weekend curve range | 12 to 676 | Broad midday peak at 12:00-15:00, no commuter spikes. |

**Overall verdict:** predictions are non-negative, bounded within the observed target range (1 to 977), respond monotonically or in plausible inverted-U shapes to weather and comfort variables, and reproduce the expected diurnal pattern. No NaNs, crashes, or implausible magnitudes were observed across the tested grid.

---

## 6. Verified correctness details

| Check | Result |
| :--- | :--- |
| Feature order matches training schema | Pass (22 features, exact order) |
| Normalization: `temp / 41`, `hum / 100`, `windspeed / 67` | Matches dataset scales |
| Cyclical encodings | Correct periods and phase (`mnth - 1`) |
| Target inverse transform | `np.expm1`, clipped at zero |
| `rush_hour` definition | Matches training (working day, hours 7-9 and 16-19) |
| Weather bad-weather flag | Matches training (`weathersit >= 3`) |
| Regime mapping usage | Matches serialized `hourly_regime_mapping` |
| `casual` / `registered` excluded | Confirmed |
| App runs across 10+ input scenarios | No exceptions |

---

## 7. Recommended priority order

1. **F4** - Replace the random split with a chronological split and recompute all reported metrics.
2. **F5** - Fit the K-Means mapping on training data only, then rerun the ablation.
3. **F1/F2** - Keep the corrected calendar derivation and add unit tests around weekday and season mapping.
4. **F6** - Remove or properly derive `temp_diff`, then retrain.
5. **F9** - Align training objective and evaluation, or add a back-transform correction.
6. **F10/F11** - Add artifact metadata and prediction intervals.
7. **F7/F8** - Keep the UI documentation; consider merging weather category 4 and clustering on `(hr, workingday)`.

---

## 8. Reproducing the checks

Key scripts used during the audit:

- Load the bundle and inspect keys: `joblib.load("bike_demand_xgboost_bundle.pkl")`.
- Rebuild a feature row and call `model.predict`.
- Compare slice means with `pandas` filters on `hour.csv`.
- Exercise the app headlessly with `streamlit.testing.v1.AppTest.from_file("app.py")`.

The numerical case studies in this report can be reproduced directly from these steps.
