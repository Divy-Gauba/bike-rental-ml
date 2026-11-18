from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# ----------------------------------------------------------------------------
# Page configuration and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bike Rental Demand Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 1280px;}
        [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0;
                                  border-radius: 10px; padding: 14px 16px;}
        [data-testid="stMetricLabel"] {color: #475569;}
        h1, h2, h3 {color: #0f172a;}
    </style>
    """,
    unsafe_allow_html=True,
)

BUNDLE_PATH = Path(__file__).with_name("bike_demand_xgboost_bundle.pkl")

# ----------------------------------------------------------------------------
# Model bundle
# ----------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    bundle = joblib.load(BUNDLE_PATH)
    return bundle["model"], bundle["hourly_regime_mapping"], bundle["feature_columns"]


model, regime_mapping, feature_columns = load_bundle()

# Dataset-level reference facts (UCI Capital Bikeshare, hourly, 2011-2012)
SEASON_BY_MONTH = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
SEASON_NAME = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_OPTIONS = {
    "Clear / few clouds": 1,
    "Mist / cloudy": 2,
    "Light rain / snow": 3,
    "Heavy rain / storm": 4,
}
def demand_level(volume: float) -> tuple[str, str]:
    if volume >= 600:
        return "Very high", "High demand expected. Rebalance docks ahead of the hour and hold a reserve fleet."
    if volume >= 400:
        return "High", "Elevated demand. Monitor dock capacity and keep standard staffing in place."
    if volume >= 200:
        return "Moderate", "Moderate demand. Standard staffing and dock buffers should be sufficient."
    return "Low", "Low demand. A good window for maintenance, repairs and bike redistribution."


# K-Means hour clusters stored in the model bundle (0 = peak, 1 = night, 2 = daytime)
REGIME_NAME = {0: "Peak-hour profile", 1: "Night profile", 2: "Daytime profile"}

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.title("Scenario controls")
st.sidebar.caption("Inputs are constrained to the model's 2011-2012 training window.")

selected_date = st.sidebar.date_input(
    "Date",
    value=pd.Timestamp("2012-06-15").date(),
    min_value=pd.Timestamp("2011-01-01").date(),
    max_value=pd.Timestamp("2012-12-31").date(),
)
selected_hour = st.sidebar.slider("Hour of day", min_value=0, max_value=23, value=8, step=1)
is_holiday = st.sidebar.checkbox("Public holiday", value=False)

st.sidebar.subheader("Weather")
selected_weather_label = st.sidebar.selectbox(
    "Condition",
    options=list(WEATHER_OPTIONS.keys()),
    index=0,
    help="Heavy rain/storm is very sparsely represented in the training data; its estimate is close to light rain/snow.",
)
weather_val = WEATHER_OPTIONS[selected_weather_label]

temperature_c = st.sidebar.slider(
    "Temperature (C)", min_value=0.0, max_value=41.0, value=25.0, step=0.5
)
humidity_pct = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=55, step=1)
windspeed_kmh = st.sidebar.slider("Wind speed (km/h)", min_value=0, max_value=67, value=12, step=1)

# ----------------------------------------------------------------------------
# Derived calendar context
# ----------------------------------------------------------------------------
date = pd.Timestamp(selected_date)
month = int(date.month)
day_of_month = int(date.day)
year_index = int(date.year - 2011)          # 2011 -> 0, 2012 -> 1
season = SEASON_BY_MONTH[month]
# Training data uses UCI weekday encoding: 0 = Sunday ... 6 = Saturday.
weekday_uci = int((date.weekday() + 1) % 7)
is_weekend = int(weekday_uci in (0, 6))
workingday = int(not is_weekend and not is_holiday)
weekday_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][weekday_uci]

st.sidebar.divider()
st.sidebar.caption(
    f"Derived context: {weekday_name}, month {month:02d}, {SEASON_NAME[season]}, "
    f"{'working day' if workingday else 'non-working day'}."
)
st.sidebar.caption(
    "Model: XGBoost regressor on log(1 + count). "
    "Holdout metrics: R2 0.9522, RMSE 38.90, MAE 23.23 bikes/hour."
)


def build_feature_row(
    hr: int,
    *,
    workingday_override: int | None = None,
    is_weekend_override: int | None = None,
    weekday_override: int | None = None,
    holiday_override: int | None = None,
    weather_override: int | None = None,
    temp_c: float | None = None,
    humidity_override: int | None = None,
    wind_override: int | None = None,
) -> dict:
    """Build a single model input row from the current scenario."""
    wd = workingday if workingday_override is None else int(workingday_override)
    weekend = is_weekend if is_weekend_override is None else int(is_weekend_override)
    weekday = weekday_uci if weekday_override is None else int(weekday_override)
    holiday = int(is_holiday) if holiday_override is None else int(holiday_override)
    weather = weather_val if weather_override is None else int(weather_override)
    temp = temperature_c if temp_c is None else float(temp_c)
    hum = humidity_pct if humidity_override is None else int(humidity_override)
    wind = windspeed_kmh if wind_override is None else int(wind_override)

    temp_norm = float(np.clip(temp / 41.0, 0.0, 1.0))
    hum_norm = float(np.clip(hum / 100.0, 0.0, 1.0))
    wind_norm = float(np.clip(wind / 67.0, 0.0, 1.0))
    rush_hour = int(wd == 1 and hr in (7, 8, 9, 16, 17, 18, 19))

    return {
        "season": season,
        "yr": year_index,
        "mnth": month,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": wd,
        "weathersit": weather,
        "temp": temp_norm,
        "hum": hum_norm,
        "windspeed": wind_norm,
        "day": day_of_month,
        "is_weekend": weekend,
        "rush_hour": rush_hour,
        "hr_sin": float(np.sin(2 * np.pi * hr / 24)),
        "hr_cos": float(np.cos(2 * np.pi * hr / 24)),
        "mnth_sin": float(np.sin(2 * np.pi * (month - 1) / 12)),
        "mnth_cos": float(np.cos(2 * np.pi * (month - 1) / 12)),
        "temp_diff": 0.0,
        "temp_hum_interaction": temp_norm * hum_norm,
        "bad_weather": int(weather >= 3),
        "demand_regime": int(regime_mapping.get(hr, 1)),
    }


def predict_counts(rows: pd.DataFrame) -> np.ndarray:
    log_pred = model.predict(rows[feature_columns])
    return np.clip(np.expm1(log_pred), 0, None)


def predict_single(row: dict) -> int:
    return int(np.round(predict_counts(pd.DataFrame([row]))[0]))


def feature_contributions(row: dict) -> pd.Series:
    """XGBoost feature contributions for one row, in log-demand space."""
    matrix = xgb.DMatrix(pd.DataFrame([row])[feature_columns], feature_names=feature_columns)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)[0]
    values = pd.Series(contributions[:-1], index=feature_columns)
    return values.reindex(values.abs().sort_values(ascending=False).index)


# ----------------------------------------------------------------------------
# Current prediction and derived KPIs
# ----------------------------------------------------------------------------
current_row = build_feature_row(selected_hour)
predicted_demand = predict_single(current_row)

all_hours = pd.DataFrame([build_feature_row(h) for h in range(24)])
hourly_predictions = predict_counts(all_hours)
daily_total = int(np.sum(hourly_predictions))
peak_hour = int(np.argmax(hourly_predictions))
peak_volume = int(np.max(hourly_predictions))
daily_mean = float(np.mean(hourly_predictions))

regime_id = int(regime_mapping.get(selected_hour, 1))
regime_name = REGIME_NAME.get(regime_id, "Standard")
level_name, advisory = demand_level(predicted_demand)

# ----------------------------------------------------------------------------
# Header and KPI cards
# ----------------------------------------------------------------------------
st.title("Bike Rental Demand Forecaster")
st.markdown(
    "Hourly demand estimates for the Capital Bikeshare system, powered by an XGBoost "
    "regressor with calendar, weather and cyclical time features."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        "Predicted demand",
        f"{predicted_demand:,} bikes/hr",
        delta=f"{predicted_demand - daily_mean:+.0f} vs daily average",
        delta_color="off",
    )
with kpi2:
    st.metric("Projected daily total", f"{daily_total:,} rentals")
with kpi3:
    st.metric("Expected peak", f"{peak_hour:02d}:00", delta=f"{peak_volume:,} bikes/hr", delta_color="off")
with kpi4:
    st.metric(
        "Operational demand level",
        level_name,
        delta=f"Hour cluster: {regime_name}",
        delta_color="off",
    )

st.info(f"Operational note: {advisory}")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
forecast_tab, drivers_tab, sensitivity_tab = st.tabs(
    ["24-hour forecast", "Prediction drivers", "Sensitivity"]
)

# --- Tab 1: 24-hour forecast ------------------------------------------------
with forecast_tab:
    compare_weekend = st.checkbox(
        "Compare with the corresponding weekend profile",
        value=False,
        help="Holds the selected weather and month fixed and switches the day type.",
    )

    curve_rows = []
    for h in range(24):
        curve_rows.append({"Hour": h, "Demand": hourly_predictions[h], "Scenario": "Selected day"})
    if compare_weekend:
        if workingday == 1:
            ref = {
                "workingday_override": 0,
                "is_weekend_override": 1,
                "weekday_override": 6,
                "holiday_override": 0,
            }
            reference_label = "Weekend reference"
        else:
            ref = {
                "workingday_override": 1,
                "is_weekend_override": 0,
                "weekday_override": 3,
                "holiday_override": 0,
            }
            reference_label = "Working-day reference"
        reference_rows = pd.DataFrame([build_feature_row(h, **ref) for h in range(24)])
        reference_predictions = predict_counts(reference_rows)
        for h in range(24):
            curve_rows.append({"Hour": h, "Demand": reference_predictions[h], "Scenario": reference_label})

    curve_df = pd.DataFrame(curve_rows)
    scenario_order = ["Selected day"] + ([reference_label] if compare_weekend else [])

    line = (
        alt.Chart(curve_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "Hour:Q",
                scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(values=list(range(0, 24, 2)), title="Hour of day"),
            ),
            y=alt.Y("Demand:Q", title="Predicted rentals (bikes/hour)"),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(domain=scenario_order, range=["#2563eb", "#94a3b8"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Hour:Q", title="Hour"),
                alt.Tooltip("Demand:Q", title="Predicted demand", format=",.0f"),
                alt.Tooltip("Scenario:N", title="Scenario"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"Hour": [selected_hour]}))
        .mark_rule(color="#cbd5e1", strokeDash=[4, 4])
        .encode(x="Hour:Q")
    )
    selected_point = (
        alt.Chart(pd.DataFrame({"Hour": [selected_hour], "Demand": [predicted_demand]}))
        .mark_circle(size=130, color="#2563eb", filled=True)
        .encode(
            x="Hour:Q",
            y="Demand:Q",
            tooltip=[
                alt.Tooltip("Hour:Q", title="Selected hour"),
                alt.Tooltip("Demand:Q", title="Demand", format=",.0f"),
            ],
        )
    )
    chart = (
        (line + rule + selected_point)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#eef2f7", domainColor="#cbd5e1", labelColor="#475569", titleColor="#334155")
        .configure_legend(labelColor="#475569", titleColor="#334155")
        .properties(height=360)
    )
    st.altair_chart(chart, width="stretch")

    table = pd.DataFrame(
        {
            "Hour": [f"{h:02d}:00" for h in range(24)],
            "Predicted demand": np.round(hourly_predictions, 0).astype(int),
            "Share of daily total": np.round(100 * hourly_predictions / hourly_predictions.sum(), 1),
        }
    )
    st.download_button(
        "Download forecast (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="bike_demand_forecast.csv",
        mime="text/csv",
    )
    st.dataframe(table, hide_index=True, height=290)

# --- Tab 2: Prediction drivers ---------------------------------------------
with drivers_tab:
    st.markdown(
        "How each input shifts the prediction for the selected hour. Contributions are "
        "expressed in log-demand units; positive values increase expected demand, negative values reduce it."
    )
    contributions = feature_contributions(current_row)
    top = contributions.head(10).rename("Contribution").reset_index().rename(columns={"index": "Feature"})
    top["Direction"] = np.where(top["Contribution"] >= 0, "Increases demand", "Reduces demand")

    contribution_chart = (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Feature:N", sort="-x", title=None),
            x=alt.X("Contribution:Q", title="Contribution to log demand"),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Increases demand", "Reduces demand"], range=["#2563eb", "#dc2626"]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Feature:N", title="Feature"),
                alt.Tooltip("Contribution:Q", title="Contribution", format="+.3f"),
            ],
        )
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#eef2f7", domainColor="#cbd5e1", labelColor="#475569", titleColor="#334155")
        .properties(height=360)
    )
    st.altair_chart(contribution_chart, width="stretch")
    st.caption(
        "Demand regime, hour encodings and rush-hour indicators dominate the model because they "
        "describe the daily travel cycle. Weather and comfort variables act as secondary adjustments."
    )

# --- Tab 3: Sensitivity -----------------------------------------------------
with sensitivity_tab:
    st.markdown(
        "Partial-dependence style checks for the selected hour. Each curve changes one input "
        "while holding the rest of the scenario fixed."
    )
    temp_left, weather_right = st.columns(2)

    with temp_left:
        st.subheader("Temperature response")
        temp_grid = np.arange(0, 41.5, 1.0)
        temp_rows = pd.DataFrame([build_feature_row(selected_hour, temp_c=t) for t in temp_grid])
        temp_pred = predict_counts(temp_rows)
        temp_df = pd.DataFrame({"Temperature": temp_grid, "Demand": temp_pred})
        temp_line = (
            alt.Chart(temp_df)
            .mark_line(color="#2563eb", strokeWidth=2.5)
            .encode(
                x=alt.X("Temperature:Q", title="Temperature (C)"),
                y=alt.Y("Demand:Q", title="Predicted rentals"),
                tooltip=[
                    alt.Tooltip("Temperature:Q", format=".0f"),
                    alt.Tooltip("Demand:Q", format=",.0f"),
                ],
            )
        )
        temp_rule = (
            alt.Chart(pd.DataFrame({"Temperature": [temperature_c]}))
            .mark_rule(color="#cbd5e1", strokeDash=[4, 4])
            .encode(x="Temperature:Q")
        )
        temp_point = (
            alt.Chart(pd.DataFrame({"Temperature": [temperature_c], "Demand": [predicted_demand]}))
            .mark_circle(size=110, color="#2563eb", filled=True)
            .encode(x="Temperature:Q", y="Demand:Q")
        )
        st.altair_chart(
            (temp_line + temp_rule + temp_point)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#eef2f7", domainColor="#cbd5e1", labelColor="#475569")
            .properties(height=300),
            width="stretch",
        )

    with weather_right:
        st.subheader("Weather response")
        weather_rows = pd.DataFrame(
            [build_feature_row(selected_hour, weather_override=w) for w in (1, 2, 3, 4)]
        )
        weather_pred = predict_counts(weather_rows)
        weather_df = pd.DataFrame(
            {
                "Condition": ["Clear", "Mist", "Rain / snow", "Storm"],
                "Demand": weather_pred,
                "Selected": [w == weather_val for w in (1, 2, 3, 4)],
            }
        )
        weather_chart = (
            alt.Chart(weather_df)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("Condition:N", sort=None, title=None),
                y=alt.Y("Demand:Q", title="Predicted rentals"),
                color=alt.Color(
                    "Selected:N",
                    scale=alt.Scale(domain=[True, False], range=["#2563eb", "#cbd5e1"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Condition:N"),
                    alt.Tooltip("Demand:Q", format=",.0f"),
                ],
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#eef2f7", domainColor="#cbd5e1", labelColor="#475569")
            .properties(height=300)
        )
        st.altair_chart(weather_chart, width="stretch")
        st.caption(
            "Storm conditions reuse the light rain/snow response because only three storm hours "
            "exist in the training data."
        )

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.caption(
    "Data: UCI Capital Bikeshare hourly dataset (17,379 observations, 2011-2012). "
    "Model artifact: bike_demand_xgboost_bundle.pkl. "
    "Estimates are conditional averages and should be treated as planning guidance, not guarantees."
)
