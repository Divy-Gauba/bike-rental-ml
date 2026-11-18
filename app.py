import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

st.set_page_config(
    page_title="Bike Rental Demand Forecaster",
    page_icon="🚲",
    layout="wide"
)

# 1. Load Model Bundle
@st.cache_resource
def load_bundle():
    return joblib.load("bike_demand_xgboost_bundle.pkl")

bundle = load_bundle()
model = bundle["model"]
regime_mapping = bundle["hourly_regime_mapping"]
feature_columns = bundle["feature_columns"]

# App Header
st.title("🚲 Capital Bikeshare - Hourly Demand Predictor")
st.markdown(
    "Predict hourly rental demand using our champion **XGBoost Regressor** ($R^2 = 0.9522$) "
    "enriched with **K-Means Operational Demand Regimes**."
)
st.divider()

# Sidebar: Operational Controls
st.sidebar.header("⚙️ Operational Parameters")

selected_date = st.sidebar.date_input("Date", pd.to_datetime("2012-06-15"))
selected_hour = st.sidebar.slider("Hour of the Day", min_value=0, max_value=23, value=8, step=1)

col_day1, col_day2 = st.sidebar.columns(2)
with col_day1:
    is_workingday = st.selectbox("Working Day?", options=["Yes", "No"], index=0)
with col_day2:
    is_holiday = st.selectbox("Holiday?", options=["No", "Yes"], index=0)

season_options = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
selected_season = st.sidebar.selectbox("Season", options=list(season_options.keys()), index=1)

weather_options = {
    "Clear / Few Clouds": 1,
    "Mist / Cloudy": 2,
    "Light Rain / Snow": 3,
    "Heavy Rain / Storm": 4
}
selected_weather = st.sidebar.selectbox("Weather Condition", options=list(weather_options.keys()), index=0)

st.sidebar.subheader("🌡️ Environmental Conditions")
temp_celsius = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=41.0, value=25.0, step=0.5)
humidity_pct = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=55, step=1)
windspeed_kmh = st.sidebar.slider("Wind Speed (km/h)", min_value=0, max_value=67, value=12, step=1)

# Normalization mapping to match model training scales
temp_norm = np.clip(temp_celsius / 41.0, 0.0, 1.0)
hum_norm = np.clip(humidity_pct / 100.0, 0.0, 1.0)
wind_norm = np.clip(windspeed_kmh / 67.0, 0.0, 1.0)
workingday_val = 1 if is_workingday == "Yes" else 0
holiday_val = 1 if is_holiday == "Yes" else 0
season_val = season_options[selected_season]
weathersit_val = weather_options[selected_weather]

# Derived Features Function
def build_feature_row(hr_val):
    dteday = pd.to_datetime(selected_date)
    day_val = dteday.day
    yr_val = max(dteday.year - 2011, 0)
    mnth_val = dteday.month
    weekday_val = dteday.weekday()  # 0=Monday, 6=Sunday
    is_weekend_val = 1 if weekday_val in [5, 6] else 0
    rush_hour_val = 1 if (workingday_val == 1 and hr_val in [7, 8, 9, 16, 17, 18, 19]) else 0
    
    hr_sin_val = np.sin(2 * np.pi * hr_val / 24)
    hr_cos_val = np.cos(2 * np.pi * hr_val / 24)
    mnth_sin_val = np.sin(2 * np.pi * (mnth_val - 1) / 12)
    mnth_cos_val = np.cos(2 * np.pi * (mnth_val - 1) / 12)
    
    temp_diff_val = 0.0
    temp_hum_val = temp_norm * hum_norm
    bad_weather_val = 1 if weathersit_val >= 3 else 0
    demand_regime_val = regime_mapping.get(hr_val, 1)
    
    return {
        'season': season_val, 'yr': yr_val, 'mnth': mnth_val, 'hr': hr_val,
        'holiday': holiday_val, 'weekday': weekday_val, 'workingday': workingday_val,
        'weathersit': weathersit_val, 'temp': temp_norm, 'hum': hum_norm,
        'windspeed': wind_norm, 'day': day_val, 'is_weekend': is_weekend_val,
        'rush_hour': rush_hour_val, 'hr_sin': hr_sin_val, 'hr_cos': hr_cos_val,
        'mnth_sin': mnth_sin_val, 'mnth_cos': mnth_cos_val,
        'temp_diff': temp_diff_val, 'temp_hum_interaction': temp_hum_val,
        'bad_weather': bad_weather_val, 'demand_regime': demand_regime_val
    }

# Predict Current Hour
current_row = pd.DataFrame([build_feature_row(selected_hour)])[feature_columns]
pred_log = model.predict(current_row)[0]
predicted_demand = int(np.round(np.clip(np.expm1(pred_log), 0, None)))

# Demand Regime Interpretation
regime_id = regime_mapping.get(selected_hour, 1)
regime_meta = {
    0: ("Commute Rush Peak", "🔴 High Priority", "Deploy station rebalancing teams to replenish empty docks."),
    1: ("Off-Peak / Night", "🟢 Low Demand", "Optimal window for overnight bike repairs and battery swaps."),
    2: ("Daytime Leisure", "🟡 Moderate Demand", "Steady turnover. Maintain standard buffer fleets.")
}
regime_title, alert_badge, recommendation = regime_meta.get(regime_id, ("Standard", "⚪ Standard", "Normal operation."))

# Display Top KPI Cards
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric(label="Predicted Rental Demand", value=f"{predicted_demand:,} bikes/hr")
with kpi2:
    st.metric(label="Operational Demand Regime", value=regime_title, delta=alert_badge)
with kpi3:
    feel_temp = "Warm" if temp_celsius > 22 else ("Mild" if temp_celsius > 12 else "Cold")
    st.metric(label="Weather Context", value=f"{temp_celsius}°C ({feel_temp})", delta=selected_weather)

st.info(f"**Operational Advisory:** {recommendation}")

st.divider()

# 24-Hour Diurnal Demand Curve
st.subheader("📈 24-Hour Diurnal Demand Simulation for Selected Day")
st.markdown("Visualizing hourly demand under the specified weather and working day conditions:")

all_24_records = [build_feature_row(h) for h in range(24)]
df_24 = pd.DataFrame(all_24_records)[feature_columns]
preds_24_log = model.predict(df_24)
preds_24 = np.clip(np.expm1(preds_24_log), 0, None)

chart_df = pd.DataFrame({
    'Hour': range(24),
    'Predicted Demand': np.round(preds_24, 0),
    'Selected': ['Current Hour' if h == selected_hour else 'Other' for h in range(24)]
})

# Altair Line + Point Chart
line_chart = alt.Chart(chart_df).mark_line(color="#1f77b4", size=3).encode(
    x=alt.X('Hour:Q', scale=alt.Scale(domain=[0, 23]), title='Hour of the Day (0–23)'),
    y=alt.Y('Predicted Demand:Q', title='Predicted Bikes Rented')
)

points = alt.Chart(chart_df).mark_circle(size=80).encode(
    x='Hour:Q',
    y='Predicted Demand:Q',
    color=alt.Color('Selected:N', scale=alt.Scale(domain=['Current Hour', 'Other'], range=['#d62728', '#1f77b4']), legend=None),
    tooltip=['Hour', 'Predicted Demand']
)

st.altair_chart(line_chart + points, use_container_width=True)

# Daily Total Projection
daily_total = int(np.sum(preds_24))
peak_hour = int(np.argmax(preds_24))
peak_vol = int(np.max(preds_24))

summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.caption(f"**Projected Daily Total:** ~{daily_total:,} total rentals")
with summary_col2:
    st.caption(f"**Expected Peak Hour:** {peak_hour}:00 with ~{peak_vol:,} rentals")
