"""Generate the animated SVG assets used by README.md.

Run from the project root:
    python assets/generate_assets.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent

INK = "#0f172a"
MUTED = "#64748b"
BORDER = "#e2e8f0"
PANEL = "#f8fafc"
BLUE = "#2563eb"
BLUE_SOFT = "#eff6ff"
TEAL = "#0891b2"
PURPLE = "#7c3aed"
AMBER = "#d97706"
GREEN = "#16a34a"
SLATE = "#94a3b8"


def svg(width: int, height: int, title: str, desc: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="svg-title svg-desc">
  <title id="svg-title">{escape(title)}</title>
  <desc id="svg-desc">{escape(desc)}</desc>
  <style>
    text {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
    .mono {{ font-family: "Cascadia Code", Consolas, "SFMono-Regular", Menlo, monospace; }}
    .label {{ font-size: 11px; letter-spacing: 2.2px; fill: {MUTED}; font-weight: 600; }}
    .heading {{ font-size: 21px; fill: {INK}; font-weight: 700; }}
    .note {{ font-size: 10.5px; fill: {MUTED}; }}
    @keyframes reveal {{ from {{ opacity: 0; }} }}
    @keyframes grow {{ from {{ width: 0; }} }}
    @keyframes draw {{ from {{ stroke-dashoffset: 1; }} }}
    .reveal {{ animation: reveal 0.6s ease-out both; }}
    .grow {{ animation: grow 1s cubic-bezier(.2,.8,.2,1) both; }}
    .draw {{ stroke-dasharray: 1; stroke-dashoffset: 0; animation: draw 2.8s ease-out both; }}
  </style>
  <rect width="{width}" height="{height}" rx="16" fill="#ffffff"/>
  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="16" fill="none" stroke="{BORDER}"/>
  {body}
</svg>
"""


def header(label: str, title: str) -> str:
    return (
        f'<text x="40" y="36" class="label">{escape(label)}</text>'
        f'<text x="40" y="66" class="heading">{escape(title)}</text>'
    )


def workflow() -> str:
    stages = [
        ("01", "Data ingestion", "hour.csv", "17,379 rows"),
        ("02", "EDA and cleaning", "leakage checks", "0 nulls"),
        ("03", "Feature engineering", "22 features", "time + weather"),
        ("04", "Model training", "XGBoost", "150 trees"),
        ("05", "Validation", "R2 0.9522", "RMSE 38.90"),
        ("06", "Deployment", "Streamlit", "real-time"),
    ]
    xs = [40 + i * 195 for i in range(len(stages))]
    y, w, h = 112, 150, 100
    cy = y + 43

    body = [header("END-TO-END WORKFLOW", "From raw hourly records to a deployed forecast")]
    body.append(
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#cbd5e1"/></marker></defs>'
    )

    for i, (num, title, line1, line2) in enumerate(stages):
        x = xs[i]
        body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="#dbe3ee"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" stroke="{BLUE}" stroke-width="2" opacity="0">'
            f'<animate attributeName="opacity" values="0;0.38;0" keyTimes="0;0.5;1" dur="3.6s" begin="{i * 0.6}s" repeatCount="indefinite"/>'
            f'</rect>'
            f'<circle cx="{x + 22}" cy="{y + 23}" r="10" fill="{BLUE_SOFT}"/>'
            f'<text x="{x + 22}" y="{y + 27}" text-anchor="middle" font-size="10.5" font-weight="700" fill="{BLUE}">{num}</text>'
            f'<text x="{x + w / 2}" y="{y + 54}" text-anchor="middle" font-size="12.8" font-weight="700" fill="{INK}">{escape(title)}</text>'
            f'<text x="{x + w / 2}" y="{y + 73}" text-anchor="middle" font-size="10.5" fill="{MUTED}">{escape(line1)}</text>'
            f'<text x="{x + w / 2}" y="{y + 89}" text-anchor="middle" font-size="10.5" fill="{MUTED}">{escape(line2)}</text>'
        )

    for i in range(len(stages) - 1):
        x1 = xs[i] + w
        x2 = xs[i + 1]
        body.append(
            f'<path d="M{x1},155 L{x2 - 9},155" fill="none" stroke="#cbd5e1" stroke-width="2" '
            f'stroke-dasharray="6 6" marker-end="url(#arrow)">'
            f'<animate attributeName="stroke-dashoffset" from="0" to="-24" dur="1.2s" repeatCount="indefinite"/>'
            f'</path>'
            f'<circle r="3.2" fill="{BLUE}" opacity="0">'
            f'<animate attributeName="opacity" from="0" to="1" begin="{i * 0.45}s" dur="0.2s" fill="freeze"/>'
            f'<animateMotion dur="1.2s" begin="{i * 0.45}s" repeatCount="indefinite" path="M{x1 + 4},155 L{x2 - 7},155"/>'
            f'</circle>'
        )

    body.append(
        f'<text x="620" y="308" text-anchor="middle" class="note">'
        f'Training window 2011-2012  |  Target log(1 + cnt)  |  Evaluation on the count scale</text>'
    )
    return svg(1240, 340, "End-to-end workflow", "Animated pipeline from raw hourly data to a deployed Streamlit forecast.", "".join(body))


def structure() -> str:
    leaves = [
        ("hour.csv", "raw hourly dataset, 17,379 rows", BLUE),
        ("Bike_Rental_EDA.ipynb", "EDA, feature engineering, benchmarks", PURPLE),
        ("bike_demand_xgboost_bundle.pkl", "trained model, regime map, feature schema", AMBER),
        ("app.py", "interactive Streamlit dashboard", GREEN),
        ("figures/", "EDA and model diagnostic charts", TEAL),
        ("README.md + docs", "summary, project guide, audit findings", MUTED),
    ]
    body = [header("PROJECT STRUCTURE", "Repository map")]
    body.append(
        f'<rect x="24" y="168" width="210" height="88" rx="14" fill="{INK}"/>'
        f'<text x="129" y="204" text-anchor="middle" font-size="13.5" font-weight="700" fill="#ffffff">ML Bike Rental</text>'
        f'<text x="129" y="226" text-anchor="middle" font-size="10.5" fill="#cbd5e1">root repository</text>'
    )

    for i, (name, role, color) in enumerate(leaves):
        ly = 32 + i * 58
        cxx = 330
        body.append(
            f'<path d="M234,212 C272,212 292,{ly + 22} {cxx},{ly + 22}" fill="none" stroke="#cbd5e1" '
            f'stroke-width="2" stroke-dasharray="5 5">'
            f'<animate attributeName="stroke-dashoffset" from="0" to="-20" dur="1.4s" repeatCount="indefinite"/>'
            f'</path>'
            f'<g class="reveal" style="animation-delay:{0.2 * i}s">'
            f'<rect x="{cxx}" y="{ly}" width="460" height="44" rx="10" fill="#ffffff" stroke="#dbe3ee"/>'
            f'<rect x="{cxx}" y="{ly}" width="4" height="44" rx="2" fill="{color}"/>'
            f'<text x="{cxx + 18}" y="{ly + 19}" font-size="12.5" font-weight="700" fill="{INK}">{escape(name)}</text>'
            f'<text x="{cxx + 18}" y="{ly + 34}" font-size="10.5" fill="{MUTED}">{escape(role)}</text>'
            f'</g>'
        )

    body.append(
        f'<rect x="830" y="30" width="340" height="356" rx="14" fill="{PANEL}" stroke="{BORDER}"/>'
        f'<text x="852" y="58" class="label">QUICK START</text>'
        f'<rect x="852" y="72" width="296" height="86" rx="8" fill="{INK}"/>'
        f'<text x="868" y="102" class="mono" font-size="11" fill="#e2e8f0">$ streamlit run app.py</text>'
        f'<rect x="1040" y="93" width="7" height="11" fill="#e2e8f0" opacity="0">'
        f'<animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.5;0.5;1" dur="1.2s" repeatCount="indefinite"/>'
        f'</rect>'
        f'<text x="868" y="126" class="mono" font-size="10" fill="{SLATE}">Local URL  http://localhost:8501</text>'
        f'<line x1="852" y1="184" x2="1148" y2="184" stroke="{BORDER}"/>'
        f'<text x="852" y="210" class="label">MODEL SNAPSHOT</text>'
        f'<text x="852" y="242" font-size="11.5" fill="{MUTED}">Holdout R2</text>'
        f'<text x="1148" y="242" text-anchor="end" font-size="13" font-weight="700" fill="{BLUE}">0.9522</text>'
        f'<text x="852" y="276" font-size="11.5" fill="{MUTED}">RMSE</text>'
        f'<text x="1148" y="276" text-anchor="end" font-size="13" font-weight="700" fill="{INK}">38.90 bikes/hr</text>'
        f'<text x="852" y="310" font-size="11.5" fill="{MUTED}">MAE</text>'
        f'<text x="1148" y="310" text-anchor="end" font-size="13" font-weight="700" fill="{INK}">23.23 bikes/hr</text>'
        f'<text x="852" y="350" class="note">Training window: January 2011 to December 2012</text>'
    )
    return svg(1200, 420, "Project structure", "Animated repository map with a quick-start terminal and model snapshot.", "".join(body))


def leaderboard() -> str:
    rows = [
        ("XGBoost", 38.90, BLUE),
        ("Random Forest", 41.01, "#64748b"),
        ("Hist Gradient Boosting", 41.18, SLATE),
        ("Ridge Regression", 127.20, "#cbd5e1"),
        ("Dummy Regressor", 198.83, "#e2e8f0"),
    ]
    body = [header("HOLDOUT PERFORMANCE", "RMSE by model - lower is better")]
    left, top, width_max, bar_h = 200, 96, 540, 24

    for value in (0, 50, 100, 150, 200):
        x = left + value / 200 * width_max
        body.append(f'<line x1="{x}" y1="84" x2="{x}" y2="356" stroke="#eef2f7"/>')
        body.append(f'<text x="{x}" y="382" text-anchor="middle" font-size="10" fill="{MUTED}">{value}</text>')

    for i, (name, value, color) in enumerate(rows):
        y = top + i * 56
        bar_w = value / 200 * width_max
        body.append(
            f'<text x="{left - 12}" y="{y + 17}" text-anchor="end" font-size="12" fill="{INK}">{escape(name)}</text>'
            f'<rect class="grow" style="animation-delay:{0.15 * i}s" x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{color}" opacity="0.95"/>'
            f'<text class="reveal" style="animation-delay:{0.15 * i + 0.7}s" x="{left + bar_w + 9}" y="{y + 17}" font-size="11.5" font-weight="700" fill="{INK}">{value:.2f}</text>'
        )

    body.append(
        f'<text x="450" y="412" text-anchor="middle" class="note">'
        f'Random 80/20 holdout. See FLAGS_AND_LIMITATIONS.md for the temporal-leakage caveat.</text>'
    )
    return svg(900, 430, "Model leaderboard", "Animated bar chart of holdout RMSE for five model families.", "".join(body))


def demand_curve() -> str:
    working = [86, 34, 18, 8, 7, 38, 162, 454, 726, 370, 243, 277, 353, 345, 331, 337,
               503, 784, 752, 514, 401, 302, 254, 188]
    weekend = [160, 120, 80, 34, 12, 15, 41, 92, 198, 337, 463, 580, 676, 660, 645, 631,
               632, 564, 513, 416, 354, 298, 243, 190]
    left, right, top, bottom = 72, 930, 84, 338
    ymax = 820.0

    def x(hour: int) -> float:
        return left + hour * (right - left) / 23

    def y(value: float) -> float:
        return bottom - value / ymax * (bottom - top)

    body = [header("24-HOUR DEMAND PROFILE", "Model estimate for a clear June day at 25 C")]

    for value in (0, 200, 400, 600, 800):
        yy = y(value)
        body.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#eef2f7"/>')
        body.append(f'<text x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{value}</text>')

    for hour in (0, 4, 8, 12, 16, 20, 23):
        body.append(f'<text x="{x(hour):.1f}" y="360" text-anchor="middle" font-size="10" fill="{MUTED}">{hour:02d}</text>')

    body.append(f'<text x="501" y="392" text-anchor="middle" font-size="11" fill="{MUTED}">Hour of day</text>')
    body.append(
        f'<text x="20" y="211" text-anchor="middle" font-size="11" fill="{MUTED}" '
        f'transform="rotate(-90 20 211)">Predicted rentals per hour</text>'
    )

    def path_for(values):
        return "M " + " L ".join(f"{x(h):.1f},{y(v):.1f}" for h, v in enumerate(values))

    def area_for(values):
        return path_for(values) + f" L {x(23):.1f},{y(0):.1f} L {x(0):.1f},{y(0):.1f} Z"

    body.append(
        f'<path class="reveal" style="animation-delay:2.2s" d="{area_for(working)}" fill="{BLUE}" opacity="0.08"/>'
        f'<path class="draw" d="{path_for(working)}" fill="none" stroke="{BLUE}" stroke-width="2.6" stroke-linecap="round" pathLength="1"/>'
        f'<path class="reveal" style="animation-delay:0.5s" d="{path_for(weekend)}" fill="none" stroke="{PURPLE}" stroke-width="2.2" stroke-linecap="round" stroke-dasharray="7 5"/>'
    )

    peaks = [(8, working[8], "08:00  726", 12, -14, BLUE),
             (17, working[17], "17:00  784", -10, -16, BLUE),
             (12, weekend[12], "12:00  676", 8, -14, PURPLE)]
    for hour, value, label, dx, dy, color in peaks:
        body.append(
            f'<g class="reveal" style="animation-delay:2.7s">'
            f'<circle cx="{x(hour):.1f}" cy="{y(value):.1f}" r="4.5" fill="#ffffff" stroke="{color}" stroke-width="2.5"/>'
            f'<text x="{x(hour) + dx:.1f}" y="{y(value) + dy:.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="{color}">{label}</text>'
            f'</g>'
        )

    body.append(
        f'<g><line x1="688" y1="66" x2="714" y2="66" stroke="{BLUE}" stroke-width="2.6"/>'
        f'<text x="720" y="70" font-size="10.5" fill="{MUTED}">Working day</text>'
        f'<line x1="812" y1="66" x2="838" y2="66" stroke="{PURPLE}" stroke-width="2.2" stroke-dasharray="7 5"/>'
        f'<text x="844" y="70" font-size="10.5" fill="{MUTED}">Weekend</text></g>'
    )
    body.append(
        f'<text x="501" y="414" text-anchor="middle" class="note">'
        f'Clear weather, 25 C, 55% humidity, 12 km/h wind. Values are point estimates.</text>'
    )
    return svg(1000, 430, "Hourly demand profile", "Animated line chart comparing working-day and weekend demand curves.", "".join(body))


def feature_importance() -> str:
    rows = [
        ("hr", 0.2147, BLUE),
        ("demand_regime", 0.1574, BLUE),
        ("hr_sin", 0.1019, BLUE),
        ("hr_cos", 0.0970, BLUE),
        ("rush_hour", 0.0928, TEAL),
        ("is_weekend", 0.0701, TEAL),
        ("workingday", 0.0578, TEAL),
        ("yr", 0.0428, PURPLE),
        ("temp", 0.0416, AMBER),
        ("season", 0.0375, AMBER),
    ]
    body = [header("FEATURE IMPORTANCE", "Top 10 XGBoost features by gain")]
    left, top, max_w, bar_h = 210, 104, 520, 18
    vmax = 0.22

    for i, (name, value, color) in enumerate(rows):
        y = top + i * 33
        bar_w = value / vmax * max_w
        body.append(
            f'<text x="{left - 14}" y="{y + 13}" text-anchor="end" font-size="11.5" fill="{INK}">{escape(name)}</text>'
            f'<rect class="grow" style="animation-delay:{0.08 * i}s" x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="0.92"/>'
            f'<text class="reveal" style="animation-delay:{0.08 * i + 0.6}s" x="{left + bar_w + 9}" y="{y + 13}" font-size="10.5" font-weight="700" fill="{MUTED}">{value:.3f}</text>'
        )

    legend = [(BLUE, "Time"), (TEAL, "Day type"), (PURPLE, "Year"), (AMBER, "Weather / season")]
    lx = 548
    for color, label in legend:
        body.append(f'<rect x="{lx}" y="58" width="9" height="9" rx="2" fill="{color}"/>')
        body.append(f'<text x="{lx + 14}" y="67" font-size="10" fill="{MUTED}">{escape(label)}</text>')
        lx += 78

    body.append(
        f'<text x="450" y="440" text-anchor="middle" class="note">'
        f'Time-of-day features account for roughly two-thirds of total gain.</text>'
    )
    return svg(900, 460, "Feature importance", "Animated horizontal bar chart of the top ten XGBoost features.", "".join(body))


ASSETS = {
    "workflow.svg": workflow,
    "structure.svg": structure,
    "model_leaderboard.svg": leaderboard,
    "demand_curve.svg": demand_curve,
    "feature_importance.svg": feature_importance,
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in ASSETS.items():
        path = OUT / name
        path.write_text(builder(), encoding="utf-8")
        print(f"wrote {path}")
