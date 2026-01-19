
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper to load a CSV (date,value) -> pandas Series (datetime index, float)
def load_series(csv_path: Path, weekly_resample: bool = True):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    s = df.set_index("date")["value"]
    if weekly_resample:
        s = s.resample("W-FRI").last().interpolate()
    return s

def zscore(s: pd.Series):
    if len(s) < 10:
        return s * np.nan
    return (s - s.rolling(260, min_periods=26).mean()) / s.rolling(260, min_periods=26).std()

# Load our three FRED series (created in Step 6)
series_map = {
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)": "unemployment_rate.csv",
    "US HY OAS (bps)": "hy_credit_spread.csv",
}

rows = []
for label, fname in series_map.items():
    path = DATA_DIR / fname
    if not path.exists():
        rows.append((label, "-", "-", "-"))
        continue
    s = load_series(path, weekly_resample=True)
    zs = zscore(s)

    last_date = s.index[-1].strftime("%Y-%m-%d")
    last_val = s.iloc[-1]
    last_z = zs.iloc[-1] if not np.isnan(zs.iloc[-1]) else np.nan

    rows.append((label, last_date, f"{last_val:.2f}", "-" if np.isnan(last_z) else f"{last_z:.2f}"))

# Compute a toy composite risk score (just average of available z-scores)
z_values = []
for label, fname in series_map.items():
    path = DATA_DIR / fname
    if not path.exists():
        continue
    s = load_series(path, weekly_resample=True)
    zs = zscore(s)
    if not np.isnan(zs.iloc[-1]):
        # Direction: inverted curve is risk => negative_is_risky
        if "Yield Curve" in label:
            z_values.append(-zs.iloc[-1])
        # HY OAS higher = risk
        elif "HY OAS" in label:
            z_values.append(zs.iloc[-1])
        # Unemployment higher = risk
        elif "Unemployment" in label:
            z_values.append(zs.iloc[-1])

composite = np.nan
if z_values:
    composite = float(np.mean(z_values))

level = "OK"
if not np.isnan(composite):
    if composite >= 2.0:
        level = "ALERT"
    elif composite >= 1.0:
        level = "WATCH"

# Render a very simple HTML
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
html = []
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Global Crisis Early Warning – Prototype</title>")
html.append("""
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:24px;}
table{border-collapse:collapse;margin-top:12px;}
th,td{border:1px solid #ddd;padding:6px 10px;}
th{background:#f4f6f8;}
.badge{display:inline-block;padding:6px 10px;border-radius:6px;margin-left:8px;}
.ok{background:#e8f5e9;} .watch{background:#fff8e1;} .alert{background:#ffebee;}
.small{color:#666;font-size:12px}
</style>
""")
html.append("</head><body>")
html.append("<h1>Global Crisis Early Warning – Prototype</h1>")
html.append(f"<p class='small'>Generated: {now}</p>")

badge = "<span class='badge ok'>OK</span>"
if level == "WATCH": badge = "<span class='badge watch'>WATCH</span>"
if level == "ALERT": badge = "<span class='badge alert'>ALERT</span>"

comp_str = "-" if np.isnan(composite) else f"{composite:.2f}"
html.append(f"<h2>Composite Risk Score: {comp_str} {badge}</h2>")

html.append("<h3>Key US Indicators</h3>")
html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z-Score</th></tr>")
for r in rows:
    html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>")
html.append("</table>")

html.append("<p class='small'>Note: Simple prototype using FRED data only. Z-scores use a rolling window and are for research use.</p>")
html.append("</body></html>")

OUT_DIR.joinpath("index.html").write_text("\n".join(html), encoding="utf-8")
print("✅ Wrote output/index.html")
