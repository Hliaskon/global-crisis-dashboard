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
        # Weekly frequency smooths daily/monthly series into a comparable cadence
        s = s.resample("W-FRI").last().interpolate()
    return s

def zscore(s: pd.Series):
    # Use a rolling window (~5 years weekly = 260 points) for stability
    if len(s) < 10:
        return s * np.nan
    roll = 260
    mu = s.rolling(roll, min_periods=max(26, roll // 10)).mean()
    sd = s.rolling(roll, min_periods=max(26, roll // 10)).std()
    return (s - mu) / sd

# --- Indicators we display ---
# Existing US indicators + ECB USD/EUR
series_map = {
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)": "unemployment_rate.csv",
    "US HY OAS (bps)": "hy_credit_spread.csv",
    "ECB USD per EUR (monthly)": "eur_usd_ecb.csv",
    "USD/CNY (derived, ECB monthly)": "usd_cny_ecb.csv",
    "OECD China CLI (amplitude adj., SA)": "china_cli_oecd.csv",
}

display_rows = []
z_values = []

for label, fname in series_map.items():
    path = DATA_DIR / fname
    if not path.exists():
        display_rows.append((label, "-", "-", "-"))
        continue

    s = load_series(path, weekly_resample=True)
    if s.empty:
        display_rows.append((label, "-", "-", "-"))
        continue

    zs = zscore(s)

    last_date = s.index[-1].strftime("%Y-%m-%d")
    last_val = s.iloc[-1]
    last_z = zs.iloc[-1] if not np.isnan(zs.iloc[-1]) else np.nan

    # Append table row
    display_rows.append((
        label,
        last_date,
        f"{last_val:.2f}",
        "-" if np.isnan(last_z) else f"{last_z:.2f}"
    ))

    # Contribution to composite (direction rules)
    if not np.isnan(last_z):
        # Yield curve: inversion (more negative) is risk => negative_is_risky
        if "Yield Curve" in label:
            z_values.append(-last_z)
        # HY OAS higher = risk
        elif "HY OAS" in label:
            z_values.append(last_z)
        # Unemployment higher = risk
        elif "Unemployment" in label:
            z_values.append(last_z)
        # ECB USD per EUR higher = stronger USD = tighter global liquidity = risk
        elif "USD per EUR" in label:
            z_values.append(last_z)      
        elif "USD/CNY" in label:
            z_values.append(last_z)


# Compute composite
composite = np.nan
if z_values:
    composite = float(np.mean(z_values))

elif "China CLI" in label:
    # Lower-than-trend CLI -> more risk (negative z should increase risk score)
    z_values.append(-last_z)

level = "OK"
if not np.isnan(composite):
    if composite >= 2.0:
        level = "ALERT"
    elif composite >= 1.0:
        level = "WATCH"

# Render a simple HTML
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

html.append("<h3>Key Indicators (US + ECB)</h3>")
html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z-Score</th></tr>")
for r in display_rows:
    html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>")
html.append("</table>")

html.append("<p class='small'>Notes: Z-scores use a rolling historical window. Direction mapping: "
            "curve inversion (more negative) = risk; HY OAS ↑ = risk; unemployment ↑ = risk; USD per EUR ↑ = risk (USD strength).</p>")

html.append("</body></html>")

OUT_DIR.joinpath("index.html").write_text("\n".join(html), encoding="utf-8")
print("✅ Wrote output/index.html")
