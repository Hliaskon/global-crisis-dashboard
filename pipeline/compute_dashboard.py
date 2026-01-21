
# pipeline/compute_dashboard.py
# Step 1 + Step 2 presentation improvements:
# - CSS tweaks, section separators, "country cards", right-aligned numerics, dim missing rows
# - Composite "hero" block and friendly direction wording in Contributors

import os, json, shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Optional YAML config
try:
    import yaml
except Exception:
    yaml = None

DATA_DIR = Path("data")
OUT_DIR  = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "data").mkdir(parents=True, exist_ok=True)  # for download links

# ---------------------
# Config & Defaults
# ---------------------
DEFAULTS = {
    "windows": {"zscore_lookback_weeks": 260},
    "thresholds": {
        "composite_watch": 1.0,
        "composite_alert": 2.0,
        "country_watch": 1.0,
        "country_alert": 2.0,
    },
    "weights": {
        "default": {"cli": 0.60, "fx": 0.40}
    },
    "indicators": {
        "table_us_ecb": [
            "US Yield Curve 10Y–3M (bps)",
            "US Unemployment Rate (%)",
            "US HY OAS (bps)",
            "USD per EUR (ECB)",
        ],
    },
    "countries": []  # filled by config.yaml
}

CFG = DEFAULTS.copy()
cfg_path = Path("config.yaml")
if yaml is not None and cfg_path.exists():
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # shallow merge
        for k in DEFAULTS:
            if isinstance(DEFAULTS[k], dict):
                CFG[k] = {**DEFAULTS[k], **(user_cfg.get(k, {}) or {})}
            else:
                CFG[k] = user_cfg.get(k, DEFAULTS[k])
        # list keys
        if "countries" in user_cfg and isinstance(user_cfg["countries"], list):
            CFG["countries"] = user_cfg["countries"]
    except Exception as e:
        print(f"⚠️ Failed to read config.yaml; using defaults. Error: {e}")

LOOKBACK = int(CFG["windows"]["zscore_lookback_weeks"])
TH_COMP_WATCH = float(CFG["thresholds"]["composite_watch"])
TH_COMP_ALERT = float(CFG["thresholds"]["composite_alert"])
TH_CT_WATCH   = float(CFG["thresholds"]["country_watch"])
TH_CT_ALERT   = float(CFG["thresholds"]["country_alert"])
W_DEF_CLI     = float(CFG["weights"]["default"]["cli"])
W_DEF_FX      = float(CFG["weights"]["default"]["fx"])

# ---------------------
# Helpers
# ---------------------
def load_series(csv_path: Path, weekly_resample: bool = True) -> pd.Series:
    if not csv_path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_csv(csv_path)
    if set(["date","value"]).issubset(df.columns):
        pass
    else:
        # fallback: pick first two columns
        df.columns = ["date","value"] + list(df.columns[2:])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    s = df.set_index("date")["value"]
    if weekly_resample:
        s = s.resample("W-FRI").last().interpolate()
    return s

def zscore(s: pd.Series) -> pd.Series:
    if len(s) < 10:
        return s * np.nan
    roll = max(10, LOOKBACK)
    mu = s.rolling(roll, min_periods=max(26, roll // 10)).mean()
    sd = s.rolling(roll, min_periods=max(26, roll // 10)).std()
    return (s - mu) / sd

def sparkline_svg(values, width: int = 120, height: int = 26, stroke: str = "#1f77b4") -> str:
    if values is None:
        return ""
    v = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty or len(v) < 2:
        return ""
    vmin, vmax = float(v.min()), float(v.max())
    span = (vmax - vmin) if vmax != vmin else 1.0
    x_step = width / (len(v) - 1)
    pts = []
    for i, y in enumerate(v):
        x = i * x_step
        y_norm = (float(y) - vmin) / span
        y_svg = height - 2 - y_norm * (height - 4)
        pts.append(f"{x:.1f},{y_svg:.1f}")
    path = "M " + " L ".join(pts)
    svg = (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<path d='{path}' fill='none' stroke='{stroke}' stroke-width='1.5' /></svg>"
    )
    return svg

# ---------------------
# Series mapping (labels -> files)
# ---------------------
series_map = {
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)":    "unemployment_rate.csv",
    "US HY OAS (bps)":             "hy_credit_spread.csv",
    "USD per EUR (ECB)":           "eur_usd_ecb.csv",

    # FX derived (USD per currency)
    "USD/JPY (derived, ECB)": "usd_jpy_ecb.csv",
    "USD/CNY (derived, ECB)": "usd_cny_ecb.csv",
    "USD/GBP (derived, ECB)": "usd_gbp_ecb.csv",
    "USD/CAD (derived, ECB)": "usd_cad_ecb.csv",
    "USD/INR (derived, ECB)": "usd_inr_ecb.csv",
    "USD/RUB (derived, ECB)": "usd_rub_ecb.csv",
}

series_map.update({
    # Commodities
    "Gold (LBMA, USD/oz)": "gold_lbma_usd.csv",            # positive_is_risky
    "Brent crude (USD/bbl)": "brent_spot_fred.csv",         # positive_is_risky

    # Equities (price level)
    "S&P 500 index (close)": "sp500_fred.csv",              # negative_is_risky (price weakness → risk↑)

    # Rates (policy)
    "Policy rate – United States (BIS)": "policy_us.csv",
    "Policy rate – Germany (BIS)": "policy_de.csv",
    "Policy rate – France (BIS)": "policy_fr.csv",
    "Policy rate – Italy (BIS)": "policy_it.csv",
    "Policy rate – United Kingdom (BIS)": "policy_gb.csv",
    "Policy rate – Canada (BIS)": "policy_ca.csv",
    "Policy rate – Japan (BIS)": "policy_jp.csv",
    "Policy rate – India (BIS)": "policy_in.csv",
    "Policy rate – Russia (BIS)": "policy_ru.csv",

    # Long-term yields (OECD/MEI via FRED)
    "10Y yield – United States (OECD)": "yield10_us.csv",
    "10Y yield – Germany (OECD)": "yield10_de.csv",
    "10Y yield – France (OECD)": "yield10_fr.csv",
    "10Y yield – Italy (OECD)": "yield10_it.csv",
    "10Y yield – United Kingdom (OECD)": "yield10_gb.csv",
    "10Y yield – Japan (OECD)": "yield10_jp.csv",
    "10Y yield – Canada (OECD)": "yield10_ca.csv",

    # Industrial production (OECD/MEI)
    "Industrial Production – Germany (OECD)": "ip_de.csv",
    "Industrial Production – France (OECD)": "ip_fr.csv",
    "Industrial Production – Italy (OECD)": "ip_it.csv",
    "Industrial Production – United States (OECD)": "ip_us.csv",
    "Industrial Production – Japan (OECD)": "ip_jp.csv",
})

def direction_for_label(label: str):
    if "S&P 500" in label:
        return "negative_is_risky", -1
    if "Gold" in label:
        return "positive_is_risky", +1
    if "Brent" in label:
        return "positive_is_risky", +1
    if "Policy rate" in label:
        return "positive_is_risky", +1
    if "10Y yield" in label:
        return "positive_is_risky", +1
    if "Industrial Production" in label:
        return "negative_is_risky", -1
    # ... keep the rest of your existing mapping

# Country CLI labels will be added dynamically from config with filenames like <country>_cli_oecd.csv
def cli_filename_from_label(label: str) -> str:
    # Simple normalized key from label (not used by logic now, reserved)
    key = (label.lower()
                 .replace("oecd", "")
                 .replace("cli", "")
                 .replace("(amplitude adj., sa)", "")
                 .replace("(", "").replace(")", "")
                 .replace(" ", "").replace(",", "").replace("–","-"))
    return f"{key}_cli_oecd.csv"

# Direction mapping & sign
def direction_for_label(label: str):
    if "Yield Curve" in label:
        return "negative_is_risky", -1
    if "HY OAS" in label:
        return "positive_is_risky", +1
    if "Unemployment" in label:
        return "positive_is_risky", +1
    if "USD per EUR" in label:      # USD strength
        return "positive_is_risky", +1
    if label.startswith("USD/"):    # USD/XXX ↑ => risk↑
        return "positive_is_risky", +1
    if "CLI" in label:
        return "negative_is_risky", -1
    return "ambiguous", +1

# Human-readable wording for direction
def human_direction(direction_name: str) -> str:
    """
    Map internal direction flags to friendly wording for display.
    """
    if direction_name == "negative_is_risky":
        return "Below trend → higher risk"
    if direction_name == "positive_is_risky":
        return "Rising → higher risk"
    return "—"

# ---------------------
# Build readings & composite
# ---------------------
display_rows = []          # [(label, last_date, last_val, last_z)]
contrib_rows  = []         # [(label, last_z, direction, contribution)]
z_values      = []         # list of contributions

# 1) US + ECB table items (also contribute to composite)
for label, fname in series_map.items():
    path = DATA_DIR / fname
    s = load_series(path, weekly_resample=True)
    if s.empty:
        continue
    zs = zscore(s)
    last_date = s.index[-1].strftime("%Y-%m-%d")
    last_val  = s.iloc[-1]
    last_z    = zs.iloc[-1] if not np.isnan(zs.iloc[-1]) else np.nan
    display_rows.append((label, last_date, f"{last_val:.2f}", "-" if np.isnan(last_z) else f"{last_z:.2f}"))
    if not np.isnan(last_z):
        dname, sign = direction_for_label(label)
        contribution = sign * float(last_z)
        contrib_rows.append((label, float(last_z), dname, contribution))
        z_values.append(contribution)

# 2) Country CLIs (from config) — also contribute to composite (as "below-trend → higher risk")
for c in CFG["countries"]:
    cli_label = c["cli_label"]
    safe = (c["name"].lower().replace(" ", "_"))
    cli_file = DATA_DIR / f"{safe}_cli_oecd.csv"
    s = load_series(cli_file, weekly_resample=True)
    if s.empty:
        continue
    zs = zscore(s)
    last_date = s.index[-1].strftime("%Y-%m-%d")
    last_val  = s.iloc[-1]
    last_z    = zs.iloc[-1] if not np.isnan(zs.iloc[-1]) else np.nan
    display_rows.append((cli_label, last_date, f"{last_val:.2f}", "-" if np.isnan(last_z) else f"{last_z:.2f}"))
    if not np.isnan(last_z):
        dname, sign = direction_for_label(cli_label)  # negative_is_risky expected for CLI
        contribution = sign * float(last_z)
        contrib_rows.append((cli_label, float(last_z), dname, contribution))
        z_values.append(contribution)

# Composite = mean of contributions
composite = np.nan if not z_values else float(np.mean(z_values))
level = "OK"
if not np.isnan(composite):
    if composite >= TH_COMP_ALERT:
        level = "ALERT"
    elif composite >= TH_COMP_WATCH:
        level = "WATCH"

# Rank contributors by absolute impact
contrib_clean = [(l, z, d, c) for (l, z, d, c) in contrib_rows if not np.isnan(c)]
abs_sum = sum(abs(c) for (_, _, _, c) in contrib_clean) or 0.0
contributors_sorted = sorted(contrib_clean, key=lambda t: abs(t[3]), reverse=True)

# ---------------------
# Country subscores
# ---------------------
country_states = []   # list of dicts for HTML + state.json
for c in CFG["countries"]:
    name      = c["name"]
    cli_label = c["cli_label"]
    fx_label  = c["fx_label"]
    fx_file   = c["fx_source"]
    w_cli     = float(c.get("weights", {}).get("cli", W_DEF_CLI))
    w_fx      = float(c.get("weights", {}).get("fx",  W_DEF_FX))

    safe = (name.lower().replace(" ", "_"))
    cli_path = DATA_DIR / f"{safe}_cli_oecd.csv"
    fx_path  = DATA_DIR / fx_file

    s_cli = load_series(cli_path, weekly_resample=True)
    s_fx  = load_series(fx_path,  weekly_resample=True)

    z_cli = zscore(s_cli)
    z_fx  = zscore(s_fx)

    z_last_cli = z_cli.iloc[-1] if len(z_cli) else np.nan
    z_last_fx  = z_fx.iloc[-1]  if len(z_fx)  else np.nan

    score_cli = (-float(z_last_cli)) if not np.isnan(z_last_cli) else np.nan   # negative_is_risky
    score_fx  = ( float(z_last_fx))  if not np.isnan(z_last_fx)  else np.nan   # positive_is_risky

    parts = []
    if not np.isnan(score_cli):
        parts.append((score_cli, w_cli))
    if not np.isnan(score_fx):
        parts.append((score_fx,  w_fx))

    subscore = np.nan
    if parts:
        wsum = sum(w for (_, w) in parts)
        if wsum > 0:
            subscore = sum(s*w for (s,w) in parts) / wsum

    lvl = "OK"
    if not np.isnan(subscore):
        if subscore >= TH_CT_ALERT:
            lvl = "ALERT"
        elif subscore >= TH_CT_WATCH:
            lvl = "WATCH"

    country_states.append({
        "name": name,
        "subscore": None if np.isnan(subscore) else float(subscore),
        "level": lvl,
        "cli_label": cli_label, "fx_label": fx_label,
        "cli_file": f"{safe}_cli_oecd.csv", "fx_file": fx_file,
    })

# ---------------------
# Δ vs previous run (for Contributors)
# ---------------------
prev_state_path = OUT_DIR / "prev_state.json"
prev_contrib = {}
if prev_state_path.exists():
    try:
        prev_state = json.loads(prev_state_path.read_text(encoding="utf-8"))
        prev_contrib = prev_state.get("contributions", {}) or {}
    except Exception as e:
        print(f"⚠️ Could not parse prev_state.json: {e}")

current_contrib = {lbl: float(contrib) for (lbl, _z, _d, contrib) in contrib_rows}

# ---------------------
# Render HTML
# ---------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
html = []
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Global Crisis Early Warning – Dashboard</title>")
html.append("""
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:24px;}
table{border-collapse:collapse;margin-top:12px;}
th,td{border:1px solid #ddd;padding:6px 10px;vertical-align:middle;}
th{background:#f4f6f8;}
.badge{display:inline-block;padding:6px 10px;border-radius:6px;margin-left:8px;}
.ok{background:#e8f5e9;} .watch{background:#fff8e1;} .alert{background:#ffebee;}
.pos{color:#b71c1c;} .neg{color:#1b5e20;}
.small{color:#666;font-size:12px}
td svg{display:block}

/* ---------- Layout & spacing ---------- */
h2 { margin-top: 6px; }
h3 { margin-top: 28px; }

.section {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 2px solid #eee;
}

/* ---------- Country cards ---------- */
.country {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 12px 16px;
  margin-top: 18px;
  background: #ffffff;
}

.country h3 {
  margin-top: 0;
}

/* ---------- Typography ---------- */
.subscore {
  font-size: 18px;
}

.dim {
  color: #999;
  font-style: italic;
}

th {
  text-align: left;
}

td.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

caption {
  text-align: left;
  font-weight: bold;
  padding-bottom: 6px;
}

/* (Optional) soft header color for section headings */
h3.section { color:#2f3b49; }
</style>
""")
html.append("</head><body>")
html.append("<h1>Global Crisis Early Warning – Dashboard</h1>")
html.append(f"<p class='small'>Generated: {now}</p>")

# Composite hero (badge & comp_str must be defined BEFORE hero block)
badge = "<span class='badge ok'>OK</span>"
if level == "WATCH":
    badge = "<span class='badge watch'>WATCH</span>"
if level == "ALERT":
    badge = "<span class='badge alert'>ALERT</span>"
comp_str = "-" if np.isnan(composite) else f"{composite:.2f}"

html.append(
    "<div style='margin:18px 0 12px 0;padding:14px 16px;border:1px solid #e0e0e0;"
    "border-radius:10px;background:#fafbfc;'>"
    "<div style='font-size:13px;color:#666;margin-bottom:4px;'>🌍 Global Composite Risk</div>"
    f"<div style='display:flex;align-items:center;gap:12px;'>"
    f"<div style='font-size:36px;line-height:1.0;font-weight:600;'>{comp_str}</div>"
    f"{badge}"
    "</div>"
    "<div class='small' style='margin-top:6px;color:#777;'>"
    "WATCH ≥ +1σ, ALERT ≥ +2σ (configurable in <code>config.yaml</code>)"
    "</div>"
    "</div>"
)

# Contributors (with Δ vs prev) — use plain language for direction
html.append("<h3 class='section'>Contributors (sign-adjusted Z)</h3>")
html.append("<table><tr><th>Indicator</th><th>Z-score</th><th>Direction</th><th>Contribution</th><th>Δ vs prev</th><th>Share</th></tr>")
if contributors_sorted:
    for (lbl, z, dname, c) in contributors_sorted:
        cls = "pos" if c >= 0 else "neg"
        prev_c = float(prev_contrib.get(lbl, 0.0)) if prev_contrib else 0.0
        delta = c - prev_c
        cls_d = "pos" if delta >= 0 else "neg"
        share = "-" if abs_sum == 0 else f"{(abs(c)/abs_sum)*100:.0f}%"
        html.append(
            f"<tr><td>{lbl}</td>"
            f"<td class='num'>{z:+.2f}</td>"
            f"<td>{human_direction(dname)}</td>"
            f"<td class='num {cls}'>{c:+.2f}</td>"
            f"<td class='num {cls_d}'>{delta:+.2f}</td>"
            f"<td class='num'>{share}</td></tr>"
        )
else:
    html.append("<tr><td colspan='6'>No contributions available.</td></tr>")
html.append("</table>")

# US + ECB table (from config order)
labels_us = CFG["indicators"]["table_us_ecb"]
html.append("<h3 class='section'>Key Indicators (US + ECB)</h3>")
html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z-Score</th></tr>")
for r in display_rows:
    if r[0] in labels_us:
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td class='num'>{r[2]}</td><td class='num'>{r[3]}</td></tr>")
html.append("</table>")

# Country blocks — wrap each country in a card, align numerics, friendlier missing FX message
def spark_from_file(file_name: str, tail_points: int = 52) -> str:
    path = DATA_DIR / file_name
    s = load_series(path, weekly_resample=True)
    if s.empty:
        return ""
    return sparkline_svg(s.tail(tail_points).tolist())

for c in country_states:
    name = c["name"]
    sub  = c["subscore"]
    lvl  = c["level"]
    badge_cn = "<span class='badge ok'>OK</span>"
    if lvl == "WATCH":
        badge_cn = "<span class='badge watch'>WATCH</span>"
    if lvl == "ALERT":
        badge_cn = "<span class='badge alert'>ALERT</span>"
    sub_str = "-" if sub is None else f"{sub:.2f}"

    # find rows for cli & fx from display_rows
    cli_row = next((r for r in display_rows if r[0] == c["cli_label"]), None)
    fx_row  = next((r for r in display_rows if r[0] == c["fx_label"]),  None)

    html.append("<div class='country'>")
    html.append(f"<h3>{name}</h3>")
    html.append(f"<p class='subscore'><b>Subscore:</b> {sub_str} {badge_cn}</p>")
    html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z-Score</th><th>Sparkline (52w)</th><th>Download</th></tr>")

    # CLI row
    if cli_row:
        cli_svg = spark_from_file(c["cli_file"])
        html.append(
            f"<tr><td>{cli_row[0]}</td><td>{cli_row[1]}</td><td class='num'>{cli_row[2]}</td><td class='num'>{cli_row[3]}</td>"
            f"<td>{cli_svg}</td><td><a href='data/{c['cli_file']}'>CSV</a></td></tr>"
        )
    else:
        html.append(f"<tr><td colspan='6' class='dim'>{c['cli_label']} not available in this run</td></tr>")

    # FX row
    if fx_row:
        fx_svg = spark_from_file(c["fx_file"])
        html.append(
            f"<tr><td>{fx_row[0]}</td><td>{fx_row[1]}</td><td class='num'>{fx_row[2]}</td><td class='num'>{fx_row[3]}</td>"
            f"<td>{fx_svg}</td><td><a href='data/{c['fx_file']}'>CSV</a></td></tr>"
        )
    else:
        html.append("<tr><td colspan='6' class='dim'>FX indicator not available in this run</td></tr>")

    html.append("</table>")
    html.append("</div>")  # end .country

# Notes (clean bullets)
html.append(
    "<ul class='small'>"
    "<li>Z-scores use a rolling historical window.</li>"
    "<li>Below-trend CLI increases risk; USD strength increases risk.</li>"
    "<li>Thresholds & weights are configurable in <code>config.yaml</code>.</li>"
    "</ul>"
)
html.append("</body></html>")

# Write HTML
(OUT_DIR / "index.html").write_text("\n".join(html), encoding="utf-8")
print("✅ Wrote output/index.html")

# Copy CSVs for download links
for f in DATA_DIR.glob("*.csv"):
    shutil.copy2(f, OUT_DIR / "data" / f.name)

# Persist state for automation (ALERT/WATCH) + contributions
state = {
    "generated_utc": now,
    "composite": None if np.isnan(composite) else round(float(composite), 3),
    "composite_level": level,
    "countries": [
        {
            "name": c["name"],
            "subscore": None if c["subscore"] is None else round(float(c["subscore"]), 3),
            "level": c["level"]
        } for c in country_states
    ],
    "contributions": {k: round(float(v), 6) for k, v in current_contrib.items()}
}
(OUT_DIR / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
print("✅ Wrote output/state.json")

