
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Optional: YAML config
try:
    import yaml  # installed by workflow
except Exception:
    yaml = None

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Defaults (used if config.yaml missing)
# -------------------------------
DEFAULTS = {
    "windows": {
        "zscore_lookback_weeks": 260,  # ~5 years of weekly points
    },
    "thresholds": {
        "composite_watch": 1.0,
        "composite_alert": 2.0,
        "china_watch": 1.0,
        "china_alert": 2.0,
    },
    "weights": {
        "china": {
            "cli": 0.60,     # OECD China CLI (amplitude adj., SA)
            "usdcny": 0.40,  # USD/CNY (derived from ECB)
        }
    },
    "indicators": {
        "table_us_ecb": [
            "US Yield Curve 10Y–3M (bps)",
            "US Unemployment Rate (%)",
            "US HY OAS (bps)",
            "ECB USD per EUR (monthly)",
        ],
        "china": [
            "OECD China CLI (amplitude adj., SA)",
            "USD/CNY (derived, ECB monthly)",
        ],
    },
}

# -------------------------------
# Load config.yaml (if present)
# -------------------------------
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
        # nested merges
        if "weights" in user_cfg and "china" in (user_cfg["weights"] or {}):
            CFG["weights"]["china"] = {
                **DEFAULTS["weights"]["china"],
                **(user_cfg["weights"]["china"] or {}),
            }
        if "indicators" in user_cfg:
            for sec, lst in (user_cfg["indicators"] or {}).items():
                if isinstance(lst, list) and lst:
                    CFG["indicators"][sec] = lst
    except Exception as e:
        print(f"⚠️  Failed to read config.yaml; using defaults. Error: {e}")

LOOKBACK = int(CFG["windows"]["zscore_lookback_weeks"])
TH_COMPOSITE_WATCH = float(CFG["thresholds"]["composite_watch"])
TH_COMPOSITE_ALERT = float(CFG["thresholds"]["composite_alert"])
TH_CHINA_WATCH = float(CFG["thresholds"]["china_watch"])
TH_CHINA_ALERT = float(CFG["thresholds"]["china_alert"])
W_CHINA_CLI = float(CFG["weights"]["china"]["cli"])
W_CHINA_USDCNY = float(CFG["weights"]["china"]["usdcny"])

# -------------------------------
# Helpers
# -------------------------------
def load_series(csv_path: Path, weekly_resample: bool = True) -> pd.Series:
    """Load CSV with columns (date,value) into a pandas Series indexed by datetime."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    s = df.set_index("date")["value"]
    if weekly_resample:
        s = s.resample("W-FRI").last().interpolate()
    return s


def zscore(s: pd.Series) -> pd.Series:
    """Rolling z-score vs history (configurable lookback)."""
    if len(s) < 10:
        return s * np.nan
    roll = max(10, LOOKBACK)
    mu = s.rolling(roll, min_periods=max(26, roll // 10)).mean()
    sd = s.rolling(roll, min_periods=max(26, roll // 10)).std()
    return (s - mu) / sd


def sparkline_svg(values, width: int = 140, height: int = 28, stroke: str = "#1f77b4") -> str:
    if values is None:
        return ""
    v = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty or len(v) < 2:
        return ""
    v_min, v_max = float(v.min()), float(v.max())
    span = v_max - v_min if v_max != v_min else 1.0
    x_step = width / (len(v) - 1)
    pts = []
    for i, y in enumerate(v):
        x = i * x_step
        y_norm = (float(y) - v_min) / span
        y_svg = height - 2 - y_norm * (height - 4)
        pts.append(f"{x:.1f},{y_svg:.1f}")
    path = "M " + " L ".join(pts)
    svg = (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<path d='{path}' fill='none' stroke='{stroke}' stroke-width='1.5' /></svg>"
    )
    return svg


def spark_from_file(file_name: str, tail_points: int = 52) -> str:
    path = DATA_DIR / file_name
    if not path.exists():
        return ""
    s = load_series(path, weekly_resample=True)
    if s.empty:
        return ""
    return sparkline_svg(s.tail(tail_points).tolist())


def direction_for_label(label: str):
    """Return (direction_name, sign) for contribution mapping."""
    if "Yield Curve" in label:
        return "negative_is_risky", -1
    if "HY OAS" in label:
        return "positive_is_risky", +1
    if "Unemployment" in label:
        return "positive_is_risky", +1
    if "USD per EUR" in label:
        return "positive_is_risky", +1
    if "USD/CNY" in label:
        return "positive_is_risky", +1
    if "China CLI" in label:
        return "negative_is_risky", -1
    return "ambiguous", +1  # fallback


# -------------------------------
# Indicator map (file names)
# -------------------------------
series_map = {
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)": "unemployment_rate.csv",
    "US HY OAS (bps)": "hy_credit_spread.csv",
    "ECB USD per EUR (monthly)": "eur_usd_ecb.csv",
    "USD/CNY (derived, ECB monthly)": "usd_cny_ecb.csv",
    "OECD China CLI (amplitude adj., SA)": "china_cli_oecd.csv",
   # NEW — Japan
    "USD/JPY (derived, ECB monthly)": "usd_jpy_ecb.csv",
    "OECD Japan CLI (amplitude adj., SA)": "japan_cli_oecd.csv,
   #europe 
    "OECD Euro Area CLI (amplitude adj., SA)": "euroarea_cli_oecd.csv"

}

# -------------------------------
# Compute latest readings, composite and contributions
# -------------------------------
display_rows = []
z_values = []
contrib_rows = []  # (label, last_z, direction_name, contribution)

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

    display_rows.append(
        (label, last_date, f"{last_val:.2f}", "-" if np.isnan(last_z) else f"{last_z:.2f}")
    )

    if not np.isnan(last_z):
        dir_name, sign = direction_for_label(label)
        contribution = sign * float(last_z)  # sign-adjusted z used by composite
        contrib_rows.append((label, float(last_z), dir_name, contribution))
        z_values.append(contribution)

# Global composite (mean of contributions)
composite = np.nan if not z_values else float(np.mean(z_values))
level = "OK"
if not np.isnan(composite):
    if composite >= TH_COMPOSITE_ALERT:
        level = "ALERT"
    elif composite >= TH_COMPOSITE_WATCH:
        level = "WATCH"

# Prepare current and previous contribution dicts for Δ
current_contrib = {lbl: contrib for (lbl, _z, _d, contrib) in contrib_rows}

prev_state_path = OUT_DIR / "prev_state.json"
prev_contrib = {}
if prev_state_path.exists():
    try:
        prev_state = json.loads(prev_state_path.read_text(encoding="utf-8"))
        prev_contrib = prev_state.get("contributions", {}) or {}
    except Exception as e:
        print(f"⚠️  Could not parse prev_state.json: {e}")

# Rank contributors by absolute *current* impact
contrib_clean = [(l, z, d, c) for (l, z, d, c) in contrib_rows if not np.isnan(c)]
abs_sum = sum(abs(c) for (_, _, _, c) in contrib_clean) or 0.0
contributors_sorted = sorted(contrib_clean, key=lambda t: abs(t[3]), reverse=True)

# -------------------------------
# China sub-bucket (config weights)
# -------------------------------
china_items = [
    ("OECD China CLI (amplitude adj., SA)", "negative_is_risky", W_CHINA_CLI),
    ("USD/CNY (derived, ECB monthly)", "positive_is_risky", W_CHINA_USDCNY),
]

china_scores = []
for label, direction, w in china_items:
    path = DATA_DIR / series_map[label]
    if not path.exists():
        continue
    s = load_series(path, weekly_resample=True)
    if s.empty:
        continue
    zs = zscore(s)
    z_last = zs.iloc[-1] if len(zs) else np.nan
    if np.isnan(z_last):
        continue
    score = -float(z_last) if direction == "negative_is_risky" else float(z_last)
    china_scores.append((score, w))

china_subscore = np.nan
if china_scores:
    total_w = sum(w for _, w in china_scores)
    if total_w > 0:
        china_subscore = sum(score * w for score, w in china_scores) / total_w

china_level = "OK"
if not np.isnan(china_subscore):
    if china_subscore >= TH_CHINA_ALERT:
        china_level = "ALERT"
    elif china_subscore >= TH_CHINA_WATCH:
        china_level = "WATCH"

# -------------------------------
# Render HTML
# -------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
html = []
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Global Crisis Early Warning – Prototype</title>")
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
</style>
""")
html.append("</head><body>")
html.append("<h1>Global Crisis Early Warning – Prototype</h1>")
html.append(f"<p class='small'>Generated: {now}</p>")

# Composite badge
badge = "<span class='badge ok'>OK</span>"
if level == "WATCH": badge = "<span class='badge watch'>WATCH</span>"
if level == "ALERT": badge = "<span class='badge alert'>ALERT</span>"
comp_str = "-" if np.isnan(composite) else f"{composite:.2f}"
html.append(f"<h2>Composite Risk Score: {comp_str} {badge}</h2>")

# ---- Contributors table (with Δ vs prev)
html.append("<h3>Contributors (sign‑adjusted Z)</h3>")
html.append("<table><tr>"
            "<th>Indicator</th><th>Z‑score</th><th>Direction</th>"
            "<th>Contribution</th><th>Δ vs prev</th><th>Share</th>"
            "</tr>")
if contributors_sorted:
    for label, z, d, c in contributors_sorted:
        cls = "pos" if c >= 0 else "neg"
        prev_c = float(prev_contrib.get(label, 0.0)) if prev_contrib else 0.0
        delta = c - prev_c
        cls_d = "pos" if delta >= 0 else "neg"
        share = "-" if abs_sum == 0 else f"{(abs(c)/abs_sum)*100:.0f}%"
        html.append(
            f"<tr><td>{label}</td>"
            f"<td>{z:+.2f}</td>"
            f"<td>{d.replace('_',' ')}</td>"
            f"<td class='{cls}'>{c:+.2f}</td>"
            f"<td class='{cls_d}'>{delta:+.2f}</td>"
            f"<td>{share}</td></tr>"
        )
else:
    html.append("<tr><td colspan='6'>No contributions available.</td></tr>")
html.append("</table>")

# ---- US + ECB table
us_ecb_labels = CFG["indicators"]["table_us_ecb"]
html.append("<h3>Key Indicators (US + ECB)</h3>")
html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z-Score</th></tr>")
for r in display_rows:
    if r[0] in us_ecb_labels:
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>")
html.append("</table>")

# ---- China block with subscore + sparklines
china_labels = CFG["indicators"]["china"]
html.append("<h3>China</h3>")
badge_cn = "<span class='badge ok'>OK</span>"
if china_level == "WATCH": badge_cn = "<span class='badge watch'>WATCH</span>"
if china_level == "ALERT": badge_cn = "<span class='badge alert'>ALERT</span>"
cn_str = "-" if np.isnan(china_subscore) else f"{china_subscore:.2f}"
html.append(f"<p><b>China Subscore:</b> {cn_str} {badge_cn}</p>")

html.append("<table><tr>"
            "<th>Indicator</th><th>Last Date</th><th>Last Value</th>"
            "<th>Z-Score</th><th>Sparkline (52w)</th>"
            "</tr>")
for label in china_labels:
    rows = [r for r in display_rows if r[0] == label]
    if not rows:
        continue
    r = rows[0]
    svg = spark_from_file(series_map[label], tail_points=52)
    html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{svg}</td></tr>")
html.append("</table>")


# ---- Japan block with subscore + sparklines ----
if "japan" in CFG["indicators"]:
    jp_labels = CFG["indicators"]["japan"]
    # Compute Japan subscore from config weights (default 60/40)
    jp_items = [
        ("OECD Japan CLI (amplitude adj., SA)", "negative_is_risky", float(CFG["weights"]["japan"]["cli"])),
        ("USD/JPY (derived, ECB monthly)", "positive_is_risky", float(CFG["weights"]["japan"]["usdjpy"])),
    ]
    jp_scores = []
    for lbl, direction, w in jp_items:
        path = DATA_DIR / series_map[lbl]
        if not path.exists():
            continue
        s = load_series(path, weekly_resample=True)
        if s.empty:
            continue
        zs = zscore(s)
        z_last = zs.iloc[-1] if len(zs) else np.nan
        if np.isnan(z_last):
            continue
        score = -float(z_last) if direction == "negative_is_risky" else float(z_last)
        jp_scores.append((score, w))
    jp_sub = np.nan
    if jp_scores:
        tw = sum(w for _, w in jp_scores)
        if tw > 0:
            jp_sub = sum(sc*w for sc, w in jp_scores) / tw
    jp_level = "OK"
    if not np.isnan(jp_sub):
        if jp_sub >= TH_CHINA_ALERT:  # reuse thresholds
            jp_level = "ALERT"
        elif jp_sub >= TH_CHINA_WATCH:
            jp_level = "WATCH"

    html.append("<h3>Japan</h3>")
    badge_jp = "<span class='badge ok'>OK</span>"
    if jp_level == "WATCH": badge_jp = "<span class='badge watch'>WATCH</span>"
    if jp_level == "ALERT": badge_jp = "<span class='badge alert'>ALERT</span>"
    jp_str = "-" if np.isnan(jp_sub) else f"{jp_sub:.2f}"
    html.append(f"<p><b>Japan Subscore:</b> {jp_str} {badge_jp}</p>")

    html.append("<table><tr>"
                "<th>Indicator</th><th>Last Date</th><th>Last Value</th>"
                "<th>Z-Score</th><th>Sparkline (52w)</th>"
                "</tr>")
    for lbl in jp_labels:
        rows = [r for r in display_rows if r[0] == lbl]
        if not rows:
            continue
        r = rows[0]
        svg = spark_from_file(series_map[lbl], tail_points=52)
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{svg}</td></tr>")
    html.append("</table>")


# ---- Eurozone block with subscore + sparklines ----
if "eurozone" in CFG["indicators"]:
    ez_labels = CFG["indicators"]["eurozone"]
    ez_items = [
        ("OECD Euro Area CLI (amplitude adj., SA)", "negative_is_risky", float(CFG["weights"]["eurozone"]["cli"])),
        ("ECB USD per EUR (monthly)", "positive_is_risky", float(CFG["weights"]["eurozone"]["usdeur"])),
    ]
    ez_scores = []
    for lbl, direction, w in ez_items:
        path = DATA_DIR / series_map[lbl]
        if not path.exists():
            continue
        s = load_series(path, weekly_resample=True)
        if s.empty:
            continue
        zs = zscore(s)
        z_last = zs.iloc[-1] if len(zs) else np.nan
        if np.isnan(z_last):
            continue
        score = -float(z_last) if direction == "negative_is_risky" else float(z_last)
        ez_scores.append((score, w))

    ez_sub = np.nan
    if ez_scores:
        tw = sum(w for _, w in ez_scores)
        if tw > 0:
            ez_sub = sum(sc*w for sc, w in ez_scores) / tw

    ez_level = "OK"
    if not np.isnan(ez_sub):
        if ez_sub >= TH_CHINA_ALERT:   # reuse thresholds (watch/alert)
            ez_level = "ALERT"
        elif ez_sub >= TH_CHINA_WATCH:
            ez_level = "WATCH"

    html.append("<h3>Eurozone</h3>")
    badge_ez = "<span class='badge ok'>OK</span>"
    if ez_level == "WATCH": badge_ez = "<span class='badge watch'>WATCH</span>"
    if ez_level == "ALERT": badge_ez = "<span class='badge alert'>ALERT</span>"
    ez_str = "-" if np.isnan(ez_sub) else f"{ez_sub:.2f}"
    html.append(f"<p><b>Eurozone Subscore:</b> {ez_str} {badge_ez}</p>")

    html.append("<table><tr>"
                "<th>Indicator</th><th>Last Date</th><th>Last Value</th>"
                "<th>Z-Score</th><th>Sparkline (52w)</th>"
                "</tr>")
    for lbl in ez_labels:
        rows = [r for r in display_rows if r[0] == lbl]
        if not rows:
            continue
        r = rows[0]
        svg = spark_from_file(series_map[lbl], tail_points=52)
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{svg}</td></tr>")
    html.append("</table>")


html.append(
    "<p class='small'>Notes: Contributions use the same direction rules as the composite: "
    "curve inversion (more negative) = risk↑; HY OAS ↑ = risk↑; unemployment ↑ = risk↑; "
    "USD per EUR ↑ = risk↑; USD/CNY ↑ = risk↑; China CLI below trend = risk↑. "
    "Edit <code>config.yaml</code> to adjust thresholds, lookback, and weights.</p>"
)

html.append("</body></html>")

# Write HTML
OUT_DIR.joinpath("index.html").write_text("\n".join(html), encoding="utf-8")
print("✅ Wrote output/index.html")

# Persist state for automation (ALERT/WATCH) + contributions (for next Δ)
state = {
    "generated_utc": now,
    "composite": None if np.isnan(composite) else round(float(composite), 3),
    "composite_level": level,
    "china_subscore": None if np.isnan(china_subscore) else round(float(china_subscore), 3),
    "china_level": china_level,
    "contributions": {k: round(float(v), 6) for k, v in current_contrib.items()}
}
OUT_DIR.joinpath("state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
print("✅ Wrote output/state.json")
