
import os
import json
import shutil
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
        "china":  {"cli": 0.60, "usdcny": 0.40},
        "japan":  {"cli": 0.60, "usdjpy": 0.40},
        "eurozone": {"cli": 0.60, "usdeur": 0.40},
        "uk":     {"cli": 0.60, "usdgbp": 0.40},
        "canada": {"cli": 0.60, "usdcad": 0.40},
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
        # These sections render only if present in config.yaml.
        # Defaults here make it easy to enable with no code changes.
        "japan": [
            "OECD Japan CLI (amplitude adj., SA)",
            "USD/JPY (derived, ECB monthly)",
        ],
        "eurozone": [
            "OECD Euro Area CLI (amplitude adj., SA)",
            "ECB USD per EUR (monthly)",
        ],
        "uk": [
            "OECD UK CLI (amplitude adj., SA)",
            "USD/GBP (derived, ECB monthly)",
        ],
        "canada": [
            "OECD Canada CLI (amplitude adj., SA)",
            "USD/CAD (derived, ECB monthly)",
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
        # shallow merge for top-level keys
        for k in DEFAULTS:
            if isinstance(DEFAULTS[k], dict):
                CFG[k] = {**DEFAULTS[k], **(user_cfg.get(k, {}) or {})}
            else:
                CFG[k] = user_cfg.get(k, DEFAULTS[k])
        # nested sub-dicts
        if "weights" in user_cfg:
            for bucket in DEFAULTS["weights"].keys():
                if bucket in user_cfg["weights"] or bucket in CFG["weights"]:
                    CFG["weights"][bucket] = {
                        **DEFAULTS["weights"].get(bucket, {}),
                        **(user_cfg["weights"].get(bucket, {}) if user_cfg["weights"] else {}),
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

# convenience weights
W_CHINA_CLI    = float(CFG["weights"]["china"]["cli"])
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
    if ("USD/CNY" in label) or ("USD/JPY" in label) or ("USD/GBP" in label) or ("USD/CAD" in label):
        return "positive_is_risky", +1
    if any(k in label for k in ["China CLI", "Japan CLI", "Euro Area CLI", "UK CLI", "Canada CLI"]):
        return "negative_is_risky", -1
    return "ambiguous", +1  # fallback


def composite_history_series(series_map: dict, tail_weeks: int = 156, min_count: int = 2) -> pd.Series:
    """
    Build a weekly composite time series from sign-adjusted z-scores
    of all indicators, then take the last `tail_weeks` points.
    """
    cols = []
    for label, fname in series_map.items():
        path = DATA_DIR / fname
        if not path.exists():
            continue
        s = load_series(path, weekly_resample=True)
        if s.empty:
            continue
        zs = zscore(s)
        _, sign = direction_for_label(label)
        cols.append(sign * zs.rename(label))

    if not cols:
        return pd.Series(dtype="float64")

    df = pd.concat(cols, axis=1)  # align by date
    counts = df.count(axis=1)
    m = df.mean(axis=1, skipna=True)
    m[counts < min_count] = np.nan

    if tail_weeks > 0 and len(m) > tail_weeks:
        m = m.iloc[-tail_weeks:]
    return m


def composite_chart_svg(series: pd.Series, smooth: pd.Series = None, width: int = 560, height: int = 140) -> str:
    """
    Render a larger inline SVG line chart for the composite history,
    with guides at 0, +1 (WATCH), +2 (ALERT) and an optional smoothing overlay.
    """
    if series is None or series.dropna().empty:
        return "<div class='small'>No composite history available yet.</div>"

    v = series.astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty or len(v) < 2:
        return "<div class='small'>Insufficient composite history.</div>"

    # Ensure we include the WATCH/ALERT bands in the scale
    lo = min(v.min(), -0.5)
    hi = max(v.max(), 2.5)
    span = hi - lo if hi != lo else 1.0
    x_step = width / (len(v) - 1)

    # main line
    pts = []
    for i, y in enumerate(v):
        x = i * x_step
        y_norm = (float(y) - lo) / span
        y_svg = height - 2 - y_norm * (height - 4)
        pts.append(f"{x:.1f},{y_svg:.1f}")
    main_path = "M " + " L ".join(pts)

    # smoothing overlay aligned to v index
    smooth_path = ""
    if smooth is not None:
        s_aligned = smooth.reindex(v.index).astype("float64").replace([np.inf, -np.inf], np.nan)
        pts_s = []
        for i, y in enumerate(s_aligned):
            if pd.isna(y):
                pts_s.append(None)
                continue
            x = i * x_step
            y_norm = (float(y) - lo) / span
            y_svg = height - 2 - y_norm * (height - 4)
            pts_s.append(f"{x:.1f},{y_svg:.1f}")
        # break into chunks on NaN
        chunks, chunk = [], []
        for p in pts_s:
            if p is None:
                if chunk:
                    chunks.append("M " + " L ".join(chunk))
                    chunk = []
            else:
                chunk.append(p)
        if chunk:
            chunks.append("M " + " L ".join(chunk))
        smooth_path = "".join(
            [f"<path d='{ch}' fill='none' stroke='#8e24aa' stroke-width='2' opacity='0.9'/>" for ch in chunks]
        )

    def y_at(val):
        y_norm = (float(val) - lo) / span
        return height - 2 - y_norm * (height - 4)

    y0, y1, y2 = y_at(0.0), y_at(1.0), y_at(2.0)

    svg = []
    svg.append(f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>")
    svg.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff'/>")
    # Guides
    svg.append(f"<line x1='0' y1='{y0:.1f}' x2='{width}' y2='{y0:.1f}' stroke='#bbb' stroke-dasharray='3,3'/>")
    svg.append(f"<line x1='0' y1='{y1:.1f}' x2='{width}' y2='{y1:.1f}' stroke='#f0b429' stroke-dasharray='4,3'/>")
    svg.append(f"<line x1='0' y1='{y2:.1f}' x2='{width}' y2='{y2:.1f}' stroke='#d32f2f' stroke-dasharray='5,3'/>")
    # Labels
    svg.append(f"<text x='{width-2}' y='{y0-4:.1f}' text-anchor='end' font-size='10' fill='#888'>0</text>")
    svg.append(f"<text x='{width-2}' y='{y1-4:.1f}' text-anchor='end' font-size='10' fill='#f0b429'>+1</text>")
    svg.append(f"<text x='{width-2}' y='{y2-4:.1f}' text-anchor='end' font-size='10' fill='#d32f2f'>+2</text>")
    # main line
    svg.append(f"<path d='{main_path}' fill='none' stroke='#1976d2' stroke-width='2'/>")
    # overlay
    if smooth_path:
        svg.append(smooth_path)
        svg.append("<text x='4' y='12' font-size='10' fill='#1976d2'>Composite</text>")
        svg.append("<text x='4' y='24' font-size='10' fill='#8e24aa'>3‑month MA</text>")
    svg.append("</svg>")
    return "".join(svg)


# -------------------------------
# Indicator map (file names)
# -------------------------------
series_map = {
    # US + ECB core
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)": "unemployment_rate.csv",
    "US HY OAS (bps)": "hy_credit_spread.csv",
    "ECB USD per EUR (monthly)": "eur_usd_ecb.csv",

    # FX crosses derived from ECB
    "USD/CNY (derived, ECB monthly)": "usd_cny_ecb.csv",
    "USD/JPY (derived, ECB monthly)": "usd_jpy_ecb.csv",
    "USD/GBP (derived, ECB monthly)": "usd_gbp_ecb.csv",
    "USD/CAD (derived, ECB monthly)": "usd_cad_ecb.csv",

    # OECD CLIs
    "OECD China CLI (amplitude adj., SA)": "china_cli_oecd.csv",
    "OECD Japan CLI (amplitude adj., SA)": "japan_cli_oecd.csv",
    "OECD Euro Area CLI (amplitude adj., SA)": "euroarea_cli_oecd.csv",
    "OECD UK CLI (amplitude adj., SA)": "uk_cli_oecd.csv",
    "OECD Canada CLI (amplitude adj., SA)": "canada_cli_oecd.csv",
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

    display_rows.append((label, last_date, f"{last_val:.2f}", "-" if np.isnan(last_z) else f"{last_z:.2f}"))

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

# Copy source CSVs to the published site so Download links work
(OUT_DIR / "data").mkdir(exist_ok=True)
for fn in set(series_map.values()):
    src = DATA_DIR / fn
    if src.exists():
        shutil.copyfile(src, OUT_DIR / "data" / fn)

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
svg { margin-top: 6px; }
a{ color:#1565c0; text-decoration:none; }
a:hover{ text-decoration:underline; }
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

# --- Composite history (3y) + 3-month MA and CSV download
comp_hist = composite_history_series(series_map, tail_weeks=156, min_count=2)
comp_ma = comp_hist.rolling(13, min_periods=4).mean()  # ~3 months (13 weeks)
html.append("<h3>Composite history (3y)</h3>")
html.append(composite_chart_svg(comp_hist, comp_ma, width=560, height=140))

# Export composite history CSV (date, composite, composite_ma_13w)
comp_df = pd.DataFrame({"date": comp_hist.index, "composite": comp_hist.values})
comp_df["composite_ma_13w"] = comp_ma.reindex(comp_hist.index).values
comp_df.to_csv(OUT_DIR / "composite_history.csv", index=False)
html.append("<p class='small'><a href='composite_history.csv' download>Download composite history (CSV)</a></p>")

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

# ---- US + ECB table (with Download)
us_ecb_labels = CFG["indicators"]["table_us_ecb"]
html.append("<h3>Key Indicators (US + ECB)</h3>")
html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z-Score</th><th>Download</th></tr>")
for r in display_rows:
    if r[0] in us_ecb_labels:
        fn = series_map[r[0]]
        link = f"<a href='data/{fn}' download>CSV</a>"
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{link}</td></tr>")
html.append("</table>")

def render_block(block_name: str, title: str, items_labels: list):
    html.append(f"<h3>{title}</h3>")
    # Build a subscore if we have weights in config
    weights = CFG["weights"].get(block_name, {})
    scoring = []
    # Build a deterministic list [(label, direction, weight)]
    bundle = []
    if block_name == "china":
        bundle = [
            ("OECD China CLI (amplitude adj., SA)", "negative_is_risky", float(weights.get("cli", 0.60))),
            ("USD/CNY (derived, ECB monthly)", "positive_is_risky", float(weights.get("usdcny", 0.40))),
        ]
    elif block_name == "japan":
        bundle = [
            ("OECD Japan CLI (amplitude adj., SA)", "negative_is_risky", float(weights.get("cli", 0.60))),
            ("USD/JPY (derived, ECB monthly)", "positive_is_risky", float(weights.get("usdjpy", 0.40))),
        ]
    elif block_name == "eurozone":
        bundle = [
            ("OECD Euro Area CLI (amplitude adj., SA)", "negative_is_risky", float(weights.get("cli", 0.60))),
            ("ECB USD per EUR (monthly)", "positive_is_risky", float(weights.get("usdeur", 0.40))),
        ]
    elif block_name == "uk":
        bundle = [
            ("OECD UK CLI (amplitude adj., SA)", "negative_is_risky", float(weights.get("cli", 0.60))),
            ("USD/GBP (derived, ECB monthly)", "positive_is_risky", float(weights.get("usdgbp", 0.40))),
        ]
    elif block_name == "canada":
        bundle = [
            ("OECD Canada CLI (amplitude adj., SA)", "negative_is_risky", float(weights.get("cli", 0.60))),
            ("USD/CAD (derived, ECB monthly)", "positive_is_risky", float(weights.get("usdcad", 0.40))),
        ]

    for lbl, direction, w in bundle:
        if lbl not in series_map:
            continue
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
        scoring.append((score, w))

    sub = np.nan
    if scoring:
        tw = sum(w for _, w in scoring)
        if tw > 0:
            sub = sum(sc * w for sc, w in scoring) / tw

    level = "OK"
    if not np.isnan(sub):
        if sub >= TH_CHINA_ALERT:
            level = "ALERT"
        elif sub >= TH_CHINA_WATCH:
            level = "WATCH"

    badge = "<span class='badge ok'>OK</span>"
    if level == "WATCH": badge = "<span class='badge watch'>WATCH</span>"
    if level == "ALERT": badge = "<span class='badge alert'>ALERT</span>"
    sub_str = "-" if np.isnan(sub) else f"{sub:.2f}"
    html.append(f"<p><b>{title} Subscore:</b> {sub_str} {badge}</p>")

    # Table with sparkline + download
    html.append("<table><tr>"
                "<th>Indicator</th><th>Last Date</th><th>Last Value</th>"
                "<th>Z-Score</th><th>Sparkline (52w)</th><th>Download</th>"
                "</tr>")
    for lbl in items_labels:
        rows = [r for r in display_rows if r[0] == lbl]
        if not rows:
            continue
        r = rows[0]
        svg = spark_from_file(series_map[lbl], tail_points=52)
        fn = series_map[lbl]
        link = f"<a href='data/{fn}' download>CSV</a>"
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{svg}</td><td>{link}</td></tr>")
    html.append("</table>")

# ---- China / Japan / Eurozone / UK / Canada blocks (render only if present in config)
if "china" in CFG["indicators"]:
    render_block("china", "China", CFG["indicators"]["china"])
if "japan" in CFG["indicators"]:
    render_block("japan", "Japan", CFG["indicators"]["japan"])
if "eurozone" in CFG["indicators"]:
    render_block("eurozone", "Eurozone", CFG["indicators"]["eurozone"])
if "uk" in CFG["indicators"]:
    render_block("uk", "United Kingdom", CFG["indicators"]["uk"])
if "canada" in CFG["indicators"]:
    render_block("canada", "Canada", CFG["indicators"]["canada"])

html.append(
    "<p class='small'>Notes: Contributions use the same direction rules as the composite: "
    "curve inversion (more negative) = risk↑; HY OAS ↑ = risk↑; unemployment ↑ = risk↑; "
    "USD per EUR ↑ = risk↑; USD crosses (USD/CNY, USD/JPY, USD/GBP, USD/CAD) ↑ = risk↑; "
    "OECD CLIs (amplitude‑adjusted) below trend = risk↑. "
    "Edit <code>config.yaml</code> to adjust thresholds, lookback, indicators and weights.</p>"
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

