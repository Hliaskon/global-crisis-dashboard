
# -*- coding: utf-8 -*-
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

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "data").mkdir(parents=True, exist_ok=True)  # for download links

# ---------------------------------------------------------------------
# Config & defaults
# ---------------------------------------------------------------------
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
    # These labels must match series_map keys (so they appear in the top table)
    "indicators": {
        "table_us_ecb": [
            "US Yield Curve 10Y–3M (bps)",
            "US Unemployment Rate (%)",
            "US HY OAS (bps)",
            "USD per EUR (ECB)",        # EURUSD (per EUR)
            "USD/JPY (derived, ECB)",
            "USD/CNY (derived, ECB)",
            "USD/GBP (derived, ECB)",
            "USD/CAD (derived, ECB)",
            "USD/INR (derived, ECB)",
            "USD/RUB (derived, ECB)",
        ],
    },
    "countries": []  # to be filled by config.yaml
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
        # list key for countries
        if "countries" in user_cfg and isinstance(user_cfg["countries"], list):
            CFG["countries"] = user_cfg["countries"]
    except Exception as e:
        print(f"⚠️ Failed to read config.yaml; using defaults. Error: {e}")

LOOKBACK = int(CFG["windows"]["zscore_lookback_weeks"])
TH_COMP_WATCH = float(CFG["thresholds"]["composite_watch"])
TH_COMP_ALERT = float(CFG["thresholds"]["composite_alert"])
TH_CT_WATCH = float(CFG["thresholds"]["country_watch"])
TH_CT_ALERT = float(CFG["thresholds"]["country_alert"])
W_DEF_CLI = float(CFG["weights"]["default"]["cli"])
W_DEF_FX  = float(CFG["weights"]["default"]["fx"])

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_series(csv_path: Path, weekly_resample: bool = True) -> pd.Series:
    """
    Load a CSV with columns [date, value] into a pandas Series (float) indexed by datetime.
    Optionally resample to weekly (Friday) with last observation and interpolate.
    """
    if not csv_path.exists():
        return pd.Series(dtype="float64")

    df = pd.read_csv(csv_path)
    if not {"date", "value"}.issubset(set(df.columns)):
        # fallback: take first two columns as date, value
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "date", cols[1]: "value"})
        else:
            return pd.Series(dtype="float64")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    s = df.set_index("date")["value"]
    if weekly_resample:
        s = s.resample("W-FRI").last().interpolate()
    return s


def zscore(s: pd.Series) -> pd.Series:
    """Rolling z-score with a default long lookback (weeks)."""
    if len(s) < 10:
        return s * np.nan
    roll = max(10, LOOKBACK)
    mu = s.rolling(roll, min_periods=max(26, roll // 10)).mean()
    sd = s.rolling(roll, min_periods=max(26, roll // 10)).std()
    return (s - mu) / sd


def sparkline_svg(values, width: int = 120, height: int = 26, stroke: str = "#1f77b4") -> str:
    """Mini inline SVG sparkline for the last N values."""
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


def series_fingerprint(s: pd.Series, tail_points: int = 52, precision: int = 6):
    """
    Build a small, hashable fingerprint of the last N weekly values.
    Rounded to avoid float noise; returns tuple() if empty.
    """
    if s is None or len(s) == 0:
        return ()
    tail = s.tail(tail_points).astype(float).round(precision)
    tail = tail[pd.notna(tail)]
    return tuple(tail.values.tolist())


def direction_for_label(label: str):
    """
    Map an indicator label to a direction rule and sign:
      - 'negative_is_risky'  -> multiply z by -1
      - 'positive_is_risky'  -> multiply z by +1
    """
    if "Yield Curve" in label:
        return "negative_is_risky", -1
    if "HY OAS" in label:
        return "positive_is_risky", +1
    if "Unemployment" in label:
        return "positive_is_risky", +1
    if "USD per EUR" in label:
        return "positive_is_risky", +1  # USD strengthens -> risk ↑
    if label.startswith("USD/"):
        return "positive_is_risky", +1  # USD/X ↑ -> risk ↑
    if "CLI" in label:
        return "negative_is_risky", -1
    return "ambiguous", +1


def human_direction(dir_name: str) -> str:
    """Human-readable direction text for UI."""
    if dir_name == "negative_is_risky":
        return "Below trend → higher risk"
    if dir_name == "positive_is_risky":
        return "Rising → higher risk"
    return "Context‑dependent"

# ---------------------------------------------------------------------
# Series mapping (labels -> files)
# ---------------------------------------------------------------------
series_map = {
    # US macro (FRED)
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)": "unemployment_rate.csv",
    "US HY OAS (bps)": "hy_credit_spread.csv",

    # FX from ECB (we derived USD per currency where applicable)
    "USD per EUR (ECB)": "eur_usd_ecb.csv",
    "USD/JPY (derived, ECB)": "usd_jpy_ecb.csv",
    "USD/CNY (derived, ECB)": "usd_cny_ecb.csv",
    "USD/GBP (derived, ECB)": "usd_gbp_ecb.csv",
    "USD/CAD (derived, ECB)": "usd_cad_ecb.csv",
    "USD/INR (derived, ECB)": "usd_inr_ecb.csv",
    "USD/RUB (derived, ECB)": "usd_rub_ecb.csv",
}
# Country CLI rows will be appended based on config (files like <country>_cli_oecd.csv)

# ---------------------------------------------------------------------
# Build display rows & contributor list
# ---------------------------------------------------------------------
display_rows = []     # [(label, last_date, last_val_str, last_z_str)]
contrib_rows  = []    # [(label, last_z, dir_name, contribution)]
z_values = []         # list of contributions (for composite)

# 1) Add series from series_map
for label, fname in series_map.items():
    path = DATA_DIR / fname
    s = load_series(path, weekly_resample=True)
    if s.empty:
        continue
    zs = zscore(s)
    last_date = s.index[-1].strftime("%Y-%m-%d")
    last_val  = s.iloc[-1]
    last_z    = zs.iloc[-1] if len(zs) else np.nan

    display_rows.append((
        label,
        last_date,
        f"{last_val:.2f}",
        "-" if np.isnan(last_z) else f"{float(last_z):+.2f}"
    ))

    if not np.isnan(last_z):
        dname, sign = direction_for_label(label)
        contribution = sign * float(last_z)
        contrib_rows.append((label, float(last_z), dname, contribution))
        z_values.append(contribution)

# 2) Append country CLIs to display/contributors
for c in CFG["countries"]:
    cli_label = c["cli_label"]
    safe = (c["name"].lower().replace(" ", "_"))
    cli_file = DATA_DIR / f"{safe}_cli_oecd.csv"

    s = load_series(cli_file, weekly_resample=True)
    if s.empty:
        # still show a placeholder row (with '-') in the country section later
        continue

    zs = zscore(s)
    last_date = s.index[-1].strftime("%Y-%m-%d")
    last_val  = s.iloc[-1]
    last_z    = zs.iloc[-1] if len(zs) else np.nan

    display_rows.append((
        cli_label,
        last_date,
        f"{last_val:.2f}",
        "-" if np.isnan(last_z) else f"{float(last_z):+.2f}"
    ))

    if not np.isnan(last_z):
        dname, sign = direction_for_label(cli_label)
        contribution = sign * float(last_z)
        contrib_rows.append((cli_label, float(last_z), dname, contribution))
        z_values.append(contribution)

# Composite (mean of contributions)
composite = np.nan if not z_values else float(np.mean(z_values))
level = "OK"
if not np.isnan(composite):
    if composite >= TH_COMP_ALERT:
        level = "ALERT"
    elif composite >= TH_COMP_WATCH:
        level = "WATCH"

# Rank contributors by absolute contribution
contrib_clean = [(l, z, d, c) for (l, z, d, c) in contrib_rows if not np.isnan(c)]
abs_sum = sum(abs(c) for (_, _, _, c) in contrib_clean) or 0.0
contributors_sorted = sorted(contrib_clean, key=lambda t: abs(t[3]), reverse=True)

# ---------------------------------------------------------------------
# Country subscores (+ integrity checks)
# ---------------------------------------------------------------------
country_states = []  # for HTML & state.json

# registries for duplicate detection
seen_cli_fingerprints = {}  # name -> fingerprint
seen_fx_fingerprints  = {}  # name -> fingerprint

for c in CFG["countries"]:
    name      = c["name"]
    cli_label = c["cli_label"]
    fx_label  = c["fx_label"]
    fx_file   = c["fx_source"]

    w_cli = float(c.get("weights", {}).get("cli", W_DEF_CLI))
    w_fx  = float(c.get("weights", {}).get("fx",  W_DEF_FX))

    safe = (name.lower().replace(" ", "_"))
    cli_path = DATA_DIR / f"{safe}_cli_oecd.csv"
    fx_path  = DATA_DIR / fx_file

    s_cli = load_series(cli_path, weekly_resample=True)
    s_fx  = load_series(fx_path,  weekly_resample=True)

    # ---- Integrity checks (non-blocking; warnings only) ----
    # (a) CLI duplicate detection
    try:
        cli_fp = series_fingerprint(s_cli, tail_points=52, precision=6)
        if cli_fp:
            dup = next((other for other, other_fp in seen_cli_fingerprints.items() if other_fp == cli_fp), None)
            if dup and dup != name:
                print(f"⚠️ Data check: {name} CLI appears IDENTICAL to {dup} over the last 52 points. "
                      f"Confirm file mapping (cli_path={cli_path}).")
            else:
                seen_cli_fingerprints[name] = cli_fp
        else:
            print(f"⚠️ Data check: {name} CLI has no usable tail (empty or NaN). File: {cli_path}")
    except Exception as e:
        print(f"⚠️ Data check (CLI) failed for {name}: {e}. File: {cli_path}")

    # (b) FX presence & duplicate detection
    try:
        if s_fx is None or len(s_fx) == 0:
            print(f"⚠️ FX data missing/empty for {name}. Expected file: {fx_path}")
        else:
            fx_fp = series_fingerprint(s_fx, tail_points=52, precision=6)
            if fx_fp:
                dup_fx = next((other for other, other_fp in seen_fx_fingerprints.items() if other_fp == fx_fp), None)
                if dup_fx and dup_fx != name:
                    print(f"⚠️ Data check: {name} FX series appears IDENTICAL to {dup_fx} over the last 52 points. "
                          f"Confirm file mapping (fx_path={fx_path}).")
                else:
                    seen_fx_fingerprints[name] = fx_fp
            else:
                print(f"⚠️ FX series for {name} has no usable tail (empty or NaN). File: {fx_path}")
    except Exception as e:
        print(f"⚠️ Data check (FX) failed for {name}: {e}. File: {fx_path}")

    # ---- Subscore calculation ----
    z_cli = zscore(s_cli)
    z_fx  = zscore(s_fx)

    z_last_cli = z_cli.iloc[-1] if len(z_cli) else np.nan
    z_last_fx  = z_fx.iloc[-1]  if len(z_fx)  else np.nan

    # Apply direction mapping for subscores:
    # CLI: negative_is_risky -> use -z
    score_cli = -float(z_last_cli) if not np.isnan(z_last_cli) else np.nan
    # FX:  positive_is_risky -> use +z (USD strength as stress)
    score_fx  =  float(z_last_fx)  if not np.isnan(z_last_fx)  else np.nan

    parts = []
    if not np.isnan(score_cli):
        parts.append((score_cli, w_cli))
    if not np.isnan(score_fx):
        parts.append((score_fx,  w_fx))

    subscore = np.nan
    if parts:
        wsum = sum(w for (_, w) in parts)
        if wsum > 0:
            subscore = sum(s * w for (s, w) in parts) / wsum

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
        "cli_label": cli_label,
        "fx_label": fx_label,
        "cli_file": f"{safe}_cli_oecd.csv",
        "fx_file": fx_file,
    })

# ---------------------------------------------------------------------
# Δ vs previous run (for Contributors)
# ---------------------------------------------------------------------
prev_state_path = OUT_DIR / "prev_state.json"
prev_contrib = {}
if prev_state_path.exists():
    try:
        prev_state = json.loads(prev_state_path.read_text(encoding="utf-8"))
        prev_contrib = prev_state.get("contributions", {}) or {}
    except Exception as e:
        print(f"⚠️ Could not parse prev_state.json: {e}")

current_contrib = {lbl: float(contrib) for (lbl, _z, _d, contrib) in contrib_rows}

# ---------------------------------------------------------------------
# Render HTML
# ---------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
html = []
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Global Crisis Early Warning – Dashboard</title>")
html.append("""
<style>
  body{font-family:Arial,Helvetica,sans-serif;margin:24px;}
  table{border-collapse:collapse;margin-top:12px;}
  th,td{border:1px solid #ddd;padding:6px 10px;vertical-align:middle;}
  th{background:#f4f6f8;text-align:left;}
  .badge{display:inline-block;padding:6px 10px;border-radius:6px;margin-left:8px;}
  .ok{background:#e8f5e9;}
  .watch{background:#fff8e1;}
  .alert{background:#ffebee;}
  .pos{color:#b71c1c;}   /* positive contribution pushes risk up */
  .neg{color:#1b5e20;}   /* negative contribution pulls risk down */
  .small{color:#666;font-size:12px}
  td svg{display:block}

  /* --- Layout & spacing --- */
  h2 { margin-top: 6px; }
  h3 { margin-top: 28px; }
  .section { margin-top: 32px; padding-top: 16px; border-top: 2px solid #eee; }
  h3.section { color: #333; }

  /* --- Country cards --- */
  .country { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-top: 18px; background: #ffffff; }
  .country h3 { margin-top: 0; }

  /* --- Typography --- */
  .subscore { font-size: 18px; }
  .dim { color: #999; font-style: italic; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  caption { text-align: left; font-weight: bold; padding-bottom: 6px; }

  /* --- Composite hero --- */
  .hero {
    border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px 16px; margin: 10px 0 14px 0;
    background: #fafafa; display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  }
  .hero .big { font-size: 36px; line-height: 1.0; font-weight: 600; }
  .hero .meta { font-size: 12px; color:#666; }
</style>
""")
html.append("</head><body>")

html.append("<h1>Global Crisis Early Warning – Dashboard</h1>")
html.append(f"<p class='small'>Generated: {now}</p>")

# Badge & value for hero (define BEFORE using them)
badge = "<span class='badge ok'>OK</span>"
if level == "WATCH":
    badge = "<span class='badge watch'>WATCH</span>"
elif level == "ALERT":
    badge = "<span class='badge alert'>ALERT</span>"

comp_str = "-" if np.isnan(composite) else f"{composite:.2f}"

# Composite hero block
html.append(
    "<div class='hero'>"
    f"<div class='big'>{comp_str}</div>"
    f"<div>{badge}</div>"
    "<div class='meta'>Composite Risk (mean of sign‑adjusted z‑scores). "
    f"Thresholds — WATCH ≥ {TH_COMP_WATCH:.1f}σ, ALERT ≥ {TH_COMP_ALERT:.1f}σ.</div>"
    "</div>"
)

# Contributors (with Δ vs prev)
html.append("<h3 class='section'>Contributors (sign‑adjusted Z)</h3>")
html.append("<table><tr><th>Indicator</th><th>Z‑score</th><th>Direction</th><th>Contribution</th><th>Δ vs prev</th><th>Share</th></tr>")
if contributors_sorted:
    for (lbl, z, dname, c) in contributors_sorted:
        cls = "pos" if c >= 0 else "neg"
        prev_c = float(prev_contrib.get(lbl, 0.0)) if prev_contrib else 0.0
        delta = c - prev_c
        cls_d = "pos" if delta >= 0 else "neg"
        share = "-" if abs_sum == 0 else f"{(abs(c)/abs_sum)*100:.0f}%"
        html.append(
            f"<tr>"
            f"<td>{lbl}</td>"
            f"<td class='num'>{float(z):+.2f}</td>"
            f"<td>{human_direction(dname)}</td>"
            f"<td class='{cls}'>{float(c):+.2f}</td>"
            f"<td class='{cls_d}'>{float(delta):+.2f}</td>"
            f"<td class='num'>{share}</td>"
            f"</tr>"
        )
else:
    html.append("<tr><td colspan='6'>No contributions available.</td></tr>")
html.append("</table>")

# Key indicators (US + ECB / global FX)
labels_us = CFG["indicators"]["table_us_ecb"]
html.append("<h3 class='section'>Key Indicators (US + ECB)</h3>")
html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z‑Score</th></tr>")
for r in display_rows:
    if r[0] in labels_us:
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td class='num'>{r[2]}</td><td class='num'>{r[3]}</td></tr>")
html.append("</table>")

# Country cards
for c in country_states:
    name = c["name"]
    sub  = c["subscore"]
    lvl  = c["level"]

    badge_cn = "<span class='badge ok'>OK</span>"
    if lvl == "WATCH":
        badge_cn = "<span class='badge watch'>WATCH</span>"
    elif lvl == "ALERT":
        badge_cn = "<span class='badge alert'>ALERT</span>"

    sub_str = "-" if sub is None else f"{sub:.2f}"

    # find rows for cli & fx (exact label match in display_rows)
    cli_row = next((r for r in display_rows if r[0] == c["cli_label"]), None)
    fx_row  = next((r for r in display_rows if r[0] == c["fx_label"]),  None)

    def spark_from_file(file_name: str, tail_points: int = 52) -> str:
        path = DATA_DIR / file_name
        s = load_series(path, weekly_resample=True)
        if s.empty:
            return ""
        return sparkline_svg(s.tail(tail_points).tolist())

    html.append("<div class='country'>")
    html.append(f"<h3>{name}</h3>")
    html.append(f"<p class='subscore'><b>Subscore:</b> {sub_str} {badge_cn}</p>")
    html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z‑Score</th><th>Sparkline (52w)</th><th>Download</th></tr>")

    # CLI row
    if cli_row:
        cli_svg = spark_from_file(c["cli_file"])
        html.append(
            f"<tr>"
            f"<td>{cli_row[0]}</td><td>{cli_row[1]}</td>"
            f"<td class='num'>{cli_row[2]}</td><td class='num'>{cli_row[3]}</td>"
            f"<td>{cli_svg}</td><td><a href='data/{c['cli_file']}'>CSV</a></td>"
            f"</tr>"
        )
    else:
        html.append(
            f"<tr><td>{c['cli_label']}</td><td>-</td><td>-</td><td>-</td>"
            f"<td></td><td class='dim'>data unavailable</td></tr>"
        )

    # FX row
    if fx_row:
        fx_svg = spark_from_file(c["fx_file"])
        html.append(
            f"<tr>"
            f"<td>{fx_row[0]}</td><td>{fx_row[1]}</td>"
            f"<td class='num'>{fx_row[2]}</td><td class='num'>{fx_row[3]}</td>"
            f"<td>{fx_svg}</td><td><a href='data/{c['fx_file']}'>CSV</a></td>"
            f"</tr>"
        )
    else:
        html.append(
            f"<tr><td>{c['fx_label']}</td><td>-</td><td>-</td><td>-</td>"
            f"<td></td><td class='dim'>data unavailable</td></tr>"
        )

    html.append("</table>")
    html.append("</div>")  # /.country

# Footnote
html.append(
    "<p class='small'>Notes: Z‑scores use a rolling window. Direction mapping — "
    "curve inversion (more negative) = risk↑; HY OAS ↑ = risk↑; unemployment ↑ = risk↑; "
    "USD strength (USD/X ↑ or USD per EUR ↑) = risk↑; CLI below trend = risk↑. "
    "Tune thresholds, lookback and weights in <code>config.yaml</code>.</p>"
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
