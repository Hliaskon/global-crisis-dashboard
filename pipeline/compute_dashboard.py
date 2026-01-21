# pipeline/compute_dashboard.py
# Step 1 – visual/layout improvements only (CSS tweaks, country cards, section separators, nicer missing-FX rows)

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

# Country CLI labels will be added dynamically from config with filenames like <country>_cli_oecd.csv
def cli_filename_from_label(label: str) -> str:
    # Simple normalized key from label
    key = (label.lower()
                 .replace("oecd", "")
                 .replace("cli", "")
                 .replace("(amplitude adj., sa)", "")
                 .replace("(", "").replace(")", "")
                 .replace(" ", "").replace(",", "").replace("–","-"))
    return f"{key}_cli_oecd.csv"  # will be overridden by workflow names

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
    if label.startswith("USD/"):     # USD/XXX ↑ => risk↑
        return "positive_is_risky", +1
    if "CLI" in label:
        return "negative_is_risky", -1
    return "ambiguous", +1

# ---------------------
# Build readings & composite
# ---------------------
display_rows = []          # [(label, last_date, last_val, last_z)]
contrib_rows  = []         # [(label, last_z, direction, contribution)]
z_values      = []         # list of contributions

# 1) US + ECB table items
for label, fname in series_map.items():
    # we'll filter by config later for display; compute contributions for all we actually have
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
        dir_name, sign = direction_for_label(label)
        contribution = sign * float(last_z)
        contrib_rows.append((label, float(last_z), dir_name, contribution))
        z_values.append(contribution)

# 2) Country CLIs (from config)
for c in CFG["countries"]:
    cli_label = c["cli_label"]
    # filename follows workflow outputs (passed as args there)
    # We infer based on common convention: <name> lower, no spaces -> *_cli_oecd.csv
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
