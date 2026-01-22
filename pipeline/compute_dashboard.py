
# -*- coding: utf-8 -*-
"""
Global Crisis Dashboard — risk-normalized composite with breadth & momentum filters

Implements your proposals:
  A) risk_score = +Z (bad_when_high) / -Z (bad_when_low)
  B) Weights normalized to 100%; reduce CLI redundancy via breadth metrics
  C) Trend/acceleration detector with persistence (YELLOW/RED)
  D) Extensible indicator registry via config.yaml
  E) Smoothing (3M MA) + standard/robust rolling Z
  F) Persist composite history for backtesting
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Optional YAML config
try:
    import yaml
except Exception:
    yaml = None

# ------------------------------- Paths --------------------------------
DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "data").mkdir(parents=True, exist_ok=True)

# ---------------------------- Defaults --------------------------------
DEFAULTS = {
    "windows": {
        "zscore_lookback_weeks": 260,   # ~5 years
        "smoothing_ma_weeks": 13,       # ~3 months pre-smoothing
        "three_month_weeks": 13,        # 3M deltas (weekly cadence)
        "persistence_weeks": 9          # ~2 months persistence
    },
    "zscore": {
        "method": "standard"            # "standard" | "robust"
    },
    "thresholds": {
        "composite_watch": 1.0,
        "composite_alert": 2.0,
        "yellow_delta": 0.3,
        "red_delta": 0.7,
        "red_level": 1.0
    },
    # For the “Key Indicators (US + ECB)” summary table (informational)
    "indicators_table_us_ecb": [
        "US Yield Curve 10Y–3M (bps)",
        "US Unemployment Rate (%)",
        "US HY OAS (bps)",
        "USD per EUR (ECB)",
        "USD/JPY (derived, ECB)",
        "USD/CNY (derived, ECB)",
        "USD/GBP (derived, ECB)",
        "USD/CAD (derived, ECB)",
        "USD/INR (derived, ECB)",
        "USD/RUB (derived, ECB)"
    ],
    # Breadth metrics from all country CLIs
    "breadth": {
        "enable": True,
        "total_weight": 0.12,           # 12% total (e.g., 6% level + 6% momentum)
        "share_level_vs_mom": [0.5, 0.5],
        "cli_level_threshold": 100.0
    },
    # Core indicators (you can extend via config.yaml -> indicators_custom)
    "indicators_core": [
        {"label": "US HY OAS (bps)",             "file": "hy_credit_spread.csv",   "bad_when": "high", "weight": 0.25},
        {"label": "US Yield Curve 10Y–3M (bps)", "file": "yield_curve_10y_3m.csv", "bad_when": "low",  "weight": 0.10},
        {"label": "US Unemployment Rate (%)",    "file": "unemployment_rate.csv",  "bad_when": "high", "weight": 0.10}
    ],
    # Country definitions (used both for country cards and CLI breadth)
    "countries": []
}

CFG = DEFAULTS.copy()
cfg_path = Path("config.yaml")
if yaml is not None and cfg_path.exists():
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # shallow/deep merge
        for k, v in DEFAULTS.items():
            if isinstance(v, dict):
                CFG[k] = {**v, **(user_cfg.get(k, {}) or {})}
            else:
                CFG[k] = user_cfg.get(k, v)
        # list overrides
        if isinstance(user_cfg.get("countries"), list):
            CFG["countries"] = user_cfg["countries"]
        if isinstance(user_cfg.get("indicators_core"), list):
            CFG["indicators_core"] = user_cfg["indicators_core"]
        if isinstance(user_cfg.get("indicators_custom"), list):
            CFG["indicators_core"] += user_cfg["indicators_custom"]
    except Exception as e:
        print(f"⚠️ Failed to read config.yaml; using defaults. Error: {e}")

LOOKBACK_W = int(CFG["windows"]["zscore_lookback_weeks"])
SMOOTH_W   = int(CFG["windows"]["smoothing_ma_weeks"])
DELTA3M_W  = int(CFG["windows"]["three_month_weeks"])
PERSIST_W  = int(CFG["windows"]["persistence_weeks"])
Z_METHOD   = (CFG.get("zscore", {}) or {}).get("method", "standard").lower()

TH_COMP_WATCH = float(CFG["thresholds"]["composite_watch"])
TH_COMP_ALERT = float(CFG["thresholds"]["composite_alert"])
TH_YELLOW_D   = float(CFG["thresholds"]["yellow_delta"])
TH_RED_D      = float(CFG["thresholds"]["red_delta"])
TH_RED_LVL    = float(CFG["thresholds"]["red_level"])

BREADTH_ON      = bool(CFG["breadth"].get("enable", True))
BREADTH_W_TOTAL = float(CFG["breadth"]["total_weight"])
BREADTH_SPLIT   = CFG["breadth"]["share_level_vs_mom"]
CLI_LEVEL_TH    = float(CFG["breadth"]["cli_level_threshold"])

# ----------------------------- Helpers --------------------------------
def load_series(csv_path: Path, weekly_resample: bool = True) -> pd.Series:
    """Load [date,value] CSV to Series. Weekly resample to Friday if requested."""
    if not csv_path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_csv(csv_path)
    if not {"date", "value"}.issubset(df.columns):
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "date", cols[1]: "value"})
        else:
            return pd.Series(dtype="float64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    s = df.set_index("date")["value"].astype(float)
    if weekly_resample:
        s = s.resample("W-FRI").last().interpolate()
    return s

def smooth_ma(s: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(s) == 0:
        return s
    return s.rolling(window, min_periods=max(2, window // 2)).mean()

def rolling_standard_z(s: pd.Series, lookback: int) -> pd.Series:
    if len(s) < 10:
        return s * np.nan
    mu = s.rolling(lookback, min_periods=max(26, lookback // 10)).mean()
    sd = s.rolling(lookback, min_periods=max(26, lookback // 10)).std()
    return (s - mu) / sd

def rolling_robust_z(s: pd.Series, lookback: int) -> pd.Series:
    if len(s) < 10:
        return s * np.nan
    med = s.rolling(lookback, min_periods=max(26, lookback // 10)).median()
    mad = s.rolling(lookback, min_periods=max(26, lookback // 10)) \
           .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    mad = mad.replace(0.0, np.nan)
    return (s - med) / (mad * 1.4826)

def roll_z(s: pd.Series) -> pd.Series:
    s2 = smooth_ma(s, SMOOTH_W)
    return rolling_robust_z(s2, LOOKBACK_W) if Z_METHOD == "robust" else rolling_standard_z(s2, LOOKBACK_W)

def series_fingerprint(s: pd.Series, tail_points: int = 52, precision: int = 6):
    if s is None or len(s) == 0:
        return ()
    tail = s.tail(tail_points).astype(float).round(precision)
    tail = tail[pd.notna(tail)]
    return tuple(tail.values.tolist())

def human_momentum_label(bad_when: str, z_delta_3m: float) -> str:
    if np.isnan(z_delta_3m):
        return "—"
    if bad_when == "high":
        return "Rising → higher risk" if z_delta_3m > 0 else "Falling → lower risk"
    if bad_when == "low":
        return "Falling → higher risk" if z_delta_3m < 0 else "Rising → lower risk"
    return "Context‑dependent"

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
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<path d='{path}' fill='none' stroke='{stroke}' stroke-width='1.5' /></svg>"
    )

# ----------------------- Key Indicators table -------------------------
SERIES_MAP_KEYS = {
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)":    "unemployment_rate.csv",
    "US HY OAS (bps)":             "hy_credit_spread.csv",
    "USD per EUR (ECB)":           "eur_usd_ecb.csv",
    "USD/JPY (derived, ECB)": "usd_jpy_ecb.csv",
    "USD/CNY (derived, ECB)": "usd_cny_ecb.csv",
    "USD/GBP (derived, ECB)": "usd_gbp_ecb.csv",
    "USD/CAD (derived, ECB)": "usd_cad_ecb.csv",
    "USD/INR (derived, ECB)": "usd_inr_ecb.csv",
    "USD/RUB (derived, ECB)": "usd_rub_ecb.csv"
}

display_rows = []  # [(label, date, last_val_str, last_z_str)]
for label, fname in SERIES_MAP_KEYS.items():
    s = load_series(DATA_DIR / fname, weekly_resample=True)
    if s.empty:
        continue
    z = roll_z(s)
    last_date = s.index[-1].strftime("%Y-%m-%d")
    last_val  = s.iloc[-1]
    last_z    = z.iloc[-1] if len(z) else np.nan
    display_rows.append((label, last_date, f"{last_val:.2f}", "-" if np.isnan(last_z) else f"{float(last_z):+.2f}"))

# ------------------ Country subscores + CLI breadth -------------------
country_states = []   # each: dict with computed stats for HTML + state.json
seen_cli_fp, seen_fx_fp = {}, {}
cli_level_flags, cli_mom_flags = [], []

for c in CFG["countries"]:
    name      = c["name"]
    cli_label = c["cli_label"]
    fx_label  = c["fx_label"]
    fx_file   = c["fx_source"]
    w_cli     = float(c.get("weights", {}).get("cli", 0.60))
    w_fx      = float(c.get("weights", {}).get("fx",  0.40))

    safe = name.lower().replace(" ", "_")
    cli_path = DATA_DIR / f"{safe}_cli_oecd.csv"
    fx_path  = DATA_DIR / fx_file

    s_cli = load_series(cli_path, weekly_resample=True)
    s_fx  = load_series(fx_path,  weekly_resample=True)

    # Non-blocking integrity notices
    try:
        fp = series_fingerprint(s_cli)
        if fp:
            dup = next((o for o, f in seen_cli_fp.items() if f == fp), None)
            if dup and dup != name:
                print(f"⚠️ Data check: {name} CLI identical to {dup} over last 52 points. Check {cli_path}.")
            else:
                seen_cli_fp[name] = fp
        else:
            print(f"⚠️ Data check: {name} CLI empty/NaN tail. File: {cli_path}")
    except Exception as e:
        print(f"⚠️ Data check (CLI) failed for {name}: {e} | {cli_path}")

    try:
        if s_fx is None or len(s_fx) == 0:
            print(f"⚠️ FX missing for {name}. Expected: {fx_path}")
        else:
            fp_fx = series_fingerprint(s_fx)
            if fp_fx:
                dup_fx = next((o for o, f in seen_fx_fp.items() if f == fp_fx), None)
                if dup_fx and dup_fx != name:
                    print(f"⚠️ Data check: {name} FX identical to {dup_fx} over last 52 points. Check {fx_path}.")
                else:
                    seen_fx_fp[name] = fp_fx
    except Exception as e:
        print(f"⚠️ Data check (FX) failed for {name}: {e} | {fx_path}")

    # Subscore: CLI bad_when_low -> -Z ; FX bad_when_high -> +Z
    z_cli = roll_z(s_cli)
    z_fx  = roll_z(s_fx)
    z_last_cli = z_cli.iloc[-1] if len(z_cli) else np.nan
    z_last_fx  = z_fx.iloc[-1]  if len(z_fx)  else np.nan

    score_cli = -float(z_last_cli) if not np.isnan(z_last_cli) else np.nan
    score_fx  =  float(z_last_fx)  if not np.isnan(z_last_fx)  else np.nan

    parts = []
    if not np.isnan(score_cli): parts.append((score_cli, w_cli))
    if not np.isnan(score_fx):  parts.append((score_fx,  w_fx))

    subscore = np.nan
    if parts:
        wsum = sum(w for _, w in parts)
        if wsum > 0:
            subscore = sum(s*w for s, w in parts) / wsum

    lvl = "OK"
    if not np.isnan(subscore):
        if subscore >= TH_COMP_ALERT: lvl = "ALERT"
        elif subscore >= TH_COMP_WATCH: lvl = "WATCH"

    # Breadth flags
    try:
        if len(s_cli) >= DELTA3M_W + 1:
            d3 = float(s_cli.iloc[-1] - s_cli.iloc[-DELTA3M_W])   # raw CLI change
            cli_level_flags.append(float(s_cli.iloc[-1] < CLI_LEVEL_TH and d3 < 0))
        if len(z_cli) >= DELTA3M_W + 1:
            z_d3 = float(z_cli.iloc[-1] - z_cli.iloc[-DELTA3M_W]) # Z change
            cli_mom_flags.append(float(z_d3 < 0))
    except Exception as e:
        print(f"⚠️ Breadth calc failed for {name}: {e}")

    # Prepare per-country stats for HTML (avoid fragile lookups)
    def last_stats(s: pd.Series, z: pd.Series):
        if s is None or len(s) == 0:
            return {"date": "-", "val": "-", "z": "-"}
        d = s.index[-1].strftime("%Y-%m-%d")
        v = f"{float(s.iloc[-1]):.2f}"
        if z is None or len(z) == 0 or np.isnan(z.iloc[-1]):
            zs = "-"
        else:
            zs = f"{float(z.iloc[-1]):+.2f}"
        return {"date": d, "val": v, "z": zs}

    country_states.append({
        "name": name,
        "level": lvl,
        "subscore": None if np.isnan(subscore) else float(subscore),
        "cli_label": cli_label,
        "fx_label":  fx_label,
        "cli_file":  f"{safe}_cli_oecd.csv",
        "fx_file":   fx_file,
        "cli_stats": last_stats(s_cli, z_cli),
        "fx_stats":  last_stats(s_fx,  z_fx),
        "cli_spark": "" if s_cli is None or len(s_cli) == 0 else sparkline_svg(s_cli.tail(52).tolist()),
        "fx_spark":  "" if s_fx  is None or len(s_fx)  == 0 else sparkline_svg(s_fx.tail(52).tolist())
    })

# ------------------------ Indicator registry --------------------------
def _load_indicator_defs():
    defs = []
    for d in CFG["indicators_core"]:
        if not isinstance(d, dict):
            continue
        f = DATA_DIR / d.get("file", "__missing__.csv")
        if f.exists():
            defs.append({
                "label": d["label"],
                "file":  d["file"],
                "bad_when": d.get("bad_when", "high"),
                "weight": float(d.get("weight", 0.0))
            })
        else:
            print(f"ℹ️ Skipping indicator (file not found): {d.get('label')} -> {f}")
    return defs

indicator_defs = _load_indicator_defs()

# Add breadth as synthetic indicators (if enabled and we have flags)
if BREADTH_ON and len(cli_level_flags) > 0 and len(cli_mom_flags) > 0:
    try:
        w_level = BREADTH_W_TOTAL * float(BREADTH_SPLIT[0])
        w_mom   = BREADTH_W_TOTAL * float(BREADTH_SPLIT[1])
        p_level = float(np.mean(cli_level_flags))
        p_mom   = float(np.mean(cli_mom_flags))

        # Small constant weekly series to pass through z machinery
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=26, freq="W-FRI")
        s_level = pd.Series([p_level]*len(idx), index=idx, dtype=float)
        s_mom   = pd.Series([p_mom]*len(idx),   index=idx, dtype=float)
        z_level = roll_z(s_level)
        z_mom   = roll_z(s_mom)

        indicator_defs.append({
            "label": "CLI breadth: % below-100 & falling",
            "file":  "__synthetic_breadth_level__",
            "bad_when": "high",
            "weight": w_level,
            "_synthetic_series": s_level,
            "_z_series": z_level
        })
        indicator_defs.append({
            "label": "CLI breadth: % with falling 3M Z",
            "file":  "__synthetic_breadth_mom__",
            "bad_when": "high",
            "weight": w_mom,
            "_synthetic_series": s_mom,
            "_z_series": z_mom
        })
    except Exception as e:
        print(f"⚠️ Could not build breadth indicators: {e}")

# Normalize all indicator weights to 1.0
w_sum = sum(max(0.0, float(d.get("weight", 0.0))) for d in indicator_defs)
if w_sum <= 0:
    n = max(1, len(indicator_defs))
    for d in indicator_defs:
        d["weight"] = 1.0 / n
else:
    for d in indicator_defs:
        d["weight"] = float(d.get("weight", 0.0)) / w_sum

# ---------------- Build per-indicator series & contributions ----------
def build_indicator_series(d):
    """Return (z_series, risk_series, trend_series, last_values_dict)."""
    if "_z_series" in d and "_synthetic_series" in d:
        z = d["_z_series"]
        base = d["_synthetic_series"]
    else:
        base = load_series(DATA_DIR / d["file"], weekly_resample=True)
        z    = roll_z(base)
    risk = -z if d["bad_when"] == "low" else z

    if len(z) >= DELTA3M_W + 1:
        trend = z - z.shift(DELTA3M_W)
        last_trend = float(trend.iloc[-1])
    else:
        trend = z * np.nan
        last_trend = np.nan

    last_z    = float(z.iloc[-1]) if len(z) else np.nan
    last_risk = float(risk.iloc[-1]) if len(risk) else np.nan
    return z, risk, trend, {"z_last": last_z, "risk_last": last_risk, "trend_last": last_trend}

contrib_rows = []         # [(label, z_last, momentum_label, risk_last, weight, weighted_contrib)]
risk_series_dict = {}     # label -> (risk_series, weight)
weights_dict     = {}     # label -> weight

for d in indicator_defs:
    try:
        z, risk, trend, info = build_indicator_series(d)
        w = float(d["weight"])
        weights_dict[d["label"]] = w
        risk_series_dict[d["label"]] = (risk, w)

        momentum_txt = human_momentum_label(d["bad_when"], info["trend_last"])
        weighted_contrib = np.nan if np.isnan(info["risk_last"]) else w * info["risk_last"]

        contrib_rows.append((d["label"], info["z_last"], momentum_txt, info["risk_last"], w, weighted_contrib))
    except Exception as e:
        print(f"⚠️ Indicator calc failed for {d.get('label')}: {e}")

# Normalize composite through time when some indicators are missing
def build_composite_series(risk_series_dict, weights_dict):
    if not risk_series_dict:
        return pd.Series(dtype=float)
    risk_df = pd.DataFrame({k: v[0] * v[1] for k, v in risk_series_dict.items()})
    w_df    = pd.DataFrame({k: v for k, v in weights_dict.items()}, index=risk_df.index).reindex(risk_df.index)
    for col in risk_df.columns:
        w_df[col] = np.where(risk_df[col].notna(), w_df[col], np.nan)
    num = risk_df.sum(axis=1, skipna=True)
    den = w_df.sum(axis=1, skipna=True)
    return num / den

composite_series = build_composite_series(risk_series_dict, weights_dict)
composite = float(composite_series.dropna().iloc[-1]) if len(composite_series.dropna()) else np.nan

level = "OK"
if not np.isnan(composite):
    if composite >= TH_COMP_ALERT: level = "ALERT"
    elif composite >= TH_COMP_WATCH: level = "WATCH"

# Detector with persistence
def detector_level(comp_series: pd.Series):
    if comp_series is None or len(comp_series) < DELTA3M_W + 1:
        return "NA", np.nan
    comp_delta = float(comp_series.iloc[-1] - comp_series.iloc[-DELTA3M_W])

    yellow_cond = (comp_series > 0) & ((comp_series - comp_series.shift(DELTA3M_W)) > TH_YELLOW_D)
    red_cond    = ((comp_series > TH_RED_LVL) | ((comp_series - comp_series.shift(DELTA3M_W)) > TH_RED_D))

    def holds_persistently(mask: pd.Series) -> bool:
        if len(mask) < PERSIST_W:
            return False
        return bool(mask.tail(PERSIST_W).all())

    if holds_persistently(red_cond):    return "RED",    comp_delta
    if holds_persistently(yellow_cond): return "YELLOW", comp_delta
    return "OK", comp_delta

det_level, comp_delta_3m = detector_level(composite_series)

# Shares for contributors table (snapshot)
abs_sum = sum(abs(x[5]) for x in contrib_rows if x[5] is not None and not np.isnan(x[5])) or 0.0
contrib_rows_out = []
for (lbl, z_last, mom_txt, risk_last, w, w_contrib) in contrib_rows:
    share = "-" if abs_sum == 0 or w_contrib is None or np.isnan(w_contrib) else f"{(abs(w_contrib)/abs_sum)*100:.0f}%"
    contrib_rows_out.append((lbl, z_last, mom_txt, risk_last, w, w_contrib, share))


# ---------------------------------------------------------------------
# Opportunity Engine (per country) — builds opportunities.json
# ---------------------------------------------------------------------

def _last_delta(z: pd.Series, window: int = DELTA3M_W) -> float:
    """Return last 3M change of a z-scored series; NaN if not enough data."""
    if z is None or len(z) < window + 1:
        return np.nan
    return float(z.iloc[-1] - z.iloc[-window])

def _try_series(file_name: str) -> pd.Series:
    s = load_series(DATA_DIR / file_name, weekly_resample=True)
    return s if isinstance(s, pd.Series) else pd.Series(dtype=float)

# Optional global commodity proxies (added in Steps 4–5)
bdi    = _try_series("bdi_stooq.csv")
gold   = _try_series("gold_imf.csv")
copper = _try_series("copper_imf.csv")

bdi_z    = roll_z(bdi)    if len(bdi)    else pd.Series(dtype=float)
gold_z   = roll_z(gold)   if len(gold)   else pd.Series(dtype=float)
copper_z = roll_z(copper) if len(copper) else pd.Series(dtype=float)

bdi_d3    = _last_delta(bdi_z)    if len(bdi_z)    else np.nan
gold_d3   = _last_delta(gold_z)   if len(gold_z)   else np.nan
copper_d3 = _last_delta(copper_z) if len(copper_z) else np.nan

def _decide_equities(country_risk: float | None, cli_d3: float) -> tuple[str,str]:
    # BUY when regime not RED and local cycle is below-trend but improving.
    if det_level == "RED" or (country_risk is not None and country_risk >= 0.8):
        return "AVOID", "Macro RED or high local risk"
    if (country_risk is not None and country_risk <= -0.2) and (not np.isnan(cli_d3) and cli_d3 > 0):
        return "BUY", "Cycle improving & below trend"
    return "WATCH", "Neutral/mixed"

def _decide_bonds(country_risk: float | None) -> tuple[str,str]:
    # Add duration in RED or when local risk high; avoid when cycle strong.
    if det_level == "RED" or (country_risk is not None and country_risk >= 0.6):
        return "BUY", "Add duration in stress"
    if country_risk is not None and country_risk <= -0.4:
        return "AVOID", "Prefer risk assets when cycle is strong"
    return "WATCH", "Neutral duration"

def _decide_commodities() -> tuple[str,str]:
    # GOLD hedge if RED or gold momentum +; cyclical beta if BDI & Copper momentum +.
    if det_level == "RED" or (not np.isnan(gold_d3) and gold_d3 > 0.2):
        return "GOLD_HEDGE", "Risk-off or inflation hedge"
    if (not np.isnan(bdi_d3) and bdi_d3 > 0) and (not np.isnan(copper_d3) and copper_d3 > 0):
        return "CYCLICAL_BETA", "BDI & Copper momentum positive"
    return "NEUTRAL", "No clear commodity tilt"

def _decide_fx(fx_d3: float) -> tuple[str,str]:
    # USD hedge in RED or when USD/LCY is rising (USD strength).
    if det_level == "RED" or (not np.isnan(fx_d3) and fx_d3 > 0):
        return "USD_HEDGE", "USD strength / hedge"
    return "LOCAL", "No FX hedge needed"

opportunities = {"generated_utc": now, "detector": det_level, "countries": []}

for cs in country_states:
    name       = cs["name"]
    subscore   = cs["subscore"]  # positive => higher risk
    cli_file   = cs["cli_file"]
    fx_file    = cs["fx_file"]

    s_cli = load_series(DATA_DIR / cli_file, weekly_resample=True)
    s_fx  = load_series(DATA_DIR / fx_file,  weekly_resample=True)
    z_cli = roll_z(s_cli)
    z_fx  = roll_z(s_fx)

    cli_d3 = _last_delta(z_cli) if len(z_cli) else np.nan
    fx_d3  = _last_delta(z_fx)  if len(z_fx)  else np.nan

    eq_act, eq_note = _decide_equities(subscore, cli_d3)
    bd_act, bd_note = _decide_bonds(subscore)
    cm_act, cm_note = _decide_commodities()
    fx_act, fx_note = _decide_fx(fx_d3)

    opportunities["countries"].append({
        "country": name,
        "subscore": None if subscore is None else round(float(subscore), 3),
        "cli_delta_3m": None if np.isnan(cli_d3) else round(float(cli_d3), 3),
        "fx_delta_3m":  None if np.isnan(fx_d3)  else round(float(fx_d3), 3),
        "equities": {"action": eq_act, "note": eq_note},
        "bonds":    {"action": bd_act, "note": bd_note},
        "commods":  {"action": cm_act, "note": cm_note},
        "fx":       {"action": fx_act, "note": fx_note}
    })

# Persist opportunities for downstream (alerts/backtests/UI)
(OUT_DIR / "opportunities.json").write_text(json.dumps(opportunities, indent=2), encoding="utf-8")
print("✅ Wrote output/opportunities.json")



# ----------------------------- HTML -----------------------------------
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
  .pos{color:#b71c1c;}
  .neg{color:#1b5e20;}
  .small{color:#666;font-size:12px}
  td svg{display:block}
  h2 { margin-top: 6px; }
  h3 { margin-top: 28px; }
  .section { margin-top: 32px; padding-top: 16px; border-top: 2px solid #eee; }
  h3.section { color: #333; }
  .country { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-top: 18px; background: #ffffff; }
  .country h3 { margin-top: 0; }
  .subscore { font-size: 18px; }
  .dim { color: #999; font-style: italic; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  caption { text-align: left; font-weight: bold; padding-bottom: 6px; }
  .hero {
    border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px 16px; margin: 10px 0 14px 0;
    background: #fafafa; display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  }
  .hero .big { font-size: 36px; line-height: 1.0; font-weight: 600; }
  .hero .meta { font-size: 12px; color:#666; }
  .det { display:inline-block; padding:4px 8px; border-radius:6px; font-weight:600; }
  .det.ok { background:#e8f5e9; }
  .det.yellow { background:#fff8e1; }
  .det.red { background:#ffebee; }
</style>
""")
html.append("</head><body>")
html.append("<h1>Global Crisis Early Warning – Dashboard</h1>")
html.append(f"<p class='small'>Generated: {now}</p>")

badge = "<span class='badge ok'>OK</span>"
if level == "WATCH":
    badge = "<span class='badge watch'>WATCH</span>"
elif level == "ALERT":
    badge = "<span class='badge alert'>ALERT</span>"

comp_str = "-" if np.isnan(composite) else f"{composite:.2f}"
det_badge = {
    "OK": "<span class='det ok'>Detector: OK</span>",
    "YELLOW": "<span class='det yellow'>Detector: YELLOW</span>",
    "RED": "<span class='det red'>Detector: RED</span>",
    "NA": "<span class='det'>Detector: NA</span>"
}.get(det_level, "<span class='det'>Detector: NA</span>")

delta_str = "–" if np.isnan(comp_delta_3m) else f"{comp_delta_3m:+.2f}"
html.append(
    "<div class='hero'>"
    f"<div class='big'>{comp_str}</div>"
    f"<div>{badge}</div>"
    f"<div class='meta'>Composite Risk (sum of weights × risk‑Z). "
    f"Detector: {det_badge} &nbsp; | &nbsp; 3‑month Δ: <b>{delta_str}</b> "
    f"(YELLOW if > {TH_YELLOW_D:+.1f} with persistence; RED if level > {TH_RED_LVL:+.1f} or Δ > {TH_RED_D:+.1f}).</div>"
    "</div>"
)

# Contributors
html.append("<h3 class='section'>Contributors (risk‑normalized Z; momentum‑aware labels)</h3>")
html.append("<table><tr><th>Indicator</th><th>Z‑score</th><th>Momentum label</th><th>Risk score</th><th>Weight</th><th>Weighted</th><th>Share</th></tr>")
if contrib_rows_out:
    for (lbl, z_last, mom_txt, risk_last, w, w_contrib, share) in contrib_rows_out:
        z_txt  = "–" if np.isnan(z_last) else f"{float(z_last):+.2f}"
        r_txt  = "–" if np.isnan(risk_last) else f"{float(risk_last):+.2f}"
        wc_txt = "–" if (w_contrib is None or np.isnan(w_contrib)) else f"{float(w_contrib):+.2f}"
        cls    = "pos" if (w_contrib is not None and not np.isnan(w_contrib) and w_contrib >= 0) else "neg"
        html.append(
            f"<tr><td>{lbl}</td>"
            f"<td class='num'>{z_txt}</td>"
            f"<td>{mom_txt}</td>"
            f"<td class='num'>{r_txt}</td>"
            f"<td class='num'>{w*100:.0f}%</td>"
            f"<td class='{cls}'>{wc_txt}</td>"
            f"<td class='num'>{share}</td></tr>"
        )
else:
    html.append("<tr><td colspan='7'>No contributors available.</td></tr>")
html.append("</table>")

# Key indicators (US + ECB)
html.append("<h3 class='section'>Key Indicators (US + ECB)</h3>")
html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z‑Score</th></tr>")
for r in display_rows:
    if r[0] in CFG["indicators_table_us_ecb"]:
        html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td class='num'>{r[2]}</td><td class='num'>{r[3]}</td></tr>")
html.append("</table>")

# Country cards (now using precomputed per-country stats; fixed CSV links)
html.append("<h3 class='section'>Country Views</h3>")
html.append("<div class='small'>Each subscore = weighted sum of (−Z(CLI), +Z(FX)).</div>")

for c in country_states:
    name = c["name"]
    sub  = c["subscore"]
    lvl  = c["level"]
    badge_cn = "<span class='badge ok'>OK</span>" if lvl == "OK" else ("<span class='badge watch'>WATCH</span>" if lvl == "WATCH" else "<span class='badge alert'>ALERT</span>")
    sub_str = "-" if sub is None else f"{sub:.2f}"

    html.append("<div class='country'>")
    html.append(f"<h3>{name}</h3>")
    html.append(f"<p class='subscore'><b>Subscore:</b> {sub_str} {badge_cn}</p>")
    html.append("<table><tr><th>Indicator</th><th>Last Date</th><th>Last Value</th><th>Z‑Score</th><th>Sparkline (52w)</th><th>Download</th></tr>")

    # CLI row
    cli_stats = c["cli_stats"]
    cli_svg   = c["cli_spark"]
    html.append(
        f"<tr><td>{c['cli_label']}</td>"
        f"<td>{cli_stats['date']}</td>"
        f"<td class='num'>{cli_stats['val']}</td>"
        f"<td class='num'>{cli_stats['z']}</td>"
        f"<td>{cli_svg}</td>"
        f"<td><a href='data/{c['cli_file']}'>CSV</a></td></tr>"
    )

    # FX row
    fx_stats = c["fx_stats"]
    fx_svg   = c["fx_spark"]
    html.append(
        f"<tr><td>{c['fx_label']}</td>"
        f"<td>{fx_stats['date']}</td>"
        f"<td class='num'>{fx_stats['val']}</td>"
        f"<td class='num'>{fx_stats['z']}</td>"
        f"<td>{fx_svg}</td>"
        f"<td><a href='data/{c['fx_file']}'>CSV</a></td></tr>"
    )

    html.append("</table>")
    html.append("</div>")

# Footnote
html.append(
    "<p class='small'>Notes: Each indicator is smoothed (3M MA) then Z‑scored on a rolling window; "
    "risk_score = +Z if bad_when_high, −Z if bad_when_low. Composite is a normalized sum of weighted risk scores. "
    "Momentum uses 3‑month Δ in Z. Detector adds persistence to avoid whipsaws. Tune parameters in <code>config.yaml</code>.</p>"
)


# ---- Opportunities section (if file exists) ----
try:
    opp = json.loads((OUT_DIR / "opportunities.json").read_text(encoding="utf-8"))
    html.append("<h3 class='section'>Opportunities (per country)</h3>")
    html.append("<table><tr><th>Country</th><th>Equities</th><th>Bonds</th><th>Commodities</th><th>FX</th></tr>")
    for item in opp.get("countries", []):
        cn = item.get("country", "-")
        eq = item.get("equities", {}).get("action", "-")
        bd = item.get("bonds",    {}).get("action", "-")
        cm = item.get("commods",  {}).get("action", "-")
        fx = item.get("fx",       {}).get("action", "-")
        html.append(
            "<tr>"
            f"<td>{cn}</td>"
            f"<td>{eq}</td>"
            f"<td>{bd}</td>"
            f"<td>{cm}</td>"
            f"<td>{fx}</td>"
            "</tr>"
        )
    html.append("</table>")
except Exception as _e:
    # Silent: keep HTML rendering robust even if opportunities.json missing
    pass

html.append("</body></html>")

# ---------------------------- Artifacts -------------------------------
(OUT_DIR / "index.html").write_text("\n".join(html), encoding="utf-8")
print("✅ Wrote output/index.html")

for f in DATA_DIR.glob("*.csv"):
    shutil.copy2(f, OUT_DIR / "data" / f.name)

# Persist snapshot state
state = {
    "generated_utc": now,
    "composite": None if np.isnan(composite) else round(float(composite), 3),
    "composite_level": level,
    "detector_level": det_level,
    "composite_delta_3m": None if np.isnan(comp_delta_3m) else round(float(comp_delta_3m), 3),
    "countries": [
        {"name": x["name"], "subscore": None if x["subscore"] is None else round(float(x["subscore"]), 3), "level": x["level"]}
        for x in country_states
    ],
    "contributions": {
        lbl: (None if (wc is None or np.isnan(wc)) else round(float(wc), 6))
        for (lbl, _z, _mom, _r, _w, wc, _sh) in contrib_rows_out
    }
}
(OUT_DIR / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
print("✅ Wrote output/state.json")

# Composite history for backtesting
comp_hist = pd.DataFrame({"date": composite_series.index, "composite": composite_series.values}).dropna()
if len(comp_hist) >= DELTA3M_W + 1:
    comp_hist["composite_delta_3m"] = comp_hist["composite"] - comp_hist["composite"].shift(DELTA3M_W)
    comp_hist["detector_flag"] = np.where(
        (comp_hist["composite"] > TH_RED_LVL) | (comp_hist["composite_delta_3m"] > TH_RED_D), "RED",
        np.where((comp_hist["composite"] > 0) & (comp_hist["composite_delta_3m"] > TH_YELLOW_D), "YELLOW", "OK")
    )
else:
    comp_hist["composite_delta_3m"] = np.nan
    comp_hist["detector_flag"] = "NA"

comp_hist.to_csv(OUT_DIR / "composite_history.csv", index=False)
print("✅ Wrote output/composite_history.csv")
