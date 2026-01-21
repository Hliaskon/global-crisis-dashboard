
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Helpers
# -------------------------------
def load_series(csv_path: Path, weekly_resample: bool = True) -> pd.Series:
    """Load CSV with columns (date,value) into a pandas Series indexed by datetime.
    Optionally resample to weekly (Fri) and interpolate for consistent cadence.
    """
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
    """Rolling z-score vs history (default ~5 years of weekly points)."""
    if len(s) < 10:
        return s * np.nan
    roll = 260  # ~5 years of weekly data
    mu = s.rolling(roll, min_periods=max(26, roll // 10)).mean()
    sd = s.rolling(roll, min_periods=max(26, roll // 10)).std()
    return (s - mu) / sd


def sparkline_svg(values, width: int = 140, height: int = 28, stroke: str = "#1f77b4") -> str:
    """Return a tiny inline SVG sparkline from a sequence of numeric values."""
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
        y_svg = height - 2 - y_norm * (height - 4)  # 2px padding
        pts.append(f"{x:.1f},{y_svg:.1f}")
    path = "M " + " L ".join(pts)
    svg = (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<path d='{path}' fill='none' stroke='{stroke}' stroke-width='1.5' /></svg>"
    )
    return svg


def spark_from_file(file_name: str, tail_points: int = 52) -> str:
    """Build a sparkline from the last N weekly points of a file in DATA_DIR."""
    path = DATA_DIR / file_name
    if not path.exists():
        return ""
    s = load_series(path, weekly_resample=True)
    if s.empty:
        return ""
    return sparkline_svg(s.tail(tail_points).tolist())


# -------------------------------
# Indicator map
# -------------------------------
# Keep US + ECB in the main table and show China items in their own section.
series_map = {
    "US Yield Curve 10Y–3M (bps)": "yield_curve_10y_3m.csv",
    "US Unemployment Rate (%)": "unemployment_rate.csv",
    "US HY OAS (bps)": "hy_credit_spread.csv",
    "ECB USD per EUR (monthly)": "eur_usd_ecb.csv",
    "USD/CNY (derived, ECB monthly)": "usd_cny_ecb.csv",
    "OECD China CLI (amplitude adj., SA)": "china_cli_oecd.csv",
}

# -------------------------------
# Calculate latest readings + composite
# -------------------------------
display_rows = []   # rows for printing
z_values = []       # contributes to global composite

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

    # Append to display table
    display_rows.append(
        (label, last_date, f"{last_val:.2f}", "-" if np.isnan(last_z) else f"{last_z:.2f}")
    )

    # Directional mapping -> composite
    if not np.isnan(last_z):
        if "Yield Curve" in label:
            # more inversion (more negative) => risk increases
            z_values.append(-last_z)
        elif "HY OAS" in label:
            z_values.append(last_z)
        elif "Unemployment" in label:
            z_values.append(last_z)
        elif "USD per EUR" in label:
            z_values.append(last_z)
        elif "USD/CNY" in label:
            z_values.append(last_z)
        elif "China CLI" in label:
            # below-trend CLI => risk increases
            z_values.append(-last_z)

# Global composite (mean of mapped z's)
composite = np.nan if not z_values else float(np.mean(z_values))
level = "OK"
if not np.isnan(composite):
    if composite >= 2.0:
