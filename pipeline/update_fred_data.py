
import os, io, json
from datetime import datetime
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FRED_KEY = os.getenv("FRED_API_KEY", "").strip()

SERIES = {
    "yield_curve_10y_3m.csv": "T10Y3M",       # 10-Year Treasury Constant Maturity Minus 3-Month Treasury
    "unemployment_rate.csv":  "UNRATE",       # Unemployment rate (%)
    "hy_credit_spread.csv":   "BAMLH0A0HYM2"  # ICE BofA US High Yield OAS (bps)
}

BASE = "https://api.stlouisfed.org/fred/series/observations"

def fetch_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": "1980-01-01"
    }
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(obs)[["date","value"]].copy()
    # Numeric coercion, keep NaN rows out
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df

def main():
    for out_name, sid in SERIES.items():
        try:
            df = fetch_series(sid)
            df.to_csv(DATA_DIR / out_name, index=False)
            print(f"✅ Wrote {DATA_DIR/out_name} with {len(df)} rows.")
        except Exception as e:
            print(f"⚠️ Failed {sid} -> {out_name}: {e}")

if __name__ == "__main__":
    main()
