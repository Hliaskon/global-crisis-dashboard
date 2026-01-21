
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Brent spot price (DCOILBRENTEU) from FRED and write data/brent_spot_fred.csv
Schema: date,value
Ref: https://fred.stlouisfed.org/series/DCOILBRENTEU
"""
import os
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
SERIES_ID = "DCOILBRENTEU"  # Brent - Europe (USD/bbl)
FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": "1980-01-01"
    }
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    r = requests.get(FRED_API, params=params, timeout=60)
    r.raise_for_status()
    data = r.json().get("observations", [])
    if not data:
        return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(data)[["date","value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    return df

def main():
    out = DATA_DIR / "brent_spot_fred.csv"
    df  = fetch_fred_series(SERIES_ID)
    df.to_csv(out, index=False)
    print(f"✅ Wrote {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
