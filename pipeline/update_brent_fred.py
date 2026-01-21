
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Brent spot price (DCOILBRENTEU) from FRED.
- Tries JSON API first (uses FRED_API_KEY if available).
- Falls back to the CSV endpoint if JSON fails (no key required).
Writes: data/brent_spot_fred.csv  with columns: date,value
Ref: https://fred.stlouisfed.org/series/DCOILBRENTEU
"""
import os
import io
import sys
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SERIES_ID = "DCOILBRENTEU"
FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()

JSON_URL  = "https://api.stlouisfed.org/fred/series/observations"
CSV_URL   = "https://fred.stlouisfed.org/graph/fredgraph.csv"

def fetch_json(series_id: str, observation_start="1980-01-01") -> pd.DataFrame:
    """Fetch via FRED JSON API (may require key / rate-limited)."""
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": observation_start,
    }
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    r = requests.get(JSON_URL, params=params, timeout=60)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(obs)[["date","value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    return df

def fetch_csv(series_id: str, start="1980-01-01") -> pd.DataFrame:
    """Fetch via FRED CSV endpoint (no key)."""
    params = {"id": series_id, "cosd": start}
    r = requests.get(CSV_URL, params=params, timeout=60)
    r.raise_for_status()
    # CSV comes as two columns, DATE and <SERIES_ID>, with missing values as '.'
    df = pd.read_csv(io.StringIO(r.text))
    # Normalize to date,value schema
    value_col = series_id
    if value_col not in df.columns and "VALUE" in df.columns:
        value_col = "VALUE"
    df = df.rename(columns={"DATE": "date", value_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    return df[["date","value"]]

def main():
    out = DATA_DIR / "brent_spot_fred.csv"

    # Try JSON first, then CSV fallback
    try:
        df = fetch_json(SERIES_ID, observation_start="1980-01-01")
        if df.empty:
            raise RuntimeError("Empty JSON response; switching to CSV fallback.")
        df.to_csv(out, index=False)
        print(f"✅ Wrote {out} (JSON API) with {len(df)} rows.")
        return
    except Exception as e:
        print(f"⚠️ JSON API failed: {e} — trying CSV fallback...")

    # CSV fallback
    df2 = fetch_csv(SERIES_ID, start="1980-01-01")
    df2.to_csv(out, index=False)
    print(f"✅ Wrote {out} (CSV fallback) with {len(df2)} rows.")

if __name__ == "__main__":
    main()
