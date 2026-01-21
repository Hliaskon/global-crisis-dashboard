
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Brent spot price (DCOILBRENTEU) from FRED with robust fallbacks:
1) FRED JSON API   (uses FRED_API_KEY if provided)
2) FRED API CSV    (file_type=csv)
3) fredgraph.csv   (graph endpoint)
Writes: data/brent_spot_fred.csv with columns: date,value
Ref (series): https://fred.stlouisfed.org/series/DCOILBRENTEU
"""

import os
import io
import sys
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SERIES_ID = "DCOILBRENTEU"
FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()

JSON_URL  = "https://api.stlouisfed.org/fred/series/observations"
API_CSV_URL = JSON_URL  # same endpoint, but file_type=csv
GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

def _normalize_date_value(df: pd.DataFrame, prefer_series_col: str = None) -> pd.DataFrame:
    # make headers case-insensitive
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    # detect date column
    date_col = None
    for candidate in ("date", "observation_date", "time", "time_period"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # Not CSV? maybe HTML returned
        head = df.head(1).to_string()
        raise KeyError(f"No date column in CSV. First row: {head}")

    # detect value column
    value_col = None
    # try 'value' first (FRED API CSV), then the series id
    if "value" in df.columns:
        value_col = "value"
    elif prefer_series_col and prefer_series_col.lower() in df.columns:
        value_col = prefer_series_col.lower()
    else:
        # try to find any numeric column other than date
        numeric_cols = [c for c in df.columns if c != date_col]
        if not numeric_cols:
            raise KeyError("No value column found.")
        # pick first numeric-ish
        value_col = numeric_cols[0]

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    # coerce types
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["date"]  = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return out

def fetch_json(series_id: str, observation_start="1980-01-01") -> pd.DataFrame:
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
    return _normalize_date_value(df)

def fetch_api_csv(series_id: str, observation_start="1980-01-01") -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "csv",
        "observation_start": observation_start,
    }
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    r = requests.get(API_CSV_URL, params=params, timeout=60)
    r.raise_for_status()
    # FRED API CSV header: date,value
    df = pd.read_csv(io.StringIO(r.text))
    return _normalize_date_value(df)  # will detect 'date'/'value'

def fetch_graph_csv(series_id: str, start="1980-01-01") -> pd.DataFrame:
    params = {"id": series_id, "cosd": start}
    r = requests.get(GRAPH_CSV_URL, params=params, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    # rudimentary HTML detection (e.g., error page)
    if text[:1] == "<":
        raise RuntimeError("fredgraph.csv returned HTML instead of CSV.")
    df = pd.read_csv(io.StringIO(text))
    # graph CSV usually has DATE and SERIES_ID columns
    prefer_series_col = series_id  # e.g., DCOILBRENTEU
    return _normalize_date_value(df, prefer_series_col=prefer_series_col)

def main():
    out = DATA_DIR / "brent_spot_fred.csv"

    # 1) JSON
    try:
        df = fetch_json(SERIES_ID, observation_start="1980-01-01")
        if not df.empty:
            df.to_csv(out, index=False)
            print(f"✅ Wrote {out} (JSON API) with {len(df)} rows.")
            return
        else:
            print("ℹ️ JSON returned empty; trying CSV API ...")
    except Exception as e:
        print(f"⚠️ JSON API failed: {e} — trying CSV API ...")

    # 2) API CSV
    try:
        df2 = fetch_api_csv(SERIES_ID, observation_start="1980-01-01")
        if not df2.empty:
            df2.to_csv(out, index=False)
            print(f"✅ Wrote {out} (CSV API) with {len(df2)} rows.")
            return
        else:
            print("ℹ️ CSV API returned empty; trying fredgraph.csv ...")
    except Exception as e:
        print(f"⚠️ CSV API failed: {e} — trying fredgraph.csv ...")

    # 3) fredgraph.csv
    df3 = fetch_graph_csv(SERIES_ID, start="1980-01-01")
    if df3.empty:
        raise SystemExit("❌ fredgraph.csv returned empty as well.")
    df3.to_csv(out, index=False)
    print(f"✅ Wrote {out} (fredgraph.csv) with {len(df3)} rows.")

if __name__ == "__main__":
    main()
