
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch S&P 500 daily close (SP500) from FRED with robust fallbacks:
1) FRED JSON API   (uses FRED_API_KEY if provided)
2) FRED API CSV    (file_type=csv)
3) fredgraph.csv   (graph endpoint)
Writes: data/sp500_fred.csv with columns: date,value
Refs:
- S&P 500 series page: https://fred.stlouisfed.org/series/SP500
- API docs (series/observations & file_type=csv/json): https://fred.stlouisfed.org/docs/api/fred/series_observations.html
"""
import os
import io
import sys
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SERIES_ID = "SP500"
FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()

JSON_URL     = "https://api.stlouisfed.org/fred/series/observations"
API_CSV_URL  = JSON_URL  # same endpoint; switch file_type to csv
GRAPH_CSV_URL= "https://fred.stlouisfed.org/graph/fredgraph.csv"

def _normalize_date_value(df: pd.DataFrame, prefer_series_col: str = None) -> pd.DataFrame:
    # Case-insensitive normalization
    lower_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)
    date_col = None
    for candidate in ("date", "observation_date", "time", "time_period"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # Likely got HTML or unexpected shape
        raise KeyError(f"No date column. Columns: {list(df.columns)}")

    # value column discovery
    value_col = None
    if "value" in df.columns:
        value_col = "value"
    elif prefer_series_col and prefer_series_col.lower() in df.columns:
        value_col = prefer_series_col.lower()
    else:
        # choose first non-date column
        candidates = [c for c in df.columns if c != date_col]
        if not candidates:
            raise KeyError("No value column found.")
        value_col = candidates[0]

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"]  = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
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
    df = pd.read_csv(io.StringIO(r.text))
    return _normalize_date_value(df)

def fetch_graph_csv(series_id: str, start="1980-01-01") -> pd.DataFrame:
    params = {"id": series_id, "cosd": start}
    r = requests.get(GRAPH_CSV_URL, params=params, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    if text[:1] == "<":
        raise RuntimeError("fredgraph.csv returned HTML instead of CSV.")
    df = pd.read_csv(io.StringIO(text))
    # The graph CSV usually has columns DATE and <SERIES_ID>
    return _normalize_date_value(df, prefer_series_col=series_id)

def main():
    out = DATA_DIR / "sp500_fred.csv"

    # 1) JSON API
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
