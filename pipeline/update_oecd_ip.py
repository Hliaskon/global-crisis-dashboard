
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Industrial Production indices with robust fallbacks:
- US: INDPRO (very long history)
- Others: OECD/MEI via FRED (e.g., DEUPROINDMISMEI)
Usage:
  python pipeline/update_oecd_ip.py <ISO2> <out_csv>
Docs:
- FRED series/observations supports file_type=csv/json: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- OECD MEI IP series via FRED (e.g., DEUPROINDMISMEI, FRAPROINDMISMEI, ...).
"""

import os, io, sys
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()
JSON_URL  = "https://api.stlouisfed.org/fred/series/observations"
API_CSV_URL  = JSON_URL
GRAPH_CSV_URL= "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "US": "INDPRO",                # US Industrial Production (since 1919)
    "DE": "DEUPROINDMISMEI",
    "FR": "FRAPROINDMISMEI",
    "IT": "ITAPROINDMISMEI",
    "JP": "JPNPROINDMISMEI",
    "GB": "GBRPROINDMISMEI",
    "CA": "CANPROINDMISMEI",
}

def _normalize_date_value(df: pd.DataFrame, prefer_series_col: str = None) -> pd.DataFrame:
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    date_col = None
    for c in ("date", "observation_date", "time", "time_period"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise KeyError(f"No date column. Columns: {list(df.columns)}")
    if "value" in df.columns:
        value_col = "value"
    elif prefer_series_col and prefer_series_col.lower() in df.columns:
        value_col = prefer_series_col.lower()
    else:
        others = [c for c in df.columns if c != date_col]
        if not others:
            raise KeyError("No value column found.")
        value_col = others[0]
    out = df[[date_col, value_col]].copy()
    out.columns = ["date","value"]
    out["date"]  = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
    return out

def fetch_json(series_id: str, observation_start="1950-01-01") -> pd.DataFrame:
    params = {"series_id": series_id, "file_type": "json", "observation_start": observation_start}
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    r = requests.get(JSON_URL, params=params, timeout=60)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(obs)[["date","value"]]
    return _normalize_date_value(df)

def fetch_api_csv(series_id: str, observation_start="1950-01-01") -> pd.DataFrame:
    params = {"series_id": series_id, "file_type": "csv", "observation_start": observation_start}
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    r = requests.get(API_CSV_URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return _normalize_date_value(df)

def fetch_graph_csv(series_id: str, start="1950-01-01") -> pd.DataFrame:
    params = {"id": series_id, "cosd": start}
    r = requests.get(GRAPH_CSV_URL, params=params, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    if text[:1] == "<":
        raise RuntimeError("fredgraph.csv returned HTML instead of CSV.")
    df = pd.read_csv(io.StringIO(text))
    return _normalize_date_value(df, prefer_series_col=series_id)

def main():
    if len(sys.argv) < 3:
        print("Usage: update_oecd_ip.py <ISO2> <out_csv>")
        sys.exit(2)
    iso2 = sys.argv[1].upper()
    out  = DATA_DIR / sys.argv[2]
    sid  = SERIES.get(iso2)
    if not sid:
        raise SystemExit(f"No series id mapping for ISO2={iso2}")

    # 1) JSON
    try:
        df = fetch_json(sid, observation_start="1950-01-01")
        if not df.empty:
            df.to_csv(out, index=False)
            print(f"✅ Wrote {out} (JSON API) with {len(df)} rows.")
            return
        else:
            print("ℹ️ JSON returned empty; trying CSV API ...")
    except Exception as e:
        print(f"⚠️ JSON API failed: {e} — trying CSV API ...")

    # 2) CSV API
    try:
        df2 = fetch_api_csv(sid, observation_start="1950-01-01")
        if not df2.empty:
            df2.to_csv(out, index=False)
            print(f"✅ Wrote {out} (CSV API) with {len(df2)} rows.")
            return
        else:
            print("ℹ️ CSV API returned empty; trying fredgraph.csv ...")
    except Exception as e:
        print(f"⚠️ CSV API failed: {e} — trying fredgraph.csv ...")

    # 3) graph CSV
    df3 = fetch_graph_csv(sid, start="1950-01-01")
    if df3.empty:
        raise SystemExit("❌ fredgraph.csv returned empty as well.")
    df3.to_csv(out, index=False)
    print(f"✅ Wrote {out} (fredgraph.csv) with {len(df3)} rows.")

if __name__ == "__main__":
    main()
