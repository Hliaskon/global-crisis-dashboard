
# -*- coding: utf-8 -*-
"""
Fetch US IG OAS (ICE BofA US Corporate Index OAS) from FRED.
Series ID: BAMLC0A0CM
Writes: data/ig_oas.csv  with columns [date,value]
Strategy: JSON API (with FRED_API_KEY if present) -> CSV API -> fredgraph.csv fallback.
The script logs and exits without error if all sources fail (keeps pipeline green).
"""
import os
import sys
import io
import csv
import json
import time
import typing as t
from pathlib import Path

import pandas as pd
import requests

SERIES_ID = "BAMLC0A0CM"
OUT_PATH = Path("data/ig_oas.csv")
START = "1980-01-01"  # early enough for history

def _is_html(text: str) -> bool:
    if not isinstance(text, str):
        return False
    tri = text.strip().lower()
    return tri.startswith("<!doctype html") or tri.startswith("<html")

def fetch_json(series_id: str, api_key: t.Optional[str]) -> pd.DataFrame:
    """FRED JSON API -> DataFrame with [date,value]"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": START
    }
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    rows = []
    for o in obs:
        v = o.get("value", "")
        if v is None or v == ".":
            continue
        try:
            rows.append((o.get("date"), float(v)))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["date", "value"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    return df

def fetch_csv_api(series_id: str, api_key: t.Optional[str]) -> pd.DataFrame:
    """FRED CSV API (same endpoint but file_type=csv)."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "file_type": "csv",
        "observation_start": START
    }
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    txt = r.text
    if _is_html(txt) or not txt or "error_code" in txt.lower():
        raise RuntimeError("CSV API returned HTML/error page")
    df = pd.read_csv(io.StringIO(txt))
    # Expect columns: date, value
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if "date" not in df.columns or "value" not in df.columns:
        # FRED sometimes returns DATE,SERIES columns; standardize
        if "date" in df.columns and series_id.lower() in df.columns:
            df = df.rename(columns={series_id.lower(): "value"})
        else:
            # Last fallback: try to locate first two columns as date/value
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "value"})
            else:
                raise RuntimeError("CSV API: could not find [date,value] columns")
    # Clean & sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df[["date", "value"]]

def fetch_fredgraph(series_id: str) -> pd.DataFrame:
    """fredgraph.csv last-resort (no key), e.g. .../fredgraph.csv?id=BAMLC0A0CM&cosd=1980-01-01"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    txt = r.text
    if _is_html(txt) or not txt:
        raise RuntimeError("fredgraph.csv returned HTML/empty")
    df = pd.read_csv(io.StringIO(txt))
    # Expect DATE,<series_id>
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # Try exact id column
    sid = series_id.lower()
    if "date" not in df.columns or sid not in df.columns:
        # If not exact, take second column as value
        if len(df.columns) >= 2 and "date" in df.columns:
            value_col = [c for c in df.columns if c != "date"][0]
            df = df.rename(columns={value_col: "value"})
        else:
            raise RuntimeError("fredgraph.csv: unexpected columns")
    else:
        df = df.rename(columns={sid: "value"})
    # Normalize
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.replace({"value": {".": None}})
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df[["date", "value"]]

def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(df_out):,} rows.")

def main():
    api_key = os.getenv("FRED_API_KEY", "").strip() or None
    # Stage 1: JSON API
    try:
        df = fetch_json(SERIES_ID, api_key)
        if not df.empty:
            write_csv(df, OUT_PATH)
            return
        else:
            print("⚠️ JSON API returned no rows; trying CSV API...")
    except Exception as e:
        print(f"⚠️ JSON API failed: {e} — trying CSV API...")

    # Stage 2: CSV API
    try:
        df = fetch_csv_api(SERIES_ID, api_key)
        if not df.empty:
            write_csv(df, OUT_PATH)
            return
        else:
            print("⚠️ CSV API returned no rows; trying fredgraph.csv...")
    except Exception as e:
        print(f"⚠️ CSV API failed: {e} — trying fredgraph.csv...")

    # Stage 3: fredgraph.csv
    try:
        df = fetch_fredgraph(SERIES_ID)
        if not df.empty:
            write_csv(df, OUT_PATH)
            return
        else:
            print("⚠️ fredgraph.csv returned no rows.")
    except Exception as e:
        print(f"⚠️ fredgraph.csv failed: {e}")

    print("⚠️ Could not fetch IG OAS today. Leaving prior file as-is (if any).")

if __name__ == "__main__":
    main()

