
# -*- coding: utf-8 -*-
"""
Fetch US 10y Breakeven Inflation from FRED (series: T10YIE)
Writes: data/t10yie.csv  with columns [date,value] (YYYY-MM-DD, float)
Strategy: JSON API (uses FRED_API_KEY if set) -> CSV API -> fredgraph.csv fallback.
If all sources fail, log and exit without hard error (keeps workflow green).
"""
import os
import io
from pathlib import Path
import pandas as pd
import requests

SERIES_ID = "T10YIE"
OUT_PATH = Path("data/t10yie.csv")
START = "1999-01-01"  # series starts around 2003; early start is fine

def _is_html(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html")

def fetch_json(series_id: str, api_key: str | None) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "file_type": "json", "observation_start": START}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for o in data.get("observations", []):
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

def fetch_csv_api(series_id: str, api_key: str | None) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "file_type": "csv", "observation_start": START}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    txt = r.text
    if _is_html(txt) or not txt or "error_code" in txt.lower():
        raise RuntimeError("CSV API returned HTML/error page")
    df = pd.read_csv(io.StringIO(txt))
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # standardize to [date,value]
    if "date" not in df.columns or "value" not in df.columns:
        if "date" in df.columns and series_id.lower() in df.columns:
            df = df.rename(columns={series_id.lower(): "value"})
        elif len(df.columns) >= 2 and "date" in df.columns:
            df = df.rename(columns={df.columns[1]: "value"})
        else:
            raise RuntimeError("CSV API did not return date/value columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df[["date", "value"]]

def fetch_fredgraph(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    txt = r.text
    if _is_html(txt) or not txt:
        raise RuntimeError("fredgraph.csv returned HTML/empty")
    df = pd.read_csv(io.StringIO(txt))
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    sid = series_id.lower()
    if "date" not in df.columns:
        raise RuntimeError("fredgraph.csv missing DATE column")
    if sid not in df.columns and len(df.columns) >= 2:
        # take second column as value if series_id-named column not present
        value_col = [c for c in df.columns if c != "date"][0]
        df = df.rename(columns={value_col: "value"})
    else:
        df = df.rename(columns={sid: "value"})
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
    api_key = (os.getenv("FRED_API_KEY") or "").strip() or None

    # 1) JSON API
    try:
        df = fetch_json(SERIES_ID, api_key)
        if not df.empty:
            write_csv(df, OUT_PATH)
            return
        else:
            print("⚠️ JSON API returned no rows; trying CSV API...")
    except Exception as e:
        print(f"⚠️ JSON API failed: {e} — trying CSV API...")

    # 2) CSV API
    try:
        df = fetch_csv_api(SERIES_ID, api_key)
        if not df.empty:
            write_csv(df, OUT_PATH)
            return
        else:
            print("⚠️ CSV API returned no rows; trying fredgraph.csv...")
    except Exception as e:
        print(f"⚠️ CSV API failed: {e} — trying fredgraph.csv...")

    # 3) fredgraph.csv
    try:
        df = fetch_fredgraph(SERIES_ID)
        if not df.empty:
            write_csv(df, OUT_PATH)
            return
        else:
            print("⚠️ fredgraph.csv returned no rows.")
    except Exception as e:
        print(f"⚠️ fredgraph.csv failed: {e}")

    print("⚠️ Could not fetch T10YIE today. Leaving any prior file as-is.")

if __name__ == "__main__":
    main()

