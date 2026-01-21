
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Industrial Production (US INDPRO, others OECD/MEI via FRED).
Usage: python pipeline/update_industrial_production_fred.py <ISO2> <out_csv>
"""
import os, sys
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY = os.getenv("FRED_API_KEY", "").strip()

SERIES = {
    "US": "INDPRO",                 # long history to 1919
    "DE": "DEUPROINDMISMEI",
    "FR": "FRAPROINDMISMEI",
    "IT": "ITAPROINDMISMEI",
    "JP": "JPNPROINDMISMEI",
    "GB": "GBRPROINDMISMEI",
    "CA": "CANPROINDMISMEI",
}

def fetch_fred(series_id: str) -> pd.DataFrame:
    params = {"series_id": series_id, "file_type": "json", "observation_start": "1950-01-01"}
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    r = requests.get(FRED_API, params=params, timeout=60)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(obs)[["date","value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    return df

def main():
    if len(sys.argv) < 3:
        print("Usage: update_industrial_production_fred.py <ISO2> <out_csv>")
        sys.exit(2)
    iso2 = sys.argv[1].upper()
    out  = DATA_DIR / sys.argv[2]
    sid  = SERIES.get(iso2)
    if not sid:
        raise SystemExit(f"No series id mapping for ISO2={iso2}")
    df = fetch_fred(sid)
    df.to_csv(out, index=False)
    print(f"✅ Wrote {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()

