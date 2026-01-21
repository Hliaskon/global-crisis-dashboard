
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch central bank policy rates from BIS SDMX API (v1 CSV), filter by country.
Usage: python pipeline/update_policy_rate_bis.py <ISO2> <out_csv>
Output: data/<out_csv> with columns date,value (monthly, percent)
Docs: BIS SDMX API (CSV supported)
"""
import sys, io
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# BIS v1 data endpoint for CBPOL; CSV response is convenient for pandas
BIS_URL = "https://stats.bis.org/api/v1/data/CBPOL/all/all"

def fetch_bis_cbpol_csv(last_n=20000) -> pd.DataFrame:
    params = {"format": "csv", "lastNObservations": str(last_n)}
    r = requests.get(BIS_URL, params=params, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "time_period" if "time_period" in df.columns else ("time" if "time" in df.columns else None)
    val_col  = "obs_value"   if "obs_value"   in df.columns else ("value" if "value" in df.columns else None)
    if date_col is None or val_col is None:
        raise RuntimeError(f"Unexpected BIS CSV columns: {df.columns.tolist()}")
    return df.rename(columns={date_col:"date", val_col:"value"})

def main():
    if len(sys.argv) < 3:
        print("Usage: update_policy_rate_bis.py <ISO2> <out_csv>")
        sys.exit(2)
    iso2 = sys.argv[1].upper()
    out  = DATA_DIR / sys.argv[2]

    df = fetch_bis_cbpol_csv(last_n=20000)
    if "ref_area" not in df.columns:
        raise RuntimeError("BIS CSV missing 'ref_area' column.")
    cdf = df[df["ref_area"].str.upper() == iso2].copy()
    if cdf.empty:
        print(f"⚠️ No BIS policy-rate rows found for {iso2}; writing empty file.")
        pd.DataFrame(columns=["date","value"]).to_csv(out, index=False)
        return
    cdf["date"]  = pd.to_datetime(cdf["date"], errors="coerce")
    cdf["value"] = pd.to_numeric(cdf["value"], errors="coerce")
    cdf = cdf.dropna(subset=["date","value"]).sort_values("date")
    cdf[["date","value"]].to_csv(out, index=False)
    print(f"✅ Wrote {out} with {len(cdf)} rows for {iso2}.")

if __name__ == "__main__":
    main()
