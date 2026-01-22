
# -*- coding: utf-8 -*-
"""
Download a country equity index from Stooq with multiple candidate symbols.
Usage (examples):
  python pipeline/update_equity_index_stooq.py --symbols "^dax,dax" --out eq_germany.csv
  python pipeline/update_equity_index_stooq.py --symbols "^ftse,ftse" --out eq_united_kingdom.csv

Writes: data/<out>  with columns [date,value] in YYYY-MM-DD
Never fails the workflow: logs warnings and exits if no symbol returns data.
"""
import argparse
import io
from pathlib import Path

import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"

def fetch_one(symbol: str) -> pd.DataFrame:
    try:
        r = requests.get(STOOQ_URL.format(sym=symbol), timeout=30)
        r.raise_for_status()
        txt = r.text
        if not txt or txt.strip().lower().startswith("<html"):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(txt))
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.strip().lower() for c in df.columns]
        # standardize columns
        if "date" not in df.columns:
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        value_col = "close" if "close" in df.columns else None
        if value_col is None:
            non_date = [c for c in df.columns if c != "date"]
            if not non_date:
                return pd.DataFrame()
            value_col = non_date[-1]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date")[["date", "value"]]
        return df
    except Exception:
        return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated Stooq symbols to try, in priority order")
    ap.add_argument("--out", required=True, help="Output filename under data/ (e.g., eq_germany.csv)")
    args = ap.parse_args()

    out_path = Path("data") / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tried = [s.strip() for s in args.symbols.split(",") if s.strip()]
    ok = False
    for sym in tried:
        df = fetch_one(sym)
        if not df.empty:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            df.to_csv(out_path, index=False)
            print(f"✅ Wrote {out_path} from '{sym}' with {len(df):,} rows.")
            ok = True
            break
        else:
            print(f"⚠️ Symbol '{sym}' returned no usable data; trying next...")
    if not ok:
        print("⚠️ All candidate symbols failed; leaving prior file as-is (if any).")

if __name__ == "__main__":
    main()
