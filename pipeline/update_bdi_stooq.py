
# -*- coding: utf-8 -*-
"""
Fetch Baltic Dry Index (BDI) from Stooq and write as date,value.
URL: https://stooq.com/q/d/l/?s=bdi&i=d
Output: data/bdi_stooq.csv  (columns: date,value)
Behavior: log warnings on failure; do NOT fail the workflow.
"""
from pathlib import Path
import io
import sys
import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/?s=bdi&i=d"
OUT_PATH = Path("data/bdi_stooq.csv")

def main():
    try:
        r = requests.get(STOOQ_URL, timeout=30)
        r.raise_for_status()
        txt = r.text
        if not txt or txt.strip().lower().startswith("<html"):
            print("⚠️ Stooq returned HTML/empty for BDI.")
            return

        # Stooq CSV columns typically: Date,Open,High,Low,Close,Volume
        df = pd.read_csv(io.StringIO(txt))
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Some mirrors may use 'date'/'close' or localized headers; handle generically
        if "date" not in df.columns:
            # try first column as date
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        # pick close as value if exists, else last numeric column
        if "close" not in df.columns:
            # find any numeric column other than date
            non_date_cols = [c for c in df.columns if c != "date"]
            if not non_date_cols:
                print("⚠️ Could not find value column in BDI CSV.")
                return
            value_col = non_date_cols[-1]
            df.rename(columns={value_col: "close"}, inplace=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values("date").loc[:, ["date", "value"]]
        if df.empty:
            print("⚠️ BDI dataframe empty after cleaning.")
            return

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.copy()
        df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
        df_out.to_csv(OUT_PATH, index=False)
        print(f"✅ Wrote {OUT_PATH} with {len(df_out):,} rows.")
    except Exception as e:
        print(f"⚠️ BDI fetch failed: {e}. Leaving prior file as-is (if any).")

if __name__ == "__main__":
    main()

