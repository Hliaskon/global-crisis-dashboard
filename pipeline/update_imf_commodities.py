
# -*- coding: utf-8 -*-
"""
Fetch Gold (USD/oz) and Copper (daily proxy) from Stooq and write as date,value CSVs.
We keep the output filenames aligned to config.yaml from Step 1:
  - data/gold_imf.csv
  - data/copper_imf.csv

Stooq daily CSV endpoints (no key needed). We try a shortlist of symbols per metal and
pick the first that returns valid data. If everything fails, we log a warning and exit
without error (pipeline stays green).

Gold candidates (USD/oz):  xauusd
Copper candidates:         hg.f (COMEX Copper futures continuous), hg

Columns normalized to: [date,value] with YYYY-MM-DD.
"""
from pathlib import Path
import io
import sys
import pandas as pd
import requests

OUT_GOLD   = Path("data/gold_imf.csv")
OUT_COPPER = Path("data/copper_imf.csv")

STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"

GOLD_SYMBOLS   = ["xauusd"]
COPPER_SYMBOLS = ["hg.f", "hg"]

def _fetch_stooq_series(symbol: str) -> pd.DataFrame:
    """Return Stooq daily OHLC CSV as DataFrame or empty on failure."""
    url = STOOQ_URL.format(sym=symbol)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        txt = r.text
        if not txt or txt.strip().lower().startswith("<html"):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(txt))
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" not in df.columns:
            # assume first column is date
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        # use 'close' if present; else last non-date column
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

def _try_write(symbols, out_path: Path, label: str):
    for sym in symbols:
        df = _fetch_stooq_series(sym)
        if not df.empty:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_out = df.copy()
            df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
            df_out.to_csv(out_path, index=False)
            print(f"✅ Wrote {out_path} from Stooq symbol '{sym}' with {len(df_out):,} rows.")
            return True
        else:
            print(f"⚠️ {label}: Stooq symbol '{sym}' returned no usable data; trying next...")
    print(f"⚠️ {label}: all symbol candidates failed. Leaving prior file as-is (if any).")
    return False

def main():
    ok_gold = _try_write(GOLD_SYMBOLS, OUT_GOLD, "Gold")
    ok_cu   = _try_write(COPPER_SYMBOLS, OUT_COPPER, "Copper")
    if not (ok_gold or ok_cu):
        # No hard failure; keep pipeline green
        pass

if __name__ == "__main__":
    main()

