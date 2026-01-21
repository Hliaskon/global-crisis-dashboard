import sys, io
import requests
import pandas as pd
from pathlib import Path

"""
Usage (actions workflow):
  python pipeline/update_oecd_cli.py CHN china_cli_oecd.csv
  python pipeline/update_oecd_cli.py DEU germany_cli_oecd.csv
"""

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# OECD MEI_CLI dataset; LOLITOAA = amplitude-adjusted CLI; monthly (M)
OECD_CSV = "https://stats.oecd.org/SDMX-JSON/data/MEI_CLI/LOLITOAA.{LOC}.M/all?contentType=csv"

def main():
    if len(sys.argv) < 3:
        print("Usage: update_oecd_cli.py <OECD_LOC> <out_csv>")
        sys.exit(2)
    loc = sys.argv[1].upper()
    out_csv = sys.argv[2]
    url = OECD_CSV.format(LOC=loc)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    # Expect TIME_PERIOD and OBS_VALUE
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(df.columns):
        raise RuntimeError(f"Unexpected OECD CSV columns: {df.columns.tolist()}")

    out = (
        df[["TIME_PERIOD", "OBS_VALUE"]]
        .rename(columns={"TIME_PERIOD":"date", "OBS_VALUE":"value"})
        .dropna(subset=["value"])
        .copy()
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out.to_csv(DATA_DIR / out_csv, index=False)
    print(f"✅ Wrote {DATA_DIR/out_csv} with {len(out)} rows.")

if __name__ == "__main__":
    main()
