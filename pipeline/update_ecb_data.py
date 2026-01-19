
import os
import pandas as pd

from ingestors.ecb_sdmx import ecb_series

# We will fetch the ECB euro foreign exchange reference rate:
# Dataflow: EXR
# Key format: <FREQ>.<CURRENCY>.<BASE_CURRENCY>.<EXR_TYPE>.<SERIES_VARIATION>
# Example for monthly USD per EUR reference rate:
# M.USD.EUR.SP00.A  (Monthly, USD vs EUR, spot reference, average)
#
# A daily variant exists (D.USD.EUR.SP00.A), but monthly is clean for a first check.

def main():
    df = ecb_series(
        flow_id="EXR",
        key="M.USD.EUR.SP00.A",
        params={"startPeriod": "2005-01"}  # trim to reasonable history
    )
    # Normalize to (date,value) CSV in data/
    df = df.rename(columns={"value": "value"})[["date", "value"]]
    os.makedirs("data", exist_ok=True)
    out = "data/eur_usd_ecb.csv"
    df.to_csv(out, index=False)
    print(f"✅ Wrote {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()

