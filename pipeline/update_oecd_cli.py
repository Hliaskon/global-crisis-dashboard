
import os, io
import pandas as pd
import requests

# OECD SDMX-JSON with CSV output (MEI_CLI / LOLITOAA = amplitude-adjusted CLI)
# Key order: SUBJECT.LOCATION.FREQUENCY -> LOLITOAA.CHN.M
OECD_CLI_CSV = "https://stats.oecd.org/SDMX-JSON/data/MEI_CLI/LOLITOAA.CHN.M/all?contentType=csv"

def main():
    os.makedirs("data", exist_ok=True)
    r = requests.get(OECD_CLI_CSV, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    # Expect TIME_PERIOD (e.g., 2024-11) and OBS_VALUE columns from OECD CSV
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(df.columns):
        raise RuntimeError(f"Unexpected OECD CSV columns: {df.columns.tolist()}")

    out = (
        df[["TIME_PERIOD", "OBS_VALUE"]]
        .rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"})
        .dropna(subset=["value"])
        .copy()
    )

    # Parse dates (monthly); our dashboard resamples to a weekly cadence later
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")

    out.to_csv("data/china_cli_oecd.csv", index=False)
    print(f"✅ Wrote data/china_cli_oecd.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()
