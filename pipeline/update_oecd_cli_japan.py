
import os
import io
import pandas as pd
import requests

# OECD MEI_CLI: LOLITOAA.JPN.M (amplitude-adjusted CLI, Japan, monthly)
URL = "https://stats.oecd.org/SDMX-JSON/data/MEI_CLI/LOLITOAA.JPN.M/all?contentType=csv"

def main():
    os.makedirs("data", exist_ok=True)

    # Download CSV (SDMX-JSON endpoint with CSV output)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()

    # Parse CSV from memory
    df = pd.read_csv(io.StringIO(r.text))

    # Expect TIME_PERIOD and OBS_VALUE columns from OECD CSV
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(df.columns):
        raise RuntimeError(f"Unexpected OECD CSV columns: {df.columns.tolist()}")

    out = (
        df[["TIME_PERIOD", "OBS_VALUE"]]
        .rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"})
        .dropna(subset=["value"])
        .copy()
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")

    out.to_csv("data/japan_cli_oecd.csv", index=False)
    print(f"✅ Wrote data/japan_cli_oecd.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()
