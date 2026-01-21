
import os, io
import pandas as pd
import requests

# OECD MEI_CLI: LOLITOAA.EA19.M (amplitude-adjusted CLI, Euro Area 19, monthly)
URL = "https://stats.oecd.org/SDMX-JSON/data/MEI_CLI/LOLITOAA.EA19.M/all?contentType=csv"

def main():
    os.makedirs("data", exist_ok=True)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
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
    out.to_csv("data/euroarea_cli_oecd.csv", index=False)
    print(f"✅ Wrote data/euroarea_cli_oecd.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()


