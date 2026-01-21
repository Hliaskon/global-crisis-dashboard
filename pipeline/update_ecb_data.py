import io
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ECB SDMX-CSV: dataset EXR, key = FREQUENCY.CURRENCY.DENOMINATOR.SERIES_VARIATION.UNIT
# We want daily (D) spot at 2pm CET (SP00) against EUR (denominator), unit A.
# Example: D.USD.EUR.SP00.A  -> USD per EUR
ECB_URL = "https://sdw-wsrest.ecb.europa.eu/service/data/EXR/D.{CUR}.EUR.SP00.A?contentType=csv"

CURRENCIES = [
    "USD", "GBP", "CAD", "JPY", "CNY", "INR", "RUB"
]

def pull_cur(cur: str) -> pd.Series:
    url = ECB_URL.format(CUR=cur)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Expect TIME_PERIOD and OBS_VALUE columns
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(df.columns):
        raise RuntimeError(f"Unexpected CSV columns for {cur}: {df.columns.tolist()}")
    df = df.rename(columns={"TIME_PERIOD":"date", "OBS_VALUE":"value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date")
    return df.set_index("date")["value"]

def main():
    series = {}
    for cur in CURRENCIES:
        try:
            s = pull_cur(cur)
            series[cur] = s
            # Save the raw per-EUR series too for debugging
            out = pd.DataFrame({"date": s.index, "value": s.values})
            out.to_csv(DATA_DIR / f"{cur.lower()}_per_eur_ecb.csv", index=False)
            print(f"✅ Wrote {cur.lower()}_per_eur_ecb.csv ({len(out)} rows)")
        except Exception as e:
            print(f"⚠️ Failed {cur}: {e}")

    # Derive USD per X (USD/X) using:
    # USD/X = (USD per EUR) / (X per EUR)
    if "USD" in series:
        usd_per_eur = series["USD"]
        def derive(pair_cur: str, out_name: str):
            if pair_cur not in series:
                print(f"⚠️ Missing {pair_cur} per EUR; cannot derive {out_name}")
                return
            x_per_eur = series[pair_cur]
            df = pd.concat([usd_per_eur, x_per_eur], axis=1, join="inner")
            df.columns = ["usd_per_eur", f"{pair_cur.lower()}_per_eur"]
            df["value"] = df["usd_per_eur"] / df[f"{pair_cur.lower()}_per_eur"]
            out = df[["value"]].reset_index().rename(columns={"index":"date"})
            out.to_csv(DATA_DIR / out_name, index=False)
            print(f"✅ Wrote {out_name} ({len(out)} rows)")

        derive("JPY", "usd_jpy_ecb.csv")
        derive("CNY", "usd_cny_ecb.csv")
        derive("GBP", "usd_gbp_ecb.csv")
        derive("CAD", "usd_cad_ecb.csv")
        derive("INR", "usd_inr_ecb.csv")
        derive("RUB", "usd_rub_ecb.csv")

        # Also store the USD per EUR directly for users of EUR countries
        eur = pd.DataFrame({"date": usd_per_eur.index, "value": usd_per_eur.values})
        eur.to_csv(DATA_DIR / "eur_usd_ecb.csv", index=False)
        print(f"✅ Wrote eur_usd_ecb.csv ({len(eur)} rows)")
    else:
        print("❌ USD per EUR not available; cannot derive USD/X")

if __name__ == "__main__":
    main()

