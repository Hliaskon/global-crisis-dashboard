
import os
import pandas as pd
from ingestors.ecb_sdmx import ecb_series

# We already pull USD per EUR monthly. We'll also pull CNY per EUR monthly and compute USD/CNY.

def main():
    os.makedirs("data", exist_ok=True)

    # 1) USD per EUR (monthly)
    usd_eur = ecb_series(
        flow_id="EXR",
        key="M.USD.EUR.SP00.A",
        params={"startPeriod": "2005-01"}  # enough history
    ).rename(columns={"value": "usd_per_eur"})

    usd_eur.to_csv("data/eur_usd_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_usd_ecb.csv with {len(usd_eur)} rows.")

    # 2) CNY per EUR (monthly)
    cny_eur = ecb_series(
        flow_id="EXR",
        key="M.CNY.EUR.SP00.A",
        params={"startPeriod": "2005-01"}
    ).rename(columns={"value": "cny_per_eur"})

    cny_eur.to_csv("data/eur_cny_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_cny_ecb.csv with {len(cny_eur)} rows.")

    # 3) Join and compute USD/CNY = (USD/EUR) / (CNY/EUR)
    df = pd.merge(usd_eur, cny_eur, on="date", how="inner")
    df["value"] = df["usd_per_eur"] / df["cny_per_eur"]
    out = df[["date", "value"]].copy()
    out.to_csv("data/usd_cny_ecb.csv", index=False)
    print(f"✅ Wrote data/usd_cny_ecb.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()
