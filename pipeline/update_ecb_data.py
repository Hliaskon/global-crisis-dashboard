
import os
import pandas as pd
from ingestors.ecb_sdmx import ecb_series

def main():
    os.makedirs("data", exist_ok=True)

    # 1) USD per EUR (monthly) -> save as date,value
    usd_eur = ecb_series(
        flow_id="EXR",
        key="M.USD.EUR.SP00.A",
        params={"startPeriod": "2005-01"}
    )
    usd_eur.to_csv("data/eur_usd_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_usd_ecb.csv with {len(usd_eur)} rows.")

    # 2) CNY per EUR (monthly) -> save as date,value
    cny_eur = ecb_series(
        flow_id="EXR",
        key="M.CNY.EUR.SP00.A",
        params={"startPeriod": "2005-01"}
    )
    cny_eur.to_csv("data/eur_cny_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_cny_ecb.csv with {len(cny_eur)} rows.")

    # 3) Join and compute USD/CNY = (USD/EUR) / (CNY/EUR)
    #    Work on copies with temporary names; final output is date,value
    usd_eur_tmp = usd_eur.rename(columns={"value": "usd_per_eur"})
    cny_eur_tmp = cny_eur.rename(columns={"value": "cny_per_eur"})
    df = pd.merge(usd_eur_tmp, cny_eur_tmp, on="date", how="inner")
    df["value"] = df["usd_per_eur"] / df["cny_per_eur"]
    out = df[["date", "value"]].copy()
    out.to_csv("data/usd_cny_ecb.csv", index=False)
    print(f"✅ Wrote data/usd_cny_ecb.csv with {len(out)} rows.")

if __name__ == "__main__":
    main()
