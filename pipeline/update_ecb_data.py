
import os
import pandas as pd
from ingestors.ecb_sdmx import ecb_series

def main():
    os.makedirs("data", exist_ok=True)

    # 1) USD per EUR (monthly) -> date,value
    usd_eur = ecb_series(
        flow_id="EXR",
        key="M.USD.EUR.SP00.A",
        params={"startPeriod": "2005-01"}
    )
    usd_eur.to_csv("data/eur_usd_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_usd_ecb.csv with {len(usd_eur)} rows.")

    # 2) CNY per EUR (monthly) -> date,value (used already for USD/CNY)
    cny_eur = ecb_series(
        flow_id="EXR",
        key="M.CNY.EUR.SP00.A",
        params={"startPeriod": "2005-01"}
    )
    cny_eur.to_csv("data/eur_cny_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_cny_ecb.csv with {len(cny_eur)} rows.")

    # 3) JPY per EUR (monthly) -> date,value (for USD/JPY)
    jpy_eur = ecb_series(
        flow_id="EXR",
        key="M.JPY.EUR.SP00.A",
        params={"startPeriod": "2005-01"}
    )
    jpy_eur.to_csv("data/eur_jpy_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_jpy_ecb.csv with {len(jpy_eur)} rows.")

    # --- Derived crosses ---
    # USD/CNY = (USD/EUR) / (CNY/EUR)
    usd_eur_tmp = usd_eur.rename(columns={"value": "usd_per_eur"})
    cny_eur_tmp = cny_eur.rename(columns={"value": "cny_per_eur"})
