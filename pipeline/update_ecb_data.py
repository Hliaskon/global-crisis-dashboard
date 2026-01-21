import os
import pandas as pd
from ingestors.ecb_sdmx import ecb_series

def main():
    os.makedirs("data", exist_ok=True)

    # Base USD/EUR (monthly)
    usd_eur = ecb_series("EXR", "M.USD.EUR.SP00.A", params={"startPeriod": "2005-01"})
    usd_eur = usd_eur.rename(columns={"value": "usd_per_eur"})
    usd_eur.to_csv("data/eur_usd_ecb.csv", index=False)  # keep legacy file for dashboard
    print(f"✅ Wrote data/eur_usd_ecb.csv with {len(usd_eur)} rows.")

    # Crosses per EUR (monthly)
    cny_eur = ecb_series("EXR", "M.CNY.EUR.SP00.A", params={"startPeriod": "2005-01"}).rename(columns={"value": "cny_per_eur"})
    cny_eur.to_csv("data/eur_cny_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_cny_ecb.csv with {len(cny_eur)} rows.")

    jpy_eur = ecb_series("EXR", "M.JPY.EUR.SP00.A", params={"startPeriod": "2005-01"}).rename(columns={"value": "jpy_per_eur"})
    jpy_eur.to_csv("data/eur_jpy_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_jpy_ecb.csv with {len(jpy_eur)} rows.")

    gbp_eur = ecb_series("EXR", "M.GBP.EUR.SP00.A", params={"startPeriod": "2005-01"}).rename(columns={"value": "gbp_per_eur"})
    gbp_eur.to_csv("data/eur_gbp_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_gbp_ecb.csv with {len(gbp_eur)} rows.")

    cad_eur = ecb_series("EXR", "M.CAD.EUR.SP00.A", params={"startPeriod": "2005-01"}).rename(columns={"value": "cad_per_eur"})
    cad_eur.to_csv("data/eur_cad_ecb.csv", index=False)
    print(f"✅ Wrote data/eur_cad_ecb.csv with {len(cad_eur)} rows.")

    # --- Derived USD crosses ---
    def derive_usd_cross(other_df, other_col, out_csv):
        df = pd.merge(usd_eur, other_df, on="date", how="inner")
        df["value"] = df["usd_per_eur"] / df[other_col]
        out = df[["date", "value"]]
        out.to_csv(out_csv, index=False)
        print(f"✅ Wrote {out_csv} with {len(out)} rows.")

    derive_usd_cross(cny_eur, "cny_per_eur", "data/usd_cny_ecb.csv")
    derive_usd_cross(jpy_eur, "jpy_per_eur", "data/usd_jpy_ecb.csv")
    derive_usd_cross(gbp_eur, "gbp_per_eur", "data/usd_gbp_ecb.csv")
    derive_usd_cross(cad_eur, "cad_per_eur", "data/usd_cad_ecb.csv")

if __name__ == "__main__":
    main()
