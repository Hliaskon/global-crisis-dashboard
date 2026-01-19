
import os
import requests
import csv

FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "yield_curve_10y_3m": "T10Y3M",
    "unemployment_rate": "UNRATE",
    "hy_credit_spread": "BAMLH0A0HYM2"
}

os.makedirs("data", exist_ok=True)

for name, series_id in SERIES.items():
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    observations = response.json()["observations"]

    filename = f"data/{name}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "value"])
        for obs in observations:
            if obs["value"] != ".":
                writer.writerow([obs["date"], obs["value"]])

print("✅ FRED data downloaded successfully")

