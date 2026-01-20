
import pandas as pd
import requests

BASE = "https://data-api.ecb.europa.eu/service"

def _parse_sdmx_json(payload: dict) -> pd.DataFrame:
    """
    Parse an SDMX-JSON response (ECB) into a tidy (date, value) DataFrame.
    Assumes a single series in the dataset.
    """
    # Time dimension labels
    time_vals = payload["structure"]["dimensions"]["observation"][0]["values"]
    dates = [v.get("id") for v in time_vals]

    # First (and typically only) series key
    series_dict = payload["dataSets"][0].get("series", {})
    if not series_dict:
        return pd.DataFrame(columns=["date", "value"])

    series_key, series_obj = next(iter(series_dict.items()))
    obs = series_obj.get("observations", {})

    rows = []
    for idx_str, val_list in obs.items():
        i = int(idx_str)
        # SDMX-JSON encodes value as a 1-element list [value]
        val = None
        if isinstance(val_list, list) and val_list:
            val = val_list[0]
        date_label = dates[i] if i < len(dates) else None
        if date_label is not None and val is not None:
            rows.append((date_label, float(val)))

    df = pd.DataFrame(rows, columns=["date", "value"])
    # Parse YYYY-MM or YYYY-MM-DD to pandas datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def ecb_series(flow_id: str, key: str, params=None) -> pd.DataFrame:
    """
    Request ECB SDMX-JSON using the new Data Portal endpoint.
    Returns DataFrame: columns ['date','value'] ascending by date.
    """
    url = f"{BASE}/data/{flow_id}/{key}"
    q = dict(params or {})
    # Force JSON SDMX to avoid content negotiation ambiguity
    q["format"] = "jsondata"

    # ECB recommends content negotiation / SDMX headers; using format param is acceptable
    # https://data.ecb.europa.eu/help/api/overview
    r = requests.get(url, params=q, timeout=60)
    r.raise_for_status()
    payload = r.json()
    return _parse_sdmx_json(payload)
