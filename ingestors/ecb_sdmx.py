
import pandas as pd
import sdmx

def ecb_series(flow_id: str, key: str, params=None) -> pd.DataFrame:
    """
    Download a time series from the ECB Data Portal (SDMX 2.1) using sdmx1.
    Returns a DataFrame with columns: date, value (ascending).
    """
    client = sdmx.Client("ECB")  # sdmx1 knows the ECB Data Portal endpoint
    msg = client.data(flow_id, key=key, params=params or {})
    df = sdmx.to_pandas(msg).reset_index()

    # Normalize to (date, value)
    # The SDMX client typically returns TIME_PERIOD and 'value'
    if "TIME_PERIOD" in df.columns and "value" in df.columns:
        out = df[["TIME_PERIOD", "value"]].rename(columns={"TIME_PERIOD": "date"})
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date")
        return out

    # Fallback: try to find time/value columns heuristically
    time_col = next((c for c in df.columns if "TIME" in c.upper()), None)
    val_col = "value" if "value" in df.columns else None
    if time_col and val_col:
        out = df[[time_col, val_col]].rename(columns={time_col: "date"})
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date")

    raise RuntimeError("Unexpected SDMX structure for ECB series; no TIME_PERIOD/value found.")

