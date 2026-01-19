
import pandas as pd
import pandasdmx as sdmx  # official SDMX client

def ecb_series(flow_id: str, key: str, params=None) -> pd.DataFrame:
    """
    Download a time series from the ECB Data Portal (SDMX 2.1) using pandasdmx.
    Returns a DataFrame with columns: date, value (ascending).
    """
    # Create a Request bound to the ECB data source
    client = sdmx.Request('ECB')  # documented API
    # Query data; EXR is the flow, key like 'M.USD.EUR.SP00.A'
    msg = client.data(resource_id=flow_id, key=key, params=params or {})
    # Convert SDMX message to pandas objects
    ser = sdmx.to_pandas(msg)   # typically returns a Series named 'value' with a multi-index that includes TIME_PERIOD
    df = ser.reset_index()      # make it a DataFrame
    # Normalize to (date, value)
    if "TIME_PERIOD" in df.columns and "value" in df.columns:
        out = df[["TIME_PERIOD", "value"]].rename(columns={"TIME_PERIOD": "date"})
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date")

    # Fallback: attempt to find time/value columns
    time_col = next((c for c in df.columns if "TIME" in c.upper()), None)
    val_col = "value" if "value" in df.columns else None
    if time_col and val_col:
        out = df[[time_col, val_col]].rename(columns={time_col: "date"})
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date")

    raise RuntimeError("Unexpected SDMX structure for ECB series; no TIME_PERIOD/value found.")
