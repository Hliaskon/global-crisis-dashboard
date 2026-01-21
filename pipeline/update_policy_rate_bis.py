
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch policy rates from BIS Bulk Downloads (WS_CBPOL_csv_flat.zip),
filter a given country (ISO2), and write data/policy_<iso2>.csv with columns: date,value.

Usage:
  python pipeline/update_policy_rate_bis.py US policy_us.csv

Docs:
- BIS Bulk downloads landing: https://data.bis.org/bulkdownload
- Dataset id commonly used in tooling: WS_CBPOL_csv_flat (policy rates, long history)
Notes:
- BIS sometimes encodes ref_area as "US: United States". We normalize to the ISO2 before colon.
- We prefer monthly frequency if both monthly and daily time stamps are present.
"""
import io, re, sys, zipfile
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

BULK_URL = "https://data.bis.org/bulkdownload/WS_CBPOL_csv_flat.zip"

def _normalize_ref_area(val: str) -> str:
    """Return the ISO2 part from 'US' or 'US: United States'."""
    if not isinstance(val, str):
        return ""
    # split on colon or whitespace, take first token with only letters
    token = val.split(":", 1)[0].strip()
    token = token.split(" ", 1)[0].strip()
    return token.upper()

