"""
step7_execution_advisor/dw_reader.py
Data Warehouse RAW_MARKET Reader.

Reads 28 fields from DW Sheet RAW_MARKET tab (L2, L4, L5, L7).
Returns latest value per indicator as a flat dict.

Degraded Mode: If DW Sheet is unreachable, returns empty dict.
Scoring dimensions 2-6 will default to 0.

Source: Trading Desk Spec Teil 4 §19
"""

import json
import logging
import os

logger = logging.getLogger("execution_advisor.dw_reader")

DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"

# All DW RAW_MARKET fields the Execution Advisor needs
DW_FIELDS_REQUIRED = {
    # L2 Sentiment
    "AAII_BULL_PCTL",
    "AAII_BEAR_PCTL",
    "PUT_CALL_RATIO",
    "VIX_LEVEL",
    "CNN_FEAR_GREED",
    "HY_OAS_SPREAD",
    "INSIDER_BUY_SELL",
    "MOVE_INDEX",

    # L4 Positioning
    "COT_SP500_COMM_NET",
    "COT_GOLD_COMM_NET",
    "COT_TREASURY_COMM_NET",
    "FUND_FLOWS_EQUITY",
    "CRYPTO_FUNDING_RATE",
    "OPTIONS_GEX",

    # L5 Fragility
    "RESERVE_DRAIN_RATE",
    "SOFR_FFR_SPREAD",
    "FIN_STRESS_INDEX",
    "ON_RRP_USAGE",
    "SPY_CONCENTRATION",
    "LIQUIDITY_AMIHUD",
    "AVG_PAIRWISE_CORR",
    "VIX_TERM_STRUCTURE",

    # L7 Cross-Asset
    "BOND_EQUITY_CORR_60D",
    "GOLD_RETURN_20D",
    "DXY_RETURN_20D",
    "COPPER_SMA50_TREND",
    "SPY_SMA50_TREND",
    "REAL_YIELD_10Y_TREND",
    "YIELD_CURVE_10Y2Y",
}


def get_sheets_service():
    """Initialize Google Sheets API service."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
    if not creds_json:
        return None
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    return build("sheets", "v4", credentials=creds)


def read_dw_raw_market(sheets_service=None) -> dict:
    """
    Read all required fields from DW Sheet RAW_MARKET tab.

    RAW_MARKET tab structure:
    Column A: date
    Column B: indicator (indicator name)
    Column C: layer
    Column D: value

    Returns:
        {indicator_name: value_string, ...}
        Latest value per indicator (last row wins, chronologically sorted).
    """
    if sheets_service is None:
        sheets_service = get_sheets_service()

    if sheets_service is None:
        logger.warning("No Google credentials — DW Sheet unavailable")
        return {}

    try:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=DW_SHEET_ID,
            range="RAW_MARKET!A:D",
        ).execute()

        rows = result.get("values", [])

        if len(rows) <= 1:
            logger.warning("RAW_MARKET tab is empty or header-only")
            return {}

        # Get latest value per indicator (last row wins)
        latest = {}
        for row in rows[1:]:  # Skip header
            if len(row) < 4:
                continue
            indicator = row[1]
            value = row[3]

            if indicator in DW_FIELDS_REQUIRED:
                latest[indicator] = value

        found = len(latest)
        missing = DW_FIELDS_REQUIRED - set(latest.keys())

        logger.info(
            f"DW RAW_MARKET: {found}/{len(DW_FIELDS_REQUIRED)} fields loaded"
        )
        if missing:
            logger.warning(f"DW missing fields: {sorted(missing)}")

        return latest

    except Exception as e:
        logger.error(f"DW Sheet read failed: {e}")
        return {}
