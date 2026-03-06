"""
step_0r_router_data_signal_generator/fetcher.py
Fetches Router raw data from yfinance, FRED, and V16 Sheet.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf
from fredapi import Fred

logger = logging.getLogger("router_data.fetcher")

# V16 Production Sheet
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"


class RouterFetcher:
    """Fetches all raw data needed for the Conviction Router."""

    def __init__(self, config: dict):
        self.config = config
        self.history_days = config.get("history_days", 300)

        # FRED
        fred_key = os.environ.get("FRED_API_KEY", "")
        self.fred = Fred(api_key=fred_key) if fred_key else None
        if not fred_key:
            logger.warning("FRED_API_KEY not set — FRED fetches will fail")

    def fetch_yfinance_prices(self) -> Dict[str, pd.Series]:
        """
        Fetch price history for all Router assets via yfinance.
        Fetches each ticker individually with a short pause to avoid rate limits.
        Returns dict of {asset_key: pd.Series of close prices}.
        """
        assets = self.config.get("yfinance_assets", {})
        start_date = (datetime.now() - timedelta(days=self.history_days + 30)).strftime("%Y-%m-%d")
        logger.info(f"yfinance: Fetching {len(assets)} assets individually, {self.history_days}d history")

        results = {}
        for key, cfg in assets.items():
            ticker = cfg["ticker"]
            try:
                t = yf.Ticker(ticker)
                hist = t.history(start=start_date, auto_adjust=True)

                if hist is not None and not hist.empty and "Close" in hist.columns:
                    close = hist["Close"].dropna()
                    if not close.empty:
                        results[key] = close
                        logger.info(
                            f"  {key:8s} ({ticker:12s}): {len(close):>4d} days, "
                            f"latest={close.iloc[-1]:.4f} ({close.index[-1].strftime('%Y-%m-%d')})"
                        )
                    else:
                        logger.warning(f"  {key:8s} ({ticker:12s}): NO DATA (empty after dropna)")
                else:
                    logger.warning(f"  {key:8s} ({ticker:12s}): NO DATA")
            except Exception as e:
                logger.warning(f"  {key:8s} ({ticker:12s}): ERROR — {e}")

            # Short pause between tickers to avoid rate limiting
            time.sleep(0.5)

        return results

    def fetch_fred_bamlem(self) -> Optional[Tuple[pd.Series, float, str]]:
        """
        Fetch BAMLEMCBPIOAS (EM Corporate Bond OAS) from FRED.
        Returns (full_series, latest_value, latest_date) or None.
        """
        if not self.fred:
            logger.warning("FRED not available — skipping BAMLEM")
            return None

        fred_cfg = self.config.get("fred_series", {}).get("BAMLEM", {})
        series_id = fred_cfg.get("series_id", "BAMLEMCBPIOAS")

        try:
            start_date = (datetime.now() - timedelta(days=self.history_days + 30)).strftime("%Y-%m-%d")
            series = self.fred.get_series(series_id, observation_start=start_date)

            if series is not None and not series.dropna().empty:
                valid = series.dropna()
                latest_val = float(valid.iloc[-1])
                latest_date = valid.index[-1].strftime("%Y-%m-%d")
                logger.info(
                    f"  BAMLEM  ({series_id:12s}): {len(valid):>4d} days, "
                    f"latest={latest_val:.2f} ({latest_date})"
                )
                return valid, latest_val, latest_date
            else:
                logger.warning(f"FRED {series_id}: no data returned")
                return None
        except Exception as e:
            logger.error(f"FRED {series_id} fetch failed: {e}")
            return None

    def read_credit_impulse_from_v16_sheet(self) -> Optional[float]:
        """
        Read Credit Impulse from V16 Production Sheet.
        Tab: DATA_K16_K17, Column E (Credit_Impulse), newest row (prepend_row mode).
        Uses the same GOOGLE_CREDENTIALS as Drive writes.
        """
        try:
            creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
            if not creds_json:
                logger.warning("GOOGLE_CREDENTIALS not set — cannot read V16 Sheet")
                return None

            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds_info = json.loads(creds_json)
            creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
            )
            sheets = build("sheets", "v4", credentials=creds)

            # Read rows 2-6 (header in row 1, newest data near top due to prepend)
            range_str = "DATA_K16_K17!A2:E6"
            result = sheets.spreadsheets().values().get(
                spreadsheetId=V16_SHEET_ID,
                range=range_str,
                valueRenderOption="UNFORMATTED_VALUE",
            ).execute()

            rows = result.get("values", [])
            if not rows:
                logger.warning("V16 Sheet DATA_K16_K17: no data rows returned")
                return None

            # Find first row with a valid Credit_Impulse value (column E = index 4)
            for row in rows:
                if len(row) >= 5:
                    ci_raw = row[4]
                    row_date = row[0] if len(row) > 0 else "?"
                    if ci_raw is not None and ci_raw != "" and ci_raw != 0:
                        try:
                            ci_val = float(ci_raw)
                            logger.info(f"  Credit Impulse from V16 Sheet: {ci_val:.6f} (date: {row_date})")
                            return ci_val
                        except (ValueError, TypeError):
                            continue

            logger.warning("V16 Sheet DATA_K16_K17: no valid Credit_Impulse found in rows 2-6")
            return None

        except Exception as e:
            logger.warning(f"V16 Sheet Credit Impulse read failed: {e}")
            return None
