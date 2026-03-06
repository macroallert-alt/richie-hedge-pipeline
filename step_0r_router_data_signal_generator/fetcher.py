"""
step_0r_router_data_signal_generator/fetcher.py
Fetches Router raw data from yfinance and FRED.
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

    def read_credit_impulse_from_dashboard(self) -> Optional[float]:
        """
        Read V16 Credit Impulse from dashboard.json (local file).
        The V16 Daily Runner writes this daily.
        """
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "dashboard", "latest.json",
        )

        try:
            if not os.path.exists(dashboard_path):
                logger.warning(f"dashboard.json not found at {dashboard_path}")
                return None

            with open(dashboard_path, "r", encoding="utf-8") as f:
                dashboard = json.load(f)

            v16 = dashboard.get("v16", {})
            ci = v16.get("credit_impulse")
            if ci is not None:
                logger.info(f"  Credit Impulse from dashboard: {ci:.6f}")
                return float(ci)
            else:
                logger.warning("Credit Impulse not found in dashboard.json v16 block")
                return None
        except Exception as e:
            logger.warning(f"Dashboard read failed: {e}")
            return None
