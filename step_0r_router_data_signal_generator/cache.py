"""
step_0r_router_data_signal_generator/cache.py
Router History Cache — Parquet + JSON fallback
Stores 300+ days of daily prices for MA and return calculations.
"""

import json
import logging
import os

import pandas as pd
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("router_data.cache")

CACHE_DIR = "data/cache/router"
HISTORY_FILE = "router_history.parquet"
HISTORY_JSON = "router_history.json"
MAX_HISTORY_DAYS = 400  # Keep ~16 months for 252d returns + buffer


def _normalize_ts(ts):
    """Convert any timestamp to tz-naive for consistent indexing."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


class RouterHistoryCache:
    """Manages price history for Router data calculations."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.data: Dict[str, pd.Series] = {}

    def load(self) -> Dict[str, pd.Series]:
        parquet_path = os.path.join(self.cache_dir, HISTORY_FILE)
        json_path = os.path.join(self.cache_dir, HISTORY_JSON)

        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                self.data = {col: df[col].dropna() for col in df.columns}
                logger.info(
                    f"Cache loaded: {len(self.data)} series, "
                    f"{max(len(s) for s in self.data.values()) if self.data else 0} max days (parquet)"
                )
                return self.data
            except Exception as e:
                logger.warning(f"Parquet load failed: {e}, trying JSON")

        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    raw = json.load(f)
                for field_name, entries in raw.items():
                    dates = [pd.Timestamp(e["date"]) for e in entries]
                    values = [e["value"] for e in entries]
                    self.data[field_name] = pd.Series(values, index=dates, dtype=float)
                logger.info(f"Cache loaded: {len(self.data)} series from JSON")
                return self.data
            except Exception as e:
                logger.warning(f"JSON load failed: {e}")

        logger.info("No router cache found — will bootstrap from yfinance history")
        return self.data

    def save(self):
        if not self.data:
            return

        # Normalize all series to tz-naive before building DataFrame
        normalized = {}
        for key, series in self.data.items():
            if series is not None and len(series) > 0:
                idx = series.index
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                normalized[key] = pd.Series(series.values, index=idx, dtype=float)

        df = pd.DataFrame(normalized)

        parquet_path = os.path.join(self.cache_dir, HISTORY_FILE)
        try:
            df.to_parquet(parquet_path)
            logger.info(f"Cache saved: {len(df.columns)} series, {len(df)} days (parquet)")
        except Exception as e:
            logger.warning(f"Parquet save failed: {e}")

        json_path = os.path.join(self.cache_dir, HISTORY_JSON)
        try:
            raw = {}
            for col in df.columns:
                series = df[col].dropna()
                raw[col] = [
                    {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
                    for idx, val in series.items()
                ]
            with open(json_path, "w") as f:
                json.dump(raw, f)
        except Exception as e:
            logger.warning(f"JSON save failed: {e}")

    def prune(self):
        for field_name in list(self.data.keys()):
            if len(self.data[field_name]) > MAX_HISTORY_DAYS + 50:
                self.data[field_name] = self.data[field_name].tail(MAX_HISTORY_DAYS)

    def update(self, field_name: str, value: float, dt: datetime = None):
        if value is None:
            return
        ts = _normalize_ts(dt or datetime.now())
        if field_name not in self.data:
            self.data[field_name] = pd.Series(dtype=float)
        self.data[field_name][ts] = value

    def update_from_df(self, field_name: str, df_series: pd.Series):
        """Bulk-update from a pandas Series (for bootstrap from yfinance)."""
        if df_series is None or df_series.empty:
            return
        if field_name not in self.data:
            self.data[field_name] = pd.Series(dtype=float)
        for idx, val in df_series.items():
            if pd.notna(val):
                ts = _normalize_ts(idx)
                self.data[field_name][ts] = float(val)

    def get(self, field_name: str) -> Optional[pd.Series]:
        s = self.data.get(field_name)
        if s is not None:
            return s.dropna().sort_index()
        return None

    def days_available(self, field_name: str) -> int:
        s = self.data.get(field_name)
        if s is not None:
            return len(s.dropna())
        return 0
