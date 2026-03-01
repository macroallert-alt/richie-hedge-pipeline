"""
cache.py — History Cache Management
=====================================
Persistent storage fuer 2-Jahres-History aller Felder.
Nutzt Parquet (schnell, kompakt) mit JSON-Fallback.
"""

import json
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger("data_collector.cache")

CACHE_DIR = "data/cache"
HISTORY_FILE = "field_history.parquet"
HISTORY_JSON = "field_history.json"
FETCH_CACHE_FILE = "last_fetch.json"
MAX_HISTORY_DAYS = 504  # 2 Jahre Handelstage


class HistoryCache:
    """Verwaltet 2-Jahres-Rolling-History fuer alle Felder."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.data: Dict[str, pd.Series] = {}

    def load(self) -> Dict[str, pd.Series]:
        """Laedt History aus Parquet oder JSON."""
        parquet_path = os.path.join(self.cache_dir, HISTORY_FILE)
        json_path = os.path.join(self.cache_dir, HISTORY_JSON)

        # Try Parquet first (faster)
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                self.data = {col: df[col].dropna() for col in df.columns}
                logger.info(f"Loaded history: {len(self.data)} fields, "
                           f"{len(df)} days from parquet")
                return self.data
            except Exception as e:
                logger.warning(f"Parquet load failed: {e}, trying JSON")

        # Fallback JSON
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    raw = json.load(f)
                for field_name, entries in raw.items():
                    dates = [pd.Timestamp(e['date']) for e in entries]
                    values = [e['value'] for e in entries]
                    self.data[field_name] = pd.Series(values, index=dates, dtype=float)
                logger.info(f"Loaded history: {len(self.data)} fields from JSON")
                return self.data
            except Exception as e:
                logger.warning(f"JSON load failed: {e}")

        logger.info("No history cache found — starting fresh")
        return self.data

    def save(self):
        """Speichert History als Parquet + JSON backup."""
        if not self.data:
            return

        # Build DataFrame
        df = pd.DataFrame(self.data)

        # Parquet
        parquet_path = os.path.join(self.cache_dir, HISTORY_FILE)
        try:
            df.to_parquet(parquet_path)
            logger.info(f"Saved history: {len(df.columns)} fields, {len(df)} days (parquet)")
        except Exception as e:
            logger.warning(f"Parquet save failed: {e}")

        # JSON backup
        json_path = os.path.join(self.cache_dir, HISTORY_JSON)
        try:
            raw = {}
            for col in df.columns:
                series = df[col].dropna()
                raw[col] = [
                    {"date": idx.strftime('%Y-%m-%d'), "value": float(val)}
                    for idx, val in series.items()
                ]
            with open(json_path, 'w') as f:
                json.dump(raw, f)
        except Exception as e:
            logger.warning(f"JSON save failed: {e}")

    def prune(self):
        """Trimmt History auf MAX_HISTORY_DAYS."""
        for field_name in list(self.data.keys()):
            if len(self.data[field_name]) > MAX_HISTORY_DAYS + 100:
                self.data[field_name] = self.data[field_name].tail(MAX_HISTORY_DAYS)

    def update(self, field_name: str, value: float, dt: datetime = None):
        """Fuegt einen neuen Wert hinzu."""
        if value is None:
            return
        ts = pd.Timestamp(dt or datetime.now())
        if field_name not in self.data:
            self.data[field_name] = pd.Series(dtype=float)
        self.data[field_name][ts] = value

    def get(self, field_name: str) -> Optional[pd.Series]:
        """Holt History fuer ein Feld."""
        return self.data.get(field_name)

    def get_latest(self, field_name: str) -> Optional[dict]:
        """Holt den letzten bekannten Wert."""
        series = self.data.get(field_name)
        if series is not None and len(series) > 0:
            valid = series.dropna()
            if len(valid) > 0:
                return {
                    "value": float(valid.iloc[-1]),
                    "date": valid.index[-1].strftime('%Y-%m-%d'),
                    "source": "cache"
                }
        return None


class FetchCache:
    """Speichert letzten erfolgreichen Fetch pro Feld fuer Stale-Fallback."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.path = os.path.join(cache_dir, FETCH_CACHE_FILE)
        self.data = {}

    def load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
                logger.info(f"Loaded fetch cache: {len(self.data)} fields")
            except Exception as e:
                logger.warning(f"Fetch cache load failed: {e}")
        return self.data

    def save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Fetch cache save failed: {e}")

    def update_from_results(self, fetch_results: dict):
        """Updated Fetch-Cache mit erfolgreichen Ergebnissen."""
        for field_name, result in fetch_results.items():
            if field_name.startswith('_'):
                continue
            if result.success and result.source_method != "STALE_CACHE":
                self.data[field_name] = {
                    "value": result.value,
                    "source": result.source,
                    "source_method": result.source_method,
                    "date": str(result.source_date),
                    "cached_at": datetime.now().isoformat(),
                }

    def get(self, field_name: str) -> Optional[dict]:
        return self.data.get(field_name)
