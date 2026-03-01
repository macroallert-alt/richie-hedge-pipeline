"""
transforms.py — Phase 2: Zeitliche Transformationen
=====================================================
T1: value, delta_1d, delta_5d, delta_21d, pctl_1y, zscore_2y + direction, freshness, confidence
T2: value, delta_5d, pctl_1y + direction, freshness, confidence
T3: value + freshness, confidence
SYSTEM: Rohdaten wie sie sind
"""

import math
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any
from scipy.stats import percentileofscore
import json
import os

logger = logging.getLogger("data_collector.transforms")


# ═══════════════════════════════════════════════════════
# TRADING DAY HELPERS
# ═══════════════════════════════════════════════════════

class TradingCalendar:
    """US-Handelstag-Logik inkl. Feiertage und Wochenenden."""

    def __init__(self, holidays_path: str = "config/us_holidays.json"):
        self.holidays = set()
        if os.path.exists(holidays_path):
            with open(holidays_path) as f:
                data = json.load(f)
            for year_key, dates in data.items():
                if year_key.startswith('_'):
                    continue
                for d in dates:
                    self.holidays.add(d)

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        return d.strftime('%Y-%m-%d') not in self.holidays

    def get_previous_trading_day(self, d: date) -> date:
        candidate = d - timedelta(days=1)
        for _ in range(10):
            if self.is_trading_day(candidate):
                return candidate
            candidate -= timedelta(days=1)
        return d - timedelta(days=1)

    def get_n_trading_days_ago(self, d: date, n: int) -> date:
        candidate = d
        count = 0
        for _ in range(n * 3):
            candidate -= timedelta(days=1)
            if self.is_trading_day(candidate):
                count += 1
            if count >= n:
                break
        return candidate


# ═══════════════════════════════════════════════════════
# CONFIDENCE
# ═══════════════════════════════════════════════════════

BASE_CONFIDENCE = {
    "API_PRIMARY": 1.00,
    "API_SECONDARY": 0.95,
    "SCRAPE": 0.85,
    "LLM_FALLBACK": 0.70,
    "STALE_CACHE": 0.50,
    "ALL_FAILED": 0.00,
}

def compute_confidence(freshness_days: int, halflife_days: int, source_method: str) -> float:
    base = BASE_CONFIDENCE.get(source_method, 0.50)
    if base == 0.0:
        return 0.0
    if freshness_days <= 0:
        decay = 1.0
    else:
        decay = math.exp(-0.693 * freshness_days / max(halflife_days, 1))
    return round(base * decay, 3)


# ═══════════════════════════════════════════════════════
# FRESHNESS
# ═══════════════════════════════════════════════════════

def compute_freshness(source_date: date, today: date, update_freq: str) -> int:
    if source_date is None:
        return 999
    raw_freshness = (today - source_date).days
    if update_freq == "daily":
        # Weekend-Exemption: Mo nach Fr = frisch
        if today.weekday() == 0 and source_date.weekday() == 4 and raw_freshness <= 3:
            return 0
        if today.weekday() <= 1 and source_date.weekday() == 4 and raw_freshness <= 3:
            return 0
    return raw_freshness


# ═══════════════════════════════════════════════════════
# DIRECTION
# ═══════════════════════════════════════════════════════

def compute_direction(delta_5d: Optional[float], threshold: float) -> Optional[str]:
    if delta_5d is None:
        return None
    if delta_5d > threshold:
        return "UP"
    elif delta_5d < -threshold:
        return "DOWN"
    return "FLAT"


# ═══════════════════════════════════════════════════════
# DELTAS
# ═══════════════════════════════════════════════════════

def compute_deltas(value_today: float, history: pd.Series, today: date,
                   cal: TradingCalendar) -> Dict[str, Optional[float]]:
    result = {"delta_1d": None, "delta_5d": None, "delta_21d": None}
    if value_today is None or history is None or len(history) == 0:
        return result

    # Build date->value lookup
    hd = {}
    for idx, val in history.items():
        if pd.notna(val):
            d = idx.date() if hasattr(idx, 'date') else idx
            hd[d] = val

    # delta_1d
    prev = cal.get_previous_trading_day(today)
    v = _find_closest(hd, prev, 3)
    if v is not None:
        result["delta_1d"] = round(value_today - v, 6)

    # delta_5d
    d5 = cal.get_n_trading_days_ago(today, 5)
    v5 = _find_closest(hd, d5, 3)
    if v5 is not None:
        result["delta_5d"] = round(value_today - v5, 6)

    # delta_21d
    d21 = cal.get_n_trading_days_ago(today, 21)
    v21 = _find_closest(hd, d21, 5)
    if v21 is not None:
        result["delta_21d"] = round(value_today - v21, 6)

    return result


def _find_closest(hd: dict, target: date, tolerance: int) -> Optional[float]:
    for offset in range(tolerance + 1):
        for sign in [0, -1, 1]:
            check = target + timedelta(days=sign * offset)
            if check in hd:
                return hd[check]
    return None


# ═══════════════════════════════════════════════════════
# PERCENTILE & Z-SCORE
# ═══════════════════════════════════════════════════════

def compute_percentile_1y(value: float, history: pd.Series) -> Optional[float]:
    if value is None or history is None:
        return None
    valid = history.dropna()
    if len(valid) < 60:
        return None
    recent = valid.tail(252)
    return round(percentileofscore(recent.values, value, kind='rank'), 1)


def compute_zscore_2y(value: float, history: pd.Series) -> Optional[float]:
    if value is None or history is None:
        return None
    valid = history.dropna()
    if len(valid) < 60:
        return None
    recent = valid.tail(504)
    mean = float(np.mean(recent))
    std = float(np.std(recent))
    if std == 0:
        return 0.0
    return round((value - mean) / std, 3)


# ═══════════════════════════════════════════════════════
# TRANSFORMED FIELD
# ═══════════════════════════════════════════════════════

class TransformedField:
    """Vollstaendig transformiertes Feld."""

    def __init__(self, field_name: str, tier: str):
        self.field_name = field_name
        self.tier = tier
        self.category = None
        self.value = None
        self.delta_1d = None
        self.delta_5d = None
        self.delta_21d = None
        self.pctl_1y = None
        self.zscore_2y = None
        self.direction = None
        self.freshness = 999
        self.confidence = 0.0
        self.expected_freq = None
        self.halflife = None
        self.source = None
        self.source_method = "ALL_FAILED"
        self.timing_class = None
        self.source_date = None
        self.anomaly_flag = "OK"
        self.is_revision = False

    def to_dict(self) -> dict:
        d = {
            "field": self.field_name,
            "tier": self.tier,
            "category": self.category,
            "value": self.value,
            "freshness": self.freshness,
            "confidence": self.confidence,
            "source": self.source,
            "source_method": self.source_method,
            "timing_class": self.timing_class,
            "anomaly_flag": self.anomaly_flag,
            "is_revision": self.is_revision,
            "expected_freq": self.expected_freq,
            "halflife": self.halflife,
        }
        if self.tier == "T1":
            d.update({
                "delta_1d": self.delta_1d, "delta_5d": self.delta_5d,
                "delta_21d": self.delta_21d, "pctl_1y": self.pctl_1y,
                "zscore_2y": self.zscore_2y, "direction": self.direction,
            })
        elif self.tier == "T2":
            d.update({
                "delta_5d": self.delta_5d, "pctl_1y": self.pctl_1y,
                "direction": self.direction,
            })
        return d

    def __repr__(self):
        return f"TF({self.field_name}={self.value}, {self.tier}, c={self.confidence})"


# ═══════════════════════════════════════════════════════
# TRANSFORM ENGINE
# ═══════════════════════════════════════════════════════

class TransformEngine:
    """Wendet alle Transformationen auf FetchResults an."""

    def __init__(self, registry: dict, history_cache: dict,
                 holidays_path: str = "config/us_holidays.json"):
        self.registry = registry
        self.history = history_cache
        self.cal = TradingCalendar(holidays_path)

    def transform_all(self, fetch_results: dict, today: date = None) -> Dict[str, TransformedField]:
        today = today or datetime.now().date()
        transformed = {}

        logger.info("=" * 60)
        logger.info("PHASE 2: TRANSFORMATIONEN")
        logger.info("=" * 60)

        for field_name, fetch_result in fetch_results.items():
            if field_name.startswith('_'):
                continue

            cfg = self.registry.get(field_name)
            if cfg is None:
                continue

            tier = cfg.get('tier', 'T3')
            tf = TransformedField(field_name, tier)
            tf.value = fetch_result.value
            tf.source = fetch_result.source
            tf.source_method = fetch_result.source_method
            tf.category = cfg.get('category')
            tf.timing_class = cfg.get('timing_class')
            tf.expected_freq = cfg.get('update_freq')
            tf.halflife = cfg.get('halflife_days')

            # Source date
            sd = fetch_result.source_date
            if isinstance(sd, str):
                try:
                    sd = datetime.strptime(sd, '%Y-%m-%d').date()
                except (ValueError, TypeError):
                    sd = None
            tf.source_date = sd

            # Freshness + Confidence
            tf.freshness = compute_freshness(sd, today, cfg.get('update_freq', 'daily'))
            tf.confidence = compute_confidence(
                tf.freshness, cfg.get('halflife_days', 7), fetch_result.source_method
            )

            # History
            hist = self.history.get(field_name)

            # Tier-specific transforms
            if tier == "T1" and tf.value is not None:
                self._apply_t1(tf, hist, today, cfg)
            elif tier == "T2" and tf.value is not None:
                self._apply_t2(tf, hist, today, cfg)

            transformed[field_name] = tf

        ok = sum(1 for t in transformed.values() if t.value is not None)
        logger.info(f"PHASE 2 COMPLETE: {ok}/{len(transformed)} fields transformed")
        return transformed

    def _apply_t1(self, tf, hist, today, cfg):
        if hist is not None and len(hist) > 0:
            deltas = compute_deltas(tf.value, hist, today, self.cal)
            tf.delta_1d = deltas["delta_1d"]
            tf.delta_5d = deltas["delta_5d"]
            tf.delta_21d = deltas["delta_21d"]
            tf.pctl_1y = compute_percentile_1y(tf.value, hist)
            tf.zscore_2y = compute_zscore_2y(tf.value, hist)
        tf.direction = compute_direction(tf.delta_5d, cfg.get('direction_threshold', 0))

    def _apply_t2(self, tf, hist, today, cfg):
        if hist is not None and len(hist) > 0:
            deltas = compute_deltas(tf.value, hist, today, self.cal)
            tf.delta_5d = deltas["delta_5d"]
            tf.pctl_1y = compute_percentile_1y(tf.value, hist)
        tf.direction = compute_direction(tf.delta_5d, cfg.get('direction_threshold', 0))

    def update_history(self, transformed: Dict[str, TransformedField], today: date = None):
        """Updated History-Cache mit heutigen Werten."""
        today = today or datetime.now().date()
        for field_name, tf in transformed.items():
            if tf.value is None:
                continue
            if field_name not in self.history:
                self.history[field_name] = pd.Series(dtype=float)
            ts = pd.Timestamp(today)
            self.history[field_name][ts] = tf.value
            if len(self.history[field_name]) > 600:
                self.history[field_name] = self.history[field_name].tail(504)
