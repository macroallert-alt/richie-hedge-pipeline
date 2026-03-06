"""
step_0r_router_data_signal_generator/calculator.py
Computes MAs, returns, deltas, z-scores from cached price history.
All calculations the Conviction Router needs.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger("router_data.calculator")


def compute_ma(series: pd.Series, window: int) -> Optional[float]:
    """Compute simple moving average. Returns None if insufficient data."""
    if series is None or len(series) < window:
        return None
    return round(float(series.tail(window).mean()), 4)


def compute_return(series: pd.Series, lookback_days: int) -> Optional[float]:
    """
    Compute percentage return over lookback_days trading days.
    return = (price_today / price_N_days_ago) - 1
    """
    if series is None or len(series) < lookback_days + 1:
        return None
    try:
        current = float(series.iloc[-1])
        past = float(series.iloc[-(lookback_days + 1)])
        if past == 0:
            return None
        return round(current / past - 1, 6)
    except (IndexError, ValueError):
        return None


def compute_delta(series: pd.Series, lookback_days: int) -> Optional[float]:
    """
    Compute absolute change over lookback_days.
    delta = value_today - value_N_days_ago
    Used for BAMLEM OAS changes and DXY.
    """
    if series is None or len(series) < lookback_days + 1:
        return None
    try:
        current = float(series.iloc[-1])
        past = float(series.iloc[-(lookback_days + 1)])
        return round(current - past, 6)
    except (IndexError, ValueError):
        return None


def compute_pct_change(series: pd.Series, lookback_days: int) -> Optional[float]:
    """
    Compute percentage change (for DXY delta_126d as fraction).
    pct_change = (value_today - value_N_days_ago) / value_N_days_ago
    """
    if series is None or len(series) < lookback_days + 1:
        return None
    try:
        current = float(series.iloc[-1])
        past = float(series.iloc[-(lookback_days + 1)])
        if past == 0:
            return None
        return round((current - past) / past, 6)
    except (IndexError, ValueError):
        return None


def compute_zscore(value: float, series: pd.Series, window: int = 504) -> Optional[float]:
    """
    Compute z-score of value relative to last `window` observations.
    Default 504 = ~2 years of trading days.
    """
    if value is None or series is None:
        return None
    valid = series.dropna()
    if len(valid) < 60:
        return None
    recent = valid.tail(window)
    mean = float(np.mean(recent))
    std = float(np.std(recent))
    if std == 0:
        return 0.0
    return round((value - mean) / std, 4)


def build_router_raw_data(cache_data: dict, bamlem_series: Optional[pd.Series],
                          credit_impulse: Optional[float], config: dict) -> dict:
    """
    Build the complete router_raw_data.json output from cached history.

    Args:
        cache_data: dict of {asset_key: pd.Series} from RouterHistoryCache
        bamlem_series: BAMLEM OAS full series (or None)
        credit_impulse: V16 Credit Impulse value (or None)
        config: router_assets.json config

    Returns:
        dict matching step0r_router_data.json schema
    """
    assets_cfg = config.get("yfinance_assets", {})
    stale_fields = []

    def _get(key: str) -> Optional[pd.Series]:
        s = cache_data.get(key)
        if s is not None:
            return s.dropna().sort_index()
        return None

    def _latest(key: str) -> Optional[float]:
        s = _get(key)
        if s is not None and len(s) > 0:
            return round(float(s.iloc[-1]), 4)
        stale_fields.append(key)
        return None

    # === Individual asset blocks ===

    # DXY
    dxy_series = _get("DXY")
    dxy_block = {
        "value": _latest("DXY"),
        "delta_126d": compute_pct_change(dxy_series, 126),
        "delta_63d": compute_pct_change(dxy_series, 63),
    }

    # VWO
    vwo_series = _get("VWO")
    vwo_block = {
        "price": _latest("VWO"),
        "ma_50d": compute_ma(vwo_series, 50),
        "ma_200d": compute_ma(vwo_series, 200),
        "return_63d": compute_return(vwo_series, 63),
        "return_126d": compute_return(vwo_series, 126),
        "return_252d": compute_return(vwo_series, 252),
    }

    # SPY
    spy_series = _get("SPY")
    spy_block = {
        "price": _latest("SPY"),
        "return_63d": compute_return(spy_series, 63),
        "return_126d": compute_return(spy_series, 126),
    }

    # BAMLEM
    bamlem_block = {
        "value": round(float(bamlem_series.iloc[-1]), 4) if bamlem_series is not None and len(bamlem_series) > 0 else None,
        "delta_63d": compute_delta(bamlem_series, 63) if bamlem_series is not None else None,
    }
    if bamlem_block["value"] is None:
        stale_fields.append("BAMLEM")

    # FXI
    fxi_series = _get("FXI")
    fxi_block = {
        "price": _latest("FXI"),
        "ma_50d": compute_ma(fxi_series, 50),
        "ma_200d": compute_ma(fxi_series, 200),
        "return_63d": compute_return(fxi_series, 63),
        "return_126d": compute_return(fxi_series, 126),
    }

    # USDCNH
    usdcnh_series = _get("USDCNH")
    usdcnh_block = {
        "value": _latest("USDCNH"),
        "delta_63d": compute_pct_change(usdcnh_series, 63),
    }

    # China Credit Impulse
    ci_zscore = None
    ci_series = _get("credit_impulse")
    if credit_impulse is not None and ci_series is not None:
        ci_zscore = compute_zscore(credit_impulse, ci_series)
    china_ci_block = {
        "value": credit_impulse,
        "zscore_2y": ci_zscore,
    }

    # DBC
    dbc_series = _get("DBC")
    dbc_block = {
        "price": _latest("DBC"),
        "return_63d": compute_return(dbc_series, 63),
        "return_126d": compute_return(dbc_series, 126),
    }

    # GLD
    gld_series = _get("GLD")
    gld_block = {
        "price": _latest("GLD"),
        "return_126d": compute_return(gld_series, 126),
    }

    # === Relative performance (pre-computed for Router) ===
    relative = {
        "vwo_spy_63d": _relative(vwo_block.get("return_63d"), spy_block.get("return_63d")),
        "vwo_spy_126d": _relative(vwo_block.get("return_126d"), spy_block.get("return_126d")),
        "fxi_spy_63d": _relative(fxi_block.get("return_63d"), spy_block.get("return_63d")),
        "dbc_spy_63d": _relative(dbc_block.get("return_63d"), spy_block.get("return_63d")),
        "dbc_spy_126d": _relative(dbc_block.get("return_126d"), spy_block.get("return_126d")),
    }

    # === Data quality ===
    all_assets = ["VWO", "FXI", "SPY", "DBC", "GLD", "DXY", "USDCNH"]
    assets_ok = sum(1 for a in all_assets if _get(a) is not None and len(_get(a)) > 0)

    data_quality = {
        "assets_fetched": len(all_assets),
        "assets_ok": assets_ok,
        "fred_ok": bamlem_block["value"] is not None,
        "credit_impulse_ok": credit_impulse is not None,
        "cache_days": {
            key: len(_get(key)) if _get(key) is not None else 0
            for key in all_assets
        },
        "stale_fields": stale_fields,
    }

    return {
        "dxy": dxy_block,
        "vwo": vwo_block,
        "spy": spy_block,
        "bamlem": bamlem_block,
        "fxi": fxi_block,
        "usdcnh": usdcnh_block,
        "china_credit_impulse": china_ci_block,
        "dbc": dbc_block,
        "gld": gld_block,
        "relative": relative,
        "data_quality": data_quality,
    }


def _relative(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Compute relative return: a - b."""
    if a is not None and b is not None:
        return round(a - b, 6)
    return None
