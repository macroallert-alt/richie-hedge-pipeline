"""
L5 Fragility Collector - step_0e_l5_fragility/main.py
Global Macro RV System - Data Warehouse Layer 5
8/8 Indicators:
  1. RESERVE_DRAIN_RATE   — FRED WRESBAL 4w-delta (Liquidity drain speed)
  2. SOFR_FFR_SPREAD      — FRED SOFR-EFFR (Overnight funding stress)
  3. FIN_STRESS_INDEX     — FRED STLFSI2 (St. Louis Fed Financial Stress)
  4. ON_RRP_USAGE         — FRED RRPONTSYD (Liquidity reserve buffer)
  5. SPY_CONCENTRATION    — yfinance SPY HHI (Market concentration risk)
  6. LIQUIDITY_AMIHUD     — yfinance SPY Amihud ratio (Market liquidity)
  7. AVG_PAIRWISE_CORR    — yfinance V16 60d rolling (Diversification erosion)
  8. VIX_TERM_STRUCTURE   — yfinance VIX/VIX3M (Vol regime stress)
Score: 0 = stable, 10 = extremely fragile.
Writes to: RAW_MARKET (R20-R27), SCORES (R6), DASHBOARD (R21)
Pattern: Identical to L2/L4 Collectors (Pull -> Normalize -> Composite -> Sheets)
Reference: V79 Statusanalyse Kap.10
Iron Rule: V16 main.py NEVER TOUCH. L2 step_0c NEVER TOUCH. L4 step_0d NEVER TOUCH.
"""

import os
import sys
import logging
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("L5_FRAGILITY")


# ==================== CONSTANTS ====================

INDICATOR_WEIGHTS = {
    "RESERVE_DRAIN_RATE":  0.15,
    "SOFR_FFR_SPREAD":     0.15,
    "FIN_STRESS_INDEX":    0.15,
    "ON_RRP_USAGE":        0.10,
    "SPY_CONCENTRATION":   0.10,
    "LIQUIDITY_AMIHUD":    0.10,
    "AVG_PAIRWISE_CORR":   0.15,
    "VIX_TERM_STRUCTURE":  0.10,
}

# Normalization ranges: (lo, hi, invert?)
# Score 0 = stable, 10 = fragile
# invert=True means lower raw value = MORE fragile
INDICATOR_RANGES = {
    "RESERVE_DRAIN_RATE":  (-300.0, 100.0, True),    # $B 4w-delta; big drain (-300) = fragile(10), growth (+100) = stable(0)
    "SOFR_FFR_SPREAD":     (-2.0, 20.0, False),      # bps; negative = stable, +20 = very stressed
    "FIN_STRESS_INDEX":    (-1.5, 3.0, False),        # index; -1.5 = calm, +3.0 = crisis
    "ON_RRP_USAGE":        (0.0, 500.0, True),        # $B; $500B = big buffer(0), $0 = no buffer(10)
    "SPY_CONCENTRATION":   (0.03, 0.12, False),       # HHI; 0.03 = diverse(0), 0.12 = concentrated(10)
    "LIQUIDITY_AMIHUD":    (0.0, 100.0, False),       # percentile 0-100 of 1Y; 0 = liquid, 100 = illiquid
    "AVG_PAIRWISE_CORR":   (0.15, 0.75, False),       # correlation; 0.15 = diverse(0), 0.75 = herding(10)
    "VIX_TERM_STRUCTURE":  (0.75, 1.25, False),       # ratio VIX/VIX3M; 0.75 = contango/calm(0), 1.25 = backwardation/panic(10)
}

# Fragility phases (0 = stable -> 10 = fragile)
FRAGILITY_PHASES = [
    (1.5,  "Minimal"),
    (3.0,  "Low"),
    (4.5,  "Moderate"),
    (6.0,  "Elevated"),
    (7.5,  "High"),
    (9.0,  "Severe"),
    (10.0, "Critical"),
]

# Phase -> SIGNAL mapping
PHASE_TO_SIGNAL = {
    "Minimal":  "Stable",
    "Low":      "Stable",
    "Moderate": "Caution",
    "Elevated": "Caution",
    "High":     "Fragile",
    "Severe":   "Fragile",
    "Critical": "Critical",
}

# Freshness parameters
FRESHNESS_PARAMS = {
    "RESERVE_DRAIN_RATE":  {"ideal": 3, "decay_rate": 0.5},   # Weekly FRED, slow decay
    "SOFR_FFR_SPREAD":     {"ideal": 1, "decay_rate": 2.0},   # Daily FRED
    "FIN_STRESS_INDEX":    {"ideal": 5, "decay_rate": 0.5},   # Weekly FRED
    "ON_RRP_USAGE":        {"ideal": 1, "decay_rate": 2.0},   # Daily FRED
    "SPY_CONCENTRATION":   {"ideal": 30, "decay_rate": 0.1},  # Quarterly, very slow decay
    "LIQUIDITY_AMIHUD":    {"ideal": 1, "decay_rate": 2.0},   # Daily
    "AVG_PAIRWISE_CORR":   {"ideal": 1, "decay_rate": 2.0},   # Daily
    "VIX_TERM_STRUCTURE":  {"ideal": 1, "decay_rate": 2.0},   # Daily
}

# Source metadata for RAW_MARKET
SOURCE_MAP = {
    "RESERVE_DRAIN_RATE":  ("FRED/WRESBAL", "T1", "$B_4w"),
    "SOFR_FFR_SPREAD":     ("FRED", "T1", "bps"),
    "FIN_STRESS_INDEX":    ("FRED/STLFSI2", "T1", "index"),
    "ON_RRP_USAGE":        ("FRED/RRPONTSYD", "T1", "$B"),
    "SPY_CONCENTRATION":   ("yfinance", "T2", "HHI"),
    "LIQUIDITY_AMIHUD":    ("yfinance", "T2", "pctl"),
    "AVG_PAIRWISE_CORR":   ("yfinance", "T2", "corr"),
    "VIX_TERM_STRUCTURE":  ("yfinance", "T2", "ratio"),
}

# V16 asset universe for pairwise correlation (ETF tickers)
V16_CORRELATION_TICKERS = [
    "GLD", "SLV",              # Precious Metals
    "SPY", "QQQ", "EFA", "EEM", "VWO",  # Equities
    "TLT", "IEF", "SHY", "LQD", "HYG",  # Bonds
    "BTC-USD", "ETH-USD",      # Crypto
]

MIN_SOURCES_FOR_COMPOSITE = 5


# ==================== WAREHOUSE CONNECTION ====================

def connect_warehouse():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    gc = gspread.authorize(creds)
    return gc.open_by_key(WAREHOUSE_SHEET_ID)


# ==================== FRED PULLS ====================

def _fred_latest(series_id, lookback_days=90):
    """Fetch latest value from FRED API for a given series.
    Returns (value, date_str, age_days) or (None, None, None) on failure.
    """
    from fredapi import Fred

    if not FRED_API_KEY:
        log.warning(f"    FRED: No API key, skipping {series_id}")
        return None, None, None

    try:
        fred = Fred(api_key=FRED_API_KEY)
        end = date.today()
        start = end - timedelta(days=lookback_days)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s = s.dropna()
        if s.empty:
            log.warning(f"    FRED {series_id}: No data in last {lookback_days}d")
            return None, None, None
        latest_val = float(s.iloc[-1])
        latest_date = s.index[-1].date()
        age_days = (end - latest_date).days
        return latest_val, latest_date.strftime("%Y-%m-%d"), age_days
    except Exception as e:
        log.warning(f"    FRED {series_id}: Failed -> {e}")
        return None, None, None


def _fred_series(series_id, lookback_days=90):
    """Fetch full series from FRED API.
    Returns pandas Series (date-indexed) or empty Series on failure.
    """
    from fredapi import Fred

    if not FRED_API_KEY:
        return pd.Series(dtype=float)

    try:
        fred = Fred(api_key=FRED_API_KEY)
        end = date.today()
        start = end - timedelta(days=lookback_days)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        return s.dropna()
    except Exception as e:
        log.warning(f"    FRED {series_id}: Series fetch failed -> {e}")
        return pd.Series(dtype=float)


def pull_reserve_drain_rate():
    """Pull Fed Bank Reserves (WRESBAL) and compute 4-week delta.
    WRESBAL = Weekly, $Millions. We convert to $Billions.
    4-week delta: current week minus 4 weeks ago.
    Negative delta = reserves draining = system more fragile.
    """
    results = {}
    today = date.today()
    log.info("  RESERVE_DRAIN: Pulling WRESBAL from FRED...")

    s = _fred_series("WRESBAL", lookback_days=120)
    if len(s) < 5:
        log.warning("    RESERVE_DRAIN: Not enough data points")
        return results

    # Convert millions to billions
    s_b = s / 1000.0

    latest_val = float(s_b.iloc[-1])
    latest_date = s_b.index[-1].date()

    # Find value ~4 weeks ago (WRESBAL is weekly, so 4 observations back)
    if len(s_b) >= 5:
        prev_val = float(s_b.iloc[-5])
    else:
        prev_val = float(s_b.iloc[0])

    delta_4w = latest_val - prev_val
    age_days = (today - latest_date).days

    results["RESERVE_DRAIN_RATE"] = {
        "raw_value": round(delta_4w, 2),
        "age_days": age_days,
        "date": latest_date.strftime("%Y-%m-%d"),
    }
    log.info(f"    RESERVE_DRAIN: {delta_4w:+.1f}$B (4w delta), Current={latest_val:.0f}$B, Date={latest_date}, Age={age_days}d")
    return results


def pull_sofr_ffr_spread():
    """Pull SOFR - EFFR spread from FRED.
    SOFR = Secured Overnight Financing Rate
    EFFR = Effective Federal Funds Rate
    Spread > 0 = repo market stressed, collateral scarce.
    Units: basis points (multiply pct difference by 100).
    """
    results = {}
    today = date.today()
    log.info("  SOFR_FFR: Pulling SOFR and EFFR from FRED...")

    sofr_val, sofr_date, sofr_age = _fred_latest("SOFR")
    effr_val, effr_date, effr_age = _fred_latest("EFFR")

    if sofr_val is None or effr_val is None:
        log.warning("    SOFR_FFR: Missing SOFR or EFFR data")
        return results

    spread_bps = (sofr_val - effr_val) * 100.0
    use_date = sofr_date
    age_days = sofr_age

    results["SOFR_FFR_SPREAD"] = {
        "raw_value": round(spread_bps, 2),
        "age_days": age_days,
        "date": use_date,
    }
    log.info(f"    SOFR_FFR: {spread_bps:+.1f}bps (SOFR={sofr_val:.2f}%, EFFR={effr_val:.2f}%), Date={use_date}, Age={age_days}d")
    return results


def pull_fin_stress_index():
    """Pull St. Louis Fed Financial Stress Index (STLFSI2) from FRED.
    Weekly composite of 18 financial market indicators.
    0 = normal, negative = calm, >1 = elevated, >2 = crisis.
    """
    results = {}
    log.info("  FIN_STRESS: Pulling STLFSI2 from FRED...")

    val, dt, age = _fred_latest("STLFSI2")
    if val is None:
        log.warning("    FIN_STRESS: No data")
        return results

    results["FIN_STRESS_INDEX"] = {
        "raw_value": round(val, 4),
        "age_days": age,
        "date": dt,
    }
    log.info(f"    FIN_STRESS: {val:.4f} (0=normal, >1=elevated, >2=crisis), Date={dt}, Age={age}d")
    return results


def pull_on_rrp_usage():
    """Pull Overnight Reverse Repo (RRPONTSYD) from FRED.
    When ON RRP is high = excess liquidity parked at Fed = buffer exists.
    When ON RRP -> 0 = no buffer left = system fragile.
    Units: $Billions.
    """
    results = {}
    log.info("  ON_RRP: Pulling RRPONTSYD from FRED...")

    val, dt, age = _fred_latest("RRPONTSYD")
    if val is None:
        log.warning("    ON_RRP: No data")
        return results

    # RRPONTSYD is in $Billions already
    results["ON_RRP_USAGE"] = {
        "raw_value": round(val, 2),
        "age_days": age,
        "date": dt,
    }
    log.info(f"    ON_RRP: ${val:.1f}B (high=buffer, low=fragile), Date={dt}, Age={age}d")
    return results


# ==================== YFINANCE PULLS ====================

def pull_spy_concentration():
    """Pull SPY top holdings and calculate HHI (Herfindahl-Hirschman Index).
    HHI = sum of squared weight fractions.
    HHI ~0.03 = well diversified, HHI ~0.10+ = concentrated.
    High concentration = fragile (single-name risk).
    """
    import yfinance as yf

    results = {}
    today = date.today()
    log.info("  SPY_HHI: Pulling SPY holdings via yfinance...")

    try:
        spy = yf.Ticker("SPY")
        data = spy.funds_data

        # Get top holdings
        holdings = data.top_holdings
        if holdings is None or (hasattr(holdings, 'empty') and holdings.empty):
            log.warning("    SPY_HHI: No holdings data returned")
            return results

        # holdings is a Series or DataFrame with holding weights
        # Weights are fractions (e.g., 0.07 = 7%)
        if isinstance(holdings, pd.DataFrame):
            if "Holding Percent" in holdings.columns:
                weights = holdings["Holding Percent"].values
            else:
                # Try first numeric column
                numeric_cols = holdings.select_dtypes(include=[np.number])
                if numeric_cols.empty:
                    log.warning("    SPY_HHI: No numeric columns in holdings")
                    return results
                weights = numeric_cols.iloc[:, 0].values
        elif isinstance(holdings, pd.Series):
            weights = holdings.values
        else:
            log.warning(f"    SPY_HHI: Unexpected holdings type: {type(holdings)}")
            return results

        weights = np.array([float(w) for w in weights if not pd.isna(w)])
        if len(weights) == 0:
            log.warning("    SPY_HHI: No valid weights")
            return results

        # Ensure weights are fractions (not percentages)
        if weights.max() > 1.0:
            weights = weights / 100.0

        hhi = float(np.sum(weights ** 2))
        n_holdings = len(weights)
        top_weight = float(weights.max()) if len(weights) > 0 else 0.0

        results["SPY_CONCENTRATION"] = {
            "raw_value": round(hhi, 6),
            "age_days": 0,  # Holdings data doesn't have a date, treat as current
            "date": today.strftime("%Y-%m-%d"),
        }
        log.info(f"    SPY_HHI: {hhi:.4f} (from {n_holdings} holdings, top={top_weight:.1%})")

    except Exception as e:
        log.warning(f"    SPY_HHI: Failed -> {e}")

    return results


def pull_liquidity_amihud():
    """Calculate Amihud Illiquidity Ratio for SPY.
    Amihud = mean(|daily_return| / dollar_volume) over 20 days.
    Higher = less liquid = more fragile.
    We express as percentile of 1-year history for normalization.
    """
    import yfinance as yf

    results = {}
    today = date.today()
    log.info("  AMIHUD: Pulling SPY daily data via yfinance...")

    try:
        spy = yf.download("SPY", period="1y", interval="1d", progress=False)
        if spy.empty or len(spy) < 30:
            log.warning("    AMIHUD: Not enough SPY data")
            return results

        # Handle MultiIndex columns from yfinance
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        close = spy["Close"].values.flatten()
        volume = spy["Volume"].values.flatten()

        # Daily returns
        returns = np.abs(np.diff(close) / close[:-1])
        # Dollar volume (in billions for numerical stability)
        dollar_vol = (close[1:] * volume[1:]) / 1e9

        # Amihud ratio per day (avoid div by zero)
        valid = dollar_vol > 0
        amihud_daily = np.where(valid, returns / dollar_vol, np.nan)
        amihud_series = pd.Series(amihud_daily)

        # 20-day rolling mean
        amihud_20d = amihud_series.rolling(20, min_periods=15).mean()
        current_amihud = amihud_20d.iloc[-1]

        if pd.isna(current_amihud):
            log.warning("    AMIHUD: Current value is NaN")
            return results

        # Percentile of current value within 1Y history
        valid_history = amihud_20d.dropna()
        if len(valid_history) < 20:
            percentile = 50.0
        else:
            percentile = float((valid_history < current_amihud).sum()) / len(valid_history) * 100.0

        # Get the latest trading date
        latest_date = spy.index[-1]
        if hasattr(latest_date, "date"):
            latest_date = latest_date.date()
        age_days = (today - latest_date).days

        results["LIQUIDITY_AMIHUD"] = {
            "raw_value": round(percentile, 2),
            "age_days": age_days,
            "date": latest_date.strftime("%Y-%m-%d"),
        }
        log.info(f"    AMIHUD: {percentile:.1f}th percentile (1Y), raw={current_amihud:.6f}, Date={latest_date}, Age={age_days}d")

    except Exception as e:
        log.warning(f"    AMIHUD: Failed -> {e}")

    return results


def pull_avg_pairwise_corr():
    """Calculate average pairwise correlation of V16 universe assets.
    60-day rolling window. Upper triangle mean of correlation matrix.
    Higher correlation = less diversification = more fragile.
    """
    import yfinance as yf

    results = {}
    today = date.today()
    log.info("  PAIRWISE_CORR: Pulling V16 universe data via yfinance...")

    try:
        tickers = V16_CORRELATION_TICKERS
        data = yf.download(tickers, period="90d", interval="1d", progress=False)

        if data.empty:
            log.warning("    PAIRWISE_CORR: No data returned")
            return results

        # Extract close prices
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]

        # Drop tickers with too little data
        close = close.dropna(axis=1, thresh=50)

        if close.shape[1] < 5:
            log.warning(f"    PAIRWISE_CORR: Only {close.shape[1]} tickers with enough data")
            return results

        # Daily returns
        returns = close.pct_change().dropna()

        # Use last 60 trading days
        returns_60d = returns.tail(60)
        if len(returns_60d) < 40:
            log.warning(f"    PAIRWISE_CORR: Only {len(returns_60d)} days of returns")
            return results

        # Correlation matrix
        corr_matrix = returns_60d.corr()

        # Average of upper triangle (excluding diagonal)
        n = corr_matrix.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        upper_vals = corr_matrix.values[mask]
        upper_vals = upper_vals[~np.isnan(upper_vals)]

        if len(upper_vals) == 0:
            log.warning("    PAIRWISE_CORR: No valid correlation pairs")
            return results

        avg_corr = float(np.mean(upper_vals))

        latest_date = returns.index[-1]
        if hasattr(latest_date, "date"):
            latest_date = latest_date.date()
        age_days = (today - latest_date).days

        results["AVG_PAIRWISE_CORR"] = {
            "raw_value": round(avg_corr, 4),
            "age_days": age_days,
            "date": latest_date.strftime("%Y-%m-%d"),
        }
        n_tickers = close.shape[1]
        n_pairs = len(upper_vals)
        log.info(f"    PAIRWISE_CORR: {avg_corr:.4f} ({n_tickers} tickers, {n_pairs} pairs, 60d), Date={latest_date}, Age={age_days}d")

    except Exception as e:
        log.warning(f"    PAIRWISE_CORR: Failed -> {e}")

    return results


def pull_vix_term_structure():
    """Pull VIX / VIX3M ratio from yfinance.
    VIX = 30-day implied vol (^VIX)
    VIX3M = 3-month implied vol (^VIX3M)
    Ratio > 1.0 = backwardation = short-term fear > long-term = fragile.
    Ratio < 1.0 = contango = normal/calm.
    """
    import yfinance as yf

    results = {}
    today = date.today()
    log.info("  VIX_TERM: Pulling ^VIX and ^VIX3M via yfinance...")

    try:
        data = yf.download(["^VIX", "^VIX3M"], period="5d", interval="1d", progress=False)

        if data.empty:
            log.warning("    VIX_TERM: No data returned")
            return results

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data

        # Drop NaN rows and get latest
        close = close.dropna()
        if close.empty:
            log.warning("    VIX_TERM: No valid close data")
            return results

        latest = close.iloc[-1]
        vix = float(latest["^VIX"])
        vix3m = float(latest["^VIX3M"])

        if vix3m <= 0 or pd.isna(vix) or pd.isna(vix3m):
            log.warning(f"    VIX_TERM: Invalid values VIX={vix}, VIX3M={vix3m}")
            return results

        ratio = vix / vix3m

        latest_date = close.index[-1]
        if hasattr(latest_date, "date"):
            latest_date = latest_date.date()
        age_days = (today - latest_date).days

        results["VIX_TERM_STRUCTURE"] = {
            "raw_value": round(ratio, 4),
            "age_days": age_days,
            "date": latest_date.strftime("%Y-%m-%d"),
        }
        state = "BACKWARDATION (stressed)" if ratio > 1.0 else "Contango (calm)"
        log.info(f"    VIX_TERM: {ratio:.4f} (VIX={vix:.1f}, VIX3M={vix3m:.1f}) — {state}, Date={latest_date}, Age={age_days}d")

    except Exception as e:
        log.warning(f"    VIX_TERM: Failed -> {e}")

    return results


# ==================== NORMALIZATION ====================

def normalize_linear(indicator_name, value):
    """Linear normalization to 0-10 scale using INDICATOR_RANGES.
    Score 0 = stable, 10 = fragile.
    If invert=True, lower values map to higher (more fragile) scores.
    """
    if pd.isna(value) or indicator_name not in INDICATOR_RANGES:
        return np.nan
    lo, hi, invert = INDICATOR_RANGES[indicator_name]
    clipped = max(lo, min(hi, value))
    normalized = (clipped - lo) / (hi - lo)
    if invert:
        normalized = 1.0 - normalized
    return round(normalized * 10.0, 2)


# ==================== COMPOSITE ====================

def calc_composite_score(indicator_scores):
    """Calculate weighted composite score from available indicators.
    Returns (composite, valid_indicators_dict).
    Renormalizes weights to available indicators.
    """
    valid = {k: v for k, v in indicator_scores.items()
             if not pd.isna(v) and k in INDICATOR_WEIGHTS}
    if not valid:
        return np.nan, {}
    total_weight = sum(INDICATOR_WEIGHTS[k] for k in valid)
    if total_weight == 0:
        return np.nan, {}
    weighted_sum = sum(INDICATOR_WEIGHTS[k] * v / total_weight for k, v in valid.items())
    return round(weighted_sum, 2), valid


def get_fragility_phase(score):
    """Map score to fragility phase name."""
    if pd.isna(score):
        return "Unknown"
    for threshold, phase in FRAGILITY_PHASES:
        if score <= threshold:
            return phase
    return "Critical"


def get_signal(phase, valid_count):
    """Get SIGNAL string from phase. If <5 sources, return Partial."""
    if valid_count < MIN_SOURCES_FOR_COMPOSITE:
        return f"Partial ({valid_count}/8)"
    return PHASE_TO_SIGNAL.get(phase, "Caution")


def calc_freshness(indicator_data):
    """Calculate weighted freshness score from available indicators.
    Returns 0-10 freshness score.
    """
    if not indicator_data:
        return 0.0

    weighted_sum = 0.0
    weight_sum = 0.0

    for name, data in indicator_data.items():
        if name not in FRESHNESS_PARAMS or name not in INDICATOR_WEIGHTS:
            continue
        age = data.get("age_days", 0)
        params = FRESHNESS_PARAMS[name]
        ideal = params["ideal"]
        decay = params["decay_rate"]

        if age <= ideal:
            fresh = 10.0
        else:
            fresh = max(0.0, 10.0 - (age - ideal) * decay)

        w = INDICATOR_WEIGHTS[name]
        weighted_sum += fresh * w
        weight_sum += w

    if weight_sum == 0:
        return 0.0
    return round(weighted_sum / weight_sum, 1)


def calc_asymmetry_adj(score, signal):
    """Fragile/Critical signals get 1.3x multiplier, others 1.0x. Capped at 13.0."""
    if pd.isna(score):
        return score
    if signal in ("Fragile", "Critical"):
        return round(min(score * 1.3, 13.0), 2)
    return round(score * 1.0, 2)


def get_direction(current, previous):
    """Compare current to previous score for direction."""
    if pd.isna(current) or pd.isna(previous):
        return "n/a"
    diff = current - previous
    return "Rising" if diff > 0.5 else "Falling" if diff < -0.5 else "Flat"


def get_speed(current, previous):
    """Assess rate of change."""
    if pd.isna(current) or pd.isna(previous):
        return "n/a"
    wc = abs(current - previous)
    return "High" if wc > 2.0 else "Medium" if wc > 0.8 else "Low"


def get_regime_weight(warehouse):
    """Read L5 regime weight from CONFIG tab. Fallback to 10%."""
    try:
        ws = warehouse.worksheet("CONFIG")
        all_data = ws.get_all_values()
        for row in all_data:
            if len(row) >= 2 and "L5" in str(row[0]) and "Fragility" in str(row[0]):
                val = str(row[1]).replace("%", "").strip()
                if val:
                    return f"{val}%"
    except Exception as e:
        log.warning(f"  CONFIG read failed: {e}")
    return "10%"


# ==================== SHEET WRITERS ====================

def write_raw_market_l5(warehouse, all_results):
    """Write L5 indicator data to RAW_MARKET tab (R20-R27).
    Only writes rows for indicators that have data.
    """
    ws = warehouse.worksheet("RAW_MARKET")
    today_str = date.today().strftime("%Y-%m-%d")

    # Build row map: find existing L5 rows by indicator name
    all_data = ws.get_all_values()
    indicator_row_map = {}
    for i, row in enumerate(all_data):
        if len(row) >= 3 and row[2] == "L5":
            indicator_row_map[row[1]] = i + 1

    updates = []

    for ind_name, data in all_results.items():
        if ind_name not in indicator_row_map:
            log.warning(f"  RAW_MARKET: {ind_name} not found in sheet (L5 row missing)")
            continue
        row_idx = indicator_row_map[ind_name]
        source, tier, unit = SOURCE_MAP.get(ind_name, ("Unknown", "T3", ""))
        raw_val = data.get("raw_value")
        if isinstance(raw_val, float) and not pd.isna(raw_val):
            raw_str = f"{raw_val:.4f}" if abs(raw_val) < 100 else f"{raw_val:.2f}"
        else:
            raw_str = "---"
        age_str = str(data.get("age_days", 0))
        date_str = data.get("date", today_str)
        row_data = [date_str, ind_name, "L5", raw_str, "---", "---", age_str, source, tier, unit]
        updates.append((f"A{row_idx}:J{row_idx}", [row_data]))

    for cell_range, values in updates:
        ws.update(values, cell_range, value_input_option="RAW")

    log.info(f"  RAW_MARKET: {len(updates)} L5 rows written")


def write_scores_l5(warehouse, composite, direction, speed, phase, signal, freshness, asym_adj, regime_weight, valid_count, total_count):
    """Write L5 composite score to SCORES tab Row 6 (5/13 columns ab Tag 1)."""
    ws = warehouse.worksheet("SCORES")

    row_data = [
        "L5 Fragility",
        str(composite) if not pd.isna(composite) else "---",
        "---",              # SCORE_7D
        "---",              # SCORE_30D
        "n/a",              # PERCENTILE
        direction,          # DIRECTION
        speed,              # SPEED
        signal,             # SIGNAL
        f"{freshness:.0f}", # FRESHNESS
        "---",              # DECAY_ADJ
        str(asym_adj) if not pd.isna(asym_adj) else "---",  # ASYMMETRY_ADJ
        regime_weight,      # REGIME_WEIGHT
        "---",              # HISTORICAL_ANALOG
    ]
    ws.update([row_data], "A6:M6", value_input_option="RAW")
    log.info(f"  SCORES: {composite} ({signal}) | {phase} | {valid_count}/{total_count}")


def write_dashboard_l5(warehouse, composite, direction, signal, freshness, speed):
    """Write L5 summary to DASHBOARD tab Row 21."""
    ws = warehouse.worksheet("DASHBOARD")
    row_data = [
        "L5 Fragility",
        str(composite) if not pd.isna(composite) else "---",
        direction,
        signal,
        f"{freshness:.0f}",
        "n/a",
        speed,
    ]
    ws.update([row_data], "A21:G21", value_input_option="RAW")
    log.info(f"  DASHBOARD: {composite}, {signal}")


# ==================== MAIN ====================

def main():
    log.info("=" * 60)
    log.info("L5 Fragility Collector - Starting (8/8 Indicators)")
    log.info("=" * 60)

    log.info("Connecting to Data Warehouse...")
    warehouse = connect_warehouse()
    log.info("  OK")

    # --- PULL PHASE ---
    log.info("--- PULL PHASE ---")

    log.info("Pulling FRED indicators...")
    reserve_results = pull_reserve_drain_rate()
    sofr_results = pull_sofr_ffr_spread()
    stress_results = pull_fin_stress_index()
    rrp_results = pull_on_rrp_usage()

    log.info("Pulling yfinance indicators...")
    hhi_results = pull_spy_concentration()
    amihud_results = pull_liquidity_amihud()
    corr_results = pull_avg_pairwise_corr()
    vix_results = pull_vix_term_structure()

    # Merge all results
    all_results = {}
    for r in [reserve_results, sofr_results, stress_results, rrp_results,
              hhi_results, amihud_results, corr_results, vix_results]:
        all_results.update(r)

    # --- NORMALIZE PHASE ---
    log.info("--- NORMALIZE PHASE ---")

    indicator_scores = {}
    indicator_data = {}

    for ind_name in INDICATOR_WEIGHTS:
        if ind_name in all_results:
            data = all_results[ind_name]
            score = normalize_linear(ind_name, data["raw_value"])
            indicator_scores[ind_name] = score
            indicator_data[ind_name] = {"age_days": data.get("age_days", 0)}
            log.info(f"  {ind_name}: raw={data['raw_value']}, score={score}")
        else:
            indicator_scores[ind_name] = np.nan
            log.warning(f"  {ind_name}: NO DATA")

    # --- COMPOSITE PHASE ---
    log.info("--- COMPOSITE PHASE ---")

    composite, valid_indicators = calc_composite_score(indicator_scores)
    valid_count = len(valid_indicators)
    total_count = len(INDICATOR_WEIGHTS)
    phase = get_fragility_phase(composite)
    signal = get_signal(phase, valid_count)
    freshness = calc_freshness(indicator_data)

    # Direction and speed from previous SCORES value
    direction = speed = "n/a"
    try:
        prev_row = warehouse.worksheet("SCORES").row_values(6)
        if len(prev_row) >= 2 and prev_row[1] not in ("---", "", "—", None):
            prev_score = float(prev_row[1])
            direction = get_direction(composite, prev_score)
            speed = get_speed(composite, prev_score)
            log.info(f"  Previous: {prev_score} -> {direction}, {speed}")
    except Exception:
        pass

    # Asymmetry adjustment
    asym_signal = PHASE_TO_SIGNAL.get(phase, "Caution") if valid_count >= MIN_SOURCES_FOR_COMPOSITE else "Caution"
    asym_adj = calc_asymmetry_adj(composite, asym_signal)

    # Regime weight from CONFIG
    regime_weight = get_regime_weight(warehouse)

    log.info(f"  Composite: {composite}/10 | {phase} | {signal}")
    log.info(f"  Sources: {valid_count}/{total_count} | Freshness: {freshness}/10")
    log.info(f"  Asymmetry: {asym_adj} | Regime Weight: {regime_weight}")

    # --- WRITE PHASE ---
    log.info("--- WRITE PHASE ---")

    write_raw_market_l5(warehouse, all_results)
    write_scores_l5(warehouse, composite, direction, speed, phase, signal,
                    freshness, asym_adj, regime_weight, valid_count, total_count)
    write_dashboard_l5(warehouse, composite, direction, signal, freshness, speed)

    # --- SUMMARY ---
    log.info("=" * 60)
    log.info(f"COMPLETE: {composite}/10 ({phase}) | {signal} | {direction}/{speed} | {valid_count}/{total_count}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
