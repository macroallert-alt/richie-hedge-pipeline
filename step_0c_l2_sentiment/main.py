"""
L2 Sentiment Collector - step_0c_l2_sentiment/main.py
Global Macro RV System - Data Warehouse Layer 2

Pulls 7 sentiment indicators, normalizes to 0-10 scores, writes to:
  - RAW_MARKET tab (individual indicator rows)
  - SCORES tab (L2 composite score row)
  - DASHBOARD tab (L2 summary line)

Sources:
  T1 (API-direct): VIX, VIX3M (term structure), HY OAS, MOVE Index
  T2 (API-indirect): Put/Call Ratio, Google Trends
  T1 (delayed): Insider Buy/Sell (SEC EDGAR placeholder)

Indicators NOT yet implemented (need scraping):
  - AAII Bull/Bear (weekly, aaii.com)
  - CNN Fear & Greed (daily, scrape)
  - Margin Debt YoY (monthly, FINRA)

Reference: V56 Statusanalyse Kap.10 (Layer 2 Sentiment)
Iron Rule: V16 main.py NEVER TOUCH. This is a separate module.
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from fredapi import Fred

# --- CONFIG ---

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("L2_SENTIMENT")

# --- NORMALIZATION ANCHORS (from V56 CONFIG Tab F) ---
# Hybrid: 60% Anchored (2015-2025 range) + 40% Rolling (252d)
# For initial implementation: use anchored ranges only (known historical bounds)
# Agent 8 will recommend adjustments after 90 days live

INDICATOR_RANGES = {
    # indicator: (min_bullish, max_bearish, invert)
    # invert=True means HIGH value = BEARISH (e.g. VIX high = fear)
    "VIX_LEVEL":           (9.0,  80.0,  True),   # VIX: low=greed, high=fear
    "VIX_TERM_STRUCTURE":  (0.75, 1.25,  False),   # VIX/VIX3M: <1=contango(calm), >1=backwardation(fear)
    "PUT_CALL_RATIO":      (0.5,  1.5,   True),   # High P/C = bearish sentiment
    "HY_OAS_SPREAD":       (250,  1000,  True),   # Wide spreads = fear
    "MOVE_INDEX":          (50,   200,   True),   # High MOVE = bond fear
    "GOOGLE_TRENDS_RATIO": (0.2,  5.0,   True),   # "stock market crash"/"buy stocks" ratio
    "INSIDER_BUY_SELL":    (0.1,  2.0,   False),  # High ratio = insiders buying = bullish
}

# Weights from V56 Kap.10 (adjusted: AAII 15% not yet available, redistributed)
INDICATOR_WEIGHTS = {
    "VIX_LEVEL":           0.12,
    "VIX_TERM_STRUCTURE":  0.12,
    "PUT_CALL_RATIO":      0.18,
    "HY_OAS_SPREAD":       0.12,
    "MOVE_INDEX":          0.06,
    "GOOGLE_TRENDS_RATIO": 0.12,
    "INSIDER_BUY_SELL":    0.12,
    # Reserved: AAII=0.15 when implemented -> redistribute
    # Currently weights sum to 0.84 -> normalized to 1.0 in calc
}

# Sentiment phase mapping (score 0-10 -> phase name)
SENTIMENT_PHASES = [
    (1.0, "Capitulation"),
    (2.0, "Depression/Panic"),
    (3.0, "Fear"),
    (4.0, "Anxiety"),
    (5.0, "Denial"),
    (6.0, "Optimism"),
    (7.0, "Excitement"),
    (8.5, "Thrill"),
    (10.0, "Euphoria"),
]


# --- GOOGLE SHEETS ---

def connect_warehouse():
    """Connect to Data Warehouse Google Sheet."""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    gc = gspread.authorize(creds)
    return gc.open_by_key(WAREHOUSE_SHEET_ID)


# --- DATA PULLS ---

def pull_fred_sentiment(fred):
    """Pull FRED-based sentiment indicators."""
    today = date.today()
    start = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    series = {
        "VIXCLS": "VIX_LEVEL",
        "VIXCLS3M": "VIX3M",
        "BAMLH0A0HYM2": "HY_OAS_SPREAD",
    }

    results = {}
    for fred_id, label in series.items():
        log.info(f"  FRED: {fred_id}...")
        try:
            raw = fred.get_series(fred_id, observation_start=start, observation_end=end)
            if raw is not None and not raw.empty:
                val = float(raw.dropna().iloc[-1])
                results[label] = val
                log.info(f"    OK: {label} = {val:.4f}")
            else:
                log.warning(f"    EMPTY: {fred_id}")
                results[label] = np.nan
        except Exception as e:
            log.error(f"    FAILED: {fred_id} -> {e}")
            results[label] = np.nan

    return results


def pull_yfinance_sentiment():
    """Pull yfinance-based sentiment indicators."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed!")
        return {}

    results = {}

    # VIX (backup if FRED fails) + VIX3M
    for ticker, label in [("^VIX", "VIX_LEVEL_YF"), ("^VIX3M", "VIX3M_YF")]:
        try:
            data = yf.download(ticker, period="5d", progress=False)
            if data is not None and not data.empty:
                val = float(data["Close"].iloc[-1])
                results[label] = val
                log.info(f"  yfinance: {label} = {val:.4f}")
        except Exception as e:
            log.warning(f"  yfinance {ticker} failed: {e}")

    # MOVE Index
    try:
        data = yf.download("^MOVE", period="5d", progress=False)
        if data is not None and not data.empty:
            results["MOVE_INDEX"] = float(data["Close"].iloc[-1])
            log.info(f"  yfinance: MOVE_INDEX = {results['MOVE_INDEX']:.2f}")
    except Exception as e:
        log.warning(f"  yfinance MOVE failed: {e}")

    # Put/Call Ratio: CBOE direct feed not available via yfinance
    # PLACEHOLDER: VIX-derived proxy until real CBOE data implemented
    if "VIX_LEVEL_YF" in results:
        vix = results["VIX_LEVEL_YF"]
        pc_approx = 0.5 + (vix / 80.0)
        results["PUT_CALL_RATIO"] = round(pc_approx, 3)
        log.info(f"  PUT_CALL_RATIO (VIX-proxy): {pc_approx:.3f}")

    return results


def pull_google_trends():
    """Pull Google Trends sentiment ratio: crash/buy."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.warning("pytrends not installed, skipping Google Trends")
        return {}

    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(
            kw_list=["stock market crash", "buy stocks"],
            timeframe="today 3-m",
            geo="US",
        )
        df = pytrends.interest_over_time()
        if df is not None and not df.empty:
            crash = df["stock market crash"].iloc[-1]
            buy = df["buy stocks"].iloc[-1]
            ratio = crash / max(buy, 1)
            log.info(f"  Google Trends: crash={crash}, buy={buy}, ratio={ratio:.3f}")
            return {"GOOGLE_TRENDS_RATIO": round(ratio, 3)}
    except Exception as e:
        log.warning(f"  Google Trends failed: {e}")

    return {}


# --- NORMALIZATION ---

def normalize_indicator(name, value):
    """
    Normalize raw indicator to 0-10 sentiment score.

    0 = Extreme Fear / Capitulation
    5 = Neutral
    10 = Extreme Greed / Euphoria

    For inverted indicators (VIX, HY spreads):
      High raw value -> LOW score (fear)
    """
    if pd.isna(value) or name not in INDICATOR_RANGES:
        return np.nan

    lo, hi, invert = INDICATOR_RANGES[name]
    clipped = max(lo, min(hi, value))
    normalized = (clipped - lo) / (hi - lo)

    if invert:
        normalized = 1.0 - normalized

    return round(normalized * 10.0, 2)


def calc_composite_score(indicator_scores):
    """
    Weighted composite L2 score.
    Only uses valid (non-NaN) indicators.
    Renormalizes weights to sum to 1.0.
    """
    valid = {k: v for k, v in indicator_scores.items()
             if not pd.isna(v) and k in INDICATOR_WEIGHTS}

    if not valid:
        return np.nan, {}

    total_weight = sum(INDICATOR_WEIGHTS[k] for k in valid)
    if total_weight == 0:
        return np.nan, {}

    weighted_sum = sum(
        INDICATOR_WEIGHTS[k] * v / total_weight
        for k, v in valid.items()
    )

    return round(weighted_sum, 2), valid


def get_sentiment_phase(score):
    """Map 0-10 score to sentiment phase name."""
    if pd.isna(score):
        return "Unknown"
    for threshold, phase in SENTIMENT_PHASES:
        if score <= threshold:
            return phase
    return "Euphoria"


def get_direction(current, previous):
    """Determine score direction."""
    if pd.isna(current) or pd.isna(previous):
        return "n/a"
    diff = current - previous
    if diff > 0.5:
        return "Rising"
    elif diff < -0.5:
        return "Falling"
    return "Flat"


def get_speed(current, prev_7d):
    """Determine rate of change."""
    if pd.isna(current) or pd.isna(prev_7d):
        return "n/a"
    weekly_change = abs(current - prev_7d)
    if weekly_change > 2.0:
        return "High"
    elif weekly_change > 0.8:
        return "Medium"
    return "Low"


# --- SHEET WRITERS ---

def write_raw_market_l2(warehouse, indicator_scores, raw_values):
    """
    Write individual L2 indicator values to RAW_MARKET tab.
    Updates existing L2 rows in-place.
    """
    ws = warehouse.worksheet("RAW_MARKET")
    today_str = date.today().strftime("%Y-%m-%d")

    # Read existing data to find L2 rows
    all_data = ws.get_all_values()
    indicator_row_map = {}
    for i, row in enumerate(all_data):
        if len(row) >= 3 and row[2] == "L2":
            indicator_row_map[row[1]] = i + 1  # 1-indexed

    source_map = {
        "VIX_LEVEL":           ("CBOE/yfinance", "T1", "index"),
        "VIX_TERM_STRUCTURE":  ("CBOE",          "T1", "ratio"),
        "PUT_CALL_RATIO":      ("CBOE",          "T1", "ratio"),
        "HY_OAS_SPREAD":       ("FRED",          "T1", "bps"),
        "MOVE_INDEX":          ("FRED/yfinance", "T1", "index"),
        "GOOGLE_TRENDS_RATIO": ("pytrends",      "T2", "ratio"),
        "INSIDER_BUY_SELL":    ("SEC EDGAR",     "T1", "ratio"),
    }

    updates = []
    for ind_name in indicator_scores:
        if ind_name not in indicator_row_map:
            log.warning(f"  RAW_MARKET: No row for {ind_name}, skipping")
            continue

        row_idx = indicator_row_map[ind_name]
        raw_val = raw_values.get(ind_name)
        source, tier, unit = source_map.get(ind_name, ("Unknown", "T3", ""))

        if isinstance(raw_val, float) and not pd.isna(raw_val):
            raw_str = f"{raw_val:.4f}" if abs(raw_val) < 100 else f"{raw_val:.2f}"
        else:
            raw_str = "—"

        row_data = [
            today_str, ind_name, "L2", raw_str,
            "—", "—",  # PREV_7D, PREV_30D (populated after history builds)
            "0",        # DATA_AGE_DAYS
            source, tier, unit,
        ]

        cell_range = f"A{row_idx}:J{row_idx}"
        updates.append((cell_range, [row_data]))

    for cell_range, values in updates:
        ws.update(cell_range, values, value_input_option="RAW")

    log.info(f"  RAW_MARKET: {len(updates)} L2 indicators written")


def write_scores_l2(warehouse, composite, direction, speed, phase, freshness, valid_count, total_count):
    """
    Write L2 composite score to SCORES tab (row 3 = L2 Sentiment).
    """
    ws = warehouse.worksheet("SCORES")

    signal = "Neutral"
    if composite is not None and not pd.isna(composite):
        if composite <= 2.5:
            signal = "Extreme Bearish"
        elif composite <= 4.0:
            signal = "Bearish"
        elif composite >= 8.0:
            signal = "Extreme Bullish"
        elif composite >= 6.5:
            signal = "Bullish"

    percentile = "n/a"  # Until 90d+ history

    # Asymmetry: bearish x1.3 (from V56 Deep Review)
    asym_adj = composite
    if composite is not None and not pd.isna(composite) and composite < 5.0:
        asym_adj = round(composite * 1.3, 2)
        asym_adj = min(asym_adj, 10.0)

    row_data = [
        "L2 Sentiment",
        str(composite) if not pd.isna(composite) else "—",
        "—",             # SCORE_7D
        "—",             # SCORE_30D
        str(percentile),
        direction,
        speed,
        signal,
        f"{freshness:.0f}",
        str(asym_adj),   # DECAY_ADJ
        str(asym_adj),   # ASYMMETRY_ADJ
        "—",             # REGIME_WEIGHT
        "—",             # HISTORICAL_ANALOG
    ]

    ws.update("A3:M3", [row_data], value_input_option="RAW")
    log.info(f"  SCORES L2: {composite} ({signal}) | Phase: {phase} | {valid_count}/{total_count} sources")


def write_dashboard_l2(warehouse, composite, direction, signal, phase, freshness, speed):
    """
    Write L2 summary to DASHBOARD tab (row 18 = L2 Sentiment).
    """
    ws = warehouse.worksheet("DASHBOARD")

    row_data = [
        "L2 Sentiment",
        str(composite) if not pd.isna(composite) else "—",
        direction,
        signal,
        f"{freshness:.0f}",
        "n/a",  # PCTL
        speed,
    ]

    ws.update("A18:G18", [row_data], value_input_option="RAW")
    log.info(f"  DASHBOARD L2: Score={composite}, Signal={signal}, Phase={phase}")


# --- MAIN ---

def main():
    log.info("=" * 60)
    log.info("L2 Sentiment Collector - Starting")
    log.info("=" * 60)

    # --- Connect ---
    log.info("Connecting to Data Warehouse Sheet...")
    warehouse = connect_warehouse()
    log.info("  OK")

    log.info("Connecting to FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    log.info("  OK")

    # --- Pull Data ---
    log.info("--- PULL PHASE ---")
    log.info("Pulling FRED sentiment data...")
    fred_data = pull_fred_sentiment(fred)

    log.info("Pulling yfinance sentiment data...")
    yf_data = pull_yfinance_sentiment()

    log.info("Pulling Google Trends...")
    gt_data = pull_google_trends()

    # --- Merge: FRED primary, yfinance fallback ---
    log.info("--- MERGE PHASE ---")
    raw_values = {}

    # VIX: prefer FRED, fallback yfinance
    raw_values["VIX_LEVEL"] = fred_data.get("VIX_LEVEL")
    if pd.isna(raw_values.get("VIX_LEVEL")):
        raw_values["VIX_LEVEL"] = yf_data.get("VIX_LEVEL_YF")
        if raw_values.get("VIX_LEVEL") is not None:
            log.info("  VIX: FRED empty, using yfinance fallback")

    # VIX Term Structure: VIX / VIX3M
    vix3m = fred_data.get("VIX3M")
    if pd.isna(vix3m) or vix3m is None:
        vix3m = yf_data.get("VIX3M_YF")
    vix_val = raw_values.get("VIX_LEVEL")
    if vix_val and vix3m and not pd.isna(vix_val) and not pd.isna(vix3m) and vix3m > 0:
        raw_values["VIX_TERM_STRUCTURE"] = round(vix_val / vix3m, 4)
        log.info(f"  VIX_TERM_STRUCTURE: {vix_val:.2f} / {vix3m:.2f} = {raw_values['VIX_TERM_STRUCTURE']:.4f}")
    else:
        raw_values["VIX_TERM_STRUCTURE"] = np.nan

    # HY OAS
    raw_values["HY_OAS_SPREAD"] = fred_data.get("HY_OAS_SPREAD")

    # MOVE Index
    raw_values["MOVE_INDEX"] = yf_data.get("MOVE_INDEX")

    # Put/Call Ratio
    raw_values["PUT_CALL_RATIO"] = yf_data.get("PUT_CALL_RATIO")

    # Google Trends
    raw_values["GOOGLE_TRENDS_RATIO"] = gt_data.get("GOOGLE_TRENDS_RATIO")

    # Insider Buy/Sell: NOT YET IMPLEMENTED
    raw_values["INSIDER_BUY_SELL"] = np.nan
    log.info("  INSIDER_BUY_SELL: placeholder (SEC EDGAR not yet implemented)")

    # --- Normalize ---
    log.info("--- NORMALIZE PHASE ---")
    indicator_scores = {}
    for name, value in raw_values.items():
        score = normalize_indicator(name, value)
        indicator_scores[name] = score
        raw_str = f"{value:.4f}" if isinstance(value, float) and not pd.isna(value) else "n/a"
        score_str = f"{score:.2f}" if not pd.isna(score) else "n/a"
        log.info(f"  {name}: raw={raw_str} -> score={score_str}/10")

    # --- Composite ---
    log.info("--- COMPOSITE PHASE ---")
    composite, valid_indicators = calc_composite_score(indicator_scores)
    valid_count = len(valid_indicators)
    total_count = len(INDICATOR_WEIGHTS)
    phase = get_sentiment_phase(composite)
    freshness = (valid_count / total_count) * 10.0

    # Direction & Speed: compare to previous score in SCORES tab
    direction = "n/a"
    speed = "n/a"
    try:
        ws_scores = warehouse.worksheet("SCORES")
        prev_row = ws_scores.row_values(3)
        if len(prev_row) >= 2 and prev_row[1] not in ("—", "", None):
            prev_score = float(prev_row[1])
            direction = get_direction(composite, prev_score)
            speed = get_speed(composite, prev_score)
            log.info(f"  Previous L2 score: {prev_score} -> Direction: {direction}, Speed: {speed}")
    except Exception as e:
        log.warning(f"  Could not read previous score: {e}")

    signal = "Neutral"
    if composite is not None and not pd.isna(composite):
        if composite <= 2.5:
            signal = "Extreme Bearish"
        elif composite <= 4.0:
            signal = "Bearish"
        elif composite >= 8.0:
            signal = "Extreme Bullish"
        elif composite >= 6.5:
            signal = "Bullish"

    log.info(f"  L2 Composite: {composite}/10")
    log.info(f"  Phase: {phase}")
    log.info(f"  Signal: {signal}")
    log.info(f"  Sources: {valid_count}/{total_count} | Freshness: {freshness:.1f}/10")

    # --- Write to Sheets ---
    log.info("--- WRITE PHASE ---")

    log.info("Writing RAW_MARKET L2...")
    write_raw_market_l2(warehouse, indicator_scores, raw_values)

    log.info("Writing SCORES L2...")
    write_scores_l2(warehouse, composite, direction, speed, phase, freshness, valid_count, total_count)

    log.info("Writing DASHBOARD L2...")
    write_dashboard_l2(warehouse, composite, direction, signal, phase, freshness, speed)

    # --- Done ---
    log.info("=" * 60)
    log.info("L2 Sentiment Collector COMPLETE")
    log.info(f"  Score: {composite}/10 ({phase})")
    log.info(f"  Signal: {signal}")
    log.info(f"  Direction: {direction} | Speed: {speed}")
    log.info(f"  Sources: {valid_count}/{total_count}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
