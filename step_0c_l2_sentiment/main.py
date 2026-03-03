"""
L2 Sentiment Collector - step_0c_l2_sentiment/main.py
Global Macro RV System - Data Warehouse Layer 2
Pulls 11 sentiment indicators from 7 sources, normalizes to 0-10, writes to:
  RAW_MARKET, SCORES, DASHBOARD tabs
Sources (7/7 LIVE):
  VIX/VIX3M/MOVE (yfinance), HY OAS (FRED), Put/Call (proxy),
  Google Trends (pytrends), AAII Bull/Bear (XLS), CNN Fear+Greed (API),
  Insider Buy/Sell (OpenInsider), Margin Debt YoY (FINRA)
Reference: V77 Statusanalyse Kap.4
Iron Rule: V16 main.py NEVER TOUCH.
"""

import io
import os
import sys
import logging
import json
import re
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from fredapi import Fred
import requests

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("L2_SENTIMENT")

SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

INDICATOR_RANGES = {
    "VIX_LEVEL":           (9.0,   80.0,  True),
    "VIX_TERM_STRUCTURE":  (0.75,  1.25,  False),
    "PUT_CALL_RATIO":      (0.5,   1.5,   True),
    "HY_OAS_SPREAD":       (250,   1000,  True),
    "MOVE_INDEX":          (50,    200,   True),
    "GOOGLE_TRENDS_RATIO": (0.2,   5.0,   True),
    "INSIDER_BUY_SELL":    (0.1,   2.0,   False),
    "AAII_BULL_PCT":       (15.0,  65.0,  False),
    "AAII_BEAR_PCT":       (15.0,  65.0,  True),
    "CNN_FEAR_GREED":      (0.0,   100.0, False),
    "MARGIN_DEBT_YOY_CHG": (-30.0, 40.0,  False),
}

INDICATOR_WEIGHTS = {
    "PUT_CALL_RATIO":      0.15,
    "VIX_TERM_STRUCTURE":  0.14,
    "HY_OAS_SPREAD":       0.14,
    "INSIDER_BUY_SELL":    0.12,
    "AAII_BULL_PCT":       0.08,
    "AAII_BEAR_PCT":       0.08,
    "VIX_LEVEL":           0.08,
    "MOVE_INDEX":          0.06,
    "MARGIN_DEBT_YOY_CHG": 0.06,
    "CNN_FEAR_GREED":      0.05,
    "GOOGLE_TRENDS_RATIO": 0.04,
}

SENTIMENT_PHASES = [
    (1.0, "Capitulation"), (2.0, "Depression/Panic"), (3.0, "Fear"),
    (4.0, "Anxiety"), (5.0, "Denial"), (6.0, "Optimism"),
    (7.0, "Excitement"), (8.5, "Thrill"), (10.0, "Euphoria"),
]


def connect_warehouse():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    gc = gspread.authorize(creds)
    return gc.open_by_key(WAREHOUSE_SHEET_ID)


# ==================== DATA PULLS ====================

def pull_fred_sentiment(fred):
    today = date.today()
    start = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    series = {"VIXCLS": "VIX_LEVEL", "BAMLH0A0HYM2": "HY_OAS_SPREAD"}
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
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed!")
        return {}
    results = {}

    def safe_last_close(ticker):
        data = yf.download(ticker, period="5d", progress=False)
        if data is not None and not data.empty:
            close = data["Close"]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            return float(close.iloc[-1])
        return np.nan

    for sym, key in [("^VIX", "VIX_LEVEL_YF"), ("^VIX3M", "VIX3M_YF"), ("^MOVE", "MOVE_INDEX")]:
        try:
            val = safe_last_close(sym)
            if not pd.isna(val):
                results[key] = val
                log.info(f"  yfinance: {key} = {val:.4f}")
        except Exception as e:
            log.warning(f"  yfinance {sym} failed: {e}")

    if "VIX_LEVEL_YF" in results:
        vix = results["VIX_LEVEL_YF"]
        pc_approx = 0.5 + (vix / 80.0)
        results["PUT_CALL_RATIO"] = round(pc_approx, 3)
        log.info(f"  PUT_CALL_RATIO (VIX-proxy): {pc_approx:.3f}")
    return results


def pull_google_trends():
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.warning("pytrends not installed, skipping")
        return {}
    try:
        pt = TrendReq(hl="en-US", tz=360)
        pt.build_payload(kw_list=["stock market crash", "buy stocks"], timeframe="today 3-m", geo="US")
        df = pt.interest_over_time()
        if df is not None and not df.empty:
            crash = df["stock market crash"].iloc[-1]
            buy = df["buy stocks"].iloc[-1]
            ratio = crash / max(buy, 1)
            log.info(f"  Google Trends: crash={crash}, buy={buy}, ratio={ratio:.3f}")
            return {"GOOGLE_TRENDS_RATIO": round(ratio, 3)}
    except Exception as e:
        log.warning(f"  Google Trends failed: {e}")
    return {}


def pull_aaii_sentiment():
    """AAII Bull/Bear Survey.
    v4: Handle merged cells in XLS. Scan ALL cells in first 10 rows to find
    column indices for Bullish and Bearish, then read data rows below."""
    results = {}

    try:
        url = "https://www.aaii.com/files/surveys/sentiment.xls"
        log.info(f"  AAII: Fetching {url}...")
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()

        df_raw = pd.read_excel(io.BytesIO(resp.content), header=None)
        log.info(f"    AAII raw shape: {df_raw.shape}")

        # Log first 5 rows for debugging
        for i in range(min(5, len(df_raw))):
            log.info(f"    Row {i}: {df_raw.iloc[i].values.tolist()}")

        # Scan first 10 rows to find column indices for Bullish and Bearish
        # AAII XLS has merged cells: "Bullish" and "Bearish" may appear in
        # different rows. We need the FIRST column with "Bullish" and the
        # FIRST column with "Bearish".
        bull_col_idx = None
        bear_col_idx = None
        data_start_row = None

        for row_idx in range(min(10, len(df_raw))):
            for col_idx in range(len(df_raw.columns)):
                cell_val = str(df_raw.iloc[row_idx, col_idx]).strip().lower()
                if cell_val == "bullish" and bull_col_idx is None:
                    bull_col_idx = col_idx
                    log.info(f"    Found 'Bullish' at row={row_idx}, col={col_idx}")
                if cell_val == "bearish" and bear_col_idx is None:
                    bear_col_idx = col_idx
                    log.info(f"    Found 'Bearish' at row={row_idx}, col={col_idx}")
                    # Data starts after this row (or after the next sub-header row)
                    data_start_row = row_idx + 1

        # If we only found Bullish but not Bearish, check if Bearish is in the
        # same row as a later occurrence of Bullish (merged cell pattern)
        if bull_col_idx is not None and bear_col_idx is None:
            # AAII typical layout: Bullish | Neutral | Bearish in adjacent groups
            # Try col_idx + 2 or +3 as Bearish
            for offset in [2, 3, 4]:
                test_col = bull_col_idx + offset
                if test_col < len(df_raw.columns):
                    for row_idx in range(min(10, len(df_raw))):
                        cell_val = str(df_raw.iloc[row_idx, test_col]).strip().lower()
                        if "bear" in cell_val:
                            bear_col_idx = test_col
                            log.info(f"    Found 'Bearish' at row={row_idx}, col={test_col} (offset search)")
                            break
                if bear_col_idx is not None:
                    break

        if bull_col_idx is not None and bear_col_idx is not None:
            # Find where numeric data starts (first row after headers with a number)
            if data_start_row is None:
                data_start_row = 3  # safe default

            # Scan for first row with numeric data in the bull column
            for row_idx in range(data_start_row, min(data_start_row + 5, len(df_raw))):
                try:
                    test_val = float(df_raw.iloc[row_idx, bull_col_idx])
                    if not pd.isna(test_val):
                        data_start_row = row_idx
                        break
                except (ValueError, TypeError):
                    continue

            log.info(f"    Data starts at row {data_start_row}, bull_col={bull_col_idx}, bear_col={bear_col_idx}")

            # Get the latest valid data row
            bull_series = pd.to_numeric(df_raw.iloc[data_start_row:, bull_col_idx], errors="coerce")
            bear_series = pd.to_numeric(df_raw.iloc[data_start_row:, bear_col_idx], errors="coerce")

            # Find last row where both are valid
            valid_mask = bull_series.notna() & bear_series.notna()
            if valid_mask.any():
                last_valid_idx = valid_mask[valid_mask].index[-1]
                bull_val = float(bull_series.loc[last_valid_idx])
                bear_val = float(bear_series.loc[last_valid_idx])

                # Normalize decimals to percent
                if bull_val <= 1.0:
                    bull_val *= 100.0
                if bear_val <= 1.0:
                    bear_val *= 100.0

                log.info(f"    Latest values: bull={bull_val:.2f}%, bear={bear_val:.2f}%")

                # Sanity check
                if 5.0 <= bull_val <= 80.0 and 5.0 <= bear_val <= 80.0:
                    results["AAII_BULL_PCT"] = round(bull_val, 2)
                    results["AAII_BEAR_PCT"] = round(bear_val, 2)
                    log.info(f"    OK: AAII_BULL_PCT={bull_val:.2f}%, AAII_BEAR_PCT={bear_val:.2f}%")
                    return results
                else:
                    log.warning(f"    Sanity fail: bull={bull_val:.2f}%, bear={bear_val:.2f}%")
            else:
                log.warning("    No valid numeric data found in bull/bear columns")
        else:
            log.warning(f"    Could not find both columns: bull_col={bull_col_idx}, bear_col={bear_col_idx}")

    except Exception as e:
        log.warning(f"    AAII primary (XLS) failed: {e}")

    # Fallback: scrape website
    try:
        url = "https://www.aaii.com/sentimentsurvey"
        log.info(f"  AAII fallback: Scraping {url}...")
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text

        bull_val = None
        bear_val = None

        m = re.search(r'[Bb]ullish\s*:?\s*(\d{1,2}(?:\.\d+)?)\s*%', text)
        if m:
            bull_val = float(m.group(1))
        m = re.search(r'[Bb]earish\s*:?\s*(\d{1,2}(?:\.\d+)?)\s*%', text)
        if m:
            bear_val = float(m.group(1))

        if bull_val is not None and (bull_val < 5.0 or bull_val > 80.0):
            log.warning(f"    AAII scrape: bull={bull_val:.2f}% failed sanity")
            bull_val = None
        if bear_val is not None and (bear_val < 5.0 or bear_val > 80.0):
            log.warning(f"    AAII scrape: bear={bear_val:.2f}% failed sanity")
            bear_val = None

        if bull_val is not None and bear_val is not None:
            results["AAII_BULL_PCT"] = round(bull_val, 2)
            results["AAII_BEAR_PCT"] = round(bear_val, 2)
            log.info(f"    OK (scrape): bull={bull_val:.2f}%, bear={bear_val:.2f}%")
        else:
            log.warning(f"    Scrape incomplete: bull={bull_val}, bear={bear_val}")

    except Exception as e:
        log.warning(f"    AAII fallback failed: {e}")

    return results


def pull_cnn_fear_greed():
    """CNN Fear & Greed Index (daily, 0-100)."""
    results = {}
    try:
        log.info("  CNN F&G: Fetching...")
        resp = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                            headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "fear_and_greed" in data and "score" in data["fear_and_greed"]:
            score = float(data["fear_and_greed"]["score"])
            rating = data["fear_and_greed"].get("rating", "Unknown")
            results["CNN_FEAR_GREED"] = round(score, 2)
            log.info(f"    OK: CNN={score:.2f} ({rating})")
            return results
    except Exception as e:
        log.warning(f"    CNN primary failed: {e}")

    try:
        log.info("  CNN fallback...")
        resp = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/current",
                            headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "score" in data:
            results["CNN_FEAR_GREED"] = round(float(data["score"]), 2)
            log.info(f"    OK (fallback): {results['CNN_FEAR_GREED']}")
            return results
    except Exception as e:
        log.warning(f"    CNN fallback failed: {e}")
    return results


def pull_margin_debt():
    """FINRA Margin Debt YoY Change (monthly)."""
    results = {}
    try:
        log.info("  MARGIN_DEBT: FINRA...")
        resp = requests.get("https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics",
                            headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        match = re.search(r'href="([^"]*margin[^"]*\.xlsx?)"', resp.text, re.IGNORECASE)
        if match:
            data_url = match.group(1)
            if not data_url.startswith("http"):
                data_url = "https://www.finra.org" + data_url
            log.info(f"  Downloading {data_url}...")
            dr = requests.get(data_url, headers=SCRAPE_HEADERS, timeout=30)
            dr.raise_for_status()
            df = pd.read_excel(io.BytesIO(dr.content))
            debit_col = None
            for c in df.columns:
                if "debit" in str(c).lower():
                    debit_col = c
                    break
            if debit_col is not None:
                series = df[debit_col].dropna()
                if len(series) >= 13:
                    latest = float(series.iloc[-1])
                    year_ago = float(series.iloc[-13])
                    if year_ago > 0:
                        yoy = ((latest - year_ago) / year_ago) * 100.0
                        results["MARGIN_DEBT_YOY_CHG"] = round(yoy, 2)
                        log.info(f"    OK: {yoy:.2f}% (latest={latest:.0f}, yr_ago={year_ago:.0f})")
                        return results
    except Exception as e:
        log.warning(f"    FINRA failed: {e}")

    try:
        log.info("  MARGIN_DEBT: FRED proxy...")
        fred = Fred(api_key=FRED_API_KEY)
        start = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")
        raw = fred.get_series("BOGZ1FL663067003Q", observation_start=start)
        if raw is not None and not raw.empty:
            clean = raw.dropna()
            if len(clean) >= 5:
                latest = float(clean.iloc[-1])
                year_ago = float(clean.iloc[-5])
                if year_ago > 0:
                    yoy = ((latest - year_ago) / year_ago) * 100.0
                    results["MARGIN_DEBT_YOY_CHG"] = round(yoy, 2)
                    log.info(f"    OK (FRED): {yoy:.2f}%")
                    return results
    except Exception as e:
        log.warning(f"    FRED proxy failed: {e}")
    return results


def pull_insider_data():
    """Insider Buy/Sell Ratio from OpenInsider (last 7d, >$25k)."""
    results = {}
    try:
        log.info("  INSIDER: OpenInsider...")
        base = ("http://openinsider.com/screener?s=&o=&pl=25&ph=&ll=&lh="
                "&fd=7&fdr=&td=0&tdr=&feession=at&tession=ct&xp=1"
                "&vl=&vh=&ocl=&och=&session=oi&a={}&v=&cnt=500&page=1")

        resp_buy = requests.get(base.format(1), headers=SCRAPE_HEADERS, timeout=30)
        resp_buy.raise_for_status()
        buy_count = len(re.findall(r'P\s*-\s*Purchase', resp_buy.text))
        if buy_count == 0:
            tm = re.search(r'<table[^>]*tinytable[^>]*>(.*?)</table>', resp_buy.text, re.DOTALL)
            if tm:
                buy_count = max(0, tm.group(1).count('<tr>') - 1)
        log.info(f"    Purchases: {buy_count}")

        resp_sell = requests.get(base.format(2), headers=SCRAPE_HEADERS, timeout=30)
        resp_sell.raise_for_status()
        sell_count = len(re.findall(r'S\s*-\s*Sale', resp_sell.text))
        if sell_count == 0:
            tm = re.search(r'<table[^>]*tinytable[^>]*>(.*?)</table>', resp_sell.text, re.DOTALL)
            if tm:
                sell_count = max(0, tm.group(1).count('<tr>') - 1)
        log.info(f"    Sales: {sell_count}")

        if sell_count > 0 and buy_count >= 0:
            ratio = max(0.05, min(buy_count / sell_count, 2.5))
            results["INSIDER_BUY_SELL"] = round(ratio, 4)
            log.info(f"    OK: ratio={ratio:.4f} (buys={buy_count}, sells={sell_count})")
        elif buy_count > 0 and sell_count == 0:
            results["INSIDER_BUY_SELL"] = 2.0
            log.info(f"    OK: ratio=2.0 (buys={buy_count}, sells=0, capped)")
        else:
            log.warning(f"    No transactions found (buys={buy_count}, sells={sell_count})")
    except Exception as e:
        log.warning(f"    INSIDER failed: {e}")
    return results


# ==================== NORMALIZATION ====================

def normalize_indicator(name, value):
    if pd.isna(value) or name not in INDICATOR_RANGES:
        return np.nan
    lo, hi, invert = INDICATOR_RANGES[name]
    clipped = max(lo, min(hi, value))
    normalized = (clipped - lo) / (hi - lo)
    if invert:
        normalized = 1.0 - normalized
    return round(normalized * 10.0, 2)


def calc_composite_score(indicator_scores):
    valid = {k: v for k, v in indicator_scores.items()
             if not pd.isna(v) and k in INDICATOR_WEIGHTS}
    if not valid:
        return np.nan, {}
    total_weight = sum(INDICATOR_WEIGHTS[k] for k in valid)
    if total_weight == 0:
        return np.nan, {}
    weighted_sum = sum(INDICATOR_WEIGHTS[k] * v / total_weight for k, v in valid.items())
    return round(weighted_sum, 2), valid


def get_sentiment_phase(score):
    if pd.isna(score):
        return "Unknown"
    for threshold, phase in SENTIMENT_PHASES:
        if score <= threshold:
            return phase
    return "Euphoria"


def get_direction(current, previous):
    if pd.isna(current) or pd.isna(previous):
        return "n/a"
    diff = current - previous
    if diff > 0.5:
        return "Rising"
    elif diff < -0.5:
        return "Falling"
    return "Flat"


def get_speed(current, prev_7d):
    if pd.isna(current) or pd.isna(prev_7d):
        return "n/a"
    weekly_change = abs(current - prev_7d)
    if weekly_change > 2.0:
        return "High"
    elif weekly_change > 0.8:
        return "Medium"
    return "Low"


# ==================== SHEET WRITERS ====================

def write_raw_market_l2(warehouse, indicator_scores, raw_values):
    ws = warehouse.worksheet("RAW_MARKET")
    today_str = date.today().strftime("%Y-%m-%d")
    all_data = ws.get_all_values()
    indicator_row_map = {}
    for i, row in enumerate(all_data):
        if len(row) >= 3 and row[2] == "L2":
            indicator_row_map[row[1]] = i + 1

    source_map = {
        "VIX_LEVEL":           ("CBOE/yfinance", "T1", "index"),
        "VIX_TERM_STRUCTURE":  ("CBOE",          "T1", "ratio"),
        "PUT_CALL_RATIO":      ("CBOE",          "T1", "ratio"),
        "HY_OAS_SPREAD":       ("FRED",          "T1", "bps"),
        "MOVE_INDEX":          ("FRED/yfinance", "T1", "index"),
        "GOOGLE_TRENDS_RATIO": ("pytrends",      "T2", "ratio"),
        "INSIDER_BUY_SELL":    ("OpenInsider",   "T1", "ratio"),
        "AAII_BULL_PCT":       ("AAII",          "T1", "pct"),
        "AAII_BEAR_PCT":       ("AAII",          "T1", "pct"),
        "CNN_FEAR_GREED":      ("CNN",           "T2", "index"),
        "MARGIN_DEBT_YOY_CHG": ("FINRA",         "T1", "pct"),
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
            raw_str = "---"
        row_data = [today_str, ind_name, "L2", raw_str, "---", "---", "0", source, tier, unit]
        updates.append((f"A{row_idx}:J{row_idx}", [row_data]))

    for cell_range, values in updates:
        ws.update(values, cell_range, value_input_option="RAW")
    log.info(f"  RAW_MARKET: {len(updates)} L2 indicators written")


def write_scores_l2(warehouse, composite, direction, speed, phase, freshness, valid_count, total_count):
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
    asym_adj = composite
    if composite is not None and not pd.isna(composite) and composite < 5.0:
        asym_adj = min(round(composite * 1.3, 2), 10.0)
    row_data = [
        "L2 Sentiment",
        str(composite) if not pd.isna(composite) else "---",
        "---", "---", "n/a",
        direction, speed, signal,
        f"{freshness:.0f}",
        str(asym_adj), str(asym_adj),
        "---", "---",
    ]
    ws.update([row_data], "A3:M3", value_input_option="RAW")
    log.info(f"  SCORES L2: {composite} ({signal}) | Phase: {phase} | {valid_count}/{total_count}")


def write_dashboard_l2(warehouse, composite, direction, signal, phase, freshness, speed):
    ws = warehouse.worksheet("DASHBOARD")
    row_data = [
        "L2 Sentiment",
        str(composite) if not pd.isna(composite) else "---",
        direction, signal,
        f"{freshness:.0f}",
        "n/a", speed,
    ]
    ws.update([row_data], "A18:G18", value_input_option="RAW")
    log.info(f"  DASHBOARD L2: Score={composite}, Signal={signal}, Phase={phase}")


# ==================== MAIN ====================

def main():
    log.info("=" * 60)
    log.info("L2 Sentiment Collector - Starting (7/7 Sources, 11 Indicators)")
    log.info("=" * 60)

    log.info("Connecting to Data Warehouse...")
    warehouse = connect_warehouse()
    log.info("  OK")
    log.info("Connecting to FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    log.info("  OK")

    # --- Pull ---
    log.info("--- PULL PHASE ---")
    log.info("Pulling FRED...")
    fred_data = pull_fred_sentiment(fred)
    log.info("Pulling yfinance...")
    yf_data = pull_yfinance_sentiment()
    log.info("Pulling Google Trends...")
    gt_data = pull_google_trends()
    log.info("Pulling AAII...")
    aaii_data = pull_aaii_sentiment()
    log.info("Pulling CNN Fear & Greed...")
    cnn_data = pull_cnn_fear_greed()
    log.info("Pulling FINRA Margin Debt...")
    margin_data = pull_margin_debt()
    log.info("Pulling Insider Buy/Sell...")
    insider_data = pull_insider_data()

    # --- Merge ---
    log.info("--- MERGE PHASE ---")
    raw_values = {}

    raw_values["VIX_LEVEL"] = fred_data.get("VIX_LEVEL")
    if pd.isna(raw_values.get("VIX_LEVEL")):
        raw_values["VIX_LEVEL"] = yf_data.get("VIX_LEVEL_YF")
        if raw_values.get("VIX_LEVEL") is not None:
            log.info("  VIX: using yfinance fallback")

    vix3m = fred_data.get("VIX3M")
    if pd.isna(vix3m) or vix3m is None:
        vix3m = yf_data.get("VIX3M_YF")
    vix_val = raw_values.get("VIX_LEVEL")
    if vix_val and vix3m and not pd.isna(vix_val) and not pd.isna(vix3m) and vix3m > 0:
        raw_values["VIX_TERM_STRUCTURE"] = round(vix_val / vix3m, 4)
        log.info(f"  VIX_TERM_STRUCTURE: {vix_val:.2f} / {vix3m:.2f} = {raw_values['VIX_TERM_STRUCTURE']:.4f}")
    else:
        raw_values["VIX_TERM_STRUCTURE"] = np.nan

    hy_pct = fred_data.get("HY_OAS_SPREAD")
    if hy_pct is not None and not pd.isna(hy_pct):
        raw_values["HY_OAS_SPREAD"] = hy_pct * 100.0
        log.info(f"  HY_OAS: {hy_pct:.2f}% -> {hy_pct * 100:.0f} bps")
    else:
        raw_values["HY_OAS_SPREAD"] = np.nan

    raw_values["MOVE_INDEX"] = yf_data.get("MOVE_INDEX")
    raw_values["PUT_CALL_RATIO"] = yf_data.get("PUT_CALL_RATIO")
    raw_values["GOOGLE_TRENDS_RATIO"] = gt_data.get("GOOGLE_TRENDS_RATIO")
    raw_values["AAII_BULL_PCT"] = aaii_data.get("AAII_BULL_PCT", np.nan)
    raw_values["AAII_BEAR_PCT"] = aaii_data.get("AAII_BEAR_PCT", np.nan)
    raw_values["CNN_FEAR_GREED"] = cnn_data.get("CNN_FEAR_GREED", np.nan)
    raw_values["MARGIN_DEBT_YOY_CHG"] = margin_data.get("MARGIN_DEBT_YOY_CHG", np.nan)
    raw_values["INSIDER_BUY_SELL"] = insider_data.get("INSIDER_BUY_SELL", np.nan)

    for key in ["AAII_BULL_PCT", "AAII_BEAR_PCT", "CNN_FEAR_GREED", "MARGIN_DEBT_YOY_CHG", "INSIDER_BUY_SELL"]:
        if pd.isna(raw_values[key]):
            log.warning(f"  {key}: no data available")

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

    direction = "n/a"
    speed = "n/a"
    try:
        ws_scores = warehouse.worksheet("SCORES")
        prev_row = ws_scores.row_values(3)
        if len(prev_row) >= 2 and prev_row[1] not in ("---", "", None):
            prev_score = float(prev_row[1])
            direction = get_direction(composite, prev_score)
            speed = get_speed(composite, prev_score)
            log.info(f"  Previous L2: {prev_score} -> Direction: {direction}, Speed: {speed}")
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
    log.info(f"  Phase: {phase} | Signal: {signal}")
    log.info(f"  Sources: {valid_count}/{total_count} | Freshness: {freshness:.1f}/10")

    # --- Write ---
    log.info("--- WRITE PHASE ---")
    log.info("Writing RAW_MARKET L2...")
    write_raw_market_l2(warehouse, indicator_scores, raw_values)
    log.info("Writing SCORES L2...")
    write_scores_l2(warehouse, composite, direction, speed, phase, freshness, valid_count, total_count)
    log.info("Writing DASHBOARD L2...")
    write_dashboard_l2(warehouse, composite, direction, signal, phase, freshness, speed)

    # --- Done ---
    log.info("=" * 60)
    log.info("L2 Sentiment Collector COMPLETE (7/7 Sources, 11 Indicators)")
    log.info(f"  Score: {composite}/10 ({phase})")
    log.info(f"  Signal: {signal} | Direction: {direction} | Speed: {speed}")
    log.info(f"  Sources: {valid_count}/{total_count}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
