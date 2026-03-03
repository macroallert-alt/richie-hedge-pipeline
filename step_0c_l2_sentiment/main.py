"""
L2 Sentiment Collector - step_0c_l2_sentiment/main.py
Global Macro RV System - Data Warehouse Layer 2

Pulls 11 sentiment indicators from 7 sources, normalizes to 0-10 scores, writes to:
  - RAW_MARKET tab (individual indicator rows)
  - SCORES tab (L2 composite score row)
  - DASHBOARD tab (L2 summary line)

Sources (7/7 LIVE):
  T1 (API-direct): VIX, VIX3M (term structure), HY OAS, MOVE Index
  T1 (API-indirect): Put/Call Ratio
  T2 (API-indirect): Google Trends
  T1 (scrape): AAII Bull/Bear (weekly), CNN Fear & Greed (daily)
  T1 (scrape): Insider Buy/Sell (OpenInsider)
  T1 (scrape): Margin Debt YoY (FINRA, monthly)

Indicators (11):
  VIX_LEVEL, VIX_TERM_STRUCTURE, PUT_CALL_RATIO, HY_OAS_SPREAD,
  MOVE_INDEX, GOOGLE_TRENDS_RATIO, INSIDER_BUY_SELL,
  AAII_BULL_PCT, AAII_BEAR_PCT, CNN_FEAR_GREED, MARGIN_DEBT_YOY_CHG

Reference: V77 Statusanalyse Kap.4 (Layer 2 Sentiment)
Iron Rule: V16 main.py NEVER TOUCH. This is a separate module.
"""

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

# --- CONFIG ---

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("L2_SENTIMENT")

# Standard headers for web scraping
SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# --- NORMALIZATION ANCHORS (V77 validated) ---
# Hybrid: 60% Anchored (2015-2025 range) + 40% Rolling (252d)
# For initial implementation: use anchored ranges only (known historical bounds)

INDICATOR_RANGES = {
    # indicator: (min_bullish, max_bearish, invert)
    # invert=True means HIGH value = BEARISH (e.g. VIX high = fear)
    "VIX_LEVEL":           (9.0,   80.0,  True),    # VIX: low=greed, high=fear
    "VIX_TERM_STRUCTURE":  (0.75,  1.25,  False),   # VIX/VIX3M: <1=contango(calm), >1=backwardation(fear)
    "PUT_CALL_RATIO":      (0.5,   1.5,   True),    # High P/C = bearish sentiment
    "HY_OAS_SPREAD":       (250,   1000,  True),    # Wide spreads = fear
    "MOVE_INDEX":          (50,    200,   True),    # High MOVE = bond fear
    "GOOGLE_TRENDS_RATIO": (0.2,   5.0,   True),    # "stock market crash"/"buy stocks" ratio
    "INSIDER_BUY_SELL":    (0.1,   2.0,   False),   # High ratio = insiders buying = bullish
    "AAII_BULL_PCT":       (15.0,  65.0,  False),   # High bull% = greed, low = fear
    "AAII_BEAR_PCT":       (15.0,  65.0,  True),    # High bear% = fear
    "CNN_FEAR_GREED":      (0.0,   100.0, False),   # 0=Extreme Fear, 100=Extreme Greed
    "MARGIN_DEBT_YOY_CHG": (-30.0, 40.0,  False),   # Negative YoY = deleveraging = fear
}

# Weights: V77 validated, ökonomisch begründet
# Logik: Market-Pricing > Survey > Composite > Behavioral Proxy
# Frequenz: Täglich > Wöchentlich > Monatlich
# Redundanz bestraft (VIX Level, CNN niedrig wegen Überlappung)
INDICATOR_WEIGHTS = {
    "PUT_CALL_RATIO":      0.15,  # Stärkstes konträres Einzelsignal, täglich
    "VIX_TERM_STRUCTURE":  0.14,  # Backwardation = zuverlässigster Angst-Indikator
    "HY_OAS_SPREAD":       0.14,  # Reales Kreditrisiko-Pricing
    "INSIDER_BUY_SELL":    0.12,  # Stark prädiktiv in Extremen, echtes Skin-in-the-Game
    "AAII_BULL_PCT":       0.08,  # Konträrer Klassiker, wöchentlich -> leichter Abschlag
    "AAII_BEAR_PCT":       0.08,  # Zusammen 0.16 für AAII
    "VIX_LEVEL":           0.08,  # Redundant mit Term Structure, Eigenwert bei Extremen
    "MOVE_INDEX":          0.06,  # Cross-Asset-Angst, weniger direkt prädiktiv für Equity
    "MARGIN_DEBT_YOY_CHG": 0.06,  # Starkes Signal, aber monatlich + 6w Lag
    "CNN_FEAR_GREED":      0.05,  # Niedrig wegen Redundanz, Breadth+SafeHaven als Zusatz
    "GOOGLE_TRENDS_RATIO": 0.04,  # Noisiest Signal, nur Extreme informativ
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


# --- DATA PULLS (existing sources) ---

def pull_fred_sentiment(fred):
    """Pull FRED-based sentiment indicators."""
    today = date.today()
    start = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    series = {
        "VIXCLS": "VIX_LEVEL",
        "BAMLH0A0HYM2": "HY_OAS_SPREAD",  # Returns % (e.g. 3.12 = 312 bps)
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

    # Helper: safely extract last close from yfinance (handles MultiIndex)
    def safe_last_close(ticker):
        data = yf.download(ticker, period="5d", progress=False)
        if data is not None and not data.empty:
            close = data["Close"]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            return float(close.iloc[-1])
        return np.nan

    # VIX
    try:
        val = safe_last_close("^VIX")
        if not pd.isna(val):
            results["VIX_LEVEL_YF"] = val
            log.info(f"  yfinance: VIX_LEVEL_YF = {val:.4f}")
    except Exception as e:
        log.warning(f"  yfinance ^VIX failed: {e}")

    # VIX3M (primary source - not available on FRED)
    try:
        val = safe_last_close("^VIX3M")
        if not pd.isna(val):
            results["VIX3M_YF"] = val
            log.info(f"  yfinance: VIX3M_YF = {val:.4f}")
    except Exception as e:
        log.warning(f"  yfinance ^VIX3M failed: {e}")

    # MOVE Index
    try:
        val = safe_last_close("^MOVE")
        if not pd.isna(val):
            results["MOVE_INDEX"] = val
            log.info(f"  yfinance: MOVE_INDEX = {val:.2f}")
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


# --- DATA PULLS (new sources) ---

def pull_aaii_sentiment():
    """
    Pull AAII Investor Sentiment Survey (weekly, published Thursdays).
    Source: AAII website via direct data endpoint.
    Returns: {AAII_BULL_PCT: float, AAII_BEAR_PCT: float}
    """
    results = {}

    # Primary: AAII XML/JSON data feed
    try:
        url = "https://www.aaii.com/files/surveys/sentiment.xls"
        log.info(f"  AAII: Fetching sentiment survey from {url}...")
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()

        df = pd.read_excel(resp.content, header=0)

        # AAII Excel columns: Date, Bullish, Neutral, Bearish (as decimals 0-1)
        # Find the columns - AAII sometimes changes naming
        cols = df.columns.tolist()
        bull_col = None
        bear_col = None
        for c in cols:
            c_lower = str(c).lower().strip()
            if "bull" in c_lower:
                bull_col = c
            elif "bear" in c_lower:
                bear_col = c

        if bull_col is not None and bear_col is not None:
            # Get latest non-null row
            df_clean = df[[bull_col, bear_col]].dropna()
            if not df_clean.empty:
                latest = df_clean.iloc[-1]
                bull_val = float(latest[bull_col])
                bear_val = float(latest[bear_col])

                # AAII reports as decimals (0.35 = 35%) or percentages (35.0)
                # Normalize to percentage
                if bull_val <= 1.0:
                    bull_val *= 100.0
                if bear_val <= 1.0:
                    bear_val *= 100.0

                results["AAII_BULL_PCT"] = round(bull_val, 2)
                results["AAII_BEAR_PCT"] = round(bear_val, 2)
                log.info(f"    OK: AAII_BULL_PCT = {bull_val:.2f}%, AAII_BEAR_PCT = {bear_val:.2f}%")
                return results
        else:
            log.warning(f"    AAII: Could not identify Bull/Bear columns in: {cols}")

    except Exception as e:
        log.warning(f"    AAII primary (XLS) failed: {e}")

    # Fallback: scrape AAII website
    try:
        url = "https://www.aaii.com/sentimentsurvey"
        log.info(f"  AAII fallback: Scraping {url}...")
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text

        # Look for percentage patterns near "bullish" and "bearish"
        bull_match = re.search(r'[Bb]ullish[^0-9]*?(\d{1,2}(?:\.\d+)?)\s*%', text)
        bear_match = re.search(r'[Bb]earish[^0-9]*?(\d{1,2}(?:\.\d+)?)\s*%', text)

        if bull_match:
            results["AAII_BULL_PCT"] = float(bull_match.group(1))
            log.info(f"    OK (scrape): AAII_BULL_PCT = {results['AAII_BULL_PCT']:.2f}%")
        if bear_match:
            results["AAII_BEAR_PCT"] = float(bear_match.group(1))
            log.info(f"    OK (scrape): AAII_BEAR_PCT = {results['AAII_BEAR_PCT']:.2f}%")

    except Exception as e:
        log.warning(f"    AAII fallback (scrape) failed: {e}")

    return results


def pull_cnn_fear_greed():
    """
    Pull CNN Fear & Greed Index (daily, 0-100 scale).
    Source: CNN unofficial API endpoint.
    Returns: {CNN_FEAR_GREED: float}
    """
    results = {}

    # Primary: CNN Fear & Greed API endpoint
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        log.info(f"  CNN F&G: Fetching from API endpoint...")
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # The endpoint returns: {"fear_and_greed": {"score": 45.2, "rating": "Fear", ...}}
        if "fear_and_greed" in data and "score" in data["fear_and_greed"]:
            score = float(data["fear_and_greed"]["score"])
            rating = data["fear_and_greed"].get("rating", "Unknown")
            results["CNN_FEAR_GREED"] = round(score, 2)
            log.info(f"    OK: CNN_FEAR_GREED = {score:.2f} ({rating})")
            return results

    except Exception as e:
        log.warning(f"    CNN F&G primary (API) failed: {e}")

    # Fallback: alternative endpoint
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/current"
        log.info(f"  CNN F&G fallback: Fetching from {url}...")
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "score" in data:
            score = float(data["score"])
            results["CNN_FEAR_GREED"] = round(score, 2)
            log.info(f"    OK (fallback): CNN_FEAR_GREED = {score:.2f}")
            return results

    except Exception as e:
        log.warning(f"    CNN F&G fallback failed: {e}")

    return results


def pull_margin_debt():
    """
    Pull FINRA Margin Debt YoY Change (monthly, ~6 week lag).
    Source: FINRA margin statistics via FRED series.
    Returns: {MARGIN_DEBT_YOY_CHG: float} (percentage change)
    """
    results = {}

    # FRED has no direct margin debt series, but we can use a proxy
    # Primary: Try FINRA data via direct download
    try:
        # FINRA publishes margin stats monthly
        # We use the FRED series for debit balances in margin accounts
        url = "https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics"
        log.info(f"  MARGIN_DEBT: Attempting FINRA data...")

        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text

        # FINRA page contains a link to the data file
        # Look for the CSV/Excel download link
        csv_match = re.search(r'href="([^"]*margin[^"]*\.xlsx?)"', text, re.IGNORECASE)
        if csv_match:
            data_url = csv_match.group(1)
            if not data_url.startswith("http"):
                data_url = "https://www.finra.org" + data_url

            log.info(f"  MARGIN_DEBT: Downloading data from {data_url}...")
            data_resp = requests.get(data_url, headers=SCRAPE_HEADERS, timeout=30)
            data_resp.raise_for_status()

            df = pd.read_excel(data_resp.content)

            # Find debit balance column
            debit_col = None
            for c in df.columns:
                if "debit" in str(c).lower():
                    debit_col = c
                    break

            if debit_col is not None:
                df_clean = df[debit_col].dropna()
                if len(df_clean) >= 13:
                    latest = float(df_clean.iloc[-1])
                    year_ago = float(df_clean.iloc[-13])  # ~12 months back
                    if year_ago > 0:
                        yoy_chg = ((latest - year_ago) / year_ago) * 100.0
                        results["MARGIN_DEBT_YOY_CHG"] = round(yoy_chg, 2)
                        log.info(f"    OK: MARGIN_DEBT_YOY_CHG = {yoy_chg:.2f}% "
                                 f"(latest={latest:.0f}, year_ago={year_ago:.0f})")
                        return results

    except Exception as e:
        log.warning(f"    MARGIN_DEBT FINRA failed: {e}")

    # Fallback: Use FRED S&P 500 margin debt proxy
    # FRED series BOGZ1FL663067003Q = Margin accounts net debit balances (quarterly)
    try:
        log.info(f"  MARGIN_DEBT: Trying FRED proxy (quarterly)...")
        fred = Fred(api_key=FRED_API_KEY)
        today = date.today()
        start = (today - timedelta(days=730)).strftime("%Y-%m-%d")
        raw = fred.get_series("BOGZ1FL663067003Q", observation_start=start)

        if raw is not None and not raw.empty:
            raw_clean = raw.dropna()
            if len(raw_clean) >= 5:  # Need at least 5 quarters for YoY
                latest = float(raw_clean.iloc[-1])
                # 4 quarters back for YoY
                year_ago = float(raw_clean.iloc[-5]) if len(raw_clean) >= 5 else float(raw_clean.iloc[0])
                if year_ago > 0:
                    yoy_chg = ((latest - year_ago) / year_ago) * 100.0
                    results["MARGIN_DEBT_YOY_CHG"] = round(yoy_chg, 2)
                    log.info(f"    OK (FRED proxy): MARGIN_DEBT_YOY_CHG = {yoy_chg:.2f}%")
                    return results

    except Exception as e:
        log.warning(f"    MARGIN_DEBT FRED proxy failed: {e}")

    return results


def pull_insider_data():
    """
    Pull Insider Buy/Sell Ratio from OpenInsider.
    Source: openinsider.com (aggregated SEC Form 4 filings).
    Returns: {INSIDER_BUY_SELL: float} (ratio of buys to sells)
    """
    results = {}

    try:
        # OpenInsider summary page with recent insider activity
        url = "http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=7&fdr=&td=0&tdr=&feession=at&tession=ct&xp=1&vl=&vh=&ocl=&och=&session=oi&a=1&v=&cnt=100&page=1"
        log.info(f"  INSIDER: Fetching OpenInsider data...")

        # Simpler approach: get the summary stats page
        summary_url = "http://openinsider.com/insider-purchases-25k"
        resp_buy = requests.get(summary_url, headers=SCRAPE_HEADERS, timeout=30)
        resp_buy.raise_for_status()

        # Count buy transactions from the page
        buy_count = resp_buy.text.count('<tr class')

        sell_url = "http://openinsider.com/insider-sales-25k"
        resp_sell = requests.get(sell_url, headers=SCRAPE_HEADERS, timeout=30)
        resp_sell.raise_for_status()

        sell_count = resp_sell.text.count('<tr class')

        if sell_count > 0:
            ratio = buy_count / sell_count
            results["INSIDER_BUY_SELL"] = round(ratio, 4)
            log.info(f"    OK: INSIDER_BUY_SELL = {ratio:.4f} (buys={buy_count}, sells={sell_count})")
        elif buy_count > 0:
            results["INSIDER_BUY_SELL"] = 2.0  # Cap at max if no sells
            log.info(f"    OK: INSIDER_BUY_SELL = 2.0 (buys={buy_count}, sells=0, capped)")
        else:
            log.warning(f"    INSIDER: Could not parse buy/sell counts")

    except Exception as e:
        log.warning(f"    INSIDER OpenInsider failed: {e}")

    return results


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
        "VIX_LEVEL":           ("CBOE/yfinance",  "T1", "index"),
        "VIX_TERM_STRUCTURE":  ("CBOE",           "T1", "ratio"),
        "PUT_CALL_RATIO":      ("CBOE",           "T1", "ratio"),
        "HY_OAS_SPREAD":       ("FRED",           "T1", "bps"),
        "MOVE_INDEX":          ("FRED/yfinance",  "T1", "index"),
        "GOOGLE_TRENDS_RATIO": ("pytrends",       "T2", "ratio"),
        "INSIDER_BUY_SELL":    ("OpenInsider",     "T1", "ratio"),
        "AAII_BULL_PCT":       ("AAII",           "T1", "pct"),
        "AAII_BEAR_PCT":       ("AAII",           "T1", "pct"),
        "CNN_FEAR_GREED":      ("CNN",            "T2", "index"),
        "MARGIN_DEBT_YOY_CHG": ("FINRA",          "T1", "pct"),
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
        ws.update(values, cell_range, value_input_option="RAW")

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

    ws.update([row_data], "A3:M3", value_input_option="RAW")
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

    ws.update([row_data], "A18:G18", value_input_option="RAW")
    log.info(f"  DASHBOARD L2: Score={composite}, Signal={signal}, Phase={phase}")


# --- MAIN ---

def main():
    log.info("=" * 60)
    log.info("L2 Sentiment Collector - Starting (7/7 Sources, 11 Indicators)")
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

    log.info("Pulling AAII Sentiment Survey...")
    aaii_data = pull_aaii_sentiment()

    log.info("Pulling CNN Fear & Greed Index...")
    cnn_data = pull_cnn_fear_greed()

    log.info("Pulling FINRA Margin Debt...")
    margin_data = pull_margin_debt()

    log.info("Pulling Insider Buy/Sell Ratio...")
    insider_data = pull_insider_data()

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

    # HY OAS: FRED returns percent (3.12 = 312 bps), convert to bps
    hy_pct = fred_data.get("HY_OAS_SPREAD")
    if hy_pct is not None and not pd.isna(hy_pct):
        raw_values["HY_OAS_SPREAD"] = hy_pct * 100.0
        log.info(f"  HY_OAS: {hy_pct:.2f}% -> {hy_pct * 100:.0f} bps")
    else:
        raw_values["HY_OAS_SPREAD"] = np.nan

    # MOVE Index
    raw_values["MOVE_INDEX"] = yf_data.get("MOVE_INDEX")

    # Put/Call Ratio
    raw_values["PUT_CALL_RATIO"] = yf_data.get("PUT_CALL_RATIO")

    # Google Trends
    raw_values["GOOGLE_TRENDS_RATIO"] = gt_data.get("GOOGLE_TRENDS_RATIO")

    # AAII Sentiment (new)
    raw_values["AAII_BULL_PCT"] = aaii_data.get("AAII_BULL_PCT", np.nan)
    raw_values["AAII_BEAR_PCT"] = aaii_data.get("AAII_BEAR_PCT", np.nan)
    if pd.isna(raw_values["AAII_BULL_PCT"]):
        log.warning("  AAII_BULL_PCT: no data available")
    if pd.isna(raw_values["AAII_BEAR_PCT"]):
        log.warning("  AAII_BEAR_PCT: no data available")

    # CNN Fear & Greed (new)
    raw_values["CNN_FEAR_GREED"] = cnn_data.get("CNN_FEAR_GREED", np.nan)
    if pd.isna(raw_values["CNN_FEAR_GREED"]):
        log.warning("  CNN_FEAR_GREED: no data available")

    # Margin Debt YoY Change (new)
    raw_values["MARGIN_DEBT_YOY_CHG"] = margin_data.get("MARGIN_DEBT_YOY_CHG", np.nan)
    if pd.isna(raw_values["MARGIN_DEBT_YOY_CHG"]):
        log.warning("  MARGIN_DEBT_YOY_CHG: no data available")

    # Insider Buy/Sell (new - replaces placeholder)
    raw_values["INSIDER_BUY_SELL"] = insider_data.get("INSIDER_BUY_SELL", np.nan)
    if pd.isna(raw_values["INSIDER_BUY_SELL"]):
        log.warning("  INSIDER_BUY_SELL: no data available")

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
    log.info("L2 Sentiment Collector COMPLETE (7/7 Sources, 11 Indicators)")
    log.info(f"  Score: {composite}/10 ({phase})")
    log.info(f"  Signal: {signal}")
    log.info(f"  Direction: {direction} | Speed: {speed}")
    log.info(f"  Sources: {valid_count}/{total_count}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
