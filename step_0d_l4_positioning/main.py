"""
L4 Positioning Collector - step_0d_l4_positioning/main.py
Global Macro RV System - Data Warehouse Layer 4
6/6 Indicators LIVE:
  Phase 1: COT x 3 (S&P 500, Gold, 10Y Treasury) — CFTC XLS ZIPs
  Phase 2: Crypto Funding Rate — OKX→Bybit→Binance fallback chain
  Phase 3: Options GEX — EODHD Marketplace API with pagination
  Phase 4: Fund Flows Equity — ICI MF-only scrape (% of Assets)
Pulls positioning indicators, normalizes to 0-10, writes to:
  RAW_MARKET (R14-R19), SCORES (R5), DASHBOARD (R20)
Pattern: Identical to L2 Sentiment Collector (Pull -> Normalize -> Composite -> Sheets)
Reference: V79 Statusanalyse Kap.10, Kap.18, Kap.19
Iron Rule: V16 main.py NEVER TOUCH. L2 step_0c NEVER TOUCH.
"""

import io
import os
import sys
import logging
import zipfile
from datetime import datetime, timedelta, date, timezone

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import requests

WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
EODHD_API_KEY = os.environ.get("EODHD_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("L4_POSITIONING")

# ==================== CONSTANTS ====================

INDICATOR_WEIGHTS = {
    "COT_SP500_COMM_NET":   0.20,
    "COT_GOLD_COMM_NET":    0.20,
    "COT_TREASURY_COMM_NET": 0.20,
    "FUND_FLOWS_EQUITY":    0.15,
    "CRYPTO_FUNDING_RATE":  0.10,
    "OPTIONS_GEX":          0.15,
}

# COT contract identification — search strings for Market_and_Exchange_Names
COT_CONTRACTS = {
    "COT_SP500_COMM_NET": {
        "search_strings": ["S&P 500", "E-MINI S&P 500"],
        "cftc_codes": ["13874A", "138741"],
    },
    "COT_GOLD_COMM_NET": {
        "search_strings": ["GOLD"],
        "cftc_codes": ["088691"],
    },
    "COT_TREASURY_COMM_NET": {
        "search_strings": ["10-YEAR", "10 YEAR", "UST 10Y", "T-NOTE", "U.S. TREASURY NOTE"],
        "cftc_codes": ["043602", "043602C"],
    },
}

# Normalization: COT uses percentile/10, others use linear ranges
# Phase 2-4 will add: FUND_FLOWS_EQUITY, CRYPTO_FUNDING_RATE, OPTIONS_GEX
INDICATOR_RANGES = {
    "FUND_FLOWS_EQUITY":   (-0.20, 0.20, False),
    "CRYPTO_FUNDING_RATE": (-0.10, 0.10, True),
    "OPTIONS_GEX":         (-5.0, 15.0, False),
}

# L4 Positioning phases (V78 Kap.10 / Addendum G1)
POSITIONING_PHASES = [
    (1.5,  "Extreme Short"),
    (3.0,  "Heavy Short"),
    (4.0,  "Lean Short"),
    (4.5,  "Slight Short"),
    (5.5,  "Neutral"),
    (6.0,  "Slight Long"),
    (7.0,  "Lean Long"),
    (8.5,  "Heavy Long"),
    (10.0, "Extreme Long"),
]

# Phase -> SIGNAL mapping (V78 Kap.10 / Addendum G1)
PHASE_TO_SIGNAL = {
    "Extreme Short": "Bearish",
    "Heavy Short":   "Bearish",
    "Lean Short":    "Bearish",
    "Slight Short":  "Neutral",
    "Neutral":       "Neutral",
    "Slight Long":   "Neutral",
    "Lean Long":     "Bullish",
    "Heavy Long":    "Bullish",
    "Extreme Long":  "Extreme",
}

# Freshness parameters (Addendum H)
FRESHNESS_PARAMS = {
    "COT_SP500_COMM_NET":   {"ideal": 3, "decay_rate": 1.0},
    "COT_GOLD_COMM_NET":    {"ideal": 3, "decay_rate": 1.0},
    "COT_TREASURY_COMM_NET": {"ideal": 3, "decay_rate": 1.0},
    "FUND_FLOWS_EQUITY":    {"ideal": 3, "decay_rate": 1.0},
    "CRYPTO_FUNDING_RATE":  {"ideal": 0, "decay_rate": 4.0},
    "OPTIONS_GEX":          {"ideal": 0, "decay_rate": 3.0},
}

# Source metadata for RAW_MARKET (with V78 corrections applied)
SOURCE_MAP = {
    "COT_SP500_COMM_NET":   ("CFTC", "T1", "pct"),
    "COT_GOLD_COMM_NET":    ("CFTC", "T1", "pct"),
    "COT_TREASURY_COMM_NET": ("CFTC", "T1", "pct"),
    "FUND_FLOWS_EQUITY":    ("ICI",  "T3", "pct_aum"),
    "CRYPTO_FUNDING_RATE":  ("OKX/Bybit/Binance", "T2", "pct"),
    "OPTIONS_GEX":          ("EODHD", "T2", "$B"),
}

SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

MIN_SOURCES_FOR_COMPOSITE = 4
COT_LOOKBACK_YEARS = 5
COT_LOOKBACK_WEEKS = 260


# ==================== WAREHOUSE CONNECTION ====================

def connect_warehouse():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    gc = gspread.authorize(creds)
    return gc.open_by_key(WAREHOUSE_SHEET_ID)


# ==================== COT DATA PULL ====================

def download_cftc_year(year):
    """Download CFTC Legacy Futures-Only Excel ZIP for a given year.
    URL pattern from CFTC Historical Compressed page:
      https://www.cftc.gov/files/dea/history/dea_fut_xls_{YEAR}.zip
    Excel files have proper column headers (unlike the TXT files which are headerless).
    """
    url = f"https://www.cftc.gov/files/dea/history/dea_fut_xls_{year}.zip"
    log.info(f"    CFTC: Downloading {url}...")
    try:
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=90)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        # Find Excel file inside ZIP
        xls_names = [n for n in zf.namelist() if n.endswith(".xls") or n.endswith(".xlsx")]
        if not xls_names:
            log.warning(f"    CFTC {year}: No XLS in ZIP, trying CSV/TXT...")
            txt_names = [n for n in zf.namelist() if n.endswith(".csv") or n.endswith(".txt")]
            if not txt_names:
                log.warning(f"    CFTC {year}: No data files in ZIP")
                return None
            with zf.open(txt_names[0]) as f:
                df = pd.read_csv(f)
            log.info(f"    CFTC {year}: {len(df)} rows loaded (CSV)")
            return df
        xls_name = xls_names[0]
        log.info(f"    CFTC {year}: Reading {xls_name}...")
        with zf.open(xls_name) as f:
            df = pd.read_excel(io.BytesIO(f.read()))
        log.info(f"    CFTC {year}: {len(df)} rows loaded (XLS)")
        return df
    except Exception as e:
        log.warning(f"    CFTC {year}: Download failed -> {e}")
        return None


def filter_cot_contract(df, indicator_name):
    """Filter COT dataframe for a specific contract using search strings and CFTC codes."""
    contract_info = COT_CONTRACTS[indicator_name]
    search_strings = contract_info["search_strings"]
    cftc_codes = contract_info["cftc_codes"]

    # Try Market_and_Exchange_Names first (contains-match)
    name_col = None
    for col in df.columns:
        if "market_and_exchange" in col.lower().replace(" ", "_"):
            name_col = col
            break
    if name_col is None:
        for col in df.columns:
            if "market" in col.lower() and "name" in col.lower():
                name_col = col
                break

    code_col = None
    for col in df.columns:
        if "cftc" in col.lower() and "code" in col.lower():
            code_col = col
            break

    mask = pd.Series([False] * len(df), index=df.index)

    if name_col is not None:
        for ss in search_strings:
            mask = mask | df[name_col].astype(str).str.contains(ss, case=False, na=False)

    if code_col is not None and mask.sum() == 0:
        for cc in cftc_codes:
            mask = mask | (df[code_col].astype(str).str.strip() == cc)

    return df[mask].copy()


def get_cot_columns(df):
    """Find the relevant COT columns regardless of exact naming."""
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "report_date" in cl.replace(" ", "_") and "yyyy" in cl:
            col_map["date"] = col
        elif cl == "report_date_as_yyyy-mm-dd":
            col_map["date"] = col
        elif "comm_positions_long_all" in cl.replace(" ", "_") or (
                "commercial" in cl and "long" in cl and "all" in cl and "spread" not in cl):
            col_map["long"] = col
        elif "comm_positions_short_all" in cl.replace(" ", "_") or (
                "commercial" in cl and "short" in cl and "all" in cl and "spread" not in cl):
            col_map["short"] = col
        elif ("open_interest_all" in cl.replace(" ", "_") or (
                "open" in cl and "interest" in cl and "all" in cl and "old" not in cl)):
            # We need absolute Open_Interest_All — exclude Pct, Change, Other variants
            if ("pct" not in cl and "percent" not in cl and 
                    "other" not in cl and "change" not in cl):
                col_map["oi"] = col

    return col_map


def calc_comm_net_pct(df_contract, col_map):
    """Calculate COMM_NET_PCT = (Long - Short) / OI * 100 for all rows."""
    df_c = df_contract.copy()
    df_c["_long"] = pd.to_numeric(df_c[col_map["long"]], errors="coerce")
    df_c["_short"] = pd.to_numeric(df_c[col_map["short"]], errors="coerce")
    df_c["_oi"] = pd.to_numeric(df_c[col_map["oi"]], errors="coerce")
    df_c["_date"] = pd.to_datetime(df_c[col_map["date"]], errors="coerce")

    # Filter out rows with invalid data
    valid = df_c["_oi"] > 0
    valid = valid & df_c["_long"].notna() & df_c["_short"].notna() & df_c["_date"].notna()
    df_c = df_c[valid].copy()

    df_c["comm_net_pct"] = (df_c["_long"] - df_c["_short"]) / df_c["_oi"] * 100.0
    df_c = df_c.sort_values("_date").reset_index(drop=True)

    return df_c[["_date", "comm_net_pct", "_long", "_short", "_oi"]]


def calc_percentile(current_value, history_series):
    """Calculate percentile of current value within history. Returns 0-100."""
    if len(history_series) == 0:
        return 50.0
    rank = (history_series < current_value).sum()
    percentile = rank / len(history_series) * 100.0
    return round(percentile, 2)


def pull_cot_data():
    """Pull COT data for all 3 contracts with 5Y percentile calculation.
    
    Downloads CFTC Legacy Futures CSV for current year + previous years
    to build up to 5 years of history for percentile ranking.
    Returns dict with indicator_name -> {raw_value, percentile, score, date, age_days}
    """
    results = {}
    today = date.today()
    current_year = today.year

    # Download all years via Excel ZIPs (current year + historical for 5Y percentile)
    # URL pattern: https://www.cftc.gov/files/dea/history/dea_fut_xls_{YEAR}.zip
    years_to_download = list(range(current_year - COT_LOOKBACK_YEARS, current_year + 1))
    log.info(f"  COT: Downloading {len(years_to_download)} years ({years_to_download[0]}-{years_to_download[-1]}) for {COT_LOOKBACK_YEARS}Y percentile...")

    all_dfs = []
    for year in years_to_download:
        df_year = download_cftc_year(year)
        if df_year is not None:
            all_dfs.append(df_year)

    if not all_dfs:
        log.error("  COT: No CFTC data downloaded at all!")
        return _cot_fallback_library()

    df_all = pd.concat(all_dfs, ignore_index=True)
    log.info(f"  COT: Combined {len(df_all)} total rows across {len(all_dfs)} years")

    # Find column names once
    col_map = get_cot_columns(df_all)
    required_cols = ["date", "long", "short", "oi"]
    missing = [c for c in required_cols if c not in col_map]
    if missing:
        log.error(f"  COT: Missing columns: {missing}")
        log.info(f"  COT: Available columns: {list(df_all.columns[:20])}")
        return _cot_fallback_library()

    log.info(f"  COT: Column mapping: date={col_map.get('date','??')}, long={col_map.get('long','??')}, short={col_map.get('short','??')}, oi={col_map.get('oi','??')}") 
    # Diagnostic: log all column names containing 'open' for debugging
    oi_candidates = [c for c in df_all.columns if "open" in c.lower() and "interest" in c.lower()]
    log.info(f"  COT: OI column candidates: {oi_candidates}")

    # Process each contract
    for indicator_name, contract_info in COT_CONTRACTS.items():
        log.info(f"  COT: Processing {indicator_name}...")
        df_contract = filter_cot_contract(df_all, indicator_name)

        if df_contract.empty:
            log.warning(f"    {indicator_name}: No matching contracts found")
            continue

        log.info(f"    {indicator_name}: {len(df_contract)} rows matched")

        # Calculate COMM_NET_PCT for all rows
        df_calc = calc_comm_net_pct(df_contract, col_map)
        if df_calc.empty:
            log.warning(f"    {indicator_name}: No valid data after calculation")
            continue

        # Get latest value
        latest = df_calc.iloc[-1]
        latest_date = latest["_date"].date()
        latest_value = round(latest["comm_net_pct"], 4)
        age_days = (today - latest_date).days

        # 5Y lookback window for percentile
        cutoff_date = today - timedelta(days=COT_LOOKBACK_YEARS * 365)
        history = df_calc[df_calc["_date"].dt.date >= cutoff_date]["comm_net_pct"]

        # Calculate percentile (excluding the latest value itself for clean ranking)
        if len(history) > 1:
            history_excl = history.iloc[:-1]
            percentile = calc_percentile(latest_value, history_excl)
        else:
            percentile = 50.0

        score = round(percentile / 10.0, 2)

        results[indicator_name] = {
            "raw_value": latest_value,
            "percentile": percentile,
            "score": score,
            "date": latest_date.strftime("%Y-%m-%d"),
            "age_days": age_days,
        }

        log.info(f"    {indicator_name}: Net={latest_value:.2f}% of OI, Percentile={percentile:.1f} ({COT_LOOKBACK_YEARS}Y), Score={score:.2f}, Date={latest_date}, Age={age_days}d")

    if not results:
        log.warning("  COT: Primary failed for all contracts, trying cot_reports library...")
        return _cot_fallback_library()

    return results


def _cot_fallback_library():
    """Fallback: try cot_reports Python library for COT data."""
    try:
        from cot_reports import cot_report
        log.info("  COT Fallback: Using cot_reports library...")
        df = cot_report(report_type="legacy_fut", cot_report_type="all_disagg")
        if df is not None and not df.empty:
            log.info(f"  COT Fallback: Got {len(df)} rows from cot_reports")
            # Process same as primary — this is a simplified fallback
            # For now return empty, will use last known value
            log.warning("  COT Fallback: cot_reports parsing not yet implemented, using last known values")
    except ImportError:
        log.warning("  COT Fallback: cot_reports not installed")
    except Exception as e:
        log.warning(f"  COT Fallback: cot_reports failed -> {e}")
    return {}


# ==================== PHASE 2: CRYPTO FUNDING ====================

def pull_crypto_funding():
    """Pull perpetual funding rates with multi-exchange fallback chain.

    Cloud/datacenter IPs are often blocked by crypto exchanges.
    We try multiple sources: OKX → Bybit → Binance.
    All are public endpoints, no API key needed.
    Funding rates are highly correlated across exchanges (arbitrage).

    Returns: {"CRYPTO_FUNDING_RATE": {"raw_value": pct, "age_days": int, "date": str}}
    """
    results = {}
    today = date.today()
    btc_rate = None

    # --- Source chain for BTCUSDT ---
    sources = [
        ("OKX", "https://www.okx.com/api/v5/public/funding-rate", 
         {"instId": "BTC-USDT-SWAP"}, _parse_okx_funding),
        ("Bybit", "https://api.bybit.com/v5/market/funding/history",
         {"category": "linear", "symbol": "BTCUSDT", "limit": "1"}, _parse_bybit_funding),
        ("Binance", "https://fapi.binance.com/fapi/v1/fundingRate",
         {"symbol": "BTCUSDT", "limit": 1}, _parse_binance_funding),
    ]

    for name, url, params, parser in sources:
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rate_pct, rate_date = parser(data)
            if rate_pct is not None:
                age_days = max(0, (today - rate_date).days)
                btc_rate = rate_pct
                results["CRYPTO_FUNDING_RATE"] = {
                    "raw_value": rate_pct,
                    "age_days": age_days,
                    "date": rate_date.strftime("%Y-%m-%d"),
                }
                log.info(f"    BTCUSDT ({name}): {rate_pct:.4f}%, Date={rate_date}, Age={age_days}d")
                break
        except Exception as e:
            log.warning(f"    BTCUSDT ({name}): failed -> {e}")

    if "CRYPTO_FUNDING_RATE" not in results:
        log.warning("    BTCUSDT: ALL sources failed")

    # --- ETHUSDT validation (same source chain, first that works) ---
    eth_rate = None
    eth_sources = [
        ("OKX", "https://www.okx.com/api/v5/public/funding-rate",
         {"instId": "ETH-USDT-SWAP"}, _parse_okx_funding),
        ("Bybit", "https://api.bybit.com/v5/market/funding/history",
         {"category": "linear", "symbol": "ETHUSDT", "limit": "1"}, _parse_bybit_funding),
    ]
    for name, url, params, parser in eth_sources:
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rate_pct, _ = parser(data)
            if rate_pct is not None:
                eth_rate = rate_pct
                log.info(f"    ETHUSDT ({name}): {eth_rate:.4f}% (validation only)")
                break
        except Exception as e:
            log.warning(f"    ETHUSDT ({name}): failed -> {e}")

    # --- Divergence check ---
    if btc_rate is not None and eth_rate is not None:
        divergence = abs(btc_rate - eth_rate)
        if divergence > 0.05:
            log.warning(f"    BTC-ETH Divergence: {divergence:.4f}% > 0.05% -> LOW CONFIDENCE")
        else:
            log.info(f"    BTC-ETH Divergence: {divergence:.4f}% (OK)")

    return results


def _parse_okx_funding(data):
    """Parse OKX /api/v5/public/funding-rate response."""
    if data.get("code") == "0" and data.get("data"):
        item = data["data"][0]
        rate = float(item["fundingRate"]) * 100.0
        ts = int(item["fundingTime"])
        d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
        return rate, d
    return None, None


def _parse_bybit_funding(data):
    """Parse Bybit /v5/market/funding/history response."""
    if data.get("retCode") == 0 and data.get("result", {}).get("list"):
        item = data["result"]["list"][0]
        rate = float(item["fundingRate"]) * 100.0
        ts = int(item["fundingRateTimestamp"])
        d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
        return rate, d
    return None, None


def _parse_binance_funding(data):
    """Parse Binance /fapi/v1/fundingRate response."""
    if data and len(data) > 0:
        rate = float(data[0]["fundingRate"]) * 100.0
        ts = int(data[0]["fundingTime"])
        d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
        return rate, d
    return None, None

def pull_options_gex():
    """Calculate Gamma Exposure (GEX) from SPY options via EODHD Marketplace API.

    GEX = Sum of (OI * Gamma * 100 * Spot * sign) for all contracts 1-7 DTE.
    Calls contribute positive gamma, Puts contribute negative gamma.
    Result in $Billion.

    EODHD Marketplace Options API (unicornbay):
      GET /api/mp/unicornbay/options/contracts
      Fields: strike, type, open_interest, gamma, delta, exp_date
      Typically <500 contracts for SPY 1-7 DTE window.

    SPY spot price from EODHD EOD API.

    Returns: {"OPTIONS_GEX": {"raw_value": gex_billions, "age_days": 0, "date": str}}
    """
    if not EODHD_API_KEY:
        log.warning("  OPTIONS_GEX: No EODHD_API_KEY set, skipping")
        return {}

    results = {}
    today = date.today()

    # --- Step 1: Get SPY spot price via EODHD EOD API ---
    spy_spot = None
    try:
        eod_url = "https://eodhd.com/api/eod/SPY.US"
        resp = requests.get(eod_url, params={
            "api_token": EODHD_API_KEY,
            "fmt": "json",
            "from": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "order": "d",
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) > 0:
            spy_spot = float(data[0]["close"])
            log.info(f"    SPY Spot: ${spy_spot:.2f} (from {data[0].get('date', 'unknown')})")
        else:
            log.warning("    SPY Spot: No EOD data returned")
    except Exception as e:
        log.warning(f"    SPY Spot: EODHD EOD failed -> {e}")

    if spy_spot is None:
        log.warning("  OPTIONS_GEX: Cannot calculate without SPY spot price")
        return {}

    # --- Step 2: Pull SPY options chain 1-7 DTE ---
    exp_from = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    exp_to = (today + timedelta(days=7)).strftime("%Y-%m-%d")

    all_contracts = []
    try:
        opts_url = "https://eodhd.com/api/mp/unicornbay/options/contracts"
        page_limit = 1000
        offset = 0
        total_raw = 0
        max_pages = 5  # safety cap: 5 * 1000 = 5000 contracts max

        for page in range(max_pages):
            params = {
                "filter[underlying_symbol]": "SPY",
                "filter[exp_date_from]": exp_from,
                "filter[exp_date_to]": exp_to,
                "fields[options-contracts]": "contract,type,strike,exp_date,open_interest,gamma,delta,dte",
                "sort": "strike",
                "page[limit]": page_limit,
                "page[offset]": offset,
                "compact": 1,
                "api_token": EODHD_API_KEY,
            }
            resp = requests.get(opts_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            contracts = data.get("data", [])
            if not contracts:
                if page == 0:
                    log.warning(f"    OPTIONS: No contracts returned for SPY {exp_from} to {exp_to}")
                    return {}
                break

            for c in contracts:
                attrs = c.get("attributes", c)
                try:
                    oi = float(attrs.get("open_interest", 0) or 0)
                    gamma = float(attrs.get("gamma", 0) or 0)
                    strike = float(attrs.get("strike", 0) or 0)
                    opt_type = str(attrs.get("type", "")).lower()
                    if oi > 0 and gamma > 0:
                        all_contracts.append({
                            "oi": oi,
                            "gamma": gamma,
                            "strike": strike,
                            "type": opt_type,
                        })
                except (ValueError, TypeError):
                    continue

            total_raw += len(contracts)
            meta = data.get("meta", {})
            total_available = meta.get("total", total_raw)

            if total_raw >= total_available or len(contracts) < page_limit:
                break
            offset += page_limit

        log.info(f"    OPTIONS: {len(all_contracts)} valid contracts from {total_raw} fetched ({total_available} total, 1-7 DTE, {page+1} pages)")

    except Exception as e:
        log.warning(f"    OPTIONS: EODHD options API failed -> {e}")
        return {}

    if not all_contracts:
        log.warning("    OPTIONS: No contracts with valid OI and gamma")
        return {}

    # --- Step 3: Calculate GEX ---
    # Dealer convention: dealers are net short options
    # Call GEX positive (dealers buy stock as price rises = stabilizing)
    # Put GEX negative (dealers sell stock as price drops = destabilizing)
    total_gex = 0.0
    call_gex = 0.0
    put_gex = 0.0
    call_count = 0
    put_count = 0

    for c in all_contracts:
        contract_gex = c["oi"] * c["gamma"] * 100 * spy_spot
        if c["type"] == "call":
            total_gex += contract_gex
            call_gex += contract_gex
            call_count += 1
        elif c["type"] == "put":
            total_gex -= contract_gex
            put_gex += contract_gex
            put_count += 1

    gex_billions = total_gex / 1e9

    log.info(f"    GEX: ${gex_billions:.2f}B (Calls: {call_count} @ ${call_gex/1e9:.2f}B, Puts: {put_count} @ -${put_gex/1e9:.2f}B)")

    results["OPTIONS_GEX"] = {
        "raw_value": round(gex_billions, 4),
        "age_days": 0,
        "date": today.strftime("%Y-%m-%d"),
    }

    return results


def pull_fund_flows():
    """Phase 4: ICI Fund Flows — Equity flows as % of Assets.

    Primary: ICI MF-only Flows page (/research/stats/flows)
      - Parses "Equity funds had estimated [in/out]flows of $X.XX billion
        (X.X percent of [Month] [Day] assets)" from page text.
      - MF-only has % of assets directly in text (Combined page does not).
      - MF flows = active investor decisions (better positioning signal than ETF).

    Fallback 1: ICI Combined Flows page (/research/stats/combined_flows)
      - Parses absolute dollar equity flows, divides by known AUM (~$20T).

    Fallback 2: datahub.io CSV (monthly, MF only, absolute $M)
      - Divides by known AUM (~$20T).

    Normalization: -0.20% to +0.20% AUM, linear, not inverted.
    Outflows = negative (bearish positioning), Inflows = positive (bullish).

    Returns: {"FUND_FLOWS_EQUITY": {"raw_value": pct_of_aum, "age_days": int, "date": str}}
    """
    from bs4 import BeautifulSoup
    import re

    results = {}
    today = date.today()

    # --- Primary: ICI MF-only Flows (has % of assets in text) ---
    log.info("  FUND_FLOWS: Trying ICI MF-only Flows (Primary)...")
    try:
        result = _parse_ici_mf_flows(today)
        if result:
            results["FUND_FLOWS_EQUITY"] = result
            log.info(f"    ICI MF-only: {result['raw_value']:.4f}% of AUM, Date={result['date']}, Age={result['age_days']}d")
            return results
    except Exception as e:
        log.warning(f"    ICI MF-only: Failed -> {e}")

    # --- Fallback 1: ICI Combined Flows (absolute $ -> divide by AUM) ---
    log.info("  FUND_FLOWS: Trying ICI Combined Flows (Fallback 1)...")
    try:
        result = _parse_ici_combined_flows(today)
        if result:
            results["FUND_FLOWS_EQUITY"] = result
            log.info(f"    ICI Combined: {result['raw_value']:.4f}% of AUM, Date={result['date']}, Age={result['age_days']}d")
            return results
    except Exception as e:
        log.warning(f"    ICI Combined: Failed -> {e}")

    # --- Fallback 2: datahub.io CSV (monthly, absolute $M -> divide by AUM) ---
    log.info("  FUND_FLOWS: Trying datahub.io CSV (Fallback 2)...")
    try:
        result = _parse_datahub_flows(today)
        if result:
            results["FUND_FLOWS_EQUITY"] = result
            log.info(f"    datahub.io: {result['raw_value']:.4f}% of AUM, Date={result['date']}, Age={result['age_days']}d")
            return results
    except Exception as e:
        log.warning(f"    datahub.io: Failed -> {e}")

    log.warning("  FUND_FLOWS: ALL sources failed")
    return results


# Approximate total US equity fund AUM for normalization ($T -> $B)
# Source: ICI Fact Book 2025 — total equity MF+ETF ~$20T domestic equity
# Updated periodically; conservative estimate is fine for % calculation
_EQUITY_AUM_BILLIONS = 20000.0


def _parse_ici_mf_flows(today):
    """Parse ICI MF-only Flows page for equity flows % of assets.

    Looks for pattern: "Equity funds2 had estimated [in|out]flows of $X.XX billion
    (X.X percent of [Month] [Day] assets)"

    KNOWN ISSUE: ICI uses Drupal with massive navigation that can exceed
    scraping buffers. We try two approaches:
    1. Parse HTML and extract only the article/main content area
    2. Use BeautifulSoup's get_text() on the article container

    Also extracts the report week-ending date from:
    "for the week ended Wednesday, [Month] [Day]"
    """
    import re
    from bs4 import BeautifulSoup

    url = "https://www.ici.org/research/stats/flows"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strategy: Extract only the main content area, skip navigation
    # ICI uses <main id="main-content"> or <article> or <div class="node-content">
    content_area = None
    for selector in [
        soup.find("main", id="main-content"),
        soup.find("main"),
        soup.find("article"),
        soup.find("div", class_="node__content"),
        soup.find("div", class_="field--name-body"),
        soup.find("div", class_="content"),
    ]:
        if selector:
            text = selector.get_text(separator=" ", strip=True)
            # Verify this section has the actual data (not just nav)
            if "Equity funds" in text and "percent" in text:
                content_area = text
                log.info(f"    ICI MF-only: Found content area ({len(text)} chars)")
                break

    # Fallback: full page text (may be truncated by nav)
    if content_area is None:
        content_area = soup.get_text(separator=" ", strip=True)
        log.info(f"    ICI MF-only: Using full page text ({len(content_area)} chars)")

    text = content_area

    # Extract week-ending date: "for the week ended Wednesday, February 18"
    # or "for the eight-day period ended Wednesday, January 7"
    date_match = re.search(
        r"(?:week|period)\s+ended\s+\w+,\s+(\w+\s+\d{1,2})",
        text, re.IGNORECASE
    )
    report_date = None
    if date_match:
        date_str = date_match.group(1)
        # Try current year first, then previous year
        for try_year in [today.year, today.year - 1]:
            try:
                report_date = datetime.strptime(f"{date_str} {try_year}", "%B %d %Y").date()
                # Sanity: report date should not be in the future
                if report_date <= today:
                    break
                report_date = None
            except ValueError:
                report_date = None

    if report_date is None:
        log.warning("    ICI MF-only: Could not parse report date")
        report_date = today - timedelta(days=7)  # Conservative fallback

    age_days = (today - report_date).days

    # Extract equity flows with % of assets
    # Pattern: "Equity funds2 had estimated [inflows|outflows] of $X.XX billion
    #           (X.X percent of [Month] [Day] assets)"
    # Also handle: "(less than 0.1 percent of ...)"
    # Note: ICI uses superscript footnote "2" directly after "funds" which becomes
    # "funds2" in get_text(). The \S* handles this.
    equity_pct_match = re.search(
        r"Equity\s+funds\S*\s+had\s+estimated\s+(inflows|outflows)\s+of\s+"
        r"\$([\d.,]+)\s+billion[^(]*\((less\s+than\s+)?([\d.]+)\s+percent\s+of\s+",
        text, re.IGNORECASE
    )

    if equity_pct_match:
        direction = equity_pct_match.group(1).lower()
        less_than = equity_pct_match.group(3) is not None
        pct_value = float(equity_pct_match.group(4))

        # "less than 0.1 percent" -> use 0.05 as midpoint estimate
        if less_than:
            pct_value = pct_value / 2.0

        # Outflows are negative (bearish), inflows are positive (bullish)
        if direction == "outflows":
            pct_value = -pct_value

        # raw_value is in percentage points matching V79 range (-0.20 to +0.20)
        # ICI "0.1 percent" = 0.10 percentage points -> raw_value = -0.10
        raw_value = round(pct_value, 4)

        return {
            "raw_value": raw_value,
            "age_days": age_days,
            "date": report_date.strftime("%Y-%m-%d"),
        }

    # Debug: log what we found to diagnose future regex failures
    equity_mentions = [m.start() for m in re.finditer(r"Equity\s+funds", text, re.IGNORECASE)]
    log.warning(f"    ICI MF-only: Equity % pattern not found. 'Equity funds' found at positions: {equity_mentions[:5]}")
    if equity_mentions:
        # Log a small snippet around first match for debugging
        pos = equity_mentions[0]
        snippet = text[pos:pos+200].replace("\n", " ")
        log.warning(f"    ICI MF-only: Snippet: {snippet[:150]}...")

    return None


def _parse_ici_combined_flows(today):
    """Fallback 1: Parse ICI Combined Flows page for absolute equity $ flows.
    Divide by known AUM to get approximate % of assets.

    Looks for: "Equity funds had estimated [in|out]flows of $X.XX billion"
    """
    import re
    from bs4 import BeautifulSoup

    url = "https://www.ici.org/research/stats/combined_flows"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract main content area (skip massive navigation)
    content_area = None
    for selector in [
        soup.find("main", id="main-content"),
        soup.find("main"),
        soup.find("article"),
        soup.find("div", class_="node__content"),
        soup.find("div", class_="field--name-body"),
        soup.find("div", class_="content"),
    ]:
        if selector:
            text = selector.get_text(separator=" ", strip=True)
            if "Equity funds" in text and "billion" in text:
                content_area = text
                break

    if content_area is None:
        content_area = soup.get_text(separator=" ", strip=True)

    text = content_area

    # Extract week-ending date
    date_match = re.search(
        r"(?:week|period)\s+ended\s+(\w+\s+\d{1,2},\s+\d{4})",
        text, re.IGNORECASE
    )
    report_date = None
    if date_match:
        try:
            report_date = datetime.strptime(date_match.group(1), "%B %d, %Y").date()
        except ValueError:
            report_date = None

    if report_date is None:
        # Try without year (same as MF-only)
        date_match2 = re.search(
            r"(?:week|period)\s+ended\s+\w+,\s+(\w+\s+\d{1,2})",
            text, re.IGNORECASE
        )
        if date_match2:
            date_str = date_match2.group(1)
            for try_year in [today.year, today.year - 1]:
                try:
                    report_date = datetime.strptime(f"{date_str} {try_year}", "%B %d %Y").date()
                    if report_date <= today:
                        break
                    report_date = None
                except ValueError:
                    report_date = None

    if report_date is None:
        report_date = today - timedelta(days=7)

    age_days = (today - report_date).days

    # Extract equity flows: "$X.XX billion"
    # Note: [^(]* allows for whitespace/newlines between "billion" and "("
    equity_match = re.search(
        r"Equity\s+funds\S*\s+had\s+estimated\s+(inflows|outflows)\s+of\s+"
        r"\$([\d.,]+)\s+billion",
        text, re.IGNORECASE
    )

    if equity_match:
        direction = equity_match.group(1).lower()
        amount_str = equity_match.group(2).replace(",", "")
        amount_b = float(amount_str)

        if direction == "outflows":
            amount_b = -amount_b

        # Convert to % of AUM: ($B / $AUM_B) * 100 -> percentage points
        # e.g., $14.64B / $20000B * 100 = 0.073 percentage points
        pct_of_aum = (amount_b / _EQUITY_AUM_BILLIONS) * 100.0
        raw_value = round(pct_of_aum, 4)

        return {
            "raw_value": raw_value,
            "age_days": age_days,
            "date": report_date.strftime("%Y-%m-%d"),
        }

    log.warning("    ICI Combined: Equity flows pattern not found in page text")
    return None


def _parse_datahub_flows(today):
    """Fallback 2: Parse datahub.io CSV for latest equity flows.
    Monthly data, absolute $M, divide by known AUM.
    """
    url = "https://datahub.io/core/investor-flow-of-funds-us/_r/-/data/weekly.csv"
    resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        log.warning("    datahub.io: Empty CSV")
        return None

    # Columns: Date, Total Equity, Domestic Equity, World Equity, ...
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    if df.empty:
        return None

    latest = df.iloc[-1]
    report_date = latest["Date"].date()
    age_days = (today - report_date).days

    # Total Equity flows in $M
    equity_flow_m = float(latest.get("Total Equity", 0))
    # Convert $M to $B, then to percentage points of AUM
    equity_flow_b = equity_flow_m / 1000.0
    pct_of_aum = (equity_flow_b / _EQUITY_AUM_BILLIONS) * 100.0
    raw_value = round(pct_of_aum, 4)

    return {
        "raw_value": raw_value,
        "age_days": age_days,
        "date": report_date.strftime("%Y-%m-%d"),
    }


# ==================== NORMALIZATION ====================

def normalize_cot(indicator_result):
    """Normalize COT indicator: percentile / 10.0 (already calculated in pull)."""
    if indicator_result is None:
        return np.nan
    return indicator_result["score"]


def normalize_linear(name, value):
    """Normalize non-COT indicators using linear scaling. Phase 2-4."""
    if pd.isna(value) or name not in INDICATOR_RANGES:
        return np.nan
    lo, hi, invert = INDICATOR_RANGES[name]
    clipped = max(lo, min(hi, value))
    normalized = (clipped - lo) / (hi - lo)
    if invert:
        normalized = 1.0 - normalized
    return round(normalized * 10.0, 2)


# ==================== COMPOSITE ====================

def calc_composite_score(indicator_scores):
    """Calculate weighted composite score from available indicators.
    Returns (composite, valid_indicators_dict).
    Same pattern as L2: renormalize weights to available indicators.
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


def get_positioning_phase(score):
    """Map score to positioning phase name."""
    if pd.isna(score):
        return "Unknown"
    for threshold, phase in POSITIONING_PHASES:
        if score <= threshold:
            return phase
    return "Extreme Long"


def get_signal(phase, valid_count):
    """Get SIGNAL string from phase. If <4 sources, return Partial."""
    if valid_count < MIN_SOURCES_FOR_COMPOSITE:
        return f"Partial ({valid_count}/6)"
    return PHASE_TO_SIGNAL.get(phase, "Neutral")


def calc_freshness(indicator_data):
    """Calculate weighted freshness score from available indicators.
    indicator_data: dict of name -> {age_days: int}
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
    """Bearish/Extreme signals get 1.3x multiplier, others 1.0x. Capped at 13.0."""
    if pd.isna(score):
        return score
    if signal in ("Bearish", "Extreme"):
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
    """Read L4 regime weight from CONFIG tab. Fallback to 15% (Risk-On default)."""
    try:
        ws = warehouse.worksheet("CONFIG")
        all_data = ws.get_all_values()
        for row in all_data:
            if len(row) >= 2 and "L4" in str(row[0]) and "Positioning" in str(row[0]):
                # CONFIG Layer Weight Matrix: RISK-ON is column B (index 1)
                val = str(row[1]).replace("%", "").strip()
                if val:
                    return f"{val}%"
    except Exception as e:
        log.warning(f"  CONFIG read failed: {e}")
    return "12%"


# ==================== SHEET WRITERS ====================

def write_raw_market_l4(warehouse, cot_results, other_results):
    """Write L4 indicator data to RAW_MARKET tab (R14-R19).
    Only writes rows for indicators that have data.
    Applies V78 corrections to unit and source fields.
    """
    ws = warehouse.worksheet("RAW_MARKET")
    today_str = date.today().strftime("%Y-%m-%d")

    # Build row map: find existing L4 rows by indicator name
    all_data = ws.get_all_values()
    indicator_row_map = {}
    for i, row in enumerate(all_data):
        if len(row) >= 3 and row[2] == "L4":
            indicator_row_map[row[1]] = i + 1

    updates = []

    # COT indicators
    for ind_name, data in cot_results.items():
        if ind_name not in indicator_row_map:
            log.warning(f"  RAW_MARKET: {ind_name} not found in sheet")
            continue
        row_idx = indicator_row_map[ind_name]
        source, tier, unit = SOURCE_MAP.get(ind_name, ("Unknown", "T3", ""))
        raw_str = f"{data['raw_value']:.4f}"
        age_str = str(data["age_days"])
        date_str = data["date"]
        row_data = [date_str, ind_name, "L4", raw_str, "---", "---", age_str, source, tier, unit]
        updates.append((f"A{row_idx}:J{row_idx}", [row_data]))

    # Phase 2-4 indicators (when implemented)
    for ind_name, data in other_results.items():
        if ind_name not in indicator_row_map:
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
        row_data = [date_str, ind_name, "L4", raw_str, "---", "---", age_str, source, tier, unit]
        updates.append((f"A{row_idx}:J{row_idx}", [row_data]))

    for cell_range, values in updates:
        ws.update(values, cell_range, value_input_option="RAW")

    log.info(f"  RAW_MARKET: {len(updates)} L4 rows written")


def write_scores_l4(warehouse, composite, direction, speed, phase, signal, freshness, asym_adj, regime_weight, valid_count, total_count):
    """Write L4 composite score to SCORES tab Row 5 (5/13 columns ab Tag 1)."""
    ws = warehouse.worksheet("SCORES")

    row_data = [
        "L4 Positioning",
        str(composite) if not pd.isna(composite) else "---",
        "---",              # SCORE_7D — needs history
        "---",              # SCORE_30D — needs history
        "n/a",              # PERCENTILE — needs 252d history
        direction,          # DIRECTION
        speed,              # SPEED
        signal,             # SIGNAL
        f"{freshness:.0f}", # FRESHNESS
        "---",              # DECAY_ADJ — needs halflife config
        str(asym_adj) if not pd.isna(asym_adj) else "---",  # ASYMMETRY_ADJ
        regime_weight,      # REGIME_WEIGHT
        "---",              # HISTORICAL_ANALOG — needs years of history
    ]
    ws.update([row_data], "A5:M5", value_input_option="RAW")
    log.info(f"  SCORES: {composite} ({signal}) | {phase} | {valid_count}/{total_count}")


def write_dashboard_l4(warehouse, composite, direction, signal, freshness, speed):
    """Write L4 summary to DASHBOARD tab Row 20."""
    ws = warehouse.worksheet("DASHBOARD")
    row_data = [
        "L4 Positioning",
        str(composite) if not pd.isna(composite) else "---",
        direction,
        signal,
        f"{freshness:.0f}",
        "n/a",
        speed,
    ]
    ws.update([row_data], "A20:G20", value_input_option="RAW")
    log.info(f"  DASHBOARD: {composite}, {signal}")


# ==================== MAIN ====================

def main():
    log.info("=" * 60)
    log.info("L4 Positioning Collector - Starting (6/6 Indicators)")
    log.info("=" * 60)

    log.info("Connecting to Data Warehouse...")
    warehouse = connect_warehouse()
    log.info("  OK")

    # --- PULL PHASE ---
    log.info("--- PULL PHASE ---")

    log.info("Pulling CFTC COT data...")
    cot_results = pull_cot_data()

    log.info("Pulling Crypto Funding Rate...")
    crypto_results = pull_crypto_funding()

    log.info("Pulling Options GEX...")
    gex_results = pull_options_gex()

    log.info("Pulling Fund Flows...")
    flows_results = pull_fund_flows()

    # --- MERGE PHASE ---
    log.info("--- MERGE PHASE ---")

    # Collect all indicator scores and metadata
    indicator_scores = {}
    indicator_data = {}  # For freshness calculation

    # COT scores (already normalized via percentile)
    for ind_name in COT_CONTRACTS:
        if ind_name in cot_results:
            data = cot_results[ind_name]
            indicator_scores[ind_name] = data["score"]
            indicator_data[ind_name] = {"age_days": data["age_days"]}
            log.info(f"  {ind_name}: raw={data['raw_value']:.2f}%, pctl={data['percentile']:.1f}, score={data['score']:.2f}")
        else:
            indicator_scores[ind_name] = np.nan
            log.warning(f"  {ind_name}: NO DATA")

    # Phase 2-4 stubs — will be populated when implemented
    for ind_name, pull_results in [
        ("CRYPTO_FUNDING_RATE", crypto_results),
        ("OPTIONS_GEX", gex_results),
        ("FUND_FLOWS_EQUITY", flows_results),
    ]:
        if ind_name in pull_results:
            data = pull_results[ind_name]
            indicator_scores[ind_name] = normalize_linear(ind_name, data["raw_value"])
            indicator_data[ind_name] = {"age_days": data.get("age_days", 0)}
        else:
            indicator_scores[ind_name] = np.nan

    # --- COMPOSITE PHASE ---
    log.info("--- COMPOSITE PHASE ---")

    composite, valid_indicators = calc_composite_score(indicator_scores)
    valid_count = len(valid_indicators)
    total_count = len(INDICATOR_WEIGHTS)
    phase = get_positioning_phase(composite)
    signal = get_signal(phase, valid_count)
    freshness = calc_freshness(indicator_data)

    # Direction and speed from previous SCORES value
    direction = speed = "n/a"
    try:
        prev_row = warehouse.worksheet("SCORES").row_values(5)
        if len(prev_row) >= 2 and prev_row[1] not in ("---", "", "—", None):
            prev_score = float(prev_row[1])
            direction = get_direction(composite, prev_score)
            speed = get_speed(composite, prev_score)
            log.info(f"  Previous: {prev_score} -> {direction}, {speed}")
    except Exception:
        pass

    # Asymmetry adjustment
    asym_signal = PHASE_TO_SIGNAL.get(phase, "Neutral") if valid_count >= MIN_SOURCES_FOR_COMPOSITE else "Neutral"
    asym_adj = calc_asymmetry_adj(composite, asym_signal)

    # Regime weight from CONFIG
    regime_weight = get_regime_weight(warehouse)

    log.info(f"  Composite: {composite}/10 | {phase} | {signal}")
    log.info(f"  Sources: {valid_count}/{total_count} | Freshness: {freshness}/10")
    log.info(f"  Asymmetry: {asym_adj} | Regime Weight: {regime_weight}")

    # --- WRITE PHASE ---
    log.info("--- WRITE PHASE ---")

    write_raw_market_l4(warehouse, cot_results, {
        **crypto_results, **gex_results, **flows_results
    })
    write_scores_l4(warehouse, composite, direction, speed, phase, signal,
                    freshness, asym_adj, regime_weight, valid_count, total_count)
    write_dashboard_l4(warehouse, composite, direction, signal, freshness, speed)

    # --- SUMMARY ---
    log.info("=" * 60)
    log.info(f"COMPLETE: {composite}/10 ({phase}) | {signal} | {direction}/{speed} | {valid_count}/{total_count}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
