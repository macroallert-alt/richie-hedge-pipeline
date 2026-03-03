"""
L4 Positioning Collector - step_0d_l4_positioning/main.py
Global Macro RV System - Data Warehouse Layer 4
Phase 1: COT x 3 (S&P 500, Gold, 10Y Treasury)
Pulls positioning indicators, normalizes to 0-10, writes to:
  RAW_MARKET (R14-R19), SCORES (R5), DASHBOARD (R20)
Sources Phase 1: CFTC Legacy Futures (COT x3)
Sources Phase 2-4: Binance (Crypto), EODHD (GEX), ICI (Fund Flows) — stubs
Pattern: Identical to L2 Sentiment Collector (Pull -> Normalize -> Composite -> Sheets)
Reference: V78 Statusanalyse Kap.10, Kap.18, Kap.19
Iron Rule: V16 main.py NEVER TOUCH. L2 step_0c NEVER TOUCH.
"""

import io
import os
import sys
import logging
import zipfile
from datetime import datetime, timedelta, date

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
    "CRYPTO_FUNDING_RATE":  ("Binance", "T2", "pct"),
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
            # Exclude Pct_of_Open_Interest columns — we need absolute OI
            if "pct" not in cl and "percent" not in cl and "other" not in cl:
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


# ==================== PHASE 2-4 STUBS ====================

def pull_crypto_funding():
    """STUB — Phase 2: Binance BTCUSDT + ETHUSDT funding rate."""
    log.info("  CRYPTO_FUNDING: [STUB - Phase 2] Skipping")
    return {}


def pull_options_gex():
    """STUB — Phase 3: EODHD SPY Options GEX calculation."""
    log.info("  OPTIONS_GEX: [STUB - Phase 3] Skipping")
    return {}


def pull_fund_flows():
    """STUB — Phase 4: ICI Combined Fund Flows scraping."""
    log.info("  FUND_FLOWS: [STUB - Phase 4] Skipping")
    return {}


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
    log.info("L4 Positioning Collector - Starting (Phase 1: COT x 3)")
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
