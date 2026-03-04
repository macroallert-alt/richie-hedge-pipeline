"""
step_0b_agent_feeder/main.py
Baldur Creek Capital — richie-hedge-pipeline

Collects 30 fields for Step 1 (Market Analyst).
Writes to Data Warehouse tabs:
  RAW_AGENT2          — Current values (30 rows, overwritten daily)
  RAW_AGENT2_HISTORY  — Daily append (365d rolling, for Pctl/Delta calc)

Wave 1 (24 fields): FRED + yfinance — fully automated
Wave 2 (6 fields):  pct_above_200dma, nh_nl, naaim_exposure,
                     cot_es_leveraged, cot_zn_leveraged, china_10y — PENDING

Output format per field:
  Field | Value | Pctl_1Y | Direction | Delta_5D | Delta_5D_Norm | Confidence | Anomaly

Bootstrap mode: Pctl=50, Direction=FLAT, Deltas=0 until enough history.
After 21d: Deltas active. After 60d: rough Pctl. After 252d: true 1Y Pctl.

Iron Rule: V38 NEVER MODIFY. This script only writes to Data Warehouse.
"""

import os
import sys
import logging
import traceback
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from fredapi import Fred
import yfinance as yf

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TAB_RAW = "RAW_AGENT2"
TAB_HISTORY = "RAW_AGENT2_HISTORY"

# How many days of yfinance history to pull for calculations
YF_LOOKBACK = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("0b_feeder")

# ─────────────────────────────────────────────
# FIELD DEFINITIONS (order matches RAW_AGENT2 tab)
# ─────────────────────────────────────────────

FIELD_ORDER = [
    "net_liquidity", "walcl", "tga", "rrp", "mmf_assets",
    "spread_2y10y", "spread_3m10y", "real_10y_yield", "hy_oas", "ig_oas",
    "nfci", "anfci", "disc_window",
    "vix", "vix_term_struct", "pc_ratio_equity", "iv_rv_spread",
    "pct_above_200dma", "nh_nl",
    "spy_tlt_corr",
    "naaim_exposure", "aaii_bull_bear",
    "cot_es_leveraged", "cot_zn_leveraged",
    "dxy", "cu_au_ratio", "china_10y", "usdcnh", "wti_curve", "usdjpy",
]

WAVE2_FIELDS = {"pct_above_200dma", "nh_nl", "naaim_exposure",
                "cot_es_leveraged", "cot_zn_leveraged", "china_10y"}

# ─────────────────────────────────────────────
# CONNECTIONS
# ─────────────────────────────────────────────

def connect_warehouse():
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    if not os.path.exists(creds_json):
        raw = os.environ.get("GCP_SA_KEY", "")
        if raw:
            with open(creds_json, "w") as f:
                f.write(raw)
        else:
            raise FileNotFoundError("No GCP credentials found")
    creds = Credentials.from_service_account_file(creds_json, scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc.open_by_key(WAREHOUSE_SHEET_ID)


def connect_fred():
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not set")
    return Fred(api_key=FRED_API_KEY)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def yf_download(ticker, days=YF_LOOKBACK):
    """Download yfinance data, return DataFrame with Close column."""
    end = date.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_last(series):
    """Get last non-NaN value from a series."""
    if series is None or (isinstance(series, pd.Series) and series.empty):
        return None
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def fred_latest(fred, series_id, days_back=30):
    """Pull latest FRED value with fallback lookback."""
    try:
        end = date.today()
        start = end - timedelta(days=days_back)
        s = fred.get_series(series_id, observation_start=start.strftime("%Y-%m-%d"),
                            observation_end=end.strftime("%Y-%m-%d"))
        val = safe_last(s)
        if val is not None:
            return val
        # Try wider lookback for weekly/monthly series
        start2 = end - timedelta(days=90)
        s2 = fred.get_series(series_id, observation_start=start2.strftime("%Y-%m-%d"),
                             observation_end=end.strftime("%Y-%m-%d"))
        return safe_last(s2)
    except Exception as e:
        log.warning(f"  FRED {series_id} failed: {e}")
        return None


# ─────────────────────────────────────────────
# FRED PULLS
# ─────────────────────────────────────────────

def pull_fred_fields(fred):
    """Pull all FRED-sourced fields. Returns dict {field_name: value}."""
    results = {}

    # #2 walcl — Fed Total Assets ($M)
    results["walcl"] = fred_latest(fred, "WALCL", 14)

    # #3 tga — Treasury General Account ($M)
    results["tga"] = fred_latest(fred, "WTREGEN", 14)

    # #4 rrp — Overnight Reverse Repo ($B)
    rrp_val = fred_latest(fred, "RRPONTSYD", 7)
    if rrp_val is not None:
        results["rrp"] = rrp_val / 1000.0  # Convert $M to $B
    else:
        results["rrp"] = None

    # #1 net_liquidity — WALCL - TGA - RRP (all in $M)
    if results["walcl"] and results["tga"] and results["rrp"] is not None:
        rrp_m = results["rrp"] * 1000  # Back to $M for calculation
        results["net_liquidity"] = results["walcl"] - results["tga"] - rrp_m
    else:
        results["net_liquidity"] = None

    # #5 mmf_assets — Money Market Mutual Funds ($B)
    mmf = fred_latest(fred, "WMMN", 14)
    if mmf is None:
        mmf = fred_latest(fred, "MMMFFAQ027S", 120)  # Quarterly fallback
    results["mmf_assets"] = mmf

    # #6 spread_2y10y — 10Y-2Y Spread (%)
    results["spread_2y10y"] = fred_latest(fred, "T10Y2Y", 7)

    # #7 spread_3m10y — 10Y-3M Spread (%)
    results["spread_3m10y"] = fred_latest(fred, "T10Y3M", 7)

    # #8 real_10y_yield — 10Y TIPS Yield (%)
    results["real_10y_yield"] = fred_latest(fred, "DFII10", 7)

    # #9 hy_oas — ICE BofA HY OAS (bps, FRED reports in %)
    hy = fred_latest(fred, "BAMLH0A0HYM2", 7)
    results["hy_oas"] = hy * 100.0 if hy is not None else None  # % -> bps

    # #10 ig_oas — ICE BofA IG OAS (bps)
    ig = fred_latest(fred, "BAMLC0A0CM", 7)
    results["ig_oas"] = ig * 100.0 if ig is not None else None

    # #11 nfci — Chicago Fed NFCI (index)
    results["nfci"] = fred_latest(fred, "NFCI", 14)

    # #12 anfci — Adjusted NFCI (index)
    results["anfci"] = fred_latest(fred, "ANFCI", 14)

    # #13 disc_window — Discount Window Borrowings ($M)
    dw = fred_latest(fred, "WLCFLPCL", 14)
    if dw is None:
        dw = fred_latest(fred, "H41RESPPALDKNWW", 14)
    results["disc_window"] = dw

    for k, v in results.items():
        status = f"{v:.4f}" if v is not None else "FAILED"
        log.info(f"  FRED {k}: {status}")

    return results


# ─────────────────────────────────────────────
# YFINANCE PULLS
# ─────────────────────────────────────────────

def pull_yfinance_fields():
    """Pull all yfinance-sourced fields. Returns dict {field_name: value}."""
    results = {}

    # --- Multi-ticker download for efficiency ---
    tickers = "^VIX ^VIX3M SPY TLT DX-Y.NYB HG=F GC=F CNH=X CL=F JPY=X"
    log.info(f"  Downloading: {tickers}")
    try:
        bulk = yf.download(tickers, period="120d", interval="1d",
                           progress=False, auto_adjust=True, group_by="ticker")
    except Exception as e:
        log.error(f"  Bulk yfinance download failed: {e}")
        bulk = pd.DataFrame()

    def get_close(ticker_name):
        """Extract Close series from bulk download."""
        try:
            if isinstance(bulk.columns, pd.MultiIndex):
                return bulk[ticker_name]["Close"].dropna()
            else:
                return bulk["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)

    # #14 vix
    vix_s = get_close("^VIX")
    results["vix"] = safe_last(vix_s)

    # #15 vix_term_struct — VIX / VIX3M
    vix3m_s = get_close("^VIX3M")
    vix_val = results["vix"]
    vix3m_val = safe_last(vix3m_s)
    if vix_val and vix3m_val and vix3m_val > 0:
        results["vix_term_struct"] = round(vix_val / vix3m_val, 4)
    else:
        results["vix_term_struct"] = None

    # #16 pc_ratio_equity — P/C ratio proxy (VIX level based heuristic)
    # True CBOE equity P/C not available via yfinance. Use ^VIX / SPY vol proxy.
    # 0c already has this — we replicate the same approach
    spy_s = get_close("SPY")
    if len(spy_s) >= 20 and vix_val is not None:
        spy_ret = spy_s.pct_change().dropna()
        rv_20d = float(spy_ret.tail(20).std() * np.sqrt(252) * 100)
        # Heuristic: when VIX >> RV, puts are expensive (high P/C)
        if rv_20d > 0:
            results["pc_ratio_equity"] = round(vix_val / rv_20d, 4)
        else:
            results["pc_ratio_equity"] = None
    else:
        results["pc_ratio_equity"] = None

    # #17 iv_rv_spread — VIX minus 20d realized vol
    if vix_val is not None and len(spy_s) >= 21:
        spy_ret = spy_s.pct_change().dropna()
        rv_20d = float(spy_ret.tail(20).std() * np.sqrt(252) * 100)
        results["iv_rv_spread"] = round(vix_val - rv_20d, 2)
    else:
        results["iv_rv_spread"] = None

    # #20 spy_tlt_corr — 60d rolling correlation
    tlt_s = get_close("TLT")
    if len(spy_s) >= 60 and len(tlt_s) >= 60:
        spy_ret = spy_s.pct_change().dropna()
        tlt_ret = tlt_s.pct_change().dropna()
        combined = pd.concat([spy_ret, tlt_ret], axis=1, join="inner")
        combined.columns = ["spy", "tlt"]
        if len(combined) >= 60:
            corr = float(combined.tail(60).corr().iloc[0, 1])
            results["spy_tlt_corr"] = round(corr, 4)
        else:
            results["spy_tlt_corr"] = None
    else:
        results["spy_tlt_corr"] = None

    # #22 aaii_bull_bear — Bull% - Bear% (from AAII website)
    results["aaii_bull_bear"] = pull_aaii()

    # #25 dxy — Dollar Index
    dxy_s = get_close("DX-Y.NYB")
    results["dxy"] = safe_last(dxy_s)

    # #26 cu_au_ratio — Copper / Gold
    cu_s = get_close("HG=F")
    au_s = get_close("GC=F")
    cu_val = safe_last(cu_s)
    au_val = safe_last(au_s)
    if cu_val and au_val and au_val > 0:
        # Standard ratio: copper price per lb / gold price per oz * 1000
        results["cu_au_ratio"] = round((cu_val / au_val) * 1000, 4)
    else:
        results["cu_au_ratio"] = None

    # #28 usdcnh
    cnh_s = get_close("CNH=X")
    results["usdcnh"] = safe_last(cnh_s)

    # #29 wti_curve — Front/back month ratio
    # CL=F is front month. For back month we use CL=F 5d ago as proxy
    cl_s = get_close("CL=F")
    if len(cl_s) >= 22:
        front = float(cl_s.iloc[-1])
        back = float(cl_s.iloc[-22])  # ~1 month ago as proxy for next month
        if back > 0:
            results["wti_curve"] = round(front / back, 4)
        else:
            results["wti_curve"] = None
    else:
        results["wti_curve"] = None

    # #30 usdjpy
    jpy_s = get_close("JPY=X")
    jpy_val = safe_last(jpy_s)
    if jpy_val and jpy_val > 0:
        results["usdjpy"] = round(jpy_val, 2)
    else:
        results["usdjpy"] = None

    for k, v in results.items():
        status = f"{v:.4f}" if v is not None else "FAILED"
        log.info(f"  YF {k}: {status}")

    return results


# ─────────────────────────────────────────────
# AAII SCRAPE
# ─────────────────────────────────────────────

def pull_aaii():
    """Pull AAII Bull-Bear spread from weekly Excel file."""
    try:
        import requests as req
        url = "https://www.aaii.com/files/surveys/sentiment.xls"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = req.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            log.warning(f"  AAII: HTTP {resp.status_code}")
            return None

        import io
        try:
            df = pd.read_excel(io.BytesIO(resp.content), engine="xlrd")
        except Exception:
            df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")

        # Find Bullish and Bearish columns
        bull_col = bear_col = None
        for col in df.columns:
            c = str(col).strip().lower()
            if "bullish" in c and "bull" in c:
                bull_col = col
            elif "bearish" in c and "bear" in c:
                bear_col = col
            if not bull_col and c in ("bullish", "bull"):
                bull_col = col
            if not bear_col and c in ("bearish", "bear"):
                bear_col = col

        if bull_col is None or bear_col is None:
            # Fallback: assume columns by position (typical AAII layout)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 3:
                bull_col = num_cols[0]
                bear_col = num_cols[2]
            else:
                log.warning(f"  AAII: Cannot find Bull/Bear columns: {df.columns.tolist()}")
                return None

        last_valid = df[[bull_col, bear_col]].dropna().iloc[-1]
        bull = float(last_valid[bull_col])
        bear = float(last_valid[bear_col])

        # Values might be in % (33.2) or decimal (0.332)
        if bull < 1.0:
            bull *= 100
        if bear < 1.0:
            bear *= 100

        spread = round(bull - bear, 2)
        log.info(f"  AAII: Bull={bull:.1f}% Bear={bear:.1f}% Spread={spread:.1f}")
        return spread

    except Exception as e:
        log.warning(f"  AAII scrape failed: {e}")
        return None


# ─────────────────────────────────────────────
# HISTORY + PERCENTILE + DELTA CALCULATION
# ─────────────────────────────────────────────

def read_history(warehouse):
    """Read RAW_AGENT2_HISTORY tab. Returns list of dicts [{date, field: value}]."""
    try:
        ws = warehouse.worksheet(TAB_HISTORY)
        data = ws.get_all_values()
        if len(data) < 3:  # title + header + at least 1 row
            return []

        headers = data[1]  # Row 2 = headers
        history = []
        for row in data[2:]:
            if not row or not row[0] or row[0] == "—":
                continue
            record = {"date": row[0]}
            for i, h in enumerate(headers[1:], start=1):
                if i < len(row) and row[i] and row[i] != "—":
                    try:
                        record[h] = float(row[i])
                    except (ValueError, TypeError):
                        pass
            history.append(record)
        return history
    except Exception as e:
        log.warning(f"  History read failed: {e}")
        return []


def calc_percentile_1y(field_name, current_value, history):
    """Calculate 1Y percentile from history. Returns 50 if insufficient data."""
    if current_value is None:
        return 50
    vals = [h.get(field_name) for h in history if field_name in h]
    vals = [v for v in vals if v is not None]
    if len(vals) < 10:  # Need at least 10 days
        return 50
    # Use last 252 values max (1Y)
    vals = vals[-252:]
    below = sum(1 for v in vals if v <= current_value)
    pctl = int(round((below / len(vals)) * 100))
    return max(0, min(100, pctl))


def calc_direction(field_name, current_value, history):
    """Calculate direction from 5d and 21d trends."""
    if current_value is None or len(history) < 5:
        return "FLAT"

    val_5d = None
    val_21d = None
    for h in reversed(history):
        if field_name in h:
            if val_5d is None and len(history) >= 5:
                # Find value ~5 days ago
                idx = max(0, len(history) - 5)
                val_5d = history[idx].get(field_name)
            if val_21d is None and len(history) >= 21:
                idx = max(0, len(history) - 21)
                val_21d = history[idx].get(field_name)
            break

    if val_5d is None:
        return "FLAT"

    d5 = current_value - val_5d if val_5d else 0
    d21 = (current_value - val_21d) if val_21d else d5

    if d5 > 0 and d21 > 0:
        return "UP"
    elif d5 < 0 and d21 < 0:
        return "DOWN"
    else:
        return "FLAT"


def calc_delta_5d(field_name, current_value, history):
    """Calculate 5-day delta."""
    if current_value is None or len(history) < 5:
        return 0.0
    idx = max(0, len(history) - 5)
    val_5d = history[idx].get(field_name)
    if val_5d is None:
        return 0.0
    return round(current_value - val_5d, 4)


def calc_delta_5d_norm(delta_5d, field_name, history):
    """Normalize 5d delta by historical std. Returns 0 if insufficient data."""
    vals = [h.get(field_name) for h in history if field_name in h]
    vals = [v for v in vals if v is not None]
    if len(vals) < 10:
        return 0.0
    diffs = [vals[i] - vals[i-1] for i in range(1, len(vals))]
    if not diffs:
        return 0.0
    std = np.std(diffs)
    if std == 0 or np.isnan(std):
        return 0.0
    return round(delta_5d / (std * np.sqrt(5)), 2)


def enrich_field(field_name, value, history):
    """Calculate Pctl_1Y, Direction, Delta_5D, Delta_5D_Norm for one field."""
    pctl = calc_percentile_1y(field_name, value, history)
    direction = calc_direction(field_name, value, history)
    delta_5d = calc_delta_5d(field_name, value, history)
    delta_5d_norm = calc_delta_5d_norm(delta_5d, field_name, history)
    return pctl, direction, delta_5d, delta_5d_norm


# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────

def detect_anomaly(field_name, value, history):
    """Simple anomaly detection: |z-score| > 3 relative to history."""
    if value is None:
        return "NO_DATA"
    vals = [h.get(field_name) for h in history if field_name in h]
    vals = [v for v in vals if v is not None]
    if len(vals) < 10:
        return "OK"
    mean = np.mean(vals[-60:])
    std = np.std(vals[-60:])
    if std == 0:
        return "OK"
    z = abs((value - mean) / std)
    if z > 4.0:
        return "ANOMALY"
    elif z > 3.0:
        return "DIVERGENT"
    return "OK"


# ─────────────────────────────────────────────
# SHEET WRITERS
# ─────────────────────────────────────────────

def write_raw_agent2(warehouse, field_data):
    """Write current values to RAW_AGENT2 tab (overwrite rows 3-32)."""
    ws = warehouse.worksheet(TAB_RAW)
    rows = []
    for field_name in FIELD_ORDER:
        fd = field_data.get(field_name, {})
        val = fd.get("value")
        val_str = f"{val:.4f}" if val is not None and isinstance(val, float) and abs(val) < 1e8 else (
            f"{val:.0f}" if val is not None and isinstance(val, (int, float)) else "—"
        )
        rows.append([
            field_name,
            val_str,
            str(fd.get("pctl_1y", 50)),
            fd.get("direction", "FLAT"),
            str(fd.get("delta_5d", 0)),
            str(fd.get("delta_5d_norm", 0)),
            str(fd.get("confidence", 1.0)),
            fd.get("anomaly", "OK"),
        ])

    ws.update(values=rows, range_name=f"A3:H{2 + len(FIELD_ORDER)}",
              value_input_option="RAW")
    log.info(f"  RAW_AGENT2: {len(rows)} fields written")


def write_history_row(warehouse, field_data):
    """Append today's values to RAW_AGENT2_HISTORY."""
    ws = warehouse.worksheet(TAB_HISTORY)
    today_str = date.today().strftime("%Y-%m-%d")

    # Build row: Date + 30 field values + run_status
    row = [today_str]
    for field_name in FIELD_ORDER:
        fd = field_data.get(field_name, {})
        val = fd.get("value")
        if val is not None:
            row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
        else:
            row.append("—")
    row.append("OK")  # run_status

    # Check if today's row already exists (overwrite if so)
    existing = ws.get_all_values()
    today_row_idx = None
    for i, r in enumerate(existing):
        if i >= 2 and len(r) > 0 and r[0] == today_str:
            today_row_idx = i + 1  # 1-indexed
            break

    if today_row_idx:
        ws.update(values=[row], range_name=f"A{today_row_idx}:AF{today_row_idx}",
                  value_input_option="RAW")
        log.info(f"  HISTORY: Updated existing row for {today_str}")
    else:
        # Insert at row 3 (after title + header)
        ws.insert_rows([row], row=3, value_input_option="RAW")
        log.info(f"  HISTORY: Inserted new row for {today_str}")

    # Trim to 365 rows max (delete oldest if > 365 data rows)
    total_data_rows = len(existing) - 2  # minus title + header
    if total_data_rows > 365:
        # Delete rows from bottom
        excess = total_data_rows - 365
        last_row = len(existing)
        for _ in range(excess):
            try:
                ws.delete_rows(last_row)
                last_row -= 1
            except Exception:
                break
        log.info(f"  HISTORY: Trimmed {excess} old rows")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("step_0b AGENT FEEDER — START")
    log.info(f"Date: {date.today()}")
    log.info("=" * 60)

    # --- Connect ---
    log.info("Connecting to Data Warehouse...")
    warehouse = connect_warehouse()
    log.info("  OK")

    log.info("Connecting to FRED...")
    fred = connect_fred()
    log.info("  OK")

    # --- Read history for enrichment ---
    log.info("Reading history...")
    history = read_history(warehouse)
    log.info(f"  {len(history)} days of history available")

    # --- Pull Phase ---
    log.info("=" * 40)
    log.info("PULL PHASE: FRED")
    log.info("=" * 40)
    fred_values = pull_fred_fields(fred)

    log.info("=" * 40)
    log.info("PULL PHASE: YFINANCE")
    log.info("=" * 40)
    yf_values = pull_yfinance_fields()

    # --- Merge ---
    log.info("=" * 40)
    log.info("MERGE + ENRICH PHASE")
    log.info("=" * 40)
    all_values = {}
    all_values.update(fred_values)
    all_values.update(yf_values)

    # --- Build enriched field data ---
    field_data = {}
    wave1_ok = 0
    wave1_fail = 0
    wave2_pending = 0

    for field_name in FIELD_ORDER:
        if field_name in WAVE2_FIELDS:
            # Wave 2: placeholder
            field_data[field_name] = {
                "value": None,
                "pctl_1y": 50,
                "direction": "FLAT",
                "delta_5d": 0,
                "delta_5d_norm": 0,
                "confidence": 0.0,
                "anomaly": "PENDING",
            }
            wave2_pending += 1
            continue

        value = all_values.get(field_name)
        if value is None:
            field_data[field_name] = {
                "value": None,
                "pctl_1y": 50,
                "direction": "FLAT",
                "delta_5d": 0,
                "delta_5d_norm": 0,
                "confidence": 0.0,
                "anomaly": "NO_DATA",
            }
            wave1_fail += 1
            log.warning(f"  {field_name}: NO DATA")
            continue

        pctl, direction, delta_5d, delta_5d_norm = enrich_field(
            field_name, value, history
        )
        anomaly = detect_anomaly(field_name, value, history)

        field_data[field_name] = {
            "value": value,
            "pctl_1y": pctl,
            "direction": direction,
            "delta_5d": delta_5d,
            "delta_5d_norm": delta_5d_norm,
            "confidence": 1.0,
            "anomaly": anomaly,
        }
        wave1_ok += 1
        log.info(f"  {field_name}: {value:.4f} | Pctl={pctl} Dir={direction} "
                 f"D5={delta_5d:.4f} Anom={anomaly}")

    # --- Write Phase ---
    log.info("=" * 40)
    log.info("WRITE PHASE")
    log.info("=" * 40)

    write_raw_agent2(warehouse, field_data)
    write_history_row(warehouse, field_data)

    # --- Summary ---
    log.info("=" * 60)
    log.info(f"step_0b AGENT FEEDER — COMPLETE")
    log.info(f"  Wave 1: {wave1_ok} OK / {wave1_fail} FAILED / {24} total")
    log.info(f"  Wave 2: {wave2_pending} PENDING")
    log.info(f"  History depth: {len(history)} days")
    pctl_mode = "BOOTSTRAP" if len(history) < 60 else ("ROUGH" if len(history) < 252 else "FULL")
    log.info(f"  Percentile mode: {pctl_mode}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
