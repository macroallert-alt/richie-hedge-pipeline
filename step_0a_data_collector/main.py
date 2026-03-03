"""
V55 Data Collector — step_0a_data_collector/main.py
Global Macro RV System — Upstream Signal Calculator

Berechnet alle 4 Tabs:
  Stufe 1: DATA_Liquidity  (FRED → Raw → Indizes → GLP → MAs → Trend/Mom/Acc)
  Stufe 2: CYCLES_Howell   (Phase Forward-Fill, Sine, MST, 26 Spalten)
  Stufe 3: DATA_K16_K17    (Cu/Au, Credit Impulse, 4 Votes, Veto, 5d Confirm)
  Stufe 4: CALC_Macro_State (Growth, Stress, State Map 1-12)

V38 liest daraus: Macro_State_Num, Growth_Signal, Stress_Score,
                  Liq_Dir_Confirmed, Vote_Sum_Magnitude

Referenz: V75 Systemstatusanalyse (Single Source of Truth)
Iron Rule: V38 NEVER MODIFY. Historische Werte = Ground Truth.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from fredapi import Fred

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
PRODUCTION_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("V55")

# ─────────────────────────────────────────────
# CONSTANTS from V75 Spec (NEVER CHANGE)
# ─────────────────────────────────────────────

# DATA_Liquidity Index Bases
BASES = {
    "Fed": 5_730_394,
    "ECB": 6_388_921.20,
    "BOJ": 5_916_723_196_439,
    "M2": 17_928.40,
}

# GLP Weights (Least-Squares Regression, RMSE=0.000000)
GLP_WEIGHTS = {
    "Fed": 0.090529,
    "ECB": 0.100933,
    "BOJ": 0.093473,
    "China": 0.431831,  # China dominiert mit 43.2%!
    "M2": 0.283234,
}

# China M2 Forward-Fill Value (from Sheet, 2026-02-20)
CHINA_M2_USD_LAST = 49_261_001_737_116

# ─────────────────────────────────────────────
# GOOGLE SHEETS CONNECTION
# ─────────────────────────────────────────────

def connect_sheets():
    """Connect to Google Sheets via Service Account."""
    creds_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json"
    )
    creds = Credentials.from_service_account_file(
        creds_path,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(PRODUCTION_SHEET_ID)

# ─────────────────────────────────────────────
# SHEET READER — Load historical context
# ─────────────────────────────────────────────

def read_sheet_tab(sheet, tab_name: str) -> pd.DataFrame:
    """Read a sheet tab into a DataFrame. Row 0 = header names, Row 1 = sub-header (skip)."""
    ws = sheet.worksheet(tab_name)
    data = ws.get_all_values()
    if len(data) < 3:
        return pd.DataFrame()

    headers = data[0]
    # Data starts at row index 2 (skip sub-header at index 1)
    rows = data[2:]
    df = pd.DataFrame(rows, columns=headers)

    # Parse Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def find_last_good_date(sheet) -> date:
    """Find last date where Detail-Strings are filled in CALC_Macro_State."""
    ws = sheet.worksheet("CALC_Macro_State")
    # Read first 100 data rows (newest first in sheet, row 1=header, row 2=subheader)
    data = ws.get("A3:C102")
    for row in data:
        if len(row) >= 3 and row[2]:  # Growth_Detail is filled
            return pd.to_datetime(row[0]).date()
    # Fallback: return a safe old date
    return date(2026, 2, 3)


# ─────────────────────────────────────────────
# FRED DATA PULL
# ─────────────────────────────────────────────

def pull_fred_series(fred: Fred, series_id: str, start: str, end: str) -> pd.Series:
    """Pull a single FRED series, forward-fill to daily."""
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    if raw is None or raw.empty:
        log.warning(f"FRED series {series_id} returned empty!")
        return pd.Series(dtype=float)
    # Resample to daily calendar and forward-fill
    daily = raw.resample("D").ffill()
    return daily


def pull_all_fred(fred: Fred, start: str, end: str) -> dict:
    """Pull all 17 FRED series needed by the Data Collector."""

    series_map = {
        # DATA_Liquidity (8 series)
        "WALCL": "WALCL",
        "WTREGEN": "WTREGEN",
        "RRPONTSYD": "RRPONTSYD",
        "ECBASSETSW": "ECBASSETSW",
        "DEXUSEU": "DEXUSEU",
        "JPNASSETS": "JPNASSETS",
        "DEXJPUS": "DEXJPUS",
        "WM2NS": "WM2NS",
        # DATA_K16_K17 (1 series)
        "TOTBKCR": "TOTBKCR",
        # CALC_Macro_State Growth (4 series)
        "PERMIT": "PERMIT",
        "UMCSENT": "UMCSENT",
        "ICSA": "ICSA",
        "INDPRO": "INDPRO",
        # CALC_Macro_State Stress (3 series)
        "VIXCLS": "VIXCLS",
        "BAMLH0A0HYM2": "BAMLH0A0HYM2",
        "NFCI": "NFCI",
        # FX for China M2
        "DEXCHUS": "DEXCHUS",
    }

    result = {}
    for key, fred_id in series_map.items():
        log.info(f"  Pulling {fred_id}...")
        try:
            result[key] = pull_fred_series(fred, fred_id, start, end)
        except Exception as e:
            log.error(f"  FAILED: {fred_id} → {e}")
            result[key] = pd.Series(dtype=float)

    return result


# ─────────────────────────────────────────────
# STUFE 1: DATA_Liquidity
# ─────────────────────────────────────────────

def calc_data_liquidity(fred_data: dict, hist_df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Calculate DATA_Liquidity for new dates.

    Uses historical GLP values from hist_df for rolling MA continuity.
    Calculates: Raw (B-F) → Indices (G-K) → GLP (L) → MAs (M-N) → Trend/Mom/Acc (O-Q)

    Returns DataFrame with columns matching the Sheet tab.
    """
    # ── Build date range for new days ──
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # ── Raw Values (Spalten B-F) ──
    # Fed_Net_Liq = WALCL - WTREGEN - RRPONTSYD*1000
    # RRPONTSYD is in Billions, rest in Millions!
    walcl = fred_data["WALCL"].reindex(dates, method="ffill")
    wtregen = fred_data["WTREGEN"].reindex(dates, method="ffill")
    rrpontsyd = fred_data["RRPONTSYD"].reindex(dates, method="ffill")
    fed_net_liq = walcl - wtregen - (rrpontsyd * 1000)

    # ECB_USD = ECBASSETSW × DEXUSEU (always FX-converted, no fallback)
    ecbassetsw = fred_data["ECBASSETSW"].reindex(dates, method="ffill")
    dexuseu = fred_data["DEXUSEU"].reindex(dates, method="ffill")
    ecb_usd = ecbassetsw * dexuseu

    # BOJ_USD = JPNASSETS × 1e8 / DEXJPUS
    jpnassets = fred_data["JPNASSETS"].reindex(dates, method="ffill")
    dexjpus = fred_data["DEXJPUS"].reindex(dates, method="ffill")
    boj_usd = jpnassets * 1e8 / dexjpus

    # China_M2_USD = Forward-fill last known value
    china_m2_usd = pd.Series(CHINA_M2_USD_LAST, index=dates, dtype=float)

    # US_M2 = WM2NS direct
    us_m2 = fred_data["WM2NS"].reindex(dates, method="ffill")

    # ── Indices (Spalten G-K) ──
    fed_idx = fed_net_liq / BASES["Fed"] * 100
    ecb_idx = ecb_usd / BASES["ECB"] * 100
    boj_idx = boj_usd / BASES["BOJ"] * 100

    # China Base: dynamic from last known pair
    # Use last historical row to derive china_base
    if not hist_df.empty and "China_M2_USD" in hist_df.columns and "China_Index" in hist_df.columns:
        last_row = hist_df.iloc[-1]
        last_raw = float(last_row["China_M2_USD"])
        last_idx = float(last_row["China_Index"])
        if last_idx > 0:
            china_base = last_raw / (last_idx / 100)
        else:
            china_base = CHINA_M2_USD_LAST / 1.024  # fallback from sheet ~102.4
    else:
        china_base = CHINA_M2_USD_LAST / 1.024

    china_idx = china_m2_usd / china_base * 100
    m2_idx = us_m2 / BASES["M2"] * 100

    # ── GLP (Spalte L) ──
    glp = (GLP_WEIGHTS["Fed"] * fed_idx +
           GLP_WEIGHTS["ECB"] * ecb_idx +
           GLP_WEIGHTS["BOJ"] * boj_idx +
           GLP_WEIGHTS["China"] * china_idx +
           GLP_WEIGHTS["M2"] * m2_idx)

    # ── MAs, Trend, Momentum, Acceleration ──
    # Need historical GLP for rolling windows (252d, 756d)
    # Concatenate historical GLP + new GLP
    if not hist_df.empty and "Global_Liq_Proxy" in hist_df.columns:
        hist_glp = hist_df.set_index("Date")["Global_Liq_Proxy"].astype(float)
        hist_glp = hist_glp[hist_glp.index < pd.Timestamp(start_date)]
        full_glp = pd.concat([hist_glp, glp]).sort_index()
        # Remove duplicates keeping last
        full_glp = full_glp[~full_glp.index.duplicated(keep="last")]
    else:
        full_glp = glp

    ma_12m = full_glp.rolling(252, min_periods=200).mean()
    ma_36m = full_glp.rolling(756, min_periods=600).mean()
    trend = (ma_12m - ma_36m) / ma_36m * 100
    momentum_6m = full_glp.pct_change(126) * 100
    acceleration = momentum_6m - momentum_6m.shift(63)

    # ── Filter to only new dates ──
    mask = full_glp.index >= pd.Timestamp(start_date)
    result = pd.DataFrame({
        "Date": full_glp.index[mask],
        "Fed_Net_Liq": fed_net_liq.reindex(full_glp.index[mask]),
        "ECB_USD": ecb_usd.reindex(full_glp.index[mask]),
        "BOJ_USD": boj_usd.reindex(full_glp.index[mask]),
        "China_M2_USD": china_m2_usd.reindex(full_glp.index[mask]),
        "US_M2": us_m2.reindex(full_glp.index[mask]),
        "Fed_Index": fed_idx.reindex(full_glp.index[mask]),
        "ECB_Index": ecb_idx.reindex(full_glp.index[mask]),
        "BOJ_Index": boj_idx.reindex(full_glp.index[mask]),
        "China_Index": china_idx.reindex(full_glp.index[mask]),
        "M2_Index": m2_idx.reindex(full_glp.index[mask]),
        "Global_Liq_Proxy": full_glp[mask],
        "MA_12M": ma_12m[mask],
        "MA_36M": ma_36m[mask],
        "Trend": trend[mask],
        "Momentum_6M": momentum_6m[mask],
        "Acceleration": acceleration[mask],
    }).reset_index(drop=True)

    return result


# ─────────────────────────────────────────────
# SHEET WRITER
# ─────────────────────────────────────────────

def write_data_liquidity(sheet, df: pd.DataFrame, hist_howell_phase: int = 4):
    """
    Write DATA_Liquidity tab.
    Inserts new rows at row 3 (after header + sub-header), newest first.
    Also writes Howell_Phase, RiskOn_Mult, RiskOff_Mult from CYCLES_Howell lookup.
    """
    ws = sheet.worksheet("DATA_Liquidity")

    # Phase-dependent lookups (V75 Section 5.6)
    RISKON_MULT = {1: 1.15, 2: 1.10, 3: 1.00, 4: 0.50}
    RISKOFF_MULT = {1: 0.90, 2: 0.95, 3: 1.00, 4: 1.20}

    phase = hist_howell_phase
    riskon = RISKON_MULT.get(phase, 1.00)
    riskoff = RISKOFF_MULT.get(phase, 1.00)

    # Build rows (newest first for sheet insertion)
    rows_to_write = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        sheet_row = [
            row["Date"].strftime("%Y-%m-%d"),
            round(row["Fed_Net_Liq"], 0) if pd.notna(row["Fed_Net_Liq"]) else "",
            round(row["ECB_USD"], 1) if pd.notna(row["ECB_USD"]) else "",
            round(row["BOJ_USD"], 0) if pd.notna(row["BOJ_USD"]) else "",
            round(row["China_M2_USD"], 0) if pd.notna(row["China_M2_USD"]) else "",
            round(row["US_M2"], 1) if pd.notna(row["US_M2"]) else "",
            round(row["Fed_Index"], 6) if pd.notna(row["Fed_Index"]) else "",
            round(row["ECB_Index"], 6) if pd.notna(row["ECB_Index"]) else "",
            round(row["BOJ_Index"], 6) if pd.notna(row["BOJ_Index"]) else "",
            round(row["China_Index"], 5) if pd.notna(row["China_Index"]) else "",
            round(row["M2_Index"], 5) if pd.notna(row["M2_Index"]) else "",
            round(row["Global_Liq_Proxy"], 6) if pd.notna(row["Global_Liq_Proxy"]) else "",
            round(row["MA_12M"], 6) if pd.notna(row["MA_12M"]) else "",
            round(row["MA_36M"], 5) if pd.notna(row["MA_36M"]) else "",
            round(row["Trend"], 6) if pd.notna(row["Trend"]) else "",
            round(row["Momentum_6M"], 6) if pd.notna(row["Momentum_6M"]) else "",
            round(row["Acceleration"], 6) if pd.notna(row["Acceleration"]) else "",
            phase,
            riskon,
            riskoff,
        ]
        rows_to_write.append(sheet_row)

    if not rows_to_write:
        log.warning("DATA_Liquidity: No rows to write.")
        return

    # Delete old rows for these dates first, then insert new ones
    # Strategy: find existing dates and overwrite, or insert at top
    num_new = len(rows_to_write)
    log.info(f"DATA_Liquidity: Writing {num_new} rows...")

    # Read existing dates to find overlap
    existing_dates = ws.col_values(1)[2:]  # Skip header + sub-header
    new_dates = [r[0] for r in rows_to_write]

    # Count how many existing rows to overwrite
    overlap_count = sum(1 for d in new_dates if d in existing_dates)
    insert_count = num_new - overlap_count

    if insert_count > 0:
        # Insert blank rows at position 3 (after header + sub-header)
        ws.insert_rows([[""] * 20] * insert_count, row=3)
        log.info(f"  Inserted {insert_count} new rows, overwriting {overlap_count} existing rows")

    # Write all rows starting at row 3
    cell_range = f"A3:T{3 + num_new - 1}"
    ws.update(cell_range, rows_to_write, value_input_option="USER_ENTERED")
    log.info(f"  Written {num_new} rows to {cell_range}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("V55 Data Collector — Starting")
    log.info("=" * 60)

    # ── Connect ──
    log.info("Connecting to Google Sheets...")
    sheet = connect_sheets()
    log.info("  ✓ Connected")

    log.info("Connecting to FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    log.info("  ✓ Connected")

    # ── Find date range ──
    last_good = find_last_good_date(sheet)
    start_date = last_good + timedelta(days=1)
    end_date = date.today()

    log.info(f"Last good date: {last_good}")
    log.info(f"Calculating: {start_date} → {end_date}")

    if start_date > end_date:
        log.info("Nothing to calculate — sheet is up to date.")
        return

    # ── Pull FRED data ──
    # Need extra history for lookbacks (756d for MA_36M + 252d buffer)
    fred_start = (start_date - timedelta(days=1200)).strftime("%Y-%m-%d")
    fred_end = end_date.strftime("%Y-%m-%d")
    log.info(f"Pulling FRED data: {fred_start} → {fred_end}")
    fred_data = pull_all_fred(fred, fred_start, fred_end)
    log.info(f"  ✓ FRED pull complete")

    # ── Read historical context ──
    log.info("Reading historical DATA_Liquidity...")
    hist_liq = read_sheet_tab(sheet, "DATA_Liquidity")
    log.info(f"  ✓ {len(hist_liq)} historical rows")

    # ── STUFE 1: DATA_Liquidity ──
    log.info("=" * 40)
    log.info("STUFE 1: DATA_Liquidity")
    log.info("=" * 40)
    liq_df = calc_data_liquidity(fred_data, hist_liq, start_date, end_date)
    log.info(f"  Calculated {len(liq_df)} rows")
    if not liq_df.empty:
        log.info(f"  Latest: {liq_df.iloc[-1]['Date'].date()} | "
                 f"GLP={liq_df.iloc[-1]['Global_Liq_Proxy']:.6f} | "
                 f"Trend={liq_df.iloc[-1]['Trend']:.6f}")

    # ── Write to Sheet ──
    log.info("Writing DATA_Liquidity to Sheet...")
    write_data_liquidity(sheet, liq_df)
    log.info("  ✓ DATA_Liquidity complete")

    # ── Done (Stufe 1 only for now) ──
    log.info("=" * 60)
    log.info("V55 Data Collector — Stufe 1 complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
