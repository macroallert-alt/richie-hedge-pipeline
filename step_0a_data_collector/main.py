"""
V55 Data Collector - step_0a_data_collector/main.py
Global Macro RV System - Upstream Signal Calculator

Stufe 1: DATA_Liquidity (FRED -> Raw -> Indizes -> GLP -> MAs -> Trend/Mom/Acc)

Referenz: V75 Systemstatusanalyse (Single Source of Truth)
Iron Rule: V38 NEVER MODIFY. Historische Werte = Ground Truth.
"""

import os
import sys
import logging
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from fredapi import Fred

# --- CONFIG ---

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
PRODUCTION_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("V55")

# --- CONSTANTS from V75 (NEVER CHANGE) ---

BASES = {
    "Fed": 5_730_394,
    "ECB": 6_388_921.20,
    "BOJ": 5_916_723_196_439,
    "M2": 17_928.40,
}

GLP_WEIGHTS = {
    "Fed": 0.090529,
    "ECB": 0.100933,
    "BOJ": 0.093473,
    "China": 0.431831,
    "M2": 0.283234,
}

CHINA_M2_USD_LAST = 49_261_001_737_116


# --- GOOGLE SHEETS ---

def connect_sheets():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    gc = gspread.authorize(creds)
    return gc.open_by_key(PRODUCTION_SHEET_ID)


# --- SHEET READER ---

def read_sheet_tab(sheet, tab_name):
    ws = sheet.worksheet(tab_name)
    data = ws.get_all_values()
    if len(data) < 3:
        return pd.DataFrame()
    headers = data[0]
    rows = data[2:]  # skip sub-header at index 1
    df = pd.DataFrame(rows, columns=headers)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    return df


def find_last_good_date(sheet):
    ws = sheet.worksheet("CALC_Macro_State")
    data = ws.get("A3:C102")
    for row in data:
        if len(row) >= 3 and row[2]:
            return pd.to_datetime(row[0]).date()
    return date(2026, 2, 3)


# --- FRED DATA PULL ---

def pull_all_fred(fred, start, end):
    series_list = {
        "WALCL": "WALCL", "WTREGEN": "WTREGEN", "RRPONTSYD": "RRPONTSYD",
        "ECBASSETSW": "ECBASSETSW", "DEXUSEU": "DEXUSEU",
        "JPNASSETS": "JPNASSETS", "DEXJPUS": "DEXJPUS", "WM2NS": "WM2NS",
        "TOTBKCR": "TOTBKCR",
        "PERMIT": "PERMIT", "UMCSENT": "UMCSENT", "ICSA": "ICSA", "INDPRO": "INDPRO",
        "VIXCLS": "VIXCLS", "BAMLH0A0HYM2": "BAMLH0A0HYM2", "NFCI": "NFCI",
        "DEXCHUS": "DEXCHUS",
    }
    full_dates = pd.date_range(start=start, end=end, freq="D")
    result = {}
    for key, fred_id in series_list.items():
        log.info(f"  Pulling {fred_id}...")
        try:
            raw = fred.get_series(fred_id, observation_start=start, observation_end=end)
            if raw is None or raw.empty:
                log.warning(f"  {fred_id} returned empty!")
                result[key] = pd.Series(np.nan, index=full_dates, dtype=float)
                continue
            # Align to full daily range, forward-fill, then backfill start
            aligned = raw.reindex(full_dates).ffill().bfill()
            result[key] = aligned
            log.info(f"  OK {fred_id}: {aligned.dropna().shape[0]} values, last={aligned.iloc[-1]:.4f}")
        except Exception as e:
            log.error(f"  FAILED: {fred_id} -> {e}")
            result[key] = pd.Series(np.nan, index=full_dates, dtype=float)
    return result


# --- STUFE 1: DATA_Liquidity ---

def calc_data_liquidity(fred_data, hist_df, start_date, end_date):
    # All FRED series share the same daily index from pull_all_fred
    
    # Raw Values (Spalten B-F)
    fed_net_liq = fred_data["WALCL"] - fred_data["WTREGEN"] - (fred_data["RRPONTSYD"] * 1000)
    ecb_usd = fred_data["ECBASSETSW"] * fred_data["DEXUSEU"]
    boj_usd = fred_data["JPNASSETS"] * 1e8 / fred_data["DEXJPUS"]
    china_m2_usd = pd.Series(CHINA_M2_USD_LAST, index=fred_data["WALCL"].index, dtype=float)
    us_m2 = fred_data["WM2NS"]

    # Indices (Spalten G-K)
    fed_idx = fed_net_liq / BASES["Fed"] * 100
    ecb_idx = ecb_usd / BASES["ECB"] * 100
    boj_idx = boj_usd / BASES["BOJ"] * 100
    m2_idx = us_m2 / BASES["M2"] * 100

    # China Base: dynamic from last known pair in historical sheet
    china_base = CHINA_M2_USD_LAST / 1.0241  # default fallback
    if not hist_df.empty and "China_M2_USD" in hist_df.columns and "China_Index" in hist_df.columns:
        try:
            last_row = hist_df.iloc[-1]
            last_raw = float(last_row["China_M2_USD"])
            last_idx = float(last_row["China_Index"])
            if last_idx > 0 and last_raw > 0:
                china_base = last_raw / (last_idx / 100)
                log.info(f"  China Base from sheet: {china_base:.0f}")
        except (ValueError, TypeError):
            pass
    china_idx = china_m2_usd / china_base * 100

    # GLP (Spalte L)
    glp_fred = (GLP_WEIGHTS["Fed"] * fed_idx +
                GLP_WEIGHTS["ECB"] * ecb_idx +
                GLP_WEIGHTS["BOJ"] * boj_idx +
                GLP_WEIGHTS["China"] * china_idx +
                GLP_WEIGHTS["M2"] * m2_idx)

    # Combine with historical GLP for MA continuity
    if not hist_df.empty and "Global_Liq_Proxy" in hist_df.columns:
        hist_glp = hist_df.set_index("Date")["Global_Liq_Proxy"].astype(float)
        hist_glp = hist_glp[hist_glp.index < pd.Timestamp(start_date)]
        full_glp = pd.concat([hist_glp, glp_fred]).sort_index()
        full_glp = full_glp[~full_glp.index.duplicated(keep="last")]
    else:
        full_glp = glp_fred

    # MAs, Trend, Momentum, Acceleration
    ma_12m = full_glp.rolling(252, min_periods=200).mean()
    ma_36m = full_glp.rolling(756, min_periods=600).mean()
    trend = (ma_12m - ma_36m) / ma_36m * 100
    momentum_6m = full_glp.pct_change(126) * 100
    acceleration = momentum_6m - momentum_6m.shift(63)

    # Filter to new dates only
    mask = full_glp.index >= pd.Timestamp(start_date)
    idx = full_glp.index[mask]

    result = pd.DataFrame({
        "Date": idx,
        "Fed_Net_Liq": fed_net_liq.reindex(idx),
        "ECB_USD": ecb_usd.reindex(idx),
        "BOJ_USD": boj_usd.reindex(idx),
        "China_M2_USD": china_m2_usd.reindex(idx),
        "US_M2": us_m2.reindex(idx),
        "Fed_Index": fed_idx.reindex(idx),
        "ECB_Index": ecb_idx.reindex(idx),
        "BOJ_Index": boj_idx.reindex(idx),
        "China_Index": china_idx.reindex(idx),
        "M2_Index": m2_idx.reindex(idx),
        "Global_Liq_Proxy": full_glp[mask],
        "MA_12M": ma_12m[mask],
        "MA_36M": ma_36m[mask],
        "Trend": trend[mask],
        "Momentum_6M": momentum_6m[mask],
        "Acceleration": acceleration[mask],
    }).reset_index(drop=True)

    result = result.ffill()
    return result


# --- SHEET WRITER ---

def fmt(val, decimals=0):
    if pd.isna(val):
        return ""
    if decimals == 0:
        return int(round(val))
    return round(float(val), decimals)


def write_data_liquidity(sheet, df, hist_howell_phase=4):
    ws = sheet.worksheet("DATA_Liquidity")

    RISKON = {1: 1.15, 2: 1.10, 3: 1.00, 4: 0.50}
    RISKOFF = {1: 0.90, 2: 0.95, 3: 1.00, 4: 1.20}
    phase = hist_howell_phase
    riskon = RISKON.get(phase, 1.00)
    riskoff = RISKOFF.get(phase, 1.00)

    rows_to_write = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        rows_to_write.append([
            row["Date"].strftime("%Y-%m-%d"),
            fmt(row["Fed_Net_Liq"], 0),
            fmt(row["ECB_USD"], 1),
            fmt(row["BOJ_USD"], 0),
            fmt(row["China_M2_USD"], 0),
            fmt(row["US_M2"], 1),
            fmt(row["Fed_Index"], 6),
            fmt(row["ECB_Index"], 6),
            fmt(row["BOJ_Index"], 6),
            fmt(row["China_Index"], 5),
            fmt(row["M2_Index"], 5),
            fmt(row["Global_Liq_Proxy"], 6),
            fmt(row["MA_12M"], 6),
            fmt(row["MA_36M"], 5),
            fmt(row["Trend"], 6),
            fmt(row["Momentum_6M"], 6),
            fmt(row["Acceleration"], 6),
            phase, riskon, riskoff,
        ])

    if not rows_to_write:
        log.warning("DATA_Liquidity: No rows to write.")
        return

    num = len(rows_to_write)
    log.info(f"DATA_Liquidity: Writing {num} rows...")

    existing_dates = ws.col_values(1)[2:]
    new_dates = set(r[0] for r in rows_to_write)
    overlap = sum(1 for d in existing_dates if d in new_dates)
    inserts = num - overlap

    if inserts > 0:
        ws.insert_rows([[""] * 20] * inserts, row=3)
        log.info(f"  Inserted {inserts} blank rows")

    cell_range = f"A3:T{3 + num - 1}"
    ws.update(cell_range, rows_to_write, value_input_option="RAW")
    log.info(f"  Written to {cell_range}")


# --- MAIN ---

def main():
    log.info("=" * 60)
    log.info("V55 Data Collector - Starting")
    log.info("=" * 60)

    log.info("Connecting to Google Sheets...")
    sheet = connect_sheets()
    log.info("  OK")

    log.info("Connecting to FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    log.info("  OK")

    last_good = find_last_good_date(sheet)
    start_date = last_good + timedelta(days=1)
    end_date = date.today()

    log.info(f"Last good date: {last_good}")
    log.info(f"Calculating: {start_date} -> {end_date}")

    if start_date > end_date:
        log.info("Sheet is up to date. Nothing to do.")
        return

    fred_start = (start_date - timedelta(days=1200)).strftime("%Y-%m-%d")
    fred_end = end_date.strftime("%Y-%m-%d")
    log.info(f"Pulling FRED: {fred_start} -> {fred_end}")
    fred_data = pull_all_fred(fred, fred_start, fred_end)

    log.info("Reading historical DATA_Liquidity...")
    hist_liq = read_sheet_tab(sheet, "DATA_Liquidity")
    log.info(f"  {len(hist_liq)} historical rows")

    log.info("STUFE 1: DATA_Liquidity")
    liq_df = calc_data_liquidity(fred_data, hist_liq, start_date, end_date)
    log.info(f"  {len(liq_df)} rows calculated")

    if not liq_df.empty:
        r = liq_df.iloc[-1]
        log.info(f"  Latest: {r['Date'].date()} GLP={r['Global_Liq_Proxy']:.6f} "
                 f"China_M2={r['China_M2_USD']:.0f} BOJ={r['BOJ_USD']:.0f}")

    log.info("Writing DATA_Liquidity...")
    write_data_liquidity(sheet, liq_df)

    log.info("=" * 60)
    log.info("Stufe 1 complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
