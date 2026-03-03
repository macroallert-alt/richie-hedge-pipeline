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

# --- CYCLES_HOWELL CONSTANTS ---

HOWELL_TROUGH_DATE = date(2022, 10, 1)  # Current cycle trough
HOWELL_CYCLE_LENGTH = 65  # Months
HOWELL_DAYS_PER_MONTH = 30.44  # For fractional month calculation

# Phase 4 TURBULENCE current stint started 2025-09-16
PHASE4_ENTRY_DATE = date(2025, 9, 16)
PHASE4_ENTRY_CONFIDENCE = 0.300
PHASE_CONFIDENCE_DAILY_RATE = 0.000954  # Linear growth from entry

# Phase-based lookup tables (100% verified from historical CSV)
RISKON_MULT = {1: 1.15, 2: 1.10, 3: 1.00, 4: 0.50}
RISKOFF_MULT = {1: 0.90, 2: 0.95, 3: 1.00, 4: 1.20}

PHASE_NAME = {1: "REBOUND", 2: "CALM", 3: "SPECULATION", 4: "TURBULENCE"}

PHASE_REGIME = {1: "RISK ON", 2: "RISK ON", 3: "RISK ON", 4: "RISK OFF"}

PHASE_SIGNALS = {
    # Phase: (Eq, Cr, Cmd, PM, GovB, Cash, Crypto, Favored)
    1: ("🟡 AMBER", "🔴 RED", "🔴 RED", "🟢 GREEN", "🟢 GREEN", "🟢 GREEN", "🔴 RED",
        "Tech, Cyclicals, SmallCaps"),
    2: ("🟢 GREEN", "🟢 GREEN", "🟢 GREEN", "🟡 YELLOW", "🟡 AMBER", "🟡 AMBER", "🟢 GREEN",
        "Financials, EM, Broad Equity"),
    3: ("🟡 AMBER", "🔴 RED", "🟢 GREEN", "🟡 YELLOW", "🟡 AMBER", "🟡 AMBER", "🟡 AMBER",
        "Energy, Mining, Commodities"),
    4: ("🔴 RED", "🔴 RED", "🔴 RED", "🟢 GREEN", "🟢 GREEN", "🟢 GREEN", "🔴 RED",
        "Cash, Govt Bonds, Defensives"),
}

# Sub_Phase thresholds: percentage of avg phase duration (124d for Phase 4)
# EARLY=0-35%, MID=35-84%, DEEP=84-114%, RECOVERY=114%+
PHASE_AVG_DURATION = {1: 170, 2: 115, 3: 620, 4: 124}

SUB_PHASE_NAMES = {
    1: ("EARLY REBOUND", "MID REBOUND", "LATE REBOUND", "LATE REBOUND"),
    2: ("EARLY CALM", "MID CALM", "LATE CALM", "LATE CALM"),
    3: ("EARLY SPECULATION", "MID SPECULATION", "LATE SPECULATION", "LATE SPECULATION"),
    4: ("EARLY TURBULENCE", "MID TURBULENCE", "DEEP TURBULENCE", "EARLY TURBULENCE RECOVERY"),
}

# Cycle_Interpretation thresholds (CP-based, verified from data)
def cycle_interpretation(cp):
    if cp < 20: return "Early Cycle (Accumulate)"
    elif cp < 40: return "Early-Mid Cycle (Risk On)"
    elif cp < 60: return "Mid Cycle (Selective)"
    elif cp < 80: return "Late Mid Cycle (Caution)"
    else: return "Late Cycle (Defensive)"


# GLI_Normalized: CBC proprietär, forward-fill last known value
GLI_NORMALIZED_LAST = 34.52


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
        hist_glp = pd.to_numeric(hist_df.set_index("Date")["Global_Liq_Proxy"], errors="coerce").dropna()
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

    phase = hist_howell_phase
    riskon = RISKON_MULT.get(phase, 1.00)
    riskoff = RISKOFF_MULT.get(phase, 1.00)

    # Build rows (newest first)
    new_dates_set = set()
    rows_to_write = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        d = row["Date"].strftime("%Y-%m-%d")
        new_dates_set.add(d)
        rows_to_write.append([
            d,
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

    # Step 1: Delete existing rows that overlap with our new dates
    # Read all dates from column A (skip header + sub-header)
    existing_dates = ws.col_values(1)[2:]  # 0-indexed: row3 onwards
    rows_to_delete = []
    for i, d in enumerate(existing_dates):
        if d in new_dates_set:
            rows_to_delete.append(i + 3)  # +3 because sheet is 1-indexed + 2 header rows

    # Delete from bottom to top so indices don't shift
    for row_idx in sorted(rows_to_delete, reverse=True):
        ws.delete_rows(row_idx)
        log.info(f"  Deleted existing row {row_idx} (overlap)")

    # Step 2: Insert new rows at top (row 3)
    ws.insert_rows(rows_to_write, row=3, value_input_option="RAW")
    log.info(f"  Inserted {num} rows at row 3")
# --- STUFE 2: CYCLES_Howell ---

def calc_cycles_howell(liq_df, start_date, end_date):
    """
    Calculate CYCLES_Howell tab for new dates.
    
    Current state: Phase 4 TURBULENCE since 2025-09-16, MST=41 (Mar 2026).
    Strategy: Forward-fill phase + compute derived metrics daily.
    GLI_Normalized: CBC proprietary, forward-fill 34.52.
    """
    rows = []
    current_phase = 4  # TURBULENCE - forward-fill until manual transition alert
    
    for _, lrow in liq_df.iterrows():
        d = lrow["Date"]
        d_date = d.date() if hasattr(d, 'date') else d
        
        # GLP, Trend, Momentum, Acceleration from DATA_Liquidity
        glp = lrow.get("Global_Liq_Proxy", np.nan)
        trend = lrow.get("Trend", np.nan)
        mom6m = lrow.get("Momentum_6M", np.nan)
        accel = lrow.get("Acceleration", np.nan)
        
        # MST: integer months since trough (increments on 15th of each month)
        # Verified: MST = (year_diff * 12 + month_diff) + 1 if day >= 15
        days_since_trough = (d_date - HOWELL_TROUGH_DATE).days
        months_frac = days_since_trough / HOWELL_DAYS_PER_MONTH  # For CP only
        y_diff = d_date.year - HOWELL_TROUGH_DATE.year
        m_diff = d_date.month - HOWELL_TROUGH_DATE.month
        mst = y_diff * 12 + m_diff
        if d_date.day >= 15:
            mst += 1
        
        # Cycle_Position: fractional, daily resolution
        cp = round(months_frac / HOWELL_CYCLE_LENGTH * 100, 1)
        
        # Sine_Wave: uses INTEGER MST (verified: sin(2*pi*40/65) = -0.663123)
        sine_wave = round(np.sin(2 * np.pi * mst / HOWELL_CYCLE_LENGTH), 6)
        
        # Months_To_Trough
        mtt = HOWELL_CYCLE_LENGTH - mst
        
        # Phase lookups
        phase = current_phase
        phase_name = PHASE_NAME[phase]
        riskon = RISKON_MULT[phase]
        riskoff = RISKOFF_MULT[phase]
        regime = PHASE_REGIME[phase]
        
        # Phase_Confidence: linear growth from 0.300 at phase entry
        days_in_phase = (d_date - PHASE4_ENTRY_DATE).days
        phase_conf = round(PHASE4_ENTRY_CONFIDENCE + days_in_phase * PHASE_CONFIDENCE_DAILY_RATE, 4)
        
        # Sine_Agrees: Phase 4 (bearish/RISK OFF) agrees when sine < 0
        if phase in (1, 2):
            sine_agrees = sine_wave > 0
        elif phase == 4:
            sine_agrees = sine_wave < 0
        else:  # Phase 3
            sine_agrees = True  # Speculation = neutral
        
        # Cycle_Interpretation (CP-based)
        cycle_interp = cycle_interpretation(cp)
        
        # Sub_Phase: based on days in current phase / avg duration
        avg_dur = PHASE_AVG_DURATION[phase]
        pct_of_avg = days_in_phase / avg_dur * 100 if avg_dur > 0 else 0
        sub_names = SUB_PHASE_NAMES[phase]
        if pct_of_avg < 35.0:
            sub_phase = sub_names[0]
        elif pct_of_avg < 83.5:
            sub_phase = sub_names[1]
        elif pct_of_avg < 113.5:
            sub_phase = sub_names[2]
        else:
            sub_phase = sub_names[3]
        
        # Signal lookups
        sigs = PHASE_SIGNALS[phase]
        
        # GLI_Normalized: forward-fill
        gli_norm = GLI_NORMALIZED_LAST
        
        rows.append({
            "Date": d,
            "Global_Liq_Proxy": glp,
            "Trend": trend,
            "Momentum_6M": mom6m,
            "Acceleration": accel,
            "Howell_Phase": phase,
            "Phase_Name": phase_name,
            "Howell_RiskOn_Mult": riskon,
            "Howell_RiskOff_Mult": riskoff,
            "Months_Since_Trough": mst,
            "Cycle_Position": cp,
            "Cycle_Interpretation": cycle_interp,
            "Months_To_Trough": mtt,
            "GLI_Normalized": gli_norm,
            "Sine_Wave": sine_wave,
            "Phase_Confidence": phase_conf,
            "Sine_Agrees": sine_agrees,
            "Regime": regime,
            "Sig_Equities": sigs[0],
            "Sig_Credits": sigs[1],
            "Sig_Commodities": sigs[2],
            "Sig_PM": sigs[3],
            "Sig_GovtBonds": sigs[4],
            "Sig_Cash": sigs[5],
            "Sig_Crypto": sigs[6],
            "Favored_Sectors": sigs[7],
            "Sub_Phase": sub_phase,
        })
    
    return pd.DataFrame(rows)


def write_cycles_howell(sheet, df):
    """Write CYCLES_Howell rows to sheet (newest first, row 3 = first data row)."""
    ws = sheet.worksheet("CYCLES_Howell")
    
    new_dates_set = set()
    rows_to_write = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        d = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        new_dates_set.add(d)
        rows_to_write.append([
            d,
            fmt(row["Global_Liq_Proxy"], 6),
            fmt(row["Trend"], 6),
            fmt(row["Momentum_6M"], 6),
            fmt(row["Acceleration"], 6),
            int(row["Howell_Phase"]),
            row["Phase_Name"],
            row["Howell_RiskOn_Mult"],
            row["Howell_RiskOff_Mult"],
            int(row["Months_Since_Trough"]),
            row["Cycle_Position"],
            row["Cycle_Interpretation"],
            int(row["Months_To_Trough"]),
            row["GLI_Normalized"],
            fmt(row["Sine_Wave"], 6),
            fmt(row["Phase_Confidence"], 4),
            row["Sine_Agrees"],
            row["Regime"],
            row["Sig_Equities"],
            row["Sig_Credits"],
            row["Sig_Commodities"],
            row["Sig_PM"],
            row["Sig_GovtBonds"],
            row["Sig_Cash"],
            row["Sig_Crypto"],
            row["Favored_Sectors"],
            row["Sub_Phase"],
        ])
    
    if not rows_to_write:
        log.warning("CYCLES_Howell: No rows to write.")
        return
    
    num = len(rows_to_write)
    log.info(f"CYCLES_Howell: Writing {num} rows...")
    
    # Delete overlapping rows (same pattern as DATA_Liquidity)
    existing_dates = ws.col_values(1)[2:]  # Skip header + sub-header
    rows_to_delete = []
    for i, d in enumerate(existing_dates):
        if d in new_dates_set:
            rows_to_delete.append(i + 3)
    
    for row_idx in sorted(rows_to_delete, reverse=True):
        ws.delete_rows(row_idx)
        log.info(f"  Deleted existing row {row_idx} (overlap)")
    
    # Insert at row 3 (newest first)
    ws.insert_rows(rows_to_write, row=3, value_input_option="RAW")
    log.info(f"  Inserted {num} rows at row 3")



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

    # --- STUFE 2: CYCLES_Howell ---
    log.info("STUFE 2: CYCLES_Howell")
    howell_df = calc_cycles_howell(liq_df, start_date, end_date)
    log.info(f"  {len(howell_df)} rows calculated")

    if not howell_df.empty:
        r = howell_df.iloc[-1]
        log.info(f"  Latest: {r['Date'].date() if hasattr(r['Date'], 'date') else r['Date']} "
                 f"Phase={r['Howell_Phase']} MST={r['Months_Since_Trough']} "
                 f"CP={r['Cycle_Position']} SW={r['Sine_Wave']:.6f}")

    log.info("Writing CYCLES_Howell...")
    write_cycles_howell(sheet, howell_df)

    log.info("=" * 60)
    log.info("Stufe 2 complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
