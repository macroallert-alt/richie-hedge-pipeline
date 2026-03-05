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
            str(fmt(row["Fed_Net_Liq"], 0)),
            str(fmt(row["ECB_USD"], 1)),
            str(fmt(row["BOJ_USD"], 0)),
            str(fmt(row["China_M2_USD"], 0)),
            str(fmt(row["US_M2"], 1)),
            str(fmt(row["Fed_Index"], 6)),
            str(fmt(row["ECB_Index"], 6)),
            str(fmt(row["BOJ_Index"], 6)),
            str(fmt(row["China_Index"], 5)),
            str(fmt(row["M2_Index"], 5)),
            str(fmt(row["Global_Liq_Proxy"], 6)),
            str(fmt(row["MA_12M"], 6)),
            str(fmt(row["MA_36M"], 5)),
            str(fmt(row["Trend"], 6)),
            str(fmt(row["Momentum_6M"], 6)),
            str(fmt(row["Acceleration"], 6)),
            str(phase), str(riskon), str(riskoff),
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
    """Write CYCLES_Howell rows to sheet (newest first, row 3 = first data row).
    
    CRITICAL: All numeric values must be written as STRINGS to prevent
    Google Sheets German locale from mangling decimals (. as thousands sep,
    1.2 interpreted as date Dec 31, 0.5 rounded to 0, etc.)
    """
    ws = sheet.worksheet("CYCLES_Howell")
    
    def s(val, decimals=6):
        """Format numeric value as string for RAW mode."""
        if pd.isna(val):
            return ""
        if decimals == 0:
            return str(int(round(val)))
        return f"{float(val):.{decimals}f}"
    
    new_dates_set = set()
    rows_to_write = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        d = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        new_dates_set.add(d)
        rows_to_write.append([
            d,
            s(row["Global_Liq_Proxy"], 6),
            s(row["Trend"], 6),
            s(row["Momentum_6M"], 6),
            s(row["Acceleration"], 6),
            int(row["Howell_Phase"]),
            row["Phase_Name"],
            s(row["Howell_RiskOn_Mult"], 2),
            s(row["Howell_RiskOff_Mult"], 2),
            int(row["Months_Since_Trough"]),
            s(row["Cycle_Position"], 1),
            row["Cycle_Interpretation"],
            int(row["Months_To_Trough"]),
            s(row["GLI_Normalized"], 2),
            s(row["Sine_Wave"], 6),
            s(row["Phase_Confidence"], 4),
            str(row["Sine_Agrees"]),
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


# --- STUFE 3: DATA_K16_K17 ---

# CBC proprietary values (frozen since Apps Script stopped 2026-02-03)
GLI_ACCEL_LAST = 2.365346
HOWELL_MOM6M_LAST = -1.330617

# Credit Impulse coefficients (from BLOCKER 4 / Addendum IV)
CI_INTERCEPT = -0.00080156
CI_BETA_US = 0.46845456
CI_BETA_CN = 0.28075508
CI_LOOKBACK_PCT = 189   # 9 months
CI_LOOKBACK_ACC = 252   # 12 months


def read_cu_au_from_prices(sheet, start_date, end_date):
    """Read COPPER and GLD prices from DATA_Prices tab, compute Cu/Au ratio."""
    ws = sheet.worksheet("DATA_Prices")
    data = ws.get_all_values()
    if len(data) < 3:
        return pd.Series(dtype=float)
    
    headers = data[0]
    rows = data[2:]  # skip sub-header
    df = pd.DataFrame(rows, columns=headers)
    
    if "Date" not in df.columns:
        log.warning("DATA_Prices: No Date column found")
        return pd.Series(dtype=float)
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    
    # Find COPPER and GLD columns
    copper_col = None
    gld_col = None
    for col in df.columns:
        cl = col.strip().upper()
        if cl == "COPPER":
            copper_col = col
        elif cl == "GLD":
            gld_col = col
    
    if copper_col is None or gld_col is None:
        available = [c for c in df.columns if c != "Date"][:20]
        log.warning(f"DATA_Prices: COPPER={copper_col}, GLD={gld_col} — columns: {available}")
        return pd.Series(dtype=float)
    
    copper = pd.to_numeric(df[copper_col].astype(str).str.replace(",", "."), errors="coerce")
    gld = pd.to_numeric(df[gld_col].astype(str).str.replace(",", "."), errors="coerce")
    
    log.info(f"  COPPER: {copper.notna().sum()} numeric vals, last={copper.dropna().iloc[-1] if copper.notna().any() else 'NONE'}")
    log.info(f"  GLD: {gld.notna().sum()} numeric vals, last={gld.dropna().iloc[-1] if gld.notna().any() else 'NONE'}")
    
    ratio = copper / gld
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    
    if ratio.empty:
        log.warning(f"  Cu/Au ratio: EMPTY after dropna (copper has {copper.notna().sum()} vals, gld has {gld.notna().sum()} vals)")
        return pd.Series(dtype=float)
    
    log.info(f"  Cu/Au ratio: {len(ratio)} values, last={ratio.iloc[-1]:.6f} ({ratio.index[-1].date()})")
    return ratio


def calc_credit_impulse(fred_data, china_m2_value):
    """
    Credit Impulse = intercept + beta_US * US_accel + beta_CN * CN_accel
    US: TOTBKCR pct_change(189) acceleration over 252 days
    CN: China M2 (constant forward-fill) — CN acceleration = 0 when constant
    """
    totbkcr = fred_data["TOTBKCR"]
    
    # US component
    us_pct = totbkcr.pct_change(CI_LOOKBACK_PCT)
    us_accel = us_pct - us_pct.shift(CI_LOOKBACK_ACC)
    
    # China component: with constant China M2, pct_change is 0 -> accel is 0
    # So CI = intercept + beta_US * us_accel + beta_CN * 0
    ci = CI_INTERCEPT + CI_BETA_US * us_accel
    
    return ci


def calc_k16_vote(cu_au_mom6m):
    """K16: Cu/Au Momentum 6M, symmetric +/-0.02 threshold."""
    if pd.isna(cu_au_mom6m):
        return 0
    if cu_au_mom6m > 0.02:
        return 1
    elif cu_au_mom6m < -0.02:
        return -1
    return 0


def calc_k17_vote(credit_impulse):
    """K17: Credit Impulse, asymmetric +0.02/-0.01 threshold."""
    if pd.isna(credit_impulse):
        return 0
    if credit_impulse > 0.02:
        return 1
    elif credit_impulse < -0.01:
        return -1
    return 0


def calc_k4_vote(gli_accel):
    """K4: GLI Acceleration, symmetric +/-0.5 threshold."""
    if pd.isna(gli_accel):
        return 0
    if gli_accel > 0.5:
        return 1
    elif gli_accel < -0.5:
        return -1
    return 0


def calc_howell_vote(phase, howell_mom6m):
    """Howell Vote: Phase-based + Momentum for Phase 3."""
    if phase in (1, 2):
        return 1
    if phase == 4:
        return -1
    if phase == 3:
        if pd.notna(howell_mom6m):
            if howell_mom6m > 0.5:
                return 1
            elif howell_mom6m < -0.5:
                return -1
        return 0
    return 0


def apply_veto(liq_dir_raw, howell_vote, howell_phase):
    """VETO_H1: Phase=1 AND Howell=+1 AND Raw=-1 -> Final=0."""
    if howell_phase == 1 and howell_vote == 1 and liq_dir_raw == -1:
        return 0
    return liq_dir_raw


def calc_data_k16_k17(fred_data, sheet, howell_df, start_date, end_date):
    """
    Calculate DATA_K16_K17 for new dates.
    
    Inputs:
        fred_data: FRED series (TOTBKCR, DEXCHUS, etc.)
        sheet: Google Sheet connection (for DATA_Prices Cu/Au)
        howell_df: CYCLES_Howell results (for Phase)
        start_date, end_date: calculation range
    """
    # --- Cu/Au Ratio from DATA_Prices ---
    log.info("  Reading Cu/Au from DATA_Prices...")
    cu_au_ratio = read_cu_au_from_prices(sheet, start_date, end_date)
    
    if cu_au_ratio.empty:
        log.warning("  No Cu/Au data — K16 will be 0")
        cu_au_mom6m = pd.Series(dtype=float)
    else:
        cu_au_mom6m = cu_au_ratio.pct_change(126)
    
    # --- Credit Impulse ---
    log.info("  Calculating Credit Impulse...")
    ci_series = calc_credit_impulse(fred_data, CHINA_M2_USD_LAST)
    
    # --- Build daily rows ---
    howell_lookup = {}
    if not howell_df.empty:
        for _, hr in howell_df.iterrows():
            d = hr["Date"]
            key = d.date() if hasattr(d, "date") else d
            howell_lookup[key] = int(hr["Howell_Phase"])
    
    # Read last confirmed state from sheet for 5-Day Confirmation continuity
    # IMPORTANT: Read from dates BEFORE start_date to avoid reading our own previous writes
    hist_k16 = read_sheet_tab(sheet, "DATA_K16_K17")
    last_confirmed = -1  # default
    if not hist_k16.empty:
        hist_before = hist_k16[hist_k16["Date"] < pd.Timestamp(start_date)]
        for _, hr in hist_before.sort_values("Date", ascending=False).iterrows():
            try:
                ldc = float(hr.get("Liq_Dir_Confirmed", np.nan))
                if not pd.isna(ldc):
                    last_confirmed = int(ldc)
                    break
            except (ValueError, TypeError):
                continue
    log.info(f"  5-Day Confirmation seed: confirmed={last_confirmed}")
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    rows = []
    confirmed = last_confirmed
    pending_dir = None
    pending_count = 0
    
    for d in dates:
        d_date = d.date()
        
        # Cu/Au
        cu_au = cu_au_ratio.get(d, np.nan) if not cu_au_ratio.empty else np.nan
        # Try nearest previous date if exact match missing
        if pd.isna(cu_au) and not cu_au_ratio.empty:
            prior = cu_au_ratio[cu_au_ratio.index <= d]
            if not prior.empty:
                cu_au = prior.iloc[-1]
        
        mom6m = cu_au_mom6m.get(d, np.nan) if not cu_au_mom6m.empty else np.nan
        if pd.isna(mom6m) and not cu_au_mom6m.empty:
            prior = cu_au_mom6m[cu_au_mom6m.index <= d]
            if not prior.empty:
                mom6m = prior.iloc[-1]
        
        # Credit Impulse
        ci = ci_series.get(d, np.nan) if ci_series is not None else np.nan
        if pd.isna(ci) and ci_series is not None and not ci_series.empty:
            prior = ci_series[ci_series.index <= d]
            if not prior.empty:
                ci = prior.iloc[-1]
        
        # CBC proprietary (forward-fill)
        gli_accel = GLI_ACCEL_LAST
        howell_mom6m = HOWELL_MOM6M_LAST
        
        # Phase from Howell calc
        phase = howell_lookup.get(d_date, 4)
        
        # 4 Votes
        k16 = calc_k16_vote(mom6m)
        k17 = calc_k17_vote(ci)
        k4 = calc_k4_vote(gli_accel)
        hv = calc_howell_vote(phase, howell_mom6m)
        
        # Vote aggregation
        vote_sum = k16 + k17 + k4 + hv
        liq_raw = int(np.sign(vote_sum)) if vote_sum != 0 else 0
        
        # Veto
        liq_final = apply_veto(liq_raw, hv, phase)
        
        # Liq Detail
        liq_detail = f"H:{hv}|A:{k4}|Cu:{k16}|Cr:{k17}={vote_sum}"
        
        # 5-Day Confirmation
        if liq_final == confirmed:
            # Same as confirmed — reset any pending
            pending_dir = None
            pending_count = 0
            confirm_detail = "CONF"
        elif pending_dir is not None and liq_final == pending_dir:
            # Continuing pending direction
            pending_count += 1
            if pending_count >= 5:
                confirmed = liq_final
                pending_dir = None
                pending_count = 0
                confirm_detail = "CONF"
            else:
                confirm_detail = f"PEND({liq_final}):{ pending_count}/5"
        else:
            # New pending direction
            pending_dir = liq_final
            pending_count = 1
            confirm_detail = f"PEND({liq_final}):{pending_count}/5"
        
        # Vote_Sum_Magnitude
        vsm = abs(vote_sum)
        
        rows.append({
            "Date": d,
            "Cu/Au_Ratio": cu_au,
            "Cu/Au_Mom6M": mom6m,
            "K16_Vote": k16,
            "Credit_Impulse": ci,
            "K17_Vote": k17,
            "GLI_Accel": gli_accel,
            "K4_Vote": k4,
            "Howell_Phase": phase,
            "Howell_Mom6M": howell_mom6m,
            "Howell_Vote": hv,
            "Vote_Sum": vote_sum,
            "Liq_Dir_Raw": liq_raw,
            "Liq_Dir_Final": liq_final,
            "Liq_Detail": liq_detail,
            "Liq_Dir_Confirmed": confirmed,
            "Liq_Confirm_Detail": confirm_detail,
            "Vote_Sum_Magnitude": vsm,
        })
    
    return pd.DataFrame(rows)


def write_data_k16_k17(sheet, df):
    """Write DATA_K16_K17 rows to sheet (newest first, row 3)."""
    ws = sheet.worksheet("DATA_K16_K17")
    
    new_dates_set = set()
    rows_to_write = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        d = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        new_dates_set.add(d)
        rows_to_write.append([
            d,
            fmt(row["Cu/Au_Ratio"], 6),
            fmt(row["Cu/Au_Mom6M"], 6),
            int(row["K16_Vote"]),
            fmt(row["Credit_Impulse"], 6),
            int(row["K17_Vote"]),
            fmt(row["GLI_Accel"], 6),
            int(row["K4_Vote"]),
            int(row["Howell_Phase"]),
            fmt(row["Howell_Mom6M"], 6),
            int(row["Howell_Vote"]),
            int(row["Vote_Sum"]),
            int(row["Liq_Dir_Raw"]),
            int(row["Liq_Dir_Final"]),
            row["Liq_Detail"],
            int(row["Liq_Dir_Confirmed"]),
            row["Liq_Confirm_Detail"],
            int(row["Vote_Sum_Magnitude"]),
        ])
    
    if not rows_to_write:
        log.warning("DATA_K16_K17: No rows to write.")
        return
    
    num = len(rows_to_write)
    log.info(f"DATA_K16_K17: Writing {num} rows...")
    
    # Delete overlapping rows
    existing_dates = ws.col_values(1)[2:]
    rows_to_delete = []
    for i, d in enumerate(existing_dates):
        if d in new_dates_set:
            rows_to_delete.append(i + 3)
    
    for row_idx in sorted(rows_to_delete, reverse=True):
        ws.delete_rows(row_idx)
        log.info(f"  Deleted existing row {row_idx} (overlap)")
    
    ws.insert_rows(rows_to_write, row=3, value_input_option="RAW")
    log.info(f"  Inserted {num} rows at row 3")


# --- STUFE 4: CALC_Macro_State ---

# Macro State Map (V55 Spec v3.0 Kap. 7.4, 31 entries, 100% purity)
MACRO_STATE_MAP = {
    (-1, -1, 0): 10, (-1, -1, 1): 10, (-1, -1, 2): 11, (-1, -1, 3): 12,
    (-1,  0, 0):  9, (-1,  0, 1):  9, (-1,  0, 2): 11, (-1,  0, 3): 12,
    (-1,  1, 0):  8, (-1,  1, 1):  9, (-1,  1, 2): 11, (-1,  1, 3): 12,
    ( 0, -1, 0):  7, ( 0, -1, 1):  7, ( 0, -1, 2): 11,
    ( 0,  0, 0):  6, ( 0,  0, 1):  6, ( 0,  0, 2): 11, ( 0,  0, 3): 12,
    ( 0,  1, 0):  5, ( 0,  1, 1):  5, ( 0,  1, 2): 11,
    ( 1, -1, 0):  3, ( 1, -1, 1):  4, ( 1, -1, 2): 11,
    ( 1,  0, 0):  2, ( 1,  0, 1):  4, ( 1,  0, 2): 11,
    ( 1,  1, 0):  1, ( 1,  1, 1):  4, ( 1,  1, 2): 11,
}

STATE_NAMES = {
    1: "FULL_EXPANSION", 2: "EXPANSION_NEUTRAL", 3: "LATE_EXPANSION",
    4: "FRAGILE_EXPANSION", 5: "GOLDILOCKS", 6: "NEUTRAL",
    7: "CAUTION", 8: "EARLY_RECOVERY", 9: "CONTRACTION",
    10: "DEEP_CONTRACTION", 11: "STRESS_ELEVATED", 12: "FINANCIAL_CRISIS",
}


def calc_p_vote(permit_value):
    """P-Vote: PERMIT Level-Threshold. >1200=+1, <900=-1. 100% Match."""
    if pd.isna(permit_value):
        return 0
    if permit_value > 1200:
        return 1
    elif permit_value < 900:
        return -1
    return 0


def calc_u_vote(umcsent_value):
    """U-Vote: UMCSENT Level-Threshold. >70=+1, <60=-1. 99.91% Match."""
    if pd.isna(umcsent_value):
        return 0
    if umcsent_value > 70:
        return 1
    elif umcsent_value < 60:
        return -1
    return 0


def calc_i_vote(icsa_value):
    """I-Vote: ICSA Level-Threshold INVERTIERT. <250k=+1, >350k=-1. 99.95% Match."""
    if pd.isna(icsa_value):
        return 0
    if icsa_value < 250000:
        return 1
    elif icsa_value > 350000:
        return -1
    return 0


def calc_ip_vote(indpro_pct10):
    """IP-Vote: INDPRO 10M pct_change. >0.015=+1, <-0.005=-1. 97.72% Match."""
    if pd.isna(indpro_pct10):
        return 0
    if indpro_pct10 > 0.015:
        return 1
    elif indpro_pct10 < -0.005:
        return -1
    return 0


def calc_stress_score(vix, hy_spread, hy_threshold, nfci):
    """Stress Score: 0-3, sum of binary indicators. V55 Spec Kap. 6."""
    stress = 0
    active = []
    if pd.notna(vix) and vix > 30.0:
        stress += 1
        active.append("VIX")
    if pd.notna(hy_spread) and pd.notna(hy_threshold) and hy_spread > hy_threshold:
        stress += 1
        active.append("HY")
    if pd.notna(nfci) and nfci > 0.5:
        stress += 1
        active.append("NFCI")
    detail = "+".join(active) if active else "NONE"
    return stress, detail


def calc_state_confidence(growth_signal, liq_dir):
    """State Confidence from Ground Truth: alignment of Growth and Liq direction."""
    if growth_signal != 0 and liq_dir != 0:
        if growth_signal == liq_dir:
            return 85  # full alignment
        else:
            return 75  # misaligned
    else:
        return 70  # one or both neutral


def calc_macro_state(growth_signal, liq_dir, stress_score):
    """Map (Growth, Liq, Stress) to State 1-12. V55 Spec Kap. 7.4."""
    s = min(stress_score, 3)
    if s == 3:
        return 12, STATE_NAMES[12]

    key = (int(growth_signal), int(liq_dir), int(s))
    state_num = MACRO_STATE_MAP.get(key)
    if state_num is None:
        log.warning(f"  State Map miss: G={growth_signal} L={liq_dir} S={s} — defaulting to State 6")
        state_num = 6
    return state_num, STATE_NAMES[state_num]


def _fred_lookup(series, d):
    """Get value from daily-ffilled FRED series, with fallback to nearest prior date."""
    if series is None or series.empty:
        return np.nan
    val = series.get(d, np.nan)
    if pd.isna(val):
        prior = series[series.index <= d]
        if not prior.empty:
            val = prior.iloc[-1]
    return val


def calc_calc_macro_state(fred_data, k16_df, howell_df, start_date, end_date):
    """
    Calculate CALC_Macro_State for new dates.

    Inputs:
        fred_data: FRED series (PERMIT, UMCSENT, ICSA, INDPRO, VIXCLS, BAMLH0A0HYM2, NFCI)
        k16_df: DATA_K16_K17 results (for Liq_Dir_Confirmed, Liq_Detail)
        howell_df: CYCLES_Howell results (for Howell_Phase)
        start_date, end_date: calculation range
    """
    # INDPRO 10-month pct_change (210 trading days on daily-ffilled series)
    indpro_pct10 = fred_data["INDPRO"].pct_change(210)

    # Build lookup dicts from Stufe 3 output
    k16_lookup = {}
    if not k16_df.empty:
        for _, r in k16_df.iterrows():
            d = r["Date"]
            key = d.date() if hasattr(d, "date") else d
            k16_lookup[key] = {
                "Liq_Dir_Confirmed": int(r["Liq_Dir_Confirmed"]),
                "Liq_Detail": r["Liq_Detail"],
            }

    howell_lookup = {}
    if not howell_df.empty:
        for _, r in howell_df.iterrows():
            d = r["Date"]
            key = d.date() if hasattr(d, "date") else d
            howell_lookup[key] = int(r["Howell_Phase"])

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rows = []
    for d in dates:
        d_date = d.date()

        # Growth Votes (forward-filled FRED)
        permit_val = _fred_lookup(fred_data["PERMIT"], d)
        umcsent_val = _fred_lookup(fred_data["UMCSENT"], d)
        icsa_val = _fred_lookup(fred_data["ICSA"], d)
        ip_pct = _fred_lookup(indpro_pct10, d)

        p = calc_p_vote(permit_val)
        u = calc_u_vote(umcsent_val)
        i = calc_i_vote(icsa_val)
        ip = calc_ip_vote(ip_pct)

        # Growth Composite & Signal
        growth_composite = p + u + i + ip
        growth_signal = int(np.sign(growth_composite)) if growth_composite != 0 else 0
        growth_detail = f"P:{p}|U:{u}|I:{i}|IP:{ip}={growth_composite}"

        # Liq_Dir_Confirmed from Stufe 3
        k16_data = k16_lookup.get(d_date, {})
        liq_dir = k16_data.get("Liq_Dir_Confirmed", 0)
        liq_detail = k16_data.get("Liq_Detail", "")

        # Stress Score
        vix_val = _fred_lookup(fred_data["VIXCLS"], d)
        hy_val = _fred_lookup(fred_data["BAMLH0A0HYM2"], d)
        nfci_val = _fred_lookup(fred_data["NFCI"], d)

        # HY Threshold ADAPTIVE: 5.0 when Growth=+1, else 7.0
        hy_threshold = 5.0 if growth_signal == 1 else 7.0

        stress_score, stress_detail = calc_stress_score(vix_val, hy_val, hy_threshold, nfci_val)

        # Macro State
        state_num, state_name = calc_macro_state(growth_signal, liq_dir, stress_score)

        # State Confidence
        state_confidence = calc_state_confidence(growth_signal, liq_dir)

        # Howell Phase (reference column)
        phase = howell_lookup.get(d_date, 4)

        rows.append({
            "Date": d,
            "Macro_State_Num": state_num,
            "Macro_State_Name": state_name,
            "Growth_Signal": growth_signal,
            "Growth_Detail": growth_detail,
            "Liq_Direction": liq_dir,
            "Liq_Detail": liq_detail,
            "Stress_Score": stress_score,
            "Stress_Detail": stress_detail,
            "VIX": vix_val,
            "HY_Spread": hy_val,
            "HY_Threshold": hy_threshold,
            "NFCI": nfci_val,
            "Howell_Phase": phase,
            "State_Confidence": state_confidence,
            "P_Vote": p,
            "U_Vote": u,
            "I_Vote": i,
            "IP_Vote": ip,
        })

    return pd.DataFrame(rows)


def write_calc_macro_state(sheet, df):
    """Write CALC_Macro_State rows to sheet (newest first, row 3)."""
    ws = sheet.worksheet("CALC_Macro_State")

    new_dates_set = set()
    rows_to_write = []
    for _, row in df.sort_values("Date", ascending=False).iterrows():
        d = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        new_dates_set.add(d)
        rows_to_write.append([
            d,
            int(row["Growth_Signal"]),
            row["Growth_Detail"],
            int(row["Liq_Direction"]),
            row["Liq_Detail"],
            int(row["Stress_Score"]),
            row["Stress_Detail"],
            fmt(row["HY_Threshold"], 1),
            int(row["Macro_State_Num"]),
            row["Macro_State_Name"],
            int(row["State_Confidence"]),
            "",  # Old_Regime_Num (legacy, empty)
            "",  # Old_Regime_Name (legacy, empty)
            int(row["Howell_Phase"]),
            fmt(row["VIX"], 2),
            fmt(row["HY_Spread"], 2),
            "",  # Old_State_V25 (legacy, empty)
        ])

    if not rows_to_write:
        log.warning("CALC_Macro_State: No rows to write.")
        return

    num = len(rows_to_write)
    log.info(f"CALC_Macro_State: Writing {num} rows...")

    # Delete overlapping rows
    existing_dates = ws.col_values(1)[2:]
    rows_to_delete = []
    for i, d in enumerate(existing_dates):
        if d in new_dates_set:
            rows_to_delete.append(i + 3)

    for row_idx in sorted(rows_to_delete, reverse=True):
        ws.delete_rows(row_idx)
        log.info(f"  Deleted existing row {row_idx} (overlap)")

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

    # --- STUFE 0: DATA_Prices (V16 Asset-Preise) ---
    # Laeuft IMMER (auch wenn Macro State schon aktuell ist),
    # weil Preise taeglich geschrieben werden muessen.
    # write_prices() hat eigenen Duplikat-Check (skippt wenn Datum schon existiert).
    log.info("=" * 60)
    log.info("STUFE 0: DATA_Prices (27 Assets)")
    log.info("=" * 60)
    try:
        from fetchers import FMPFetcher, YFinanceFetcher, V16PriceFetcher
        from writers import V16SheetWriter

        fmp_key = os.environ.get('FMP_API_KEY', os.environ.get('EODHD_API_KEY', ''))
        fmp = FMPFetcher(fmp_key)
        yf = YFinanceFetcher()
        price_fetcher = V16PriceFetcher(fmp, yf)

        log.info("  Fetching 27 asset prices...")
        v16_prices = price_fetcher.fetch_all()
        ok_count = sum(1 for v in v16_prices.values() if v is not None)
        log.info(f"  Prices fetched: {ok_count}/27")

        if ok_count >= 20:  # Mindestens 20/27 Preise muessen da sein
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
            writer = V16SheetWriter(PRODUCTION_SHEET_ID, creds_path)
            success = writer.write_prices(v16_prices, end_date)
            if success:
                log.info(f"  DATA_Prices: {end_date} geschrieben")
            else:
                log.warning("  DATA_Prices: write_prices returned False")
        else:
            log.warning(f"  DATA_Prices: Nur {ok_count}/27 Preise — SKIP (brauche mindestens 20)")

    except Exception as e:
        log.error(f"  STUFE 0 FAILED (non-fatal): {e}")
        log.info("  Pipeline continues without price update")

    log.info("")

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

    # --- STUFE 3: DATA_K16_K17 ---
    log.info("STUFE 3: DATA_K16_K17")
    k16_df = calc_data_k16_k17(fred_data, sheet, howell_df, start_date, end_date)
    log.info(f"  {len(k16_df)} rows calculated")

    if not k16_df.empty:
        r = k16_df.iloc[-1]
        log.info(f"  Latest: {r['Date'].date() if hasattr(r['Date'], 'date') else r['Date']} "
                 f"K16={r['K16_Vote']} K17={r['K17_Vote']} K4={r['K4_Vote']} HV={r['Howell_Vote']} "
                 f"VS={r['Vote_Sum']} Conf={r['Liq_Dir_Confirmed']}")

    log.info("Writing DATA_K16_K17...")
    write_data_k16_k17(sheet, k16_df)

    log.info("=" * 60)
    log.info("Stufe 3 complete")
    log.info("=" * 60)

    # --- STUFE 4: CALC_Macro_State ---
    log.info("STUFE 4: CALC_Macro_State")
    macro_df = calc_calc_macro_state(fred_data, k16_df, howell_df, start_date, end_date)
    log.info(f"  {len(macro_df)} rows calculated")

    if not macro_df.empty:
        r = macro_df.iloc[-1]
        log.info(f"  Latest: {r['Date'].date() if hasattr(r['Date'], 'date') else r['Date']} "
                 f"State={r['Macro_State_Num']} ({r['Macro_State_Name']}) "
                 f"GS={r['Growth_Signal']} Liq={r['Liq_Direction']} "
                 f"Stress={r['Stress_Score']} SC={r['State_Confidence']}")
        log.info(f"  Growth: {r['Growth_Detail']}")
        log.info(f"  Stress: {r['Stress_Detail']} (VIX={r['VIX']:.2f} HY={r['HY_Spread']:.2f} "
                 f"HY_Thr={r['HY_Threshold']:.1f})")

    log.info("Writing CALC_Macro_State...")
    write_calc_macro_state(sheet, macro_df)

    log.info("=" * 60)
    log.info("Stufe 4 complete — V55 Data Collector KOMPLETT")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
