"""
L8 Seasonality Collector — step_0g_l8_seasonality/main.py
Baldur Creek Capital — richie-hedge-pipeline

6 Signals:
  1. HOWELL_CYCLE_POS     (weight 0.30) — 65-month cycle, trough Oct 2022
  2. MONTHLY_SEASONAL_SPY (weight 0.25) — 20Y avg monthly return SPY
  3. PRESIDENTIAL_CYCLE   (weight 0.20) — 4Y cycle, intra-year scoring
  4. OPEX_PROXIMITY       (weight 0.10) — distance to monthly OPEX (3rd Friday)
  5. QUARTER_END_EFFECT   (weight 0.10) — Q-end selling + Q-start buying
  6. EARNINGS_SEASON      (weight 0.05) — earnings season proximity

Writes to Data Warehouse:
  RAW_MACRO  : Rows 9-14  (L8, 6 indicators)
  SCORES     : Row 9      (L8 Seasonality composite)
  DASHBOARD  : Row 24     (L8 score + signal)
"""

import os
import sys
import math
import logging
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS — DO NOT CHANGE WITHOUT DISCUSSION
# ─────────────────────────────────────────────

# Howell Cycle
HOWELL_TROUGH_DATE     = date(2022, 10, 1)   # Oct 2022 trough (validated)
HOWELL_CYCLE_MONTHS    = 65                   # 65-month cycle length
HOWELL_PHASE4_ENTRY    = date(2026, 9, 16)    # Phase 4 entry date

# Layer weights
WEIGHTS = {
    "HOWELL_CYCLE_POS":     0.30,
    "MONTHLY_SEASONAL_SPY": 0.25,
    "PRESIDENTIAL_CYCLE":   0.20,
    "OPEX_PROXIMITY":       0.10,
    "QUARTER_END_EFFECT":   0.10,
    "EARNINGS_SEASON":      0.05,
}

# Asymmetry multiplier (bearish)
BEARISH_MULTIPLIER = 1.3

# Minimum indicators for valid composite
MIN_VALID = 4

# Regime weight from CONFIG (Risk-On default)
REGIME_WEIGHT_RISK_ON  = 0.10
REGIME_WEIGHT_RISK_OFF = 0.05
REGIME_WEIGHT_DD       = 0.05

# Google Sheets
WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# RAW_MACRO rows for L8 (1-indexed)
RAW_MACRO_ROWS = {
    "PRESIDENTIAL_CYCLE":   9,
    "MONTHLY_SEASONAL_SPY": 10,
    "OPEX_PROXIMITY":       11,
    "QUARTER_END_EFFECT":   12,
    "HOWELL_CYCLE_POS":     13,
    "EARNINGS_SEASON":      14,
}

# SCORES row for L8
SCORES_ROW = 9

# DASHBOARD row for L8
DASHBOARD_ROW = 24

# ─────────────────────────────────────────────
# GOOGLE SHEETS CONNECTION
# ─────────────────────────────────────────────

def get_gspread_client():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return gspread.authorize(creds)


def open_warehouse(client):
    return client.open_by_key(WAREHOUSE_SHEET_ID)


# ─────────────────────────────────────────────
# SIGNAL 1 — HOWELL CYCLE (weight 0.30)
# ─────────────────────────────────────────────

def calc_howell_cycle(today: date) -> dict:
    """
    65-month sinusoidal cycle anchored to Oct 2022 trough.
    Trough = cycle bottom (score 0 = bearish headwind).
    Peak = cycle top at ~32.5 months after trough.
    
    Score 0-10:
      0   = trough (maximum bearish)
      10  = peak   (maximum bullish tailwind)
    
    Phase context logged for reference.
    """
    try:
        months_elapsed = (
            (today.year - HOWELL_TROUGH_DATE.year) * 12
            + (today.month - HOWELL_TROUGH_DATE.month)
            + (today.day - HOWELL_TROUGH_DATE.day) / 30.0
        )

        cycle_position = months_elapsed % HOWELL_CYCLE_MONTHS
        # sin starts at -1 (trough), peaks at +1 halfway
        angle_rad = 2 * math.pi * (cycle_position / HOWELL_CYCLE_MONTHS) - (math.pi / 2)
        sin_val = math.sin(angle_rad)  # -1 to +1

        # Normalize to 0-10
        score = (sin_val + 1.0) / 2.0 * 10.0
        score = max(0.0, min(10.0, score))

        # Determine phase label
        pct = cycle_position / HOWELL_CYCLE_MONTHS
        if pct < 0.25:
            phase_label = "Phase 1 (Early Expansion)"
        elif pct < 0.50:
            phase_label = "Phase 2 (Late Expansion)"
        elif pct < 0.75:
            phase_label = "Phase 3 (Early Contraction)"
        else:
            phase_label = "Phase 4 (Late Contraction)"

        log.info(
            f"HOWELL_CYCLE: months_elapsed={months_elapsed:.1f}, "
            f"cycle_pos={cycle_position:.1f}/{HOWELL_CYCLE_MONTHS}, "
            f"sin={sin_val:.4f}, Score={score:.2f}, Phase={phase_label}"
        )

        return {
            "value": round(cycle_position, 2),
            "score": round(score, 4),
            "raw_value": round(cycle_position, 2),
            "age_days": 0,
            "source": "Calculated",
            "tier": "T2",
            "unit": "month",
            "meta": phase_label,
        }

    except Exception as e:
        log.error(f"HOWELL_CYCLE error: {e}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 2 — MONTHLY SEASONAL SPY (weight 0.25)
# ─────────────────────────────────────────────

def calc_monthly_seasonal_spy(today: date) -> dict:
    """
    Downloads 20Y of SPY monthly returns (yfinance).
    Computes average return for the current calendar month.
    Normalizes: -3% avg = Score 0, +3% avg = Score 10.
    """
    try:
        start = (today - timedelta(days=365 * 21)).strftime("%Y-%m-%d")
        end   = today.strftime("%Y-%m-%d")

        spy = yf.download("SPY", start=start, end=end, interval="1mo",
                          progress=False, auto_adjust=True)

        if spy.empty or len(spy) < 12:
            log.warning("MONTHLY_SEASONAL_SPY: insufficient data")
            return None

        spy["monthly_return"] = spy["Close"].pct_change() * 100
        spy = spy.dropna(subset=["monthly_return"])
        spy.index = pd.to_datetime(spy.index)
        spy["month"] = spy.index.month

        current_month = today.month
        monthly_avg = spy.groupby("month")["monthly_return"].mean()

        if current_month not in monthly_avg.index:
            log.warning(f"MONTHLY_SEASONAL_SPY: month {current_month} not in history")
            return None

        avg_return = float(monthly_avg[current_month])

        # Normalize: -3% → 0, +3% → 10
        score = (avg_return + 3.0) / 6.0 * 10.0
        score = max(0.0, min(10.0, score))

        month_name = today.strftime("%B")
        log.info(
            f"MONTHLY_SEASONAL_SPY: month={month_name}, "
            f"avg_20Y={avg_return:.3f}%, Score={score:.2f}"
        )

        return {
            "value": round(avg_return, 4),
            "score": round(score, 4),
            "raw_value": round(avg_return, 4),
            "age_days": 0,
            "source": "Calculated",
            "tier": "T2",
            "unit": "% avg",
        }

    except Exception as e:
        log.error(f"MONTHLY_SEASONAL_SPY error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 3 — PRESIDENTIAL CYCLE (weight 0.20)
# ─────────────────────────────────────────────

def calc_presidential_cycle(today: date) -> dict:
    """
    4-year US Presidential cycle.
    Base election years: 2000, 2004, 2008, 2012, 2016, 2020, 2024.
    
    Year in cycle:
      Year 1 (post-election): moderate bullish (7/10)
      Year 2 (midterm H1):    bearish (3/10), H2 rally (6/10)
      Year 3 (pre-election):  strongest (8/10)
      Year 4 (election):      moderate (6/10)
    
    Intra-year scoring for Year 2:
      Jan-Sep = bearish phase (3.0)
      Oct-Dec = midterm rally phase (6.5)
    """
    try:
        election_years = list(range(2000, 2040, 4))  # 2000, 2004, ... (election = year 0)
        # Find position in cycle
        # Year 1 = election_year + 1, Year 2 = election_year + 2, etc.
        cycle_year = None
        for ey in sorted(election_years, reverse=True):
            if today.year >= ey:
                cycle_year = today.year - ey  # 0=election, 1=post, 2=midterm, 3=pre
                break

        if cycle_year is None:
            log.warning("PRESIDENTIAL_CYCLE: could not determine cycle year")
            return None

        cycle_year_label = cycle_year % 4  # 0,1,2,3

        # Score map
        if cycle_year_label == 0:
            # Election year — moderate bullish
            score = 6.0
            label = "Year 4 (Election)"
        elif cycle_year_label == 1:
            # Post-election — moderate bullish
            score = 7.0
            label = "Year 1 (Post-Election)"
        elif cycle_year_label == 2:
            # Midterm — intra-year split
            if today.month <= 9:
                score = 3.0
                label = "Year 2 (Midterm H1 — Bearish)"
            else:
                score = 6.5
                label = "Year 2 (Midterm H2 — Rally)"
        else:
            # Pre-election — strongest year
            score = 8.0
            label = "Year 3 (Pre-Election)"

        log.info(
            f"PRESIDENTIAL_CYCLE: year={today.year}, cycle_year={cycle_year_label}, "
            f"label={label}, Score={score:.2f}"
        )

        return {
            "value": round(score, 4),
            "score": round(score, 4),
            "raw_value": cycle_year_label,
            "age_days": 0,
            "source": "Calendar",
            "tier": "T1",
            "unit": "Year 2 (Midterm)",
            "meta": label,
        }

    except Exception as e:
        log.error(f"PRESIDENTIAL_CYCLE error: {e}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 4 — OPEX PROXIMITY (weight 0.10)
# ─────────────────────────────────────────────

def get_third_friday(year: int, month: int) -> date:
    """Returns the third Friday of the given month."""
    first_day = date(year, month, 1)
    # weekday(): Monday=0, Friday=4
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday


def calc_opex_proximity(today: date) -> dict:
    """
    Monthly OPEX = 3rd Friday of each month.
    Score reflects proximity to OPEX (elevated vol / pin risk):
      0-2 days before OPEX  → Score 9.0 (maximum regime risk)
      3-5 days before OPEX  → Score 6.5
      6-10 days before OPEX → Score 4.0
      >10 days              → Score 2.0 (low OPEX influence)
      Post-OPEX (0-3 days after) → Score 5.0 (vol collapse, repositioning)
    """
    try:
        this_opex = get_third_friday(today.year, today.month)

        # Check next month's OPEX if this month's already passed
        if today > this_opex + timedelta(days=3):
            if today.month == 12:
                next_opex = get_third_friday(today.year + 1, 1)
            else:
                next_opex = get_third_friday(today.year, today.month + 1)
            relevant_opex = next_opex
        else:
            relevant_opex = this_opex

        days_delta = (relevant_opex - today).days  # positive = before, negative = after

        if -3 <= days_delta < 0:
            # Just past OPEX — vol collapse / repositioning
            score = 5.0
            proximity_label = f"Post-OPEX ({abs(days_delta)}d after)"
        elif days_delta == 0:
            score = 9.0
            proximity_label = "OPEX Day"
        elif 1 <= days_delta <= 2:
            score = 9.0
            proximity_label = f"Pre-OPEX ({days_delta}d before) — HIGH"
        elif 3 <= days_delta <= 5:
            score = 6.5
            proximity_label = f"Pre-OPEX ({days_delta}d before) — MODERATE"
        elif 6 <= days_delta <= 10:
            score = 4.0
            proximity_label = f"Pre-OPEX ({days_delta}d before) — LOW"
        else:
            score = 2.0
            proximity_label = f"Pre-OPEX ({days_delta}d before) — MINIMAL"

        log.info(
            f"OPEX_PROXIMITY: next_opex={relevant_opex}, "
            f"days_to_opex={days_delta}, label={proximity_label}, Score={score:.2f}"
        )

        return {
            "value": days_delta,
            "score": round(score, 4),
            "raw_value": days_delta,
            "age_days": 0,
            "source": "Calendar",
            "tier": "T1",
            "unit": "days",
            "meta": proximity_label,
        }

    except Exception as e:
        log.error(f"OPEX_PROXIMITY error: {e}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 5 — QUARTER-END EFFECT (weight 0.10)
# ─────────────────────────────────────────────

def calc_quarter_end_effect(today: date) -> dict:
    """
    Quarter-end rebalancing pressure (last 5 trading days of quarter)
    and quarter-start inflow effect (first 3 trading days of quarter).
    
    Quarter-end months: March, June, September, December.
    
    Score:
      Last 1-2 days of quarter  → Score 8.0 (max rebalancing pressure, bearish)
      Last 3-5 days of quarter  → Score 6.5
      First 1-3 days of quarter → Score 3.5 (fresh capital inflow, bullish)
      Otherwise                 → Score 5.0 (neutral)
    """
    try:
        # Quarter-end months and their last day
        quarter_ends = {
            3:  date(today.year, 3, 31),
            6:  date(today.year, 6, 30),
            9:  date(today.year, 9, 30),
            12: date(today.year, 12, 31),
        }

        # Quarter-start months and first day
        quarter_starts = {
            1:  date(today.year, 1, 1),
            4:  date(today.year, 4, 1),
            7:  date(today.year, 7, 1),
            10: date(today.year, 10, 1),
        }

        # Find nearest quarter end
        nearest_qend = None
        min_dist = 999
        for qe in quarter_ends.values():
            dist = abs((today - qe).days)
            if dist < min_dist:
                min_dist = dist
                nearest_qend = qe

        days_to_qend   = (nearest_qend - today).days   # negative = past
        days_from_qend = (today - nearest_qend).days   # positive = after

        # Find nearest quarter start
        nearest_qstart = None
        min_dist2 = 999
        for qs in quarter_starts.values():
            dist = abs((today - qs).days)
            if dist < min_dist2:
                min_dist2 = dist
                nearest_qstart = qs

        days_from_qstart = (today - nearest_qstart).days  # positive = after start

        # Score logic
        if 0 <= days_to_qend <= 2:
            score = 8.0
            label = f"Q-End in {days_to_qend}d — MAX REBALANCING"
        elif 3 <= days_to_qend <= 5:
            score = 6.5
            label = f"Q-End in {days_to_qend}d — ELEVATED REBALANCING"
        elif 0 <= days_from_qstart <= 3:
            score = 3.5
            label = f"Q-Start +{days_from_qstart}d — FRESH CAPITAL INFLOW"
        else:
            score = 5.0
            label = "Mid-Quarter — NEUTRAL"

        log.info(
            f"QUARTER_END_EFFECT: nearest_qend={nearest_qend}, "
            f"days_to_qend={days_to_qend}, label={label}, Score={score:.2f}"
        )

        return {
            "value": days_to_qend if days_to_qend >= 0 else -days_from_qend,
            "score": round(score, 4),
            "raw_value": days_to_qend if days_to_qend >= 0 else -days_from_qend,
            "age_days": 0,
            "source": "Calendar",
            "tier": "T1",
            "unit": "days",
            "meta": label,
        }

    except Exception as e:
        log.error(f"QUARTER_END_EFFECT error: {e}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 6 — EARNINGS SEASON (weight 0.05)
# ─────────────────────────────────────────────

def calc_earnings_season(today: date) -> dict:
    """
    Earnings season windows (approximate, US large-cap):
      Q4 (Jan 15 - Feb 28):  peak uncertainty — Score 6.5
      Q1 (Apr 15 - May 31):  peak uncertainty — Score 6.5
      Q2 (Jul 15 - Aug 31):  peak uncertainty — Score 6.5
      Q3 (Oct 15 - Nov 30):  peak uncertainty — Score 6.5
      Post-season (within 14d after end): Score 4.5 (uncertainty abating)
      Pre-season (within 7d before start): Score 5.5 (building uncertainty)
      Off-season: Score 3.0 (low earnings-driven vol)
    
    Raw value: 1.0 = in season, 0.0 = out of season (for sheet TRUE/FALSE).
    """
    try:
        year = today.year

        # Earnings season windows: (start, end)
        seasons = [
            (date(year, 1, 15),  date(year, 2, 28)),   # Q4
            (date(year, 4, 15),  date(year, 5, 31)),   # Q1
            (date(year, 7, 15),  date(year, 8, 31)),   # Q2
            (date(year, 10, 15), date(year, 11, 30)),  # Q3
        ]

        in_season     = False
        pre_season    = False
        post_season   = False
        days_to_start = None
        days_from_end = None

        for (start, end) in seasons:
            if start <= today <= end:
                in_season = True
                break
            # Pre-season: within 7 days before start
            delta_to_start = (start - today).days
            if 0 < delta_to_start <= 7:
                pre_season = True
                days_to_start = delta_to_start
                break
            # Post-season: within 14 days after end
            delta_from_end = (today - end).days
            if 0 < delta_from_end <= 14:
                post_season = True
                days_from_end = delta_from_end
                break

        if in_season:
            score = 6.5
            label = "IN EARNINGS SEASON"
            raw_value = 1.0
        elif pre_season:
            score = 5.5
            label = f"PRE-SEASON ({days_to_start}d to start)"
            raw_value = 0.0
        elif post_season:
            score = 4.5
            label = f"POST-SEASON ({days_from_end}d after end)"
            raw_value = 0.0
        else:
            score = 3.0
            label = "OFF-SEASON"
            raw_value = 0.0

        log.info(
            f"EARNINGS_SEASON: label={label}, Score={score:.2f}"
        )

        return {
            "value": raw_value,
            "score": round(score, 4),
            "raw_value": raw_value,
            "age_days": 0,
            "source": "Calendar",
            "tier": "T1",
            "unit": "TRUE/FALSE",
            "meta": label,
        }

    except Exception as e:
        log.error(f"EARNINGS_SEASON error: {e}")
        return None


# ─────────────────────────────────────────────
# COMPOSITE CALCULATION
# ─────────────────────────────────────────────

PHASE_MAP = [
    (0.0,  1.5,  "Minimal Tailwind",  "Bearish"),
    (1.5,  3.0,  "Weak Tailwind",     "Bearish"),
    (3.0,  4.5,  "Slight Headwind",   "Neutral"),
    (4.5,  5.5,  "Neutral",           "Neutral"),
    (5.5,  7.0,  "Slight Tailwind",   "Bullish"),
    (7.0,  8.5,  "Strong Tailwind",   "Bullish"),
    (8.5, 10.0,  "Maximum Tailwind",  "Extreme"),
]


def get_phase(score: float) -> tuple:
    for lo, hi, phase, signal in PHASE_MAP:
        if lo <= score <= hi:
            return phase, signal
    return "Neutral", "Neutral"


def calc_composite(results: dict) -> dict:
    """
    Weighted composite from all valid signals.
    Applies bearish asymmetry (x1.3) if signal is Bearish.
    """
    weighted_sum  = 0.0
    weight_total  = 0.0
    valid_count   = 0
    freshness_sum = 0.0

    for indicator, weight in WEIGHTS.items():
        r = results.get(indicator)
        if r is None:
            log.warning(f"COMPOSITE: {indicator} missing — skipping")
            continue
        weighted_sum  += r["score"] * weight
        weight_total  += weight
        valid_count   += 1
        freshness_sum += max(0.0, 10.0 - r["age_days"] * 0.5)

    if valid_count < MIN_VALID:
        log.error(f"COMPOSITE: only {valid_count}/{len(WEIGHTS)} valid — aborting")
        return None

    # Normalize by actual weights used
    score_raw = weighted_sum / weight_total * 10.0 / 10.0
    # (weighted_sum is already in 0-10 space since weights sum to 1.0)
    score_raw = weighted_sum / weight_total
    score_raw = max(0.0, min(10.0, score_raw))

    freshness = freshness_sum / valid_count

    phase, signal = get_phase(score_raw)

    # Asymmetry adjustment
    if signal in ("Bearish", "Extreme"):
        asymmetry_adj = min(13.0, score_raw * BEARISH_MULTIPLIER)
    else:
        asymmetry_adj = score_raw

    log.info(
        f"COMPOSITE: score={score_raw:.4f}, phase={phase}, signal={signal}, "
        f"valid={valid_count}/{len(WEIGHTS)}, freshness={freshness:.1f}/10, "
        f"asymmetry_adj={asymmetry_adj:.4f}"
    )

    return {
        "score_raw":     round(score_raw, 4),
        "signal":        signal,
        "phase":         phase,
        "freshness":     round(freshness, 2),
        "asymmetry_adj": round(asymmetry_adj, 4),
        "valid_count":   valid_count,
        "regime_weight": REGIME_WEIGHT_RISK_ON,
    }


# ─────────────────────────────────────────────
# GOOGLE SHEETS WRITE
# ─────────────────────────────────────────────

def write_raw_macro(ws_raw_macro, today: date, results: dict):
    """
    Writes 6 L8 indicator rows to RAW_MACRO tab (Rows 9-14).
    Columns: A=DATE B=INDICATOR C=LAYER D=VALUE E=PREV_7D F=PREV_30D
             G=DATA_AGE_DAYS H=SOURCE I=RELIABILITY_TIER J=UNIT
    """
    today_str = today.strftime("%Y-%m-%d")

    indicator_order = [
        "PRESIDENTIAL_CYCLE",
        "MONTHLY_SEASONAL_SPY",
        "OPEX_PROXIMITY",
        "QUARTER_END_EFFECT",
        "HOWELL_CYCLE_POS",
        "EARNINGS_SEASON",
    ]

    for indicator in indicator_order:
        row_num = RAW_MACRO_ROWS[indicator]
        r = results.get(indicator)

        if r is None:
            val_str = "ERROR"
            age_str = "—"
            src_str = "—"
            tier_str = "—"
            unit_str = "—"
        else:
            val_str  = str(round(r["value"], 6)) if isinstance(r["value"], float) else str(r["value"])
            age_str  = str(r["age_days"])
            src_str  = r["source"]
            tier_str = r["tier"]
            unit_str = r["unit"]

        # Read existing PREV_7D and PREV_30D from sheet (columns E, F = index 4, 5)
        try:
            existing_row = ws_raw_macro.row_values(row_num)
            prev_7d  = existing_row[4] if len(existing_row) > 4 else "—"
            prev_30d = existing_row[5] if len(existing_row) > 5 else "—"
        except Exception:
            prev_7d  = "—"
            prev_30d = "—"

        row_data = [
            today_str,   # A: DATE
            indicator,   # B: INDICATOR
            "L8",        # C: LAYER
            val_str,     # D: VALUE
            prev_7d,     # E: PREV_7D (preserve)
            prev_30d,    # F: PREV_30D (preserve)
            age_str,     # G: DATA_AGE_DAYS
            src_str,     # H: SOURCE
            tier_str,    # I: RELIABILITY_TIER
            unit_str,    # J: UNIT
        ]

        ws_raw_macro.update(
            range_name=f"A{row_num}:J{row_num}",
            values=[row_data],
        )
        log.info(f"RAW_MACRO Row {row_num} written: {indicator} = {val_str}")


def write_scores(ws_scores, today: date, composite: dict, results: dict):
    """
    Writes L8 composite to SCORES tab Row 9.
    5/13 columns available from day 1:
    A=LAYER B=SCORE_RAW C=SCORE_7D D=SCORE_30D E=PERCENTILE F=DIRECTION
    G=SPEED H=SIGNAL I=FRESHNESS J=DECAY_ADJ K=ASYMMETRY_ADJ
    L=REGIME_WEIGHT M=HISTORICAL_ANALOG
    """
    today_str = today.strftime("%Y-%m-%d")

    row_data = [
        f"L8 Seasonality ({today_str})",  # A: LAYER
        composite["score_raw"],            # B: SCORE_RAW
        "—",                               # C: SCORE_7D (needs history)
        "—",                               # D: SCORE_30D (needs history)
        "—",                               # E: PERCENTILE (needs history)
        "—",                               # F: DIRECTION (needs history)
        "—",                               # G: SPEED (needs history)
        composite["signal"],               # H: SIGNAL
        composite["freshness"],            # I: FRESHNESS
        "—",                               # J: DECAY_ADJ (needs history)
        composite["asymmetry_adj"],        # K: ASYMMETRY_ADJ
        composite["regime_weight"],        # L: REGIME_WEIGHT
        "—",                               # M: HISTORICAL_ANALOG (needs history)
    ]

    ws_scores.update(
        range_name=f"A{SCORES_ROW}:M{SCORES_ROW}",
        values=[row_data],
    )
    log.info(
        f"SCORES Row {SCORES_ROW} written: "
        f"L8={composite['score_raw']:.4f} | {composite['phase']} | {composite['signal']}"
    )


def write_dashboard(ws_dashboard, today: date, composite: dict):
    """
    Writes L8 score + signal to DASHBOARD tab Row 24.
    """
    today_str = today.strftime("%Y-%m-%d")

    row_data = [
        f"L8 Seasonality",          # A
        composite["score_raw"],     # B: Score
        composite["signal"],        # C: Signal
        composite["phase"],         # D: Phase
        composite["freshness"],     # E: Freshness
        today_str,                  # F: Last Update
        composite["valid_count"],   # G: Sources OK
        f"{composite['valid_count']}/{len(WEIGHTS)}",  # H: Coverage
    ]

    ws_dashboard.update(
        range_name=f"A{DASHBOARD_ROW}:H{DASHBOARD_ROW}",
        values=[row_data],
    )
    log.info(
        f"DASHBOARD Row {DASHBOARD_ROW} written: "
        f"L8={composite['score_raw']:.4f} | {composite['signal']}"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("L8 SEASONALITY COLLECTOR — START")
    log.info("=" * 60)

    today = date.today()
    log.info(f"Date: {today}")

    # ── Calculate all signals ──
    results = {}

    log.info("--- Signal 1: Howell Cycle ---")
    results["HOWELL_CYCLE_POS"] = calc_howell_cycle(today)

    log.info("--- Signal 2: Monthly Seasonal SPY ---")
    results["MONTHLY_SEASONAL_SPY"] = calc_monthly_seasonal_spy(today)

    log.info("--- Signal 3: Presidential Cycle ---")
    results["PRESIDENTIAL_CYCLE"] = calc_presidential_cycle(today)

    log.info("--- Signal 4: OPEX Proximity ---")
    results["OPEX_PROXIMITY"] = calc_opex_proximity(today)

    log.info("--- Signal 5: Quarter-End Effect ---")
    results["QUARTER_END_EFFECT"] = calc_quarter_end_effect(today)

    log.info("--- Signal 6: Earnings Season ---")
    results["EARNINGS_SEASON"] = calc_earnings_season(today)

    # ── Composite ──
    log.info("--- Composite Calculation ---")
    composite = calc_composite(results)

    if composite is None:
        log.error("FATAL: Composite calculation failed — not enough valid signals")
        sys.exit(1)

    log.info(
        f"COMPOSITE RESULT: {composite['score_raw']:.4f}/10 | "
        f"{composite['phase']} | {composite['signal']} | "
        f"{composite['valid_count']}/{len(WEIGHTS)} Sources | "
        f"Freshness {composite['freshness']:.1f}/10"
    )

    # ── Google Sheets ──
    log.info("--- Writing to Google Sheets ---")
    try:
        client    = get_gspread_client()
        warehouse = open_warehouse(client)

        ws_raw_macro  = warehouse.worksheet("RAW_MACRO")
        ws_scores     = warehouse.worksheet("SCORES")
        ws_dashboard  = warehouse.worksheet("DASHBOARD")

        write_raw_macro(ws_raw_macro, today, results)
        write_scores(ws_scores, today, composite, results)
        write_dashboard(ws_dashboard, today, composite)

        log.info("All sheets written successfully.")

    except Exception as e:
        log.error(f"Google Sheets write error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    log.info("=" * 60)
    log.info(
        f"L8 SEASONALITY COLLECTOR — DONE | "
        f"{composite['score_raw']:.2f}/10 | {composite['phase']} | "
        f"{composite['signal']} | {composite['valid_count']}/{len(WEIGHTS)} Sources"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
