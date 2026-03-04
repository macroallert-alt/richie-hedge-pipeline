"""
L7 Cross-Asset Collector — step_0f_l7_cross_asset/main.py
Baldur Creek Capital — richie-hedge-pipeline

7 Signals:
  1. BOND_EQUITY_CORR_60D   (weight 0.20) — Rolling 60d corr TLT/SPY > 0.3 = warning
  2. GOLD_DXY_BOTH_UP       (weight 0.15) — Gold 20d return + DXY 20d return both positive
  3. COPPER_VS_SPY_DIVERG   (weight 0.20) — Copper SMA50 falling + SPY rising = divergence
  4. CREDIT_VS_EQUITY       (weight 0.20) — HY spreads widening + SPY near high = warning
  5. REAL_YIELD_TREND       (weight 0.10) — TIPS 10Y 30d SMA rising = headwind
  6. YIELD_CURVE_10Y2Y      (weight 0.15) — 10Y-2Y steepener after inversion
  7. GOLD_RETURN_20D        (weight 0.00) — Raw value only, embedded in GOLD_DXY check

Writes to Data Warehouse:
  RAW_MARKET : Rows 28-34 (L7, 7 indicators)
  SCORES     : Row 8      (L7 Cross-Asset composite)
  DASHBOARD  : Row 23     (L7 score + signal)
"""

import os
import sys
import logging
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
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

# Signal weights (sum = 1.0, GOLD_RETURN_20D/DXY_RETURN_20D are raw-only rows)
WEIGHTS = {
    "BOND_EQUITY_CORR_60D": 0.20,
    "GOLD_DXY_BOTH_UP":     0.15,
    "COPPER_SPY_DIVERGENCE":0.20,
    "CREDIT_VS_EQUITY":     0.20,
    "REAL_YIELD_TREND":     0.10,
    "YIELD_CURVE_10Y2Y":    0.15,
}

# Asymmetry multiplier
BEARISH_MULTIPLIER = 1.3

# Minimum valid indicators for composite
MIN_VALID = 4

# Regime weights
REGIME_WEIGHT_RISK_ON  = 0.10
REGIME_WEIGHT_RISK_OFF = 0.05
REGIME_WEIGHT_DD       = 0.10

# Google Sheets
WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# RAW_MARKET rows for L7 (1-indexed, rows 28-34)
RAW_MARKET_ROWS = {
    "BOND_EQUITY_CORR_60D": 28,
    "GOLD_RETURN_20D":       29,
    "DXY_RETURN_20D":        30,
    "COPPER_SMA50_TREND":    31,
    "SPY_SMA50_TREND":       32,
    "REAL_YIELD_10Y_TREND":  33,
    "YIELD_CURVE_10Y2Y":     34,
}

# SCORES row for L7
SCORES_ROW = 8

# DASHBOARD row for L7
DASHBOARD_ROW = 23

# ─────────────────────────────────────────────
# GOOGLE SHEETS + FRED CONNECTION
# ─────────────────────────────────────────────

def get_gspread_client():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return gspread.authorize(creds)


def open_warehouse(client):
    return client.open_by_key(WAREHOUSE_SHEET_ID)


def get_fred_client():
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise ValueError("FRED_API_KEY not set in environment")
    return Fred(api_key=api_key)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def download_yf(ticker: str, days: int) -> pd.DataFrame:
    """Download yfinance daily data, flatten MultiIndex if present."""
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


# ─────────────────────────────────────────────
# SIGNAL 1 — BOND/EQUITY CORRELATION 60D (weight 0.20)
# ─────────────────────────────────────────────

def calc_bond_equity_corr(today: date) -> dict:
    """
    Rolling 60d correlation between TLT (bonds) and SPY (equities).
    Normal regime: negative correlation (bonds hedge equities).
    Warning: correlation > 0.3 = both moving together = no diversification.
    
    Score:
      corr <= -0.3  → Score 2.0  (strong negative = healthy hedge)
      -0.3 < corr <= 0.0 → Score 4.0  (mild negative = ok)
      0.0 < corr <= 0.3  → Score 6.5  (positive = weakening)
      corr > 0.3         → Score 9.0  (strong positive = warning)
    """
    try:
        spy = download_yf("SPY", 120)
        tlt = download_yf("TLT", 120)

        if spy.empty or tlt.empty or "Close" not in spy.columns or "Close" not in tlt.columns:
            log.warning("BOND_EQUITY_CORR: missing data")
            return None

        spy_ret = spy["Close"].pct_change().dropna()
        tlt_ret = tlt["Close"].pct_change().dropna()

        combined = pd.concat([spy_ret, tlt_ret], axis=1).dropna()
        combined.columns = ["SPY", "TLT"]

        if len(combined) < 60:
            log.warning(f"BOND_EQUITY_CORR: only {len(combined)} rows, need 60")
            return None

        corr = float(combined["SPY"].tail(60).corr(combined["TLT"].tail(60)))

        if corr <= -0.3:
            score = 2.0
            label = "Strong Negative (healthy hedge)"
        elif corr <= 0.0:
            score = 4.0
            label = "Mild Negative (ok)"
        elif corr <= 0.3:
            score = 6.5
            label = "Positive (weakening diversification)"
        else:
            score = 9.0
            label = "Strong Positive (WARNING — no hedge)"

        log.info(f"BOND_EQUITY_CORR_60D: corr={corr:.4f}, label={label}, Score={score:.2f}")

        return {
            "value": round(corr, 4),
            "score": round(score, 4),
            "age_days": 0,
            "source": "Calculated",
            "tier": "T2",
            "unit": "corr",
            "sheet_key": "BOND_EQUITY_CORR_60D",
        }

    except Exception as e:
        log.error(f"BOND_EQUITY_CORR error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 2 — GOLD/DXY BOTH UP (weight 0.15)
# ─────────────────────────────────────────────

def calc_gold_dxy_both_up(today: date) -> dict:
    """
    Gold (GLD) and DXY (DX-Y.NYB) 20-day returns.
    Both positive = flight-to-safety AND dollar strength = unusual stress signal.
    Normal: Gold up + DXY down (inverse relationship).
    
    Score:
      Both up (Gold>0 AND DXY>0)  → Score 8.0 (stress signal)
      Both down                    → Score 3.0 (risk-on, benign)
      Gold up, DXY down            → Score 4.0 (normal, gold hedge working)
      Gold down, DXY up            → Score 6.0 (dollar strength = mild headwind)
    
    Also returns raw GOLD_RETURN_20D and DXY_RETURN_20D for sheet.
    """
    try:
        gld = download_yf("GLD", 60)
        dxy = download_yf("DX-Y.NYB", 60)

        if gld.empty or dxy.empty or "Close" not in gld.columns or "Close" not in dxy.columns:
            log.warning("GOLD_DXY: missing data")
            return None, None, None

        gld_close = gld["Close"].dropna()
        dxy_close = dxy["Close"].dropna()

        if len(gld_close) < 22 or len(dxy_close) < 22:
            log.warning("GOLD_DXY: insufficient history")
            return None, None, None

        gold_ret = float((gld_close.iloc[-1] / gld_close.iloc[-21] - 1) * 100)
        dxy_ret  = float((dxy_close.iloc[-1] / dxy_close.iloc[-21] - 1) * 100)

        gold_up = gold_ret > 0
        dxy_up  = dxy_ret  > 0

        if gold_up and dxy_up:
            score = 8.0
            label = "BOTH UP — Stress Signal"
        elif not gold_up and not dxy_up:
            score = 3.0
            label = "Both Down — Risk-On"
        elif gold_up and not dxy_up:
            score = 4.0
            label = "Gold Up / DXY Down — Normal Hedge"
        else:
            score = 6.0
            label = "Gold Down / DXY Up — Dollar Strength"

        log.info(
            f"GOLD_DXY: gold_ret={gold_ret:.2f}%, dxy_ret={dxy_ret:.2f}%, "
            f"label={label}, Score={score:.2f}"
        )

        composite_result = {
            "value": round(score, 4),
            "score": round(score, 4),
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "score",
            "sheet_key": "GOLD_DXY_BOTH_UP",
        }

        gold_raw = {
            "value": round(gold_ret, 4),
            "score": None,
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "%",
            "sheet_key": "GOLD_RETURN_20D",
        }

        dxy_raw = {
            "value": round(dxy_ret, 4),
            "score": None,
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "%",
            "sheet_key": "DXY_RETURN_20D",
        }

        return composite_result, gold_raw, dxy_raw

    except Exception as e:
        log.error(f"GOLD_DXY error: {e}\n{traceback.format_exc()}")
        return None, None, None


# ─────────────────────────────────────────────
# SIGNAL 3 — COPPER vs SPY DIVERGENCE (weight 0.20)
# ─────────────────────────────────────────────

def calc_copper_spy_divergence(today: date) -> dict:
    """
    Copper (HG=F) SMA50 trend vs SPY SMA50 trend.
    Copper = leading economic indicator (Dr. Copper).
    Divergence: Copper SMA50 Falling + SPY SMA50 Rising = warning (equity leading copper).
    
    Score:
      Both Rising                        → Score 3.0 (bullish confirmation)
      Copper Rising / SPY Falling        → Score 4.0 (copper leading, recovery)
      Both Falling                       → Score 6.0 (bearish confirmation)
      Copper Falling / SPY Rising        → Score 8.5 (DIVERGENCE WARNING)
    
    SMA50 trend: current SMA50 vs SMA50 10 days ago.
    """
    try:
        copper = download_yf("HG=F", 120)
        spy    = download_yf("SPY", 120)

        if copper.empty or spy.empty or "Close" not in copper.columns or "Close" not in spy.columns:
            log.warning("COPPER_SPY: missing data")
            return None, None, None

        copper_close = copper["Close"].dropna()
        spy_close    = spy["Close"].dropna()

        if len(copper_close) < 60 or len(spy_close) < 60:
            log.warning("COPPER_SPY: insufficient history")
            return None, None, None

        copper_sma50_now  = float(copper_close.tail(50).mean())
        copper_sma50_prev = float(copper_close.tail(60).head(50).mean())
        spy_sma50_now     = float(spy_close.tail(50).mean())
        spy_sma50_prev    = float(spy_close.tail(60).head(50).mean())

        copper_rising = copper_sma50_now > copper_sma50_prev
        spy_rising    = spy_sma50_now    > spy_sma50_prev

        copper_trend = "Rising" if copper_rising else "Falling"
        spy_trend    = "Rising" if spy_rising    else "Falling"

        if copper_rising and spy_rising:
            score = 3.0
            label = "Both Rising — Bullish Confirmation"
        elif copper_rising and not spy_rising:
            score = 4.0
            label = "Copper Rising / SPY Falling — Recovery Signal"
        elif not copper_rising and not spy_rising:
            score = 6.0
            label = "Both Falling — Bearish Confirmation"
        else:
            score = 8.5
            label = "DIVERGENCE — Copper Falling / SPY Rising (WARNING)"

        log.info(
            f"COPPER_SPY: copper={copper_trend}, spy={spy_trend}, "
            f"label={label}, Score={score:.2f}"
        )

        composite_result = {
            "value": round(score, 4),
            "score": round(score, 4),
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "score",
            "sheet_key": "COPPER_SPY_DIVERGENCE",
        }

        copper_raw = {
            "value": copper_trend,
            "score": None,
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "Rising/Falling",
            "sheet_key": "COPPER_SMA50_TREND",
        }

        spy_raw = {
            "value": spy_trend,
            "score": None,
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "Rising/Falling",
            "sheet_key": "SPY_SMA50_TREND",
        }

        return composite_result, copper_raw, spy_raw

    except Exception as e:
        log.error(f"COPPER_SPY error: {e}\n{traceback.format_exc()}")
        return None, None, None


# ─────────────────────────────────────────────
# SIGNAL 4 — CREDIT vs EQUITY (weight 0.20)
# ─────────────────────────────────────────────

def calc_credit_vs_equity(today: date, fred: Fred) -> dict:
    """
    HY OAS spread (BAMLH0A0HYM2) trend vs SPY proximity to 52W high.
    Warning: HY spreads widening + SPY near 52W high = credit/equity divergence.
    
    HY spread 20d change:
      > +50bps  = significant widening
      +20 to +50 = moderate widening
      -20 to +20 = stable
      < -20      = tightening (bullish)
    
    SPY distance from 52W high:
      < 2%  = near high
      2-5%  = close
      > 5%  = away from high
    
    Score matrix:
      HY widening significantly + SPY near high → Score 9.0
      HY widening moderately   + SPY near high → Score 7.0
      HY widening              + SPY away       → Score 5.5
      HY stable/tightening     + SPY near high  → Score 4.0
      HY tightening            + SPY away       → Score 2.5
    """
    try:
        # HY spread from FRED
        hy_series = fred.get_series("BAMLH0A0HYM2", observation_start=(
            date.today() - timedelta(days=60)).strftime("%Y-%m-%d"))

        if hy_series is None or len(hy_series) < 22:
            log.warning("CREDIT_VS_EQUITY: insufficient HY data")
            return None

        hy_series = hy_series.dropna()
        hy_now    = float(hy_series.iloc[-1])
        hy_prev20 = float(hy_series.iloc[-21]) if len(hy_series) >= 21 else float(hy_series.iloc[0])
        hy_change = hy_now - hy_prev20  # in bps

        # SPY 52W high
        spy = download_yf("SPY", 260)
        if spy.empty or "Close" not in spy.columns:
            log.warning("CREDIT_VS_EQUITY: SPY data missing")
            return None

        spy_close   = spy["Close"].dropna()
        spy_now     = float(spy_close.iloc[-1])
        spy_52w_high = float(spy_close.tail(252).max())
        spy_dist_pct = (spy_52w_high - spy_now) / spy_52w_high * 100  # % below 52W high

        # Classify
        hy_widening_sig  = hy_change > 50
        hy_widening_mod  = 20 < hy_change <= 50
        hy_stable        = -20 <= hy_change <= 20
        spy_near_high    = spy_dist_pct < 2.0
        spy_close_high   = 2.0 <= spy_dist_pct < 5.0

        if hy_widening_sig and spy_near_high:
            score = 9.0
            label = f"HY +{hy_change:.0f}bps + SPY near high — CRITICAL DIVERGENCE"
        elif hy_widening_sig and spy_close_high:
            score = 7.5
            label = f"HY +{hy_change:.0f}bps + SPY close to high — HIGH WARNING"
        elif hy_widening_mod and spy_near_high:
            score = 7.0
            label = f"HY +{hy_change:.0f}bps + SPY near high — WARNING"
        elif hy_widening_mod and spy_close_high:
            score = 6.0
            label = f"HY +{hy_change:.0f}bps + SPY close — MODERATE"
        elif hy_widening_sig or hy_widening_mod:
            score = 5.5
            label = f"HY widening ({hy_change:.0f}bps) — CAUTION"
        elif hy_stable and spy_near_high:
            score = 4.0
            label = f"HY stable + SPY near high — NEUTRAL"
        else:
            score = 2.5
            label = f"HY tightening/stable + SPY away — BENIGN"

        hy_age = int((date.today() - hy_series.index[-1].date()).days)

        log.info(
            f"CREDIT_VS_EQUITY: HY={hy_now:.0f}bps, 20d_chg={hy_change:.1f}bps, "
            f"SPY_dist={spy_dist_pct:.2f}%, label={label}, Score={score:.2f}"
        )

        return {
            "value": round(hy_change, 2),
            "score": round(score, 4),
            "age_days": hy_age,
            "source": "FRED",
            "tier": "T1",
            "unit": "bps",
            "sheet_key": "CREDIT_VS_EQUITY",
        }

    except Exception as e:
        log.error(f"CREDIT_VS_EQUITY error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 5 — REAL YIELD TREND (weight 0.10)
# ─────────────────────────────────────────────

def calc_real_yield_trend(today: date, fred: Fred) -> dict:
    """
    TIPS 10Y real yield (DFII10) — 30d SMA trend.
    Rising real yields = headwind for equities and gold.
    
    Score:
      30d SMA strongly rising (>+20bps)  → Score 8.0
      30d SMA rising (+5 to +20bps)      → Score 6.5
      30d SMA flat (-5 to +5bps)         → Score 5.0
      30d SMA falling (-5 to -20bps)     → Score 3.5
      30d SMA strongly falling (<-20bps) → Score 2.0
    
    Also writes REAL_YIELD_10Y_TREND (Rising/Falling) to RAW_MARKET.
    """
    try:
        ry_series = fred.get_series("DFII10", observation_start=(
            date.today() - timedelta(days=90)).strftime("%Y-%m-%d"))

        if ry_series is None or len(ry_series) < 32:
            log.warning("REAL_YIELD_TREND: insufficient data")
            return None, None

        ry_series = ry_series.dropna()

        sma30_now  = float(ry_series.tail(30).mean())
        sma30_prev = float(ry_series.tail(60).head(30).mean()) if len(ry_series) >= 60 else float(ry_series.head(30).mean())
        sma30_change = (sma30_now - sma30_prev) * 100  # in bps (series is in %)

        trend_label = "Rising" if sma30_change > 0 else "Falling"

        if sma30_change > 20:
            score = 8.0
            label = f"Strongly Rising (+{sma30_change:.0f}bps) — HEADWIND"
        elif sma30_change > 5:
            score = 6.5
            label = f"Rising (+{sma30_change:.0f}bps) — MILD HEADWIND"
        elif sma30_change >= -5:
            score = 5.0
            label = f"Flat ({sma30_change:.0f}bps) — NEUTRAL"
        elif sma30_change >= -20:
            score = 3.5
            label = f"Falling ({sma30_change:.0f}bps) — TAILWIND"
        else:
            score = 2.0
            label = f"Strongly Falling ({sma30_change:.0f}bps) — STRONG TAILWIND"

        ry_age = int((date.today() - ry_series.index[-1].date()).days)

        log.info(
            f"REAL_YIELD_TREND: sma30_chg={sma30_change:.1f}bps, "
            f"trend={trend_label}, label={label}, Score={score:.2f}"
        )

        composite_result = {
            "value": round(sma30_change, 2),
            "score": round(score, 4),
            "age_days": ry_age,
            "source": "FRED",
            "tier": "T1",
            "unit": "bps",
            "sheet_key": "REAL_YIELD_TREND",
        }

        raw_result = {
            "value": trend_label,
            "score": None,
            "age_days": ry_age,
            "source": "FRED",
            "tier": "T1",
            "unit": "Rising/Falling",
            "sheet_key": "REAL_YIELD_10Y_TREND",
        }

        return composite_result, raw_result

    except Exception as e:
        log.error(f"REAL_YIELD_TREND error: {e}\n{traceback.format_exc()}")
        return None, None


# ─────────────────────────────────────────────
# SIGNAL 6 — YIELD CURVE 10Y-2Y (weight 0.15)
# ─────────────────────────────────────────────

def calc_yield_curve(today: date, fred: Fred) -> dict:
    """
    10Y-2Y yield spread (T10Y2Y from FRED, in bps).
    Classic recession predictor + steepener signal.
    
    Context:
      Deeply inverted (<-50bps): recession risk high → Score 8.0
      Inverted (-50 to 0bps):    caution → Score 6.5
      Flat (0 to +25bps):        steepening after inversion → Score 5.5
      Normal (+25 to +100bps):   healthy → Score 3.5
      Steep (>+100bps):          strong growth signal → Score 2.0
    
    Note: T10Y2Y is already the spread in percentage points on FRED.
    We convert to bps (* 100).
    """
    try:
        yc_series = fred.get_series("T10Y2Y", observation_start=(
            date.today() - timedelta(days=60)).strftime("%Y-%m-%d"))

        if yc_series is None or len(yc_series) < 2:
            log.warning("YIELD_CURVE: insufficient data")
            return None

        yc_series = yc_series.dropna()
        yc_now_pct = float(yc_series.iloc[-1])
        yc_bps     = yc_now_pct * 100  # convert to bps

        if yc_bps < -50:
            score = 8.0
            label = f"Deeply Inverted ({yc_bps:.0f}bps) — RECESSION RISK"
        elif yc_bps < 0:
            score = 6.5
            label = f"Inverted ({yc_bps:.0f}bps) — CAUTION"
        elif yc_bps < 25:
            score = 5.5
            label = f"Flat/Steepening ({yc_bps:.0f}bps) — WATCH"
        elif yc_bps < 100:
            score = 3.5
            label = f"Normal ({yc_bps:.0f}bps) — HEALTHY"
        else:
            score = 2.0
            label = f"Steep ({yc_bps:.0f}bps) — STRONG GROWTH SIGNAL"

        yc_age = int((date.today() - yc_series.index[-1].date()).days)

        log.info(
            f"YIELD_CURVE_10Y2Y: spread={yc_bps:.1f}bps, "
            f"label={label}, Score={score:.2f}"
        )

        return {
            "value": round(yc_bps, 2),
            "score": round(score, 4),
            "age_days": yc_age,
            "source": "FRED",
            "tier": "T1",
            "unit": "bps",
            "sheet_key": "YIELD_CURVE_10Y2Y",
        }

    except Exception as e:
        log.error(f"YIELD_CURVE error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# COMPOSITE CALCULATION
# ─────────────────────────────────────────────

PHASE_MAP = [
    (0.0,  1.5,  "Minimal Stress",    "Bullish"),
    (1.5,  3.0,  "Low Stress",        "Bullish"),
    (3.0,  4.5,  "Mild Caution",      "Neutral"),
    (4.5,  5.5,  "Neutral",           "Neutral"),
    (5.5,  7.0,  "Elevated Warning",  "Bearish"),
    (7.0,  8.5,  "High Warning",      "Bearish"),
    (8.5, 10.0,  "Critical Warning",  "Extreme"),
]


def get_phase(score: float) -> tuple:
    for lo, hi, phase, signal in PHASE_MAP:
        if lo <= score <= hi:
            return phase, signal
    return "Neutral", "Neutral"


def calc_composite(signal_results: dict) -> dict:
    """
    Weighted composite. signal_results keys must match WEIGHTS keys.
    """
    weighted_sum  = 0.0
    weight_total  = 0.0
    valid_count   = 0
    freshness_sum = 0.0

    for indicator, weight in WEIGHTS.items():
        r = signal_results.get(indicator)
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

    score_raw = weighted_sum / weight_total
    score_raw = max(0.0, min(10.0, score_raw))
    freshness = freshness_sum / valid_count

    phase, signal = get_phase(score_raw)

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

def write_raw_market(ws_raw_market, today: date, raw_rows: dict):
    """
    Writes 7 L7 indicator rows to RAW_MARKET tab (Rows 28-34).
    raw_rows: dict keyed by sheet_key (matching RAW_MARKET_ROWS).
    """
    today_str = today.strftime("%Y-%m-%d")

    for sheet_key, row_num in RAW_MARKET_ROWS.items():
        r = raw_rows.get(sheet_key)

        if r is None:
            val_str  = "ERROR"
            age_str  = "—"
            src_str  = "—"
            tier_str = "—"
            unit_str = "—"
        else:
            val_str  = str(round(r["value"], 6)) if isinstance(r["value"], float) else str(r["value"])
            age_str  = str(r["age_days"])
            src_str  = r["source"]
            tier_str = r["tier"]
            unit_str = r["unit"]

        try:
            existing_row = ws_raw_market.row_values(row_num)
            prev_7d  = existing_row[4] if len(existing_row) > 4 else "—"
            prev_30d = existing_row[5] if len(existing_row) > 5 else "—"
        except Exception:
            prev_7d  = "—"
            prev_30d = "—"

        row_data = [
            today_str,
            sheet_key,
            "L7",
            val_str,
            prev_7d,
            prev_30d,
            age_str,
            src_str,
            tier_str,
            unit_str,
        ]

        ws_raw_market.update(
            range_name=f"A{row_num}:J{row_num}",
            values=[row_data],
        )
        log.info(f"RAW_MARKET Row {row_num} written: {sheet_key} = {val_str}")


def write_scores(ws_scores, today: date, composite: dict):
    today_str = today.strftime("%Y-%m-%d")

    row_data = [
        f"L7 Cross-Asset ({today_str})",
        composite["score_raw"],
        "—",
        "—",
        "—",
        "—",
        "—",
        composite["signal"],
        composite["freshness"],
        "—",
        composite["asymmetry_adj"],
        composite["regime_weight"],
        "—",
    ]

    ws_scores.update(
        range_name=f"A{SCORES_ROW}:M{SCORES_ROW}",
        values=[row_data],
    )
    log.info(
        f"SCORES Row {SCORES_ROW} written: "
        f"L7={composite['score_raw']:.4f} | {composite['phase']} | {composite['signal']}"
    )


def write_dashboard(ws_dashboard, today: date, composite: dict):
    today_str = today.strftime("%Y-%m-%d")

    row_data = [
        "L7 Cross-Asset",
        composite["score_raw"],
        composite["signal"],
        composite["phase"],
        composite["freshness"],
        today_str,
        composite["valid_count"],
        f"{composite['valid_count']}/{len(WEIGHTS)}",
    ]

    ws_dashboard.update(
        range_name=f"A{DASHBOARD_ROW}:H{DASHBOARD_ROW}",
        values=[row_data],
    )
    log.info(
        f"DASHBOARD Row {DASHBOARD_ROW} written: "
        f"L7={composite['score_raw']:.4f} | {composite['signal']}"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("L7 CROSS-ASSET COLLECTOR — START")
    log.info("=" * 60)

    today = date.today()
    log.info(f"Date: {today}")

    fred = get_fred_client()

    # ── Calculate all signals ──
    signal_results = {}  # keyed by WEIGHTS keys
    raw_rows       = {}  # keyed by RAW_MARKET_ROWS keys

    log.info("--- Signal 1: Bond/Equity Correlation 60D ---")
    r = calc_bond_equity_corr(today)
    if r:
        signal_results["BOND_EQUITY_CORR_60D"] = r
        raw_rows["BOND_EQUITY_CORR_60D"] = r

    log.info("--- Signal 2: Gold/DXY Both Up ---")
    r_gold_dxy, r_gold_raw, r_dxy_raw = calc_gold_dxy_both_up(today)
    if r_gold_dxy:
        signal_results["GOLD_DXY_BOTH_UP"] = r_gold_dxy
    if r_gold_raw:
        raw_rows["GOLD_RETURN_20D"] = r_gold_raw
    if r_dxy_raw:
        raw_rows["DXY_RETURN_20D"] = r_dxy_raw

    log.info("--- Signal 3: Copper vs SPY Divergence ---")
    r_copper_comp, r_copper_raw, r_spy_raw = calc_copper_spy_divergence(today)
    if r_copper_comp:
        signal_results["COPPER_SPY_DIVERGENCE"] = r_copper_comp
    if r_copper_raw:
        raw_rows["COPPER_SMA50_TREND"] = r_copper_raw
    if r_spy_raw:
        raw_rows["SPY_SMA50_TREND"] = r_spy_raw

    log.info("--- Signal 4: Credit vs Equity ---")
    r = calc_credit_vs_equity(today, fred)
    if r:
        signal_results["CREDIT_VS_EQUITY"] = r

    log.info("--- Signal 5: Real Yield Trend ---")
    r_ry_comp, r_ry_raw = calc_real_yield_trend(today, fred)
    if r_ry_comp:
        signal_results["REAL_YIELD_TREND"] = r_ry_comp
    if r_ry_raw:
        raw_rows["REAL_YIELD_10Y_TREND"] = r_ry_raw

    log.info("--- Signal 6: Yield Curve 10Y-2Y ---")
    r = calc_yield_curve(today, fred)
    if r:
        signal_results["YIELD_CURVE_10Y2Y"] = r
        raw_rows["YIELD_CURVE_10Y2Y"] = r

    # ── Composite ──
    log.info("--- Composite Calculation ---")
    composite = calc_composite(signal_results)

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

        ws_raw_market = warehouse.worksheet("RAW_MARKET")
        ws_scores     = warehouse.worksheet("SCORES")
        ws_dashboard  = warehouse.worksheet("DASHBOARD")

        write_raw_market(ws_raw_market, today, raw_rows)
        write_scores(ws_scores, today, composite)
        write_dashboard(ws_dashboard, today, composite)

        log.info("All sheets written successfully.")

    except Exception as e:
        log.error(f"Google Sheets write error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    log.info("=" * 60)
    log.info(
        f"L7 CROSS-ASSET COLLECTOR — DONE | "
        f"{composite['score_raw']:.2f}/10 | {composite['phase']} | "
        f"{composite['signal']} | {composite['valid_count']}/{len(WEIGHTS)} Sources"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
