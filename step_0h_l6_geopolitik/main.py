"""
L6 Geopolitik Collector — step_0h_l6_geopolitik/main.py
Baldur Creek Capital — richie-hedge-pipeline

6 Signals:
  1. GPR_INDEX          (weight 0.25) — Caldara-Iacoviello Geopolitical Risk Index
  2. OIL_VOLATILITY_20D (weight 0.20) — OVX (Oil VIX) 20d avg vs 1Y avg
  3. BALTIC_DRY_INDEX   (weight 0.15) — VIX/OVX Ratio proxy (geopolitical stress)
  4. EM_FX_STRESS_BASKET(weight 0.20) — EEM/DXY ratio 20d change
  5. ELECTION_CYCLE_POS (weight 0.10) — Calendar-based election cycle position
  6. SANCTIONS_TARIFF_CT(weight 0.10) — Gold/Oil ratio 60d trend (proxy)

Writes to Data Warehouse:
  RAW_MACRO  : Rows 3-8   (L6, 6 indicators)
  SCORES     : Row 7      (L6 Geopolitik composite)
  DASHBOARD  : Row 22     (L6 score + signal)
"""

import os
import sys
import logging
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
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

WEIGHTS = {
    "GPR_INDEX":           0.25,
    "OIL_VOLATILITY_20D":  0.20,
    "BALTIC_DRY_INDEX":    0.15,  # VIX/OVX Ratio proxy
    "EM_FX_STRESS_BASKET": 0.20,
    "ELECTION_CYCLE_POS":  0.10,
    "SANCTIONS_TARIFF_CT": 0.10,  # Gold/Oil Ratio proxy
}

BEARISH_MULTIPLIER = 1.3
MIN_VALID          = 4

REGIME_WEIGHT_RISK_ON  = 0.05
REGIME_WEIGHT_RISK_OFF = 0.08
REGIME_WEIGHT_DD       = 0.10

WAREHOUSE_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# RAW_MACRO rows for L6 (1-indexed)
RAW_MACRO_ROWS = {
    "GPR_INDEX":           3,
    "OIL_VOLATILITY_20D":  4,
    "BALTIC_DRY_INDEX":    5,
    "EM_FX_STRESS_BASKET": 6,
    "ELECTION_CYCLE_POS":  7,
    "SANCTIONS_TARIFF_CT": 8,
}

SCORES_ROW    = 7
DASHBOARD_ROW = 22

# GPR Index CSV URL (Caldara & Iacoviello, updated monthly)
GPR_CSV_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_CSV_FALLBACK = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xlsx"

# ─────────────────────────────────────────────
# CONNECTIONS
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
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ─────────────────────────────────────────────
# SIGNAL 1 — GPR INDEX (weight 0.25)
# ─────────────────────────────────────────────

def calc_gpr_index(today: date) -> dict:
    """
    Caldara & Iacoviello Geopolitical Risk Index.
    Monthly data, downloaded directly from matteoiacoviello.com.
    
    Historical context:
      Normal baseline: ~100
      Elevated:        150-200
      Crisis:          >200 (Gulf War ~600, 9/11 ~450, Ukraine 2022 ~350)
    
    Score normalization: 50 (calm/0) to 300 (crisis/10).
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(GPR_CSV_URL, headers=headers, timeout=30)

        if resp.status_code != 200:
            log.warning(f"GPR: primary URL failed ({resp.status_code}), trying fallback")
            resp = requests.get(GPR_CSV_FALLBACK, headers=headers, timeout=30)

        if resp.status_code != 200:
            log.warning(f"GPR: both URLs failed, status={resp.status_code}")
            return None

        import io
        content = io.BytesIO(resp.content)

        try:
            df = pd.read_excel(content, engine="xlrd")
        except Exception:
            content.seek(0)
            df = pd.read_excel(content, engine="openpyxl")

        # Find GPR column (usually "GPRC" or "GPR")
        gpr_col = None
        for col in df.columns:
            if str(col).upper() in ("GPRC", "GPR", "GPRH"):
                gpr_col = col
                break
        if gpr_col is None:
            # Try second numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                gpr_col = num_cols[1]
            else:
                log.warning(f"GPR: cannot find GPR column. Columns: {df.columns.tolist()}")
                return None

        gpr_series = df[gpr_col].dropna()
        if len(gpr_series) < 12:
            log.warning("GPR: insufficient data rows")
            return None

        gpr_now  = float(gpr_series.iloc[-1])
        gpr_1y   = float(gpr_series.tail(12).mean())
        age_days = 15  # Monthly data, average age ~15 days into month

        # Normalize: 50 = Score 0 (calm), 300 = Score 10 (crisis)
        score = (gpr_now - 50.0) / 250.0 * 10.0
        score = max(0.0, min(10.0, score))

        if gpr_now < 100:
            label = f"Calm ({gpr_now:.0f})"
        elif gpr_now < 150:
            label = f"Moderate ({gpr_now:.0f})"
        elif gpr_now < 200:
            label = f"Elevated ({gpr_now:.0f})"
        else:
            label = f"Crisis ({gpr_now:.0f})"

        log.info(f"GPR_INDEX: value={gpr_now:.1f}, 1Y_avg={gpr_1y:.1f}, label={label}, Score={score:.2f}")

        return {
            "value": round(gpr_now, 2),
            "score": round(score, 4),
            "age_days": age_days,
            "source": "Caldara-Iacoviello",
            "tier": "T1",
            "unit": "index",
        }

    except Exception as e:
        log.error(f"GPR_INDEX error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 2 — OIL VOLATILITY 20D (weight 0.20)
# ─────────────────────────────────────────────

def calc_oil_volatility(today: date) -> dict:
    """
    OVX (CBOE Crude Oil Volatility Index) via yfinance (^OVX).
    20d average vs 1Y average — elevated OVX = geopolitical/supply stress.
    
    Score: 20d avg / 1Y avg ratio:
      < 0.8  → Score 2.0 (very calm)
      0.8-1.0 → Score 4.0 (below average)
      1.0-1.3 → Score 6.0 (elevated)
      1.3-1.6 → Score 8.0 (high stress)
      > 1.6  → Score 9.5 (crisis)
    """
    try:
        ovx = download_yf("^OVX", 400)

        if ovx.empty or "Close" not in ovx.columns:
            log.warning("OIL_VOLATILITY: OVX data missing")
            return None

        ovx_close = ovx["Close"].dropna()

        if len(ovx_close) < 22:
            log.warning("OIL_VOLATILITY: insufficient data")
            return None

        ovx_20d = float(ovx_close.tail(20).mean())
        ovx_1y  = float(ovx_close.tail(252).mean()) if len(ovx_close) >= 252 else float(ovx_close.mean())
        ratio   = ovx_20d / ovx_1y if ovx_1y > 0 else 1.0

        if ratio < 0.8:
            score = 2.0
            label = f"Very Calm (ratio={ratio:.2f})"
        elif ratio < 1.0:
            score = 4.0
            label = f"Below Average (ratio={ratio:.2f})"
        elif ratio < 1.3:
            score = 6.0
            label = f"Elevated (ratio={ratio:.2f})"
        elif ratio < 1.6:
            score = 8.0
            label = f"High Stress (ratio={ratio:.2f})"
        else:
            score = 9.5
            label = f"Crisis (ratio={ratio:.2f})"

        log.info(f"OIL_VOLATILITY_20D: ovx_20d={ovx_20d:.1f}, ratio={ratio:.2f}, label={label}, Score={score:.2f}")

        return {
            "value": round(ovx_20d, 2),
            "score": round(score, 4),
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "%",
        }

    except Exception as e:
        log.error(f"OIL_VOLATILITY error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 3 — BALTIC DRY (VIX/OVX RATIO PROXY) (weight 0.15)
# ─────────────────────────────────────────────

def calc_vix_ovx_ratio(today: date) -> dict:
    """
    VIX/OVX Ratio — geopolitical stress proxy.
    Baltic Dry Index not freely available; VIX/OVX ratio captures
    whether oil-specific fear exceeds general equity fear.
    
    OVX > VIX (ratio < 1.0): oil-specific stress = geopolitical/supply shock
    OVX < VIX (ratio > 1.0): equity fear > oil fear = financial stress, not geo
    
    Score (inverted — low ratio = high geo stress):
      ratio < 0.5  → Score 9.0 (extreme oil fear = geo crisis)
      0.5-0.8      → Score 7.0 (elevated oil fear)
      0.8-1.0      → Score 5.5 (mild oil stress)
      1.0-1.5      → Score 4.0 (equity fear dominant, normal)
      > 1.5        → Score 2.0 (equity fear >> oil, geo calm)
    """
    try:
        vix = download_yf("^VIX", 30)
        ovx = download_yf("^OVX", 30)

        if vix.empty or ovx.empty or "Close" not in vix.columns or "Close" not in ovx.columns:
            log.warning("VIX_OVX_RATIO: missing data")
            return None

        vix_now = float(vix["Close"].dropna().iloc[-1])
        ovx_now = float(ovx["Close"].dropna().iloc[-1])

        if ovx_now <= 0:
            log.warning("VIX_OVX_RATIO: OVX is zero or negative")
            return None

        ratio = vix_now / ovx_now

        if ratio < 0.5:
            score = 9.0
            label = f"Extreme Oil Fear (VIX/OVX={ratio:.2f})"
        elif ratio < 0.8:
            score = 7.0
            label = f"Elevated Oil Fear (VIX/OVX={ratio:.2f})"
        elif ratio < 1.0:
            score = 5.5
            label = f"Mild Oil Stress (VIX/OVX={ratio:.2f})"
        elif ratio < 1.5:
            score = 4.0
            label = f"Normal (VIX/OVX={ratio:.2f})"
        else:
            score = 2.0
            label = f"Geo Calm (VIX/OVX={ratio:.2f})"

        log.info(f"BALTIC_DRY(VIX/OVX): vix={vix_now:.1f}, ovx={ovx_now:.1f}, ratio={ratio:.2f}, label={label}, Score={score:.2f}")

        return {
            "value": round(ratio, 4),
            "score": round(score, 4),
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "ratio",
        }

    except Exception as e:
        log.error(f"VIX_OVX_RATIO error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 4 — EM FX STRESS BASKET (weight 0.20)
# ─────────────────────────────────────────────

def calc_em_fx_stress(today: date) -> dict:
    """
    EM stress proxy: EEM (EM equities) / UUP (USD ETF) ratio, 20d change.
    When EM falls relative to USD = EM stress = geopolitical/macro risk.
    
    Using UUP (Invesco DB USD Bull ETF) as DXY proxy — more reliable on yfinance.
    
    20d ratio change:
      < -5%   → Score 9.0 (severe EM stress)
      -5 to -2% → Score 7.0 (elevated stress)
      -2 to 0% → Score 5.5 (mild stress)
      0 to +2% → Score 4.0 (stable)
      > +2%    → Score 2.0 (EM outperforming, risk-on)
    """
    try:
        eem = download_yf("EEM", 60)
        uup = download_yf("UUP", 60)

        if eem.empty or uup.empty or "Close" not in eem.columns or "Close" not in uup.columns:
            log.warning("EM_FX_STRESS: missing data")
            return None

        eem_close = eem["Close"].dropna()
        uup_close = uup["Close"].dropna()

        # Align on common dates
        combined = pd.concat([eem_close, uup_close], axis=1).dropna()
        combined.columns = ["EEM", "UUP"]

        if len(combined) < 22:
            log.warning(f"EM_FX_STRESS: only {len(combined)} rows after align")
            return None

        ratio_now  = float(combined["EEM"].iloc[-1]  / combined["UUP"].iloc[-1])
        ratio_prev = float(combined["EEM"].iloc[-21] / combined["UUP"].iloc[-21])
        ratio_chg  = (ratio_now / ratio_prev - 1) * 100  # % change

        if ratio_chg < -5.0:
            score = 9.0
            label = f"Severe EM Stress ({ratio_chg:.1f}%)"
        elif ratio_chg < -2.0:
            score = 7.0
            label = f"Elevated EM Stress ({ratio_chg:.1f}%)"
        elif ratio_chg < 0.0:
            score = 5.5
            label = f"Mild EM Stress ({ratio_chg:.1f}%)"
        elif ratio_chg < 2.0:
            score = 4.0
            label = f"Stable ({ratio_chg:.1f}%)"
        else:
            score = 2.0
            label = f"EM Outperforming ({ratio_chg:.1f}%)"

        log.info(f"EM_FX_STRESS: ratio_chg={ratio_chg:.2f}%, label={label}, Score={score:.2f}")

        return {
            "value": round(ratio_chg, 4),
            "score": round(score, 4),
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "index",
        }

    except Exception as e:
        log.error(f"EM_FX_STRESS error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 5 — ELECTION CYCLE POSITION (weight 0.10)
# ─────────────────────────────────────────────

def calc_election_cycle(today: date) -> dict:
    """
    US midterm + global election cycle position.
    2026 = US Midterm Year (November 2026).
    
    Geopolitical risk tends to rise in election years due to:
    - Policy uncertainty
    - Trade/tariff posturing
    - International relations shifts
    
    Score based on months to next major US election:
      > 18 months  → Score 3.0 (far, low uncertainty)
      12-18 months → Score 5.0 (approaching)
      6-12 months  → Score 7.0 (election year, elevated uncertainty)
      < 6 months   → Score 8.5 (imminent, maximum uncertainty)
    """
    try:
        # Next major elections
        elections = [
            date(2026, 11, 3),   # US Midterms
            date(2028, 11, 7),   # US Presidential
        ]

        nearest = min(elections, key=lambda d: abs((d - today).days))
        days_to = (nearest - today).days
        months_to = days_to / 30.44

        if months_to > 18:
            score = 3.0
            label = f"Far ({months_to:.0f}mo to election)"
        elif months_to > 12:
            score = 5.0
            label = f"Approaching ({months_to:.0f}mo to election)"
        elif months_to > 6:
            score = 7.0
            label = f"Election Year ({months_to:.0f}mo to election)"
        else:
            score = 8.5
            label = f"Imminent ({months_to:.0f}mo to election)"

        log.info(f"ELECTION_CYCLE: next={nearest}, months_to={months_to:.1f}, label={label}, Score={score:.2f}")

        return {
            "value": round(months_to, 1),
            "score": round(score, 4),
            "age_days": 0,
            "source": "Calendar",
            "tier": "T1",
            "unit": "Year 2",
        }

    except Exception as e:
        log.error(f"ELECTION_CYCLE error: {e}")
        return None


# ─────────────────────────────────────────────
# SIGNAL 6 — SANCTIONS/TARIFF (GOLD/OIL RATIO PROXY) (weight 0.10)
# ─────────────────────────────────────────────

def calc_gold_oil_ratio(today: date) -> dict:
    """
    Gold/Oil ratio (GLD/USO) 60d trend.
    Rising ratio = Gold outperforming Oil = sanctions/supply disruption signal.
    When sanctions hit oil supply, oil prices spike but gold also rises as
    safe haven — the ratio captures the net geopolitical premium.
    
    60d SMA trend of GLD/USO ratio:
      Rising strongly (>+10%)  → Score 8.0 (geo premium building)
      Rising moderately (+3-10%) → Score 6.5
      Flat (-3 to +3%)         → Score 5.0
      Falling (-3 to -10%)     → Score 3.5 (oil outperforming, supply ok)
      Falling strongly (<-10%) → Score 2.0 (oil boom, geo calm)
    """
    try:
        gld = download_yf("GLD", 120)
        uso = download_yf("USO", 120)

        if gld.empty or uso.empty or "Close" not in gld.columns or "Close" not in uso.columns:
            log.warning("GOLD_OIL_RATIO: missing data")
            return None

        gld_close = gld["Close"].dropna()
        uso_close = uso["Close"].dropna()

        combined = pd.concat([gld_close, uso_close], axis=1).dropna()
        combined.columns = ["GLD", "USO"]

        if len(combined) < 62:
            log.warning(f"GOLD_OIL_RATIO: only {len(combined)} rows, need 62")
            return None

        combined["ratio"] = combined["GLD"] / combined["USO"]
        ratio_now  = float(combined["ratio"].iloc[-1])
        ratio_60d  = float(combined["ratio"].iloc[-61])
        ratio_chg  = (ratio_now / ratio_60d - 1) * 100  # % change over 60d

        if ratio_chg > 10.0:
            score = 8.0
            label = f"Geo Premium Building ({ratio_chg:.1f}% 60d)"
        elif ratio_chg > 3.0:
            score = 6.5
            label = f"Moderate Geo Premium ({ratio_chg:.1f}% 60d)"
        elif ratio_chg >= -3.0:
            score = 5.0
            label = f"Flat ({ratio_chg:.1f}% 60d)"
        elif ratio_chg >= -10.0:
            score = 3.5
            label = f"Oil Outperforming ({ratio_chg:.1f}% 60d)"
        else:
            score = 2.0
            label = f"Oil Boom / Geo Calm ({ratio_chg:.1f}% 60d)"

        log.info(f"SANCTIONS_TARIFF(Gold/Oil): ratio_chg={ratio_chg:.2f}%, label={label}, Score={score:.2f}")

        return {
            "value": round(ratio_chg, 4),
            "score": round(score, 4),
            "age_days": 0,
            "source": "yfinance",
            "tier": "T2",
            "unit": "count",
        }

    except Exception as e:
        log.error(f"GOLD_OIL_RATIO error: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────
# COMPOSITE
# ─────────────────────────────────────────────

PHASE_MAP = [
    (0.0,  1.5,  "Minimal Risk",    "Bullish"),
    (1.5,  3.0,  "Low Risk",        "Bullish"),
    (3.0,  4.5,  "Moderate",        "Neutral"),
    (4.5,  5.5,  "Neutral",         "Neutral"),
    (5.5,  7.0,  "Elevated Risk",   "Bearish"),
    (7.0,  8.5,  "High Risk",       "Bearish"),
    (8.5, 10.0,  "Critical Risk",   "Extreme"),
]


def get_phase(score: float) -> tuple:
    for lo, hi, phase, signal in PHASE_MAP:
        if lo <= score <= hi:
            return phase, signal
    return "Neutral", "Neutral"


def calc_composite(results: dict) -> dict:
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

def write_raw_macro(ws_raw_macro, today: date, results: dict):
    today_str = today.strftime("%Y-%m-%d")

    indicator_order = [
        "GPR_INDEX",
        "OIL_VOLATILITY_20D",
        "BALTIC_DRY_INDEX",
        "EM_FX_STRESS_BASKET",
        "ELECTION_CYCLE_POS",
        "SANCTIONS_TARIFF_CT",
    ]

    for indicator in indicator_order:
        row_num = RAW_MACRO_ROWS[indicator]
        r = results.get(indicator)

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
            existing_row = ws_raw_macro.row_values(row_num)
            prev_7d  = existing_row[4] if len(existing_row) > 4 else "—"
            prev_30d = existing_row[5] if len(existing_row) > 5 else "—"
        except Exception:
            prev_7d  = "—"
            prev_30d = "—"

        row_data = [
            today_str,
            indicator,
            "L6",
            val_str,
            prev_7d,
            prev_30d,
            age_str,
            src_str,
            tier_str,
            unit_str,
        ]

        ws_raw_macro.update(
            range_name=f"A{row_num}:J{row_num}",
            values=[row_data],
        )
        log.info(f"RAW_MACRO Row {row_num} written: {indicator} = {val_str}")


def write_scores(ws_scores, today: date, composite: dict):
    today_str = today.strftime("%Y-%m-%d")

    row_data = [
        f"L6 Geopolitik ({today_str})",
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
        f"L6={composite['score_raw']:.4f} | {composite['phase']} | {composite['signal']}"
    )


def write_dashboard(ws_dashboard, today: date, composite: dict):
    today_str = today.strftime("%Y-%m-%d")

    row_data = [
        "L6 Geopolitik",
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
        f"L6={composite['score_raw']:.4f} | {composite['signal']}"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("L6 GEOPOLITIK COLLECTOR — START")
    log.info("=" * 60)

    today = date.today()
    log.info(f"Date: {today}")

    fred = get_fred_client()

    results = {}

    log.info("--- Signal 1: GPR Index ---")
    results["GPR_INDEX"] = calc_gpr_index(today)

    log.info("--- Signal 2: Oil Volatility 20D ---")
    results["OIL_VOLATILITY_20D"] = calc_oil_volatility(today)

    log.info("--- Signal 3: VIX/OVX Ratio (Baltic Dry Proxy) ---")
    results["BALTIC_DRY_INDEX"] = calc_vix_ovx_ratio(today)

    log.info("--- Signal 4: EM FX Stress Basket ---")
    results["EM_FX_STRESS_BASKET"] = calc_em_fx_stress(today)

    log.info("--- Signal 5: Election Cycle Position ---")
    results["ELECTION_CYCLE_POS"] = calc_election_cycle(today)

    log.info("--- Signal 6: Gold/Oil Ratio (Sanctions Proxy) ---")
    results["SANCTIONS_TARIFF_CT"] = calc_gold_oil_ratio(today)

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

    log.info("--- Writing to Google Sheets ---")
    try:
        client    = get_gspread_client()
        warehouse = open_warehouse(client)

        ws_raw_macro = warehouse.worksheet("RAW_MACRO")
        ws_scores    = warehouse.worksheet("SCORES")
        ws_dashboard = warehouse.worksheet("DASHBOARD")

        write_raw_macro(ws_raw_macro, today, results)
        write_scores(ws_scores, today, composite)
        write_dashboard(ws_dashboard, today, composite)

        log.info("All sheets written successfully.")

    except Exception as e:
        log.error(f"Google Sheets write error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    log.info("=" * 60)
    log.info(
        f"L6 GEOPOLITIK COLLECTOR — DONE | "
        f"{composite['score_raw']:.2f}/10 | {composite['phase']} | "
        f"{composite['signal']} | {composite['valid_count']}/{len(WEIGHTS)} Sources"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
