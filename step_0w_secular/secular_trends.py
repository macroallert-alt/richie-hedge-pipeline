"""
Säkulare Trends Circle — Main Script
Baldur Creek Capital | Step 0w (V1.5 — yfinance for market data + all V1.4 fixes)

Pipeline:
  1. FRED + yfinance Daten fetchen (18 FRED + 4 yfinance + 2 statisch = 24 Serien)
  2. Regime-Blöcke berechnen (Ratios, Perzentile, Directional Scores)
  3. Fragilitäts-Indikatoren berechnen
  4. Regime-Activation Scores + Gewichteter Tailwind Score
  5. Conviction Summary
  6. Bewertungs-Kaskade (6 Ratios, Perzentile, Half-Life O-U, Kaskaden-Logik)
  7. LLM-Call (Narrativ + Web Search Fundamental-Bestätigung)
  8. Combined Signal berechnen (Preis x Fundamental Multiplier)
  9. JSON schreiben
  10. Git commit + push

Fixes V1.5 (over V1.4):
  - EODHD API replaced with yfinance (free, full history, no API key needed)
  - EODHD had 402 Payment Required for historical data beyond ~1 year
  - yfinance gives SPY since 1993, GLD since 2004, SLV since 2006, DBC since 2006
  - All V1.4 fixes retained (Series ambiguous, LLM parser, etc.)

Usage:
  python -m step_0w_secular.secular_trends [--skip-git] [--skip-llm]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import linregress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("secular")

from .config import (
    FRED_SERIES, FRED_BASE_URL,
    EOD_TICKERS, EOD_BASE_URL,
    STATIC_WAP_DATA,
    REGIME_BLOCKS, REGIME_WEIGHTS, REGIME_ORDER,
    FRAGILITY_INDICATORS,
    ASSET_CLASSES, ASSET_CLASS_LABELS,
    ACTIVE_THRESHOLD, ROBUSTNESS_MAP, ROBUST_CATEGORIES,
    VALUATION_RATIOS,
    CLAUDE_MODEL, LLM_MAX_TOKENS,
    DATA_DIR, OUTPUT_FILE,
)

# yfinance tickers matching our EOD_TICKERS keys
YFINANCE_TICKERS = {
    "SPY":    "SPY",
    "DBC":    "DBC",
    "GOLD":   "GLD",
    "SILVER": "SLV",
}


# ===================================================================
# 1. DATA FETCHING
# ===================================================================

def fetch_fred(series_id, fred_api_key):
    """Fetch a single FRED series."""
    url = f"{FRED_BASE_URL}/series/observations"
    params = {
        "series_id": series_id, "api_key": fred_api_key,
        "file_type": "json", "sort_order": "asc",
        "observation_start": "1940-01-01",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            logger.warning(f"FRED {series_id}: no observations")
            return None
        dates, values = [], []
        for o in obs:
            if o["value"] == ".":
                continue
            dates.append(pd.Timestamp(o["date"]))
            values.append(float(o["value"]))
        if not dates:
            return None
        s = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        logger.info(f"FRED {series_id}: {len(s)} obs, {s.index[0].date()} -> {s.index[-1].date()}")
        return s
    except Exception as e:
        logger.error(f"FRED {series_id} failed: {e}")
        return None


def fetch_yfinance_monthly(key, yf_ticker):
    """Fetch full history via yfinance and resample to monthly.

    yfinance is free, no API key needed, gives full history:
    SPY since 1993, GLD since 2004, SLV since 2006, DBC since 2006.
    """
    try:
        import yfinance as yf
        logger.info(f"yfinance {yf_ticker}: downloading full history...")
        ticker_obj = yf.Ticker(yf_ticker)
        df = ticker_obj.history(period="max", auto_adjust=True)
        if df is None or df.empty:
            logger.warning(f"yfinance {yf_ticker}: no data")
            return None
        s = df["Close"].dropna()
        s.name = key
        # Remove timezone info if present
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # Resample to month-end
        s = s.resample("ME").last().dropna()
        logger.info(f"yfinance {yf_ticker}: {len(s)} monthly obs, {s.index[0].date()} -> {s.index[-1].date()}")
        return s
    except Exception as e:
        logger.error(f"yfinance {yf_ticker} failed: {e}")
        return None


def load_static_wap():
    """Load static Working Age Population data for China + Germany."""
    result = {}
    for key, wap_def in STATIC_WAP_DATA.items():
        data = wap_def["data"]
        dates = [pd.Timestamp(f"{year}-07-01") for year in sorted(data.keys())]
        values = [data[year] for year in sorted(data.keys())]
        s = pd.Series(values, index=pd.DatetimeIndex(dates), name=key)
        s = s.resample("ME").ffill()
        logger.info(f"Static {key}: {len(s)} monthly obs, {s.index[0].date()} -> {s.index[-1].date()}")
        result[key] = s
    return result


def fetch_all_data():
    """Fetch all series: 18 FRED + 4 yfinance + 2 static."""
    fred_api_key = os.environ.get("FRED_API_KEY", "")
    if not fred_api_key:
        logger.error("FRED_API_KEY not set"); sys.exit(1)

    all_data = {}
    for key, sid in FRED_SERIES.items():
        s = fetch_fred(sid, fred_api_key)
        if s is not None:
            all_data[key] = s

    # Market data via yfinance (replaces EODHD which had 402 errors)
    for key, yf_ticker in YFINANCE_TICKERS.items():
        s = fetch_yfinance_monthly(key, yf_ticker)
        if s is not None:
            all_data[key] = s

    static = load_static_wap()
    all_data.update(static)

    total = len(FRED_SERIES) + len(YFINANCE_TICKERS) + len(STATIC_WAP_DATA)
    logger.info(f"Fetched {len(all_data)}/{total} series")
    if len(all_data) < total * 0.5:
        logger.error(f"Only {len(all_data)}/{total} — aborting"); sys.exit(1)
    return all_data


# ===================================================================
# 2. HELPERS
# ===================================================================

def to_monthly(s):
    if s is None: return None
    return s.resample("ME").last().dropna()

def compute_yoy(s):
    if s is None or len(s) < 13: return None
    return s.pct_change(12) * 100

def compute_ratio(numerator, denominator, multiply=1.0, denom_scale=1.0):
    if numerator is None or denominator is None: return None
    num_m = to_monthly(numerator)
    den_m = to_monthly(denominator)
    aligned = pd.concat([num_m, den_m], axis=1, sort=True).dropna()
    if aligned.empty: return None
    ratio = (aligned.iloc[:, 0] / (aligned.iloc[:, 1] / denom_scale)) * multiply
    ratio.name = "ratio"
    return ratio

def compute_percentile(series, current_value):
    if series is None or len(series) < 12: return None
    return round(float((series < current_value).sum() / len(series) * 100), 1)

def compute_percentile_20y(series, current_value):
    if series is None or len(series) < 12: return None
    cutoff = max(0, len(series) - 240)
    recent = series.iloc[cutoff:]
    if len(recent) < 12: return None
    return round(float((recent < current_value).sum() / len(recent) * 100), 1)


# ===================================================================
# 3. DIRECTIONAL SCORES
# ===================================================================

def calc_directional_score(method, series, percentile, all_data):
    if method == "none" or method is None:
        return 0.0
    if percentile is None and method not in (
        "wap_growth", "mfg_employment", "trade_deficit",
        "interest_defense_ratio", "real_rate",
    ):
        return 0.5
    if method == "low_is_active":
        return round(1.0 - (percentile / 100.0), 3)
    elif method == "high_is_active":
        return round(min(percentile / 100.0, 1.0), 3)
    elif method == "wap_growth":
        if not isinstance(series, dict): return 0.5
        scores = []
        for k, v in series.items():
            if v is None: scores.append(0.5)
            elif v < 0: scores.append(1.0)
            elif v <= 1.0: scores.append(0.7)
            else: scores.append(0.3)
        return round(np.mean(scores), 3) if scores else 0.5
    elif method == "mfg_employment":
        if series is None or len(series) < 25: return 0.5
        return 0.8 if series.iloc[-1] - series.iloc[-25] > 0 else 0.3
    elif method == "trade_deficit":
        if series is None or len(series) < 12: return 0.5
        mx = series.min()
        if mx >= 0: return 0.3
        return round(min(abs(series.iloc[-1]) / abs(mx), 1.0), 3)
    elif method == "interest_defense_ratio":
        if series is None: return 0.5
        val = float(series) if not isinstance(series, pd.Series) else float(series.iloc[-1])
        return round(min(max(val, 0.0), 1.0), 3)
    elif method == "real_rate":
        if series is None: return 0.5
        val = float(series) if not isinstance(series, pd.Series) else float(series.iloc[-1])
        return round(1.0 - max(0.0, min(1.0, (val + 2.0) / 6.0)), 3)
    else:
        logger.warning(f"Unknown method: {method}"); return 0.5


# ===================================================================
# 4. REGIME BLOCK COMPUTATION
# ===================================================================

def compute_chart_data(chart_def, all_data):
    cid = chart_def["id"]
    ctype = chart_def["type"]
    method = chart_def.get("directional_score_method", "none")

    result = {
        "id": cid, "name": chart_def["name"], "type": ctype,
        "unit": chart_def.get("unit", ""),
        "current": None, "alltime_mean": None, "rolling_20y_mean": None,
        "percentile": None, "percentile_20y": None,
        "directional_score": 0.5, "data": [],
        "annotations": chart_def.get("annotations", []),
    }
    for k in ("reference_line", "lines", "color_zones"):
        if k in chart_def: result[k] = chart_def[k]

    try:
        if ctype == "single_line":
            s = all_data.get(chart_def["series"][0])
            if s is None: return result
            s = to_monthly(s)
            cur = float(s.iloc[-1])
            result["current"] = round(cur, 2)
            result["alltime_mean"] = round(float(s.mean()), 2)
            r20 = s.rolling(240).mean()
            result["rolling_20y_mean"] = round(float(r20.iloc[-1]), 2) if len(r20.dropna()) > 0 else None
            result["percentile"] = compute_percentile(s, cur)
            result["percentile_20y"] = compute_percentile_20y(s, cur)
            result["directional_score"] = calc_directional_score(method, s, result["percentile"], all_data)
            result["data"] = _s2j(s)

        elif ctype == "multi_line":
            lines_data, latest_yoy = {}, {}
            for ld in chart_def["lines"]:
                k = ld["key"]
                s = all_data.get(k)
                if s is None:
                    logger.warning(f"Chart {cid}: missing {k}"); continue
                s = to_monthly(s)
                if chart_def.get("transform") == "yoy_growth":
                    s = compute_yoy(s)
                    if s is None: continue
                lines_data[k] = s
                latest_yoy[k] = float(s.iloc[-1]) if len(s) > 0 else None
            if lines_data:
                merged = pd.DataFrame(lines_data).dropna(how="all")
                result["data"] = [
                    {**{"date": dt.strftime("%Y-%m")},
                     **{k: round(float(row[k]), 3) if pd.notna(row.get(k)) else None for k in lines_data}}
                    for dt, row in merged.iterrows()
                ]
            result["directional_score"] = calc_directional_score(method, latest_yoy, None, all_data)
            valid = [v for v in latest_yoy.values() if v is not None]
            result["current"] = round(np.mean(valid), 2) if valid else None

        elif ctype == "ratio":
            nk, dk = chart_def["numerator"], chart_def["denominator"]
            mul = chart_def.get("multiply", 1.0)
            ds = chart_def.get("denominator_scale", 1.0)
            ratio = compute_ratio(all_data.get(nk), all_data.get(dk), multiply=mul, denom_scale=ds)
            if ratio is None or len(ratio) < 12:
                logger.warning(f"Chart {cid}: ratio {nk}/{dk} failed"); return result
            cur = float(ratio.iloc[-1])
            result["current"] = round(cur, 4)
            result["alltime_mean"] = round(float(ratio.mean()), 4)
            r20 = ratio.rolling(240).mean()
            result["rolling_20y_mean"] = round(float(r20.iloc[-1]), 4) if len(r20.dropna()) > 0 else None
            result["percentile"] = compute_percentile(ratio, cur)
            result["percentile_20y"] = compute_percentile_20y(ratio, cur)
            result["directional_score"] = calc_directional_score(method, ratio, result["percentile"], all_data)
            result["data"] = _s2j(ratio)
            if result["alltime_mean"] and result["rolling_20y_mean"]:
                d = abs(result["rolling_20y_mean"] - result["alltime_mean"]) / abs(result["alltime_mean"]) * 100
                result["structural_shift_warning"] = d > 30

        elif ctype == "dual_line":
            lo = {}
            for ld in chart_def["lines"]:
                s = all_data.get(ld["key"])
                if s is not None:
                    lo[ld["key"]] = to_monthly(s)
            if len(lo) == 2:
                ks = list(lo.keys())
                merged = pd.concat([lo[ks[0]], lo[ks[1]]], axis=1, sort=True).dropna()
                merged.columns = ks
                result["data"] = [
                    {**{"date": dt.strftime("%Y-%m")}, **{k: round(float(row[k]), 2) for k in ks}}
                    for dt, row in merged.iterrows()
                ]
                iv, dv = float(merged.iloc[-1][ks[0]]), float(merged.iloc[-1][ks[1]])
                t = iv + dv
                if t > 0:
                    ratio_val = iv / t
                    result["directional_score"] = calc_directional_score(method, ratio_val, None, all_data)
                result["current"] = round(iv, 2)

        elif ctype == "computed_real_rate":
            gs10 = all_data.get("GS10")
            cpi = all_data.get("CPIAUCSL")
            if gs10 is None or cpi is None: return result
            gs10_m = to_monthly(gs10)
            cpi_yoy = compute_yoy(to_monthly(cpi))
            if cpi_yoy is None: return result
            aligned = pd.concat([gs10_m, cpi_yoy], axis=1, sort=True).dropna()
            if aligned.empty: return result
            aligned.columns = ["gs10", "cpi_yoy"]
            rr = aligned["gs10"] - aligned["cpi_yoy"]
            cur = float(rr.iloc[-1])
            result["current"] = round(cur, 2)
            result["alltime_mean"] = round(float(rr.mean()), 2)
            r20 = rr.rolling(240).mean()
            result["rolling_20y_mean"] = round(float(r20.iloc[-1]), 2) if len(r20.dropna()) > 0 else None
            result["percentile"] = compute_percentile(rr, cur)
            result["percentile_20y"] = compute_percentile_20y(rr, cur)
            result["directional_score"] = calc_directional_score(method, cur, result["percentile"], all_data)
            result["data"] = _s2j(rr)

        elif ctype == "dual_axis_gold_realrate":
            gold = all_data.get("GOLD")
            gs10 = all_data.get("GS10")
            cpi = all_data.get("CPIAUCSL")
            if gold is None or gs10 is None or cpi is None: return result
            gold_m, gs10_m = to_monthly(gold), to_monthly(gs10)
            cpi_yoy = compute_yoy(to_monthly(cpi))
            if cpi_yoy is None: return result
            aligned = pd.concat([gold_m, gs10_m, cpi_yoy], axis=1, sort=True).dropna()
            if aligned.empty: return result
            aligned.columns = ["gold", "gs10", "cpi_yoy"]
            aligned["real_rate"] = aligned["gs10"] - aligned["cpi_yoy"]
            result["data"] = [
                {"date": dt.strftime("%Y-%m"), "gold": round(float(r["gold"]), 2),
                 "real_rate": round(float(r["real_rate"]), 2)}
                for dt, r in aligned.iterrows()
            ]
            result["current"] = round(float(aligned.iloc[-1]["gold"]), 2)
            result["directional_score"] = 0.0
    except Exception as e:
        logger.error(f"Chart {cid} failed: {e}")
    return result


def _s2j(s):
    if s is None: return []
    return [{"date": dt.strftime("%Y-%m"), "value": round(float(v), 4)}
            for dt, v in s.items() if pd.notna(v)]


def compute_regime_block(rk, bd, all_data):
    logger.info(f"Regime: {rk} -- {bd['name']}")
    charts, wd, tw = [], 0.0, 0.0
    for cd in bd["charts"]:
        cr = compute_chart_data(cd, all_data)
        charts.append(cr)
        w = cd.get("chart_weight", 0.0)
        if w > 0: wd += cr["directional_score"] * w; tw += w
    act = round(wd / tw, 3) if tw > 0 else 0.5
    logger.info(f"  -> Activation: {act:.3f} ({'ACTIVE' if act >= ACTIVE_THRESHOLD else 'INACTIVE'})")
    return {"charts": charts, "activation": act, "active": act >= ACTIVE_THRESHOLD}


# ===================================================================
# 5. FRAGILITY INDICATORS
# ===================================================================

def compute_fragility(rk, fd, all_data):
    tr, th, di = fd["transform"], fd["threshold"], fd["threshold_direction"]
    try:
        if tr == "yoy_growth":
            sk = fd["series"]
            s = all_data.get(sk)
            if s is None: return {"status": "INACTIVE", "current_value": None}
            yoy = compute_yoy(to_monthly(s))
            if yoy is None or len(yoy) < 4: return {"status": "INACTIVE", "current_value": None}
            cur = float(yoy.iloc[-1])
            sq = fd.get("sustained_quarters", 4)
            qc = yoy.iloc[-sq * 3::3] if len(yoy) >= sq * 3 else yoy.iloc[-sq:]
            hit = cur > th if di == "above" else cur < th
            sus = all((float(v) > th if di == "above" else float(v) < th) for v in qc.values if pd.notna(v))
            if sus and len(qc) >= sq: return {"status": "ACTIVE", "current_value": round(cur, 2)}
            if hit: return {"status": "WATCH", "current_value": round(cur, 2)}
            return {"status": "INACTIVE", "current_value": round(cur, 2)}

        elif tr == "gdp_minus_gs10":
            gdp = all_data.get("GDP")
            gs10 = all_data.get("GS10")
            if gdp is None or gs10 is None: return {"status": "INACTIVE", "current_value": None}
            gy = compute_yoy(to_monthly(gdp))
            gm = to_monthly(gs10)
            if gy is None: return {"status": "INACTIVE", "current_value": None}
            al = pd.concat([gy, gm], axis=1, sort=True).dropna()
            if al.empty: return {"status": "INACTIVE", "current_value": None}
            al.columns = ["gy", "g10"]
            sp = al["gy"] - al["g10"]
            cur = float(sp.iloc[-1])
            sq = fd.get("sustained_quarters", 4)
            rc = sp.iloc[-sq * 3:] if len(sp) >= sq * 3 else sp
            hit = cur > th if di == "above" else cur < th
            sus = all((float(v) > th if di == "above" else float(v) < th) for v in rc.values if pd.notna(v))
            if sus and len(rc) >= sq: return {"status": "ACTIVE", "current_value": round(cur, 2)}
            if hit: return {"status": "WATCH", "current_value": round(cur, 2)}
            return {"status": "INACTIVE", "current_value": round(cur, 2)}

        elif tr == "real_rate":
            gs10 = all_data.get("GS10")
            cpi = all_data.get("CPIAUCSL")
            if gs10 is None or cpi is None: return {"status": "INACTIVE", "current_value": None}
            gm = to_monthly(gs10)
            cy = compute_yoy(to_monthly(cpi))
            if cy is None: return {"status": "INACTIVE", "current_value": None}
            al = pd.concat([gm, cy], axis=1, sort=True).dropna()
            if al.empty: return {"status": "INACTIVE", "current_value": None}
            al.columns = ["g10", "cy"]
            rr = al["g10"] - al["cy"]
            cur = float(rr.iloc[-1])
            sm = fd.get("sustained_months", 6)
            ab = cur > th
            if len(rr) >= sm:
                rc = rr.iloc[-sm:]
                rising = float(rc.iloc[-1]) > float(rc.iloc[0])
                aab = all(float(v) > th for v in rc.values)
            else:
                rising, aab = False, False
            if ab and aab and rising: return {"status": "ACTIVE", "current_value": round(cur, 2)}
            if ab: return {"status": "WATCH", "current_value": round(cur, 2)}
            return {"status": "INACTIVE", "current_value": round(cur, 2)}

        elif tr == "ratio_momentum_12m":
            gold = all_data.get("GOLD")
            spy = all_data.get("SPY")
            if gold is None or spy is None: return {"status": "INACTIVE", "current_value": None}
            ratio = compute_ratio(gold, spy)
            if ratio is None or len(ratio) < 13: return {"status": "INACTIVE", "current_value": None}
            mom = (float(ratio.iloc[-1]) / float(ratio.iloc[-13])) - 1.0
            bl = mom < th
            sm = fd.get("sustained_months", 12)
            sc = False
            if len(ratio) >= sm + 12:
                sc = True
                for i in range(sm):
                    ix, ix12 = -(i+1), -(i+1)-12
                    if abs(ix12) > len(ratio): sc = False; break
                    if (float(ratio.iloc[ix]) / float(ratio.iloc[ix12])) - 1.0 >= 0: sc = False; break
                sc = sc and bl
            if sc: return {"status": "ACTIVE", "current_value": round(mom * 100, 2)}
            if bl: return {"status": "WATCH", "current_value": round(mom * 100, 2)}
            return {"status": "INACTIVE", "current_value": round(mom * 100, 2)}
    except Exception as e:
        logger.error(f"Fragility {rk}: {e}")
    return {"status": "INACTIVE", "current_value": None}


# ===================================================================
# 6. TAILWIND + CONVICTION
# ===================================================================

def compute_tailwind_scores(rr):
    tw = {a: 0.0 for a in ASSET_CLASSES}
    mx = {a: 0.0 for a in ASSET_CLASSES}
    for rk in REGIME_ORDER:
        bd, w, act = REGIME_BLOCKS[rk], REGIME_WEIGHTS[rk], rr[rk]["activation"]
        for a in ASSET_CLASSES:
            imp = bd["asset_implications"].get(a, 0.0)
            tw[a] += act * imp * w; mx[a] += abs(imp) * w
    return {a: int(round(tw[a] / mx[a] * 100)) if mx[a] > 0 else 0 for a in ASSET_CLASSES}


def compute_conviction_summary(rr, ts):
    ac = sum(1 for r in REGIME_ORDER if rr[r]["active"])
    wa = sum(rr[r]["activation"] * REGIME_WEIGHTS[r] for r in REGIME_ORDER)
    ra = sum(1 for r in REGIME_ORDER if rr[r]["active"] and ROBUSTNESS_MAP[r] in ROBUST_CATEGORIES)
    fa = sum(1 for r in REGIME_ORDER if rr[r]["active"] and ROBUSTNESS_MAP[r] == "FRAGIL")
    pos = [a for a in ASSET_CLASSES if ts.get(a, 0) > 20]
    neg = [a for a in ASSET_CLASSES if ts.get(a, 0) < -20]
    if len(pos) >= 3 and "gold" in pos: conv = "REAL ASSETS BEVORZUGT"
    elif len(neg) >= 3: conv = "FINANCIAL ASSETS BEVORZUGT"
    else: conv = "GEMISCHT -- kein klares saekulares Regime"
    rs = {}
    for r in REGIME_ORDER:
        bd = REGIME_BLOCKS[r]
        rs[r] = {
            "name": bd["name"], "name_de": bd["name_de"],
            "activation": rr[r]["activation"], "active": rr[r]["active"],
            "robustness": ROBUSTNESS_MAP[r], "robustness_bar": bd["robustness_bar"],
            "horizon": bd["horizon"], "weight": REGIME_WEIGHTS[r],
            "fragility_indicator": rr[r].get("fragility_indicator", ""),
            "fragility_status": rr[r].get("fragility_status", "INACTIVE"),
            "fragility_detail": rr[r].get("fragility_detail", ""),
            "fragility_current_value": rr[r].get("fragility_current_value"),
        }
    return {
        "active_regimes": ac, "total_regimes": len(REGIME_ORDER),
        "weighted_activation": round(wa, 3), "convergence_direction": conv,
        "robust_active": ra, "fragile_active": fa,
        "regime_status": rs, "tailwind_scores": ts, "narrative": "",
    }


# ===================================================================
# 7. BEWERTUNGS-KASKADE
# ===================================================================

def calculate_half_life(rs):
    if rs is None or len(rs) < 36:
        return {"half_life_months": None, "significant": False, "r_squared": None, "p_value": None}
    try:
        c = rs[rs > 0].dropna()
        if len(c) < 36: return {"half_life_months": None, "significant": False, "r_squared": None, "p_value": None}
        lr = np.log(c.values); d = np.diff(lr); lg = lr[:-1]
        sl, _, rv, pv, _ = linregress(lg, d)
        if sl >= 0: return {"half_life_months": None, "significant": False, "r_squared": round(rv**2, 4), "p_value": round(pv, 4)}
        hl = -np.log(2) / sl
        return {"half_life_months": round(float(hl), 1), "significant": pv < 0.05, "r_squared": round(rv**2, 4), "p_value": round(pv, 4)}
    except Exception as e:
        logger.error(f"Half-life: {e}")
        return {"half_life_months": None, "significant": False, "r_squared": None, "p_value": None}

def estimate_normalization(h, sig):
    if not sig or h is None: return "Kein statistischer Rueckkehr-Trend nachweisbar"
    t = h * 2
    if t < 12: return "< 1 Jahr"
    elif t < 24: return "~1-2 Jahre"
    elif t < 36: return "~2-3 Jahre"
    elif t < 60: return "~3-5 Jahre"
    elif t < 120: return "~5-10 Jahre"
    else: return "> 10 Jahre"

def classify_signal(p, d):
    if p is None: return "KEINE DATEN"
    e = (100 - p) if d == "high_is_cheap" else p
    if e <= 10: return "EXTREM BILLIG"
    elif e <= 25: return "SEHR BILLIG"
    elif e <= 40: return "BILLIG"
    elif e <= 60: return "FAIR"
    elif e <= 75: return "TEUER"
    elif e <= 90: return "SEHR TEUER"
    else: return "EXTREM TEUER"

def compute_valuation_ratio(rk, rd, all_data):
    logger.info(f"  Ratio: {rk}")
    ds = rd.get("denominator_scale", 1.0)
    ratio = compute_ratio(all_data.get(rd["numerator"]), all_data.get(rd["denominator"]), denom_scale=ds)
    if ratio is None or len(ratio) < 24:
        logger.warning(f"  {rk}: insufficient data ({0 if ratio is None else len(ratio)} pts)"); return None
    cur = float(ratio.iloc[-1]); am = float(ratio.mean())
    r20 = ratio.rolling(240).mean()
    r20m = float(r20.iloc[-1]) if len(r20.dropna()) > 0 else None
    shift = r20m is not None and am != 0 and abs(r20m - am) / abs(am) * 100 > 30
    p = compute_percentile(ratio, cur); p20 = compute_percentile_20y(ratio, cur)
    sig = classify_signal(p, rd["direction"])
    hl = calculate_half_life(ratio); norm = estimate_normalization(hl["half_life_months"], hl["significant"])
    logger.info(f"    Pctl: {p}%, Signal: {sig}, HL: {hl['half_life_months']} Mo")
    return {
        "name": rd["name"], "current": round(cur, 4), "alltime_mean": round(am, 4),
        "rolling_20y_mean": round(r20m, 4) if r20m else None,
        "structural_shift_warning": shift,
        "percentile_alltime": p, "percentile_20y": p20, "signal": sig,
        "half_life": hl, "estimated_normalization": norm, "level": rd["level"],
        "fundamental_confirmation": "KEINE DATEN", "fundamental_factors": [],
        "combined_signal": sig,
    }

def build_cascade(ratios):
    cascade = []
    sm2 = ratios.get("SPY_M2")
    if sm2 and sm2["percentile_alltime"] and sm2["percentile_alltime"] > 60:
        cascade.append({"level": 1, "ratio": "SPY/M2", "signal": "AKTIEN REAL TEUER",
                        "percentile": sm2["percentile_alltime"], "implication": "-> Alternativen zu Financial Assets"})
        for k, n, thr in [("GOLD_SPY", "Gold/SPY", 40), ("DBC_SPY", "DBC/SPY", 40)]:
            r = ratios.get(k)
            if r and r["percentile_alltime"] and r["percentile_alltime"] < thr:
                cascade.append({"level": 1, "ratio": n, "signal": r["signal"],
                                "percentile": r["percentile_alltime"], "implication": f"-> {n.split('/')[0]} bevorzugt"})
    e2 = []
    for k, asset, thr_hi in [("GOLD_SILVER", "Silber", True), ("OIL_GOLD", "Oel", False), ("COPPER_GOLD", "Kupfer", False)]:
        r = ratios.get(k)
        if not r or r["percentile_alltime"] is None: continue
        if thr_hi and r["percentile_alltime"] > 70:
            e2.append({"asset": asset, "vs": "Gold", "percentile": r["percentile_alltime"], "signal": r["signal"]})
        elif not thr_hi and r["percentile_alltime"] < 30:
            e2.append({"asset": asset, "vs": "Gold", "percentile": r["percentile_alltime"], "signal": r["signal"]})
    e2.sort(key=lambda x: abs(x["percentile"] - 50), reverse=True)
    for it in e2:
        cascade.append({"level": 2, "ratio": f"{it['asset']}/{it['vs']}", "signal": it["signal"],
                        "percentile": it["percentile"], "implication": f"-> {it['asset']} innerhalb Real Assets"})
    if cascade:
        st = max(cascade, key=lambda x: abs(x["percentile"] - 50))
        sm = {"strongest_signal": {"ratio": st["ratio"], "signal": st["signal"], "percentile": st["percentile"]},
              "chain_length": len(cascade), "direction": "REAL ASSETS" if any(c["level"] == 1 for c in cascade) else "UNKLAR"}
    else:
        sm = {"strongest_signal": None, "chain_length": 0, "direction": "KEIN KLARES SIGNAL"}
    return cascade, sm

def compute_valuation_cascade(all_data):
    logger.info("Computing valuation cascade...")
    ratios = {}
    for rk, rd in VALUATION_RATIOS.items():
        r = compute_valuation_ratio(rk, rd, all_data)
        if r: ratios[rk] = r
    cas, sm = build_cascade(ratios)
    logger.info(f"  Cascade: {sm['chain_length']} steps, dir: {sm['direction']}")
    return {"ratios": ratios, "cascade": cas, "cascade_summary": sm, "narrative": ""}


# ===================================================================
# 8. LLM INTEGRATION
# ===================================================================

SECULAR_SYSTEM_PROMPT = """Du bist der Saekulare-Trends-Analyst von Baldur Creek Capital, einem systematischen Macro-Hedgefund.

Deine Aufgabe: Interpretiere die berechneten saekularen Regime-Daten und formuliere ein Investor-Briefing auf Deutsch.

REGELN:
1. Schreibe auf Deutsch, professioneller Investoren-Ton (wie Bridgewater Daily Observations)
2. Kein "KI" oder "AI" in deinem Text
3. Jede Aussage muss durch die gelieferten Daten oder Web-Search-Ergebnisse gestuetzt sein
4. Keine erfundenen Zahlen -- nur Zahlen die in den Inputs stehen oder die du per Web Search findest
5. Sei direkt und meinungsstark -- kein "es koennte sein" wenn die Daten klar sind
6. Erwaehne Fragilitaets-Indikatoren nur wenn sie auf WATCH oder ACTIVE stehen
7. Halte dich an die kausale Kette: Demografie -> Deglobalisierung -> Fiscal Dominance -> Financial Repression -> Great Divergence

KRITISCH -- OUTPUT-FORMAT:
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt. KEIN Text vor dem JSON. KEIN Markdown. KEINE Backticks.
Deine Antwort muss DIREKT mit { beginnen und mit } enden. Nichts davor, nichts danach.
{
    "regime_narratives": {
        "demographic_cliff": "3-5 Saetze...",
        "deglobalization": "3-5 Saetze...",
        "fiscal_dominance": "3-5 Saetze...",
        "financial_repression": "3-5 Saetze...",
        "great_divergence": "3-5 Saetze..."
    },
    "conviction_narrative": "3-5 Saetze Gesamtbild...",
    "fundamental_assessments": [
        {"ratio": "RATIO_KEY", "factors": [{"factor": "...", "detail": "...", "direction": "BESTAETIGT"}]}
    ],
    "valuation_narrative": "3-5 Saetze Bewertungs-Kaskade..."
}"""


def build_llm_input(cs, rr, vc):
    sec = [f"## CONVICTION\nAktiv: {cs['active_regimes']}/{cs['total_regimes']}\nActivation: {cs['weighted_activation']:.2f}\nRichtung: {cs['convergence_direction']}\nTailwinds: {json.dumps(cs['tailwind_scores'])}"]
    for rk in REGIME_ORDER:
        bd, r, rs = REGIME_BLOCKS[rk], rr[rk], cs["regime_status"][rk]
        sec.append(f"\n## {bd['name']}\nActivation: {r['activation']:.2f} ({'AKTIV' if r['active'] else 'INAKTIV'})\nRobust: {rs['robustness']} | {rs['horizon']}\nFragility: {rs.get('fragility_status','INACTIVE')}")
        for c in r["charts"]:
            sec.append(f"  - {c['name']}: {c['current']}, Pzl={c.get('percentile')}%")
    sec.append("\n## KASKADE")
    for rk, rv in vc["ratios"].items():
        h = rv["half_life"]
        sec.append(f"  {rk}: {rv['current']}, Pzl={rv['percentile_alltime']}%, {rv['signal']}, HL={h['half_life_months']}Mo")
    ext = [(k, v) for k, v in vc["ratios"].items() if v["percentile_alltime"] and (v["percentile_alltime"] > 80 or v["percentile_alltime"] < 20)]
    if ext:
        sec.append("\n## FUNDAMENTAL-BESTAETIGUNG (Web Search)")
        for k, v in ext:
            a = v["name"].split("/")[0].strip()
            sec.append(f"### {k} (Pzl: {v['percentile_alltime']}%)\nSuche: Produktionsdefizite {a}, Nachfrage-Treiber, Capex, Lagerbestaende")
    return "\n".join(sec)


def parse_llm_response(resp):
    """Parse LLM response, robustly extracting JSON even if preceded by prose."""
    txt = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not txt:
        logger.error("LLM returned no text content")
        return None

    # Strategy 1: Find first { and last }
    first_brace = txt.find("{")
    last_brace = txt.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidate = txt[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 2: ```json ... ``` block
    match = re.search(r'```json\s*(.*?)```', txt, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Strip prefixes
    for prefix in ["```json", "```"]:
        if txt.startswith(prefix):
            txt = txt[len(prefix):]
    if txt.endswith("```"):
        txt = txt[:-3]
    try:
        return json.loads(txt.strip())
    except json.JSONDecodeError:
        logger.error(f"LLM parse fail -- first 500 chars: {txt[:500]}")
        return None


def calc_fund_conf(factors):
    if not factors: return "KEINE DATEN"
    s = sum((1 if f.get("direction") == "BESTAETIGT" else -1 if f.get("direction") == "WIDERSPRICHT" else 0) for f in factors) / len(factors)
    if s > 0.5: return "STARK BESTAETIGT"
    elif s > 0: return "TEILWEISE BESTAETIGT"
    elif s > -0.5: return "GEMISCHT"
    else: return "WIDERSPRICHT"


def calc_combined(ps, fc):
    ps_map = {"EXTREM BILLIG": 5, "SEHR BILLIG": 4, "BILLIG": 3, "FAIR": 0, "TEUER": -3, "SEHR TEUER": -4, "EXTREM TEUER": -5}
    fm = {"STARK BESTAETIGT": 1.5, "TEILWEISE BESTAETIGT": 1.2, "KEINE DATEN": 1.0, "GEMISCHT": 0.8, "WIDERSPRICHT": 0.5}
    c = ps_map.get(ps, 0) * fm.get(fc, 1.0)
    if c >= 6: return "EXTREM STARK"
    elif c >= 4: return "STARK"
    elif c >= 2: return "MODERAT"
    elif c >= -2: return "NEUTRAL"
    elif c >= -4: return "MODERAT NEGATIV"
    else: return "STARK NEGATIV"


def run_llm(cs, rr, vc):
    try: import anthropic
    except: logger.error("anthropic not installed"); return {}, False
    if not os.environ.get("ANTHROPIC_API_KEY"): logger.error("No API key"); return {}, False
    try:
        client = anthropic.Anthropic()
        logger.info("Calling LLM...")
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=LLM_MAX_TOKENS,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=SECULAR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_llm_input(cs, rr, vc)}])
        parsed = parse_llm_response(resp)
        if parsed: logger.info("LLM OK"); return parsed, True
        return {}, False
    except Exception as e:
        logger.error(f"LLM failed: {e}"); return {}, False


def apply_llm(ld, cs, rr, vc):
    for rk in REGIME_ORDER:
        rr[rk]["narrative"] = ld.get("regime_narratives", {}).get(rk, "")
    cs["narrative"] = ld.get("conviction_narrative", "")
    for a in ld.get("fundamental_assessments", []):
        rk, factors = a.get("ratio", ""), a.get("factors", [])
        if rk in vc["ratios"]:
            rv = vc["ratios"][rk]
            rv["fundamental_factors"] = factors
            rv["fundamental_confirmation"] = calc_fund_conf(factors)
            rv["combined_signal"] = calc_combined(rv["signal"], rv["fundamental_confirmation"])
    vc["narrative"] = ld.get("valuation_narrative", "")
    return cs, rr, vc


# ===================================================================
# 9. JSON + GIT
# ===================================================================

def write_json(cs, rr, vc, llm_ok):
    now = datetime.now(timezone.utc)
    regimes = {r: {"charts": rr[r]["charts"], "narrative": rr[r].get("narrative", "")} for r in REGIME_ORDER}
    out = {
        "metadata": {"generated_at": now.isoformat(), "version": "1.5",
                     "data_through": now.strftime("%Y-%m-%d"),
                     "fred_series_count": len(FRED_SERIES), "yfinance_series_count": len(YFINANCE_TICKERS),
                     "llm_model": CLAUDE_MODEL if llm_ok else "", "llm_success": llm_ok},
        "conviction_summary": cs, "regimes": regimes, "valuation_cascade": vc,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Written: {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE):,} bytes)")


def git_push():
    try:
        subprocess.run(["git", "add", "step_0w_secular/data/secular_trends.json"], check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
            logger.info("No changes"); return
        subprocess.run(["git", "commit", "-m", "Secular Trends monthly update"], check=True)
        subprocess.run(["git", "pull", "--rebase"], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info("Git push OK")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git: {e}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--skip-git", action="store_true")
    pa.add_argument("--skip-llm", action="store_true")
    args = pa.parse_args()

    logger.info("=" * 60)
    logger.info("SECULAR TRENDS PIPELINE -- V1.5")
    logger.info("=" * 60)

    all_data = fetch_all_data()

    rr = {}
    for rk in REGIME_ORDER:
        bd = REGIME_BLOCKS[rk]
        r = compute_regime_block(rk, bd, all_data)
        fd = FRAGILITY_INDICATORS.get(rk)
        if fd:
            fr = compute_fragility(rk, fd, all_data)
            r["fragility_indicator"] = fd["name"]
            r["fragility_status"] = fr["status"]
            r["fragility_detail"] = fd["description_de"]
            r["fragility_current_value"] = fr.get("current_value")
        rr[rk] = r

    ts = compute_tailwind_scores(rr)
    for a, s in ts.items(): logger.info(f"  {ASSET_CLASS_LABELS.get(a,a):20s} -> {s:+d}%")

    cs = compute_conviction_summary(rr, ts)
    logger.info(f"  Active: {cs['active_regimes']}/{cs['total_regimes']}, Conv: {cs['convergence_direction']}")

    vc = compute_valuation_cascade(all_data)

    llm_ok = False
    if not args.skip_llm:
        ld, llm_ok = run_llm(cs, rr, vc)
        if llm_ok: cs, rr, vc = apply_llm(ld, cs, rr, vc)

    write_json(cs, rr, vc, llm_ok)
    if not args.skip_git: git_push()

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
