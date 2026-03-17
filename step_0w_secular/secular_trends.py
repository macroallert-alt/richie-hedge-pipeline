"""
Säkulare Trends Circle — Main Script
Baldur Creek Capital | Step 0w (V1.2 — Etappe 1+2+3: Regime + Kaskade + LLM)

Pipeline:
  1. FRED + EOD Daten fetchen (22 FRED + 2 EOD = 24 Serien)
  2. Regime-Blöcke berechnen (Ratios, Perzentile, Directional Scores)
  3. Fragilitäts-Indikatoren berechnen
  4. Regime-Activation Scores + Gewichteter Tailwind Score
  5. Conviction Summary
  6. Bewertungs-Kaskade (6 Ratios, Perzentile, Half-Life O-U, Kaskaden-Logik)
  7. LLM-Call (Narrativ + Web Search Fundamental-Bestätigung)
  8. Combined Signal berechnen (Preis × Fundamental Multiplier)
  9. JSON schreiben
  10. Git commit + push

Usage:
  python -m step_0w_secular.secular_trends [--skip-git] [--skip-llm]
"""

import argparse
import json
import logging
import os
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
    FRED_SERIES,
    FRED_BASE_URL,
    EOD_TICKERS,
    EOD_BASE_URL,
    REGIME_BLOCKS,
    REGIME_WEIGHTS,
    REGIME_ORDER,
    FRAGILITY_INDICATORS,
    ASSET_CLASSES,
    ASSET_CLASS_LABELS,
    ACTIVE_THRESHOLD,
    ROBUSTNESS_MAP,
    ROBUST_CATEGORIES,
    VALUATION_RATIOS,
    CLAUDE_MODEL,
    LLM_MAX_TOKENS,
    DATA_DIR,
    OUTPUT_FILE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_fred(series_id, fred_api_key):
    """Fetch a single FRED series. Returns pandas Series (DatetimeIndex, float)."""
    url = f"{FRED_BASE_URL}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
        "sort_order": "asc",
        "observation_start": "1940-01-01",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        if not obs:
            logger.warning(f"FRED {series_id}: no observations returned")
            return None

        dates, values = [], []
        for o in obs:
            if o["value"] == ".":
                continue
            dates.append(pd.Timestamp(o["date"]))
            values.append(float(o["value"]))

        if not dates:
            logger.warning(f"FRED {series_id}: all values are '.'")
            return None

        s = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
        s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()
        logger.info(f"FRED {series_id}: {len(s)} obs, {s.index[0].date()} → {s.index[-1].date()}")
        return s

    except Exception as e:
        logger.error(f"FRED {series_id} fetch failed: {e}")
        return None


def fetch_eod_monthly(ticker, eod_api_key):
    """Fetch monthly close prices from EOD. Returns pandas Series."""
    url = f"{EOD_BASE_URL}/eod/{ticker}"
    params = {
        "api_token": eod_api_key,
        "fmt": "json",
        "period": "m",
        "from": "1940-01-01",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            logger.warning(f"EOD {ticker}: no data returned")
            return None

        dates = [pd.Timestamp(d["date"]) for d in data]
        closes = [float(d["adjusted_close"]) for d in data]

        s = pd.Series(closes, index=pd.DatetimeIndex(dates), name=ticker)
        s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()
        logger.info(f"EOD {ticker}: {len(s)} obs, {s.index[0].date()} → {s.index[-1].date()}")
        return s

    except Exception as e:
        logger.error(f"EOD {ticker} fetch failed: {e}")
        return None


def fetch_all_data():
    """Fetch all 22 FRED + 2 EOD series. Returns dict key → pd.Series."""
    fred_api_key = os.environ.get("FRED_API_KEY", "")
    eod_api_key = os.environ.get("EODHD_API_KEY", "")

    if not fred_api_key:
        logger.error("FRED_API_KEY not set")
        sys.exit(1)
    if not eod_api_key:
        logger.error("EODHD_API_KEY not set")
        sys.exit(1)

    all_data = {}

    for key, series_id in FRED_SERIES.items():
        s = fetch_fred(series_id, fred_api_key)
        if s is not None:
            all_data[key] = s

    for key, ticker in EOD_TICKERS.items():
        s = fetch_eod_monthly(ticker, eod_api_key)
        if s is not None:
            all_data[key] = s

    logger.info(f"Fetched {len(all_data)}/{len(FRED_SERIES) + len(EOD_TICKERS)} series")

    total_expected = len(FRED_SERIES) + len(EOD_TICKERS)
    if len(all_data) < total_expected * 0.5:
        logger.error(f"Only {len(all_data)}/{total_expected} series fetched — aborting")
        sys.exit(1)

    return all_data


# ═══════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def to_monthly(s):
    """Resample any series to month-end frequency (last value)."""
    if s is None:
        return None
    return s.resample("ME").last().dropna()


def compute_yoy(s):
    """Compute YoY growth rate (%) from a level series."""
    if s is None or len(s) < 13:
        return None
    return s.pct_change(12) * 100


def compute_ratio(numerator, denominator, multiply=1.0, denom_scale=1.0):
    """Compute ratio of two series aligned by date."""
    if numerator is None or denominator is None:
        return None
    num_m = to_monthly(numerator)
    den_m = to_monthly(denominator)
    aligned = pd.concat([num_m, den_m], axis=1).dropna()
    if aligned.empty:
        return None
    ratio = (aligned.iloc[:, 0] / (aligned.iloc[:, 1] / denom_scale)) * multiply
    ratio.name = "ratio"
    return ratio


def compute_percentile(series, current_value):
    """Compute historical percentile of current_value within series."""
    if series is None or len(series) < 12:
        return None
    return round(float((series < current_value).sum() / len(series) * 100), 1)


def compute_percentile_20y(series, current_value):
    """Compute percentile within last 20 years only."""
    if series is None or len(series) < 12:
        return None
    cutoff = max(0, len(series) - 240)
    recent = series.iloc[cutoff:]
    if len(recent) < 12:
        return None
    return round(float((recent < current_value).sum() / len(recent) * 100), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. DIRECTIONAL SCORES
# ═══════════════════════════════════════════════════════════════════════════

def calc_directional_score(method, series, percentile, all_data):
    """Calculate directional score (0.0-1.0). 0=inactive, 0.5=neutral, 1=active."""
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
        if not isinstance(series, dict):
            return 0.5
        scores = []
        for key, yoy_val in series.items():
            if yoy_val is None:
                scores.append(0.5)
            elif yoy_val < 0:
                scores.append(1.0)
            elif yoy_val <= 1.0:
                scores.append(0.7)
            else:
                scores.append(0.3)
        return round(np.mean(scores), 3) if scores else 0.5

    elif method == "mfg_employment":
        if series is None or len(series) < 25:
            return 0.5
        change_24m = series.iloc[-1] - series.iloc[-25] if len(series) >= 25 else 0
        return 0.8 if change_24m > 0 else 0.3

    elif method == "trade_deficit":
        if series is None or len(series) < 12:
            return 0.5
        current = series.iloc[-1]
        max_deficit = series.min()
        if max_deficit >= 0:
            return 0.3
        return round(min(abs(current) / abs(max_deficit), 1.0), 3)

    elif method == "interest_defense_ratio":
        if series is None:
            return 0.5
        ratio_val = float(series)
        return round(min(max(ratio_val, 0.0), 1.0), 3)

    elif method == "real_rate":
        if series is None:
            return 0.5
        real_rate = float(series)
        return round(1.0 - max(0.0, min(1.0, (real_rate + 2.0) / 6.0)), 3)

    else:
        logger.warning(f"Unknown directional score method: {method}")
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 4. REGIME BLOCK COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_chart_data(chart_def, all_data):
    """Compute a single chart's data, current value, percentile, directional score."""
    chart_id = chart_def["id"]
    chart_type = chart_def["type"]
    method = chart_def["directional_score_method"]

    result = {
        "id": chart_id,
        "name": chart_def["name"],
        "type": chart_type,
        "unit": chart_def.get("unit", ""),
        "current": None,
        "alltime_mean": None,
        "rolling_20y_mean": None,
        "percentile": None,
        "percentile_20y": None,
        "directional_score": 0.5,
        "data": [],
        "annotations": chart_def.get("annotations", []),
    }

    if "reference_line" in chart_def:
        result["reference_line"] = chart_def["reference_line"]
    if "lines" in chart_def:
        result["lines"] = chart_def["lines"]
    if "color_zones" in chart_def:
        result["color_zones"] = chart_def["color_zones"]

    try:
        if chart_type == "single_line":
            key = chart_def["series"][0]
            s = all_data.get(key)
            if s is None:
                logger.warning(f"Chart {chart_id}: missing series {key}")
                return result
            s = to_monthly(s)
            current = float(s.iloc[-1])
            result["current"] = round(current, 2)
            result["alltime_mean"] = round(float(s.mean()), 2)
            rolling_20y = s.rolling(240).mean()
            result["rolling_20y_mean"] = round(float(rolling_20y.iloc[-1]), 2) if len(rolling_20y.dropna()) > 0 else None
            result["percentile"] = compute_percentile(s, current)
            result["percentile_20y"] = compute_percentile_20y(s, current)
            result["directional_score"] = calc_directional_score(method, s, result["percentile"], all_data)
            result["data"] = _series_to_json(s)

        elif chart_type == "multi_line":
            lines_data = {}
            latest_yoy = {}
            for line_def in chart_def["lines"]:
                key = line_def["key"]
                s = all_data.get(key)
                if s is None:
                    logger.warning(f"Chart {chart_id}: missing series {key}")
                    continue
                s = to_monthly(s)
                if chart_def.get("transform") == "yoy_growth":
                    s = compute_yoy(s)
                    if s is None:
                        continue
                lines_data[key] = s
                latest_yoy[key] = float(s.iloc[-1]) if len(s) > 0 else None

            if lines_data:
                merged = pd.DataFrame(lines_data).dropna(how="all")
                data_points = []
                for dt, row in merged.iterrows():
                    point = {"date": dt.strftime("%Y-%m")}
                    for k in lines_data:
                        val = row.get(k)
                        point[k] = round(float(val), 3) if pd.notna(val) else None
                    data_points.append(point)
                result["data"] = data_points

            result["directional_score"] = calc_directional_score(method, latest_yoy, None, all_data)
            valid_yoy = [v for v in latest_yoy.values() if v is not None]
            result["current"] = round(np.mean(valid_yoy), 2) if valid_yoy else None

        elif chart_type == "ratio":
            num_key = chart_def["numerator"]
            den_key = chart_def["denominator"]
            multiply = chart_def.get("multiply", 1.0)
            denom_scale = chart_def.get("denominator_scale", 1.0)
            ratio = compute_ratio(all_data.get(num_key), all_data.get(den_key),
                                  multiply=multiply, denom_scale=denom_scale)
            if ratio is None or len(ratio) < 12:
                logger.warning(f"Chart {chart_id}: could not compute ratio {num_key}/{den_key}")
                return result

            current = float(ratio.iloc[-1])
            result["current"] = round(current, 4)
            result["alltime_mean"] = round(float(ratio.mean()), 4)
            rolling_20y = ratio.rolling(240).mean()
            result["rolling_20y_mean"] = round(float(rolling_20y.iloc[-1]), 4) if len(rolling_20y.dropna()) > 0 else None
            result["percentile"] = compute_percentile(ratio, current)
            result["percentile_20y"] = compute_percentile_20y(ratio, current)
            result["directional_score"] = calc_directional_score(method, ratio, result["percentile"], all_data)
            result["data"] = _series_to_json(ratio)
            if result["alltime_mean"] and result["rolling_20y_mean"]:
                diff_pct = abs(result["rolling_20y_mean"] - result["alltime_mean"]) / abs(result["alltime_mean"]) * 100
                result["structural_shift_warning"] = diff_pct > 30

        elif chart_type == "dual_line":
            lines_out = {}
            for line_def in chart_def["lines"]:
                key = line_def["key"]
                s = all_data.get(key)
                if s is None:
                    continue
                lines_out[key] = to_monthly(s)

            if len(lines_out) == 2:
                keys = list(lines_out.keys())
                merged = pd.concat([lines_out[keys[0]], lines_out[keys[1]]], axis=1).dropna()
                merged.columns = keys
                data_points = []
                for dt, row in merged.iterrows():
                    point = {"date": dt.strftime("%Y-%m")}
                    for k in keys:
                        point[k] = round(float(row[k]), 2)
                    data_points.append(point)
                result["data"] = data_points
                interest_val = float(merged.iloc[-1][keys[0]])
                defense_val = float(merged.iloc[-1][keys[1]])
                total = interest_val + defense_val
                if total > 0:
                    result["directional_score"] = calc_directional_score(
                        method, interest_val / total, None, all_data)
                result["current"] = round(interest_val, 2)

        elif chart_type == "computed_real_rate":
            gs10 = all_data.get("GS10")
            cpi = all_data.get("CPIAUCSL")
            if gs10 is None or cpi is None:
                return result
            gs10_m = to_monthly(gs10)
            cpi_yoy = compute_yoy(to_monthly(cpi))
            if cpi_yoy is None:
                return result
            aligned = pd.concat([gs10_m, cpi_yoy], axis=1).dropna()
            if aligned.empty:
                return result
            aligned.columns = ["gs10", "cpi_yoy"]
            real_rate = aligned["gs10"] - aligned["cpi_yoy"]
            current = float(real_rate.iloc[-1])
            result["current"] = round(current, 2)
            result["alltime_mean"] = round(float(real_rate.mean()), 2)
            rolling_20y = real_rate.rolling(240).mean()
            result["rolling_20y_mean"] = round(float(rolling_20y.iloc[-1]), 2) if len(rolling_20y.dropna()) > 0 else None
            result["percentile"] = compute_percentile(real_rate, current)
            result["percentile_20y"] = compute_percentile_20y(real_rate, current)
            result["directional_score"] = calc_directional_score(method, current, result["percentile"], all_data)
            result["data"] = _series_to_json(real_rate)

        elif chart_type == "dual_axis_gold_realrate":
            gold = all_data.get("GOLD")
            gs10 = all_data.get("GS10")
            cpi = all_data.get("CPIAUCSL")
            if gold is None or gs10 is None or cpi is None:
                return result
            gold_m = to_monthly(gold)
            gs10_m = to_monthly(gs10)
            cpi_yoy = compute_yoy(to_monthly(cpi))
            if cpi_yoy is None:
                return result
            aligned = pd.concat([gold_m, gs10_m, cpi_yoy], axis=1).dropna()
            if aligned.empty:
                return result
            aligned.columns = ["gold", "gs10", "cpi_yoy"]
            aligned["real_rate"] = aligned["gs10"] - aligned["cpi_yoy"]
            data_points = []
            for dt, row in aligned.iterrows():
                data_points.append({
                    "date": dt.strftime("%Y-%m"),
                    "gold": round(float(row["gold"]), 2),
                    "real_rate": round(float(row["real_rate"]), 2),
                })
            result["data"] = data_points
            result["current"] = round(float(aligned.iloc[-1]["gold"]), 2)
            result["directional_score"] = 0.0

    except Exception as e:
        logger.error(f"Chart {chart_id} computation failed: {e}")

    return result


def _series_to_json(s):
    """Convert pandas Series to list of {date, value} dicts."""
    if s is None:
        return []
    return [{"date": dt.strftime("%Y-%m"), "value": round(float(val), 4)}
            for dt, val in s.items() if pd.notna(val)]


def compute_regime_block(regime_key, block_def, all_data):
    """Compute a full regime block: charts + activation score."""
    logger.info(f"Computing regime: {regime_key} — {block_def['name']}")
    charts_output = []
    weighted_directional = 0.0
    total_weight = 0.0

    for chart_def in block_def["charts"]:
        chart_result = compute_chart_data(chart_def, all_data)
        charts_output.append(chart_result)
        weight = chart_def.get("chart_weight", 0.0)
        if weight > 0:
            weighted_directional += chart_result["directional_score"] * weight
            total_weight += weight

    activation = round(weighted_directional / total_weight, 3) if total_weight > 0 else 0.5
    is_active = activation >= ACTIVE_THRESHOLD
    logger.info(f"  → Activation: {activation:.3f} ({'ACTIVE' if is_active else 'INACTIVE'})")

    return {"charts": charts_output, "activation": activation, "active": is_active}


# ═══════════════════════════════════════════════════════════════════════════
# 5. FRAGILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def compute_fragility(regime_key, frag_def, all_data):
    """Compute fragility indicator status: INACTIVE / WATCH / ACTIVE."""
    transform = frag_def["transform"]
    threshold = frag_def["threshold"]
    direction = frag_def["threshold_direction"]

    try:
        if transform == "yoy_growth":
            series_key = frag_def["series"]
            s = all_data.get(series_key)
            if s is None:
                return {"status": "INACTIVE", "current_value": None}
            yoy = compute_yoy(to_monthly(s))
            if yoy is None or len(yoy) < 4:
                return {"status": "INACTIVE", "current_value": None}
            current = float(yoy.iloc[-1])
            sustained = frag_def.get("sustained_quarters", 4)
            quarterly_checks = yoy.iloc[-sustained * 3::3] if len(yoy) >= sustained * 3 else yoy.iloc[-sustained:]
            if direction == "above":
                hit = current > threshold
                sustained_hit = all(float(v) > threshold for v in quarterly_checks.values if pd.notna(v))
            else:
                hit = current < threshold
                sustained_hit = all(float(v) < threshold for v in quarterly_checks.values if pd.notna(v))
            if sustained_hit and len(quarterly_checks) >= sustained:
                return {"status": "ACTIVE", "current_value": round(current, 2)}
            elif hit:
                return {"status": "WATCH", "current_value": round(current, 2)}
            return {"status": "INACTIVE", "current_value": round(current, 2)}

        elif transform == "gdp_minus_gs10":
            gdp = all_data.get("GDP")
            gs10 = all_data.get("GS10")
            if gdp is None or gs10 is None:
                return {"status": "INACTIVE", "current_value": None}
            gdp_yoy = compute_yoy(to_monthly(gdp))
            gs10_m = to_monthly(gs10)
            if gdp_yoy is None:
                return {"status": "INACTIVE", "current_value": None}
            aligned = pd.concat([gdp_yoy, gs10_m], axis=1).dropna()
            if aligned.empty:
                return {"status": "INACTIVE", "current_value": None}
            aligned.columns = ["gdp_yoy", "gs10"]
            spread = aligned["gdp_yoy"] - aligned["gs10"]
            current = float(spread.iloc[-1])
            sustained = frag_def.get("sustained_quarters", 4)
            recent = spread.iloc[-sustained * 3:] if len(spread) >= sustained * 3 else spread
            hit = current > threshold if direction == "above" else current < threshold
            sustained_hit = all(
                (float(v) > threshold if direction == "above" else float(v) < threshold)
                for v in recent.values if pd.notna(v))
            if sustained_hit and len(recent) >= sustained:
                return {"status": "ACTIVE", "current_value": round(current, 2)}
            elif hit:
                return {"status": "WATCH", "current_value": round(current, 2)}
            return {"status": "INACTIVE", "current_value": round(current, 2)}

        elif transform == "real_rate":
            gs10 = all_data.get("GS10")
            cpi = all_data.get("CPIAUCSL")
            if gs10 is None or cpi is None:
                return {"status": "INACTIVE", "current_value": None}
            gs10_m = to_monthly(gs10)
            cpi_yoy = compute_yoy(to_monthly(cpi))
            if cpi_yoy is None:
                return {"status": "INACTIVE", "current_value": None}
            aligned = pd.concat([gs10_m, cpi_yoy], axis=1).dropna()
            if aligned.empty:
                return {"status": "INACTIVE", "current_value": None}
            aligned.columns = ["gs10", "cpi_yoy"]
            real_rate = aligned["gs10"] - aligned["cpi_yoy"]
            current = float(real_rate.iloc[-1])
            sustained_months = frag_def.get("sustained_months", 6)
            above = current > threshold
            if len(real_rate) >= sustained_months:
                recent = real_rate.iloc[-sustained_months:]
                is_rising = float(recent.iloc[-1]) > float(recent.iloc[0])
                all_above = all(float(v) > threshold for v in recent.values)
            else:
                is_rising = False
                all_above = False
            if above and all_above and is_rising:
                return {"status": "ACTIVE", "current_value": round(current, 2)}
            elif above:
                return {"status": "WATCH", "current_value": round(current, 2)}
            return {"status": "INACTIVE", "current_value": round(current, 2)}

        elif transform == "ratio_momentum_12m":
            gold = all_data.get("GOLD")
            spy = all_data.get("SPY")
            if gold is None or spy is None:
                return {"status": "INACTIVE", "current_value": None}
            ratio = compute_ratio(gold, spy)
            if ratio is None or len(ratio) < 13:
                return {"status": "INACTIVE", "current_value": None}
            momentum = (float(ratio.iloc[-1]) / float(ratio.iloc[-13])) - 1.0
            below = momentum < threshold
            sustained_months = frag_def.get("sustained_months", 12)
            sustained_check = False
            if len(ratio) >= sustained_months + 12:
                sustained_check = True
                for i in range(sustained_months):
                    idx = -(i + 1)
                    idx_12 = idx - 12
                    if abs(idx_12) > len(ratio):
                        sustained_check = False
                        break
                    m = (float(ratio.iloc[idx]) / float(ratio.iloc[idx_12])) - 1.0
                    if m >= 0:
                        sustained_check = False
                        break
                sustained_check = sustained_check and below
            if sustained_check:
                return {"status": "ACTIVE", "current_value": round(momentum * 100, 2)}
            elif below:
                return {"status": "WATCH", "current_value": round(momentum * 100, 2)}
            return {"status": "INACTIVE", "current_value": round(momentum * 100, 2)}

    except Exception as e:
        logger.error(f"Fragility {regime_key}: {e}")
    return {"status": "INACTIVE", "current_value": None}


# ═══════════════════════════════════════════════════════════════════════════
# 6. TAILWIND SCORES + CONVICTION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def compute_tailwind_scores(regime_results):
    """Compute Secular Tailwind Score per asset class (-100% to +100%)."""
    tailwinds = {a: 0.0 for a in ASSET_CLASSES}
    max_possible = {a: 0.0 for a in ASSET_CLASSES}
    for rk in REGIME_ORDER:
        block = REGIME_BLOCKS[rk]
        w = REGIME_WEIGHTS[rk]
        act = regime_results[rk]["activation"]
        for a in ASSET_CLASSES:
            imp = block["asset_implications"].get(a, 0.0)
            tailwinds[a] += act * imp * w
            max_possible[a] += 1.0 * abs(imp) * w
    return {a: int(round(tailwinds[a] / max_possible[a] * 100, 0)) if max_possible[a] > 0 else 0
            for a in ASSET_CLASSES}


def compute_conviction_summary(regime_results, tailwind_scores):
    """Build the Conviction Summary block."""
    active_count = sum(1 for r in REGIME_ORDER if regime_results[r]["active"])
    weighted_act = sum(regime_results[r]["activation"] * REGIME_WEIGHTS[r] for r in REGIME_ORDER)
    robust_active = sum(1 for r in REGIME_ORDER if regime_results[r]["active"] and ROBUSTNESS_MAP[r] in ROBUST_CATEGORIES)
    fragile_active = sum(1 for r in REGIME_ORDER if regime_results[r]["active"] and ROBUSTNESS_MAP[r] == "FRAGIL")

    pos = [a for a in ASSET_CLASSES if tailwind_scores.get(a, 0) > 20]
    neg = [a for a in ASSET_CLASSES if tailwind_scores.get(a, 0) < -20]
    if len(pos) >= 3 and "gold" in pos:
        convergence = "REAL ASSETS BEVORZUGT"
    elif len(neg) >= 3:
        convergence = "FINANCIAL ASSETS BEVORZUGT"
    else:
        convergence = "GEMISCHT — kein klares säkulares Regime"

    regime_status = {}
    for r in REGIME_ORDER:
        bd = REGIME_BLOCKS[r]
        regime_status[r] = {
            "name": bd["name"], "name_de": bd["name_de"],
            "activation": regime_results[r]["activation"], "active": regime_results[r]["active"],
            "robustness": ROBUSTNESS_MAP[r], "robustness_bar": bd["robustness_bar"],
            "horizon": bd["horizon"], "weight": REGIME_WEIGHTS[r],
            "fragility_indicator": regime_results[r].get("fragility_indicator", ""),
            "fragility_status": regime_results[r].get("fragility_status", "INACTIVE"),
            "fragility_detail": regime_results[r].get("fragility_detail", ""),
            "fragility_current_value": regime_results[r].get("fragility_current_value"),
        }

    return {
        "active_regimes": active_count, "total_regimes": len(REGIME_ORDER),
        "weighted_activation": round(weighted_act, 3), "convergence_direction": convergence,
        "robust_active": robust_active, "fragile_active": fragile_active,
        "regime_status": regime_status, "tailwind_scores": tailwind_scores,
        "narrative": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. BEWERTUNGS-KASKADE
# ═══════════════════════════════════════════════════════════════════════════

def calculate_half_life(ratio_series):
    """Ornstein-Uhlenbeck Half-Life. Returns dict."""
    if ratio_series is None or len(ratio_series) < 36:
        return {"half_life_months": None, "significant": False, "r_squared": None, "p_value": None}
    try:
        clean = ratio_series[ratio_series > 0].dropna()
        if len(clean) < 36:
            return {"half_life_months": None, "significant": False, "r_squared": None, "p_value": None}
        log_ratio = np.log(clean.values)
        delta = np.diff(log_ratio)
        lagged = log_ratio[:-1]
        slope, intercept, r_value, p_value, std_err = linregress(lagged, delta)
        if slope >= 0:
            return {"half_life_months": None, "significant": False,
                    "r_squared": round(r_value**2, 4), "p_value": round(p_value, 4)}
        hl = -np.log(2) / slope
        return {"half_life_months": round(float(hl), 1), "significant": p_value < 0.05,
                "r_squared": round(r_value**2, 4), "p_value": round(p_value, 4)}
    except Exception as e:
        logger.error(f"Half-life failed: {e}")
        return {"half_life_months": None, "significant": False, "r_squared": None, "p_value": None}


def estimate_normalization(hl_months, significant):
    """2 × Half-Life ≈ 75% distance covered."""
    if not significant or hl_months is None:
        return "Kein statistischer Rückkehr-Trend nachweisbar"
    t = hl_months * 2
    if t < 12: return "< 1 Jahr"
    elif t < 24: return "~1-2 Jahre"
    elif t < 36: return "~2-3 Jahre"
    elif t < 60: return "~3-5 Jahre"
    elif t < 120: return "~5-10 Jahre"
    else: return "> 10 Jahre"


def classify_signal(percentile, direction):
    """Classify valuation signal."""
    if percentile is None:
        return "KEINE DATEN"
    eff = (100 - percentile) if direction == "high_is_cheap" else percentile
    if eff <= 10: return "EXTREM BILLIG"
    elif eff <= 25: return "SEHR BILLIG"
    elif eff <= 40: return "BILLIG"
    elif eff <= 60: return "FAIR"
    elif eff <= 75: return "TEUER"
    elif eff <= 90: return "SEHR TEUER"
    else: return "EXTREM TEUER"


def compute_valuation_ratio(ratio_key, ratio_def, all_data):
    """Compute a single valuation ratio."""
    logger.info(f"  Ratio: {ratio_key}")
    denom_scale = ratio_def.get("denominator_scale", 1.0)
    ratio = compute_ratio(all_data.get(ratio_def["numerator"]),
                          all_data.get(ratio_def["denominator"]),
                          denom_scale=denom_scale)
    if ratio is None or len(ratio) < 24:
        logger.warning(f"  {ratio_key}: insufficient data")
        return None

    current = float(ratio.iloc[-1])
    alltime_mean = float(ratio.mean())
    rolling_20y = ratio.rolling(240).mean()
    r20y = float(rolling_20y.iloc[-1]) if len(rolling_20y.dropna()) > 0 else None
    shift = False
    if r20y is not None and alltime_mean != 0:
        shift = abs(r20y - alltime_mean) / abs(alltime_mean) * 100 > 30

    pctl = compute_percentile(ratio, current)
    pctl_20y = compute_percentile_20y(ratio, current)
    signal = classify_signal(pctl, ratio_def["direction"])
    hl = calculate_half_life(ratio)
    norm = estimate_normalization(hl["half_life_months"], hl["significant"])

    logger.info(f"    Pctl: {pctl}%, Signal: {signal}, HL: {hl['half_life_months']} Mo")

    return {
        "name": ratio_def["name"], "current": round(current, 4),
        "alltime_mean": round(alltime_mean, 4),
        "rolling_20y_mean": round(r20y, 4) if r20y else None,
        "structural_shift_warning": shift,
        "percentile_alltime": pctl, "percentile_20y": pctl_20y,
        "signal": signal, "half_life": hl, "estimated_normalization": norm,
        "level": ratio_def["level"],
        "fundamental_confirmation": "KEINE DATEN",
        "fundamental_factors": [],
        "combined_signal": signal,
    }


def build_cascade(ratios):
    """Build the Relative Value Cascade."""
    cascade = []

    spy_m2 = ratios.get("SPY_M2")
    if spy_m2 and spy_m2["percentile_alltime"] is not None and spy_m2["percentile_alltime"] > 60:
        cascade.append({"level": 1, "ratio": "SPY/M2", "signal": "AKTIEN REAL TEUER",
                        "percentile": spy_m2["percentile_alltime"],
                        "implication": "→ Schaue auf Alternativen zu Financial Assets"})
        for key, name, thr in [("GOLD_SPY", "Gold/SPY", 40), ("DBC_SPY", "DBC/SPY", 40)]:
            r = ratios.get(key)
            if r and r["percentile_alltime"] is not None and r["percentile_alltime"] < thr:
                sig = "GOLD BILLIG VS. AKTIEN" if "GOLD" in key else "COMMODITIES BILLIG VS. AKTIEN"
                imp = "→ Gold bevorzugt vs. SPY" if "GOLD" in key else "→ Commodities bevorzugt vs. SPY"
                cascade.append({"level": 1, "ratio": name, "signal": sig,
                                "percentile": r["percentile_alltime"], "implication": imp})

    ebene2 = []
    gs = ratios.get("GOLD_SILVER")
    if gs and gs["percentile_alltime"] is not None and gs["percentile_alltime"] > 70:
        ebene2.append({"asset": "Silber", "vs": "Gold", "percentile": gs["percentile_alltime"], "signal": gs["signal"]})
    og = ratios.get("OIL_GOLD")
    if og and og["percentile_alltime"] is not None and og["percentile_alltime"] < 30:
        ebene2.append({"asset": "Öl", "vs": "Gold", "percentile": og["percentile_alltime"], "signal": og["signal"]})
    cg = ratios.get("COPPER_GOLD")
    if cg and cg["percentile_alltime"] is not None and cg["percentile_alltime"] < 30:
        ebene2.append({"asset": "Kupfer", "vs": "Gold", "percentile": cg["percentile_alltime"], "signal": cg["signal"]})

    ebene2.sort(key=lambda x: abs(x["percentile"] - 50), reverse=True)
    for item in ebene2:
        cascade.append({"level": 2, "ratio": f"{item['asset']}/{item['vs']}",
                        "signal": item["signal"], "percentile": item["percentile"],
                        "implication": f"→ {item['asset']} stärkstes Signal innerhalb Real Assets"})

    if cascade:
        strongest = max(cascade, key=lambda x: abs(x["percentile"] - 50))
        summary = {"strongest_signal": {"ratio": strongest["ratio"], "signal": strongest["signal"],
                                        "percentile": strongest["percentile"]},
                   "chain_length": len(cascade),
                   "direction": "REAL ASSETS" if any(c["level"] == 1 for c in cascade) else "UNKLAR"}
    else:
        summary = {"strongest_signal": None, "chain_length": 0, "direction": "KEIN KLARES SIGNAL"}

    return cascade, summary


def compute_valuation_cascade(all_data):
    """Compute 6 ratios + cascade logic."""
    logger.info("Computing valuation cascade...")
    ratios = {}
    for rk, rd in VALUATION_RATIOS.items():
        result = compute_valuation_ratio(rk, rd, all_data)
        if result is not None:
            ratios[rk] = result
    cascade, summary = build_cascade(ratios)
    logger.info(f"  Cascade: {summary['chain_length']} steps, direction: {summary['direction']}")
    return {"ratios": ratios, "cascade": cascade, "cascade_summary": summary, "narrative": ""}


# ═══════════════════════════════════════════════════════════════════════════
# 8. LLM INTEGRATION (Etappe 3)
# ═══════════════════════════════════════════════════════════════════════════

SECULAR_SYSTEM_PROMPT = """Du bist der Säkulare-Trends-Analyst von Baldur Creek Capital, einem systematischen Macro-Hedgefund.

Deine Aufgabe: Interpretiere die berechneten säkularen Regime-Daten und formuliere ein Investor-Briefing auf Deutsch.

REGELN:
1. Schreibe auf Deutsch, professioneller Investoren-Ton (wie Bridgewater Daily Observations)
2. Kein "KI" oder "AI" in deinem Text
3. Jede Aussage muss durch die gelieferten Daten oder Web-Search-Ergebnisse gestützt sein
4. Keine erfundenen Zahlen — nur Zahlen die in den Inputs stehen oder die du per Web Search findest
5. Sei direkt und meinungsstark — kein "es könnte sein" wenn die Daten klar sind
6. Erwähne Fragilitäts-Indikatoren nur wenn sie auf WATCH oder ACTIVE stehen
7. Halte dich an die kausale Kette: Demografie → Deglobalisierung → Fiscal Dominance → Financial Repression → Great Divergence

DEMOGRAFISCHER KONTEXT (statisch, für Referenz):
- US Working Age Population Growth: von 1.2% (1990er) auf ~0.3% (2020er) verlangsamt
- China: seit 2022 in absolutem Bevölkerungsrückgang
- Japan: seit 1995 schrumpfend — 30 Jahre Laborexperiment für demografischen Gegenwind
- Europa (Deutschland): Working Age Population schrumpft seit ~2010

OUTPUT-FORMAT:
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt (kein Markdown, keine Backticks, kein Preamble):
{
    "regime_narratives": {
        "demographic_cliff": "3-5 Sätze...",
        "deglobalization": "3-5 Sätze...",
        "fiscal_dominance": "3-5 Sätze...",
        "financial_repression": "3-5 Sätze...",
        "great_divergence": "3-5 Sätze..."
    },
    "conviction_narrative": "3-5 Sätze Gesamtbild für Conviction Summary...",
    "fundamental_assessments": [
        {
            "ratio": "RATIO_KEY",
            "factors": [
                {
                    "factor": "Produktionsdefizit",
                    "detail": "Konkretes Detail aus Web Search...",
                    "direction": "BESTÄTIGT"
                }
            ]
        }
    ],
    "valuation_narrative": "3-5 Sätze über die Bewertungs-Kaskade..."
}"""


def build_llm_input(conviction_summary, regime_results, valuation_cascade):
    """Build the user prompt from computed data."""
    sections = []

    # Conviction Summary
    cs = conviction_summary
    sections.append(f"""## CONVICTION SUMMARY
Aktive Regimes: {cs['active_regimes']}/{cs['total_regimes']}
Gewichteter Activation Score: {cs['weighted_activation']:.2f}
Konvergenz-Richtung: {cs['convergence_direction']}
Tailwind Scores: {json.dumps(cs['tailwind_scores'])}
""")

    # Regime blocks
    for rk in REGIME_ORDER:
        bd = REGIME_BLOCKS[rk]
        rr = regime_results[rk]
        rs = cs["regime_status"][rk]
        sections.append(f"""## REGIME: {bd['name']}
Activation Score: {rr['activation']:.2f} ({'AKTIV' if rr['active'] else 'INAKTIV'})
Robustheit: {rs['robustness']}
Zeithorizont: {rs['horizon']}
Fragilitäts-Status: {rs.get('fragility_status', 'INACTIVE')}
Charts:""")
        for chart in rr["charts"]:
            sections.append(f"  - {chart['name']}: aktuell={chart['current']}, "
                          f"Perzentil={chart.get('percentile')}%, "
                          f"Alltime-Mean={chart.get('alltime_mean')}, "
                          f"20J-Mean={chart.get('rolling_20y_mean')}")

    # Valuation cascade
    sections.append("\n## BEWERTUNGS-KASKADE (Ratios + Perzentile + Half-Life)")
    for rk, rv in valuation_cascade["ratios"].items():
        hl = rv["half_life"]
        sections.append(
            f"  {rk}: aktuell={rv['current']}, "
            f"Pzl={rv['percentile_alltime']}%, "
            f"Signal={rv['signal']}, "
            f"Half-Life={hl['half_life_months']} Mo "
            f"({'signifikant' if hl['significant'] else 'NICHT signifikant'}), "
            f"Normalisierung={rv['estimated_normalization']}"
        )

    # Extreme ratios for fundamental confirmation
    extreme = [
        (rk, rv) for rk, rv in valuation_cascade["ratios"].items()
        if rv["percentile_alltime"] is not None
        and (rv["percentile_alltime"] > 80 or rv["percentile_alltime"] < 20)
    ]

    if extreme:
        sections.append("""
## AUFGABE: FUNDAMENTAL-BESTÄTIGUNG (Web Search)
Für folgende Ratios im extremen Bereich, suche nach aktuellen Supply/Demand-Fundamentals:""")
        for rk, rv in extreme:
            asset = rv["name"].split("/")[0].strip()
            sections.append(f"""
### {rk} (Perzentil: {rv['percentile_alltime']}%)
Suche nach:
1. Produktionsdefizite oder -überschüsse für {asset}
2. Nachfrage-Treiber (Technologie, Regulierung, geopolitisch)
3. Capex-Trends (wird investiert oder nicht?)
4. Lagerbestands-Trends
Bewerte ob die Fundamentals das Preis-Signal BESTÄTIGEN oder WIDERSPRECHEN.""")

    return "\n".join(sections)


def parse_llm_response(response):
    """Parse LLM response. Robust against backticks and preamble."""
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON parse error: {e}")
        logger.error(f"Raw response (first 500 chars): {text[:500]}")
        return None


def get_llm_fallback():
    """Fallback when LLM fails."""
    return {
        "regime_narratives": {
            k: "Narrativ konnte nicht generiert werden."
            for k in REGIME_ORDER
        },
        "conviction_narrative": "Narrativ konnte nicht generiert werden.",
        "fundamental_assessments": [],
        "valuation_narrative": "Narrativ konnte nicht generiert werden.",
    }


def calculate_fundamental_confirmation(factors):
    """Calculate fundamental confirmation from LLM factors."""
    if not factors:
        return "KEINE DATEN"
    score = sum(
        (1.0 if f.get("direction") == "BESTÄTIGT" else
         -1.0 if f.get("direction") == "WIDERSPRICHT" else 0.0)
        for f in factors
    ) / len(factors)
    if score > 0.5: return "STARK BESTÄTIGT"
    elif score > 0.0: return "TEILWEISE BESTÄTIGT"
    elif score > -0.5: return "GEMISCHT"
    else: return "WIDERSPRICHT"


def calculate_combined_signal(price_signal, fundamental_confirmation):
    """Combine price signal with fundamental confirmation."""
    price_scores = {
        "EXTREM BILLIG": 5, "SEHR BILLIG": 4, "BILLIG": 3, "FAIR": 0,
        "TEUER": -3, "SEHR TEUER": -4, "EXTREM TEUER": -5, "KEINE DATEN": 0,
    }
    fund_multipliers = {
        "STARK BESTÄTIGT": 1.5, "TEILWEISE BESTÄTIGT": 1.2,
        "KEINE DATEN": 1.0, "GEMISCHT": 0.8, "WIDERSPRICHT": 0.5,
    }
    raw = price_scores.get(price_signal, 0)
    mult = fund_multipliers.get(fundamental_confirmation, 1.0)
    combined = raw * mult

    if combined >= 6: return "EXTREM STARK"
    elif combined >= 4: return "STARK"
    elif combined >= 2: return "MODERAT"
    elif combined >= -2: return "NEUTRAL"
    elif combined >= -4: return "MODERAT NEGATIV"
    else: return "STARK NEGATIV"


def run_llm(conviction_summary, regime_results, valuation_cascade):
    """Run the LLM call with Web Search. Returns parsed response or fallback."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed — skipping LLM")
        return get_llm_fallback(), False

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — skipping LLM")
        return get_llm_fallback(), False

    try:
        client = anthropic.Anthropic()
        user_prompt = build_llm_input(conviction_summary, regime_results, valuation_cascade)

        logger.info("Calling LLM (Claude Sonnet + Web Search)...")
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
            }],
            system=SECULAR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        parsed = parse_llm_response(response)
        if parsed is None:
            logger.warning("LLM response could not be parsed — using fallback")
            return get_llm_fallback(), False

        logger.info("LLM call successful")
        return parsed, True

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return get_llm_fallback(), False


def apply_llm_results(llm_data, conviction_summary, regime_results, valuation_cascade):
    """Apply LLM results to the computed data structures."""
    # Regime narratives
    narratives = llm_data.get("regime_narratives", {})
    for rk in REGIME_ORDER:
        if rk in narratives:
            regime_results[rk]["narrative"] = narratives[rk]

    # Conviction narrative
    conviction_summary["narrative"] = llm_data.get("conviction_narrative", "")

    # Fundamental assessments → update valuation ratios
    assessments = llm_data.get("fundamental_assessments", [])
    for assessment in assessments:
        ratio_key = assessment.get("ratio", "")
        factors = assessment.get("factors", [])
        if ratio_key in valuation_cascade["ratios"]:
            rv = valuation_cascade["ratios"][ratio_key]
            rv["fundamental_factors"] = factors
            rv["fundamental_confirmation"] = calculate_fundamental_confirmation(factors)
            rv["combined_signal"] = calculate_combined_signal(
                rv["signal"], rv["fundamental_confirmation"])

    # Valuation narrative
    valuation_cascade["narrative"] = llm_data.get("valuation_narrative", "")

    # Rebuild cascade summary with updated combined signals
    # (strongest might change with fundamental confirmation)
    cascade = valuation_cascade["cascade"]
    if cascade:
        for step in cascade:
            ratio_name = step["ratio"].replace("/", "_").replace(" ", "_").upper()
            # Try to match ratio key
            for rk, rv in valuation_cascade["ratios"].items():
                if rv["name"].replace(" ", "") == step["ratio"].replace(" ", ""):
                    step["combined_signal"] = rv.get("combined_signal", step["signal"])
                    break

    return conviction_summary, regime_results, valuation_cascade


# ═══════════════════════════════════════════════════════════════════════════
# 9. JSON OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def build_output_json(conviction_summary, regime_results, valuation_cascade, llm_success):
    """Build the complete secular_trends.json structure."""
    now = datetime.now(timezone.utc)

    regimes = {}
    for r in REGIME_ORDER:
        regimes[r] = {
            "charts": regime_results[r]["charts"],
            "narrative": regime_results[r].get("narrative", ""),
        }

    return {
        "metadata": {
            "generated_at": now.isoformat(),
            "version": "1.2",
            "data_through": now.strftime("%Y-%m-%d"),
            "fred_series_count": len(FRED_SERIES),
            "eod_series_count": len(EOD_TICKERS),
            "llm_model": CLAUDE_MODEL if llm_success else "",
            "llm_success": llm_success,
        },
        "conviction_summary": conviction_summary,
        "regimes": regimes,
        "valuation_cascade": valuation_cascade,
    }


def write_json(output):
    """Write secular_trends.json."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(OUTPUT_FILE)
    logger.info(f"Written: {OUTPUT_FILE} ({size:,} bytes)")


# ═══════════════════════════════════════════════════════════════════════════
# 10. GIT PUSH
# ═══════════════════════════════════════════════════════════════════════════

def git_push():
    """Git add, commit, push."""
    try:
        subprocess.run(["git", "add", "step_0w_secular/data/secular_trends.json"], check=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode == 0:
            logger.info("No changes to commit")
            return
        subprocess.run(["git", "commit", "-m", "Secular Trends monthly update"], check=True)
        subprocess.run(["git", "pull", "--rebase"], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info("Git push complete")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git push failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Secular Trends Pipeline")
    parser.add_argument("--skip-git", action="store_true", help="Skip git commit/push")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM call")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SECULAR TRENDS PIPELINE — V1.2 (Etappe 1+2+3)")
    logger.info("=" * 60)

    # ── Step 1: Fetch ──
    logger.info("Step 1: Fetching all data...")
    all_data = fetch_all_data()

    # ── Step 2: Regime blocks ──
    logger.info("Step 2: Computing regime blocks...")
    regime_results = {}
    for rk in REGIME_ORDER:
        bd = REGIME_BLOCKS[rk]
        result = compute_regime_block(rk, bd, all_data)
        frag_def = FRAGILITY_INDICATORS.get(rk)
        if frag_def:
            fr = compute_fragility(rk, frag_def, all_data)
            result["fragility_indicator"] = frag_def["name"]
            result["fragility_status"] = fr["status"]
            result["fragility_detail"] = frag_def["description_de"]
            result["fragility_current_value"] = fr.get("current_value")
        regime_results[rk] = result

    # ── Step 3: Tailwind ──
    logger.info("Step 3: Computing tailwind scores...")
    tailwind_scores = compute_tailwind_scores(regime_results)
    for a, s in tailwind_scores.items():
        logger.info(f"  {ASSET_CLASS_LABELS.get(a, a):20s} → {s:+d}%")

    # ── Step 4: Conviction Summary ──
    logger.info("Step 4: Building conviction summary...")
    conviction_summary = compute_conviction_summary(regime_results, tailwind_scores)
    logger.info(f"  Active: {conviction_summary['active_regimes']}/{conviction_summary['total_regimes']}")
    logger.info(f"  Weighted: {conviction_summary['weighted_activation']:.3f}")
    logger.info(f"  Convergence: {conviction_summary['convergence_direction']}")

    # ── Step 5: Valuation Cascade ──
    logger.info("Step 5: Computing valuation cascade...")
    valuation_cascade = compute_valuation_cascade(all_data)

    # ── Step 6: LLM ──
    llm_success = False
    if not args.skip_llm:
        logger.info("Step 6: Running LLM...")
        llm_data, llm_success = run_llm(conviction_summary, regime_results, valuation_cascade)
        conviction_summary, regime_results, valuation_cascade = apply_llm_results(
            llm_data, conviction_summary, regime_results, valuation_cascade)
        logger.info(f"  LLM success: {llm_success}")
    else:
        logger.info("Step 6: LLM SKIPPED")

    # ── Step 7: JSON ──
    logger.info("Step 7: Writing JSON...")
    output = build_output_json(conviction_summary, regime_results, valuation_cascade, llm_success)
    write_json(output)

    # ── Step 8: Git ──
    if not args.skip_git:
        logger.info("Step 8: Git push...")
        git_push()
    else:
        logger.info("Step 8: Git push SKIPPED")

    logger.info("=" * 60)
    logger.info("SECULAR TRENDS PIPELINE — COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
