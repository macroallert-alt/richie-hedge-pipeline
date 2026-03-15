"""
Lead-Engine V1.1 — Cycles Calibration & Early Warning System
Baldur Creek Capital | Step 0v Phase 2 | March 2026

Architecture (4 Modules, ~1500 lines, zero external dependencies):
  Modul B: Conditional Forward Returns (27 assets, tail-risk, max drawdown)
  Modul C: Regime Interaction (clusters, V16 transition, Fed pivot, crash/correction, analogues)
  Modul D: Transition Engine (transition matrix, phase position, remaining durations, cascade, confirmation)
  Modul E: Orchestrator + Output (run_lead_engine entry point, 3 output JSONs)

Spec: LEAD_ENGINE_V1.1_SPEC_TEIL1-5.md (source of truth)
Replaces: lead_engine.py V1.0 (archived as lead_engine_v1_reference.py)
"""

import json
import logging
import math
import os
from datetime import datetime, timezone

logger = logging.getLogger("cycles.lead_engine")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG (Spec TEIL5 §14)
# ═══════════════════════════════════════════════════════════════════════════

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_MODULE_DIR, "data")

# 27 Assets — full V16 DATA_Prices universe (Spec TEIL1 §3 / TEIL5 §14.1)
ASSETS = [
    "GLD", "SLV", "GDX", "GDXJ", "SIL", "PLATINUM",
    "SPY", "IWM", "EEM", "VGK",
    "XLY", "XLI", "XLF", "XLE", "XLV", "XLP", "XLU", "VNQ", "XLK",
    "TLT", "TIP", "LQD", "HYG",
    "DBC", "COPPER",
    "BTC", "ETH",
]

# 9 non-political cycles (Spec TEIL5 §14.2)
LEAD_LAG_CYCLES = [
    "LIQUIDITY", "CREDIT", "COMMODITY", "CHINA_CREDIT",
    "DOLLAR", "BUSINESS", "FED_RATES", "EARNINGS", "TRADE",
]
ALL_CYCLES = LEAD_LAG_CYCLES + ["POLITICAL"]

# Horizons (Spec TEIL5 §14.10)
HORIZONS = [3, 6, 12]
PHASE_DETECTION_LAG_MONTHS = 2
TRANSITION_WINDOW_MONTHS = 6
CASCADE_LOOKBACK_MONTHS = 6
N_HISTORICAL_ANALOGUES = 5

# Phase Buckets (Spec TEIL1 §6 / TEIL5 §14.7)
PHASE_BUCKETS = {
    "BULLISH": [
        "EXPANSION", "EARLY_RECOVERY", "RECOVERY", "MID_BULL",
        "EARLY_BULL", "EARLY_STIMULUS", "EASING", "NEUTRAL",
        "PRE_ELECTION", "TROUGH",
        "WEAKENING",       # Dollar WEAKENING = bullish for risk assets
    ],
    "BEARISH": [
        "CONTRACTION", "DETERIORATION", "DISTRESS", "RECESSION",
        "COLLAPSE", "BEAR", "WITHDRAWAL",
        "STRENGTHENING",   # Dollar STRENGTHENING = bearish for risk assets
    ],
    "NEUTRAL_MIXED": [
        "LATE_EXPANSION", "PEAK", "PLATEAU", "LATE",
        "OVERINVESTMENT", "TIGHTENING", "RESTRICTIVE",
        "MIDTERM", "POST_INAUGURATION", "ELECTION",
        "PRE_PIVOT", "PIVOT", "REPAIR",
        "EUPHORIA",
    ],
}

# Reverse lookup: phase → bucket
_PHASE_BUCKET_MAP = {}
for _bucket, _phases in PHASE_BUCKETS.items():
    for _p in _phases:
        _PHASE_BUCKET_MAP[_p] = _bucket

# Cycle Clusters (Spec TEIL1 §7 / TEIL5 §14.3)
CYCLE_CLUSTERS = {
    "CREDIT_CLUSTER": {
        "cycles": ["CREDIT", "LIQUIDITY", "EARNINGS"],
        "dominant": "CREDIT",
    },
    "REAL_ECONOMY_CLUSTER": {
        "cycles": ["BUSINESS", "TRADE", "CHINA_CREDIT", "COMMODITY"],
        "dominant": "BUSINESS",
    },
    "MONETARY_POLICY_CLUSTER": {
        "cycles": ["FED_RATES"],
        "dominant": "FED_RATES",
    },
    "CURRENCY_CLUSTER": {
        "cycles": ["DOLLAR"],
        "dominant": "DOLLAR",
    },
}
CLUSTER_ORDER = [
    "CREDIT_CLUSTER", "REAL_ECONOMY_CLUSTER",
    "MONETARY_POLICY_CLUSTER", "CURRENCY_CLUSTER",
]

# Cluster Distance Weights for analogues (Spec TEIL5 §14.4)
CLUSTER_DISTANCE_WEIGHTS = {
    "CREDIT_CLUSTER": 2.0,
    "REAL_ECONOMY_CLUSTER": 1.5,
    "MONETARY_POLICY_CLUSTER": 1.0,
    "CURRENCY_CLUSTER": 0.8,
}

# Causal Chain Pairs for conditional remaining durations (Spec TEIL5 §14.5)
CAUSAL_CHAIN_PAIRS = [
    ("LIQUIDITY", "CREDIT"),
    ("LIQUIDITY", "BUSINESS"),
    ("CREDIT", "BUSINESS"),
    ("CHINA_CREDIT", "COMMODITY"),
    ("CHINA_CREDIT", "BUSINESS"),
    ("FED_RATES", "DOLLAR"),
    ("DOLLAR", "COMMODITY"),
    ("BUSINESS", "EARNINGS"),
    ("EARNINGS", "FED_RATES"),
    ("COMMODITY", "FED_RATES"),
]

# V16 State Groups (Spec TEIL3 §9.3.2 / TEIL5 §14.6)
V16_STATE_GROUPS = {
    "GROWTH": [1, 2, 3, 4, 5, 6],
    "STRESS": [7, 8, 9],
    "CRISIS": [10, 11, 12],
}

# V16 State Name → Number (macro_state_history.json stores names, not numbers)
V16_STATE_NAME_TO_NUM = {
    "STEADY_GROWTH": 1, "FRAGILE_EXPANSION": 2, "LATE_EXPANSION": 3,
    "FULL_EXPANSION": 4, "REFLATION": 5, "NEUTRAL": 6,
    "SOFT_LANDING": 7, "STRESS_ELEVATED": 8, "CONTRACTION": 9,
    "DEEP_CONTRACTION": 10, "FINANCIAL_CRISIS": 11, "EARLY_RECOVERY": 12,
}

def _parse_v16_state(s):
    """Convert V16 state (string name or int) to state number."""
    if isinstance(s, int):
        return s
    if isinstance(s, str):
        # Try name lookup first
        num = V16_STATE_NAME_TO_NUM.get(s)
        if num is not None:
            return num
        # Try direct int parse
        try:
            return int(s)
        except (ValueError, TypeError):
            pass
    return None

# Entry Rules (Spec TEIL3 §9.5.3 / TEIL5 §14.11)
ENTRY_RULES = {
    "CREDIT_BEARISH__BUSINESS_BEARISH": {
        "type": "CRASH",
        "typical_drawdown": "-25% bis -35%",
        "entry_zone": "-20% bis -25% vom Hoch",
    },
    "CREDIT_BEARISH__BUSINESS_BULLISH": {
        "type": "CORRECTION",
        "typical_drawdown": "-8% bis -15%",
        "entry_zone": "-8% bis -10% vom Hoch",
    },
    "CREDIT_BEARISH__BUSINESS_BEARISH__FED_PIVOT_LIKELY": {
        "type": "CRASH_WITH_RECOVERY",
        "typical_drawdown": "-20% bis -35%",
        "entry_zone": "-15% bis -20% vom Hoch (aggressiver weil Fed kommt)",
    },
}

# Fed Pivot phase sets (Spec TEIL3 §9.4.2)
_RESTRICTIVE_PHASES = {"RESTRICTIVE", "TIGHTENING", "PRE_PIVOT"}
_EASING_PHASES = {"EASING", "NEUTRAL", "PIVOT"}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path):
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None


def _phase_to_bucket(phase):
    return _PHASE_BUCKET_MAP.get(phase, "NEUTRAL_MIXED")


def _next_month(ym):
    """'2024-12' → '2025-01'"""
    y, m = int(ym[:4]), int(ym[5:7])
    m += 1
    if m > 12:
        m = 1
        y += 1
    return f"{y:04d}-{m:02d}"


def _add_months(ym, n):
    """Add n months to 'YYYY-MM'."""
    y, m = int(ym[:4]), int(ym[5:7])
    m += n
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    return f"{y:04d}-{m:02d}"


def _months_between(ym1, ym2):
    """Number of months from ym1 to ym2 (inclusive = +0, ym2 > ym1 = positive)."""
    y1, m1 = int(ym1[:4]), int(ym1[5:7])
    y2, m2 = int(ym2[:4]), int(ym2[5:7])
    return (y2 - y1) * 12 + (m2 - m1)


def _expand_zone_months(zones):
    """Expand phase_zones list to {month: phase} dict."""
    month_phase = {}
    for z in zones:
        phase = z.get("phase", "UNKNOWN")
        start = z.get("start", "")
        end = z.get("end", "")
        if not start or not end:
            continue
        m = start
        while m <= end:
            month_phase[m] = phase
            m = _next_month(m)
    return month_phase


def _cumulative_return(ret_dict, start_month, horizon):
    """Compute cumulative log-return from start_month over horizon months."""
    total = 0.0
    valid = 0
    m = start_month
    for _ in range(horizon):
        m = _next_month(m) if _ > 0 else m
        if _ == 0:
            m = start_month
        r = ret_dict.get(m)
        if r is not None:
            total += r
            valid += 1
        # If we have a gap, still continue (partial)
    # Need at least 2/3 of months to be valid
    if valid < max(2, horizon * 2 // 3):
        return None
    return total


def _cumulative_return_clean(ret_dict, start_month, horizon):
    """Compute cumulative log-return: sum of returns for months [start, start+1, ..., start+horizon-1]."""
    total = 0.0
    valid = 0
    m = start_month
    for i in range(horizon):
        r = ret_dict.get(m)
        if r is not None:
            total += r
            valid += 1
        m = _next_month(m)
    if valid < max(2, horizon * 2 // 3):
        return None
    return total


def _compute_n_independent(months_list, horizon):
    """Compute number of non-overlapping windows in months_list for given horizon."""
    if not months_list:
        return 0
    sorted_m = sorted(months_list)
    count = 1
    last = sorted_m[0]
    for m in sorted_m[1:]:
        if _months_between(last, m) >= horizon:
            count += 1
            last = m
    return count


# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_monthly_returns(prices_history):
    """
    Compute monthly log-returns for all 27 assets from daily prices.
    (Spec TEIL2 §8.5)
    """
    returns = {}
    for ticker in ASSETS:
        price_list = prices_history.get(ticker, [])
        if not price_list:
            continue
        # Resample to monthly (last price per month)
        monthly = {}
        for pt in price_list:
            p = pt.get("price")
            d = pt.get("date")
            if p and d and p > 0:
                monthly[d[:7]] = p

        sorted_months = sorted(monthly.keys())
        ret = {}
        for i in range(1, len(sorted_months)):
            prev = monthly[sorted_months[i - 1]]
            curr = monthly[sorted_months[i]]
            if prev > 0 and curr > 0:
                ret[sorted_months[i]] = round(math.log(curr / prev), 6)

        returns[ticker] = ret
    return returns


def _compute_baselines(monthly_returns):
    """
    Compute unconditional average forward returns per asset per horizon.
    Each asset uses its own full available history.
    Also returns history_info for SHORT_HISTORY flagging.
    (Spec TEIL2 §8.6)
    """
    baselines = {}
    history_info = {}

    for ticker, ret_dict in monthly_returns.items():
        months = sorted(ret_dict.keys())
        if not months:
            continue

        history_info[ticker] = {
            "history_start": months[0],
            "history_end": months[-1],
            "n_months": len(months),
            "short_history": months[0] > "2010-01",
        }

        b = {}
        for h in HORIZONS:
            cum_returns = []
            for i in range(len(months) - h):
                cr = sum(ret_dict.get(months[i + j], 0) for j in range(h))
                cum_returns.append(cr)
            b[f"baseline_{h}m"] = round(sum(cum_returns) / len(cum_returns), 6) if cum_returns else None
        baselines[ticker] = b

    return baselines, history_info


def _ensure_chart_data_complete(chart_data):
    """
    Patch V4.0 chart_data: if phase_zones or smoothed are missing,
    compute them on-the-fly from indicator + ma_12m.
    (Spec TEIL1 §4.1 — V4.0 Fallback)
    """
    cycles = chart_data.get("cycles", {})
    for cid, cdata in cycles.items():
        # Phase zones fallback
        if "phase_zones" not in cdata or not cdata["phase_zones"]:
            indicator = cdata.get("indicator", [])
            ma = cdata.get("ma_12m", [])
            if indicator and ma and len(indicator) == len(ma):
                zones = _compute_phase_zones_fallback(cdata)
                cdata["phase_zones"] = zones

        # Smoothed fallback
        if "smoothed" not in cdata or not cdata["smoothed"]:
            ma = cdata.get("ma_12m", [])
            if ma:
                cdata["smoothed"] = _compute_smoothed_fallback(ma)

    return chart_data


def _compute_phase_zones_fallback(cdata):
    """Simple phase zone computation from indicator vs MA for V4.0 fallback."""
    indicator = cdata.get("indicator", [])
    ma = cdata.get("ma_12m", [])
    dates = cdata.get("dates", [])
    if not indicator or not ma or not dates or len(indicator) != len(ma):
        return []

    zones = []
    current_phase = None
    current_start = None

    for i in range(len(indicator)):
        val = indicator[i]
        ma_val = ma[i]
        if val is None or ma_val is None:
            continue

        if val > ma_val:
            phase = "EXPANSION"
        else:
            phase = "CONTRACTION"

        d = dates[i] if i < len(dates) else None
        if not d:
            continue
        m = d[:7] if len(d) >= 7 else d

        if phase != current_phase:
            if current_phase is not None and current_start:
                zones.append({"phase": current_phase, "start": current_start, "end": prev_m})
            current_phase = phase
            current_start = m
        prev_m = m

    if current_phase and current_start:
        zones.append({"phase": current_phase, "start": current_start, "end": prev_m})

    return zones


def _compute_smoothed_fallback(ma_values):
    """Double-smooth: 12-month MA of the MA values."""
    if len(ma_values) < 12:
        return ma_values
    smoothed = [None] * 11
    for i in range(11, len(ma_values)):
        window = [v for v in ma_values[i - 11:i + 1] if v is not None]
        smoothed.append(round(sum(window) / len(window), 4) if window else None)
    return smoothed


# ═══════════════════════════════════════════════════════════════════════════
# MODUL B: CONDITIONAL FORWARD RETURNS (Spec TEIL2 §8)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_return_stats(cum_returns, baseline, n_indep):
    """
    Compute return statistics for a set of cumulative forward returns.
    V1.1: p10/p90 tail-risk, SE based on n_independent.
    (Spec TEIL2 §8.7)
    """
    if not cum_returns or len(cum_returns) < 2:
        return None

    n = len(cum_returns)
    sr = sorted(cum_returns)

    avg = sum(cum_returns) / n
    median = sr[n // 2] if n % 2 == 1 else (sr[n // 2 - 1] + sr[n // 2]) / 2
    worst = sr[0]
    best = sr[-1]

    # Tail-Risk Percentiles
    p10_idx = max(0, int(n * 0.10))
    p90_idx = min(n - 1, int(n * 0.90))
    p10 = sr[p10_idx]
    p90 = sr[p90_idx]

    # Standard Deviation
    var = sum((r - avg) ** 2 for r in cum_returns) / n
    std = var ** 0.5

    # Standard Error: based on n_independent (V1.1 strictification)
    ni = max(1, n_indep)
    se = round(std / ni ** 0.5, 6) if ni > 0 else None

    # Hit Rate
    hit = sum(1 for r in cum_returns if r > 0) / n

    # Excess Return
    avg_ex = round(avg - baseline, 6) if baseline is not None else None

    # Signal Strength: |avg_excess| / SE
    sig_str = round(abs(avg_ex) / se, 2) if se and se > 0 and avg_ex is not None else None

    # Confidence Interval (68% CI = ±1 SE)
    ci_lo = round(avg_ex - se, 6) if avg_ex is not None and se else None
    ci_hi = round(avg_ex + se, 6) if avg_ex is not None and se else None
    significant = bool(ci_lo is not None and ci_hi is not None and (ci_lo > 0 or ci_hi < 0))

    return {
        "avg": round(avg, 6),
        "median": round(median, 6),
        "worst": round(worst, 6),
        "best": round(best, 6),
        "p10": round(p10, 6),
        "p90": round(p90, 6),
        "std": round(std, 6),
        "std_err": se,
        "hit_rate": round(hit, 3),
        "n": n,
        "n_independent": ni,
        "baseline": round(baseline, 6) if baseline is not None else None,
        "avg_excess": avg_ex,
        "signal_strength": sig_str,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "significant": significant,
    }


def _compute_conditional_max_drawdown(daily_prices, phase_months, horizon_months):
    """
    Compute max drawdown statistics within forward return windows.
    (Spec TEIL2 §8.8)

    Args:
        daily_prices: [{date, price}, ...] for one ticker
        phase_months: sorted list of 'YYYY-MM' (already offset by detection lag)
        horizon_months: int
    Returns:
        dict with max_dd_median, max_dd_worst, max_dd_p90, n_windows or None
    """
    # Build date→price map
    price_map = {}
    for pt in daily_prices:
        if pt.get("price") and pt.get("date"):
            price_map[pt["date"]] = pt["price"]

    all_dates = sorted(price_map.keys())
    if not all_dates:
        return None

    max_drawdowns = []

    for start_month in phase_months:
        end_month = _add_months(start_month, horizon_months)

        window_prices = [price_map[d] for d in all_dates
                         if d[:7] >= start_month and d[:7] < end_month
                         and d in price_map]

        if len(window_prices) < 20:
            continue

        # Max Drawdown in window
        peak = window_prices[0]
        max_dd = 0
        for p in window_prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd

        max_drawdowns.append(round(-max_dd, 6))  # Negative (loss)

    if len(max_drawdowns) < 3:
        return None

    sr = sorted(max_drawdowns)  # Sorted from worst (most negative) to best
    n = len(sr)

    return {
        "max_dd_median": sr[n // 2],
        "max_dd_worst": sr[0],
        "max_dd_p90": sr[max(0, int(n * 0.10))],  # 10th percentile of DDs = worst 10%
        "n_windows": n,
    }


def _returns_for_months(phase_months, monthly_returns, baselines, transition_months,
                        history_info=None):
    """
    Compute forward return stats for all assets across all horizons for given months.
    Also splits into transition vs steady-state for 6M horizon.
    V1.1.1: SHORT_HISTORY gate — all short_history assets forced to significant=False.
    """
    offset_months = [_add_months(m, PHASE_DETECTION_LAG_MONTHS) for m in phase_months]

    # Split transition vs steady
    trans_offset = []
    steady_offset = []
    for orig, offset in zip(phase_months, offset_months):
        if orig in transition_months:
            trans_offset.append(offset)
        else:
            # Check if within TRANSITION_WINDOW_MONTHS of a transition
            is_near = False
            for tm in transition_months:
                diff = _months_between(tm, orig)
                if 0 <= diff <= TRANSITION_WINDOW_MONTHS:
                    is_near = True
                    break
            if is_near:
                trans_offset.append(offset)
            else:
                steady_offset.append(offset)

    result = {}
    for ticker in ASSETS:
        rd = monthly_returns.get(ticker, {})
        bl = baselines.get(ticker, {})
        ticker_result = {}

        for h in HORIZONS:
            baseline = bl.get(f"baseline_{h}m")
            crs = [_cumulative_return_clean(rd, m, h) for m in offset_months]
            crs = [r for r in crs if r is not None]
            ni = _compute_n_independent(offset_months, h)
            ticker_result[f"{h}m"] = _compute_return_stats(crs, baseline, ni)

        # Transition vs Steady for 6M only
        bl_6m = bl.get("baseline_6m")
        if trans_offset:
            crs_t = [_cumulative_return_clean(rd, m, 6) for m in trans_offset]
            crs_t = [r for r in crs_t if r is not None]
            ni_t = _compute_n_independent(trans_offset, 6)
            ticker_result["transition_6m"] = _compute_return_stats(crs_t, bl_6m, ni_t)
        if steady_offset:
            crs_s = [_cumulative_return_clean(rd, m, 6) for m in steady_offset]
            crs_s = [r for r in crs_s if r is not None]
            ni_s = _compute_n_independent(steady_offset, 6)
            ticker_result["steady_6m"] = _compute_return_stats(crs_s, bl_6m, ni_s)

        result[ticker] = ticker_result

        # V1.1.1 SHORT_HISTORY gate: all short_history assets → never significant
        # (Crypto with thin sample sizes, V132 §9.1)
        if history_info and history_info.get(ticker, {}).get("short_history"):
            for key, stats in ticker_result.items():
                if isinstance(stats, dict) and stats.get("significant"):
                    stats["significant"] = False

    return result


def compute_conditional_returns(chart_data, monthly_returns, baselines,
                                history_info, daily_prices=None):
    """
    Modul B: Compute forward returns conditioned on cycle phases.
    V1.1: 27 Assets, Tail-Risk, optional Max Drawdown.
    (Spec TEIL2 §8.9)
    """
    results = {}

    for cycle_id in ALL_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cycle_id, {})
        zones = cdata.get("phase_zones", [])
        if not zones:
            continue

        month_phase = _expand_zone_months(zones)
        transition_months = {z["start"] for z in zones}
        cycle_results = {}

        # Detail phases
        all_phases = sorted(set(month_phase.values()))
        for phase in all_phases:
            pm = sorted([m for m, p in month_phase.items() if p == phase])
            if not pm:
                continue
            cycle_results[phase] = _returns_for_months(
                pm, monthly_returns, baselines, transition_months, history_info)

        # Aggregated Buckets
        bucket_groups = {}
        for m, p in month_phase.items():
            b = _phase_to_bucket(p)
            bucket_groups.setdefault(b, []).append(m)

        for bname, bmonths in bucket_groups.items():
            if not bmonths:
                continue
            bucket_result = _returns_for_months(
                sorted(bmonths), monthly_returns, baselines, transition_months, history_info)

            # Max Drawdown only for Buckets, only 6M, only if daily_prices present
            if daily_prices:
                offset_months = [_add_months(m, PHASE_DETECTION_LAG_MONTHS)
                                 for m in sorted(bmonths)]
                for ticker in ASSETS:
                    ticker_daily = daily_prices.get(ticker, [])
                    if ticker_daily and ticker in bucket_result:
                        dd = _compute_conditional_max_drawdown(
                            ticker_daily, offset_months, 6)
                        if dd and bucket_result[ticker].get("6m"):
                            bucket_result[ticker]["6m"]["max_drawdown"] = dd

            cycle_results[f"BUCKET_{bname}"] = bucket_result

        results[cycle_id] = cycle_results

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MODUL C: REGIME INTERACTION (Spec TEIL3 §9)
# ═══════════════════════════════════════════════════════════════════════════

# --- C1: Cluster-Conditional-Returns (Spec TEIL3 §9.2) ---

def compute_cluster_conditional_returns(chart_data, monthly_returns, baselines,
                                        history_info=None):
    """
    Compute forward returns conditioned on cluster-state combinations.
    Returns both full combinations (n>=6) and cluster marginals (robust).
    (Spec TEIL3 §9.2)
    V1.1.1: SHORT_HISTORY gate applied to cluster returns too.
    """
    # Step 1: Build month→phase maps for all cycles
    cycle_month_maps = {}
    for cid in LEAD_LAG_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cid, {})
        zones = cdata.get("phase_zones", [])
        if zones:
            cycle_month_maps[cid] = _expand_zone_months(zones)

    # Step 2: Find common months across all dominant cycles
    dominant_cycles = [CYCLE_CLUSTERS[c]["dominant"] for c in CLUSTER_ORDER]
    common_months = None
    for dc in dominant_cycles:
        if dc in cycle_month_maps:
            dc_months = set(cycle_month_maps[dc].keys())
            common_months = dc_months if common_months is None else common_months & dc_months

    if not common_months:
        return {}

    common_months = sorted(common_months)

    # Step 3: Map each month to cluster combination
    month_combos = {}
    for m in common_months:
        combo_parts = []
        for cluster_name in CLUSTER_ORDER:
            dominant = CYCLE_CLUSTERS[cluster_name]["dominant"]
            phase = cycle_month_maps.get(dominant, {}).get(m)
            if not phase:
                combo_parts.append("UNKNOWN")
            else:
                combo_parts.append(_phase_to_bucket(phase))

        combo_key = "_".join(combo_parts)
        month_combos.setdefault(combo_key, []).append(m)

    # Step 4: Forward returns per combination (n>=6 only)
    combo_results = {}
    for combo_key, months in month_combos.items():
        if len(months) < 6:
            continue

        offset_months = [_add_months(m, PHASE_DETECTION_LAG_MONTHS) for m in months]

        assets_result = {}
        for ticker in ASSETS:
            rd = monthly_returns.get(ticker, {})
            bl = baselines.get(ticker, {})
            tr = {}
            for h in HORIZONS:
                baseline = bl.get(f"baseline_{h}m")
                crs = [_cumulative_return_clean(rd, m, h) for m in offset_months]
                crs = [r for r in crs if r is not None]
                ni = _compute_n_independent(offset_months, h)
                tr[f"{h}m"] = _compute_return_stats(crs, baseline, ni) if crs else None
            assets_result[ticker] = tr
            # V1.1.1 SHORT_HISTORY gate
            if history_info and history_info.get(ticker, {}).get("short_history"):
                for key, stats in tr.items():
                    if isinstance(stats, dict) and stats.get("significant"):
                        stats["significant"] = False

        combo_results[combo_key] = {
            "months_sample": months[:5],
            "n_months": len(months),
            "assets": assets_result,
        }

    # Step 5: Cluster Marginals (individual cluster, independent of others)
    marginals = {}
    for cluster_name in CLUSTER_ORDER:
        dominant = CYCLE_CLUSTERS[cluster_name]["dominant"]
        dom_map = cycle_month_maps.get(dominant, {})

        bucket_months = {}
        for m in common_months:
            phase = dom_map.get(m)
            if phase:
                bucket = _phase_to_bucket(phase)
                bucket_months.setdefault(bucket, []).append(m)

        cluster_marginal = {}
        for bucket, months in bucket_months.items():
            offset = [_add_months(m, PHASE_DETECTION_LAG_MONTHS) for m in months]
            assets_result = {}
            for ticker in ASSETS:
                rd = monthly_returns.get(ticker, {})
                bl = baselines.get(ticker, {})
                tr = {}
                for h in HORIZONS:
                    baseline = bl.get(f"baseline_{h}m")
                    crs = [_cumulative_return_clean(rd, m, h) for m in offset]
                    crs = [r for r in crs if r is not None]
                    ni = _compute_n_independent(offset, h)
                    tr[f"{h}m"] = _compute_return_stats(crs, baseline, ni) if crs else None
                assets_result[ticker] = tr
                # V1.1.1 SHORT_HISTORY gate
                if history_info and history_info.get(ticker, {}).get("short_history"):
                    for key, stats in tr.items():
                        if isinstance(stats, dict) and stats.get("significant"):
                            stats["significant"] = False
            cluster_marginal[bucket] = {
                "n_months": len(months),
                "assets": assets_result,
            }
        marginals[cluster_name] = cluster_marginal

    return {"cluster_combinations": combo_results, "cluster_marginals": marginals}


# --- C2: V16 State Transition Probability (Spec TEIL3 §9.3) ---

def _state_to_group(state_num):
    """Map V16 state number to GROWTH/STRESS/CRISIS group."""
    for group, states in V16_STATE_GROUPS.items():
        if state_num in states:
            return group
    return "GROWTH"


def _group_severity(group):
    return {"GROWTH": 0, "STRESS": 1, "CRISIS": 2}.get(group, 0)


def compute_v16_transition_probability(chart_data, macro_state_history, cycle_month_maps):
    """
    Compute probability of V16 state transitions conditioned on cycle cluster states.
    (Spec TEIL3 §9.3)
    """
    # V16 state per month
    v16_monthly = {}
    for entry in macro_state_history:
        d = entry.get("date", "")
        s = entry.get("state")
        if d and s is not None:
            dm = d[:7]
            num = _parse_v16_state(s)
            if num is not None:
                v16_monthly[dm] = num

    if not v16_monthly:
        return {}

    # Common months across dominant cycles
    dominant_cycles = [CYCLE_CLUSTERS[c]["dominant"] for c in CLUSTER_ORDER]
    common_months = None
    for dc in dominant_cycles:
        if dc in cycle_month_maps:
            dc_months = set(cycle_month_maps[dc].keys())
            common_months = dc_months if common_months is None else common_months & dc_months

    if not common_months:
        return {}

    common_months = sorted(common_months)

    # Group months by Credit cluster state (by_credit_cluster)
    credit_dom = CYCLE_CLUSTERS["CREDIT_CLUSTER"]["dominant"]
    credit_map = cycle_month_maps.get(credit_dom, {})

    credit_groups = {}
    for m in common_months:
        phase = credit_map.get(m)
        if phase:
            bucket = _phase_to_bucket(phase)
            credit_groups.setdefault(bucket, []).append(m)

    # Group months by dual cluster (Credit × Real Economy)
    real_dom = CYCLE_CLUSTERS["REAL_ECONOMY_CLUSTER"]["dominant"]
    real_map = cycle_month_maps.get(real_dom, {})

    dual_groups = {}
    for m in common_months:
        cp = credit_map.get(m)
        rp = real_map.get(m)
        if cp and rp:
            cb = _phase_to_bucket(cp)
            rb = _phase_to_bucket(rp)
            key = f"CREDIT_{cb}__REAL_{rb}"
            dual_groups.setdefault(key, []).append(m)

    def _compute_transitions(month_group):
        transitions = {}
        for h in [3, 6]:
            h_key = f"{h}m"
            counts = {"growth": 0, "stress": 0, "crisis": 0}
            valid = 0
            for m in month_group:
                worst_group = "GROWTH"
                has_data = False
                for fwd in range(1, h + 1):
                    fwd_month = _add_months(m, fwd)
                    fwd_state = v16_monthly.get(fwd_month)
                    if fwd_state is not None:
                        has_data = True
                        fwd_group = _state_to_group(fwd_state)
                        if _group_severity(fwd_group) > _group_severity(worst_group):
                            worst_group = fwd_group
                if has_data:
                    counts[worst_group.lower()] += 1
                    valid += 1

            if valid > 0:
                transitions[f"v16_stays_growth_{h_key}"] = round(counts["growth"] / valid, 3)
                transitions[f"v16_to_stress_{h_key}"] = round(counts["stress"] / valid, 3)
                transitions[f"v16_to_crisis_{h_key}"] = round(counts["crisis"] / valid, 3)
            transitions[f"n_months"] = len(month_group)
        return transitions

    by_credit = {}
    for bucket, months in credit_groups.items():
        by_credit[bucket] = _compute_transitions(months)

    by_dual = {}
    for key, months in dual_groups.items():
        by_dual[key] = _compute_transitions(months)

    return {
        "by_credit_cluster": by_credit,
        "by_dual_cluster": by_dual,
        "methodology": "Monthly V16 states mapped to GROWTH/STRESS/CRISIS groups. "
                        "Forward transition = worst V16 group reached within horizon.",
    }


# --- C3: Fed Pivot Probability (Spec TEIL3 §9.4) ---

def _identify_fed_pivot_events(chart_data):
    """
    Identify months where Fed_Rates transitioned from RESTRICTIVE/TIGHTENING to EASING/NEUTRAL.
    (Spec TEIL3 §9.4.2)
    """
    fed_zones = chart_data.get("cycles", {}).get("FED_RATES", {}).get("phase_zones", [])
    if not fed_zones:
        return []

    pivots = []
    for i in range(1, len(fed_zones)):
        prev_phase = fed_zones[i - 1]["phase"]
        curr_phase = fed_zones[i]["phase"]
        if prev_phase in _RESTRICTIVE_PHASES and curr_phase in _EASING_PHASES:
            pivots.append({
                "month": fed_zones[i]["start"],
                "from_phase": prev_phase,
                "to_phase": curr_phase,
            })

    return pivots


def compute_fed_pivot_probability(chart_data, cycle_month_maps):
    """
    Compute probability of Fed pivot within N months, conditioned on cluster states.
    (Spec TEIL3 §9.4.3)
    """
    pivot_events = _identify_fed_pivot_events(chart_data)
    pivot_months_set = {e["month"] for e in pivot_events}

    # Common months
    dominant_cycles = [CYCLE_CLUSTERS[c]["dominant"] for c in CLUSTER_ORDER]
    common_months = None
    for dc in dominant_cycles:
        if dc in cycle_month_maps:
            dc_months = set(cycle_month_maps[dc].keys())
            common_months = dc_months if common_months is None else common_months & dc_months

    if not common_months:
        return {"fed_pivot_events": pivot_events, "pivot_probability_by_cluster": {}}

    common_months = sorted(common_months)

    # Build cluster state per month
    credit_dom = CYCLE_CLUSTERS["CREDIT_CLUSTER"]["dominant"]
    real_dom = CYCLE_CLUSTERS["REAL_ECONOMY_CLUSTER"]["dominant"]
    credit_map = cycle_month_maps.get(credit_dom, {})
    real_map = cycle_month_maps.get(real_dom, {})

    # Group months by relevant cluster keys
    cluster_groups = {}
    for m in common_months:
        cp = credit_map.get(m)
        rp = real_map.get(m)
        if not cp:
            continue
        cb = _phase_to_bucket(cp)

        # Single: CREDIT_BEAR, CREDIT_BULL, etc.
        key_single = f"CREDIT_{cb}"
        cluster_groups.setdefault(key_single, []).append(m)

        # Dual: CREDIT_BEAR__REAL_BEAR, etc.
        if rp:
            rb = _phase_to_bucket(rp)
            key_dual = f"CREDIT_{cb}__REAL_{rb}"
            cluster_groups.setdefault(key_dual, []).append(m)

    # For each group: how often did a pivot occur within 6M / 12M?
    pivot_probs = {}
    for key, months in cluster_groups.items():
        pivots_6m = 0
        pivots_12m = 0
        for m in months:
            for horizon in [6, 12]:
                found = False
                for fwd in range(1, horizon + 1):
                    if _add_months(m, fwd) in pivot_months_set:
                        found = True
                        break
                if found:
                    if horizon == 6:
                        pivots_6m += 1
                    else:
                        pivots_12m += 1

        n = len(months)
        pivot_probs[key] = {
            "pivot_within_6m": round(pivots_6m / n, 3) if n > 0 else 0,
            "pivot_within_12m": round(pivots_12m / n, 3) if n > 0 else 0,
            "n_months_in_state": n,
            "n_pivots_6m": pivots_6m,
            "n_pivots_12m": pivots_12m,
        }

    # Enrich pivot events with cluster state at time of pivot
    enriched_events = []
    for ev in pivot_events:
        pm = ev["month"]
        ev_enriched = dict(ev)
        cluster_state = {}
        for cluster_name in CLUSTER_ORDER:
            dominant = CYCLE_CLUSTERS[cluster_name]["dominant"]
            phase = cycle_month_maps.get(dominant, {}).get(pm)
            cluster_state[cluster_name] = _phase_to_bucket(phase) if phase else "UNKNOWN"
        ev_enriched["cluster_state_at_pivot"] = cluster_state
        enriched_events.append(ev_enriched)

    return {
        "fed_pivot_events": enriched_events,
        "pivot_probability_by_cluster": pivot_probs,
    }


# --- C4: Crash vs. Correction (Spec TEIL3 §9.5) ---

def _build_month_bucket_map(phase_zones):
    """Build month → bucket mapping from phase zones."""
    month_map = _expand_zone_months(phase_zones)
    return {m: _phase_to_bucket(p) for m, p in month_map.items()}


def compute_crash_vs_correction(chart_data, monthly_returns, baselines):
    """
    Compute historical drawdown statistics conditioned on Credit+Business dual state.
    (Spec TEIL3 §9.5)
    """
    credit_zones = chart_data.get("cycles", {}).get("CREDIT", {}).get("phase_zones", [])
    business_zones = chart_data.get("cycles", {}).get("BUSINESS", {}).get("phase_zones", [])

    if not credit_zones or not business_zones:
        return {}

    credit_map = _build_month_bucket_map(credit_zones)
    business_map = _build_month_bucket_map(business_zones)

    common = sorted(set(credit_map.keys()) & set(business_map.keys()))

    # Group months by dual-state
    dual_groups = {}
    for m in common:
        cb = credit_map[m]
        bb = business_map[m]
        key = f"CREDIT_{cb}__BUSINESS_{bb}"
        dual_groups.setdefault(key, []).append(m)

    # Forward Returns + stats per group
    results = {}
    for key, months in dual_groups.items():
        offset = [_add_months(m, PHASE_DETECTION_LAG_MONTHS) for m in months]

        spy_rd = monthly_returns.get("SPY", {})
        spy_bl = baselines.get("SPY", {})

        group_result = {}
        for h in HORIZONS:
            bl = spy_bl.get(f"baseline_{h}m")
            crs = [_cumulative_return_clean(spy_rd, m, h) for m in offset]
            crs = [r for r in crs if r is not None]
            ni = _compute_n_independent(offset, h)
            group_result[f"spy_{h}m"] = _compute_return_stats(crs, bl, ni) if crs else None

        group_result["n_months"] = len(months)
        results[key] = group_result

    return {
        "dual_state_drawdowns": results,
        "entry_rules": ENTRY_RULES,
    }


# --- C5: Historical Analogues (Spec TEIL3 §9.6) ---

BUCKET_NUMERIC = {"BULLISH": 1, "NEUTRAL_MIXED": 0, "BEARISH": -1}


def _cluster_state_vector(cluster_states):
    """Convert cluster states to numeric vector for distance calculation."""
    return [BUCKET_NUMERIC.get(cluster_states.get(c, "NEUTRAL_MIXED"), 0)
            for c in CLUSTER_ORDER]


def compute_historical_analogues(chart_data, cycle_month_maps, monthly_returns,
                                 current_cluster_state, n_analogues=N_HISTORICAL_ANALOGUES):
    """
    Find N most similar historical periods to current cluster state.
    (Spec TEIL3 §9.6)
    """
    current_vec = _cluster_state_vector(current_cluster_state)

    # Common months
    dominant_cycles = [CYCLE_CLUSTERS[c]["dominant"] for c in CLUSTER_ORDER]
    common_months = None
    for dc in dominant_cycles:
        if dc in cycle_month_maps:
            dc_months = set(cycle_month_maps[dc].keys())
            common_months = dc_months if common_months is None else common_months & dc_months

    if not common_months:
        return {"analogues": []}

    common_months = sorted(common_months)

    # Build cluster state for each historical month
    monthly_cluster_states = {}
    for m in common_months:
        state = {}
        for cluster_name in CLUSTER_ORDER:
            dominant = CYCLE_CLUSTERS[cluster_name]["dominant"]
            phase = cycle_month_maps.get(dominant, {}).get(m)
            state[cluster_name] = _phase_to_bucket(phase) if phase else "NEUTRAL_MIXED"
        monthly_cluster_states[m] = state

    # Weighted Euclidean distance
    weights = [CLUSTER_DISTANCE_WEIGHTS[c] for c in CLUSTER_ORDER]

    distances = []
    for m, hist_state in monthly_cluster_states.items():
        hist_vec = _cluster_state_vector(hist_state)
        dist = sum(w * (a - b) ** 2 for a, b, w
                   in zip(current_vec, hist_vec, weights)) ** 0.5
        distances.append((m, dist, hist_state))

    distances.sort(key=lambda x: x[1])

    # De-duplicate: consecutive months → only one per episode (same year = same episode)
    analogues = []
    used_periods = set()
    for m, dist, state in distances:
        period_key = m[:4]  # Same year = same episode
        if period_key in used_periods:
            continue
        used_periods.add(period_key)

        # Forward returns for key assets
        offset_m = _add_months(m, PHASE_DETECTION_LAG_MONTHS)
        spy_6m = _cumulative_return_clean(monthly_returns.get("SPY", {}), offset_m, 6)
        gld_6m = _cumulative_return_clean(monthly_returns.get("GLD", {}), offset_m, 6)
        tlt_6m = _cumulative_return_clean(monthly_returns.get("TLT", {}), offset_m, 6)

        similarity = round(1 / (1 + dist), 3)

        analogues.append({
            "period_start": m,
            "similarity_score": similarity,
            "cluster_state": state,
            "what_happened_next": {
                "spy_6m_return": round(spy_6m, 4) if spy_6m is not None else None,
                "gld_6m_return": round(gld_6m, 4) if gld_6m is not None else None,
                "tlt_6m_return": round(tlt_6m, 4) if tlt_6m is not None else None,
            },
        })

        if len(analogues) >= n_analogues:
            break

    # Consensus from analogues
    consensus = _compute_analogue_consensus(analogues)

    return {"analogues": analogues, "consensus": consensus}


def _compute_analogue_consensus(analogues):
    """Compute consensus direction and average returns from analogues."""
    if not analogues:
        return {}

    spy_returns = [a["what_happened_next"]["spy_6m_return"]
                   for a in analogues if a["what_happened_next"].get("spy_6m_return") is not None]
    gld_returns = [a["what_happened_next"]["gld_6m_return"]
                   for a in analogues if a["what_happened_next"].get("gld_6m_return") is not None]

    avg_spy = round(sum(spy_returns) / len(spy_returns), 4) if spy_returns else None

    if avg_spy is not None:
        if avg_spy > 0.03:
            direction = "BULLISH"
        elif avg_spy < -0.03:
            direction = "BEARISH"
        else:
            direction = "MIXED"
    else:
        direction = "UNKNOWN"

    return {
        "direction": direction,
        "avg_spy_6m": avg_spy,
        "avg_gld_6m": round(sum(gld_returns) / len(gld_returns), 4) if gld_returns else None,
        "n_analogues": len(analogues),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODUL D: TRANSITION ENGINE (Spec TEIL4 §11)
# ═══════════════════════════════════════════════════════════════════════════

# --- D1: Transition Matrices (Spec TEIL4 §11.2) ---

def compute_transition_matrices(chart_data):
    """
    Compute phase transition statistics for all cycles.
    (Spec TEIL4 §11.2)
    """
    results = {}

    for cycle_id in LEAD_LAG_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cycle_id, {})
        zones = cdata.get("phase_zones", [])
        if not zones or len(zones) < 3:
            continue

        # Compute duration per zone
        zone_durations = []
        for z in zones:
            start = z["start"]
            end = z["end"]
            duration = _months_between(start, end) + 1  # Inclusive
            zone_durations.append({
                "phase": z["phase"],
                "start": start,
                "end": end,
                "duration": duration,
            })

        # Phase statistics
        phase_stats = {}
        phases = sorted(set(z["phase"] for z in zone_durations))

        for phase in phases:
            phase_zones_list = [z for z in zone_durations if z["phase"] == phase]
            durations = [z["duration"] for z in phase_zones_list]
            n = len(durations)

            if n == 0:
                continue

            sd = sorted(durations)
            mean_d = sum(durations) / n
            dur_stats = {
                "median": sd[n // 2],
                "mean": round(mean_d, 1),
                "min": sd[0],
                "max": sd[-1],
                "std": round((sum((d - mean_d) ** 2 for d in durations) / n) ** 0.5, 1),
                "all_durations": durations,
            }

            # Transitions: what followed this phase?
            transitions = {}
            for i, z in enumerate(zone_durations):
                if z["phase"] == phase and i + 1 < len(zone_durations):
                    next_phase = zone_durations[i + 1]["phase"]
                    transitions[next_phase] = transitions.get(next_phase, 0) + 1

            total_trans = sum(transitions.values())
            trans_probs = {}
            for next_p, count in transitions.items():
                trans_probs[next_p] = {
                    "count": count,
                    "probability": round(count / total_trans, 3) if total_trans > 0 else 0,
                }

            phase_stats[phase] = {
                "n_occurrences": n,
                "duration_months": dur_stats,
                "transitions_to": trans_probs,
            }

        results[cycle_id] = {
            "phases": phase_stats,
            "total_transitions": len(zone_durations) - 1,
        }

    return results


# --- D2: Phase Position Score (Spec TEIL4 §11.3) ---

def compute_phase_positions(chart_data, transition_matrices):
    """
    Compute current phase position (0-100%+) for each cycle.
    (Spec TEIL4 §11.3)
    """
    now_month = datetime.now().strftime("%Y-%m")
    results = {}

    for cycle_id in LEAD_LAG_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cycle_id, {})
        zones = cdata.get("phase_zones", [])
        if not zones:
            continue

        # Current phase = last zone
        current_zone = zones[-1]
        current_phase = current_zone["phase"]
        current_start = current_zone["start"]
        current_duration = _months_between(current_start, now_month)

        # Transition matrix for this phase
        tm = transition_matrices.get(cycle_id, {}).get("phases", {}).get(current_phase)
        if not tm:
            results[cycle_id] = {
                "current_phase": current_phase,
                "current_duration_months": current_duration,
                "phase_position_pct": None,
                "status": "NO_HISTORY",
            }
            continue

        median_dur = tm["duration_months"]["median"]
        position = round(current_duration / median_dur * 100) if median_dur > 0 else None

        # Remaining: based on historical durations that were LONGER than current
        all_durs = tm["duration_months"].get("all_durations", [])
        longer = [d for d in all_durs if d > current_duration]
        if longer:
            sl = sorted(longer)
            remaining_median = round(sl[len(sl) // 2] - current_duration)
            remaining_p25 = max(0, round(sl[int(len(sl) * 0.25)] - current_duration))
            remaining_p75 = round(sl[min(len(sl) - 1, int(len(sl) * 0.75))] - current_duration)
        else:
            remaining_median = 0
            remaining_p25 = 0
            remaining_p75 = 0

        # Status Label (Spec TEIL4 §11.3.3)
        if position is None:
            status = "NO_HISTORY"
        elif position < 40:
            status = "EARLY_PHASE"
        elif position < 70:
            status = "MID_PHASE"
        elif position < 100:
            status = "LATE_PHASE"
        else:
            status = "EXTENDED"

        results[cycle_id] = {
            "current_phase": current_phase,
            "current_duration_months": current_duration,
            "median_duration": median_dur,
            "phase_position_pct": position,
            "remaining_median": remaining_median,
            "remaining_p25": remaining_p25,
            "remaining_p75": remaining_p75,
            "status": status,
            "transitions_ahead": tm.get("transitions_to", {}),
        }

    return results


# --- D3: Conditional Remaining Durations (Spec TEIL4 §11.4) ---

def _compute_bucket_zones(phase_zones):
    """Convert detail phase zones to bucket zones (BULLISH/BEARISH/NEUTRAL_MIXED)."""
    if not phase_zones:
        return []

    bucket_zones = []
    for z in phase_zones:
        bucket = _phase_to_bucket(z["phase"])
        if bucket_zones and bucket_zones[-1]["bucket"] == bucket:
            bucket_zones[-1]["end"] = z["end"]
        else:
            bucket_zones.append({
                "start": z["start"],
                "end": z["end"],
                "bucket": bucket,
            })
    return bucket_zones


def _compute_smoothed_bucket_zones(cdata):
    """
    Compute stable bucket zones from smoothed indicator vs MA.
    Uses 2x12M smoothed curve (the 'sine wave') compared to 12M MA.
    This eliminates noise from short-lived phase flips.
    
    Smoothed > MA * 1.01 → BULLISH
    Smoothed < MA * 0.99 → BEARISH
    Otherwise → NEUTRAL_MIXED
    
    Falls back to _compute_bucket_zones(phase_zones) if smoothed data unavailable.
    """
    smoothed_raw = cdata.get("smoothed", [])
    ma_raw = cdata.get("ma_12m", [])
    
    if not smoothed_raw or not ma_raw:
        # Fallback to phase-based
        return _compute_bucket_zones(cdata.get("phase_zones", []))
    
    # Build month→value maps (handle both dict and array formats)
    smooth_map = {}
    ma_map = {}
    
    if isinstance(smoothed_raw[0], dict):
        # Dict format: [{date, value}, ...]
        for pt in smoothed_raw:
            if pt.get("value") is not None and pt.get("date"):
                smooth_map[pt["date"][:7]] = pt["value"]
        for pt in ma_raw:
            if pt.get("value") is not None and pt.get("date"):
                ma_map[pt["date"][:7]] = pt["value"]
    else:
        # Array format: need dates from indicator
        dates = []
        indicator = cdata.get("indicator", [])
        if isinstance(indicator, list) and indicator and isinstance(indicator[0], dict):
            dates = [pt.get("date", "")[:7] for pt in indicator if pt.get("date")]
        
        if dates:
            for i, d in enumerate(dates):
                if i < len(smoothed_raw) and smoothed_raw[i] is not None:
                    smooth_map[d] = smoothed_raw[i]
                if i < len(ma_raw) and ma_raw[i] is not None:
                    ma_map[d] = ma_raw[i]
    
    if not smooth_map or not ma_map:
        return _compute_bucket_zones(cdata.get("phase_zones", []))
    
    # Smoothed vs MA → bucket per month
    common = sorted(set(smooth_map.keys()) & set(ma_map.keys()))
    if not common:
        return _compute_bucket_zones(cdata.get("phase_zones", []))
    
    bucket_zones = []
    for m in common:
        sv = smooth_map[m]
        mv = ma_map[m]
        
        if mv == 0:
            bucket = "NEUTRAL_MIXED"
        elif sv > mv * 1.01:
            bucket = "BULLISH"
        elif sv < mv * 0.99:
            bucket = "BEARISH"
        else:
            bucket = "NEUTRAL_MIXED"
        
        if bucket_zones and bucket_zones[-1]["bucket"] == bucket:
            bucket_zones[-1]["end"] = m
        else:
            bucket_zones.append({"start": m, "end": m, "bucket": bucket})
    
    return bucket_zones


def _get_bucket_at_month(bucket_zones, month):
    """Find bucket state at a given month."""
    for bz in bucket_zones:
        if bz["start"] <= month <= bz["end"]:
            return bz["bucket"]
    return None


def _find_next_bucket_change(bucket_zones, after_month, from_bucket):
    """Find the month when bucket changes from from_bucket to something else, after after_month."""
    for bz in bucket_zones:
        if bz["start"] > after_month and bz["bucket"] != from_bucket:
            return bz["start"]
    return None


def compute_conditional_remaining_durations(chart_data):
    """
    For each causal chain pair: How long does Cycle B remain BULLISH
    after Cycle A transitions to BEARISH?
    (Spec TEIL4 §11.4)
    """
    results = {}

    for ca, cb in CAUSAL_CHAIN_PAIRS:
        key = f"{ca}_warns_{cb}"

        zones_a = chart_data.get("cycles", {}).get(ca, {}).get("phase_zones", [])
        zones_b = chart_data.get("cycles", {}).get(cb, {}).get("phase_zones", [])
        if not zones_a or not zones_b:
            continue

        a_bucket_zones = _compute_smoothed_bucket_zones(chart_data.get("cycles", {}).get(ca, {}))
        b_bucket_zones = _compute_smoothed_bucket_zones(chart_data.get("cycles", {}).get(cb, {}))

        # Find events: A transitions from BULLISH → BEARISH
        events = []
        for i in range(1, len(a_bucket_zones)):
            if (a_bucket_zones[i - 1]["bucket"] == "BULLISH" and
                    a_bucket_zones[i]["bucket"] == "BEARISH"):
                a_turned_month = a_bucket_zones[i]["start"]

                # Was B still BULLISH at this moment?
                b_state = _get_bucket_at_month(b_bucket_zones, a_turned_month)

                if b_state == "BULLISH":
                    b_turned_month = _find_next_bucket_change(
                        b_bucket_zones, a_turned_month, "BULLISH")

                    if b_turned_month:
                        remaining = _months_between(a_turned_month, b_turned_month)
                        events.append({
                            f"{ca.lower()}_turned_bearish": a_turned_month,
                            f"{cb.lower()}_was_bullish_until": b_turned_month,
                            f"{cb.lower()}_remaining_months": remaining,
                        })

        if events:
            remaining_months = [e[f"{cb.lower()}_remaining_months"] for e in events]
            sr = sorted(remaining_months)
            n = len(sr)
            stats = {
                "median": sr[n // 2],
                "mean": round(sum(remaining_months) / n, 1),
                "min": sr[0],
                "max": sr[-1],
                "n_events": n,
            }
        else:
            stats = {"n_events": 0}

        results[key] = {
            "description": f"Wenn {ca} BEARISH wird — wie lange bleibt {cb} noch BULLISH?",
            "events": events,
            "remaining_months_stats": stats,
        }

    return results


# --- D4: Cascade Speed Index (Spec TEIL4 §11.5) ---

def _bucket_severity(bucket):
    return {"BULLISH": 0, "NEUTRAL_MIXED": 1, "BEARISH": 2}.get(bucket, 0)


def _compute_historical_cascade_speeds(chart_data, lookback_months):
    """Compute cascade speed for every historical month using smoothed bucket zones."""
    # Build smoothed bucket zones for all cycles
    all_bucket_zones = {}
    for cycle_id in LEAD_LAG_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cycle_id, {})
        bz = _compute_smoothed_bucket_zones(cdata)
        if bz:
            all_bucket_zones[cycle_id] = bz

    if not all_bucket_zones:
        return []

    # Find date range
    all_starts = []
    all_ends = []
    for bz_list in all_bucket_zones.values():
        for bz in bz_list:
            all_starts.append(bz["start"])
            all_ends.append(bz["end"])
    if not all_starts:
        return []

    first_month = _add_months(min(all_starts), lookback_months)
    last_month = max(all_ends)

    historical = []
    m = first_month
    while m <= last_month:
        lookback_start = _add_months(m, -lookback_months)
        n_transitions = 0

        for cycle_id, bz_list in all_bucket_zones.items():
            for i in range(1, len(bz_list)):
                trans_month = bz_list[i]["start"]
                if lookback_start < trans_month <= m:
                    prev_b = bz_list[i - 1]["bucket"]
                    curr_b = bz_list[i]["bucket"]
                    if _bucket_severity(curr_b) > _bucket_severity(prev_b):
                        n_transitions += 1
                        break  # Only latest per cycle

        speed = round(n_transitions / lookback_months, 3)
        historical.append({"month": m, "speed": speed})
        m = _next_month(m)

    return historical


def _calibrate_cascade_speeds(historical_speeds, monthly_returns, v16_monthly):
    """
    Calibrate cascade speed thresholds against V16 outcomes and SPY returns.
    (Spec TEIL4 §11.5.3)
    """
    buckets = {
        "speed_below_0.2": [],
        "speed_0.2_to_0.5": [],
        "speed_above_0.5": [],
        "speed_above_0.8": [],
    }

    for entry in historical_speeds:
        s = entry["speed"]
        m = entry["month"]

        if s >= 0.8:
            buckets["speed_above_0.8"].append(m)
        if s >= 0.5:
            buckets["speed_above_0.5"].append(m)
        elif s >= 0.2:
            buckets["speed_0.2_to_0.5"].append(m)
        else:
            buckets["speed_below_0.2"].append(m)

    results = {}
    label_map = {
        "speed_below_0.2": "CALM",
        "speed_0.2_to_0.5": "MODERATE",
        "speed_above_0.5": "CASCADE",
        "speed_above_0.8": "CRISIS",
    }

    for bname, months in buckets.items():
        if not months:
            results[bname] = {"label": label_map.get(bname, ""), "n_months": 0}
            continue

        # V16 stays growth in 6M
        growth_count = 0
        v16_valid = 0
        for m in months:
            worst = "GROWTH"
            has_data = False
            for i in range(1, 7):
                fwd_state = v16_monthly.get(_add_months(m, i))
                if fwd_state is not None:
                    has_data = True
                    fg = _state_to_group(fwd_state)
                    if _group_severity(fg) > _group_severity(worst):
                        worst = fg
            if has_data:
                v16_valid += 1
                if worst == "GROWTH":
                    growth_count += 1

        # SPY 6M returns
        spy_rd = monthly_returns.get("SPY", {})
        spy_returns = [_cumulative_return_clean(spy_rd, _add_months(m, 2), 6) for m in months]
        spy_returns = [r for r in spy_returns if r is not None]

        results[bname] = {
            "label": label_map.get(bname, ""),
            "v16_stays_growth_pct": round(growth_count / v16_valid * 100) if v16_valid > 0 else None,
            "avg_spy_6m": round(sum(spy_returns) / len(spy_returns), 4) if spy_returns else None,
            "n_months": len(months),
        }

    return results


def compute_cascade_speed(chart_data, monthly_returns=None, v16_monthly=None,
                          lookback_months=CASCADE_LOOKBACK_MONTHS):
    """
    Compute current cascade speed and historical calibration.
    (Spec TEIL4 §11.5)
    """
    now_month = datetime.now().strftime("%Y-%m")
    lookback_start = _add_months(now_month, -lookback_months)

    transitions = []
    not_transitioned = []

    for cycle_id in LEAD_LAG_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cycle_id, {})
        if not cdata:
            continue

        bucket_zones = _compute_smoothed_bucket_zones(cdata)

        found = False
        for i in range(1, len(bucket_zones)):
            trans_month = bucket_zones[i]["start"]
            if lookback_start < trans_month <= now_month:
                prev_bucket = bucket_zones[i - 1]["bucket"]
                curr_bucket = bucket_zones[i]["bucket"]

                if _bucket_severity(curr_bucket) > _bucket_severity(prev_bucket):
                    transitions.append({
                        "cycle": cycle_id,
                        "month": trans_month,
                        "from": prev_bucket,
                        "to": curr_bucket,
                    })
                    found = True
                    break

        if not found:
            not_transitioned.append(cycle_id)

    speed = round(len(transitions) / lookback_months, 3)

    if speed >= 0.8:
        severity = "CRISIS"
    elif speed >= 0.5:
        severity = "CASCADE"
    elif speed >= 0.2:
        severity = "MODERATE"
    else:
        severity = "CALM"

    # Historical cascade speeds + calibration
    historical = _compute_historical_cascade_speeds(chart_data, lookback_months)

    calibration = {}
    if monthly_returns and v16_monthly:
        calibration = _calibrate_cascade_speeds(historical, monthly_returns, v16_monthly)

    return {
        "current": {
            "cascade_speed": speed,
            "n_transitions": len(transitions),
            "lookback_months": lookback_months,
            "transitioned_cycles": transitions,
            "not_yet_transitioned": not_transitioned,
            "severity": severity,
        },
        "historical_cascade_speeds": historical,
        "calibration": calibration,
    }


# --- D5: Confirmation Counter (Spec TEIL4 §11.6) ---

def compute_confirmation_counter(chart_data):
    """
    Count how many of the 9 non-political cycles are in each bucket.
    Uses smoothed bucket zones for stable regime assessment.
    (Spec TEIL4 §11.6)
    """
    counts = {"BULLISH": [], "BEARISH": [], "NEUTRAL_MIXED": []}

    for cycle_id in LEAD_LAG_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cycle_id, {})
        if not cdata:
            continue

        bucket_zones = _compute_smoothed_bucket_zones(cdata)
        if not bucket_zones:
            continue

        # Current bucket = last smoothed bucket zone
        current_bucket = bucket_zones[-1]["bucket"]
        counts[current_bucket].append(cycle_id)

    n_bull = len(counts["BULLISH"])
    n_bear = len(counts["BEARISH"])
    n_neut = len(counts["NEUTRAL_MIXED"])

    score = n_bull - n_bear

    if score >= 5:
        interp = "Starker Konsens bullish. Risk-On."
    elif score >= 2:
        interp = "Mehrheitlich bullish, aber nicht einstimmig. Vorsichtig Risk-On."
    elif score >= -1:
        interp = "Divergenz. Cluster-Dominance entscheidet (Credit > Real Economy)."
    elif score >= -4:
        interp = "Mehrheitlich bearish. Defensive Positionierung empfohlen."
    else:
        interp = "Breiter Konsens bearish. Maximale Vorsicht. Crash-Risiko erhoeht."

    return {
        "bullish_count": n_bull,
        "bearish_count": n_bear,
        "neutral_count": n_neut,
        "bullish_cycles": counts["BULLISH"],
        "bearish_cycles": counts["BEARISH"],
        "neutral_cycles": counts["NEUTRAL_MIXED"],
        "confirmation_score": score,
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODUL E: ORCHESTRATOR + OUTPUT (Spec TEIL5 §13)
# ═══════════════════════════════════════════════════════════════════════════

def _get_current_cluster_state(chart_data):
    """Determine current cluster state from smoothed bucket zones. (Spec TEIL5 §13.2)"""
    state = {}
    for cluster_name, cluster_def in CYCLE_CLUSTERS.items():
        dominant = cluster_def["dominant"]
        cdata = chart_data.get("cycles", {}).get(dominant, {})
        bucket_zones = _compute_smoothed_bucket_zones(cdata)
        if bucket_zones:
            state[cluster_name] = bucket_zones[-1]["bucket"]
        else:
            state[cluster_name] = "NEUTRAL_MIXED"
    return state


def _compute_overall_assessment_v11(cond_returns, cluster_returns, v16_transitions,
                                     fed_pivot, crash_corr, cascade, confirmation,
                                     phase_positions):
    """
    Compute honest overall assessment with V1.1 metrics.
    (Spec TEIL5 §13.3)
    """
    # Significant conditional returns
    n_sig, n_total = 0, 0
    for cid, phases in cond_returns.items():
        for phase, assets in phases.items():
            if not isinstance(assets, dict):
                continue
            for asset, horizons in assets.items():
                if not isinstance(horizons, dict):
                    continue
                for hk, stats in horizons.items():
                    if stats and isinstance(stats, dict) and stats.get("avg") is not None:
                        n_total += 1
                        if stats.get("significant"):
                            n_sig += 1

    # Cascade severity
    cascade_severity = cascade.get("current", {}).get("severity", "CALM")

    # Confirmation score
    conf_score = confirmation.get("confirmation_score", 0)

    # Phase positions: any EXTENDED?
    extended_cycles = [cid for cid, pp in phase_positions.items()
                       if pp.get("status") == "EXTENDED"]
    late_cycles = [cid for cid, pp in phase_positions.items()
                   if pp.get("status") == "LATE_PHASE"]

    # Build verdict
    if cascade_severity in ("CASCADE", "CRISIS"):
        verdict = (f"WARNING — Cascade Speed {cascade_severity}. "
                   f"{cascade.get('current', {}).get('n_transitions', 0)} cycles transitioning. "
                   f"Defensive positioning recommended.")
    elif conf_score <= -4:
        verdict = "BEARISH — Broad consensus across cycles. Maximum caution."
    elif conf_score <= -1:
        verdict = "CAUTIOUS — Majority bearish. Watch Credit+Business dual state."
    elif conf_score >= 5:
        verdict = "BULLISH — Strong consensus. Risk-on positioning supported."
    elif conf_score >= 2:
        verdict = "MODERATE BULLISH — Majority positive but not unanimous."
    else:
        verdict = "MIXED — Cycles diverge. Follow dominant cluster signals."

    if extended_cycles:
        verdict += f" WATCH: {', '.join(extended_cycles)} extended beyond median duration."

    return {
        "n_significant_returns": n_sig,
        "n_total_returns": n_total,
        "pct_significant": round(n_sig / n_total * 100, 1) if n_total > 0 else 0,
        "cascade_severity": cascade_severity,
        "confirmation_score": conf_score,
        "extended_cycles": extended_cycles,
        "late_cycles": late_cycles,
        "verdict": verdict,
    }


def run_lead_engine(data_dir=None):
    """
    Main orchestrator for Lead-Engine V1.1.
    Reads inputs from disk, runs all modules, writes 3 output JSONs.
    (Spec TEIL5 §13.1)

    Args:
        data_dir: Override DATA_DIR (for Colab). If None, uses module-level DATA_DIR.

    Returns:
        dict — overall assessment + all results (for Colab inspection)
    """
    dd = data_dir or DATA_DIR

    logger.info("=" * 60)
    logger.info("LEAD-ENGINE V1.1 START")
    logger.info(f"  Data dir: {dd}")
    logger.info("=" * 60)

    # ── LOAD INPUTS ──
    chart_data = _load_json(os.path.join(dd, "cycles_chart_data.json"))
    prices = _load_json(os.path.join(dd, "raw", "prices_history.json"))
    macro_states = _load_json(os.path.join(dd, "raw", "macro_state_history.json"))

    if not chart_data or not prices:
        logger.error("Missing input data — aborting")
        return None

    # Patch V4.0 chart_data
    chart_data = _ensure_chart_data_complete(chart_data)

    # ── DATA PREPARATION ──
    logger.info("Computing monthly returns (27 assets)...")
    monthly_returns = _compute_monthly_returns(prices)
    logger.info(f"  Assets with returns: {len(monthly_returns)}")

    logger.info("Computing baselines...")
    baselines, history_info = _compute_baselines(monthly_returns)

    # Pre-compute cycle month maps (used by multiple modules)
    cycle_month_maps = {}
    for cid in LEAD_LAG_CYCLES:
        cdata = chart_data.get("cycles", {}).get(cid, {})
        zones = cdata.get("phase_zones", [])
        if zones:
            cycle_month_maps[cid] = _expand_zone_months(zones)

    # V16 monthly states (for Modul C + D calibration)
    v16_monthly = {}
    if macro_states:
        for entry in macro_states:
            d = entry.get("date", "")
            s = entry.get("state")
            if d and s is not None:
                num = _parse_v16_state(s)
                if num is not None:
                    v16_monthly[d[:7]] = num

    # ── MODUL B: CONDITIONAL FORWARD RETURNS ──
    logger.info("MODUL B: Conditional Forward Returns (27 assets)...")
    cond_returns = compute_conditional_returns(
        chart_data, monthly_returns, baselines, history_info,
        daily_prices=prices
    )

    # ── MODUL C: REGIME INTERACTION ──
    logger.info("MODUL C: Regime-Interaktion...")

    logger.info("  C1: Cluster-Conditional-Returns...")
    cluster_returns = compute_cluster_conditional_returns(
        chart_data, monthly_returns, baselines, history_info)

    logger.info("  C2: V16 State Transition Probability...")
    v16_transitions = compute_v16_transition_probability(
        chart_data, macro_states or [], cycle_month_maps) if macro_states else {}

    logger.info("  C3: Fed Pivot Probability...")
    fed_pivot = compute_fed_pivot_probability(chart_data, cycle_month_maps)

    logger.info("  C4: Crash vs. Korrektur...")
    crash_corr = compute_crash_vs_correction(chart_data, monthly_returns, baselines)

    logger.info("  C5: Historische Analoga...")
    current_cluster_state = _get_current_cluster_state(chart_data)
    analogues = compute_historical_analogues(
        chart_data, cycle_month_maps, monthly_returns, current_cluster_state)

    # ── MODUL D: TRANSITION ENGINE ──
    logger.info("MODUL D: Transition Engine...")

    logger.info("  D1: Transition-Matrices...")
    transition_matrices = compute_transition_matrices(chart_data)

    logger.info("  D2: Phase-Positions...")
    phase_positions = compute_phase_positions(chart_data, transition_matrices)

    logger.info("  D3: Bedingte Restlaufzeiten...")
    remaining_durations = compute_conditional_remaining_durations(chart_data)

    logger.info("  D4: Cascade Speed...")
    cascade = compute_cascade_speed(chart_data, monthly_returns, v16_monthly)

    logger.info("  D5: Confirmation Counter...")
    confirmation = compute_confirmation_counter(chart_data)

    # ── OVERALL ASSESSMENT ──
    logger.info("Overall Assessment...")
    assessment = _compute_overall_assessment_v11(
        cond_returns, cluster_returns, v16_transitions,
        fed_pivot, crash_corr, cascade, confirmation, phase_positions)

    # ── WRITE OUTPUTS ──
    timestamp = datetime.now(timezone.utc).isoformat()

    for fname, data in [
        ("conditional_returns.json", {
            "computed_at": timestamp,
            "engine_version": "1.1",
            "phase_detection_lag_months": PHASE_DETECTION_LAG_MONTHS,
            "horizons": HORIZONS,
            "assets": ASSETS,
            "history_info": history_info,
            "baselines": baselines,
            "disruption_integration": "PREPARED",
            "overall_assessment": assessment,
            "conditional_returns": cond_returns,
        }),
        ("regime_interaction.json", {
            "computed_at": timestamp,
            "engine_version": "1.1",
            "cluster_definitions": {k: {"cycles": v["cycles"], "dominant": v["dominant"]}
                                    for k, v in CYCLE_CLUSTERS.items()},
            "cluster_conditional_returns": cluster_returns,
            "v16_transition_probability": v16_transitions,
            "fed_pivot_probability": fed_pivot,
            "crash_vs_correction": crash_corr,
            "historical_analogues": analogues,
            "overall_assessment": assessment,
        }),
        ("transition_engine.json", {
            "computed_at": timestamp,
            "engine_version": "1.1",
            "transition_matrices": transition_matrices,
            "phase_positions": phase_positions,
            "conditional_remaining_durations": remaining_durations,
            "cascade_speed": cascade,
            "confirmation_counter": confirmation,
            "overall_assessment": assessment,
        }),
    ]:
        path = os.path.join(dd, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False, default=str)
            sz = os.path.getsize(path) / 1024
            logger.info(f"  Written: {fname} ({sz:.0f} KB)")
        except Exception as e:
            logger.error(f"  Write failed {fname}: {e}")

    # ── SUMMARY LOG ──
    logger.info("=" * 60)
    logger.info("LEAD-ENGINE V1.1 COMPLETE")
    logger.info(f"  Assets: {len(monthly_returns)}")
    logger.info(f"  Cascade Speed: {cascade.get('current', {}).get('severity', 'N/A')}")
    logger.info(f"  Confirmation: {confirmation.get('confirmation_score', 'N/A')}")
    logger.info(f"  Verdict: {assessment.get('verdict', 'N/A')}")
    logger.info("=" * 60)

    return {
        "assessment": assessment,
        "conditional_returns": cond_returns,
        "cluster_returns": cluster_returns,
        "v16_transitions": v16_transitions,
        "fed_pivot": fed_pivot,
        "crash_correction": crash_corr,
        "analogues": analogues,
        "transition_matrices": transition_matrices,
        "phase_positions": phase_positions,
        "remaining_durations": remaining_durations,
        "cascade": cascade,
        "confirmation": confirmation,
        "baselines": baselines,
        "history_info": history_info,
    }
