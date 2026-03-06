"""
step2_signal_generator/router_proximity.py
Router Proximity — Trigger Proximity Calculation, Quick Check, Trends

Source: Signal Generator Spec Teil 2 §10, Teil 4 §21.1, Dashboard Writer Addendum
"""

import logging
from typing import Optional

logger = logging.getLogger("signal_generator.router_proximity")


# ============================================================
# FIELD VALUE EXTRACTION
# ============================================================

def get_field_value(data: dict, field_path: str):
    """
    Extract a value from nested dict using dot notation.
    Example: get_field_value(data, "dxy.delta_126d") → data["dxy"]["delta_126d"]
    """
    parts = field_path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


# ============================================================
# THRESHOLD ADJUSTMENT (Fragility)
# ============================================================

def get_adjusted_thresholds(trigger_config: dict, fragility_state: str, fragility_overlay: dict) -> dict:
    """
    Get thresholds adjusted for current Fragility State.
    Spec Teil 2 §8.4

    HEALTHY:  threshold_healthy (standard)
    ELEVATED: threshold_elevated (lowered, easier to fire)
    EXTREME:  threshold_extreme (same as elevated)
    CRISIS:   threshold_elevated (router override handled separately)
    """
    overlay = fragility_overlay.get(fragility_state, fragility_overlay.get("HEALTHY", {}))
    threshold_key = overlay.get("threshold_key", "threshold_healthy")

    adjusted = {}
    conditions = trigger_config.get("conditions", {})
    for cond_name, cond in conditions.items():
        # Try fragility-specific threshold, fall back to healthy, fall back to generic
        value = cond.get(threshold_key)
        if value is None:
            value = cond.get("threshold_healthy")
        if value is None:
            value = cond.get("threshold")
        adjusted[cond_name] = value

    return adjusted


# ============================================================
# FULL PROXIMITY CALCULATION
# ============================================================

def calculate_trigger_proximity(trigger_config: dict, router_raw_data: dict,
                                v16_regime: str, fragility_state: str,
                                fragility_overlay: dict) -> dict:
    """
    Calculate how close a single trigger is to firing.
    Spec Teil 2 §10.1

    Returns:
        {
            "composite": 0.0-1.0 (minimum of all individual proximities),
            "individual": {cond_name: 0.0-1.0, ...},
            "all_conditions_met": bool,
            "closest_to_trigger": str,
            "furthest_from_trigger": str,
            "dual_signal": {"fast_met": bool, "slow_met": bool}
        }
    """
    thresholds = get_adjusted_thresholds(trigger_config, fragility_state, fragility_overlay)
    conditions = trigger_config.get("conditions", {})
    proximities = {}

    for cond_name, cond in conditions.items():
        threshold = thresholds.get(cond_name)
        operator = cond.get("operator", "")

        # Handle regime checks (in / not_in)
        if operator == "in":
            allowed = cond.get("allowed", [])
            proximities[cond_name] = 1.0 if v16_regime in allowed else 0.0
            continue

        if operator == "not_in":
            blocked = cond.get("blocked", [])
            proximities[cond_name] = 1.0 if v16_regime not in blocked else 0.0
            continue

        # Handle numeric conditions
        field_path = cond.get("field", "")
        current_value = get_field_value(router_raw_data, field_path)

        if current_value is None or threshold is None:
            proximities[cond_name] = 0.0
            continue

        try:
            current_value = float(current_value)
            threshold = float(threshold)
        except (ValueError, TypeError):
            proximities[cond_name] = 0.0
            continue

        if operator == "<":
            if current_value <= threshold:
                proximities[cond_name] = 1.0
            else:
                distance = current_value - threshold
                max_distance = abs(threshold) if threshold != 0 else 1.0
                proximities[cond_name] = max(0.0, round(1.0 - (distance / max_distance), 4)) if max_distance > 0 else 0.0

        elif operator == ">":
            if current_value >= threshold:
                proximities[cond_name] = 1.0
            else:
                distance = threshold - current_value
                max_distance = abs(threshold) if threshold != 0 else 1.0
                proximities[cond_name] = max(0.0, round(1.0 - (distance / max_distance), 4)) if max_distance > 0 else 0.0

        else:
            proximities[cond_name] = 0.0

    # Composite: minimum of all (all must be met)
    composite = min(proximities.values()) if proximities else 0.0
    all_met = all(v >= 1.0 for v in proximities.values()) if proximities else False

    # Closest / furthest
    closest = max(proximities, key=proximities.get) if proximities else None
    furthest = min(proximities, key=proximities.get) if proximities else None

    # Dual Signal check
    dual_signal = _check_dual_signal(trigger_config, router_raw_data)

    return {
        "composite": round(composite, 4),
        "individual": {k: round(v, 4) for k, v in proximities.items()},
        "all_conditions_met": all_met,
        "closest_to_trigger": closest,
        "furthest_from_trigger": furthest,
        "dual_signal": dual_signal,
    }


# ============================================================
# DUAL SIGNAL CHECK
# ============================================================

def _check_dual_signal(trigger_config: dict, router_raw_data: dict) -> dict:
    """
    Check fast and slow signals for entry.
    Spec Teil 2 §8.1-8.3: Both fast AND slow must be positive for entry.
    """
    dual_cfg = trigger_config.get("dual_signal_entry", {})
    if not dual_cfg:
        return {"fast_met": False, "slow_met": False}

    fast_met = _check_signal_group(dual_cfg.get("fast_signal", {}), router_raw_data)
    slow_met = _check_signal_group(dual_cfg.get("slow_signal", {}), router_raw_data)

    return {"fast_met": fast_met, "slow_met": slow_met}


def _check_signal_group(signal_group: dict, router_raw_data: dict) -> bool:
    """
    Check if ALL conditions in a signal group are met.
    Supports: field > threshold, field_a > field_b (a_gt_b)
    """
    if not signal_group:
        return False

    for signal_name, signal_cfg in signal_group.items():
        operator = signal_cfg.get("operator", "")

        if operator == "a_gt_b":
            val_a = get_field_value(router_raw_data, signal_cfg.get("field_a", ""))
            val_b = get_field_value(router_raw_data, signal_cfg.get("field_b", ""))
            if val_a is None or val_b is None:
                return False
            try:
                if float(val_a) <= float(val_b):
                    return False
            except (ValueError, TypeError):
                return False

        elif operator == ">":
            val = get_field_value(router_raw_data, signal_cfg.get("field", ""))
            threshold = signal_cfg.get("threshold", 0)
            if val is None:
                return False
            try:
                if float(val) <= float(threshold):
                    return False
            except (ValueError, TypeError):
                return False

        elif operator == "<":
            val = get_field_value(router_raw_data, signal_cfg.get("field", ""))
            threshold = signal_cfg.get("threshold", 0)
            if val is None:
                return False
            try:
                if float(val) >= float(threshold):
                    return False
            except (ValueError, TypeError):
                return False

    return True


# ============================================================
# QUICK PROXIMITY CHECK (Fast Path)
# ============================================================

def quick_proximity_check(router_raw_data: dict) -> float:
    """
    Fast, rough estimate of max Router proximity.
    Only checks 2 key indicators per trigger.
    Spec Teil 4 §21.1

    Returns: max proximity across all triggers (0.0-1.0)
    """
    # EM_BROAD: DXY 6M + VWO/SPY relative
    dxy_6m = _safe_float(get_field_value(router_raw_data, "dxy.delta_126d"), 0.0)
    vwo_spy = _safe_float(get_field_value(router_raw_data, "relative.vwo_spy_126d"), 0.0)
    em_prox = min(
        min(1.0, max(0.0, dxy_6m / -0.05)) if dxy_6m < 0 else 0.0,
        min(1.0, max(0.0, vwo_spy / 0.10)) if vwo_spy > 0 else 0.0,
    )

    # CHINA_STIMULUS: Credit Impulse Z-Score
    china_ci = _safe_float(get_field_value(router_raw_data, "china_credit_impulse.zscore_2y"), 0.0)
    china_prox = min(1.0, max(0.0, china_ci / 2.0)) if china_ci > 0 else 0.0

    # COMMODITY_SUPER: DBC/SPY relative
    dbc_spy = _safe_float(get_field_value(router_raw_data, "relative.dbc_spy_126d"), 0.0)
    comm_prox = min(1.0, max(0.0, dbc_spy / 0.05)) if dbc_spy > 0 else 0.0

    return round(max(em_prox, china_prox, comm_prox), 4)


# ============================================================
# PROXIMITY TRENDS (Dashboard Writer Addendum)
# ============================================================

def compute_proximity_trends(today_proximity: dict, yesterday_proximity: dict,
                             config: dict) -> dict:
    """
    Compute trend per Router target: RISING, FALLING, STABLE, or NEW.
    Source: Dashboard Writer Addendum

    Args:
        today_proximity: {trigger_id: {composite: float, ...}}
        yesterday_proximity: {trigger_id: float} or None
        config: full config (for trend threshold)
    """
    timing = config.get("router_timing", {}).get("trend", {})
    threshold = timing.get("stable_threshold", 0.03)

    if yesterday_proximity is None:
        trends = {}
        for trigger_id, prox_data in today_proximity.items():
            composite = prox_data.get("composite", 0.0) if isinstance(prox_data, dict) else 0.0
            trends[trigger_id] = {
                "trend": "NEW",
                "delta": 0.0,
                "yesterday_composite": None,
            }
        return trends

    trends = {}
    for trigger_id, prox_data in today_proximity.items():
        today_val = prox_data.get("composite", 0.0) if isinstance(prox_data, dict) else 0.0
        yesterday_val = yesterday_proximity.get(trigger_id, 0.0)
        if yesterday_val is None:
            yesterday_val = 0.0

        delta = round(today_val - yesterday_val, 4)

        if delta > threshold:
            trend = "RISING"
        elif delta < -threshold:
            trend = "FALLING"
        else:
            trend = "STABLE"

        trends[trigger_id] = {
            "trend": trend,
            "delta": delta,
            "yesterday_composite": yesterday_val,
        }

    return trends


def inject_trends_into_proximity(proximity: dict, trends: dict) -> dict:
    """
    Inject trend/delta/yesterday_composite into each proximity entry.
    Dashboard Writer Addendum: trend fields live inside proximity.{target}.
    """
    for trigger_id, trend_data in trends.items():
        if trigger_id in proximity and isinstance(proximity[trigger_id], dict):
            proximity[trigger_id]["trend"] = trend_data["trend"]
            proximity[trigger_id]["delta"] = trend_data["delta"]
            proximity[trigger_id]["yesterday_composite"] = trend_data["yesterday_composite"]
    return proximity


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float safely."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
