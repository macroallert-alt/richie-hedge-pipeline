"""
step2_signal_generator/router_engine.py
Conviction Router Engine — Main Router Logic

Runs EVERY DAY. Entry evaluation only monthly.
Exit check and Proximity ALWAYS.

Source: Signal Generator Spec Teil 2 §12, Teil 4 §21.2
"""

import logging
from datetime import date
from typing import Optional

from .router_state import (
    check_emergency_exit,
    is_monthly_evaluation_day,
    get_next_evaluation_day,
    days_in_state,
)
from .router_proximity import (
    calculate_trigger_proximity,
    _check_dual_signal,
    get_field_value,
)

logger = logging.getLogger("signal_generator.router_engine")


def run_router_engine(router_raw_data: dict, v16_regime: str,
                      fragility_state: str, router_state: dict,
                      g7_input: Optional[dict], today: date,
                      config: dict) -> dict:
    """
    Full Router Engine run.
    Spec Teil 2 §12.1

    Steps:
      1. Emergency Exit Check
      2. Crisis Override Check
      3. Exit Check (daily, if router active)
      4. Trigger Proximity (daily, always)
      5. Entry Evaluation (monthly only)
      6. Assemble result

    Returns: router_output dict
    """
    current_state = router_state.get("current_state", "US_DOMESTIC")
    triggers_config = config.get("router_triggers", {}).get("triggers", {})
    fragility_overlay = config.get("router_fragility_overlay", {}).get("fragility_overlay", {})

    # ============================================================
    # STEP 1: EMERGENCY EXIT CHECK (always, before everything)
    # ============================================================
    emergency = check_emergency_exit(v16_regime, router_state, config)
    if emergency is not None:
        logger.warning(f"EMERGENCY EXIT: {emergency['reason']}")
        # Build minimal output with emergency
        return _build_emergency_output(emergency, router_state, today, fragility_state, config)

    # ============================================================
    # STEP 2: CRISIS OVERRIDE CHECK (Fragility CRISIS)
    # ============================================================
    crisis_override = False
    crisis_recommendation = None
    if fragility_state == "CRISIS" and current_state == "US_DOMESTIC":
        crisis_override = True
        crisis_recommendation = (
            "Fragility CRISIS active. Router empfiehlt volle International-Allokation "
            "(bis 25%) unabhaengig von Triggern. G7 Emergency Review empfohlen."
        )
        logger.warning(f"CRISIS OVERRIDE: {crisis_recommendation}")

    # ============================================================
    # STEP 3: EXIT CHECK (daily, if router active)
    # ============================================================
    exit_check = None
    if current_state != "US_DOMESTIC":
        exit_check = _evaluate_exit_conditions(
            current_state, router_raw_data, v16_regime, g7_input, triggers_config,
        )
        if exit_check is not None and exit_check.get("exit_triggered", False):
            logger.info(f"EXIT SIGNAL: {exit_check['exit_id']} — {exit_check['reason']}")

    # ============================================================
    # STEP 4: TRIGGER PROXIMITY (daily, always)
    # ============================================================
    proximity = {}
    for trigger_id, trigger_cfg in triggers_config.items():
        proximity[trigger_id] = calculate_trigger_proximity(
            trigger_cfg, router_raw_data, v16_regime, fragility_state, fragility_overlay,
        )

    max_prox = 0.0
    max_prox_trigger = None
    for tid, pdata in proximity.items():
        comp = pdata.get("composite", 0.0)
        if comp > max_prox:
            max_prox = comp
            max_prox_trigger = tid

    # ============================================================
    # STEP 5: ENTRY EVALUATION (monthly only)
    # ============================================================
    entry_evaluation = _build_entry_evaluation(
        current_state, router_state, today, proximity, triggers_config,
        router_raw_data, v16_regime, g7_input, fragility_state, fragility_overlay, config,
    )

    # ============================================================
    # STEP 6: ASSEMBLE RESULT
    # ============================================================
    return {
        "current_state": current_state,
        "state_since": router_state.get("state_since", "2025-01-01"),
        "days_in_state": days_in_state(router_state, today),
        "proximity": proximity,
        "max_proximity": round(max_prox, 4),
        "max_proximity_trigger": max_prox_trigger.upper() if max_prox_trigger else None,
        "entry_evaluation": entry_evaluation,
        "exit_check": exit_check,
        "emergency": None,
        "crisis_override": crisis_override,
        "crisis_recommendation": crisis_recommendation,
        "fragility_state_used": fragility_state,
        "thresholds_adjusted": fragility_state != "HEALTHY",
        "threshold_adjustment_note": _threshold_note(fragility_state) if fragility_state != "HEALTHY" else None,
    }


# ============================================================
# EXIT CONDITIONS
# ============================================================

def _evaluate_exit_conditions(current_state: str, router_raw_data: dict,
                              v16_regime: str, g7_input: Optional[dict],
                              triggers_config: dict) -> Optional[dict]:
    """
    Evaluate exit conditions for the current active state.
    Spec Teil 2 §9: ONE signal = exit recommendation.
    """
    trigger_cfg = triggers_config.get(current_state.lower())
    if trigger_cfg is None:
        return None

    exit_conditions = trigger_cfg.get("exit_conditions", [])

    for exit_cond in exit_conditions:
        exit_id = exit_cond.get("id", "unknown")
        severity = exit_cond.get("severity", "EXIT")

        triggered = _check_single_exit_condition(exit_cond, router_raw_data, v16_regime, g7_input)

        if triggered:
            return {
                "exit_triggered": True,
                "exit_id": exit_id,
                "severity": severity,
                "reason": exit_cond.get("action", "Exit condition met."),
                "description": exit_cond.get("description", ""),
            }

    return {
        "exit_triggered": False,
        "reason": "No exit conditions met",
    }


def _check_single_exit_condition(exit_cond: dict, router_raw_data: dict,
                                  v16_regime: str, g7_input: Optional[dict]) -> bool:
    """Check if a single exit condition is met."""

    # Type: g7_veto_check
    if exit_cond.get("type") == "g7_veto_check":
        return _check_g7_veto(exit_cond, g7_input)

    # Combined conditions (AND)
    if "conditions" in exit_cond and exit_cond.get("combine") == "AND":
        return all(
            _check_atomic_condition(sub, router_raw_data, v16_regime)
            for sub in exit_cond["conditions"]
        )

    # Regime check (in / not_in)
    operator = exit_cond.get("operator", "")
    if operator == "in" and "values" in exit_cond:
        return v16_regime in exit_cond["values"]
    if operator == "not_in" and "values" in exit_cond:
        return v16_regime not in exit_cond["values"]

    # Comparison operators (a_lt_b, <, >)
    return _check_atomic_condition(exit_cond, router_raw_data, v16_regime)


def _check_atomic_condition(cond: dict, router_raw_data: dict, v16_regime: str) -> bool:
    """Check a single atomic condition."""
    operator = cond.get("operator", "")

    if operator == "a_lt_b":
        val_a = get_field_value(router_raw_data, cond.get("field_a", ""))
        val_b = get_field_value(router_raw_data, cond.get("field_b", ""))
        if val_a is None or val_b is None:
            return False
        try:
            return float(val_a) < float(val_b)
        except (ValueError, TypeError):
            return False

    if operator == "a_gt_b":
        val_a = get_field_value(router_raw_data, cond.get("field_a", ""))
        val_b = get_field_value(router_raw_data, cond.get("field_b", ""))
        if val_a is None or val_b is None:
            return False
        try:
            return float(val_a) > float(val_b)
        except (ValueError, TypeError):
            return False

    if operator in ("<", ">"):
        field_path = cond.get("field", "")
        threshold = cond.get("threshold")
        val = get_field_value(router_raw_data, field_path)
        if val is None or threshold is None:
            return False
        try:
            val = float(val)
            threshold = float(threshold)
            if operator == "<":
                return val < threshold
            else:
                return val > threshold
        except (ValueError, TypeError):
            return False

    if operator == "in" and "values" in cond:
        return v16_regime in cond["values"]

    if operator == "not_in" and "values" in cond:
        return v16_regime not in cond["values"]

    return False


def _check_g7_veto(exit_cond: dict, g7_input: Optional[dict]) -> bool:
    """Check if G7 veto is active for given keywords. V1: G7 not available → no veto."""
    if g7_input is None:
        return False

    keywords = exit_cond.get("keywords", [])
    active_vetos = g7_input.get("active_vetos", [])

    for veto in active_vetos:
        veto_upper = str(veto).upper()
        for kw in keywords:
            if kw.upper() in veto_upper:
                return True

    return False


# ============================================================
# ENTRY EVALUATION
# ============================================================

def _build_entry_evaluation(current_state: str, router_state: dict, today: date,
                            proximity: dict, triggers_config: dict,
                            router_raw_data: dict, v16_regime: str,
                            g7_input: Optional[dict], fragility_state: str,
                            fragility_overlay: dict, config: dict) -> dict:
    """
    Build entry evaluation result.
    Entry only on monthly evaluation day, only from US_DOMESTIC.
    """
    is_eval_day = is_monthly_evaluation_day(today, router_state, config)

    entry_eval = {
        "is_evaluation_day": is_eval_day,
        "last_evaluation": router_state.get("last_entry_evaluation"),
        "next_evaluation": get_next_evaluation_day(today),
        "recommendation": None,
    }

    if not is_eval_day:
        return entry_eval

    if current_state != "US_DOMESTIC":
        entry_eval["recommendation"] = {
            "action": "SKIP",
            "reason": f"Router already in {current_state} — no new entry evaluation",
        }
        return entry_eval

    # Check each trigger
    for trigger_id, trigger_cfg in triggers_config.items():
        prox_data = proximity.get(trigger_id, {})
        if not prox_data.get("all_conditions_met", False):
            continue

        # Dual signal check
        dual = prox_data.get("dual_signal", {})
        if not (dual.get("fast_met", False) and dual.get("slow_met", False)):
            continue

        # G7 veto check (if required)
        if trigger_cfg.get("g7_veto_required", False):
            veto = _check_g7_veto_for_entry(trigger_cfg, g7_input)
            if veto is not None:
                entry_eval["recommendation"] = {
                    "action": "ENTRY_BLOCKED_BY_VETO",
                    "trigger": trigger_cfg.get("id", trigger_id.upper()),
                    "veto_reason": veto,
                }
                return entry_eval

        # All checks passed → entry recommendation
        allocation = _calculate_allocation(trigger_cfg, g7_input, fragility_state, fragility_overlay)
        confidence = _assess_entry_confidence(prox_data, fragility_state)

        entry_eval["recommendation"] = {
            "action": "ENTRY_RECOMMENDATION",
            "trigger": trigger_cfg.get("id", trigger_id.upper()),
            "trigger_name": trigger_cfg.get("name", trigger_id),
            "allocation": allocation,
            "confidence": confidence,
            "recommendation_text": _build_entry_text(trigger_cfg, allocation),
        }
        # Only one trigger can fire at a time
        break

    return entry_eval


def _check_g7_veto_for_entry(trigger_cfg: dict, g7_input: Optional[dict]) -> Optional[str]:
    """Check G7 veto for entry. V1: G7 not available → no veto (conservative pass)."""
    if g7_input is None:
        return None  # V1: No G7 data → assume no veto

    active_vetos = g7_input.get("active_vetos", [])
    if active_vetos:
        return f"G7 veto active: {', '.join(active_vetos)}"

    return None


def _calculate_allocation(trigger_cfg: dict, g7_input: Optional[dict],
                          fragility_state: str, fragility_overlay: dict) -> dict:
    """Calculate allocation for a triggered entry."""
    alloc_cfg = trigger_cfg.get("allocation", {})

    # Total percentage
    total_range = alloc_cfg.get("total_international_pct_range")
    if total_range:
        # Use lower end for initial recommendation
        total_pct = total_range[0]
    else:
        total_pct = alloc_cfg.get("default_total_pct", 0.15)

    # Distribution
    distribution = alloc_cfg.get("default_distribution", {})

    # G7 override (V1: not available)
    if g7_input is not None:
        g7_dist = g7_input.get("preferred_targets", {})
        override_key = alloc_cfg.get("g7_override_field", "")
        if override_key and "." in override_key:
            parts = override_key.split(".")
            g7_override = g7_dist
            for p in parts:
                g7_override = g7_override.get(p, {}) if isinstance(g7_override, dict) else {}
            if g7_override and isinstance(g7_override, (list, dict)):
                pass  # V1: Don't override defaults yet

    # Build distribution with actual weights
    actual_distribution = {}
    for etf, share in distribution.items():
        actual_distribution[etf] = round(total_pct * share, 4)

    return {
        "total_pct": total_pct,
        "distribution": actual_distribution,
        "source": "DEFAULT",
    }


def _assess_entry_confidence(prox_data: dict, fragility_state: str) -> str:
    """Assess confidence level of entry recommendation."""
    composite = prox_data.get("composite", 0.0)
    dual = prox_data.get("dual_signal", {})

    if composite >= 1.0 and dual.get("fast_met") and dual.get("slow_met"):
        if fragility_state == "HEALTHY":
            return "HIGH"
        else:
            return "MEDIUM"
    return "LOW"


def _build_entry_text(trigger_cfg: dict, allocation: dict) -> str:
    """Build human-readable entry recommendation text."""
    trigger_name = trigger_cfg.get("name", "Unknown Trigger")
    total_pct = allocation.get("total_pct", 0)
    dist = allocation.get("distribution", {})
    dist_str = ", ".join(f"{etf} ({pct:.0%})" for etf, pct in dist.items())
    return (
        f"{trigger_name} trigger fired. "
        f"Empfehlung: {total_pct:.0%} International — {dist_str}. "
        f"Umsetzung via Agent R mit Operator."
    )


# ============================================================
# FAST PATH OUTPUT
# ============================================================

def build_fast_path_router_output(router_state: dict, quick_proximity: float,
                                  fragility_state: str, today: date,
                                  config: dict) -> dict:
    """
    Minimal Router output for Fast Path.
    Spec Teil 4 §21.2
    """
    return {
        "current_state": "US_DOMESTIC",
        "state_since": router_state.get("state_since", "2025-01-01"),
        "days_in_state": days_in_state(router_state, today),
        "proximity": {
            "em_broad": {"composite": round(quick_proximity, 4), "all_conditions_met": False},
            "china_stimulus": {"composite": 0.0, "all_conditions_met": False},
            "commodity_super": {"composite": 0.0, "all_conditions_met": False},
        },
        "max_proximity": round(quick_proximity, 4),
        "max_proximity_trigger": "EM_BROAD" if quick_proximity > 0 else None,
        "entry_evaluation": {
            "is_evaluation_day": False,
            "last_evaluation": router_state.get("last_entry_evaluation"),
            "next_evaluation": get_next_evaluation_day(today),
            "recommendation": None,
        },
        "exit_check": None,
        "emergency": None,
        "crisis_override": False,
        "crisis_recommendation": None,
        "fragility_state_used": fragility_state,
        "thresholds_adjusted": False,
        "threshold_adjustment_note": None,
    }


# ============================================================
# HELPERS
# ============================================================

def _build_emergency_output(emergency: dict, router_state: dict, today: date,
                            fragility_state: str, config: dict) -> dict:
    """Build output dict for emergency exit case."""
    return {
        "current_state": "US_DOMESTIC",
        "state_since": today.isoformat(),
        "days_in_state": 0,
        "proximity": {
            "em_broad": {"composite": 0.0, "all_conditions_met": False},
            "china_stimulus": {"composite": 0.0, "all_conditions_met": False},
            "commodity_super": {"composite": 0.0, "all_conditions_met": False},
        },
        "max_proximity": 0.0,
        "max_proximity_trigger": None,
        "entry_evaluation": {
            "is_evaluation_day": False,
            "last_evaluation": router_state.get("last_entry_evaluation"),
            "next_evaluation": get_next_evaluation_day(today),
            "recommendation": None,
        },
        "exit_check": None,
        "emergency": emergency,
        "crisis_override": False,
        "crisis_recommendation": None,
        "fragility_state_used": fragility_state,
        "thresholds_adjusted": False,
        "threshold_adjustment_note": None,
    }


def _threshold_note(fragility_state: str) -> str:
    """Generate human-readable note about threshold adjustments."""
    if fragility_state == "ELEVATED":
        return "EM_BROAD: DXY threshold -3% (statt -5%), VWO/SPY threshold +5% (statt +10%)"
    elif fragility_state == "EXTREME":
        return "EM_BROAD: DXY threshold -3%, VWO/SPY +5%. EXTREME minimum international 5% empfohlen."
    elif fragility_state == "CRISIS":
        return "CRISIS: Router empfiehlt volle International-Allokation unabhaengig von Triggern."
    return ""
