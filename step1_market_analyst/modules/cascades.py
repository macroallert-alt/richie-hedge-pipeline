"""
Market Analyst — Cascade Detection Module
Causal chains between layers with known time lags.

Checks: Did Source layer trigger X weeks ago?
        Has Target layer ALREADY reacted?
        If not: EXPECTED cascade — early warning.

Statuses:
  EXPECTED   — Trigger fired, target hasn't reacted yet (early warning)
  PROPAGATED — Target reacted as expected (confirmation)

Source: AGENT2_SPEC_TEIL5 Section 16
"""

import re
from datetime import datetime, timedelta


def check_cascades(
    layer_results: dict,
    history_30d: list,
    cascade_config: list,
    today: datetime = None,
) -> list:
    """
    For each causal chain: did source trigger within the lag window?
    If yes, has target already reacted?

    layer_results: current layer results
    history_30d: list of daily records
    cascade_config: list from cascades.json
    today: current date (for testing)

    Returns: list of active cascades
    """
    if today is None:
        today = datetime.utcnow().date()
    elif isinstance(today, datetime):
        today = today.date()

    active_cascades = []

    for cascade in cascade_config:
        source_layer = cascade["source"]
        target_layer = cascade["target"]
        lag_min_days = cascade["lag_weeks_min"] * 7
        lag_max_days = cascade["lag_weeks_max"] * 7

        # Check if source triggered within the lag window
        triggered_date = _find_trigger_in_window(
            history_30d, source_layer, cascade["source_trigger"],
            lag_min_days, lag_max_days,
        )

        if triggered_date is None:
            continue

        # Source triggered. Has target already reacted?
        target_current = layer_results.get(target_layer, {})
        target_has_reacted = _evaluate_expected_effect(
            target_current, cascade["expected_effect"]
        )

        status = "PROPAGATED" if target_has_reacted else "EXPECTED"

        # Calculate timing
        triggered_dt = _parse_date(triggered_date)
        if triggered_dt:
            days_since = (today - triggered_dt).days
            days_remaining = lag_max_days - days_since
        else:
            days_since = 0
            days_remaining = lag_max_days

        active_cascades.append({
            "cascade_id": cascade["id"],
            "name": cascade["name"],
            "source_layer": source_layer,
            "target_layer": target_layer,
            "triggered_date": triggered_date,
            "lag_window": f"{cascade['lag_weeks_min']}-{cascade['lag_weeks_max']} weeks",
            "status": status,
            "description": cascade["description"],
            "days_since_trigger": days_since,
            "days_remaining_in_window": max(0, days_remaining),
        })

    return active_cascades


def _find_trigger_in_window(
    history_30d: list,
    source_layer: str,
    trigger_condition: str,
    lag_min_days: int,
    lag_max_days: int,
) -> str:
    """
    Searches history for the source trigger within the lag window.
    Returns the date string if found, None otherwise.
    """
    if not history_30d:
        return None

    for i, day in enumerate(history_30d):
        days_ago = len(history_30d) - 1 - i

        if lag_min_days <= days_ago <= lag_max_days:
            source_data = day.get("layers", {}).get(source_layer, {})
            if _evaluate_trigger(source_data, trigger_condition):
                return day.get("date")

    return None


def _evaluate_trigger(source_data: dict, trigger_str: str) -> bool:
    """
    Evaluates trigger condition against source layer data.
    Supports: "regime == 'TIGHTENING'", "regime == 'TIGHTENING' OR regime == 'DRAIN'"
    """
    if " OR " in trigger_str:
        parts = trigger_str.split(" OR ")
        return any(_eval_single_trigger(source_data, p.strip()) for p in parts)
    return _eval_single_trigger(source_data, trigger_str)


def _eval_single_trigger(data: dict, condition: str) -> bool:
    """Evaluates a single trigger condition."""
    match = re.match(r"(\w+)\s*(==|!=|<|>|<=|>=)\s*(.+)", condition.strip())
    if not match:
        return False

    field, op, value_str = match.groups()
    value_str = value_str.strip().strip("'\"")

    actual = data.get(field)
    if actual is None:
        return False

    # Try numeric
    try:
        expected = float(value_str)
        actual_num = float(actual)
        if op == "==":
            return actual_num == expected
        elif op == "<":
            return actual_num < expected
        elif op == ">":
            return actual_num > expected
    except (ValueError, TypeError):
        pass

    # String comparison
    if op == "==":
        return str(actual) == value_str
    elif op == "!=":
        return str(actual) != value_str

    return False


def _evaluate_expected_effect(target_data: dict, expected_effect: str) -> bool:
    """
    Checks if target layer has reacted as expected.
    Supports: "score decreases", "score increases",
              "regime moves to ELEVATED or worse"
    """
    if "score decreases" in expected_effect:
        # Check if score is negative (indicating it has moved down)
        score = target_data.get("score", 0)
        direction = target_data.get("direction", "STABLE")
        return score < -2 or direction == "DETERIORATING"

    elif "score increases" in expected_effect:
        score = target_data.get("score", 0)
        direction = target_data.get("direction", "STABLE")
        return score > 2 or direction == "IMPROVING"

    elif "regime moves to" in expected_effect:
        # Extract target regime(s) from string
        regime = target_data.get("regime", "")
        # Simple check: does the current regime appear in the expected effect?
        return regime in expected_effect

    return False


def _parse_date(date_str: str):
    """Parses date string to date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
