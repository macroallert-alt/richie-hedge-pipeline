"""
step2_signal_generator/router_state.py
Router State Machine — State Management, History, Evaluation Day Check

Source: Signal Generator Spec Teil 2 §7, Teil 4 §21.3-21.5
"""

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger("signal_generator.router_state")


def default_router_state() -> dict:
    """
    Initial Router State when no history exists (first run).
    Spec Teil 4 §21.3
    """
    return {
        "current_state": "US_DOMESTIC",
        "state_since": "2025-01-01",
        "last_entry_evaluation": None,
        "last_exit_check": None,
        "trigger_proximity_yesterday": {
            "em_broad": 0.0,
            "china_stimulus": 0.0,
            "commodity_super": 0.0,
        },
        "history_30d": [],
    }


def is_monthly_evaluation_day(today: date, router_state: dict, config: dict) -> bool:
    """
    Entry-Evaluation: monthly, 1st-3rd business day of month.
    Spec Teil 4 §21.4

    Rules:
      - today.day <= entry.evaluation_day_max (default 3)
      - today is weekday (Mon-Fri)
      - Not already evaluated this month
    """
    timing = config.get("router_timing", {}).get("entry", {})
    max_day = timing.get("evaluation_day_max", 3)

    if today.day > max_day:
        return False

    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    last_eval = router_state.get("last_entry_evaluation")
    if last_eval is not None:
        last_eval_date = _parse_date(last_eval)
        if last_eval_date is not None:
            if (today.year, today.month) == (last_eval_date.year, last_eval_date.month):
                return False

    return True


def get_next_evaluation_day(today: date) -> str:
    """
    Estimate next monthly evaluation day (1st of next month, or next business day).
    """
    if today.month == 12:
        next_first = date(today.year + 1, 1, 1)
    else:
        next_first = date(today.year, today.month + 1, 1)

    # Adjust to weekday
    while next_first.weekday() >= 5:
        next_first += timedelta(days=1)

    return next_first.isoformat()


def check_emergency_exit(v16_regime: str, router_state: dict, config: dict) -> Optional[dict]:
    """
    Emergency Exit: V16 in crisis regime → immediate reset to US_DOMESTIC.
    Spec Teil 2 §7.3

    Returns emergency dict or None.
    """
    sm_config = config.get("router_state_machine", {})
    emergency_regimes = sm_config.get("emergency_regimes", ["FINANCIAL_CRISIS", "DEEP_CONTRACTION"])

    current_state = router_state.get("current_state", "US_DOMESTIC")

    if v16_regime in emergency_regimes and current_state != "US_DOMESTIC":
        return {
            "action": "EMERGENCY_EXIT",
            "reason": f"V16 regime changed to {v16_regime} — immediate router reset to US_DOMESTIC",
            "previous_state": current_state,
            "new_state": "US_DOMESTIC",
            "recommendation": "Exit all international positions immediately. Discuss with Agent R.",
        }

    return None


def validate_state_transition(current_state: str, target_state: str, config: dict) -> bool:
    """
    Validate that a state transition is allowed.
    Spec Teil 2 §7.2: No direct transitions between international states.
    Must go through US_DOMESTIC.
    """
    sm_config = config.get("router_state_machine", {})
    states = sm_config.get("states", {})

    state_def = states.get(current_state, {})
    allowed = state_def.get("transitions_to", [])
    blocked = state_def.get("cannot_transition_to", [])

    if target_state in blocked:
        return False

    if target_state not in allowed:
        return False

    return True


def update_router_state_history(router_state: dict, router_output: dict, today: date) -> dict:
    """
    Update Router State and 30-day history after engine run.
    Spec Teil 4 §21.5

    Updates:
      - current_state (if emergency exit)
      - state_since (if state changed)
      - last_entry_evaluation (if evaluation day)
      - last_exit_check (if router active)
      - history_30d (append today, trim to 30)
      - trigger_proximity_yesterday
    """
    new_state = router_state.get("current_state", "US_DOMESTIC")

    # Emergency exit changes state
    emergency = router_output.get("emergency")
    if emergency is not None and emergency.get("action") == "EMERGENCY_EXIT":
        new_state = "US_DOMESTIC"
        router_state["state_since"] = today.isoformat()

    # Update last_entry_evaluation if evaluation day
    entry_eval = router_output.get("entry_evaluation", {})
    if entry_eval.get("is_evaluation_day", False):
        router_state["last_entry_evaluation"] = today.isoformat()

    # Update last_exit_check if router was active
    if router_state.get("current_state", "US_DOMESTIC") != "US_DOMESTIC":
        router_state["last_exit_check"] = today.isoformat()

    # Build today's history record
    today_proximity = {}
    for trigger_id, prox_data in router_output.get("proximity", {}).items():
        if isinstance(prox_data, dict):
            today_proximity[trigger_id] = prox_data.get("composite", 0.0)
        else:
            today_proximity[trigger_id] = 0.0

    today_record = {
        "date": today.isoformat(),
        "state": new_state,
        "proximity": today_proximity,
    }

    # Append to history, trim to 30
    history = router_state.get("history_30d", [])
    history.append(today_record)
    if len(history) > 30:
        history = history[-30:]

    # Write back
    router_state["current_state"] = new_state
    router_state["history_30d"] = history
    router_state["trigger_proximity_yesterday"] = today_proximity

    return router_state


def days_in_state(router_state: dict, today: date) -> int:
    """Calculate how many days the router has been in current state."""
    state_since = router_state.get("state_since")
    if state_since is None:
        return 0
    since_date = _parse_date(state_since)
    if since_date is None:
        return 0
    return (today - since_date).days


def _parse_date(date_str: str) -> Optional[date]:
    """Parse ISO date string to date object."""
    if date_str is None:
        return None
    try:
        return date.fromisoformat(str(date_str)[:10])
    except (ValueError, TypeError):
        return None
