"""
step2_signal_generator/outcome_tracker.py
Outcome Tracker — Router Recommendation Lifecycle Tracking

Tracks: PENDING → EXECUTED/REJECTED/EXPIRED → ACTIVE → CLOSED
Source: Signal Generator Spec Teil 3 §19
"""

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger("signal_generator.outcome_tracker")


def manage_outcome_tracker(tracker_entries: list, router_output: dict,
                           today: date, config: dict) -> list:
    """
    Daily update of outcome tracker entries.
    Spec Teil 3 §19.3

    - ACTIVE entries: check for exit signal → CLOSED
    - PENDING entries: check if trigger still active, expire after N days
    """
    timing = config.get("router_timing", {}).get("outcome_tracker", {})
    expiry_days = timing.get("pending_expiry_days", 5)

    for entry in tracker_entries:
        status = entry.get("status", "")

        if status == "ACTIVE":
            # Check if exit signal for this recommendation
            exit_check = router_output.get("exit_check")
            emergency = router_output.get("emergency")

            if emergency is not None and emergency.get("action") == "EMERGENCY_EXIT":
                entry["status"] = "CLOSED"
                entry["exit_date"] = today.isoformat()
                entry["exit_reason"] = emergency.get("reason", "Emergency exit")
                logger.info(f"Outcome {entry.get('signal_id')}: ACTIVE → CLOSED (emergency)")

            elif exit_check is not None and exit_check.get("exit_triggered", False):
                entry["status"] = "CLOSED"
                entry["exit_date"] = today.isoformat()
                entry["exit_reason"] = exit_check.get("reason", "Exit signal")
                logger.info(f"Outcome {entry.get('signal_id')}: ACTIVE → CLOSED (exit signal)")

        elif status == "PENDING":
            # Check if trigger still active
            trigger_id = _extract_trigger_id(entry.get("signal_id", ""))
            if trigger_id:
                prox = router_output.get("proximity", {}).get(trigger_id, {})
                still_met = prox.get("all_conditions_met", False)
            else:
                still_met = False

            # Expire after N days if trigger no longer active
            if not still_met:
                issued_date = _parse_date(entry.get("date_issued"))
                if issued_date is not None:
                    days_since = (today - issued_date).days
                    if days_since > expiry_days:
                        entry["status"] = "EXPIRED"
                        entry["exit_date"] = today.isoformat()
                        entry["exit_reason"] = f"Trigger conditions no longer met after {days_since} days"
                        logger.info(f"Outcome {entry.get('signal_id')}: PENDING → EXPIRED ({days_since}d)")

    return tracker_entries


def create_outcome_entry(entry_recommendation: dict, today: date) -> dict:
    """
    Create a new outcome tracker entry from an entry recommendation.
    Spec Teil 3 §19.2
    """
    trigger = entry_recommendation.get("trigger", "UNKNOWN")
    month_str = today.strftime("%Y_%m")
    signal_id = f"ROUTER_{trigger}_{month_str}"

    allocation = entry_recommendation.get("allocation", {})
    dist = allocation.get("distribution", {})
    dist_str = ", ".join(f"{etf} ({pct:.0%})" for etf, pct in dist.items())

    return {
        "signal_id": signal_id,
        "type": "ROUTER",
        "date_issued": today.isoformat(),
        "recommendation": f"{allocation.get('total_pct', 0):.0%} International — {dist_str}. {trigger} trigger fired.",
        "trigger_details": {
            "trigger": trigger,
            "allocation": allocation,
            "confidence": entry_recommendation.get("confidence", "UNKNOWN"),
        },
        "status": "PENDING",
        "execution_date": None,
        "execution_details": None,
        "rejection_reason": None,
        "exit_date": None,
        "exit_reason": None,
        "review_date": (today + timedelta(days=90)).isoformat(),
        "outcome": None,
        "outcome_vs_v16": None,
        "notes": None,
    }


def _extract_trigger_id(signal_id: str) -> Optional[str]:
    """Extract trigger ID from signal_id. E.g. ROUTER_EM_BROAD_2026_03 → em_broad."""
    if not signal_id:
        return None
    parts = signal_id.split("_")
    if len(parts) >= 3 and parts[0] == "ROUTER":
        # Reconstruct trigger: ROUTER_EM_BROAD_2026_03 → EM_BROAD
        # Find where the date part starts (4-digit year)
        trigger_parts = []
        for i, p in enumerate(parts[1:], 1):
            if len(p) == 4 and p.isdigit():
                break
            trigger_parts.append(p)
        if trigger_parts:
            return "_".join(trigger_parts).lower()
    return None


def _parse_date(date_str: str) -> Optional[date]:
    """Parse ISO date string."""
    if date_str is None:
        return None
    try:
        return date.fromisoformat(str(date_str)[:10])
    except (ValueError, TypeError):
        return None
