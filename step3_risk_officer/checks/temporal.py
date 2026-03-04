"""
step_3_risk_officer/checks/temporal.py
Temporal Checks — zeitabhaengige Risiken.
Spec: Risk Officer Teil 2 §12
"""

from ..utils.helpers import make_alert
from ..utils.mappings import get_upcoming_events


# ═══════════════════════════════════════════════════════════════════
# TMP_EVENT_CALENDAR — Makro-Event Severity Boost
# Spec Teil 2 §12.3
# ═══════════════════════════════════════════════════════════════════

def check_event_calendar(reference_date, config):
    """
    Prueft ob ein wichtiges Makro-Event innerhalb der naechsten 48h liegt.
    Gibt eigenen Alert + Event-Context fuer Severity-Boosts zurueck.

    Returns: (alert_or_None, event_context_dict)
    """
    if not config.get("enabled", True):
        return None, {"event_in_48h": False}

    horizon_days = config.get("boost_horizon_days", 2)
    upcoming = get_upcoming_events(reference_date)

    imminent_events = [
        evt for evt in upcoming
        if evt["days_until"] <= horizon_days
    ]

    event_context = {
        "event_in_48h": len(imminent_events) > 0,
        "imminent_events": imminent_events,
        "next_event_name": upcoming[0]["event"] if upcoming else None,
        "next_event_days": upcoming[0]["days_until"] if upcoming else None
    }

    if not imminent_events:
        return None, event_context

    event_names = ", ".join(
        f"{e['event']} in {e['days_until']}d ({e['date']})"
        for e in imminent_events
    )

    alert = make_alert(
        severity="MONITOR",
        message=(
            f"Upcoming macro event(s): {event_names}. "
            f"Increased uncertainty may affect existing risk assessments."
        ),
        check_id="TMP_EVENT_CALENDAR",
        recommendation=(
            "Macro event approaching. Existing risk assessments carry "
            "elevated uncertainty. No preemptive action recommended."
        )
    )

    return alert, event_context


# ═══════════════════════════════════════════════════════════════════
# TMP_CC_EXPIRY — F6 Covered Call Verfall
# Spec Teil 2 §12.1
# V1: DEAKTIVIERT (F6 nicht live)
# ═══════════════════════════════════════════════════════════════════

def check_cc_expiry(f6_positions, config):
    """V1: Deaktiviert."""
    if not config.get("enabled", False):
        return None
    return None


# ═══════════════════════════════════════════════════════════════════
# TMP_V16_REBALANCE — V16 Rebalance Proximity
# Spec Teil 2 §12.2
# V1: DEAKTIVIERT (next_rebalance_expected nicht im V16 Output)
# ═══════════════════════════════════════════════════════════════════

def check_v16_rebalance(v16_state, config):
    """V1: Deaktiviert."""
    if not config.get("enabled", False):
        return None
    return None