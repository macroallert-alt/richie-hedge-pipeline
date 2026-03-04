"""
step_3_risk_officer/portfolio/status.py
Portfolio Status Ampel (GREEN/YELLOW/RED/BLACK).
Spec: Risk Officer Teil 3 §19
"""


def determine_portfolio_status(active_alerts, emergency_triggers, config=None):
    """
    Bestimmt Portfolio Status Ampel.

    BLACK:  Emergency Trigger aktiv
    RED:    Mindestens 1 CRITICAL
    YELLOW: Mindestens 1 WARNING ODER 3+ MONITORs
    GREEN:  Alles ok
    """
    yellow_monitor_threshold = 3
    if config:
        yellow_monitor_threshold = config.get("portfolio_status", {}).get(
            "yellow_monitor_threshold", 3
        )

    # BLACK: Emergency Trigger aktiv
    if emergency_triggers:
        trigger_ids = [e.get("check_id", "?") for e in emergency_triggers]
        return {
            "status": "BLACK",
            "reason": (
                f"{len(emergency_triggers)} EMERGENCY trigger(s) active: "
                f"{', '.join(trigger_ids)}"
            ),
            "severity_counts": _count_severities(active_alerts)
        }

    # Zaehle aktive Alerts (ohne RESOLVED)
    non_resolved = [
        a for a in (active_alerts or [])
        if a.get("severity") not in ("RESOLVED",)
    ]

    counts = _count_severities(non_resolved)

    # RED: Mindestens 1 CRITICAL
    if counts["CRITICAL"] > 0:
        return {
            "status": "RED",
            "reason": f"{counts['CRITICAL']} CRITICAL alert(s). Immediate review recommended.",
            "severity_counts": counts
        }

    # YELLOW: Mindestens 1 WARNING ODER 3+ MONITORs
    if counts["WARNING"] > 0 or counts["MONITOR"] >= yellow_monitor_threshold:
        reason_parts = []
        if counts["WARNING"] > 0:
            reason_parts.append(f"{counts['WARNING']} WARNING(s)")
        if counts["MONITOR"] >= yellow_monitor_threshold:
            reason_parts.append(f"{counts['MONITOR']} MONITORs (≥{yellow_monitor_threshold})")
        return {
            "status": "YELLOW",
            "reason": f"{' and '.join(reason_parts)}. Review recommended.",
            "severity_counts": counts
        }

    # GREEN
    monitor_note = ""
    if counts["MONITOR"] > 0:
        monitor_note = f" {counts['MONITOR']} MONITOR(s) noted."
    return {
        "status": "GREEN",
        "reason": f"All limits within bounds.{monitor_note}",
        "severity_counts": counts
    }


def _count_severities(alerts):
    """Zaehlt Alerts nach Severity."""
    counts = {"EMERGENCY": 0, "CRITICAL": 0, "WARNING": 0, "MONITOR": 0}
    for a in (alerts or []):
        sev = a.get("severity", "MONITOR")
        if sev in counts:
            counts[sev] += 1
    return counts