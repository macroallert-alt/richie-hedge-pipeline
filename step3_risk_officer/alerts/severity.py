"""
step_3_risk_officer/alerts/severity.py
Severity-Bestimmung mit Boost-Logik.
Spec: Risk Officer Teil 3 §17
"""

from ..utils.helpers import SEVERITY_LADDER


def determine_severity(alert, fragility_state, event_context, g7_status, config):
    """
    Bestimmt finale Severity eines Alerts basierend auf:
    1. Basis-Severity (vom Check selbst)
    2. Kontext-Boosts (Fragility, Events, G7)

    REGELN:
    - Maximal EIN Boost pro Alert (der hoechste gewinnt)
    - Kein Boost kann zu EMERGENCY fuehren (nur explizite Triggers)
    - Boost-Grund wird dokumentiert

    Returns: dict mit severity, base_severity, boost_applied, boost_amount
    """
    base = alert.get("severity", "MONITOR")

    # EMERGENCY wird nie geboosted — direkt zurueck
    if base == "EMERGENCY":
        return {
            "severity": "EMERGENCY",
            "base_severity": "EMERGENCY",
            "boost_applied": None,
            "boost_amount": 0
        }

    boost_rules = config.get("severity", {}).get("boost_rules", {})

    # Moegliche Boosts sammeln
    potential_boosts = []

    # Fragility Boost
    if fragility_state == "CRISIS":
        amount = boost_rules.get("FRAGILITY_CRISIS", 2)
        potential_boosts.append(("FRAGILITY_CRISIS", amount))
    elif fragility_state == "EXTREME":
        amount = boost_rules.get("FRAGILITY_EXTREME", 1)
        potential_boosts.append(("FRAGILITY_EXTREME", amount))
    # ELEVATED: kein automatischer Boost (nur im Kontext erwaehnt)
    # HEALTHY: kein Boost

    # Event Boost
    if event_context.get("event_in_48h", False):
        amount = boost_rules.get("EVENT_IMMINENT", 1)
        potential_boosts.append(("EVENT_IMMINENT", amount))

    # G7 Boost
    if g7_status == "STRUCTURAL_BREAK":
        amount = boost_rules.get("G7_STRUCTURAL_BREAK", 2)
        potential_boosts.append(("G7_STRUCTURAL_BREAK", amount))
    elif g7_status == "ELEVATED_RISK":
        amount = boost_rules.get("G7_ELEVATED_RISK", 1)
        potential_boosts.append(("G7_ELEVATED_RISK", amount))

    # Hoechsten Boost waehlen
    if not potential_boosts:
        return {
            "severity": base,
            "base_severity": base,
            "boost_applied": None,
            "boost_amount": 0
        }

    potential_boosts.sort(key=lambda x: x[1], reverse=True)
    boost_reason, boost_amount = potential_boosts[0]

    # Boost anwenden
    if base in SEVERITY_LADDER:
        base_idx = SEVERITY_LADDER.index(base)
        boosted_idx = min(base_idx + boost_amount, len(SEVERITY_LADDER) - 1)
        boosted_severity = SEVERITY_LADDER[boosted_idx]
    else:
        boosted_severity = base

    return {
        "severity": boosted_severity,
        "base_severity": base,
        "boost_applied": boost_reason,
        "boost_amount": boost_amount
    }


def enrich_alert_context(alert, fragility_state, event_context, g7_status,
                          v16_state):
    """
    Fuegt Kontext-Informationen zum Alert hinzu.
    Spec Teil 3 §21.1
    """
    alert["context"] = {
        "fragility_state": fragility_state,
        "event_in_48h": event_context.get("event_in_48h", False),
        "next_event": event_context.get("next_event_name"),
        "next_event_days": event_context.get("next_event_days"),
        "g7_status": g7_status,
        "v16_state": v16_state.get("v16_state", "UNKNOWN") if v16_state else "UNAVAILABLE",
        "dd_protect_active": v16_state.get("dd_protect_active", False) if v16_state else False
    }
    return alert