"""
step_3_risk_officer/portfolio/fast_path.py
Fast Path Evaluation — wenn nichts passiert ist.
Spec: Risk Officer Teil 2 §14
"""


def evaluate_fast_path(v16_state, fragility_state, risk_history,
                        event_context, config):
    """
    ALLE Bedingungen muessen True sein fuer Fast Path.
    Wenn EINE False ist → Full Path.

    Spec Teil 2 §14.1
    """
    if not config.get("fast_path", {}).get("enabled", True):
        return {"fast_path": False, "reason": "FAST_PATH_DISABLED"}

    yesterday = risk_history or {}

    conditions = {
        "v16_state_unchanged": (
            v16_state.get("v16_state") ==
            yesterday.get("v16_state")
        ),
        "fragility_healthy_or_unchanged": (
            fragility_state == "HEALTHY" or
            fragility_state == yesterday.get("fragility_state")
        ),
        "no_alerts_escalated": (
            not any(
                a.get("trend") == "ESCALATING"
                for a in yesterday.get("alerts", [])
            )
        ),
        "no_event_in_48h": (
            not event_context.get("event_in_48h", False)
        ),
        "yesterday_was_green": (
            yesterday.get("portfolio_status") == "GREEN"
        ),
        "has_history": (
            bool(risk_history)
        )
    }

    fast_path = all(conditions.values())
    failed = [k for k, v in conditions.items() if not v]

    return {
        "fast_path": fast_path,
        "conditions": conditions,
        "failed_conditions": failed,
        "reason": "ALL_MET" if fast_path else f"FAILED: {', '.join(failed)}"
    }


def weights_changed_significantly(v16_weights, yesterday_weights, threshold=0.02):
    """
    Prueft ob sich V16-Gewichte seit gestern um mehr als threshold geaendert haben.
    Fuer Fast Path Quick Exposure Check.
    """
    if not yesterday_weights:
        return True  # Kein Vergleich moeglich → Full Path

    all_assets = set(v16_weights.keys()) | set(yesterday_weights.keys())
    for asset in all_assets:
        curr = v16_weights.get(asset, 0.0)
        prev = yesterday_weights.get(asset, 0.0)
        if abs(curr - prev) > threshold:
            return True

    return False