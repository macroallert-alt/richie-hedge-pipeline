"""
step2_signal_generator/recommendations.py
Recommendations Builder — Summary for CIO

V1 Scope: Router + Concentration warnings. F6 Advisory, PermOpt, Fragility stubs.
Source: Signal Generator Spec Teil 3 §18
"""

import logging

logger = logging.getLogger("signal_generator.recommendations")


def build_recommendations(router_output: dict, concentration: dict,
                          config: dict) -> dict:
    """
    Build recommendations output block.
    Spec Teil 3 §18.1

    Returns structured recommendations for CIO, Risk Officer, Agent R.
    """
    timing_cfg = config.get("router_timing", {}).get("proximity", {})

    # Router recommendations
    router_rec = _build_router_recommendations(router_output, timing_cfg)

    # Concentration warning
    concentration_warning = _build_concentration_warning(concentration)

    # Has anything actionable?
    has_actionable = (
        router_rec.get("has_recommendation", False)
        or concentration_warning.get("has_warning", False)
    )

    # Summary for CIO
    summary = _build_summary_for_cio(router_rec, concentration_warning, router_output)

    return {
        "date": router_output.get("date"),
        "has_actionable_recommendations": has_actionable,
        "router": router_rec,
        "perm_opt": {
            "has_recommendation": False,
            "status": "UNAVAILABLE",
            "note": "PermOpt available in V2 (after G7 Monitor)",
        },
        "fragility": {
            "has_recommendation": False,
            "status": "UNAVAILABLE",
            "note": "Fragility recommendations available in V2",
        },
        "concentration_warning": concentration_warning,
        "f6_advisory": {
            "has_advisory": False,
            "status": "UNAVAILABLE",
            "note": "F6 Advisory available in V2 (after F6 live)",
        },
        "summary_for_cio": summary,
    }


def _build_router_recommendations(router_output: dict, timing_cfg: dict) -> dict:
    """Build Router-specific recommendations."""
    entry_eval = router_output.get("entry_evaluation", {})
    exit_check = router_output.get("exit_check")
    emergency = router_output.get("emergency")
    crisis = router_output.get("crisis_override", False)
    max_prox = router_output.get("max_proximity", 0.0)
    max_prox_trigger = router_output.get("max_proximity_trigger")

    has_rec = False
    entry = None
    exit_rec = None
    proximity_note = None

    # Emergency
    if emergency is not None:
        has_rec = True
        exit_rec = {
            "type": "EMERGENCY_EXIT",
            "reason": emergency.get("reason", ""),
            "recommendation": emergency.get("recommendation", ""),
        }

    # Crisis override
    elif crisis:
        has_rec = True
        entry = {
            "type": "CRISIS_OVERRIDE",
            "recommendation": router_output.get("crisis_recommendation", ""),
        }

    # Exit signal
    elif exit_check is not None and exit_check.get("exit_triggered", False):
        has_rec = True
        exit_rec = {
            "type": "EXIT_SIGNAL",
            "exit_id": exit_check.get("exit_id"),
            "reason": exit_check.get("reason", ""),
            "severity": exit_check.get("severity", "EXIT"),
        }

    # Entry recommendation
    elif (entry_eval.get("recommendation") or {}).get("action") == "ENTRY_RECOMMENDATION":
        has_rec = True
        rec = entry_eval["recommendation"]
        entry = {
            "type": "ENTRY_RECOMMENDATION",
            "trigger": rec.get("trigger"),
            "trigger_name": rec.get("trigger_name"),
            "allocation": rec.get("allocation"),
            "confidence": rec.get("confidence"),
            "recommendation_text": rec.get("recommendation_text"),
        }

    # Proximity note
    cio_threshold = timing_cfg.get("cio_mention_threshold", 0.5)
    if max_prox >= cio_threshold and max_prox_trigger:
        proximity_note = (
            f"{max_prox_trigger} proximity at {max_prox:.0%}. "
            f"{'Approaching trigger.' if max_prox >= 0.7 else 'Monitor closely.'}"
        )
    elif max_prox > 0:
        proximity_note = f"Max proximity: {max_prox:.0%} ({max_prox_trigger or 'none'}). No action needed."

    return {
        "has_recommendation": has_rec,
        "entry": entry,
        "exit": exit_rec,
        "proximity_note": proximity_note,
        "next_evaluation": entry_eval.get("next_evaluation"),
    }


def _build_concentration_warning(concentration: dict) -> dict:
    """Build concentration warning from projection."""
    if not concentration:
        return {"has_warning": False}

    return {
        "has_warning": concentration.get("warning", False),
        "effective_tech_pct": concentration.get("effective_tech_pct"),
        "message": concentration.get("warning_message"),
        "top5_assets": concentration.get("top5_assets", []),
        "top5_concentration_pct": concentration.get("top5_concentration_pct"),
        "action_type": "REVIEW" if concentration.get("warning") else None,
    }


def _build_summary_for_cio(router_rec: dict, concentration_warning: dict,
                           router_output: dict) -> str:
    """Build human-readable summary for CIO briefing."""
    parts = []

    # Router
    if router_rec.get("entry") is not None:
        entry = router_rec["entry"]
        if entry.get("type") == "ENTRY_RECOMMENDATION":
            parts.append(f"Router {entry.get('trigger', '')} Entry-Empfehlung aktiv")
        elif entry.get("type") == "CRISIS_OVERRIDE":
            parts.append("CRISIS: Router empfiehlt volle International-Allokation")

    if router_rec.get("exit") is not None:
        exit_r = router_rec["exit"]
        if exit_r.get("type") == "EMERGENCY_EXIT":
            parts.append(f"EMERGENCY EXIT: {exit_r.get('reason', '')}")
        else:
            parts.append(f"Router Exit-Signal: {exit_r.get('exit_id', '')}")

    if router_rec.get("proximity_note"):
        parts.append(f"Router: {router_rec['proximity_note']}")

    # Concentration
    if concentration_warning.get("has_warning"):
        tech = concentration_warning.get("effective_tech_pct", 0)
        parts.append(f"Konzentrations-Warnung: Tech bei {tech:.1%}")

    # V1 stubs
    parts.append("F6/PermOpt/Fragility: UNAVAILABLE (V2)")

    if not parts:
        parts.append("Keine Empfehlungen. Router inaktiv, V16 unveraendert.")

    return " | ".join(parts)
