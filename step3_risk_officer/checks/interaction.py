"""
step_3_risk_officer/checks/interaction.py
Interaction Checks — Wechselwirkungen zwischen Subsystemen.
Spec: Risk Officer Teil 2 §11
"""

from ..utils.helpers import make_alert


# ═══════════════════════════════════════════════════════════════════
# INT_REGIME_CONFLICT — V16 vs Market Analyst Regime-Widerspruch
# Spec Teil 2 §11.1
# ═══════════════════════════════════════════════════════════════════

def check_regime_conflict(v16_state, market_analyst_regime, config):
    """
    V16 Regime vs Market Analyst System Regime.

    WICHTIG: V16 hat RECHT (validiert). Dieser Check stellt V16 nicht
    in Frage, sondern informiert den CIO ueber die Divergenz.

    Returns: alert_or_None
    """
    if not config.get("enabled", True):
        return None

    compatible = config.get("compatible_mappings", {
        "Risk-On": ["BROAD_RISK_ON", "SELECTIVE"],
        "Risk-Off": ["BROAD_RISK_OFF", "RISK_OFF_FORCED", "CONFLICTED"],
        "DD-Protect": ["BROAD_RISK_OFF", "RISK_OFF_FORCED"]
    })

    v16 = v16_state.get("v16_state", "Risk-On")
    v16_regime = v16_state.get("v16_regime", "UNKNOWN")
    ma = market_analyst_regime.get("regime", "UNKNOWN")
    ma_lean = market_analyst_regime.get("lean", "UNKNOWN")

    if ma in compatible.get(v16, []):
        return None  # Kein Conflict

    # Conflict erkannt — Severity bestimmen
    severe_conflicts = {
        ("Risk-On", "BROAD_RISK_OFF"),
        ("Risk-On", "RISK_OFF_FORCED"),
        ("Risk-Off", "BROAD_RISK_ON")
    }

    if (v16, ma) in severe_conflicts:
        base_severity = "WARNING"
        recommendation = (
            f"Significant regime divergence. V16 may transition soon. "
            f"Prepare for potential large portfolio reallocation. "
            f"Do NOT preempt V16 — let V16's validated signals drive timing."
        )
    else:
        base_severity = "MONITOR"
        recommendation = (
            f"V16 and Market Analyst slightly divergent. V16 validated — "
            f"no action on V16 required. Monitor for V16 regime transition."
        )

    return make_alert(
        severity=base_severity,
        message=(
            f"V16 state '{v16}' (regime: {v16_regime}) diverges from "
            f"Market Analyst assessment '{ma}' (lean: {ma_lean}). "
            f"V16 operates on validated signals — this divergence may indicate "
            f"V16 will transition soon. No action required on V16."
        ),
        check_id="INT_REGIME_CONFLICT",
        affected_systems=["V16", "MARKET_ANALYST"],
        trade_class="A",
        recommendation=recommendation
    )


# ═══════════════════════════════════════════════════════════════════
# INT_RECOMMENDATION_OVERLOAD — Zu viele Empfehlungen gleichzeitig
# Spec Teil 2 §11.2
# V1: DEAKTIVIERT (Signal Generator nicht live)
# ═══════════════════════════════════════════════════════════════════

def check_recommendation_overload(signal_generator_output, config):
    """
    Zaehlt aktive unvalidierte Empfehlungen.
    V1: Deaktiviert, gibt immer None zurueck.
    """
    if not config.get("enabled", False):
        return None

    # Placeholder fuer spaeter — wenn Signal Generator live ist
    return None


# ═══════════════════════════════════════════════════════════════════
# INT_CORRELATION_SHOCK — Diversifikations-Illusion
# Spec Teil 2 §11.3
# V1: DEAKTIVIERT (Return-Zeitreihen nicht im DW)
# ═══════════════════════════════════════════════════════════════════

def check_correlation_shock(position_returns, portfolio_weights, config):
    """
    Berechnet paarweise Korrelationen der Top-N Positionen.
    V1: Deaktiviert, gibt immer None zurueck.
    """
    if not config.get("enabled", False):
        return None

    # Placeholder fuer spaeter
    return None