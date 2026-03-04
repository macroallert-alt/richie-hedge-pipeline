"""
Market Analyst — Fragility Monitor Module
Measures US market concentration risk (Mag7 dependency).

4 indicators → 4 states → structured recommendations.
Shadow mode: informational only, no automatic weight changes.

States:
  HEALTHY  — No fragility concerns
  ELEVATED — Concentration elevated, increase attention
  EXTREME  — Concentration extreme, de-escalation recommended
  CRISIS   — Mag7 concentration breaking down (qualitatively different)

Source: AGENT2_SPEC_ADDENDUM_FRAGILITY
"""

from datetime import datetime, date, timedelta


STATE_HIERARCHY = {"HEALTHY": 0, "ELEVATED": 1, "EXTREME": 2, "CRISIS": 3}


def calculate_fragility_state(
    hhi: float,
    breadth_pct: float,
    spy_rsp_6m_delta: float,
    ai_gap_data: dict,
    fragility_config: dict,
) -> dict:
    """
    Determines fragility state from 4 indicators.
    Worst-case principle: highest (worst) state from ANY indicator wins.

    Returns: {"state": str, "triggers_active": list, "indicators": dict}
    """
    triggers = []
    max_state = "HEALTHY"

    thresholds = fragility_config.get("indicators", {})

    # HHI
    if hhi is not None:
        hhi_t = thresholds.get("hhi", {}).get("thresholds", {})
        if hhi > hhi_t.get("EXTREME", {}).get("min", 2000):
            triggers.append({
                "indicator": "HHI", "value": hhi,
                "threshold": f">{hhi_t['EXTREME']['min']}", "implies": "EXTREME",
            })
            max_state = _escalate(max_state, "EXTREME")
        elif hhi > hhi_t.get("ELEVATED", {}).get("min", 1500):
            triggers.append({
                "indicator": "HHI", "value": hhi,
                "threshold": f">{hhi_t['ELEVATED']['min']}", "implies": "ELEVATED",
            })
            max_state = _escalate(max_state, "ELEVATED")

    # Breadth
    if breadth_pct is not None:
        b_t = thresholds.get("breadth_pct", {}).get("thresholds", {})
        extreme_max = b_t.get("EXTREME", {}).get("max", 50)
        elevated_max = b_t.get("ELEVATED", {}).get("max", 70)
        if breadth_pct < extreme_max:
            triggers.append({
                "indicator": "Breadth", "value": breadth_pct,
                "threshold": f"<{extreme_max}%", "implies": "EXTREME",
            })
            max_state = _escalate(max_state, "EXTREME")
        elif breadth_pct < elevated_max:
            triggers.append({
                "indicator": "Breadth", "value": breadth_pct,
                "threshold": f"<{elevated_max}%", "implies": "ELEVATED",
            })
            max_state = _escalate(max_state, "ELEVATED")

    # SPY/RSP 6M Delta
    if spy_rsp_6m_delta is not None:
        s_t = thresholds.get("spy_rsp_6m_delta", {}).get("thresholds", {})
        if spy_rsp_6m_delta > s_t.get("EXTREME", {}).get("min", 0.20):
            triggers.append({
                "indicator": "SPY/RSP 6M Delta", "value": spy_rsp_6m_delta,
                "threshold": f">{s_t['EXTREME']['min']:.0%}", "implies": "EXTREME",
            })
            max_state = _escalate(max_state, "EXTREME")
        elif spy_rsp_6m_delta > s_t.get("ELEVATED", {}).get("min", 0.10):
            triggers.append({
                "indicator": "SPY/RSP 6M Delta", "value": spy_rsp_6m_delta,
                "threshold": f">{s_t['ELEVATED']['min']:.0%}", "implies": "ELEVATED",
            })
            max_state = _escalate(max_state, "ELEVATED")

    # AI Capex/Revenue Gap
    if ai_gap_data is not None and not is_stale(ai_gap_data, fragility_config):
        gap_ratio = ai_gap_data.get("gap_ratio", 0)
        gap_direction = ai_gap_data.get("gap_direction", "STABLE")

        if gap_ratio > 6.0:
            triggers.append({
                "indicator": "AI Capex/Revenue Gap", "value": gap_ratio,
                "threshold": ">6.0", "implies": "EXTREME",
            })
            max_state = _escalate(max_state, "EXTREME")
        elif gap_ratio > 4.0 and gap_direction == "WIDENING":
            triggers.append({
                "indicator": "AI Capex/Revenue Gap", "value": gap_ratio,
                "direction": gap_direction,
                "threshold": ">4.0 + WIDENING", "implies": "ELEVATED",
            })
            max_state = _escalate(max_state, "ELEVATED")

    return {
        "state": max_state,
        "triggers_active": triggers,
        "indicators": {
            "hhi": hhi,
            "breadth_pct": breadth_pct,
            "spy_rsp_6m_delta": spy_rsp_6m_delta,
            "ai_capex_revenue_gap": ai_gap_data,
        },
    }


def check_crisis_condition(
    mag7_drawdown_pct: float,
    spy_drawdown_pct: float,
) -> bool:
    """
    CRISIS = Mag7 falls hard while broad market holds.
    Signals that concentration is BREAKING.

    mag7_drawdown_pct: negative (e.g., -0.35)
    spy_drawdown_pct: negative (e.g., -0.12)
    """
    if mag7_drawdown_pct is None or spy_drawdown_pct is None:
        return False
    return mag7_drawdown_pct < -0.30 and spy_drawdown_pct > -0.15


def get_consequence_recommendations(state: str, fragility_config: dict) -> dict:
    """Returns structured recommendations for the given state."""
    recommendations = fragility_config.get("consequence_recommendations", {})
    return recommendations.get(state, recommendations.get("HEALTHY", {}))


def is_stale(ai_gap_data: dict, fragility_config: dict = None, max_age_days: int = 100) -> bool:
    """Checks if manual AI Gap entry is too old."""
    if ai_gap_data is None:
        return True

    if fragility_config:
        max_age_days = fragility_config.get("indicators", {}).get(
            "ai_capex_revenue_gap", {}
        ).get("max_staleness_days", max_age_days)

    date_entered = ai_gap_data.get("date_entered")
    if not date_entered:
        return True

    try:
        entered = datetime.strptime(date_entered, "%Y-%m-%d").date()
        age = (date.today() - entered).days
        return age > max_age_days
    except (ValueError, TypeError):
        return True


def _escalate(current_state: str, new_state: str) -> str:
    """Returns the higher (worse) state."""
    if STATE_HIERARCHY.get(new_state, 0) > STATE_HIERARCHY.get(current_state, 0):
        return new_state
    return current_state
