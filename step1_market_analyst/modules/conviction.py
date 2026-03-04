"""
Market Analyst — Conviction Module
4-dimensional conviction assessment per layer.

Dimensions:
  1. Data Clarity       — Are sub-scores consistent?
  2. Narrative Alignment — Does IC confirm or contradict data?
  3. Catalyst Fragility  — Is a major catalyst approaching?
  4. Regime Duration     — How stable is the current regime?

Composite = MIN(all 4) — only as strong as the weakest link.
Limiting Factor = the dimension with the lowest value.

Source: AGENT2_SPEC_TEIL4 Section 10
"""

from modules.transitions import calculate_regime_duration_score


def calculate_conviction(layer_data: dict) -> dict:
    """
    Master conviction calculation for a single layer.

    layer_data must contain:
      - "raw_data_clarity": float (from layer_calculator.calculate_data_clarity)
      - "ic_confirmation": str ("CONFIRMING" | "CONTRADICTING" | "NO_DATA")
      - "ic_dissent": bool
      - "signal_phase": str
      - "catalyst_exposure": list
      - "regime_history": dict
      - "surprise": dict (with "category")

    Returns: {
        "data_clarity": float,
        "narrative_alignment": float,
        "catalyst_fragility": float,
        "regime_duration": float,
        "composite": str (HIGH/MEDIUM/LOW/CONFLICTED),
        "limiting_factor": {"factor": str, "value": float, "label": str}
    }
    """
    # 1. Data Clarity (already computed during layer calculation)
    data_clarity = layer_data.get("raw_data_clarity", 0.5)

    # 2. Narrative Alignment
    narrative_alignment = calculate_narrative_alignment(
        layer_data.get("ic_confirmation", "NO_DATA"),
        layer_data.get("ic_dissent", False),
        layer_data.get("signal_phase", "MIXED"),
    )

    # 3. Catalyst Fragility
    catalyst_fragility = calculate_catalyst_fragility(
        layer_data.get("catalyst_exposure", [])
    )

    # Apply surprise cap to catalyst_fragility
    surprise_cat = layer_data.get("surprise", {}).get("category", "NORMAL")
    catalyst_fragility = _apply_surprise_cap(surprise_cat, catalyst_fragility)

    # 4. Regime Duration
    regime_duration = calculate_regime_duration_score(
        layer_data.get("regime_history", {})
    )

    # Composite + Limiting Factor
    composite = categorize_conviction(
        data_clarity, narrative_alignment, catalyst_fragility, regime_duration
    )
    limiting_factor = find_limiting_factor(
        data_clarity, narrative_alignment, catalyst_fragility, regime_duration
    )

    return {
        "data_clarity": data_clarity,
        "narrative_alignment": narrative_alignment,
        "catalyst_fragility": catalyst_fragility,
        "regime_duration": regime_duration,
        "composite": composite,
        "limiting_factor": limiting_factor,
    }


def categorize_conviction(dc: float, na: float, cf: float, rd: float) -> str:
    """
    Composite conviction is NOT the average.
    It's as strong as the weakest link.

    Source: AGENT2_SPEC_TEIL4 Section 10.2
    """
    minimum = min(dc, na, cf, rd)

    if minimum >= 0.7:
        return "HIGH"
    elif minimum >= 0.4:
        return "MEDIUM"
    elif minimum >= 0.2:
        return "LOW"
    else:
        return "CONFLICTED"


def find_limiting_factor(dc: float, na: float, cf: float, rd: float) -> dict:
    """
    What holds conviction back? THIS is the most important information.

    Source: AGENT2_SPEC_TEIL4 Section 10.3
    """
    factors = {
        "data_clarity": dc,
        "narrative_alignment": na,
        "catalyst_fragility": cf,
        "regime_duration": rd,
    }
    weakest = min(factors, key=factors.get)

    labels = {
        "data_clarity": "Sub-scores conflicting — data sends mixed signals",
        "narrative_alignment": "IC narrative contradicts or missing",
        "catalyst_fragility": "Major catalyst approaching — outcome uncertain",
        "regime_duration": "Regime too young or unstable",
    }

    return {
        "factor": weakest,
        "value": factors[weakest],
        "label": labels[weakest],
    }


def calculate_narrative_alignment(
    ic_confirmation: str,
    ic_dissent: bool,
    signal_phase: str,
) -> float:
    """
    How well does the IC narrative align with data?

    Source: AGENT2_SPEC_TEIL4 Section 10.4
    """
    base_map = {
        "CONFIRMING": 0.9,
        "NO_DATA": 0.5,
        "CONTRADICTING": 0.2,
    }
    base = base_map.get(ic_confirmation, 0.5)

    if ic_dissent:
        base *= 0.7  # Internal IC disagreement reduces alignment

    if signal_phase == "EARLY_SIGNAL":
        base *= 0.8  # Not yet confirmed
    elif signal_phase == "CONFLICTED":
        base *= 0.6  # Leading vs lagging contradict

    return round(min(1.0, max(0.0, base)), 2)


def calculate_catalyst_fragility(catalyst_exposures: list) -> float:
    """
    How fragile is the current state due to upcoming catalysts?
    No catalysts = 1.0 (robust). FOMC tomorrow = 0.1 (fragile).

    Source: AGENT2_SPEC_TEIL4 Section 10.5
    """
    if not catalyst_exposures:
        return 1.0

    fragility = 1.0

    for cat in catalyst_exposures:
        days = cat.get("days_until", 999)
        direction = cat.get("direction", "INCREMENTAL")

        if direction == "BINARY":
            # FOMC, CPI: outcome can change everything
            if days <= 1:
                impact = 0.1
            elif days <= 3:
                impact = 0.3
            elif days <= 7:
                impact = 0.6
            else:
                impact = 0.8
        elif direction == "DIRECTIONAL":
            # OpEx, Earnings: known direction, uncertain magnitude
            if days <= 1:
                impact = 0.3
            elif days <= 3:
                impact = 0.5
            else:
                impact = 0.8
        else:  # INCREMENTAL or UNKNOWN
            impact = 0.8

        fragility = min(fragility, impact)

    return round(fragility, 2)


def _apply_surprise_cap(surprise_category: str, catalyst_fragility: float) -> float:
    """
    Caps catalyst_fragility based on surprise severity.
    EXTREME surprise -> max 0.4, HIGH -> max 0.6.

    Source: AGENT2_SPEC_TEIL4 Section 12.2
    """
    caps = {
        "EXTREME": 0.4,
        "HIGH": 0.6,
    }
    cap = caps.get(surprise_category)
    if cap is not None and catalyst_fragility > cap:
        return cap
    return catalyst_fragility
