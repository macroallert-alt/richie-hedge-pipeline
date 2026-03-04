"""
Market Analyst — Surprise Detection Module
How unusual is today's score change compared to history?

Categories:
  EXTREME  (|z| >= 3.0) — Auto-flag, KS check, conviction capped at 0.4
  HIGH     (|z| >= 2.0) — Flag for CIO, conviction capped at 0.6
  MODERATE (|z| >= 1.5) — No special consequences
  NORMAL   (|z| < 1.5)  — No special consequences

Source: AGENT2_SPEC_TEIL4 Section 12
"""


def calculate_surprise(
    score_today: int,
    score_yesterday: int,
    historical_daily_deltas: list,
) -> dict:
    """
    Z-score of today's delta relative to historical daily deltas.

    score_today: current layer score
    score_yesterday: yesterday's layer score
    historical_daily_deltas: list of ints (past daily score changes)

    Returns: {"z_score": float, "category": str}
    """
    if score_yesterday is None:
        return {"z_score": 0.0, "category": "NORMAL"}

    delta = score_today - score_yesterday

    if not historical_daily_deltas or len(historical_daily_deltas) < 5:
        return {"z_score": 0.0, "category": "NORMAL"}

    mean_delta = sum(historical_daily_deltas) / len(historical_daily_deltas)
    variance = sum((d - mean_delta) ** 2 for d in historical_daily_deltas) / len(
        historical_daily_deltas
    )
    std_delta = variance**0.5

    if std_delta == 0:
        z = 0.0
    else:
        z = (delta - mean_delta) / std_delta

    if abs(z) >= 3.0:
        category = "EXTREME"
    elif abs(z) >= 2.0:
        category = "HIGH"
    elif abs(z) >= 1.5:
        category = "MODERATE"
    else:
        category = "NORMAL"

    return {"z_score": round(z, 2), "category": category}


def apply_surprise_to_conviction(
    surprise_category: str,
    catalyst_fragility: float,
) -> float:
    """
    Caps catalyst_fragility based on surprise severity.
    Called after surprise detection, before conviction composite.

    Source: AGENT2_SPEC_TEIL4 Section 12.2

    Returns: adjusted catalyst_fragility (may be lower, never higher)
    """
    caps = {
        "EXTREME": 0.4,
        "HIGH": 0.6,
        "MODERATE": None,  # No cap
        "NORMAL": None,
    }

    cap = caps.get(surprise_category)
    if cap is not None and catalyst_fragility > cap:
        return cap
    return catalyst_fragility
