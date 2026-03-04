"""
Market Analyst — System Synthesis Module
Derives categorical system regime from all 8 layers.

NO aggregated score. "L1=+5, L8=-6" and "L1=-1, L8=+1" both average to ~0
but are fundamentally different situations.

System Regimes:
  RISK_OFF_FORCED — L8 Tail Risk dominates everything
  BROAD_RISK_ON   — 6+ layers positive, 0 negative
  BROAD_RISK_OFF  — 6+ layers negative, 0 positive
  CONFLICTED      — 3+ positive AND 3+ negative (most dangerous state)
  SELECTIVE       — Some positive, some negative, with a lean
  NEUTRAL         — Most layers near zero

Source: AGENT2_SPEC_TEIL6 Section 20
"""


def synthesize_system_regime(layer_results: dict) -> dict:
    """
    Derives system regime status from all 8 layers.
    Categorical, not numerical.

    layer_results: {layer_name: {"score": int, ...}}

    Returns: {
        "regime": str,
        "reason": str,
        "positive_layers": list,
        "negative_layers": list,
        ...
    }
    """
    scores = {name: data.get("score", 0) for name, data in layer_results.items()}

    positive_layers = [n for n, s in scores.items() if s >= 3]
    negative_layers = [n for n, s in scores.items() if s <= -3]
    neutral_layers = [n for n, s in scores.items() if -3 < s < 3]

    n_pos = len(positive_layers)
    n_neg = len(negative_layers)

    # L8 Override: Tail Risk dominates everything
    l8_score = scores.get("Tail Risk & Black Swan (L8)", 0)
    if l8_score <= -5:
        return {
            "regime": "RISK_OFF_FORCED",
            "reason": f"Tail Risk & Black Swan (L8) at {l8_score} — overrides all other signals",
            "positive_layers": positive_layers,
            "negative_layers": negative_layers,
        }

    # Broad Risk On
    if n_pos >= 6 and n_neg == 0:
        return {
            "regime": "BROAD_RISK_ON",
            "reason": f"{n_pos} layers positive, 0 negative — broad alignment",
            "positive_layers": positive_layers,
            "negative_layers": negative_layers,
        }

    # Broad Risk Off
    if n_neg >= 6 and n_pos == 0:
        return {
            "regime": "BROAD_RISK_OFF",
            "reason": f"{n_neg} layers negative, 0 positive — broad deterioration",
            "positive_layers": positive_layers,
            "negative_layers": negative_layers,
        }

    # Conflicted — MOST IMPORTANT state
    if n_pos >= 3 and n_neg >= 3:
        return {
            "regime": "CONFLICTED",
            "reason": f"{n_pos} positive AND {n_neg} negative — mixed signals, reduce conviction",
            "positive_layers": positive_layers,
            "negative_layers": negative_layers,
            "note": "Most dangerous state. Reduce position sizes across system.",
        }

    # Selective
    if n_pos >= 2 or n_neg >= 2:
        dominant = "POSITIVE" if n_pos > n_neg else "NEGATIVE"
        return {
            "regime": "SELECTIVE",
            "lean": dominant,
            "reason": f"{n_pos} positive, {n_neg} negative — opportunities in specific areas",
            "positive_layers": positive_layers,
            "negative_layers": negative_layers,
        }

    # Neutral
    return {
        "regime": "NEUTRAL",
        "reason": "Most layers near zero — no strong directional signal",
        "positive_layers": positive_layers,
        "negative_layers": negative_layers,
    }
