"""
Market Analyst — Layer Calculator Module
Computes weighted layer scores from sub-scores and assigns regimes.

- Weighted average using PRIMARY(3) / SECONDARY(2) / CONTEXTUAL(1)
- Integer scores -10 to +10
- Regime assignment from layer_regimes.json thresholds
- NO aggregated system score (by design)

Source: AGENT2_SPEC_TEIL3 Section 7, AGENT2_SPEC_TEIL2 Section 4.1
"""

WEIGHT_MAP = {"PRIMARY": 3, "SECONDARY": 2, "CONTEXTUAL": 1}


def calculate_layer_score(
    sub_scores: dict,
    field_weights: dict,
    v16_state: str,
) -> int:
    """
    Weighted average of sub-scores using weight categories.

    sub_scores: {field_name: int(-10..+10)}
    field_weights: {field_name: {"risk_on": "PRIMARY", "risk_off": "SECONDARY"}}
    v16_state: "Risk-On" | "Risk-Off" | "DD-Protect"

    Returns: integer layer score -10 to +10.
    """
    weight_key = "risk_on" if v16_state == "Risk-On" else "risk_off"

    total_weighted = 0
    total_weight = 0

    for field, score in sub_scores.items():
        weight_config = field_weights.get(field)
        if weight_config is None:
            continue

        weight_cat = weight_config.get(weight_key, "CONTEXTUAL")
        w = WEIGHT_MAP.get(weight_cat, 1)
        total_weighted += score * w
        total_weight += w

    if total_weight == 0:
        return 0

    raw = total_weighted / total_weight
    return int(max(-10, min(10, round(raw))))


def assign_regime(
    layer_score: int,
    layer_name: str,
    layer_regimes_config: dict,
    direction: str = None,
) -> str:
    """
    Assigns a regime category based on layer score and config thresholds.

    Special case: L2 RECOVERY is directional (score negative but IMPROVING).

    layer_score: integer -10 to +10
    layer_name: full layer name (e.g., "Global Liquidity Cycle (L1)")
    layer_regimes_config: full layer_regimes.json content
    direction: "IMPROVING"/"DETERIORATING"/etc. (for L2 RECOVERY detection)

    Returns: regime string (e.g., "EXPANSION", "TIGHTENING")
    """
    config = layer_regimes_config.get(layer_name)
    if config is None:
        return "UNKNOWN"

    regimes = config.get("regimes", {})

    # Special case: L2 Macro Regime — RECOVERY detection
    if layer_name == "Macro Regime (L2)" and direction == "IMPROVING" and layer_score < 0:
        return "RECOVERY"

    # Standard threshold-based assignment
    for regime_name, thresholds in regimes.items():
        # Skip special regimes (like RECOVERY with _special key)
        if "_special" in thresholds:
            continue

        score_min = thresholds.get("score_min", -999)
        score_max = thresholds.get("score_max", 999)

        if score_min <= layer_score <= score_max:
            return regime_name

    # Edge case: score exactly at boundary — use regime_order to find
    # the regime whose range contains the score
    regime_order = config.get("regime_order", [])
    for regime_name in regime_order:
        thresholds = regimes.get(regime_name, {})
        if "_special" in thresholds:
            continue
        score_min = thresholds.get("score_min", -999)
        score_max = thresholds.get("score_max", 999)
        if score_min <= layer_score <= score_max:
            return regime_name

    return "UNKNOWN"


def get_layer_id(layer_name: str) -> str:
    """
    Extracts layer ID from full name.
    "Global Liquidity Cycle (L1)" -> "L1"
    """
    start = layer_name.rfind("(")
    end = layer_name.rfind(")")
    if start != -1 and end != -1:
        return layer_name[start + 1 : end]
    return layer_name


def calculate_data_clarity(sub_scores: dict) -> float:
    """
    How uniform are the sub-scores? Used for IC weight determination.
    0.0 = completely contradictory, 1.0 = all agree.

    Source: AGENT2_SPEC_TEIL3 Section 7.2
    """
    if len(sub_scores) < 2:
        return 1.0

    signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in sub_scores.values()]
    non_zero = [s for s in signs if s != 0]

    if not non_zero:
        return 0.5  # All neutral

    agreement = abs(sum(non_zero)) / len(non_zero)
    return round(agreement, 2)


def build_sub_score_detail(
    layer_sub_scores: dict,
    field_weights: dict,
    all_sub_scores: dict,
    v16_state: str,
) -> dict:
    """
    Builds the detailed sub_scores dict for output.
    Each entry includes score, weight used, and timing.

    Returns: {field_name: {"score": int, "weight": str, "timing": str}}
    """
    weight_key = "risk_on" if v16_state == "Risk-On" else "risk_off"
    result = {}

    for field, score in layer_sub_scores.items():
        weight_config = field_weights.get(field, {})
        weight_cat = weight_config.get(weight_key, "CONTEXTUAL")

        timing = "COINCIDENT"  # default
        if field in all_sub_scores:
            timing = all_sub_scores[field].get("timing", "COINCIDENT")

        result[field] = {
            "score": score,
            "weight": weight_cat,
            "timing": timing,
        }

    return result
