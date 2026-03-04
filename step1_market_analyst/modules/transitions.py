"""
Market Analyst — Transitions Module
Tracks regime stability and proximity to regime changes.

- Regime History: duration, changes in 30d, oscillation, CHAOTIC flag
- Transition Proximity: how close is the score to a regime boundary?

Source: AGENT2_SPEC_TEIL4 Sections 13-14
"""


def calculate_regime_history(history_30d: list, layer_name: str) -> dict:
    """
    From 30 days of history, compute regime stability metrics.

    history_30d: list of daily records with {"date": ..., "layers": {layer_name: {"regime": ...}}}
    layer_name: full layer name

    Returns: {
        "current_regime": str,
        "duration_days": int,
        "regime_changes_30d": int,
        "oscillation_flag": bool,
        "chaotic_flag": bool,
        "unique_regimes_30d": list
    }
    """
    regimes = []
    for day in history_30d:
        layer_data = day.get("layers", {}).get(layer_name, {})
        regime = layer_data.get("regime")
        if regime:
            regimes.append(regime)

    if not regimes:
        return {
            "current_regime": "UNKNOWN",
            "duration_days": 0,
            "regime_changes_30d": 0,
            "oscillation_flag": False,
            "chaotic_flag": False,
            "unique_regimes_30d": [],
        }

    current_regime = regimes[-1]

    # Duration of current regime (counting back from today)
    duration = 0
    for r in reversed(regimes):
        if r == current_regime:
            duration += 1
        else:
            break

    # Regime changes in period
    changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])

    # Oscillation: pendulum between exactly 2 regimes?
    unique_regimes = list(set(regimes))
    oscillation = len(unique_regimes) == 2 and changes >= 4

    # CHAOTIC: 6+ changes in 30 days
    chaotic = changes >= 6

    return {
        "current_regime": current_regime,
        "duration_days": duration,
        "regime_changes_30d": changes,
        "oscillation_flag": oscillation,
        "chaotic_flag": chaotic,
        "unique_regimes_30d": unique_regimes,
    }


def calculate_regime_duration_score(regime_history: dict) -> float:
    """
    Converts regime history into a conviction dimension (0.0-1.0).
    Used as input to 4D conviction calculation.

    Source: AGENT2_SPEC_TEIL4 Section 10.6
    """
    days = regime_history.get("duration_days", 0)
    changes_30d = regime_history.get("regime_changes_30d", 0)
    oscillation = regime_history.get("oscillation_flag", False)

    # Base from duration
    if days >= 60:
        base = 1.0
    elif days >= 30:
        base = 0.8
    elif days >= 15:
        base = 0.6
    elif days >= 5:
        base = 0.4
    else:
        base = 0.2

    # Penalty for frequent changes
    if changes_30d >= 4:
        base *= 0.5  # CHAOTIC territory
    elif changes_30d >= 2:
        base *= 0.7

    # Penalty for oscillation
    if oscillation:
        base *= 0.6

    return round(min(1.0, max(0.0, base)), 2)


def calculate_transition_proximity(
    current_score: int,
    current_regime: str,
    layer_regime_config: dict,
    velocity: str,
    acceleration: str,
) -> dict:
    """
    How close is the layer to a regime change?

    current_score: integer -10 to +10
    current_regime: e.g., "EXPANSION"
    layer_regime_config: config for this specific layer from layer_regimes.json
    velocity: "ACCELERATING" | "MOVING" | "STEADY" | "DECELERATING"
    acceleration: "STRONGLY_ACCELERATING" | ... | "FLAT"

    Returns: {
        "proximity": float (0.0=far, 1.0=imminent),
        "target_regime": str,
        "target_direction": "UP" | "DOWN",
        "estimated_days": int | None,
        "distance_to_boundary": int
    }
    """
    regimes = layer_regime_config.get("regimes", {})
    current_thresholds = regimes.get(current_regime, {})

    # Handle special regimes (like RECOVERY with _special key)
    if "_special" in current_thresholds:
        return _default_proximity()

    score_min = current_thresholds.get("score_min", -10)
    score_max = current_thresholds.get("score_max", 10)

    # Distance to each boundary
    dist_to_lower = (current_score - score_min) if score_min > -10 else 999
    dist_to_upper = (score_max - current_score) if score_max < 10 else 999
    min_distance = min(dist_to_lower, dist_to_upper)

    # Determine direction of nearest transition
    if dist_to_lower < dist_to_upper:
        target_direction = "DOWN"
        target_regime = _find_adjacent_regime(
            current_regime, "below", layer_regime_config
        )
    else:
        target_direction = "UP"
        target_regime = _find_adjacent_regime(
            current_regime, "above", layer_regime_config
        )

    # Base proximity from distance (0-1)
    range_size = score_max - score_min
    max_distance = range_size / 2 if range_size > 0 else 5
    proximity = 1.0 - (min_distance / max_distance)
    proximity = max(0.0, min(1.0, proximity))

    # Velocity adjustment: if moving toward boundary, closer
    moving_toward = (
        (target_direction == "DOWN" and velocity in ["DECELERATING", "MOVING"])
        or (target_direction == "UP" and velocity in ["ACCELERATING", "MOVING"])
    )
    if moving_toward:
        proximity = min(1.0, proximity * 1.3)

    # Acceleration adjustment
    if acceleration and "STRONGLY" in acceleration:
        proximity = min(1.0, proximity * 1.2)

    # Estimated days (rough heuristic)
    estimated_days = None
    if min_distance > 0 and velocity != "STEADY":
        estimated_days = max(1, int(min_distance * 3))

    return {
        "proximity": round(proximity, 2),
        "target_regime": target_regime,
        "target_direction": target_direction,
        "estimated_days": estimated_days,
        "distance_to_boundary": min_distance,
    }


# --- Internal helpers ---


def _default_proximity() -> dict:
    """Default response when proximity can't be calculated."""
    return {
        "proximity": 0.0,
        "target_regime": "UNKNOWN",
        "target_direction": "UNKNOWN",
        "estimated_days": None,
        "distance_to_boundary": 999,
    }


def _find_adjacent_regime(
    current_regime: str, direction: str, layer_regime_config: dict
) -> str:
    """
    Finds the regime above or below the current one in the regime_order list.

    direction: "above" | "below"
    """
    regime_order = layer_regime_config.get("regime_order", [])
    if current_regime not in regime_order:
        return "UNKNOWN"

    idx = regime_order.index(current_regime)

    if direction == "below" and idx > 0:
        return regime_order[idx - 1]
    elif direction == "above" and idx < len(regime_order) - 1:
        return regime_order[idx + 1]

    return "UNKNOWN"
