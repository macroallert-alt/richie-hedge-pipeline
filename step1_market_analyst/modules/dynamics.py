"""
Market Analyst — Dynamics Module
Calculates temporal dynamics: how fast and in which direction scores move.

- Velocity (1st derivative): ACCELERATING / MOVING / STEADY / DECELERATING
- Acceleration (2nd derivative): STRONGLY_ACCELERATING / MILDLY_ACCELERATING / FLAT / etc.
- Direction (medium-term trend): IMPROVING / STABLE / DETERIORATING / RECOVERING / WEAKENING

Source: AGENT2_SPEC_TEIL4 Sections 11.1-11.3
"""


def calculate_velocity(
    score_today: int,
    score_yesterday: int,
    score_5d_ago: int,
) -> str:
    """
    Short-term speed of change based on 5-day delta.

    Returns: "ACCELERATING" | "DECELERATING" | "MOVING" | "STEADY"
    """
    if score_yesterday is None or score_5d_ago is None:
        return "STEADY"

    delta_5d = score_today - score_5d_ago

    if abs(delta_5d) >= 4:
        return "ACCELERATING" if delta_5d > 0 else "DECELERATING"
    elif abs(delta_5d) >= 2:
        return "MOVING"
    else:
        return "STEADY"


def calculate_acceleration(
    delta_5d_current: int,
    delta_5d_previous: int,
) -> str:
    """
    Is the SPEED of change itself changing? (2nd derivative)
    2008/2020: The problem was acceleration, not level.

    delta_5d_current: score_today - score_5d_ago
    delta_5d_previous: score_5d_ago - score_10d_ago

    Returns: "STRONGLY_ACCELERATING" | "MILDLY_ACCELERATING" | "FLAT" |
             "MILDLY_DECELERATING" | "STRONGLY_DECELERATING"
    """
    if delta_5d_current is None or delta_5d_previous is None:
        return "FLAT"

    accel = delta_5d_current - delta_5d_previous

    if abs(accel) >= 3:
        return "STRONGLY_ACCELERATING" if accel > 0 else "STRONGLY_DECELERATING"
    elif abs(accel) >= 1:
        return "MILDLY_ACCELERATING" if accel > 0 else "MILDLY_DECELERATING"
    else:
        return "FLAT"


def calculate_direction(
    score_today: int,
    score_5d_ago: int,
    score_21d_ago: int,
) -> str:
    """
    Medium-term trend combining 5d and 21d lookback.

    Returns: "IMPROVING" | "DETERIORATING" | "RECOVERING" | "WEAKENING" | "STABLE"
    """
    if score_5d_ago is None:
        return "STABLE"

    trend_5d = score_today - score_5d_ago
    trend_21d = (score_today - score_21d_ago) if score_21d_ago is not None else trend_5d

    if trend_5d > 0 and trend_21d > 0:
        return "IMPROVING"
    elif trend_5d < 0 and trend_21d < 0:
        return "DETERIORATING"
    elif trend_5d > 0 and trend_21d <= 0:
        return "RECOVERING"  # Short-term better, long-term still negative
    elif trend_5d < 0 and trend_21d >= 0:
        return "WEAKENING"  # Short-term worse, long-term still positive
    else:
        return "STABLE"


def get_score_n_days_ago(history: list, n: int) -> int:
    """
    Gets score from N days ago in history list.
    history: list of {"date": ..., "score": int} sorted oldest first.

    Returns: score int or None if not enough history.
    """
    if not history or len(history) < n + 1:
        return None
    return history[-(n + 1)].get("score")


def get_historical_daily_deltas(history: list) -> list:
    """
    Calculates daily score deltas from history.
    Returns list of ints (delta per day).
    """
    if not history or len(history) < 2:
        return []

    deltas = []
    for i in range(1, len(history)):
        prev = history[i - 1].get("score", 0)
        curr = history[i].get("score", 0)
        deltas.append(curr - prev)

    return deltas


def extract_layer_history(history_30d: list, layer_name: str) -> list:
    """
    Extracts a single layer's history from the full 30-day records.

    history_30d: list of daily records, each with {"date": ..., "layers": {layer_name: {...}}}
    layer_name: full layer name

    Returns: list of {"date": str, "score": int, "regime": str}
    """
    result = []
    for day in history_30d:
        layer_data = day.get("layers", {}).get(layer_name, {})
        if layer_data:
            result.append(
                {
                    "date": day.get("date"),
                    "score": layer_data.get("score", 0),
                    "regime": layer_data.get("regime", "UNKNOWN"),
                }
            )
    return result
