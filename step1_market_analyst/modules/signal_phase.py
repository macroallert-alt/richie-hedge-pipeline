"""
Market Analyst — Signal Phase Module
Detects whether leading indicators are diverging from lagging ones.

Signal phases:
  EARLY_SIGNAL — Leading turned but lagging hasn't yet
  CONFIRMED   — Leading and lagging agree
  CONFLICTED  — Leading and lagging contradict each other
  MIXED       — No clear pattern
  NO_SIGNAL   — No leading indicators with signal

Source: AGENT2_SPEC_TEIL3 Section 9
"""


def detect_signal_phase(
    layer_sub_scores: dict,
    all_sub_scores_info: dict,
    field_properties: dict,
) -> str:
    """
    Determines signal phase for a layer by comparing leading vs lagging indicators.

    layer_sub_scores: {field_name: int} — sub-scores for this layer
    all_sub_scores_info: {field_name: {"score": int, "timing": str, ...}} — all fields
    field_properties: field_properties.json content

    Returns: "EARLY_SIGNAL" | "CONFIRMED" | "CONFLICTED" | "MIXED" | "NO_SIGNAL"
    """
    leading_scores = []
    lagging_scores = []

    for field, score in layer_sub_scores.items():
        if score == 0:
            continue

        # IC fields don't have timing classification
        if field.startswith("ic_"):
            continue

        # Get timing from field_properties
        props = field_properties.get(field, {})
        timing = props.get("timing", "COINCIDENT")

        if timing == "LEADING":
            leading_scores.append(score)
        elif timing == "LAGGING":
            lagging_scores.append(score)

    if not leading_scores:
        return "NO_SIGNAL"

    leading_direction = sum(1 if s > 0 else -1 for s in leading_scores) / len(
        leading_scores
    )

    if lagging_scores:
        lagging_direction = sum(1 if s > 0 else -1 for s in lagging_scores) / len(
            lagging_scores
        )
    else:
        lagging_direction = 0

    # Leading has clearly turned but lagging hasn't
    if abs(leading_direction) > 0.5 and abs(lagging_direction) < 0.3:
        return "EARLY_SIGNAL"

    # Leading and lagging agree
    if (leading_direction > 0.3 and lagging_direction > 0.3) or (
        leading_direction < -0.3 and lagging_direction < -0.3
    ):
        return "CONFIRMED"

    # Leading and lagging contradict
    if (leading_direction > 0.3 and lagging_direction < -0.3) or (
        leading_direction < -0.3 and lagging_direction > 0.3
    ):
        return "CONFLICTED"

    return "MIXED"
