"""
Market Analyst — IC Integration Module
Dynamically integrates Intelligence Collector data into layer scoring.

IC weight is NOT fixed — it depends on the SITUATION:
  - Data clear, IC confirms   → CONTEXTUAL
  - Data clear, IC contradicts → SECONDARY (creates tension)
  - Data unclear               → PRIMARY (resolves ambiguity)
  - Data unclear + thesis shift → PRIMARY (IC sees what data doesn't yet)

Source: AGENT2_SPEC_TEIL3 Section 7.2, AGENT2_SPEC_TEIL5 Sections 18.1-18.3
"""


def calculate_ic_status(
    ic_data: dict,
    layer_name: str,
    ic_config: dict,
    data_direction: str = "NEUTRAL",
) -> dict:
    """
    Determines if IC confirms, contradicts, or is silent for this layer.

    ic_data: full IC output with "consensus", "high_novelty_claims", "catalysts"
    layer_name: full layer name
    ic_config: from ic_integration.json
    data_direction: "POSITIVE" | "NEGATIVE" | "NEUTRAL" (from data sub-scores)

    Returns: {
        "ic_confirmation": "CONFIRMING" | "CONTRADICTING" | "NO_DATA",
        "ic_dissent": bool,
        "ic_thesis_shift": list | None,
        "ic_score": int | None,
        "ic_weight_used": str
    }
    """
    topic_mapping = ic_config.get("topic_layer_mapping", {})
    relevant_topics = topic_mapping.get(layer_name, [])

    if not relevant_topics or not ic_data:
        return _no_data_result()

    consensus = ic_data.get("consensus", {})

    # Find the best matching topic
    best_topic = None
    best_confidence = None
    for topic in relevant_topics:
        topic_data = consensus.get(topic)
        if topic_data and topic_data.get("confidence") != "NO_DATA":
            if best_topic is None:
                best_topic = topic
                best_confidence = topic_data
            elif _confidence_rank(topic_data.get("confidence", "LOW")) > \
                 _confidence_rank(best_confidence.get("confidence", "LOW")):
                best_topic = topic
                best_confidence = topic_data

    if best_topic is None:
        return _no_data_result()

    topic_data = best_confidence

    # Determine IC direction
    ic_score_raw = topic_data.get("score", 0)
    if ic_score_raw > 0:
        ic_direction = "POSITIVE"
    elif ic_score_raw < 0:
        ic_direction = "NEGATIVE"
    else:
        ic_direction = "NEUTRAL"

    # Confirmation
    if ic_direction == data_direction:
        confirmation = "CONFIRMING"
    elif ic_direction == "NEUTRAL":
        confirmation = "NO_DATA"
    else:
        confirmation = "CONTRADICTING"

    # Dissent: sources disagree internally
    source_count = topic_data.get("source_count", 0)
    confidence = topic_data.get("confidence", "HIGH")
    dissent = source_count >= 3 and confidence in ["LOW", "MEDIUM"]

    # Normalize IC score to -10..+10
    ic_score = _normalize_ic_score(ic_score_raw)

    return {
        "ic_confirmation": confirmation,
        "ic_dissent": dissent,
        "ic_thesis_shift": None,  # Set separately by detect_thesis_shifts
        "ic_score": ic_score,
        "ic_weight_used": "CONTEXTUAL",  # Default, overridden by determine_ic_weight
        "ic_topic": best_topic,
    }


def determine_ic_weight(
    data_clarity: float,
    ic_status: dict,
    ic_config: dict,
) -> str:
    """
    Dynamically determines IC weight based on situation.

    Source: AGENT2_SPEC_TEIL3 Section 7.2

    Returns: "PRIMARY" | "SECONDARY" | "CONTEXTUAL"
    """
    has_thesis_shift = ic_status.get("ic_thesis_shift") is not None

    # Rule priority (highest first)
    if data_clarity < 0.7 and has_thesis_shift:
        return "PRIMARY"  # IC sees what data doesn't yet

    if data_clarity < 0.5:
        return "PRIMARY"  # Data unclear, IC resolves ambiguity

    if data_clarity >= 0.7 and ic_status.get("ic_confirmation") == "CONTRADICTING":
        return "SECONDARY"  # Creates tension

    if data_clarity >= 0.7 and ic_status.get("ic_confirmation") == "CONFIRMING":
        return "CONTEXTUAL"  # Data sufficient, IC merely confirms

    return "CONTEXTUAL"  # Default


def detect_thesis_shifts(
    ic_high_novelty_claims: list,
    layer_name: str,
    topic_mapping: dict,
    threshold: int = 9,
) -> list:
    """
    Checks if IC has a fundamental thesis shift for this layer.
    novelty_score >= threshold = potential game-changer.

    Source: AGENT2_SPEC_TEIL5 Section 18.3

    Returns: list of shifts (empty if none)
    """
    relevant_topics = topic_mapping.get(layer_name, [])
    shifts = []

    for claim in (ic_high_novelty_claims or []):
        if claim.get("topic") in relevant_topics and \
           claim.get("novelty_score", 0) >= threshold:
            shifts.append({
                "source": claim.get("source"),
                "claim": claim.get("claim"),
                "direction": claim.get("direction"),
                "novelty": claim.get("novelty_score"),
            })

    return shifts if shifts else None


def get_data_direction(sub_scores: dict) -> str:
    """
    Derives the overall data direction from sub-scores (excluding IC fields).
    """
    data_scores = [v for k, v in sub_scores.items() if not k.startswith("ic_") and v != 0]
    if not data_scores:
        return "NEUTRAL"

    avg = sum(data_scores) / len(data_scores)
    if avg > 1:
        return "POSITIVE"
    elif avg < -1:
        return "NEGATIVE"
    return "NEUTRAL"


def _normalize_ic_score(raw_score: float) -> int:
    """Normalizes IC consensus score to integer -10..+10."""
    return int(max(-10, min(10, round(raw_score))))


def _confidence_rank(confidence: str) -> int:
    """Ranks confidence for comparison."""
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NO_DATA": 0}.get(confidence, 0)


def _no_data_result() -> dict:
    return {
        "ic_confirmation": "NO_DATA",
        "ic_dissent": False,
        "ic_thesis_shift": None,
        "ic_score": None,
        "ic_weight_used": "CONTEXTUAL",
        "ic_topic": None,
    }
