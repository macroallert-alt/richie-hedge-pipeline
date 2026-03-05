"""
IC Pipeline — Stufe 2: Consensus Calculator
Weighted consensus per topic. Deterministic, no LLM.

FIX.1 Applied: Aggregates claims per source FIRST (average),
then weights per source with expertise. A source with 5 claims
has the same weight as one with 1 claim (at equal expertise).
"""

import logging

logger = logging.getLogger(__name__)


def calculate_all_consensus(
    claims: list[dict],
    sources_config: list[dict],
    expertise_matrix: dict,
    taxonomy_topics: list[str],
) -> dict:
    """
    Calculate weighted consensus for all 15 topics.

    Konsens-Formel (Addendum FIX.1 corrected):
    1. For each source: average all bias-adjusted signals for the topic
    2. weighted_consensus = Σ(avg_signal[source] × expertise[source][topic]) / Σ(expertise[source][topic])

    Only OPINION and PREDICTION claims flow into consensus.

    Returns:
        dict: topic -> {consensus_score, source_count, total_claims, confidence, contributors}
    """
    # Build bias lookup from sources config
    bias_lookup = {}
    for src in sources_config:
        bias_lookup[src["source_id"]] = src.get("known_bias", 0)

    expertise = expertise_matrix.get("expertise", expertise_matrix)

    results = {}

    for topic in taxonomy_topics:
        # Filter: only OPINION + PREDICTION for this topic's primary_topic
        topic_claims = [
            c for c in claims
            if c.get("primary_topic") == topic
            and c.get("claim_type") in ("OPINION", "PREDICTION")
        ]

        if not topic_claims:
            results[topic] = {
                "consensus_score": 0.0,
                "source_count": 0,
                "total_claims": 0,
                "confidence": "NO_DATA",
                "contributors": [],
            }
            continue

        # STEP 1: Aggregate claims per source (average bias-adjusted signal)
        source_signals = {}   # source_id -> list of bias_adjusted signals
        source_claim_ids = {} # source_id -> list of claim IDs

        for claim in topic_claims:
            source_id = claim["source_id"]
            intensity = claim["sentiment"]["intensity"]
            direction = claim["sentiment"]["direction"]

            if direction == "BULLISH":
                signed = intensity
            elif direction == "BEARISH":
                signed = -intensity
            elif direction == "MIXED":
                signed = 0
            else:  # NEUTRAL
                signed = 0

            known_bias = bias_lookup.get(source_id, 0)
            bias_adjusted = signed - known_bias

            source_signals.setdefault(source_id, []).append(bias_adjusted)
            source_claim_ids.setdefault(source_id, []).append(claim.get("id", ""))

        # STEP 2: Per-source average, then expertise-weighted consensus
        numerator = 0.0
        denominator = 0.0
        contributors = []

        for source_id, signals in source_signals.items():
            avg_signal = sum(signals) / len(signals)
            source_expertise = expertise.get(source_id, {}).get(topic, 1)

            numerator += avg_signal * source_expertise
            denominator += source_expertise  # only once per source

            contributors.append({
                "source_id": source_id,
                "claim_count": len(signals),
                "avg_bias_adjusted_signal": round(avg_signal, 2),
                "expertise_weight": source_expertise,
                "claim_ids": source_claim_ids.get(source_id, []),
            })

        consensus = numerator / denominator if denominator > 0 else 0.0

        # Confidence based on unique source count
        unique_sources = len(source_signals)
        if unique_sources >= 4:
            confidence = "HIGH"
        elif unique_sources >= 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        results[topic] = {
            "consensus_score": round(consensus, 2),
            "source_count": unique_sources,
            "total_claims": len(topic_claims),
            "confidence": confidence,
            "contributors": contributors,
        }

    return results