"""
IC Pipeline — Stufe 2: Novelty & Belief-State Tracking
Updates belief states and validates/adjusts novelty scores.
Deterministic, no LLM.
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

BELIEFS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "history", "beliefs.json"
)


def load_beliefs() -> dict:
    path = os.path.normpath(BELIEFS_PATH)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"beliefs": {}, "last_updated": None}


def save_beliefs(beliefs_data: dict) -> None:
    path = os.path.normpath(BELIEFS_PATH)
    beliefs_data["last_updated"] = date.today().isoformat()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(beliefs_data, f, indent=2)


def _direction_changed(old_dir: str, new_dir: str) -> bool:
    """Check if direction actually changed (ignoring NEUTRAL<->MIXED)."""
    if old_dir == new_dir:
        return False
    # BULLISH <-> BEARISH is a major shift
    if {old_dir, new_dir} == {"BULLISH", "BEARISH"}:
        return True
    # Any other change
    return old_dir != new_dir


def update_beliefs_and_validate_novelty(
    claims: list[dict],
    beliefs_data: dict,
) -> tuple[list[dict], dict]:
    """
    Update belief states based on today's claims and validate novelty scores.

    Logic (per Spec Teil 3 §3.5):
    - For each claim: compare with current belief state of source+topic
    - If direction or intensity changes (delta >= 2): update belief, log change
    - Claims that change beliefs get higher novelty (minimum 5)
    - Claims that confirm beliefs get capped novelty (max 3)

    Returns:
        (updated_claims, updated_beliefs_data)
    """
    beliefs = beliefs_data.get("beliefs", {})
    today = date.today().isoformat()

    for claim in claims:
        source_id = claim["source_id"]
        topic = claim.get("primary_topic", "")
        direction = claim["sentiment"]["direction"]
        intensity = claim["sentiment"]["intensity"]

        if not topic:
            continue

        # Get current belief for this source + topic
        source_beliefs = beliefs.setdefault(source_id, {})
        current = source_beliefs.get(topic)

        if current is None:
            # First time seeing this source+topic — set initial belief
            source_beliefs[topic] = {
                "current_direction": direction,
                "current_intensity": intensity,
                "last_change_date": today,
                "last_change_from": None,
                "last_change_to": f"{direction}:{intensity}",
                "change_history": [],
            }
            # First observation: novelty stays as-is (LLM's initial assessment)
            continue

        old_dir = current["current_direction"]
        old_int = current["current_intensity"]
        intensity_delta = abs(intensity - old_int)
        dir_changed = _direction_changed(old_dir, direction)

        if dir_changed or intensity_delta >= 2:
            # BELIEF CHANGE — significant shift
            change_entry = {
                "date": today,
                "from": f"{old_dir}:{old_int}",
                "to": f"{direction}:{intensity}",
            }
            current["change_history"].append(change_entry)
            current["last_change_from"] = f"{old_dir}:{old_int}"
            current["last_change_to"] = f"{direction}:{intensity}"
            current["last_change_date"] = today
            current["current_direction"] = direction
            current["current_intensity"] = intensity

            # Validate novelty: belief-changing claims should be >= 5
            if claim["novelty_score"] < 5:
                old_novelty = claim["novelty_score"]
                claim["novelty_score"] = max(5, claim["novelty_score"])
                claim["novelty_note"] = (
                    f"[AUTO-ADJUSTED from {old_novelty}] "
                    f"Belief shift: {old_dir}:{old_int} → {direction}:{intensity}. "
                    + claim.get("novelty_note", "")
                )
                logger.info(
                    f"[{source_id}/{topic}] Novelty adjusted {old_novelty}→{claim['novelty_score']} "
                    f"(belief shift)"
                )

            # Major direction reversal: novelty minimum 7
            if dir_changed and {old_dir, direction} == {"BULLISH", "BEARISH"}:
                if claim["novelty_score"] < 7:
                    claim["novelty_score"] = 7
                    claim["novelty_note"] = (
                        f"[MAJOR SHIFT] {old_dir}→{direction}. " + claim.get("novelty_note", "")
                    )
        else:
            # BELIEF CONFIRMED — no significant change
            # Cap novelty for confirming claims
            if claim["novelty_score"] > 3 and claim["claim_type"] in ("OPINION", "PREDICTION"):
                old_novelty = claim["novelty_score"]
                claim["novelty_score"] = min(3, claim["novelty_score"])
                claim["novelty_note"] = (
                    f"[AUTO-CAPPED from {old_novelty}] "
                    f"Confirms existing belief {old_dir}:{old_int}. "
                    + claim.get("novelty_note", "")
                )

            # Update intensity if minor change
            current["current_intensity"] = intensity

    beliefs_data["beliefs"] = beliefs
    return claims, beliefs_data


def compute_freshness(
    sources_config: list[dict],
    claims: list[dict],
    beliefs_data: dict,
) -> dict:
    """
    Compute intelligence freshness per source.
    Three dimensions per Spec Teil 3 §3.4.
    """
    today = date.today()
    freshness = {}

    # Group claims by source to find latest content date
    source_latest = {}
    for claim in claims:
        sid = claim["source_id"]
        cd = claim.get("content_date", "")
        if cd:
            try:
                dt = datetime.strptime(cd, "%Y-%m-%d").date()
                if sid not in source_latest or dt > source_latest[sid]:
                    source_latest[sid] = dt
            except ValueError:
                pass

    beliefs = beliefs_data.get("beliefs", {})

    for src in sources_config:
        sid = src["source_id"]
        if not src.get("active", True):
            continue

        # days_since_last_content
        last_content = source_latest.get(sid)
        days_since = (today - last_content).days if last_content else None

        # belief_unchanged_days — days since last novelty >= 5 claim for this source
        source_beliefs = beliefs.get(sid, {})
        last_change_dates = []
        for topic, belief in source_beliefs.items():
            lcd = belief.get("last_change_date")
            if lcd:
                try:
                    last_change_dates.append(datetime.strptime(lcd, "%Y-%m-%d").date())
                except ValueError:
                    pass
        if last_change_dates:
            most_recent_change = max(last_change_dates)
            belief_unchanged = (today - most_recent_change).days
        else:
            belief_unchanged = None

        # content_expected_next — based on cadence
        cadence = src.get("cadence", "weekly")
        cadence_map = {
            "daily": 1, "3x_weekly": 2, "2x_weekly": 3,
            "weekly": 7, "bi-weekly": 14, "monthly": 30,
        }
        expected_interval = cadence_map.get(cadence, 7)
        if days_since is not None:
            expected_next = max(0, expected_interval - days_since)
        else:
            expected_next = expected_interval

        freshness[sid] = {
            "days_since_content": days_since,
            "belief_unchanged_days": belief_unchanged,
            "expected_next": expected_next,
        }

    return freshness