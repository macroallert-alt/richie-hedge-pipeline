"""
IC Intelligence Pipeline — Main Entry Point
Usage: python -m step_0i_ic_pipeline.main --stage all
Stages: extraction, intelligence, briefing, all

IC V2 Phase 1: 7-Day Claims Carry-Forward + Source Cards
- claims_archive.json persists ALL claims from last 7 days
- New claims are ADDED (not replaced)
- Old claims get Freshness tags: FRESH/AGING/FADING/ARCHIVED/EXPIRED
- Intelligence Engine receives ALL active claims, not just today's
- source_cards[] in latest.json intelligence block

IC V2 Phase 2: Cadence Anomaly Detection + 90-Day Archive
- cadence_anomalies[] in latest.json intelligence block
- ARCHIVED window extended to 90 days (was 14)
- EXPIRED threshold moved from 15 to 91 days

IC V2 Phase 2: Narrative Threads
- threads.json persists active narrative threads across runs
- Deterministic thread matching (Topic + Direction + Asset overlap)
- Lifecycle: SEED -> BUILDING -> ESTABLISHED -> FADING -> ARCHIVED
- challenged flag (not a status — can coexist with any lifecycle status)
- Conviction formula: Max Expertise 35%, Source Diversity 20% (capped 4),
  Data Confirmation 20%, Freshness 15%, Conviction Trend 10%
- Portfolio Alignment: CONFIRMING/THREATENING/MIXED/OPPORTUNITY/NEUTRAL
  + numeric portfolio_relevance_score
- Thread creation gated: Novelty >= 7, CORE/Expert >= 7, V16 asset overlap
- Aggressive decay: FADING 10d (17d for THREATENING), ARCHIVED 21d (28d)

IC V2 Phase 2: Position Pre-Mortems
- pre_mortems.json persists failure scenarios for V16 positions >10%
- LLM-generated weekly (or on V16 position change >5%)
- Daily deterministic evidence update (new claims matched against scenarios)
- 2-4 failure scenarios per position with early warning indicators
- Output: pre_mortems[] in latest.json intelligence block
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ic_pipeline")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")

CLAIMS_ARCHIVE_PATH = os.path.join(DATA_DIR, "history", "claims_archive.json")
SOURCE_HISTORY_PATH = os.path.join(DATA_DIR, "history", "source_history.json")
THREADS_PATH = os.path.join(DATA_DIR, "history", "threads.json")
PRE_MORTEMS_PATH = os.path.join(DATA_DIR, "history", "pre_mortems.json")
BELIEF_STATE_PATH = os.path.join(DATA_DIR, "history", "belief_state.json")

# Path to Vercel dashboard JSON (written by V16_DAILY_RUNNER, updated by IC)
DASHBOARD_JSON_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "data", "dashboard", "latest.json"
)


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Claims Archive — 7-Day Active Window + 90-Day Archive (IC V2 Phase 2)
# ---------------------------------------------------------------------------
def _compute_freshness(content_date_str: str, today: date) -> dict:
    """Compute freshness category and decay weight for a claim.

    Categories per IC V2 Spec Teil 5, Kapitel 17 / Teil 3, Kapitel 7:
      FRESH:    0-2 days,  decay 1.0
      AGING:    3-5 days,  decay 0.7
      FADING:   6-7 days,  decay 0.4
      ARCHIVED: 8-90 days, decay 0.15  (extended from 14 in Phase 2)
      EXPIRED:  91+ days,  decay 0.0   (removed from archive)
    """
    try:
        content_date = datetime.strptime(content_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        content_date = today

    age_days = (today - content_date).days

    if age_days <= 2:
        return {"freshness": "FRESH", "decay_weight": 1.0, "age_days": age_days}
    elif age_days <= 5:
        return {"freshness": "AGING", "decay_weight": 0.7, "age_days": age_days}
    elif age_days <= 7:
        return {"freshness": "FADING", "decay_weight": 0.4, "age_days": age_days}
    elif age_days <= 90:
        return {"freshness": "ARCHIVED", "decay_weight": 0.15, "age_days": age_days}
    else:
        return {"freshness": "EXPIRED", "decay_weight": 0.0, "age_days": age_days}


def _load_claims_archive() -> dict:
    """Load claims_archive.json from data/history/."""
    if os.path.exists(CLAIMS_ARCHIVE_PATH):
        try:
            return _load_json(CLAIMS_ARCHIVE_PATH)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Claims archive corrupt, starting fresh: {e}")
    return {"claims": [], "last_updated": None}


def _save_claims_archive(archive: dict) -> None:
    """Save claims_archive.json to data/history/."""
    archive["last_updated"] = date.today().isoformat()
    _save_json(archive, CLAIMS_ARCHIVE_PATH)


def _make_claim_key(claim: dict) -> str:
    """Create a unique key for deduplication.

    Key = source_id + content_date + first 80 chars of claim_text.
    This catches exact repeats from same source on same date.
    """
    text_prefix = claim.get("claim_text", "")[:80].strip().lower()
    return f"{claim.get('source_id', '')}|{claim.get('content_date', '')}|{text_prefix}"


def merge_claims_into_archive(
    new_claims: list[dict], archive: dict
) -> tuple[list[dict], dict]:
    """Merge new claims into the archive and return all active claims.

    Steps:
      1. Build key set from existing archive claims for dedup
      2. Add new claims that aren't duplicates
      3. Update freshness tags on ALL claims
      4. Remove EXPIRED claims (>90 days)
      5. Return (active_claims for Intelligence Engine, updated archive)

    Active claims = FRESH + AGING + FADING (0-7 days, used for scoring).
    ARCHIVED claims (8-90 days) stay in archive but are NOT passed to
    Intelligence Engine — they're only visible in thread detail (Phase 2+).
    """
    today = date.today()

    # Index existing claims by key for dedup
    existing_keys = {}
    for i, claim in enumerate(archive.get("claims", [])):
        key = _make_claim_key(claim)
        existing_keys[key] = i

    # Add new claims (skip duplicates)
    added_count = 0
    duplicate_count = 0
    for claim in new_claims:
        key = _make_claim_key(claim)
        if key in existing_keys:
            # Duplicate — update freshness on existing claim (reset to FRESH)
            idx = existing_keys[key]
            freshness = _compute_freshness(claim.get("content_date", ""), today)
            archive["claims"][idx]["freshness"] = freshness["freshness"]
            archive["claims"][idx]["decay_weight"] = freshness["decay_weight"]
            archive["claims"][idx]["age_days"] = freshness["age_days"]
            duplicate_count += 1
        else:
            # New claim — add with freshness
            freshness = _compute_freshness(claim.get("content_date", ""), today)
            claim["freshness"] = freshness["freshness"]
            claim["decay_weight"] = freshness["decay_weight"]
            claim["age_days"] = freshness["age_days"]
            archive["claims"].append(claim)
            existing_keys[key] = len(archive["claims"]) - 1
            added_count += 1

    # Update freshness on ALL existing claims and filter out EXPIRED
    surviving_claims = []
    for claim in archive["claims"]:
        freshness = _compute_freshness(claim.get("content_date", ""), today)
        claim["freshness"] = freshness["freshness"]
        claim["decay_weight"] = freshness["decay_weight"]
        claim["age_days"] = freshness["age_days"]
        if freshness["freshness"] != "EXPIRED":
            surviving_claims.append(claim)

    expired_count = len(archive["claims"]) - len(surviving_claims)
    archive["claims"] = surviving_claims

    # Active claims = FRESH + AGING + FADING (passed to Intelligence Engine)
    active_claims = [
        c for c in surviving_claims
        if c.get("freshness") in ("FRESH", "AGING", "FADING")
    ]

    # Log summary
    freshness_counts = {}
    for c in surviving_claims:
        f = c.get("freshness", "UNKNOWN")
        freshness_counts[f] = freshness_counts.get(f, 0) + 1

    logger.info(
        f"Claims Archive: +{added_count} new, {duplicate_count} dupes, "
        f"{expired_count} expired, {len(surviving_claims)} total in archive"
    )
    logger.info(
        f"Claims Freshness: {freshness_counts}"
    )
    logger.info(
        f"Active claims for Intelligence Engine: {len(active_claims)} "
        f"(FRESH+AGING+FADING)"
    )

    return active_claims, archive


# ---------------------------------------------------------------------------
# Cadence Anomaly Detection (IC V2 Phase 2)
# ---------------------------------------------------------------------------
# Mapping from sources.json cadence field to expected posts per week
CADENCE_BASELINE = {
    "weekly": 1.0,
    "2x_weekly": 2.0,
    "3x_weekly": 3.0,
    "5x_weekly": 5.0,
    "daily": 5.0,
    "monthly": 0.25,
}
# cadence values that skip anomaly detection (too irregular to measure)
CADENCE_SKIP = {"irregular", "irregular_free"}


def _detect_cadence_anomalies(
    claims_archive: dict,
    sources_config: list[dict],
) -> list[dict]:
    """Detect cadence anomalies per IC V2 Spec Kapitel 15.

    For each active source with a measurable cadence baseline:
      cadence_ratio = actual_posts_this_week / baseline_posts_per_week

      ratio > 1.5 -> ELEVATED
      ratio > 2.0 -> HIGH
      ratio > 3.0 -> EXTREME

    Special case: monthly sources that post again within 20 days -> HIGH

    Returns list of anomaly dicts for sources that exceed threshold.
    """
    today = date.today()
    week_ago = today - timedelta(days=7)

    # Count distinct posts per source in last 7 days
    # Use (source_id, content_date, title_prefix) as post proxy
    # since one post can produce multiple claims
    source_posts = {}
    for claim in claims_archive.get("claims", []):
        sid = claim.get("source_id", "")
        content_date_str = claim.get("content_date", "")
        try:
            content_date = datetime.strptime(content_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        if content_date < week_ago:
            continue

        # Use source_id + content_date as post key
        # (multiple claims from same post share same content_date)
        post_key = f"{sid}|{content_date_str}"
        if sid not in source_posts:
            source_posts[sid] = set()
        source_posts[sid].add(post_key)

    # Build source lookup
    source_lookup = {s["source_id"]: s for s in sources_config}

    anomalies = []
    for src in sources_config:
        sid = src["source_id"]
        if not src.get("active", True):
            continue

        cadence = src.get("cadence", "")
        if cadence in CADENCE_SKIP:
            continue

        baseline = CADENCE_BASELINE.get(cadence)
        if baseline is None:
            continue

        actual_posts = len(source_posts.get(sid, set()))

        # Skip if no posts (silence is not a cadence anomaly —
        # that's handled by Silence Map in Phase 3)
        if actual_posts == 0:
            continue

        cadence_ratio = actual_posts / baseline if baseline > 0 else 0

        # Determine anomaly level
        anomaly_level = None
        if cadence_ratio > 3.0:
            anomaly_level = "EXTREME"
        elif cadence_ratio > 2.0:
            anomaly_level = "HIGH"
        elif cadence_ratio > 1.5:
            anomaly_level = "ELEVATED"

        # Special case: monthly sources posting again within 20 days
        if cadence == "monthly" and actual_posts >= 2:
            anomaly_level = "HIGH"

        if anomaly_level is None:
            continue

        # Collect topics from this source's recent claims
        recent_topics = set()
        for claim in claims_archive.get("claims", []):
            if claim.get("source_id") != sid:
                continue
            try:
                cd = datetime.strptime(
                    claim.get("content_date", ""), "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                continue
            if cd >= week_ago:
                for t in claim.get("topics", []):
                    recent_topics.add(t)

        anomalies.append({
            "source_id": sid,
            "source_name": src.get("source_name", sid),
            "anomaly_level": anomaly_level,
            "cadence_ratio": round(cadence_ratio, 2),
            "actual_posts_7d": actual_posts,
            "baseline_posts_week": baseline,
            "cadence": cadence,
            "topics": sorted(recent_topics),
            "detected_at": today.isoformat(),
        })

    # Sort by severity: EXTREME > HIGH > ELEVATED
    severity_order = {"EXTREME": 0, "HIGH": 1, "ELEVATED": 2}
    anomalies.sort(key=lambda a: severity_order.get(a["anomaly_level"], 9))

    if anomalies:
        for a in anomalies:
            logger.info(
                f"CADENCE ANOMALY: {a['source_name']} — {a['anomaly_level']} "
                f"({a['actual_posts_7d']} posts in 7d, baseline {a['baseline_posts_week']}/week, "
                f"ratio {a['cadence_ratio']}x) — Topics: {', '.join(a['topics'])}"
            )
    else:
        logger.info("Cadence Anomaly Detection: no anomalies detected")

    return anomalies


# ---------------------------------------------------------------------------
# Source Conviction History (IC V2 Phase 2, Spec Kapitel 9)
# ---------------------------------------------------------------------------
def _load_source_history() -> dict:
    """Load source_history.json from data/history/."""
    if os.path.exists(SOURCE_HISTORY_PATH):
        try:
            return _load_json(SOURCE_HISTORY_PATH)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Source history corrupt, starting fresh: {e}")
    return {"sources": {}, "last_updated": None}


def _save_source_history(history: dict) -> None:
    """Save source_history.json to data/history/."""
    history["last_updated"] = date.today().isoformat()
    _save_json(history, SOURCE_HISTORY_PATH)


def _update_source_history(
    new_claims: list[dict],
    sources_config: list[dict],
    source_history: dict,
) -> dict:
    """Update source conviction history with new extraction data.

    Per IC V2 Spec Kapitel 9: 4-week rolling window per source tracking
    conviction direction, intensity, bias-adjusted signal, temperature,
    and speaker confidence per content_date.

    Detects:
      - conviction_trend: RISING / STABLE / FALLING
      - temperature_trend: RISING / STABLE / FALLING
      - shift_detected: true if direction changed or intensity jumped >=3

    Args:
        new_claims: claims from current extraction (may be empty)
        sources_config: list of source configs from sources.json
        source_history: existing history dict

    Returns:
        updated source_history dict
    """
    today = date.today()
    cutoff = today - timedelta(days=30)

    source_lookup = {s["source_id"]: s for s in sources_config}
    sources_dict = source_history.get("sources", {})

    # Group new claims by source_id + content_date
    new_by_source: dict[str, dict[str, list[dict]]] = {}
    for claim in new_claims:
        sid = claim.get("source_id", "")
        cd = claim.get("content_date", "")
        if not sid or not cd:
            continue
        if sid not in new_by_source:
            new_by_source[sid] = {}
        if cd not in new_by_source[sid]:
            new_by_source[sid][cd] = []
        new_by_source[sid][cd].append(claim)

    # Update history for each source that has new claims
    for sid, date_claims in new_by_source.items():
        if sid not in sources_dict:
            sources_dict[sid] = {"conviction_history": []}

        src_config = source_lookup.get(sid, {})
        known_bias = src_config.get("known_bias", 0)

        for cd, claims_list in date_claims.items():
            # Aggregate claims for this date: use highest-novelty claim
            # for direction/intensity, average for temperature/confidence
            best_novelty = -1
            direction = "NEUTRAL"
            intensity = 5
            temperatures = []
            confidences = []

            for c in claims_list:
                nov = c.get("novelty_score", 0)
                if nov > best_novelty:
                    best_novelty = nov
                    sent = c.get("sentiment", {})
                    direction = sent.get("direction", "NEUTRAL")
                    intensity = sent.get("intensity", 5)

                lt = c.get("linguistic_temperature")
                if isinstance(lt, (int, float)) and lt > 0:
                    temperatures.append(lt)

                sc = c.get("speaker_confidence")
                if isinstance(sc, (int, float)) and sc > 0:
                    confidences.append(sc)

            # Bias-adjusted signal
            signed = intensity if direction == "BULLISH" else (
                -intensity if direction == "BEARISH" else 0
            )
            bias_adjusted = signed - known_bias

            avg_temperature = (
                round(sum(temperatures) / len(temperatures), 1)
                if temperatures else None
            )
            avg_confidence = (
                round(sum(confidences) / len(confidences), 1)
                if confidences else None
            )

            # Check for duplicate entry (same source + date)
            existing_dates = {
                e["date"] for e in sources_dict[sid]["conviction_history"]
            }
            if cd in existing_dates:
                # Update existing entry
                for entry in sources_dict[sid]["conviction_history"]:
                    if entry["date"] == cd:
                        entry["direction"] = direction
                        entry["intensity"] = intensity
                        entry["bias_adjusted"] = bias_adjusted
                        entry["temperature"] = avg_temperature
                        entry["speaker_confidence"] = avg_confidence
                        break
            else:
                sources_dict[sid]["conviction_history"].append({
                    "date": cd,
                    "direction": direction,
                    "intensity": intensity,
                    "bias_adjusted": bias_adjusted,
                    "temperature": avg_temperature,
                    "speaker_confidence": avg_confidence,
                })

    # For ALL sources: prune old entries + compute trends
    for sid in list(sources_dict.keys()):
        history_list = sources_dict[sid].get("conviction_history", [])

        # Remove entries older than 30 days
        history_list = [
            e for e in history_list
            if e.get("date", "") >= cutoff.isoformat()
        ]

        # Sort by date ascending
        history_list.sort(key=lambda e: e.get("date", ""))

        # Compute conviction_trend from last 4 entries
        conviction_trend = "STABLE"
        temperature_trend = "STABLE"
        shift_detected = False
        shift_detail = None
        conviction_4w_delta = 0

        if len(history_list) >= 2:
            # Conviction trend based on bias_adjusted signal
            recent = history_list[-4:] if len(history_list) >= 4 else history_list
            signals = [e.get("bias_adjusted", 0) for e in recent]

            if len(signals) >= 2:
                first_half = sum(signals[:len(signals)//2]) / max(len(signals)//2, 1)
                second_half = sum(signals[len(signals)//2:]) / max(len(signals) - len(signals)//2, 1)
                delta = second_half - first_half
                conviction_4w_delta = round(delta, 1)

                if delta > 1.5:
                    conviction_trend = "RISING"
                elif delta < -1.5:
                    conviction_trend = "FALLING"

            # Temperature trend
            temps = [
                e.get("temperature") for e in recent
                if e.get("temperature") is not None
            ]
            if len(temps) >= 2:
                t_first = sum(temps[:len(temps)//2]) / max(len(temps)//2, 1)
                t_second = sum(temps[len(temps)//2:]) / max(len(temps) - len(temps)//2, 1)
                t_delta = t_second - t_first
                if t_delta > 1.5:
                    temperature_trend = "RISING"
                elif t_delta < -1.5:
                    temperature_trend = "FALLING"

            # Shift detection: direction change or intensity jump >= 3
            if len(history_list) >= 2:
                prev = history_list[-2]
                curr = history_list[-1]
                prev_dir = prev.get("direction", "NEUTRAL")
                curr_dir = curr.get("direction", "NEUTRAL")
                prev_int = prev.get("intensity", 5)
                curr_int = curr.get("intensity", 5)

                if (prev_dir in ("BULLISH", "BEARISH") and
                        curr_dir in ("BULLISH", "BEARISH") and
                        prev_dir != curr_dir):
                    shift_detected = True
                    shift_detail = f"Direction change: {prev_dir} -> {curr_dir}"
                elif abs(curr_int - prev_int) >= 3:
                    shift_detected = True
                    shift_detail = (
                        f"Intensity jump: {prev_int} -> {curr_int} "
                        f"(delta {curr_int - prev_int:+d})"
                    )

        sources_dict[sid]["conviction_history"] = history_list
        sources_dict[sid]["conviction_trend"] = conviction_trend
        sources_dict[sid]["conviction_4w_delta"] = conviction_4w_delta
        sources_dict[sid]["temperature_trend"] = temperature_trend
        sources_dict[sid]["shift_detected"] = shift_detected
        sources_dict[sid]["shift_detail"] = shift_detail
        sources_dict[sid]["entry_count"] = len(history_list)

    source_history["sources"] = sources_dict

    # Log summary
    trends = {}
    shifts = []
    for sid, data in sources_dict.items():
        ct = data.get("conviction_trend", "STABLE")
        if ct != "STABLE":
            trends[sid] = ct
        if data.get("shift_detected"):
            shifts.append(f"{sid}: {data.get('shift_detail', '')}")

    total_entries = sum(d.get("entry_count", 0) for d in sources_dict.values())
    logger.info(
        f"Source Conviction History: {len(sources_dict)} sources, "
        f"{total_entries} total entries (30d window)"
    )
    if trends:
        logger.info(f"Conviction trends: {trends}")
    if shifts:
        for s in shifts:
            logger.info(f"SHIFT DETECTED: {s}")

    return source_history


# ---------------------------------------------------------------------------
# Narrative Threads (IC V2 Phase 2)
# ---------------------------------------------------------------------------
THREAD_MATCH_THRESHOLD = 0.55
THREAD_SEED_MIN_NOVELTY = 7
THREAD_SEED_MIN_EXPERTISE = 7
THREAD_FADING_DAYS = 10
THREAD_FADING_DAYS_THREATENING = 17
THREAD_ARCHIVED_DAYS = 21
THREAD_ARCHIVED_DAYS_THREATENING = 28
THREAD_SOURCE_DIVERSITY_CAP = 4


def _load_threads() -> dict:
    """Load threads.json from data/history/."""
    if os.path.exists(THREADS_PATH):
        try:
            return _load_json(THREADS_PATH)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Threads file corrupt, starting fresh: {e}")
    return {"active_threads": [], "archived_threads": [], "last_updated": None}


def _save_threads(threads_data: dict) -> None:
    """Save threads.json to data/history/."""
    threads_data["last_updated"] = date.today().isoformat()
    _save_json(threads_data, THREADS_PATH)


def _generate_thread_id(primary_topic: str, today_str: str, existing_ids: set) -> str:
    """Generate unique thread ID: THR_{TOPIC}_{YYYYMMDD}_{seq}."""
    date_part = today_str.replace("-", "")
    seq = 1
    while True:
        tid = f"THR_{primary_topic}_{date_part}_{seq:03d}"
        if tid not in existing_ids:
            return tid
        seq += 1


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compute_thread_match_score(thesis: dict, thread: dict) -> float:
    """Compute deterministic match score between a thesis and a thread.

    Weights:
      0.35 Topic Overlap (Jaccard)
      0.30 Direction Alignment (same=1.0, NEUTRAL=0.5, opposite=0.0)
      0.35 Asset Overlap (Jaccard on affected_assets)

    Returns: float 0.0-1.0
    """
    # Topic overlap
    thesis_topics = set(thesis.get("topics", []))
    thread_topics = set(thread.get("topics", []))
    topic_score = _jaccard(thesis_topics, thread_topics)

    # Direction alignment
    thesis_dir = thesis.get("sentiment", {}).get("direction", "NEUTRAL")
    thread_dir = thread.get("direction", "NEUTRAL")

    if thesis_dir == thread_dir:
        dir_score = 1.0
    elif thesis_dir == "NEUTRAL" or thread_dir == "NEUTRAL":
        dir_score = 0.5
    elif thesis_dir == "MIXED" or thread_dir == "MIXED":
        dir_score = 0.5
    else:
        # Opposite directions (BULLISH vs BEARISH)
        dir_score = 0.0

    # Asset overlap
    thesis_assets = set()
    for aa in thesis.get("affected_assets", []):
        if isinstance(aa, dict):
            thesis_assets.add(aa.get("asset", ""))
        elif isinstance(aa, str):
            thesis_assets.add(aa)
    thesis_assets.discard("")

    thread_assets = set(thread.get("affected_assets", []))
    asset_score = _jaccard(thesis_assets, thread_assets)

    return 0.35 * topic_score + 0.30 * dir_score + 0.35 * asset_score


def _get_thesis_assets(thesis: dict) -> list[str]:
    """Extract asset tickers from thesis affected_assets (handles V1+V2 format)."""
    assets = set()
    for aa in thesis.get("affected_assets", []):
        if isinstance(aa, dict):
            a = aa.get("asset", "")
            if a:
                assets.add(a.upper())
        elif isinstance(aa, str) and aa:
            assets.add(aa.upper())
    return sorted(assets)


def _get_v16_assets(v16_context: dict | None) -> set[str]:
    """Get set of active V16 position tickers."""
    if not v16_context:
        return set()
    weights = v16_context.get("current_weights", {})
    return {
        asset.upper() for asset, w in weights.items()
        if isinstance(w, (int, float)) and w > 0.005
    }


def _thesis_qualifies_for_seed(
    thesis: dict,
    source_config: dict,
    expertise_matrix: dict,
    v16_assets: set[str],
) -> bool:
    """Check if a thesis qualifies to create a new SEED thread.

    Requirements (ALL must be met):
      1. novelty_score >= 7
      2. Source is CORE tier OR has expertise >= 7 in primary_topic
      3. At least 1 affected_asset is a current V16 position
    """
    # Requirement 1: Novelty
    if thesis.get("novelty_score", 0) < THREAD_SEED_MIN_NOVELTY:
        return False

    # Requirement 2: Source quality
    tier = source_config.get("tier", "")
    source_id = thesis.get("source_id", "")
    primary_topic = thesis.get("primary_topic", thesis.get("topics", [""])[0])

    source_expertise = (
        expertise_matrix.get("expertise", {})
        .get(source_id, {})
        .get(primary_topic, 0)
    )

    if tier != "CORE" and source_expertise < THREAD_SEED_MIN_EXPERTISE:
        return False

    # Requirement 3: V16 asset overlap
    thesis_assets = set(_get_thesis_assets(thesis))
    if not thesis_assets & v16_assets:
        return False

    return True


def _create_seed_thread(
    thesis: dict,
    source_config: dict,
    expertise_matrix: dict,
    today_str: str,
    existing_ids: set,
) -> dict:
    """Create a new SEED thread from a qualifying thesis."""
    source_id = thesis.get("source_id", "")
    primary_topic = thesis.get("primary_topic", thesis.get("topics", [""])[0])
    thesis_assets = _get_thesis_assets(thesis)
    direction = thesis.get("sentiment", {}).get("direction", "NEUTRAL")

    thread_id = _generate_thread_id(primary_topic, today_str, existing_ids)

    # core_hypothesis = claim_text (deterministic; later LLM-generated)
    core_hypothesis = thesis.get("claim_text", "")[:300]

    # Get expertise score for conviction
    source_expertise = (
        expertise_matrix.get("expertise", {})
        .get(source_id, {})
        .get(primary_topic, 5)
    )

    return {
        "thread_id": thread_id,
        "core_hypothesis": core_hypothesis,
        "status": "SEED",
        "challenged": False,
        "challenger_detail": None,
        "created_at": today_str,
        "last_evidence_date": thesis.get("content_date", today_str),
        "topics": list(thesis.get("topics", [primary_topic])),
        "direction": direction,
        "sources": [source_id],
        "source_count": 1,
        "evidence": [
            {
                "thesis_id": thesis.get("id", ""),
                "source_id": source_id,
                "claim_text": thesis.get("claim_text", "")[:200],
                "content_date": thesis.get("content_date", today_str),
                "direction": direction,
                "novelty_score": thesis.get("novelty_score", 0),
                "speaker_confidence": thesis.get("speaker_confidence", 5),
            }
        ],
        "affected_assets": thesis_assets,
        "max_expertise": source_expertise,
        "max_expertise_source": source_id,
        "conviction": 0.0,  # Computed after creation
        "conviction_components": {},
        "portfolio_alignment": "NEUTRAL",
        "portfolio_detail": "",
        "portfolio_relevance_score": 0.0,
        "threatened_positions": [],
    }


def _add_evidence_to_thread(thread: dict, thesis: dict, expertise_matrix: dict) -> dict:
    """Add a thesis as evidence to an existing thread."""
    source_id = thesis.get("source_id", "")
    direction = thesis.get("sentiment", {}).get("direction", "NEUTRAL")
    content_date = thesis.get("content_date", "")

    # Add evidence entry
    thread["evidence"].append({
        "thesis_id": thesis.get("id", ""),
        "source_id": source_id,
        "claim_text": thesis.get("claim_text", "")[:200],
        "content_date": content_date,
        "direction": direction,
        "novelty_score": thesis.get("novelty_score", 0),
        "speaker_confidence": thesis.get("speaker_confidence", 5),
    })

    # Update last_evidence_date
    if content_date > thread.get("last_evidence_date", ""):
        thread["last_evidence_date"] = content_date

    # Update sources list (unique)
    if source_id not in thread["sources"]:
        thread["sources"].append(source_id)
    thread["source_count"] = len(thread["sources"])

    # Update topics (union)
    for t in thesis.get("topics", []):
        if t not in thread["topics"]:
            thread["topics"].append(t)

    # Update affected_assets (union)
    for a in _get_thesis_assets(thesis):
        if a not in thread["affected_assets"]:
            thread["affected_assets"].append(a)

    # Update max_expertise
    primary_topic = thread["topics"][0] if thread["topics"] else ""
    source_expertise = (
        expertise_matrix.get("expertise", {})
        .get(source_id, {})
        .get(primary_topic, 0)
    )
    if source_expertise > thread.get("max_expertise", 0):
        thread["max_expertise"] = source_expertise
        thread["max_expertise_source"] = source_id

    # Check for challenge: opposite direction from expert
    if direction in ("BULLISH", "BEARISH") and thread["direction"] in ("BULLISH", "BEARISH"):
        if direction != thread["direction"] and source_expertise >= 6:
            thread["challenged"] = True
            thread["challenger_detail"] = (
                f"{source_id} (expertise {source_expertise}) says {direction} "
                f"vs thread direction {thread['direction']}"
            )

    return thread


def _compute_thread_conviction(
    thread: dict,
    expertise_matrix: dict,
    source_history: dict,
    v16_context: dict | None,
    taxonomy: dict | None,
) -> float:
    """Compute thread conviction score (0.0-10.0).

    Formula:
      0.35 x max_single_expertise (normalized /10)
      0.20 x source_diversity (unique_sources / 4, capped 1.0)
      0.20 x data_confirmation (1.0 V16-aligned, 0.5 neutral, 0.0 contra)
      0.15 x freshness (avg decay_weight of evidence)
      0.10 x conviction_trend (from source_history: RISING=1.0, STABLE=0.5, FALLING=0.0)

    Returns: float 0.0-10.0
    """
    # 1. Max single expertise (already tracked on thread)
    max_exp = thread.get("max_expertise", 5)
    exp_score = max_exp / 10.0  # 0.0-1.0

    # 2. Source diversity (capped at 4)
    source_count = thread.get("source_count", 1)
    diversity_score = min(source_count / THREAD_SOURCE_DIVERSITY_CAP, 1.0)

    # 3. Data confirmation (V16 alignment)
    data_score = 0.5  # default: neutral
    if v16_context and taxonomy:
        thread_topics = thread.get("topics", [])
        thread_dir = thread.get("direction", "NEUTRAL")
        topic_to_layers = taxonomy.get("topic_to_layers", {})

        # Check if V16 regime aligns with thread direction
        # Simple heuristic: if thread is BULLISH and touches V16-related topics,
        # check if V16 has corresponding positions
        v16_weights = v16_context.get("current_weights", {})
        thread_assets = set(thread.get("affected_assets", []))
        v16_assets = {
            a.upper() for a, w in v16_weights.items()
            if isinstance(w, (int, float)) and w > 0.005
        }

        overlap = thread_assets & v16_assets
        if overlap:
            # V16 has positions in assets the thread discusses
            # If thread says BULLISH and V16 is long -> CONFIRMING
            if thread_dir in ("BULLISH", "NEUTRAL"):
                data_score = 0.8
            elif thread_dir == "BEARISH":
                # Thread bearish but V16 is long -> contradicting
                data_score = 0.2
            else:
                data_score = 0.5
        # No overlap = neutral

    # 4. Freshness (avg decay_weight of evidence)
    evidence = thread.get("evidence", [])
    if evidence:
        today = date.today()
        weights = []
        for e in evidence:
            f = _compute_freshness(e.get("content_date", ""), today)
            weights.append(f["decay_weight"])
        freshness_score = sum(weights) / len(weights) if weights else 0.5
    else:
        freshness_score = 0.0

    # 5. Conviction trend (from source_history for thread's primary source)
    trend_score = 0.5  # default: STABLE
    sources_data = source_history.get("sources", {})
    # Use the max_expertise source's conviction trend
    primary_source = thread.get("max_expertise_source", "")
    if primary_source in sources_data:
        ct = sources_data[primary_source].get("conviction_trend", "STABLE")
        if ct == "RISING":
            trend_score = 1.0
        elif ct == "FALLING":
            trend_score = 0.0

    # Weighted combination -> scale to 0-10
    raw = (
        0.35 * exp_score
        + 0.20 * diversity_score
        + 0.20 * data_score
        + 0.15 * freshness_score
        + 0.10 * trend_score
    )

    conviction = round(raw * 10.0, 1)
    conviction = max(0.0, min(10.0, conviction))

    # Store components for transparency
    thread["conviction_components"] = {
        "max_expertise": max_exp,
        "max_expertise_source": primary_source,
        "source_diversity": round(diversity_score, 2),
        "data_confirmation": round(data_score, 2),
        "freshness": round(freshness_score, 2),
        "conviction_trend": trend_score,
    }
    thread["conviction"] = conviction

    return conviction


def _compute_portfolio_alignment(
    thread: dict,
    v16_context: dict | None,
) -> dict:
    """Compute portfolio alignment and relevance score for a thread.

    Categories:
      CONFIRMING:  Thread direction supports V16 position direction
      THREATENING: Thread direction opposes V16 position
      MIXED:       Some positions confirmed, some threatened
      OPPORTUNITY: Thread affects asset NOT in V16 portfolio
      NEUTRAL:     No overlap between thread assets and V16 positions

    portfolio_relevance_score = SUM(position_weight_pct * severity)
      THREATENING=1.0, MIXED=0.5, CONFIRMING=0.3, NEUTRAL/OPPORTUNITY=0.0

    Returns: dict with alignment, detail, relevance_score, threatened_positions
    """
    if not v16_context:
        return {
            "portfolio_alignment": "NEUTRAL",
            "portfolio_detail": "No V16 context available",
            "portfolio_relevance_score": 0.0,
            "threatened_positions": [],
        }

    weights = v16_context.get("current_weights", {})
    v16_positions = {
        a.upper(): w for a, w in weights.items()
        if isinstance(w, (int, float)) and w > 0.005
    }

    thread_assets = set(thread.get("affected_assets", []))
    thread_dir = thread.get("direction", "NEUTRAL")

    if not thread_assets or not v16_positions:
        return {
            "portfolio_alignment": "NEUTRAL",
            "portfolio_detail": "No asset overlap or no V16 positions",
            "portfolio_relevance_score": 0.0,
            "threatened_positions": [],
        }

    overlap = thread_assets & set(v16_positions.keys())
    non_overlap = thread_assets - set(v16_positions.keys())

    if not overlap:
        if non_overlap:
            return {
                "portfolio_alignment": "OPPORTUNITY",
                "portfolio_detail": (
                    f"Thread affects {', '.join(sorted(non_overlap))} "
                    f"— not in V16 portfolio"
                ),
                "portfolio_relevance_score": 0.0,
                "threatened_positions": [],
            }
        return {
            "portfolio_alignment": "NEUTRAL",
            "portfolio_detail": "No relevance to current portfolio",
            "portfolio_relevance_score": 0.0,
            "threatened_positions": [],
        }

    # For each overlapping position, determine alignment
    confirmed = []
    threatened = []
    relevance_score = 0.0

    for asset in sorted(overlap):
        weight_pct = round(v16_positions[asset] * 100, 1)

        # V16 is LONG all positions (weight > 0)
        # Thread BULLISH on asset = CONFIRMING (supports long position)
        # Thread BEARISH on asset = THREATENING (opposes long position)
        # Check per-asset direction from evidence if available
        asset_dir = thread_dir  # fallback to thread-level direction

        # Try to find asset-specific direction from affected_assets in evidence
        for ev in thread.get("evidence", []):
            # This is a simplification; full per-asset tracking in Phase B
            pass

        if asset_dir == "BULLISH":
            confirmed.append(f"{asset} {weight_pct}%")
            relevance_score += weight_pct * 0.3
        elif asset_dir == "BEARISH":
            threatened.append({"asset": asset, "weight_pct": weight_pct})
            relevance_score += weight_pct * 1.0
        else:
            relevance_score += weight_pct * 0.1

    # Determine overall alignment
    if threatened and confirmed:
        alignment = "MIXED"
    elif threatened:
        alignment = "THREATENING"
    elif confirmed:
        alignment = "CONFIRMING"
    else:
        alignment = "NEUTRAL"

    detail_parts = []
    if confirmed:
        detail_parts.append(f"Confirms: {', '.join(confirmed)}")
    if threatened:
        threat_strs = [f"{t['asset']} {t['weight_pct']}%" for t in threatened]
        detail_parts.append(f"Threatens: {', '.join(threat_strs)}")
    if non_overlap:
        detail_parts.append(f"Opportunity: {', '.join(sorted(non_overlap))}")

    return {
        "portfolio_alignment": alignment,
        "portfolio_detail": "; ".join(detail_parts) if detail_parts else "",
        "portfolio_relevance_score": round(relevance_score, 1),
        "threatened_positions": threatened,
    }


def _update_thread_lifecycle(thread: dict, today: date) -> str:
    """Update thread lifecycle status based on evidence and time.

    Lifecycle: SEED -> BUILDING -> ESTABLISHED -> FADING -> ARCHIVED
    challenged is a flag, not a status.

    Transitions:
      SEED -> BUILDING: 2+ unique sources OR same source 2+ weeks
      BUILDING -> ESTABLISHED: 3+ unique sources OR data confirmation
      any -> FADING: no new evidence for 10d (17d if THREATENING)
      FADING -> ARCHIVED: no new evidence for 21d (28d if THREATENING)

    Returns: new status string
    """
    old_status = thread.get("status", "SEED")

    # If already ARCHIVED, stay ARCHIVED
    if old_status == "ARCHIVED":
        return "ARCHIVED"

    last_evidence_str = thread.get("last_evidence_date", "")
    try:
        last_evidence = datetime.strptime(last_evidence_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        last_evidence = today

    days_silent = (today - last_evidence).days

    # Determine decay thresholds based on portfolio alignment
    is_threatening = thread.get("portfolio_alignment") == "THREATENING"
    fading_threshold = (
        THREAD_FADING_DAYS_THREATENING if is_threatening
        else THREAD_FADING_DAYS
    )
    archived_threshold = (
        THREAD_ARCHIVED_DAYS_THREATENING if is_threatening
        else THREAD_ARCHIVED_DAYS
    )

    # Check for ARCHIVED first (takes precedence)
    if days_silent >= archived_threshold:
        thread["status"] = "ARCHIVED"
        return "ARCHIVED"

    # Check for FADING
    if days_silent >= fading_threshold:
        thread["status"] = "FADING"
        return "FADING"

    # Forward transitions (only if not fading)
    source_count = thread.get("source_count", 1)
    data_conf = thread.get("conviction_components", {}).get("data_confirmation", 0.5)

    if old_status == "SEED":
        # SEED -> BUILDING: 2+ unique sources
        if source_count >= 2:
            thread["status"] = "BUILDING"
            return "BUILDING"
        # OR same source posting about this over 2+ weeks
        evidence = thread.get("evidence", [])
        if len(evidence) >= 2:
            dates = sorted(set(e.get("content_date", "") for e in evidence))
            if len(dates) >= 2:
                try:
                    first = datetime.strptime(dates[0], "%Y-%m-%d").date()
                    last = datetime.strptime(dates[-1], "%Y-%m-%d").date()
                    if (last - first).days >= 14:
                        thread["status"] = "BUILDING"
                        return "BUILDING"
                except (ValueError, TypeError):
                    pass

    if old_status in ("SEED", "BUILDING"):
        # -> ESTABLISHED: 3+ unique sources OR strong data confirmation
        if source_count >= 3 or data_conf >= 0.8:
            thread["status"] = "ESTABLISHED"
            return "ESTABLISHED"

    # No transition — keep current status
    return old_status


def _update_threads(
    new_claims: list[dict],
    threads_data: dict,
    v16_context: dict | None,
    expertise_matrix: dict,
    sources_config: list[dict],
    source_history: dict,
    taxonomy: dict | None,
) -> dict:
    """Orchestrate narrative thread updates for this run.

    Steps:
      1. For each new thesis: match against active threads
      2. If match: add evidence, update conviction + lifecycle
      3. If no match + seed criteria met: create new SEED thread
      4. For all threads: lifecycle decay check (FADING, ARCHIVED)
      5. Recompute portfolio alignment for all active threads
      6. Move ARCHIVED threads to archived_threads list

    Returns: updated threads_data dict
    """
    today = date.today()
    today_str = today.isoformat()

    active_threads = threads_data.get("active_threads", [])
    archived_threads = threads_data.get("archived_threads", [])

    source_lookup = {s["source_id"]: s for s in sources_config}
    v16_assets = _get_v16_assets(v16_context)

    # Collect all existing thread IDs for unique ID generation
    all_ids = {t["thread_id"] for t in active_threads}
    all_ids.update(t["thread_id"] for t in archived_threads)

    matched_count = 0
    new_thread_count = 0
    unmatched_count = 0

    for thesis in new_claims:
        # Only process FRESH claims (just extracted)
        if thesis.get("freshness", "FRESH") not in ("FRESH",):
            continue

        # Find best matching thread
        best_score = 0.0
        best_thread_idx = -1

        for i, thread in enumerate(active_threads):
            if thread.get("status") == "ARCHIVED":
                continue
            score = _compute_thread_match_score(thesis, thread)
            if score > best_score:
                best_score = score
                best_thread_idx = i

        if best_score >= THREAD_MATCH_THRESHOLD and best_thread_idx >= 0:
            # Match found — add evidence to existing thread
            active_threads[best_thread_idx] = _add_evidence_to_thread(
                active_threads[best_thread_idx], thesis, expertise_matrix
            )
            matched_count += 1
            logger.debug(
                f"Thread match: {thesis.get('id', '')} -> "
                f"{active_threads[best_thread_idx]['thread_id']} "
                f"(score {best_score:.2f})"
            )
        else:
            # No match — check if this thesis qualifies for a new thread
            src_config = source_lookup.get(thesis.get("source_id", ""), {})
            if _thesis_qualifies_for_seed(
                thesis, src_config, expertise_matrix, v16_assets
            ):
                new_thread = _create_seed_thread(
                    thesis, src_config, expertise_matrix, today_str, all_ids
                )
                active_threads.append(new_thread)
                all_ids.add(new_thread["thread_id"])
                new_thread_count += 1
                logger.info(
                    f"NEW THREAD: {new_thread['thread_id']} — "
                    f"'{new_thread['core_hypothesis'][:80]}...'"
                )
            else:
                unmatched_count += 1

    # Recompute conviction and portfolio alignment for ALL active threads
    newly_archived = []
    surviving_active = []

    for thread in active_threads:
        # Compute conviction
        _compute_thread_conviction(
            thread, expertise_matrix, source_history, v16_context, taxonomy
        )

        # Compute portfolio alignment
        pa = _compute_portfolio_alignment(thread, v16_context)
        thread["portfolio_alignment"] = pa["portfolio_alignment"]
        thread["portfolio_detail"] = pa["portfolio_detail"]
        thread["portfolio_relevance_score"] = pa["portfolio_relevance_score"]
        thread["threatened_positions"] = pa["threatened_positions"]

        # Update lifecycle (needs portfolio_alignment for decay thresholds)
        new_status = _update_thread_lifecycle(thread, today)

        if new_status == "ARCHIVED":
            newly_archived.append(thread)
        else:
            surviving_active.append(thread)

    # Move archived threads
    archived_threads.extend(newly_archived)

    # Cap archived_threads to last 50 (prevent unbounded growth)
    if len(archived_threads) > 50:
        archived_threads = archived_threads[-50:]

    # Sort active threads by portfolio_relevance_score descending
    surviving_active.sort(
        key=lambda t: t.get("portfolio_relevance_score", 0), reverse=True
    )

    threads_data["active_threads"] = surviving_active
    threads_data["archived_threads"] = archived_threads

    # Log summary
    status_counts = {}
    for t in surviving_active:
        s = t.get("status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1

    challenged = [t["thread_id"] for t in surviving_active if t.get("challenged")]
    threatening = [
        t["thread_id"] for t in surviving_active
        if t.get("portfolio_alignment") == "THREATENING"
    ]

    logger.info(
        f"Narrative Threads: {len(surviving_active)} active "
        f"({status_counts}), {len(newly_archived)} newly archived"
    )
    logger.info(
        f"Thread matching: {matched_count} matched, {new_thread_count} new seeds, "
        f"{unmatched_count} unmatched (below seed threshold)"
    )
    if challenged:
        logger.warning(f"CHALLENGED threads: {', '.join(challenged)}")
    if threatening:
        logger.warning(f"THREATENING threads: {', '.join(threatening)}")

    return threads_data


# ---------------------------------------------------------------------------
# Position Pre-Mortems (IC V2 Phase 2, Spec Kapitel 14)
# ---------------------------------------------------------------------------
PRE_MORTEM_POSITION_THRESHOLD = 0.10  # Only positions >10% weight
PRE_MORTEM_REGEN_DAYS = 7  # Regenerate weekly
PRE_MORTEM_WEIGHT_CHANGE_TRIGGER = 0.05  # Regenerate if weight changed >5%
PRE_MORTEM_MODEL = "claude-sonnet-4-6"


def _load_pre_mortems() -> dict:
    """Load pre_mortems.json from data/history/."""
    if os.path.exists(PRE_MORTEMS_PATH):
        try:
            return _load_json(PRE_MORTEMS_PATH)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Pre-mortems file corrupt, starting fresh: {e}")
    return {"positions": {}, "last_updated": None, "last_generated": None}


def _save_pre_mortems(pm_data: dict) -> None:
    """Save pre_mortems.json to data/history/."""
    pm_data["last_updated"] = date.today().isoformat()
    _save_json(pm_data, PRE_MORTEMS_PATH)


def _needs_regeneration(
    pm_data: dict,
    asset: str,
    current_weight: float,
    today: date,
) -> bool:
    """Check if pre-mortem for this asset needs LLM regeneration.

    Triggers:
      1. No existing pre-mortem for this asset
      2. Last generation >7 days ago
      3. V16 weight changed by >5% since last generation
      4. A THREATENING thread was created (checked by caller)
    """
    pos_data = pm_data.get("positions", {}).get(asset)
    if not pos_data:
        return True

    # Check age
    last_gen = pos_data.get("generated_at", "")
    if last_gen:
        try:
            gen_date = datetime.strptime(last_gen, "%Y-%m-%d").date()
            if (today - gen_date).days >= PRE_MORTEM_REGEN_DAYS:
                return True
        except (ValueError, TypeError):
            return True
    else:
        return True

    # Check weight change
    prev_weight = pos_data.get("v16_weight", 0)
    if abs(current_weight - prev_weight) > PRE_MORTEM_WEIGHT_CHANGE_TRIGGER:
        return True

    return False


def _build_pre_mortem_claims_text(
    asset: str,
    claims_archive: dict,
) -> str:
    """Build claims text relevant to an asset for pre-mortem prompt."""
    relevant_claims = []
    asset_upper = asset.upper()

    for claim in claims_archive.get("claims", []):
        if claim.get("freshness") not in ("FRESH", "AGING", "FADING"):
            continue

        # Check if claim mentions this asset in affected_assets
        claim_assets = set()
        for aa in claim.get("affected_assets", []):
            if isinstance(aa, dict):
                claim_assets.add(aa.get("asset", "").upper())
            elif isinstance(aa, str):
                claim_assets.add(aa.upper())

        # Also check v16_position_impact
        for vpi in claim.get("v16_position_impact", []):
            if isinstance(vpi, dict):
                claim_assets.add(vpi.get("position", "").upper())

        # Also check system_relevance.affected_assets
        sr = claim.get("system_relevance", {})
        for a in sr.get("affected_assets", []):
            claim_assets.add(a.upper())

        if asset_upper not in claim_assets:
            continue

        source = claim.get("source_id", "unknown")
        text = claim.get("claim_text", "")[:200]
        direction = claim.get("sentiment", {}).get("direction", "")
        novelty = claim.get("novelty_score", 0)
        relevant_claims.append({
            "source": source,
            "text": text,
            "direction": direction,
            "novelty": novelty,
        })

    # Sort by novelty descending, take top 10
    relevant_claims.sort(key=lambda c: c["novelty"], reverse=True)
    relevant_claims = relevant_claims[:10]

    if not relevant_claims:
        return "No IC claims directly reference this asset in the last 7 days."

    lines = []
    for c in relevant_claims:
        lines.append(
            f"- [{c['source']}] ({c['direction']}, novelty {c['novelty']}): "
            f"{c['text']}"
        )
    return "\n".join(lines)


def _build_pre_mortem_threads_text(
    asset: str,
    threads_data: dict,
) -> str:
    """Build threads text relevant to an asset for pre-mortem prompt."""
    if not threads_data:
        return "No active narrative threads."

    relevant = []
    asset_upper = asset.upper()

    for thread in threads_data.get("active_threads", []):
        if asset_upper in [a.upper() for a in thread.get("affected_assets", [])]:
            relevant.append(thread)

    if not relevant:
        return "No active threads directly reference this asset."

    lines = []
    for t in relevant:
        alignment = t.get("portfolio_alignment", "NEUTRAL")
        lines.append(
            f"- [{t['thread_id']}] \"{t.get('core_hypothesis', '')[:100]}\" "
            f"| Status: {t.get('status', '?')} | Conviction: {t.get('conviction', 0)} "
            f"| Alignment: {alignment} | Sources: {', '.join(t.get('sources', []))}"
        )
    return "\n".join(lines)


def _generate_pre_mortem_llm(
    asset: str,
    weight_pct: float,
    regime: str,
    portfolio_text: str,
    claims_text: str,
    threads_text: str,
    previous_pm_text: str,
) -> list[dict]:
    """Call LLM to generate failure scenarios for a position.

    Returns list of scenario dicts, or empty list on failure.
    """
    import anthropic

    prompts_dir = os.path.join(BASE_DIR, "src", "extraction", "prompts")

    try:
        with open(os.path.join(prompts_dir, "pre_mortem_system.txt"), "r") as f:
            system_prompt = f.read()
        with open(os.path.join(prompts_dir, "pre_mortem_user.txt"), "r") as f:
            user_template = f.read()
    except FileNotFoundError as e:
        logger.error(f"Pre-mortem prompt file not found: {e}")
        return []

    user_prompt = user_template.format(
        asset=asset,
        weight_pct=weight_pct,
        regime=regime,
        portfolio_text=portfolio_text,
        claims_text=claims_text,
        threads_text=threads_text,
        previous_pm_text=previous_pm_text,
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=PRE_MORTEM_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = response.content[0].text.strip()

        # Parse JSON (reuse pattern from extractor)
        import re
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```\s*$", "", raw_text)

        bracket_pos = raw_text.find("[")
        if bracket_pos == -1:
            logger.error(f"[PM:{asset}] No JSON array in LLM response")
            return []

        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(raw_text, bracket_pos)

        if not isinstance(parsed, list):
            logger.error(f"[PM:{asset}] LLM returned non-list: {type(parsed)}")
            return []

        # Validate scenarios
        valid_categories = {
            "MACRO_REGIME_SHIFT", "NARRATIVE_COLLAPSE",
            "LIQUIDITY_FLOW", "EXTERNAL_SHOCK", "CORRELATION_BREAK",
        }
        valid_prob = {"LOW", "MEDIUM", "HIGH"}
        valid_horizon = {"IMMEDIATE", "SHORT", "MEDIUM", "LONG"}

        validated = []
        for i, sc in enumerate(parsed[:4]):  # max 4
            if not isinstance(sc, dict):
                continue
            if not sc.get("description"):
                continue

            # Normalize fields
            sc["scenario_id"] = f"PM_{asset}_{i + 1:03d}"
            cat = str(sc.get("failure_category", "")).upper()
            if cat not in valid_categories:
                cat = "MACRO_REGIME_SHIFT"
            sc["failure_category"] = cat

            prob = str(sc.get("probability_label", "MEDIUM")).upper()
            if prob not in valid_prob:
                prob = "MEDIUM"
            sc["probability_label"] = prob

            horizon = str(sc.get("time_horizon", "MEDIUM")).upper()
            if horizon not in valid_horizon:
                horizon = "MEDIUM"
            sc["time_horizon"] = horizon

            sc["reasoning"] = str(sc.get("reasoning", ""))[:500]
            sc["early_warning_indicator"] = str(
                sc.get("early_warning_indicator", "")
            )[:300]
            sc["early_warning_source"] = str(
                sc.get("early_warning_source", "")
            )[:200]
            sc["portfolio_impact"] = str(sc.get("portfolio_impact", ""))[:100]

            # Ensure evidence arrays
            if not isinstance(sc.get("ic_evidence_for"), list):
                sc["ic_evidence_for"] = []
            sc["ic_evidence_for"] = [
                str(e)[:200] for e in sc["ic_evidence_for"][:5]
            ]
            if not isinstance(sc.get("ic_evidence_against"), list):
                sc["ic_evidence_against"] = []
            sc["ic_evidence_against"] = [
                str(e)[:200] for e in sc["ic_evidence_against"][:5]
            ]

            validated.append(sc)

        logger.info(
            f"[PM:{asset}] LLM generated {len(validated)} failure scenarios"
        )
        return validated

    except Exception as e:
        logger.error(f"[PM:{asset}] LLM call failed: {e}")
        return []


def _update_evidence_deterministic(
    pm_position: dict,
    claims_archive: dict,
    asset: str,
) -> dict:
    """Deterministically update ic_evidence_for/against from new claims.

    This runs daily (no LLM). It scans active claims for the asset and
    checks if they support or contradict any existing failure scenario.
    Simple heuristic: BEARISH claims on the asset = evidence_for failure,
    BULLISH claims = evidence_against failure.
    """
    asset_upper = asset.upper()
    today_str = date.today().isoformat()

    # Collect today's relevant claims
    todays_claims = []
    for claim in claims_archive.get("claims", []):
        if claim.get("freshness") != "FRESH":
            continue
        if claim.get("extraction_date", "") != today_str:
            continue

        # Check if claim mentions this asset
        claim_assets = set()
        for aa in claim.get("affected_assets", []):
            if isinstance(aa, dict):
                claim_assets.add(aa.get("asset", "").upper())
        for vpi in claim.get("v16_position_impact", []):
            if isinstance(vpi, dict):
                claim_assets.add(vpi.get("position", "").upper())

        if asset_upper in claim_assets:
            todays_claims.append(claim)

    if not todays_claims:
        return pm_position

    # For each scenario, add relevant claims as evidence
    for scenario in pm_position.get("failure_scenarios", []):
        for claim in todays_claims:
            source = claim.get("source_id", "unknown")
            direction = claim.get("sentiment", {}).get("direction", "")
            text = claim.get("claim_text", "")[:150]
            evidence_str = f"{source}: {text}"

            if direction == "BEARISH":
                # Bearish on asset = supports the failure scenario
                if evidence_str not in scenario.get("ic_evidence_for", []):
                    scenario.setdefault("ic_evidence_for", []).append(
                        evidence_str
                    )
            elif direction == "BULLISH":
                # Bullish on asset = argues against failure
                if evidence_str not in scenario.get("ic_evidence_against", []):
                    scenario.setdefault("ic_evidence_against", []).append(
                        evidence_str
                    )

    pm_position["evidence_last_updated"] = today_str
    return pm_position


def _run_pre_mortems(
    v16_context: dict | None,
    claims_archive: dict,
    threads_data: dict | None,
    pm_data: dict,
) -> dict:
    """Orchestrate pre-mortem generation and updates.

    Daily:
      - Identify V16 positions >10%
      - For positions needing regeneration: LLM call
      - For all positions: deterministic evidence update
      - Remove pre-mortems for positions no longer >10%

    Returns: updated pm_data dict
    """
    if not v16_context:
        logger.info("Pre-Mortems: No V16 context — skipping")
        return pm_data

    today = date.today()
    today_str = today.isoformat()
    regime = v16_context.get("regime", "UNKNOWN")
    weights = v16_context.get("current_weights", {})

    # Find positions >10%
    large_positions = {
        asset.upper(): w
        for asset, w in weights.items()
        if isinstance(w, (int, float)) and w > PRE_MORTEM_POSITION_THRESHOLD
    }

    if not large_positions:
        logger.info("Pre-Mortems: No positions >10% — skipping")
        return pm_data

    # Build portfolio text for prompt
    portfolio_lines = []
    for asset, w in sorted(large_positions.items(), key=lambda x: -x[1]):
        portfolio_lines.append(f"  {asset}: {round(w * 100, 1)}%")
    portfolio_text = "\n".join(portfolio_lines)

    positions_dict = pm_data.get("positions", {})
    generated_count = 0
    updated_count = 0

    # Check for new THREATENING threads (trigger for regeneration)
    has_new_threatening = False
    if threads_data:
        for t in threads_data.get("active_threads", []):
            if t.get("portfolio_alignment") == "THREATENING":
                # Check if thread was recently created or updated
                last_ev = t.get("last_evidence_date", "")
                if last_ev == today_str:
                    has_new_threatening = True
                    break

    for asset, weight in large_positions.items():
        weight_pct = round(weight * 100, 1)

        needs_regen = _needs_regeneration(pm_data, asset, weight, today)

        # Also regenerate if new threatening thread appeared
        if has_new_threatening and not needs_regen:
            # Check if any threatening thread affects THIS asset
            if threads_data:
                for t in threads_data.get("active_threads", []):
                    if (t.get("portfolio_alignment") == "THREATENING" and
                            asset in [a.upper() for a in t.get("affected_assets", [])]):
                        needs_regen = True
                        logger.info(
                            f"[PM:{asset}] Regenerating due to THREATENING "
                            f"thread {t['thread_id']}"
                        )
                        break

        if needs_regen:
            # Build context for LLM
            claims_text = _build_pre_mortem_claims_text(asset, claims_archive)
            threads_text = _build_pre_mortem_threads_text(asset, threads_data)

            # Previous pre-mortem text for continuity
            prev_pm = positions_dict.get(asset, {})
            if prev_pm.get("failure_scenarios"):
                previous_pm_text = json.dumps(
                    prev_pm["failure_scenarios"], indent=2
                )[:2000]
            else:
                previous_pm_text = "No previous pre-mortem exists."

            scenarios = _generate_pre_mortem_llm(
                asset, weight_pct, regime, portfolio_text,
                claims_text, threads_text, previous_pm_text,
            )

            if scenarios:
                positions_dict[asset] = {
                    "asset": asset,
                    "v16_weight": weight,
                    "v16_weight_pct": weight_pct,
                    "regime": regime,
                    "generated_at": today_str,
                    "evidence_last_updated": today_str,
                    "failure_scenarios": scenarios,
                    "scenario_count": len(scenarios),
                    "aggregate_risk": _compute_aggregate_risk(scenarios),
                }
                generated_count += 1
            else:
                logger.warning(
                    f"[PM:{asset}] LLM generation failed — keeping previous"
                )
        else:
            # Deterministic evidence update only
            if asset in positions_dict:
                positions_dict[asset] = _update_evidence_deterministic(
                    positions_dict[asset], claims_archive, asset
                )
                # Update weight in case it changed slightly
                positions_dict[asset]["v16_weight"] = weight
                positions_dict[asset]["v16_weight_pct"] = weight_pct
                updated_count += 1

    # Remove pre-mortems for positions no longer >10%
    removed = []
    for asset in list(positions_dict.keys()):
        if asset not in large_positions:
            removed.append(asset)
            del positions_dict[asset]

    pm_data["positions"] = positions_dict
    if generated_count > 0:
        pm_data["last_generated"] = today_str

    # Log summary
    logger.info(
        f"Pre-Mortems: {len(positions_dict)} positions tracked "
        f"({generated_count} regenerated, {updated_count} evidence-updated"
        f"{f', {len(removed)} removed' if removed else ''})"
    )
    for asset, pos in positions_dict.items():
        sc_count = pos.get("scenario_count", 0)
        risk = pos.get("aggregate_risk", "UNKNOWN")
        logger.info(
            f"  {asset} ({pos.get('v16_weight_pct', 0)}%): "
            f"{sc_count} scenarios, aggregate risk {risk}"
        )

    return pm_data


def _compute_aggregate_risk(scenarios: list[dict]) -> str:
    """Compute aggregate risk from failure scenarios.

    HIGH if any scenario is HIGH probability.
    MEDIUM if any scenario is MEDIUM or 2+ are LOW.
    LOW if all scenarios are LOW.
    """
    probs = [s.get("probability_label", "LOW") for s in scenarios]
    if "HIGH" in probs:
        return "HIGH"
    if "MEDIUM" in probs:
        return "MEDIUM"
    if len([p for p in probs if p == "LOW"]) >= 2:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Bayesian Belief State (IC V2 Phase 3, Spec Kapitel 6)
# ---------------------------------------------------------------------------
BELIEF_NEUTRAL = 5.0
BELIEF_MIN = 0.0
BELIEF_MAX = 10.0
BELIEF_BASE_LEARNING_RATE = 0.15
BELIEF_EVIDENCE_DAMPING = 0.05  # lr = base / (1 + damping * evidence_count)
BELIEF_UNCERTAINTY_MIN = 0.10
BELIEF_UNCERTAINTY_MAX = 0.90
BELIEF_UNCERTAINTY_CONFIRM_DELTA = -0.02
BELIEF_UNCERTAINTY_CONTRADICT_DELTA = 0.05
BELIEF_DAILY_DECAY_RATE = 0.02  # toward neutral per day without evidence
BELIEF_DAILY_UNCERTAINTY_RISE = 0.005  # per day without evidence
BELIEF_SIGNIFICANT_SHIFT = 1.5
BELIEF_STALE_DAYS = 7
BELIEF_STALE_UNCERTAINTY_THRESHOLD = 0.60
BELIEF_HISTORY_WEEKS = 4

ALL_TOPICS = [
    "LIQUIDITY", "FED_POLICY", "CREDIT", "RECESSION", "INFLATION",
    "EQUITY_VALUATION", "CHINA_EM", "GEOPOLITICS", "ENERGY",
    "COMMODITIES", "TECH_AI", "CRYPTO", "DOLLAR", "VOLATILITY", "POSITIONING",
]


def _load_belief_state() -> dict:
    """Load belief_state.json from data/history/."""
    if os.path.exists(BELIEF_STATE_PATH):
        try:
            return _load_json(BELIEF_STATE_PATH)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Belief state corrupt, starting fresh: {e}")
    return {"beliefs": {}, "belief_shifts": [], "stale_beliefs": [], "last_updated": None}


def _save_belief_state(bs: dict) -> None:
    """Save belief_state.json to data/history/."""
    bs["last_updated"] = date.today().isoformat()
    _save_json(bs, BELIEF_STATE_PATH)


def _initialize_belief(topic: str) -> dict:
    """Create a new topic belief with neutral defaults."""
    return {
        "belief_score": BELIEF_NEUTRAL,
        "belief_direction": "NEUTRAL",
        "uncertainty": 0.50,
        "evidence_count": 0,
        "evidence_count_7d": 0,
        "last_evidence_date": None,
        "last_significant_shift": None,
        "shift_magnitude": 0.0,
        "shift_cause": None,
        "strongest_bull": None,
        "strongest_bear": None,
        "stale_warning": False,
        "history_4w": [],
    }


def _compute_evidence_weight(
    thesis: dict,
    source_config: dict,
    expertise_matrix: dict,
) -> float:
    """Compute evidence weight for a single thesis.

    weight = (expertise/10) * (conviction/10) * freshness_decay * bias_adjustment

    bias_adjustment rewards sources speaking AGAINST their known bias:
      alignment_with_bias: 1.0 if direction matches bias sign, -1.0 if opposite
      adjustment = 1.0 - (alignment * abs(known_bias) / 20)
      Range: ~0.65 (max bias-aligned) to ~1.35 (max bias-contrary)
    """
    source_id = thesis.get("source_id", "")
    topics = thesis.get("topics", [])
    primary_topic = topics[0] if topics else ""

    # Expertise score (0-10) for the primary topic
    expertise = (
        expertise_matrix.get("expertise", {})
        .get(source_id, {})
        .get(primary_topic, 3)
    )
    expertise_norm = expertise / 10.0

    # Speaker confidence / conviction (1-10)
    conviction = thesis.get("speaker_confidence", 5)
    if not isinstance(conviction, (int, float)):
        conviction = 5
    conviction_norm = max(1, min(10, conviction)) / 10.0

    # Freshness decay
    freshness = thesis.get("freshness", "FRESH")
    decay_map = {"FRESH": 1.0, "AGING": 0.7, "FADING": 0.4, "ARCHIVED": 0.15}
    freshness_decay = decay_map.get(freshness, 1.0)

    # Bias adjustment
    known_bias = source_config.get("known_bias", 0)
    direction = thesis.get("sentiment", {}).get("direction", "NEUTRAL")

    if known_bias != 0 and direction in ("BULLISH", "BEARISH"):
        # Determine if direction aligns with bias
        # Positive bias = expected to be bullish
        # Negative bias = expected to be bearish
        bias_sign = 1 if known_bias > 0 else -1
        dir_sign = 1 if direction == "BULLISH" else -1
        alignment = 1.0 if bias_sign == dir_sign else -1.0
        bias_adjustment = 1.0 - (alignment * abs(known_bias) / 20.0)
        # Clamp to reasonable range
        bias_adjustment = max(0.5, min(1.5, bias_adjustment))
    else:
        bias_adjustment = 1.0

    weight = expertise_norm * conviction_norm * freshness_decay * bias_adjustment
    return max(0.0, min(1.0, weight))


def _update_belief_for_thesis(
    belief: dict,
    thesis: dict,
    evidence_weight: float,
    source_config: dict,
    expertise_matrix: dict,
) -> dict:
    """Update a topic belief with a single thesis.

    Applies:
      1. Dynamic learning rate (slower with more evidence)
      2. Direction signal (+weight for BULLISH, -weight for BEARISH)
      3. Uncertainty adjustment (down for confirm, up for contradict)
      4. Strongest bull/bear tracking
    """
    direction = thesis.get("sentiment", {}).get("direction", "NEUTRAL")

    if direction == "NEUTRAL":
        # Neutral evidence doesn't move the belief
        belief["evidence_count"] = belief.get("evidence_count", 0) + 1
        return belief

    # Dynamic learning rate
    evidence_count = belief.get("evidence_count", 0)
    lr = BELIEF_BASE_LEARNING_RATE / (1.0 + BELIEF_EVIDENCE_DAMPING * evidence_count)

    # Direction signal
    signal = evidence_weight if direction == "BULLISH" else -evidence_weight

    # Update belief score
    old_score = belief.get("belief_score", BELIEF_NEUTRAL)
    new_score = old_score + lr * signal
    new_score = max(BELIEF_MIN, min(BELIEF_MAX, new_score))
    belief["belief_score"] = round(new_score, 2)

    # Update direction label
    if new_score > 5.5:
        belief["belief_direction"] = "BULLISH"
    elif new_score < 4.5:
        belief["belief_direction"] = "BEARISH"
    else:
        belief["belief_direction"] = "NEUTRAL"

    # Update uncertainty
    old_uncertainty = belief.get("uncertainty", 0.50)
    belief_is_bullish = old_score > BELIEF_NEUTRAL
    signal_is_bullish = direction == "BULLISH"

    if belief_is_bullish == signal_is_bullish:
        # Confirmation — reduce uncertainty
        new_uncertainty = old_uncertainty + BELIEF_UNCERTAINTY_CONFIRM_DELTA
    else:
        # Contradiction — increase uncertainty
        new_uncertainty = old_uncertainty + BELIEF_UNCERTAINTY_CONTRADICT_DELTA

    belief["uncertainty"] = round(
        max(BELIEF_UNCERTAINTY_MIN, min(BELIEF_UNCERTAINTY_MAX, new_uncertainty)), 3
    )

    # Update evidence count
    belief["evidence_count"] = evidence_count + 1

    # Track strongest bull/bear
    source_id = thesis.get("source_id", "")
    topics = thesis.get("topics", [])
    primary_topic = topics[0] if topics else ""
    expertise = (
        expertise_matrix.get("expertise", {})
        .get(source_id, {})
        .get(primary_topic, 0)
    )

    if direction == "BULLISH":
        current_bull = belief.get("strongest_bull")
        if not current_bull or expertise > current_bull.get("expertise", 0):
            belief["strongest_bull"] = {
                "source_id": source_id,
                "expertise": expertise,
            }
    elif direction == "BEARISH":
        current_bear = belief.get("strongest_bear")
        if not current_bear or expertise > current_bear.get("expertise", 0):
            belief["strongest_bear"] = {
                "source_id": source_id,
                "expertise": expertise,
            }

    return belief


def _apply_daily_decay(beliefs: dict, today: date) -> dict:
    """Apply daily decay toward neutral for topics without new evidence.

    Per Spec Kapitel 6.4:
      belief_score += 0.02 * (5.0 - belief_score)  per day without evidence
      uncertainty += 0.005 per day without evidence
    """
    today_str = today.isoformat()
    week_ago = (today - timedelta(days=7)).isoformat()

    for topic, belief in beliefs.items():
        last_ev = belief.get("last_evidence_date")

        if last_ev == today_str:
            # Had evidence today — no decay
            belief["stale_warning"] = False
            continue

        # Apply decay
        old_score = belief.get("belief_score", BELIEF_NEUTRAL)
        decay_step = BELIEF_DAILY_DECAY_RATE * (BELIEF_NEUTRAL - old_score)
        new_score = old_score + decay_step
        belief["belief_score"] = round(new_score, 2)

        # Update direction
        if new_score > 5.5:
            belief["belief_direction"] = "BULLISH"
        elif new_score < 4.5:
            belief["belief_direction"] = "BEARISH"
        else:
            belief["belief_direction"] = "NEUTRAL"

        # Uncertainty rises
        old_unc = belief.get("uncertainty", 0.50)
        new_unc = min(BELIEF_UNCERTAINTY_MAX, old_unc + BELIEF_DAILY_UNCERTAINTY_RISE)
        belief["uncertainty"] = round(new_unc, 3)

        # Stale warning
        if last_ev and last_ev < week_ago and new_unc >= BELIEF_STALE_UNCERTAINTY_THRESHOLD:
            belief["stale_warning"] = True
        elif not last_ev:
            belief["stale_warning"] = False

    return beliefs


def _update_belief_history(beliefs: dict, today: date) -> dict:
    """Update weekly history snapshots for each belief."""
    today_str = today.isoformat()

    for topic, belief in beliefs.items():
        history = belief.get("history_4w", [])

        # Add today's snapshot if not already present for this date
        if not history or history[-1].get("date") != today_str:
            history.append({
                "date": today_str,
                "belief": belief.get("belief_score", BELIEF_NEUTRAL),
                "uncertainty": belief.get("uncertainty", 0.50),
            })

        # Keep only last 4 weeks (28 entries max if daily, typically ~20)
        if len(history) > 28:
            history = history[-28:]

        belief["history_4w"] = history

    return beliefs


def _update_belief_state(
    new_claims: list[dict],
    belief_state: dict,
    expertise_matrix: dict,
    sources_config: list[dict],
) -> dict:
    """Orchestrate Bayesian Belief State update.

    Steps:
      1. Initialize missing topic beliefs
      2. For each new FRESH claim: compute evidence weight, update belief
      3. Update evidence_count_7d for all topics
      4. Apply daily decay for topics without new evidence
      5. Detect significant belief shifts
      6. Update weekly history
      7. Build stale_beliefs list

    Returns: updated belief_state dict
    """
    today = date.today()
    today_str = today.isoformat()
    week_ago = (today - timedelta(days=7)).isoformat()

    beliefs = belief_state.get("beliefs", {})
    source_lookup = {s["source_id"]: s for s in sources_config}

    # 1. Initialize missing topics
    for topic in ALL_TOPICS:
        if topic not in beliefs:
            beliefs[topic] = _initialize_belief(topic)

    # Save previous scores for shift detection
    previous_scores = {t: b.get("belief_score", BELIEF_NEUTRAL) for t, b in beliefs.items()}

    # Track which topics received new evidence today
    topics_with_evidence_today = set()

    # 2. Process new FRESH claims
    for thesis in new_claims:
        if thesis.get("freshness", "FRESH") != "FRESH":
            continue

        source_id = thesis.get("source_id", "")
        src_config = source_lookup.get(source_id, {})
        topics = thesis.get("topics", [])

        # Compute evidence weight once per thesis
        weight = _compute_evidence_weight(thesis, src_config, expertise_matrix)

        # Apply to each topic the thesis covers
        for topic in topics:
            if topic not in beliefs:
                beliefs[topic] = _initialize_belief(topic)

            beliefs[topic] = _update_belief_for_thesis(
                beliefs[topic], thesis, weight, src_config, expertise_matrix
            )
            beliefs[topic]["last_evidence_date"] = today_str
            topics_with_evidence_today.add(topic)

    # 3. Update evidence_count_7d
    # Count from claims_archive would be more accurate, but for simplicity
    # we increment for topics that got evidence today and decay for others
    for topic in ALL_TOPICS:
        if topic in topics_with_evidence_today:
            # Rough count: increment by number of new claims for this topic
            count_7d = beliefs[topic].get("evidence_count_7d", 0)
            new_for_topic = sum(
                1 for c in new_claims
                if c.get("freshness") == "FRESH" and topic in c.get("topics", [])
            )
            beliefs[topic]["evidence_count_7d"] = count_7d + new_for_topic
        # Note: 7d count will be recalculated properly when we have enough data

    # 4. Apply daily decay
    beliefs = _apply_daily_decay(beliefs, today)

    # 5. Detect significant shifts
    belief_shifts = []
    for topic in ALL_TOPICS:
        if topic not in beliefs:
            continue
        old_score = previous_scores.get(topic, BELIEF_NEUTRAL)
        new_score = beliefs[topic].get("belief_score", BELIEF_NEUTRAL)
        magnitude = round(new_score - old_score, 2)

        if abs(magnitude) >= BELIEF_SIGNIFICANT_SHIFT:
            # Find cause (highest-weight thesis for this topic)
            cause = "Decay/accumulation"
            for thesis in new_claims:
                if topic in thesis.get("topics", []) and thesis.get("freshness") == "FRESH":
                    cause = f"{thesis.get('source_id', '?')}: {thesis.get('claim_text', '')[:80]}"
                    break

            shift = {
                "topic": topic,
                "from_score": round(old_score, 2),
                "to_score": round(new_score, 2),
                "magnitude": magnitude,
                "cause": cause,
                "date": today_str,
            }
            belief_shifts.append(shift)

            beliefs[topic]["last_significant_shift"] = today_str
            beliefs[topic]["shift_magnitude"] = magnitude
            beliefs[topic]["shift_cause"] = cause

            logger.info(
                f"BELIEF SHIFT: {topic} {old_score:.1f} → {new_score:.1f} "
                f"({magnitude:+.1f}) — {cause[:60]}"
            )

    # 6. Update history
    beliefs = _update_belief_history(beliefs, today)

    # 7. Build stale_beliefs list
    stale_beliefs = []
    for topic in ALL_TOPICS:
        if beliefs.get(topic, {}).get("stale_warning"):
            last_ev = beliefs[topic].get("last_evidence_date", "")
            days_stale = 0
            if last_ev:
                try:
                    led = datetime.strptime(last_ev, "%Y-%m-%d").date()
                    days_stale = (today - led).days
                except (ValueError, TypeError):
                    pass
            stale_beliefs.append({
                "topic": topic,
                "belief_score": beliefs[topic].get("belief_score", BELIEF_NEUTRAL),
                "uncertainty": beliefs[topic].get("uncertainty", 0.50),
                "days_without_evidence": days_stale,
            })

    belief_state["beliefs"] = beliefs
    belief_state["belief_shifts"] = belief_shifts
    belief_state["stale_beliefs"] = stale_beliefs

    # Log summary
    non_neutral = {
        t: round(b.get("belief_score", 5.0), 1)
        for t, b in beliefs.items()
        if abs(b.get("belief_score", 5.0) - 5.0) > 0.3
    }
    logger.info(
        f"Belief State: {len(beliefs)} topics, "
        f"{len(topics_with_evidence_today)} updated today, "
        f"{len(belief_shifts)} significant shifts, "
        f"{len(stale_beliefs)} stale"
    )
    if non_neutral:
        logger.info(f"Non-neutral beliefs: {non_neutral}")
    if belief_shifts:
        for s in belief_shifts:
            logger.info(
                f"  SHIFT: {s['topic']} {s['from_score']} → {s['to_score']} "
                f"({s['magnitude']:+.1f})"
            )

    return belief_state


# ---------------------------------------------------------------------------
# Cross-System Confirmation: IC vs V16 (IC V2 Phase 3, Spec Kapitel 13)
# ---------------------------------------------------------------------------
# Layer name mapping from latest.json to short codes
LAYER_NAME_TO_CODE = {
    "Global Liquidity Cycle (L1)": "L1",
    "Macro Regime (L2)": "L2",
    "Earnings & Fundamentals (L3)": "L3",
    "Cross-Border Flows & FX (L4)": "L4",
    "Risk Appetite & Sentiment (L5)": "L5",
    "Relative Value & Asset Rotation (L6)": "L6",
    "Central Bank Policy Divergence (L7)": "L7",
    "Tail Risk & Black Swan (L8)": "L8",
}


def _get_layer_scores_from_dashboard(dashboard_data: dict) -> dict:
    """Extract layer scores from latest.json, mapped to L1-L8 codes.

    Returns: {"L1": {"score": 0, "direction": "STABLE"}, ...}
    """
    layer_scores = {}
    raw_layers = (
        dashboard_data.get("layers", {}).get("layer_scores", {})
    )
    for full_name, data in raw_layers.items():
        code = LAYER_NAME_TO_CODE.get(full_name)
        if code and isinstance(data, dict):
            layer_scores[code] = {
                "score": data.get("score", 0),
                "direction": data.get("direction", "STABLE"),
            }
    return layer_scores


def _layer_score_to_direction(score: int | float) -> str:
    """Convert V16 layer score (-5 to +5) to direction label."""
    if score >= 2:
        return "BULLISH"
    elif score <= -2:
        return "BEARISH"
    else:
        return "NEUTRAL"


def _compute_cross_system_confirmation(
    belief_state: dict,
    taxonomy: dict,
    dashboard_data: dict,
) -> list[dict]:
    """Compare IC Belief State against V16 Layer Scores.

    Per Spec Kapitel 13: For each topic with divergence_possible=true,
    compare IC belief direction against V16 layer signal.

    Alignment types:
      CONFIRMING:    IC and V16 same direction
      DIVERGING:     One is NEUTRAL, other has direction
      CONTRADICTING: IC and V16 opposite directions (CRITICAL)

    Returns: list of cross_system dicts
    """
    if not belief_state or not dashboard_data:
        return []

    beliefs = belief_state.get("beliefs", {})
    layer_scores = _get_layer_scores_from_dashboard(dashboard_data)
    topic_to_layers = taxonomy.get("topic_to_layers", {})
    divergence_config = taxonomy.get("divergence_config", {})

    if not layer_scores:
        logger.info("Cross-System: No layer scores available — skipping")
        return []

    results = []

    for topic, config in divergence_config.items():
        if not config.get("divergence_possible", False):
            continue

        belief = beliefs.get(topic)
        if not belief:
            continue

        ic_score = belief.get("belief_score", 5.0)
        ic_direction = belief.get("belief_direction", "NEUTRAL")

        # Get V16 layers for this topic
        layers_for_topic = topic_to_layers.get(topic, [])
        if not layers_for_topic:
            continue

        # Average the layer scores for this topic
        layer_vals = []
        layer_details = []
        for layer_code in layers_for_topic:
            ls = layer_scores.get(layer_code)
            if ls:
                layer_vals.append(ls["score"])
                layer_details.append(f"{layer_code}={ls['score']}")

        if not layer_vals:
            continue

        avg_layer_score = sum(layer_vals) / len(layer_vals)
        v16_direction = _layer_score_to_direction(avg_layer_score)

        # Determine alignment
        if ic_direction == v16_direction:
            alignment = "CONFIRMING"
        elif ic_direction == "NEUTRAL" or v16_direction == "NEUTRAL":
            alignment = "DIVERGING"
        elif (ic_direction == "BULLISH" and v16_direction == "BEARISH") or \
             (ic_direction == "BEARISH" and v16_direction == "BULLISH"):
            alignment = "CONTRADICTING"
        else:
            alignment = "DIVERGING"

        results.append({
            "topic": topic,
            "ic_belief": round(ic_score, 1),
            "ic_direction": ic_direction,
            "ic_uncertainty": round(belief.get("uncertainty", 0.50), 2),
            "v16_layers": ", ".join(layer_details),
            "v16_avg_score": round(avg_layer_score, 1),
            "v16_direction": v16_direction,
            "alignment": alignment,
        })

    # Sort: CONTRADICTING first, then DIVERGING, then CONFIRMING
    alignment_order = {"CONTRADICTING": 0, "DIVERGING": 1, "CONFIRMING": 2}
    results.sort(key=lambda r: alignment_order.get(r["alignment"], 9))

    # Log
    contradictions = [r for r in results if r["alignment"] == "CONTRADICTING"]
    divergences = [r for r in results if r["alignment"] == "DIVERGING"]
    confirmations = [r for r in results if r["alignment"] == "CONFIRMING"]

    logger.info(
        f"Cross-System Confirmation: {len(results)} topics checked — "
        f"{len(confirmations)} confirming, {len(divergences)} diverging, "
        f"{len(contradictions)} contradicting"
    )
    for c in contradictions:
        logger.warning(
            f"  CONTRADICTING: {c['topic']} — IC {c['ic_direction']} "
            f"({c['ic_belief']}) vs V16 {c['v16_direction']} ({c['v16_layers']})"
        )

    return results


# ---------------------------------------------------------------------------
# Source Disagreement Tracking (IC V2 Phase 3, Spec Kapitel 12)
# ---------------------------------------------------------------------------
DISAGREEMENT_MIN_EXPERTISE = 6


def _detect_disagreements(
    active_claims: list[dict],
    sources_config: list[dict],
    expertise_matrix: dict,
    source_history: dict,
    v16_context: dict | None,
) -> list[dict]:
    """Detect expert disagreements per topic.

    A disagreement exists when:
      1. Two sources have active claims on the same topic
      2. Their directions are opposite (BULLISH vs BEARISH)
      3. Both have expertise >= 6 in that topic

    For each disagreement, tracks:
      - Side A (bullish) and Side B (bearish) with strongest source
      - V16 alignment (which side does V16 support?)
      - Portfolio exposure (which positions are affected?)
      - Second derivative signal (is one side's conviction fading?)

    Returns: list of disagreement dicts
    """
    source_lookup = {s["source_id"]: s for s in sources_config}
    expertise = expertise_matrix.get("expertise", {})
    sources_hist = source_history.get("sources", {}) if source_history else {}

    # Group claims by topic + direction, tracking best source per side
    # Structure: topic -> direction -> [{source_id, expertise, claim_text, ...}]
    topic_sides: dict[str, dict[str, list[dict]]] = {}

    for claim in active_claims:
        direction = claim.get("sentiment", {}).get("direction", "NEUTRAL")
        if direction not in ("BULLISH", "BEARISH"):
            continue

        for topic in claim.get("topics", []):
            source_id = claim.get("source_id", "")
            exp_score = expertise.get(source_id, {}).get(topic, 0)

            if exp_score < DISAGREEMENT_MIN_EXPERTISE:
                continue

            if topic not in topic_sides:
                topic_sides[topic] = {"BULLISH": [], "BEARISH": []}

            # Check if this source is already on this side for this topic
            existing = [
                e for e in topic_sides[topic][direction]
                if e["source_id"] == source_id
            ]
            if existing:
                # Keep highest novelty claim
                if claim.get("novelty_score", 0) > existing[0].get("novelty_score", 0):
                    existing[0].update({
                        "claim_text": claim.get("claim_text", "")[:200],
                        "novelty_score": claim.get("novelty_score", 0),
                        "intensity": claim.get("sentiment", {}).get("intensity", 5),
                    })
            else:
                topic_sides[topic][direction].append({
                    "source_id": source_id,
                    "expertise": exp_score,
                    "direction": direction,
                    "claim_text": claim.get("claim_text", "")[:200],
                    "novelty_score": claim.get("novelty_score", 0),
                    "intensity": claim.get("sentiment", {}).get("intensity", 5),
                })

    # Build disagreements where both sides have qualified sources
    disagreements = []

    for topic, sides in topic_sides.items():
        bulls = sides.get("BULLISH", [])
        bears = sides.get("BEARISH", [])

        if not bulls or not bears:
            continue

        # Get strongest source per side (by expertise, then novelty)
        best_bull = max(bulls, key=lambda s: (s["expertise"], s["novelty_score"]))
        best_bear = max(bears, key=lambda s: (s["expertise"], s["novelty_score"]))

        # V16 alignment: check belief state direction or layer scores
        v16_alignment = "UNKNOWN"
        if v16_context:
            # Simple heuristic: if V16 has positive weights in assets
            # associated with this topic's bullish direction, V16 = SIDE_A
            # This is approximate; full implementation would use layer scores
            v16_alignment = "NEUTRAL"

        # Second derivative: check conviction trends
        bull_trend = sources_hist.get(
            best_bull["source_id"], {}
        ).get("conviction_trend", "STABLE")
        bear_trend = sources_hist.get(
            best_bear["source_id"], {}
        ).get("conviction_trend", "STABLE")

        second_derivative = None
        if bull_trend == "FALLING" and bear_trend != "FALLING":
            second_derivative = (
                f"{best_bull['source_id']}'s conviction FALLING — "
                f"disagreement may resolve toward BEARISH"
            )
        elif bear_trend == "FALLING" and bull_trend != "FALLING":
            second_derivative = (
                f"{best_bear['source_id']}'s conviction FALLING — "
                f"disagreement may resolve toward BULLISH"
            )
        elif bull_trend == "RISING" and bear_trend != "RISING":
            second_derivative = (
                f"{best_bull['source_id']}'s conviction RISING — "
                f"bullish side strengthening"
            )
        elif bear_trend == "RISING" and bull_trend != "RISING":
            second_derivative = (
                f"{best_bear['source_id']}'s conviction RISING — "
                f"bearish side strengthening"
            )

        # Portfolio exposure: check if V16 has positions in assets for this topic
        portfolio_exposure = "NONE"
        if v16_context:
            from step_0i_ic_pipeline.src.extraction.extractor import _load_taxonomy
            try:
                tax = _load_taxonomy()
                topic_assets = set(tax.get("topic_to_assets", {}).get(topic, []))
                v16_weights = v16_context.get("current_weights", {})
                v16_assets = {
                    a.upper() for a, w in v16_weights.items()
                    if isinstance(w, (int, float)) and w > 0.005
                }
                overlap = topic_assets & v16_assets
                if overlap:
                    total_weight = sum(
                        v16_weights.get(a, 0) for a in overlap
                    )
                    if total_weight > 0.20:
                        portfolio_exposure = "HIGH"
                    elif total_weight > 0.10:
                        portfolio_exposure = "MEDIUM"
                    else:
                        portfolio_exposure = "LOW"
            except Exception:
                pass

        disagreements.append({
            "topic": topic,
            "side_a": {
                "source_id": best_bull["source_id"],
                "direction": "BULLISH",
                "expertise": best_bull["expertise"],
                "intensity": best_bull["intensity"],
                "claim_text": best_bull["claim_text"],
                "conviction_trend": bull_trend,
                "supporter_count": len(bulls),
            },
            "side_b": {
                "source_id": best_bear["source_id"],
                "direction": "BEARISH",
                "expertise": best_bear["expertise"],
                "intensity": best_bear["intensity"],
                "claim_text": best_bear["claim_text"],
                "conviction_trend": bear_trend,
                "supporter_count": len(bears),
            },
            "v16_alignment": v16_alignment,
            "portfolio_exposure": portfolio_exposure,
            "second_derivative_signal": second_derivative,
        })

    # Sort by portfolio exposure (HIGH first) then by max expertise
    exposure_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    disagreements.sort(key=lambda d: (
        exposure_order.get(d["portfolio_exposure"], 9),
        -max(d["side_a"]["expertise"], d["side_b"]["expertise"]),
    ))

    # Log
    if disagreements:
        for d in disagreements:
            sd_sig = d.get("second_derivative_signal", "")
            logger.info(
                f"EXPERT DISAGREEMENT: {d['topic']} — "
                f"{d['side_a']['source_id']} (BULL, exp {d['side_a']['expertise']}) "
                f"vs {d['side_b']['source_id']} (BEAR, exp {d['side_b']['expertise']}) "
                f"| Portfolio: {d['portfolio_exposure']}"
                f"{f' | {sd_sig}' if sd_sig else ''}"
            )
    else:
        logger.info("Source Disagreement Tracking: no expert disagreements detected")

    return disagreements


# ---------------------------------------------------------------------------
# Google Drive Output
# ---------------------------------------------------------------------------
DRIVE_ROOT_ID = "1Tng3i4Cly7isKOxIkGqiTmGiZNEtPj3D"


def _get_drive_service():
    """Build Google Drive service from env credentials."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
    if not creds_json:
        return None
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


def _get_sheets_service():
    """Build Google Sheets service from env credentials."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
    if not creds_json:
        return None
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)


def _find_or_create_folder(service, name: str, parent_id: str) -> str:
    """Find or create a folder in Drive."""
    query = (
        f"name='{name}' and '{parent_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {
        "name": name,
        "parents": [parent_id],
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def _upload_to_drive(service, data: dict, filename: str, folder_id: str) -> None:
    """Upload or update a JSON file in a Drive folder."""
    from googleapiclient.http import MediaInMemoryUpload

    content = json.dumps(data, indent=2).encode("utf-8")
    media = MediaInMemoryUpload(content, mimetype="application/json")

    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        service.files().update(fileId=files[0]["id"], media_body=media).execute()
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        service.files().create(body=metadata, media_body=media).execute()


def write_drive_outputs(
    claims_output: dict, intel: dict, briefing: dict, claims_archive: dict
) -> None:
    """Write all outputs to Google Drive CURRENT/ and HISTORY/ic/YYYY-MM-DD/."""
    try:
        service = _get_drive_service()
        if service is None:
            logger.warning("No Drive credentials — skipping Drive writes")
            return

        today_str = date.today().isoformat()

        # Find CURRENT folder
        current_id = _find_or_create_folder(service, "CURRENT", DRIVE_ROOT_ID)

        # Write to CURRENT/
        _upload_to_drive(service, intel, "step0b_ic_intelligence.json", current_id)
        _upload_to_drive(service, claims_output, "step0b_ic_claims.json", current_id)
        _upload_to_drive(service, briefing, "step0b_ic_briefing.json", current_id)
        _upload_to_drive(
            service, claims_archive, "step0b_ic_claims_archive.json", current_id
        )
        logger.info("Drive: CURRENT/ updated (4 files incl. claims_archive)")

        # Write to HISTORY/ic/YYYY-MM-DD/ via local archive (committed by GitHub Actions)
        _write_local_archive(claims_output, "step0b_ic_claims.json", today_str)
        _write_local_archive(intel, "step0b_ic_intelligence.json", today_str)
        _write_local_archive(briefing, "step0b_ic_briefing.json", today_str)
        logger.info(f"Local archive: archive/{today_str}/ (3 files)")

    except ImportError:
        logger.warning("Google API libraries not installed — Drive writes skipped")
    except Exception as e:
        logger.error(f"Drive write failed: {e}")


def _write_local_archive(data: dict, filename: str, date_str: str) -> None:
    """Write JSON to archive/YYYY-MM-DD/ for Git-based archiving."""
    try:
        archive_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "archive", date_str,
        )
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, filename)
        with open(archive_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"  Local archive: archive/{date_str}/{filename}")
    except Exception as e:
        logger.warning(f"  Local archive write failed (non-fatal): {e}")


DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"


def write_intelligence_tab(claims: list[dict], sources_config: list[dict]) -> None:
    """Write 1 row per source to INTELLIGENCE tab (12 columns A:L)."""
    try:
        service = _get_sheets_service()
        if service is None:
            return

        today = date.today().isoformat()

        # Build source lookup for tier + bias
        source_lookup = {s["source_id"]: s for s in sources_config}

        # Aggregate per source: use highest-novelty claim as core thesis
        source_rows = {}
        for claim in claims:
            sid = claim["source_id"]
            if sid not in source_rows:
                source_rows[sid] = {
                    "source_name": claim.get("source_name", sid),
                    "core_thesis": "",
                    "direction": "",
                    "intensity": 0,
                    "bias_adj": 0,
                    "claim_type": "",
                    "confidence": 0.0,
                    "novelty_max": 0,
                    "topics": set(),
                    "content_date": "",
                }
            row = source_rows[sid]
            if claim.get("novelty_score", 0) >= row["novelty_max"]:
                row["novelty_max"] = claim["novelty_score"]
                row["core_thesis"] = claim.get("claim_text", "")[:300]
                row["direction"] = claim["sentiment"]["direction"]
                row["intensity"] = claim["sentiment"]["intensity"]
                row["claim_type"] = claim.get("claim_type", "")
                row["confidence"] = claim.get("confidence", {}).get(
                    "extraction_confidence", 0.0
                )
                row["content_date"] = claim.get("content_date", "")
                # Bias-adjusted signal
                known_bias = source_lookup.get(sid, {}).get("known_bias", 0)
                signed = row["intensity"] if row["direction"] == "BULLISH" else (
                    -row["intensity"] if row["direction"] == "BEARISH" else 0
                )
                row["bias_adj"] = signed - known_bias
            row["topics"].update(claim.get("topics", []))

        rows = []
        for sid, row in source_rows.items():
            tier = source_lookup.get(sid, {}).get("tier", "")
            rows.append([
                today,                                  # A: DATE
                sid,                                    # B: SOURCE
                tier,                                   # C: TIER
                row["core_thesis"],                     # D: CORE_THESIS
                row["direction"],                       # E: DIRECTION
                str(row["intensity"]),                  # F: INTENSITY
                str(row["bias_adj"]),                   # G: BIAS_ADJ
                row["claim_type"],                      # H: CLAIM_TYPE
                str(row["novelty_max"]),                # I: NOVELTY
                ", ".join(sorted(row["topics"])),        # J: TOPICS
                str(round(row["confidence"], 2)),        # K: CONFIDENCE
                row["content_date"],                    # L: CONTENT_DATE
            ])

        if rows:
            service.spreadsheets().values().append(
                spreadsheetId=DW_SHEET_ID,
                range="INTELLIGENCE!A:L",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": rows},
            ).execute()
            logger.info(f"Sheet INTELLIGENCE: {len(rows)} rows written")

    except ImportError:
        logger.warning("Google API libs missing — Sheet write skipped")
    except Exception as e:
        logger.error(f"INTELLIGENCE tab write failed: {e}")


def write_agent_summary_tab(briefing: dict) -> None:
    """Write 1 row IC summary to AGENT_SUMMARY tab."""
    try:
        service = _get_sheets_service()
        if service is None:
            return

        today = date.today().isoformat()
        meta = briefing.get("metadata", {})

        row = [
            today, "IC_PIPELINE", "Intelligence Collector",
            briefing.get("briefing_text", "")[:300],
            str(meta.get("topics_covered", 0)),
            f"{meta.get('divergences_flagged', 0)} divergences",
            "", "", "",
        ]

        service.spreadsheets().values().append(
            spreadsheetId=DW_SHEET_ID,
            range="AGENT_SUMMARY!A:I",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
        logger.info("Sheet AGENT_SUMMARY: IC row written")

    except ImportError:
        pass
    except Exception as e:
        logger.error(f"AGENT_SUMMARY write failed: {e}")


# ---------------------------------------------------------------------------
# Stage Runners
# ---------------------------------------------------------------------------
def run_extraction(
    sources: list[dict],
    v16_context: dict | None = None,
    claims_archive: dict | None = None,
) -> tuple[list[dict], dict]:
    """Run Stufe 1: Fetch + Extract (Extraction V2).

    Args:
        sources: list of source configs from sources.json
        v16_context: V16 data from latest.json (regime, current_weights)
        claims_archive: claims archive for source history in Call B
    """
    from step_0i_ic_pipeline.src.extraction.fetcher import fetch_all_sources
    from step_0i_ic_pipeline.src.extraction.extractor import extract_claims

    logger.info("=" * 60)
    logger.info("STUFE 1: EXTRACTION (V2)")
    logger.info("=" * 60)

    all_content, fetch_state, failed_sources = fetch_all_sources(sources)
    logger.info(f"Fetched {len(all_content)} items, {len(failed_sources)} failed")

    source_lookup = {s["source_id"]: s for s in sources}
    all_claims = []

    for content in all_content:
        sid = content["source_id"]
        src_config = source_lookup.get(sid, {})
        try:
            claims = extract_claims(
                content, src_config,
                v16_context=v16_context,
                claims_archive=claims_archive,
            )
            all_claims.extend(claims)
        except Exception as e:
            logger.error(f"[{sid}] Extraction failed: {e}")
            failed_sources.append({
                "source_id": sid,
                "error": f"Extraction: {e}",
                "retry_next_run": True,
            })

    today = date.today().isoformat()
    run_id = f"run_{today.replace('-', '')}_{datetime.utcnow().strftime('%H%M%S')}"

    no_new = [
        s["source_id"] for s in sources
        if s.get("active", True)
        and s["source_id"] not in {c["source_id"] for c in all_claims}
        and s["source_id"] not in {f["source_id"] for f in failed_sources}
    ]

    claims_output = {
        "extraction_date": today,
        "extraction_run_id": run_id,
        "sources_attempted": sum(1 for s in sources if s.get("active", True)),
        "sources_successful": len(set(c["source_id"] for c in all_claims)),
        "sources_failed": len(failed_sources),
        "sources_no_new_content": len(no_new),
        "total_claims_extracted": len(all_claims),
        "failed_sources": failed_sources,
        "no_new_content": no_new,
        "claims": all_claims,
    }

    # Save locally
    claims_path = os.path.join(DATA_DIR, "claims", f"claims_{today}.json")
    _save_json(claims_output, claims_path)

    logger.info(
        f"Extraction complete: {len(all_claims)} claims from "
        f"{claims_output['sources_successful']} sources"
    )
    return all_claims, claims_output


def run_intelligence(
    claims: list[dict],
    sources: list[dict],
    expertise_matrix: dict,
    taxonomy: dict,
) -> dict:
    """Run Stufe 2: Intelligence Engine (deterministic)."""
    from step_0i_ic_pipeline.src.intelligence.engine import run_intelligence_engine

    logger.info("=" * 60)
    logger.info("STUFE 2: INTELLIGENCE ENGINE")
    logger.info("=" * 60)

    intel = run_intelligence_engine(claims, sources, expertise_matrix, taxonomy)

    today = date.today().isoformat()
    intel_path = os.path.join(DATA_DIR, "intelligence", f"intel_{today}.json")
    _save_json(intel, intel_path)

    return intel


def run_briefing(intel: dict) -> dict:
    """Run Stufe 3: Agent 0 Briefing."""
    from step_0i_ic_pipeline.src.briefing.agent0 import generate_briefing

    logger.info("=" * 60)
    logger.info("STUFE 3: AGENT 0 BRIEFING")
    logger.info("=" * 60)

    briefing = generate_briefing(intel)

    today = date.today().isoformat()
    briefing_path = os.path.join(DATA_DIR, "briefings", f"briefing_{today}.json")
    _save_json(briefing, briefing_path)

    return briefing


# ---------------------------------------------------------------------------
# Source Cards Builder (IC V2 Phase 1)
# ---------------------------------------------------------------------------
def _build_source_cards(
    claims_archive: dict,
    sources_config: list[dict],
    fetch_state_path: str,
) -> list[dict]:
    """Build source_cards[] array for latest.json intelligence block.

    Per source: name, tier, active claims count, freshness breakdown,
    latest content date, days since content, stale warning, direction,
    bias-adjusted signal, top claim text, topics covered.
    """
    today = date.today()

    # Load fetch_state for last_content_date per source
    fetch_state = {}
    fs_path = os.path.normpath(fetch_state_path)
    if os.path.exists(fs_path):
        try:
            with open(fs_path, "r") as f:
                fs_data = json.load(f)
            fetch_state = fs_data.get("fetch_state", {})
        except Exception:
            pass

    # Build source lookup
    source_lookup = {s["source_id"]: s for s in sources_config}

    # Group archive claims by source_id (only active: FRESH/AGING/FADING)
    source_claims = {}
    for claim in claims_archive.get("claims", []):
        if claim.get("freshness") not in ("FRESH", "AGING", "FADING"):
            continue
        sid = claim.get("source_id", "")
        if sid not in source_claims:
            source_claims[sid] = []
        source_claims[sid].append(claim)

    cards = []
    for src in sources_config:
        sid = src["source_id"]
        if not src.get("active", True):
            continue

        claims_list = source_claims.get(sid, [])

        # Freshness breakdown
        freshness_breakdown = {"FRESH": 0, "AGING": 0, "FADING": 0}
        for c in claims_list:
            f = c.get("freshness", "")
            if f in freshness_breakdown:
                freshness_breakdown[f] += 1

        # Latest content date from fetch_state
        fs_entry = fetch_state.get(sid, {})
        last_content_date = fs_entry.get("last_content_date", "")
        days_since = None
        if last_content_date and last_content_date != "2000-01-01":
            try:
                lcd = datetime.strptime(last_content_date, "%Y-%m-%d").date()
                days_since = (today - lcd).days
            except (ValueError, TypeError):
                pass

        # Stale warning: CORE source >14 days without content
        tier = src.get("tier", "")
        stale_warning = False
        if tier == "CORE" and days_since is not None and days_since > 14:
            stale_warning = True

        # Direction + intensity from highest-novelty active claim
        direction = ""
        intensity = 0
        bias_adj = 0
        top_claim_text = ""
        all_topics = set()
        best_novelty = -1

        known_bias = src.get("known_bias", 0)

        for c in claims_list:
            # Collect topics
            for t in c.get("topics", []):
                all_topics.add(t)

            nov = c.get("novelty_score", 0)
            if nov > best_novelty:
                best_novelty = nov
                sentiment = c.get("sentiment", {})
                direction = sentiment.get("direction", "")
                intensity = sentiment.get("intensity", 0)
                top_claim_text = c.get("claim_text", "")[:200]

                # Bias-adjusted signal
                signed = intensity if direction == "BULLISH" else (
                    -intensity if direction == "BEARISH" else 0
                )
                bias_adj = signed - known_bias

        # Build claims list for frontend (sorted by novelty desc, then date desc)
        claims_for_card = []
        for c in sorted(
            claims_list,
            key=lambda x: (x.get("novelty_score", 0), x.get("content_date", "")),
            reverse=True,
        ):
            sentiment = c.get("sentiment", {})
            claims_for_card.append({
                "claim_text": c.get("claim_text", "")[:300],
                "freshness": c.get("freshness", "FRESH"),
                "content_date": c.get("content_date", ""),
                "topics": c.get("topics", []),
                "novelty_score": c.get("novelty_score", 0),
                "direction": sentiment.get("direction", ""),
                "intensity": sentiment.get("intensity", 0),
            })

        cards.append({
            "source_id": sid,
            "source_name": src.get("source_name", sid),
            "tier": tier,
            "active_claims": len(claims_list),
            "freshness_breakdown": freshness_breakdown,
            "latest_content_date": last_content_date,
            "days_since_content": days_since,
            "stale_warning": stale_warning,
            "direction": direction,
            "intensity": intensity,
            "bias_adjusted_signal": bias_adj,
            "known_bias": known_bias,
            "top_claim": top_claim_text,
            "claims": claims_for_card,
            "topics": sorted(all_topics),
        })
    # Sort: CORE first, then by active_claims descending
    tier_order = {"CORE": 0, "SECONDARY": 1, "NOISE_FILTER": 2}
    cards.sort(key=lambda c: (
        tier_order.get(c["tier"], 9),
        -c["active_claims"],
    ))

    # Log stale warnings
    stale_sources = [c["source_id"] for c in cards if c["stale_warning"]]
    if stale_sources:
        logger.warning(
            f"STALE SOURCE WARNING: {', '.join(stale_sources)} "
            f"(CORE, >14 days without content)"
        )

    logger.info(
        f"Source cards built: {len(cards)} sources, "
        f"{sum(c['active_claims'] for c in cards)} total active claims"
    )

    return cards


# ---------------------------------------------------------------------------
# Dashboard Intelligence Block Builder
# ---------------------------------------------------------------------------
def build_intelligence_block(
    intel: dict,
    briefing: dict,
    claims_archive: dict,
    sources_config: list[dict],
    cadence_anomalies: list[dict] | None = None,
    threads_data: dict | None = None,
    pm_data: dict | None = None,
    belief_state: dict | None = None,
    cross_system: list | None = None,
    expert_disagreements: list | None = None,
) -> dict:
    """
    Map IC Pipeline output to dashboard.json intelligence block.
    Format matches IntelDetail.jsx expectations exactly:
      consensus[TOPIC] = {score, direction, sources, confidence}
      divergences[] = {theme, divergence_type, magnitude, ic_signal, dc_signal, ...}
      high_novelty_claims[] = {source, claim, novelty, signal, theme}
      catalyst_timeline[] = {event, date, days_until, impact, themes}
      source_cards[] = {source_id, source_name, tier, active_claims, ...}
      cadence_anomalies[] = {source_id, anomaly_level, cadence_ratio, ...}
      active_threads[] = {thread_id, core_hypothesis, status, conviction, ...}
      pre_mortems[] = {asset, weight_pct, scenarios, aggregate_risk, ...}  (NEW)
    """
    today = date.today()

    # --- Consensus: IC format -> Dashboard format ---
    consensus_out = {}
    ic_consensus = intel.get("consensus", {})
    for topic, data in ic_consensus.items():
        score = data.get("consensus_score", 0.0)
        source_count = data.get("source_count", 0)
        confidence_label = data.get("confidence", "NO_DATA")

        if confidence_label == "NO_DATA":
            continue

        conf_map = {"HIGH": 0.85, "MEDIUM": 0.65, "LOW": 0.40}
        confidence_num = conf_map.get(confidence_label, 0.5)

        if score > 1.0:
            direction = "BULLISH"
        elif score < -1.0:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        consensus_out[topic] = {
            "score": score,
            "direction": direction,
            "sources": source_count,
            "confidence": confidence_num,
        }

    # --- Divergences: IC format -> Dashboard format ---
    divergences_out = []
    for div in intel.get("divergences", []):
        topic = div.get("topic", "")
        ic_contrib = ic_consensus.get(topic, {}).get("contributors", [])
        top_names = [c["source_id"] for c in sorted(
            ic_contrib,
            key=lambda x: abs(x.get("avg_bias_adjusted_signal", 0))
            * x.get("expertise_weight", 1),
            reverse=True,
        )[:3]]

        ic_score = div.get("ic_consensus_score", 0.0)
        dc_score = div.get("dc_signal_score", 0.0)

        divergences_out.append({
            "theme": topic,
            "divergence_type": div.get("divergence_type", "UNKNOWN"),
            "magnitude": round(div.get("severity", 0.0) / 10.0, 2),
            "ic_signal": ic_score,
            "dc_signal": dc_score,
            "ic_top_contributors": top_names,
            "dc_source_field": div.get("dc_source_field", ""),
            "dc_confidence": div.get("dc_confidence", None),
            "interpretation_hint": div.get("interpretation", ""),
        })

    # --- High Novelty Claims: IC format -> Dashboard format ---
    claims_out = []
    for hn in intel.get("high_novelty_claims", []):
        primary_topic = hn.get("topics", [""])[0] if hn.get("topics") else ""
        topic_consensus = ic_consensus.get(primary_topic, {})
        signal = topic_consensus.get("consensus_score", 0.0)

        claims_out.append({
            "source": hn.get("source_id", ""),
            "claim": hn.get("claim_text", ""),
            "novelty": hn.get("novelty_score", 0),
            "signal": round(signal, 1),
            "theme": primary_topic,
            "date": hn.get("content_date", ""),
            "freshness": hn.get("freshness", "FRESH"),
        })

    claims_out.sort(key=lambda x: x["novelty"], reverse=True)

    # --- Catalyst Timeline: IC format -> Dashboard format ---
    catalysts_out = []
    for cat in intel.get("catalyst_timeline", []):
        cat_date_str = cat.get("date", "")
        days_until = None
        if cat_date_str:
            try:
                if len(cat_date_str) == 10:
                    cat_date = datetime.strptime(cat_date_str, "%Y-%m-%d").date()
                    days_until = (cat_date - today).days
                elif len(cat_date_str) == 7:
                    cat_date = datetime.strptime(
                        cat_date_str + "-01", "%Y-%m-%d"
                    ).date()
                    days_until = (cat_date - today).days
            except (ValueError, TypeError):
                pass

        if days_until is not None and days_until < 0:
            continue

        sources_count = len(cat.get("sources_mentioning", []))
        impact = "HIGH" if sources_count >= 2 else "MEDIUM"

        catalysts_out.append({
            "event": cat.get("event", ""),
            "date": cat_date_str,
            "days_until": days_until if days_until is not None else 99,
            "impact": impact,
            "themes": cat.get("topics", []),
        })

    catalysts_out.sort(key=lambda x: x["days_until"])

    # --- Source Cards (IC V2 Phase 1) ---
    fetch_state_path = os.path.join(DATA_DIR, "history", "fetch_state.json")
    source_cards = _build_source_cards(
        claims_archive, sources_config, fetch_state_path
    )

    # --- Active Threads (IC V2 Phase 2) ---
    active_threads_out = []
    if threads_data:
        for t in threads_data.get("active_threads", []):
            # Compact version for dashboard (no full evidence array)
            active_threads_out.append({
                "thread_id": t.get("thread_id", ""),
                "core_hypothesis": t.get("core_hypothesis", "")[:200],
                "status": t.get("status", "SEED"),
                "challenged": t.get("challenged", False),
                "conviction": t.get("conviction", 0.0),
                "direction": t.get("direction", "NEUTRAL"),
                "topics": t.get("topics", []),
                "sources": t.get("sources", []),
                "source_count": t.get("source_count", 0),
                "affected_assets": t.get("affected_assets", []),
                "created_at": t.get("created_at", ""),
                "last_evidence_date": t.get("last_evidence_date", ""),
                "portfolio_alignment": t.get("portfolio_alignment", "NEUTRAL"),
                "portfolio_relevance_score": t.get("portfolio_relevance_score", 0.0),
                "threatened_positions": t.get("threatened_positions", []),
            })

    # --- Pre-Mortems (IC V2 Phase 2) ---
    pre_mortems_out = []
    if pm_data:
        for asset, pos in pm_data.get("positions", {}).items():
            pre_mortems_out.append({
                "asset": pos.get("asset", asset),
                "v16_weight_pct": pos.get("v16_weight_pct", 0),
                "regime": pos.get("regime", ""),
                "generated_at": pos.get("generated_at", ""),
                "aggregate_risk": pos.get("aggregate_risk", "LOW"),
                "scenario_count": pos.get("scenario_count", 0),
                "failure_scenarios": [
                    {
                        "scenario_id": s.get("scenario_id", ""),
                        "description": s.get("description", "")[:200],
                        "failure_category": s.get("failure_category", ""),
                        "probability_label": s.get("probability_label", "LOW"),
                        "early_warning_indicator": s.get("early_warning_indicator", "")[:200],
                        "portfolio_impact": s.get("portfolio_impact", ""),
                    }
                    for s in pos.get("failure_scenarios", [])
                ],
            })
        # Sort by weight descending
        pre_mortems_out.sort(key=lambda p: p["v16_weight_pct"], reverse=True)

    # --- Belief State (IC V2 Phase 3) ---
    belief_state_out = {}
    belief_shifts_out = []
    stale_beliefs_out = []
    if belief_state:
        for topic, b in belief_state.get("beliefs", {}).items():
            belief_state_out[topic] = {
                "belief": round(b.get("belief_score", 5.0), 1),
                "direction": b.get("belief_direction", "NEUTRAL"),
                "uncertainty": round(b.get("uncertainty", 0.50), 2),
                "evidence_count": b.get("evidence_count", 0),
                "last_evidence_date": b.get("last_evidence_date"),
                "stale_warning": b.get("stale_warning", False),
            }
        belief_shifts_out = belief_state.get("belief_shifts", [])
        stale_beliefs_out = belief_state.get("stale_beliefs", [])

    # --- Cross-System Confirmation (IC V2 Phase 3) ---
    cross_system_out = cross_system or []

    return {
        "status": "AVAILABLE",
        "consensus": consensus_out,
        "divergences": divergences_out,
        "divergences_count": len(divergences_out),
        "high_novelty_claims": claims_out,
        "catalyst_timeline": catalysts_out,
        "source_cards": source_cards,
        "cadence_anomalies": cadence_anomalies or [],
        "active_threads": active_threads_out,
        "pre_mortems": pre_mortems_out,
        "belief_state": belief_state_out,
        "belief_shifts": belief_shifts_out,
        "stale_beliefs": stale_beliefs_out,
        "cross_system": cross_system_out,
        "expert_disagreements": expert_disagreements or [],
    }


def update_dashboard_json(
    intel: dict,
    briefing: dict,
    active_claims_count: int,
    claims_archive: dict,
    sources_config: list[dict],
    cadence_anomalies: list[dict] | None = None,
    threads_data: dict | None = None,
    pm_data: dict | None = None,
    belief_state: dict | None = None,
    cross_system: list | None = None,
    expert_disagreements: list | None = None,
) -> None:
    """
    Read data/dashboard/latest.json, replace intelligence block,
    update pipeline_health, and write back.

    IC V2: Uses active_claims_count (from archive, includes carry-forward)
    instead of just today's new claims. Dashboard always gets updated
    as long as there are active claims in the 7-day window.
    """
    if not os.path.exists(DASHBOARD_JSON_PATH):
        logger.warning(
            f"Dashboard JSON not found at {DASHBOARD_JSON_PATH} "
            f"— skipping update"
        )
        return

    # Guard: Only update intelligence block if we have active claims
    # (either new or carried forward from last 7 days)
    if active_claims_count == 0:
        logger.info(
            "No active claims (new or carry-forward) — keeping existing "
            "intelligence block in dashboard."
        )
        try:
            with open(DASHBOARD_JSON_PATH, "r") as f:
                dashboard = json.load(f)
            now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            steps = dashboard.get("pipeline_health", {}).get("steps", {})
            steps["step_0b_ic"] = {
                "status": "OK",
                "completed_at": now_utc,
                "summary": "0 active claims — previous data retained",
            }
            dashboard.setdefault("pipeline_health", {})["steps"] = steps
            with open(DASHBOARD_JSON_PATH, "w") as f:
                json.dump(dashboard, f, indent=2, ensure_ascii=False)
            logger.info(
                "Dashboard pipeline_health updated (intelligence block unchanged)"
            )
        except Exception as e:
            logger.error(f"Dashboard pipeline_health update failed: {e}")
        return

    try:
        with open(DASHBOARD_JSON_PATH, "r") as f:
            dashboard = json.load(f)

        # Replace intelligence block (with source_cards + cadence_anomalies + threads + pre-mortems)
        dashboard["intelligence"] = build_intelligence_block(
            intel, briefing, claims_archive, sources_config,
            cadence_anomalies, threads_data, pm_data, belief_state,
            cross_system, expert_disagreements
        )

        # Update pipeline health
        now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        new_claims = intel.get("extraction_summary", {}).get("total_claims", 0)
        anomaly_count = len(cadence_anomalies) if cadence_anomalies else 0
        thread_count = len(
            threads_data.get("active_threads", [])
        ) if threads_data else 0
        pm_count = len(
            pm_data.get("positions", {})
        ) if pm_data else 0
        steps = dashboard.get("pipeline_health", {}).get("steps", {})
        steps["step_0b_ic"] = {
            "status": "OK",
            "completed_at": now_utc,
            "summary": (
                f"{active_claims_count} active claims "
                f"({new_claims} new + carry-forward), "
                f"{len(intel.get('divergences', []))} divergences, "
                f"{intel.get('extraction_summary', {}).get('high_novelty_claims', 0)} high-novelty"
                f"{f', {anomaly_count} cadence anomalies' if anomaly_count else ''}"
                f"{f', {thread_count} threads' if thread_count else ''}"
                f"{f', {pm_count} pre-mortems' if pm_count else ''}"
            ),
        }

        # Update header divergences count
        dashboard.get("header", {})["divergences_count"] = len(
            intel.get("divergences", [])
        )

        # Remove INTELLIGENCE from known_unknowns
        kus = dashboard.get("known_unknowns", [])
        dashboard["known_unknowns"] = [
            ku for ku in kus if ku.get("gap") != "INTELLIGENCE"
        ]

        with open(DASHBOARD_JSON_PATH, "w") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)

        logger.info(f"Dashboard JSON updated: {DASHBOARD_JSON_PATH}")

    except Exception as e:
        logger.error(f"Dashboard JSON update failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IC Intelligence Pipeline")
    parser.add_argument(
        "--stage",
        choices=["extraction", "intelligence", "briefing", "all"],
        default="all",
        help="Which stage to run",
    )
    args = parser.parse_args()

    logger.info(f"IC Pipeline starting — stage={args.stage}")
    start_time = datetime.utcnow()

    # Load config
    sources_config = _load_json(os.path.join(CONFIG_DIR, "sources.json"))
    sources = sources_config["sources"]
    expertise_matrix = _load_json(os.path.join(CONFIG_DIR, "expertise_matrix.json"))
    taxonomy = _load_json(os.path.join(CONFIG_DIR, "taxonomy.json"))

    new_claims = []
    claims_output = {}
    intel = {}
    briefing = {}
    active_claims_count = 0
    cadence_anomalies = []
    threads_data = None
    pm_data = None
    belief_state = None
    cross_system = None
    expert_disagreements = None

    # Load claims archive for carry-forward
    claims_archive = _load_claims_archive()
    logger.info(
        f"Claims archive loaded: {len(claims_archive.get('claims', []))} "
        f"existing claims"
    )

    # Load V16 context from latest.json for Extraction V2 Portfolio Transmission
    v16_context = None
    if os.path.exists(DASHBOARD_JSON_PATH):
        try:
            dashboard_data = _load_json(DASHBOARD_JSON_PATH)
            v16_raw = dashboard_data.get("v16", {})
            if v16_raw.get("current_weights"):
                v16_context = {
                    "regime": v16_raw.get("regime", "UNKNOWN"),
                    "current_weights": v16_raw.get("current_weights", {}),
                }
                active_pos = sum(
                    1 for w in v16_context["current_weights"].values()
                    if w > 0.005
                )
                logger.info(
                    f"V16 context loaded: regime={v16_context['regime']}, "
                    f"{active_pos} active positions"
                )
            else:
                logger.warning("V16 data in latest.json has no current_weights")
        except Exception as e:
            logger.warning(f"Failed to load V16 context from latest.json: {e}")
    else:
        logger.warning(
            f"latest.json not found at {DASHBOARD_JSON_PATH} — "
            f"Extraction V2 runs without V16 context"
        )

    try:
        # Stufe 1: Extraction (V2 — with V16 context + claims archive)
        if args.stage in ("extraction", "all"):
            new_claims, claims_output = run_extraction(
                sources, v16_context=v16_context, claims_archive=claims_archive
            )

        # Merge new claims into archive + get all active claims
        active_claims, claims_archive = merge_claims_into_archive(
            new_claims, claims_archive
        )
        active_claims_count = len(active_claims)

        # Save updated archive
        _save_claims_archive(claims_archive)

        # Cadence Anomaly Detection (IC V2 Phase 2)
        cadence_anomalies = _detect_cadence_anomalies(claims_archive, sources)

        # Source Conviction History (IC V2 Phase 2)
        source_history = _load_source_history()
        source_history = _update_source_history(
            new_claims, sources, source_history
        )
        _save_source_history(source_history)

        # Narrative Threads (IC V2 Phase 2)
        threads_data = _load_threads()
        threads_data = _update_threads(
            new_claims, threads_data, v16_context,
            expertise_matrix, sources, source_history, taxonomy
        )
        _save_threads(threads_data)

        # Position Pre-Mortems (IC V2 Phase 2)
        pm_data = _load_pre_mortems()
        pm_data = _run_pre_mortems(
            v16_context, claims_archive, threads_data, pm_data
        )
        _save_pre_mortems(pm_data)

        # Bayesian Belief State (IC V2 Phase 3)
        belief_state = _load_belief_state()
        belief_state = _update_belief_state(
            new_claims, belief_state, expertise_matrix, sources
        )
        _save_belief_state(belief_state)

        # Cross-System Confirmation: IC vs V16 (IC V2 Phase 3)
        if belief_state and os.path.exists(DASHBOARD_JSON_PATH):
            try:
                dashboard_for_cs = _load_json(DASHBOARD_JSON_PATH)
                cross_system = _compute_cross_system_confirmation(
                    belief_state, taxonomy, dashboard_for_cs
                )
            except Exception as e:
                logger.warning(f"Cross-System Confirmation failed: {e}")
                cross_system = []

        # Source Disagreement Tracking (IC V2 Phase 3)
        expert_disagreements = _detect_disagreements(
            active_claims, sources, expertise_matrix,
            source_history, v16_context
        )

        # Stufe 2: Intelligence (uses ALL active claims, not just new)
        if args.stage in ("intelligence", "all"):
            if args.stage == "intelligence" and not active_claims:
                # Load today's claims from file
                today = date.today().isoformat()
                claims_path = os.path.join(
                    DATA_DIR, "claims", f"claims_{today}.json"
                )
                if os.path.exists(claims_path):
                    claims_data = _load_json(claims_path)
                    active_claims = claims_data.get("claims", [])
                    claims_output = claims_data
                else:
                    logger.error("No claims file found for today")
                    sys.exit(1)

            intel = run_intelligence(
                active_claims, sources, expertise_matrix, taxonomy
            )

        # Stufe 3: Briefing
        if args.stage in ("briefing", "all"):
            if args.stage == "briefing" and not intel:
                today = date.today().isoformat()
                intel_path = os.path.join(
                    DATA_DIR, "intelligence", f"intel_{today}.json"
                )
                if os.path.exists(intel_path):
                    intel = _load_json(intel_path)
                else:
                    logger.error("No intel file found for today")
                    sys.exit(1)

            briefing = run_briefing(intel)

        # Write to Google Drive + Sheets
        if args.stage == "all":
            write_drive_outputs(claims_output, intel, briefing, claims_archive)
            write_intelligence_tab(active_claims, sources)
            write_agent_summary_tab(briefing)
            update_dashboard_json(
                intel, briefing, active_claims_count,
                claims_archive, sources, cadence_anomalies, threads_data,
                pm_data, belief_state, cross_system,
                expert_disagreements
            )

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"IC Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
