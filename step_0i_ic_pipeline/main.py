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
) -> dict:
    """
    Map IC Pipeline output to dashboard.json intelligence block.
    Format matches IntelDetail.jsx expectations exactly:
      consensus[TOPIC] = {score, direction, sources, confidence}
      divergences[] = {theme, divergence_type, magnitude, ic_signal, dc_signal, ...}
      high_novelty_claims[] = {source, claim, novelty, signal, theme}
      catalyst_timeline[] = {event, date, days_until, impact, themes}
      source_cards[] = {source_id, source_name, tier, active_claims, ...}
      cadence_anomalies[] = {source_id, anomaly_level, cadence_ratio, ...}  (NEW Phase 2)
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

    return {
        "status": "AVAILABLE",
        "consensus": consensus_out,
        "divergences": divergences_out,
        "divergences_count": len(divergences_out),
        "high_novelty_claims": claims_out,
        "catalyst_timeline": catalysts_out,
        "source_cards": source_cards,
        "cadence_anomalies": cadence_anomalies or [],
    }


def update_dashboard_json(
    intel: dict,
    briefing: dict,
    active_claims_count: int,
    claims_archive: dict,
    sources_config: list[dict],
    cadence_anomalies: list[dict] | None = None,
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

        # Replace intelligence block (with source_cards + cadence_anomalies)
        dashboard["intelligence"] = build_intelligence_block(
            intel, briefing, claims_archive, sources_config, cadence_anomalies
        )

        # Update pipeline health
        now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        new_claims = intel.get("extraction_summary", {}).get("total_claims", 0)
        anomaly_count = len(cadence_anomalies) if cadence_anomalies else 0
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
                claims_archive, sources, cadence_anomalies
            )

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"IC Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
