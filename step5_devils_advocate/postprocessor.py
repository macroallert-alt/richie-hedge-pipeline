"""
step5_devils_advocate/postprocessor.py
Devil's Advocate Post-Processor — 6 Steps, Deterministic

Step 1: Challenge Validation (incl. Master Protection)
Step 2: Persistent Challenge Merger
Step 3: DA History Update
Step 4: Perspective Seed Rotation Check
Step 5: Output Formatting
Step 6: Write (Google Sheets)

Source: DA Spec Teil 4
"""

import json
import logging
from datetime import date, datetime, timedelta

logger = logging.getLogger("da_postprocessor")


# =============================================================================
# STEP 1: CHALLENGE VALIDATION (Spec Teil 4 §4.2)
# =============================================================================

def validate_challenges(challenges: list, preprocessor_output: dict,
                        inputs: dict, config: dict) -> tuple:
    """
    Validate LLM-generated challenges.
    Returns (valid: bool, cleaned_challenges: list, errors: list).
    """
    errors = []
    cleaned = []

    min_required = preprocessor_output["asymmetry"]["min_challenges"]
    max_allowed = config.get("challenges", {}).get("max_per_run", 5)

    # Check: minimum challenges
    if len(challenges) < min_required:
        errors.append({
            "type": "TOO_FEW_CHALLENGES",
            "expected_min": min_required,
            "actual": len(challenges),
            "action": "RETRY",
        })
        return (False, [], errors)

    # Check: maximum challenges — trim to best 5
    if len(challenges) > max_allowed:
        severity_order = {"SUBSTANTIVE": 0, "MODERATE": 1, "MINOR": 2}
        challenges = sorted(challenges, key=lambda c: severity_order.get(c.get("severity"), 99))
        challenges = challenges[:max_allowed]
        errors.append({
            "type": "TOO_MANY_CHALLENGES",
            "original_count": len(challenges),
            "trimmed_to": max_allowed,
            "action": "TRIMMED",
        })

    # Load master protection phrases from config
    master_cfg = config.get("master_protection", {})

    for challenge in challenges:
        challenge_errors = []

        # Mandatory fields
        if not challenge.get("challenge_text"):
            challenge_errors.append("MISSING_CHALLENGE_TEXT")
        if not challenge.get("type"):
            challenge_errors.append("MISSING_TYPE")
        if not challenge.get("severity"):
            challenge_errors.append("MISSING_SEVERITY")

        # Type validation
        valid_types = ["NARRATIVE", "UNASKED_QUESTION", "PREMISE_ATTACK"]
        if challenge.get("type") not in valid_types:
            challenge_errors.append(f"INVALID_TYPE: {challenge.get('type')}")
            challenge["type"] = "PREMISE_ATTACK"

        # Target section validation
        valid_sections = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "SYSTEM", None]
        if challenge.get("target_section") not in valid_sections:
            challenge_errors.append(f"INVALID_TARGET_SECTION: {challenge.get('target_section')}")
            challenge["target_section"] = "SYSTEM"

        # Target assumption validation
        if challenge.get("target_assumption"):
            valid_kas = [ka.get("id") for ka in inputs.get("draft_memo", {}).get("key_assumptions", [])]
            if valid_kas and challenge["target_assumption"] not in valid_kas:
                challenge_errors.append(
                    f"INVALID_TARGET_ASSUMPTION: {challenge['target_assumption']} not in {valid_kas}"
                )
                challenge["target_assumption"] = None

        # Evidence check (non-fatal)
        if not challenge.get("evidence") or len(challenge["evidence"]) == 0:
            challenge_errors.append("MISSING_EVIDENCE")

        # Master Protection check (FATAL — challenge removed)
        text = (challenge.get("challenge_text", "") + " " +
                " ".join(challenge.get("evidence", [])))
        violations = _check_master_protection(text, master_cfg)
        if violations:
            challenge_errors.append(f"MASTER_PROTECTION_VIOLATION: {violations}")
            logger.warning(f"Challenge {challenge.get('id')} removed: Master Protection violation")
            continue  # Skip this challenge entirely

        if challenge_errors:
            challenge["validation_warnings"] = challenge_errors

        cleaned.append(challenge)

    # Check if enough valid challenges remain
    if len(cleaned) < min_required:
        errors.append({
            "type": "INSUFFICIENT_VALID_CHALLENGES",
            "valid_count": len(cleaned),
            "required": min_required,
            "action": "RETRY",
        })
        return (False, cleaned, errors)

    all_ok = len(errors) == 0 or all(e.get("action") != "RETRY" for e in errors)
    return (all_ok, cleaned, errors)


def _check_master_protection(text: str, master_cfg: dict) -> list | None:
    """Check if DA output violates Master Protection rules."""
    violations = []
    lower = text.lower()

    phrase_groups = {
        "V16_ATTACK": master_cfg.get("v16_attack_phrases", []),
        "F6_ATTACK": master_cfg.get("f6_attack_phrases", []),
        "TRADE_RECOMMENDATION": master_cfg.get("trade_phrases", []),
        "RO_SEVERITY_CHANGE": master_cfg.get("ro_severity_phrases", []),
    }

    for violation_type, phrases in phrase_groups.items():
        for phrase in phrases:
            if phrase.lower() in lower:
                violations.append(f"{violation_type}: '{phrase}'")

    return violations if violations else None


# =============================================================================
# STEP 2: PERSISTENT CHALLENGE MERGER (Spec Teil 4 §4.3)
# =============================================================================

def merge_persistent_challenges(llm_challenges: list,
                                persistent_from_history: list,
                                forced_from_history: list) -> list:
    """
    Merge LLM challenges with persistent/forced challenges from history.
    - If LLM renewed a persistent challenge: mark as persistent
    - If LLM missed a persistent challenge: add from history
    - Forced decision challenges: always add prominently
    """
    merged = list(llm_challenges)

    for pc in persistent_from_history:
        renewed = _find_matching_challenge(merged, pc)
        if renewed:
            renewed["is_persistent"] = True
            renewed["persistent_days"] = pc.get("days_open", 0) + 1
            renewed["original_id"] = pc["id"]
        else:
            # LLM missed it — insert from history
            from_history = {
                "id": pc["id"],
                "type": pc.get("challenge_type", "PREMISE_ATTACK"),
                "target_section": pc.get("target_section"),
                "target_assumption": pc.get("target_assumption"),
                "severity": "SUBSTANTIVE",
                "challenge_text": (
                    pc.get("challenge_text", "") +
                    f"\n[PERSISTENT: Tag {pc.get('days_open', 0) + 1}, "
                    f"erneuert aus History]"
                ),
                "evidence": pc.get("evidence", []),
                "is_persistent": True,
                "persistent_days": pc.get("days_open", 0) + 1,
                "original_id": pc["id"],
            }
            merged.append(from_history)

    # Forced decision challenges — always at front
    for fd in forced_from_history:
        fd_challenge = {
            "id": fd["id"],
            "type": fd.get("challenge_type", "PREMISE_ATTACK"),
            "target_section": fd.get("target_section"),
            "target_assumption": fd.get("target_assumption"),
            "severity": "SUBSTANTIVE",
            "challenge_text": (
                f"FORCED DECISION (Tag {fd.get('days_open', 0) + 1}, "
                f"{fd.get('noted_count', 3)}x NOTED): "
                f"{fd.get('challenge_text', '')}\n"
                f"CIO MUSS mit ACCEPTED oder REJECTED antworten. "
                f"NOTED ist nicht mehr erlaubt."
            ),
            "evidence": fd.get("evidence", []),
            "is_persistent": True,
            "persistent_days": fd.get("days_open", 0) + 1,
            "is_forced_decision": True,
            "original_id": fd["id"],
        }
        merged.insert(0, fd_challenge)

    return merged


def _find_matching_challenge(challenges: list, persistent: dict):
    """Find an LLM challenge matching a persistent challenge by target."""
    for c in challenges:
        # Match by section + assumption
        if (c.get("target_section") == persistent.get("target_section") and
                c.get("target_assumption") == persistent.get("target_assumption") and
                persistent.get("target_section") is not None):
            return c
        # Fallback: text similarity (first 50 chars)
        pc_text = persistent.get("challenge_text", "")[:50].lower()
        c_text = c.get("challenge_text", "").lower()
        if pc_text and pc_text in c_text:
            return c
    return None


# =============================================================================
# STEP 3: DA HISTORY UPDATE (Spec Teil 4 §4.4)
# =============================================================================

def update_da_history(da_history: dict | None, new_challenges: list,
                      preprocessor_output: dict) -> dict:
    """
    Update DA History with new challenges.
    NOTE: Acceptance rate is NOT updated here — that happens after CIO Final.
    """
    today_str = date.today().isoformat()

    if da_history is None:
        da_history = {
            "last_updated": None,
            "open_challenges": [],
            "resolved_challenges_7d": [],
            "challenge_effectiveness": {
                "total_30d": 0, "accepted_30d": 0, "noted_30d": 0, "rejected_30d": 0,
                "acceptance_rate_overall": 0, "acceptance_rate_narrative": 0,
                "acceptance_rate_unasked": 0, "acceptance_rate_premise": 0,
                "noted_rate_overall": 0,
            },
            "perspective_seed_index": 0,
            "last_seed_rotation": today_str,
        }

    # Build updated open challenges
    updated_open = []

    # Keep existing open challenges that weren't renewed
    renewed_ids = {nc.get("original_id") for nc in new_challenges if nc.get("is_persistent")}
    for existing in da_history.get("open_challenges", []):
        if existing["id"] in renewed_ids:
            continue  # Replaced by renewed version
        existing["days_open"] = existing.get("days_open", 0) + 1
        updated_open.append(existing)

    # Add new challenges
    for nc in new_challenges:
        # Get previous CIO responses if persistent
        prev_responses = []
        if nc.get("is_persistent") and nc.get("original_id"):
            for oc in da_history.get("open_challenges", []):
                if oc["id"] == nc.get("original_id"):
                    prev_responses = oc.get("cio_responses", [])
                    break

        entry = {
            "id": nc.get("original_id", nc["id"]),
            "first_raised": today_str if not nc.get("is_persistent") else _get_first_raised(da_history, nc),
            "challenge_text": nc["challenge_text"],
            "challenge_type": nc["type"],
            "target_section": nc.get("target_section"),
            "target_assumption": nc.get("target_assumption"),
            "evidence": nc.get("evidence", []),
            "severity": nc["severity"],
            "cio_responses": prev_responses,
            "days_open": nc.get("persistent_days", 1),
            "escalated": nc.get("is_forced_decision", False),
        }
        updated_open.append(entry)

    # Clean resolved challenges older than 7 days
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    resolved_7d = [
        r for r in da_history.get("resolved_challenges_7d", [])
        if r.get("resolved_date", "2000-01-01") > cutoff
    ]

    return {
        "last_updated": today_str,
        "open_challenges": updated_open,
        "resolved_challenges_7d": resolved_7d,
        "challenge_effectiveness": da_history.get("challenge_effectiveness", {}),
        "perspective_seed_index": preprocessor_output["perspective_seed"]["seed_index"],
        "last_seed_rotation": da_history.get("last_seed_rotation", today_str),
    }


def _get_first_raised(da_history: dict, challenge: dict) -> str:
    """Get original first_raised date for a persistent challenge."""
    original_id = challenge.get("original_id")
    if original_id:
        for oc in da_history.get("open_challenges", []):
            if oc["id"] == original_id:
                return oc.get("first_raised", date.today().isoformat())
    return date.today().isoformat()


# =============================================================================
# FUNCTIONS CALLED BY CIO POST-PROCESSOR (bidirectional write-back)
# =============================================================================

def update_effectiveness_after_cio_final(da_history: dict,
                                         cio_resolutions: list) -> dict:
    """
    Called by CIO Post-Processor AFTER CIO Final run.
    Updates acceptance rate based on ACCEPTED/NOTED/REJECTED.
    This is the ONLY bidirectional data flow in the system.
    """
    today_str = date.today().isoformat()
    eff = da_history.get("challenge_effectiveness", {})

    for res in cio_resolutions:
        challenge_id = res.get("challenge_id")
        result = res.get("resolution")  # ACCEPTED | NOTED | REJECTED
        challenge_type = res.get("challenge_type", "PREMISE_ATTACK")

        eff["total_30d"] = eff.get("total_30d", 0) + 1

        if result == "ACCEPTED":
            eff["accepted_30d"] = eff.get("accepted_30d", 0) + 1
            _move_to_resolved(da_history, challenge_id, today_str, result)
        elif result == "NOTED":
            eff["noted_30d"] = eff.get("noted_30d", 0) + 1
            _add_cio_response(da_history, challenge_id, today_str, result)
        elif result == "REJECTED":
            eff["rejected_30d"] = eff.get("rejected_30d", 0) + 1
            _move_to_resolved(da_history, challenge_id, today_str, result)

        # Per-type tracking
        type_total_key = f"total_{challenge_type.lower()}_30d"
        type_accepted_key = f"accepted_{challenge_type.lower()}_30d"
        eff[type_total_key] = eff.get(type_total_key, 0) + 1
        if result == "ACCEPTED":
            eff[type_accepted_key] = eff.get(type_accepted_key, 0) + 1

        # Update per-type acceptance rate
        type_total = eff.get(type_total_key, 1)
        type_accepted = eff.get(type_accepted_key, 0)
        rate_key = f"acceptance_rate_{challenge_type.lower()}"
        eff[rate_key] = type_accepted / type_total if type_total > 0 else 0

    # Update overall rates
    total = eff.get("total_30d", 1)
    if total > 0:
        eff["acceptance_rate_overall"] = eff.get("accepted_30d", 0) / total
        eff["noted_rate_overall"] = eff.get("noted_30d", 0) / total

    da_history["challenge_effectiveness"] = eff
    return da_history


def _move_to_resolved(da_history: dict, challenge_id: str,
                      resolved_date: str, resolution: str) -> None:
    """Move a challenge from open to resolved."""
    open_challenges = da_history.get("open_challenges", [])
    resolved = da_history.get("resolved_challenges_7d", [])

    for i, oc in enumerate(open_challenges):
        if oc["id"] == challenge_id:
            oc["resolved_date"] = resolved_date
            oc["final_resolution"] = resolution
            resolved.append(oc)
            open_challenges.pop(i)
            break

    da_history["open_challenges"] = open_challenges
    da_history["resolved_challenges_7d"] = resolved


def _add_cio_response(da_history: dict, challenge_id: str,
                      response_date: str, resolution: str) -> None:
    """Add a CIO response to an open challenge."""
    for oc in da_history.get("open_challenges", []):
        if oc["id"] == challenge_id:
            oc.setdefault("cio_responses", []).append({
                "date": response_date,
                "resolution": resolution,
            })
            break


# =============================================================================
# STEP 4: PERSPECTIVE SEED ROTATION CHECK (Spec Teil 4 §4.5)
# =============================================================================

def check_seed_rotation(da_history: dict | None, today: date, config: dict) -> dict:
    """Check if perspective seed needs rotation."""
    interval = config.get("perspective_seeds", {}).get("rotation_interval_trading_days", 20)
    seed_count = len(config.get("perspective_seeds", {}).get("seeds", []))
    if seed_count == 0:
        seed_count = 5

    if da_history is None:
        return {"rotation_needed": False, "new_index": 0, "new_rotation_date": str(today)}

    last_rotation = da_history.get("last_seed_rotation")
    current_index = da_history.get("perspective_seed_index", 0)

    if last_rotation:
        try:
            last_date = datetime.strptime(last_rotation, "%Y-%m-%d").date()
            calendar_days = (today - last_date).days
            trading_days_approx = int(calendar_days * 5 / 7)
            if trading_days_approx >= interval:
                new_index = (current_index + 1) % seed_count
                return {"rotation_needed": True, "new_index": new_index,
                        "new_rotation_date": str(today)}
        except (ValueError, TypeError):
            pass

    return {"rotation_needed": False, "new_index": current_index,
            "new_rotation_date": last_rotation or str(today)}


# =============================================================================
# STEP 5: OUTPUT FORMATTING (Spec Teil 4 §4.6)
# =============================================================================

def format_da_output(validated_challenges: list, preprocessor_output: dict,
                     da_history_updated: dict, inputs: dict,
                     llm_result: dict) -> dict:
    """Build the complete DA output object for CIO Final."""
    today_str = date.today().isoformat()
    now_str = datetime.utcnow().strftime("%H%M%S")

    persistent_renewed = [c for c in validated_challenges if c.get("is_persistent")]
    forced_decision = [c for c in validated_challenges if c.get("is_forced_decision")]

    flags_unused = preprocessor_output["flags"].get("unused", [])

    draft_memo = inputs.get("draft_memo", {})

    return {
        "date": today_str,
        "run_id": f"da_{today_str.replace('-', '')}_{now_str}",
        "draft_run_id": draft_memo.get("run_id", "unknown"),
        "primary_focus": preprocessor_output["focus"]["primary_focus"],
        "perspective_seed": preprocessor_output["perspective_seed"]["seed_label"],
        "challenges": [
            {
                "id": c["id"],
                "type": c["type"],
                "target_section": c.get("target_section"),
                "target_assumption": c.get("target_assumption"),
                "challenge_text": c["challenge_text"],
                "evidence": c.get("evidence", []),
                "severity": c["severity"],
                "is_persistent": c.get("is_persistent", False),
                "persistent_days": c.get("persistent_days", 0),
                "is_forced_decision": c.get("is_forced_decision", False),
                "validation_warnings": c.get("validation_warnings", []),
            }
            for c in validated_challenges
        ],
        "preprocessor_flags_unused": [
            {
                "flag_type": f.get("flag_type"),
                "detail": f.get("detail"),
                "significance": f.get("significance"),
            }
            for f in flags_unused
        ],
        "persistent_challenges_renewed": [
            {
                "original_id": c.get("original_id"),
                "original_challenge": c["challenge_text"][:100],
                "days_open": c.get("persistent_days", 0),
                "cio_response_history": _get_response_history(da_history_updated, c),
                "escalation_note": (
                    f"FORCED DECISION nach {c.get('persistent_days', 0)} Tagen"
                    if c.get("is_forced_decision") else
                    f"Tag {c.get('persistent_days', 0)}, erneuert"
                ),
            }
            for c in persistent_renewed
        ],
        "confidence_saturation": preprocessor_output["confidence_saturation"],
        "metadata": {
            "total_challenges": len(validated_challenges),
            "by_type": {
                "NARRATIVE": sum(1 for c in validated_challenges if c["type"] == "NARRATIVE"),
                "UNASKED_QUESTION": sum(1 for c in validated_challenges if c["type"] == "UNASKED_QUESTION"),
                "PREMISE_ATTACK": sum(1 for c in validated_challenges if c["type"] == "PREMISE_ATTACK"),
            },
            "by_severity": {
                "SUBSTANTIVE": sum(1 for c in validated_challenges if c["severity"] == "SUBSTANTIVE"),
                "MODERATE": sum(1 for c in validated_challenges if c["severity"] == "MODERATE"),
                "MINOR": sum(1 for c in validated_challenges if c["severity"] == "MINOR"),
            },
            "persistent_renewed": len(persistent_renewed),
            "forced_decisions": len(forced_decision),
            "preprocessor_flags_total": preprocessor_output["flags"]["total_found"],
            "preprocessor_flags_used": preprocessor_output["flags"]["sent_to_llm"],
            "llm_model": "claude-sonnet-4-5-20250929",
            "llm_temperature": 0.4,
            "llm_attempt": llm_result.get("attempt", 1),
            "generation_time_seconds": llm_result.get("generation_time_seconds", 0),
        },
    }


def _get_response_history(da_history: dict, challenge: dict) -> list:
    """Get CIO response history for a challenge."""
    original_id = challenge.get("original_id")
    if original_id:
        for oc in da_history.get("open_challenges", []):
            if oc["id"] == original_id:
                return [r.get("resolution") for r in oc.get("cio_responses", [])]
    return []


# =============================================================================
# STEP 6: WRITE — AGENT_SUMMARY (Spec Teil 4 §4.7)
# =============================================================================

def write_agent_summary(sheets_service, da_output: dict) -> None:
    """Write DA summary row to AGENT_SUMMARY tab in DW Sheet."""
    if sheets_service is None:
        return

    DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
    today_str = date.today().isoformat()
    meta = da_output.get("metadata", {})

    row = [
        today_str,
        "Step5_DevilsAdvocate",
        f"{meta.get('total_challenges', 0)} challenges",
        da_output.get("primary_focus", ""),
        da_output.get("perspective_seed", ""),
        f"SUBST:{meta.get('by_severity', {}).get('SUBSTANTIVE', 0)} "
        f"MOD:{meta.get('by_severity', {}).get('MODERATE', 0)} "
        f"MIN:{meta.get('by_severity', {}).get('MINOR', 0)}",
        f"Persistent:{meta.get('persistent_renewed', 0)} "
        f"Forced:{meta.get('forced_decisions', 0)}",
        f"Flags:{meta.get('preprocessor_flags_total', 0)}/{meta.get('preprocessor_flags_used', 0)}",
        da_output.get("run_id", ""),
    ]

    try:
        sheets_service.spreadsheets().values().append(
            spreadsheetId=DW_SHEET_ID,
            range="AGENT_SUMMARY!A:I",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
        logger.info(f"AGENT_SUMMARY: DA row written ({meta.get('total_challenges', 0)} challenges)")
    except Exception as e:
        logger.error(f"AGENT_SUMMARY write failed: {e}")
