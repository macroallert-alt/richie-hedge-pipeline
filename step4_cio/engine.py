"""
step4_cio/engine.py
CIO Agent — 3-Stage Orchestrator (Pre-Processor → LLM → Post-Processor)
Spec: CIO Spec Teil 6 §6.1

Two modes:
  - draft (Step 4): Full Pre-Processor → LLM → Post-Processor
  - final (Step 6): Reuses Draft Pre-Processor → LLM with DA → Post-Processor
  V1: final without DA = draft promoted to final with annotation
"""

import json
import logging
from datetime import date, datetime, timedelta

from step4_cio.preprocessor import (
    validate_inputs,
    calculate_data_quality,
    can_run,
    build_temporal_context,
    check_ic_blind_spot,
    check_near_miss,
    check_extended_calm,
    match_patterns,
    detect_anti_patterns,
    determine_briefing_type,
    compress_ongoing_conditions,
    calculate_confidence_markers,
    calculate_system_conviction,
    assemble_preprocessor_output,
)
from step4_cio.llm import (
    build_system_prompt,
    build_draft_user_prompt,
    build_final_user_prompt,
    call_cio_llm,
)
from step4_cio.postprocessor import (
    validate_output,
    fact_check_briefing,
    handle_fact_check_flags,
    extract_action_items,
    extract_sections,
    parse_sections,
    update_history_digest,
    update_action_item_tracking,
    extract_da_resolution,
    build_cio_resolutions,
    generate_fallback_briefing,
)

logger = logging.getLogger("cio_engine")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HISTORY = {
    "last_updated": None,
    "consecutive_routine_days": 0,
    "active_threads": [],
    "resolved_threads_last_7d": [],
    "patterns_last_7d": {},
    "open_action_items": [],
}


def _extract_fragility_string(val) -> str:
    """Market Analyst returns fragility_state as dict — extract string."""
    if isinstance(val, dict):
        return val.get("state", val.get("level", "HEALTHY"))
    return val if val else "HEALTHY"


def _extract_regime_string(val) -> str:
    """Market Analyst returns system_regime as dict — extract string."""
    if isinstance(val, dict):
        return val.get("regime", "UNKNOWN")
    return val if val else "UNKNOWN"


def is_monday_mode(d: date) -> bool:
    """Monday = special mode. Delta refers to Friday, not yesterday."""
    return d.weekday() == 0


def get_reference_date(d: date) -> str:
    """Reference date for delta calculation."""
    if is_monday_mode(d):
        return (d - timedelta(days=3)).isoformat()
    return (d - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# DRAFT RUN (Step 4)
# ---------------------------------------------------------------------------
def run_cio_draft(inputs: dict, config: dict,
                  cio_history: dict | None = None,
                  yesterday_briefing: dict | None = None) -> dict:
    """
    Full CIO Draft run.
    Called after Risk Officer (Step 3).

    Args:
        inputs: Dict with keys: v16_production, risk_alerts, layer_analysis,
                ic_intelligence, f6_production, signals, beliefs
        config: Parsed CIO_CONFIG.yaml
        cio_history: CIO History Digest (None = first run)
        yesterday_briefing: Yesterday's FINAL_MEMO (None = first run)

    Returns:
        Draft output dict (written to Drive + Sheet by main.py)
    """
    today_date = date.today()
    today_str = today_date.isoformat()
    now_str = datetime.utcnow().strftime("%H%M%S")

    if cio_history is None:
        cio_history = DEFAULT_HISTORY.copy()

    # ===== PRE-PROCESSOR =====
    logger.info("PRE-PROCESSOR: Phase 1 — Input Validation")
    completeness = validate_inputs(inputs, config)
    data_quality = calculate_data_quality(completeness, config)

    if not can_run(completeness, config):
        logger.warning("Pflicht-Inputs missing — generating fallback briefing")
        return _build_fallback_output(
            None, inputs, config, today_str, now_str, "draft"
        )

    logger.info("PRE-PROCESSOR: Phase 2 — History + Temporal Setup")
    monday = is_monday_mode(today_date)
    reference_date = get_reference_date(today_date)

    logger.info("PRE-PROCESSOR: Phase 3 — Temporal Context")
    temporal_context = build_temporal_context(inputs, config, today_date)

    logger.info("PRE-PROCESSOR: Phase 4 — Absence Checks")
    absence_flags = []
    blind_spots = check_ic_blind_spot(
        inputs.get("ic_intelligence", {}),
        inputs.get("layer_analysis", {}),
        config,
    )
    absence_flags.extend(blind_spots)

    near_miss = check_near_miss(inputs.get("v16_production", {}), temporal_context, config)
    if near_miss:
        absence_flags.append(near_miss)

    extended_calm = check_extended_calm(cio_history, inputs.get("risk_alerts", {}), config)
    if extended_calm:
        absence_flags.append(extended_calm)

    logger.info("PRE-PROCESSOR: Phase 5 — Pattern Matching")
    active_patterns = match_patterns(inputs, temporal_context, absence_flags, cio_history, config)
    anti_patterns = detect_anti_patterns(inputs.get("ic_intelligence", {}), config)

    logger.info("PRE-PROCESSOR: Phase 6 — Briefing Type")
    fragility_state = _extract_fragility_string(
        inputs.get("layer_analysis", {}).get("fragility_state", "HEALTHY")
    )
    briefing_type = determine_briefing_type(
        inputs.get("risk_alerts", {}),
        active_patterns,
        inputs.get("ic_intelligence", {}),
        temporal_context,
        fragility_state,
        cio_history,
        config,
    )

    logger.info("PRE-PROCESSOR: Phase 7 — Ongoing Conditions")
    ongoing = compress_ongoing_conditions(inputs.get("risk_alerts", {}), config)

    logger.info("PRE-PROCESSOR: Phase 8 — Confidence + Conviction")
    confidence_markers = calculate_confidence_markers(
        inputs, inputs.get("ic_intelligence", {}), active_patterns, absence_flags
    )
    system_conviction = calculate_system_conviction(
        inputs, inputs.get("ic_intelligence", {}), active_patterns,
        confidence_markers, config,
    )

    logger.info("PRE-PROCESSOR: Phase 9 — Assembly")
    preprocessor_output = assemble_preprocessor_output(
        completeness=completeness,
        data_quality=data_quality,
        temporal_context=temporal_context,
        absence_flags=absence_flags,
        active_patterns=active_patterns,
        anti_patterns=anti_patterns,
        briefing_type=briefing_type,
        ongoing_conditions=ongoing,
        confidence_markers=confidence_markers,
        system_conviction=system_conviction,
        cio_history=cio_history,
        reference_date=reference_date,
        is_monday=monday,
        inputs=inputs,
        today_str=today_str,
    )

    # ===== LLM CALL =====
    logger.info("LLM: Building Draft prompt")
    system_prompt = build_system_prompt()
    user_prompt = build_draft_user_prompt(
        preprocessor_output, inputs, yesterday_briefing
    )

    logger.info("LLM: Calling Claude Sonnet (Draft)")
    llm_result = call_cio_llm(system_prompt, user_prompt, config.get("llm", {}))

    if not llm_result["success"]:
        logger.warning(f"LLM failed: {llm_result.get('error')} — fallback")
        return _build_fallback_output(
            preprocessor_output, inputs, config, today_str, now_str, "draft"
        )

    briefing_text = llm_result["briefing_text"]

    # ===== POST-PROCESSOR =====
    logger.info("POST-PROCESSOR: Step 1 — Output Validation")
    valid, errors = validate_output(briefing_text, is_final=False)
    if not valid:
        logger.warning(f"Output validation failed: {errors} — fallback")
        return _build_fallback_output(
            preprocessor_output, inputs, config, today_str, now_str, "draft"
        )

    logger.info("POST-PROCESSOR: Step 2 — Fact-Check")
    fact_flags = fact_check_briefing(
        briefing_text, inputs, preprocessor_output
    )
    fact_result = handle_fact_check_flags(fact_flags)
    if fact_result["action"] == "RETRY_OR_FALLBACK":
        logger.warning(f"Fact-check CRITICAL: {fact_flags} — fallback")
        return _build_fallback_output(
            preprocessor_output, inputs, config, today_str, now_str, "draft"
        )

    logger.info("POST-PROCESSOR: Step 3 — Action Items Extraction")
    action_items = extract_action_items(briefing_text, preprocessor_output)

    logger.info("POST-PROCESSOR: Step 4 — History Digest Update")
    new_history = update_history_digest(
        cio_history, briefing_text, inputs, preprocessor_output
    )

    logger.info("POST-PROCESSOR: Step 5 — Action Item Tracking")
    updated_action_items = update_action_item_tracking(
        cio_history.get("open_action_items", []),
        action_items, inputs, preprocessor_output, config,
    )
    new_history["open_action_items"] = updated_action_items

    # Parse sections for dashboard
    section_texts = parse_sections(briefing_text)
    section_word_counts = {k: len(v.split()) for k, v in section_texts.items()}

    # Build Draft Output
    draft_output = {
        "date": today_str,
        "run_id": f"cio_draft_{today_str.replace('-', '')}_{now_str}",
        "generation_model": config.get("llm", {}).get("model", "claude-sonnet-4-5-20250929"),
        "temperature": config.get("llm", {}).get("temperature", 0.3),
        "briefing_type": briefing_type,
        "system_conviction": system_conviction,
        "risk_ampel": inputs.get("risk_alerts", {}).get("portfolio_status", "GREEN"),
        "fragility_state": fragility_state,
        "data_quality": data_quality,
        "v16_regime": inputs.get("v16_production", {}).get("regime", "UNKNOWN"),
        "briefing_text": briefing_text,
        "section_texts": section_texts,
        "section_word_counts": section_word_counts,
        "action_items": updated_action_items,
        "confidence_markers": confidence_markers,
        "fact_check_flags": fact_flags,
        "fact_check_corrections_count": len(fact_flags),
        "is_fallback": False,
        "metadata": {
            "sections_present": extract_sections(briefing_text),
            "word_count": len(briefing_text.split()),
            "active_patterns_class_a": [p["pattern"] for p in active_patterns],
            "active_patterns_class_b": [],
            "absences_flagged": [f["type"] for f in absence_flags],
            "llm_attempts": llm_result.get("attempt", 1),
        },
        "preprocessor_output": preprocessor_output,
        "cio_history_digest": new_history,
    }

    logger.info(
        f"CIO Draft complete. Type={briefing_type}, "
        f"Conviction={system_conviction}, Words={len(briefing_text.split())}"
    )
    return draft_output


# ---------------------------------------------------------------------------
# FINAL RUN (Step 6)
# ---------------------------------------------------------------------------
def run_cio_final(inputs: dict, draft_output: dict,
                  devils_advocate: dict | None, config: dict) -> dict:
    """
    Full CIO Final run.
    Called after Devil's Advocate (Step 5).

    V1: DA not built yet → draft promoted to final with annotation.

    Args:
        inputs: Same inputs dict as draft run
        draft_output: Output from run_cio_draft()
        devils_advocate: DA output (None if DA not available)
        config: Parsed CIO_CONFIG.yaml

    Returns:
        Final output dict
    """
    today_str = draft_output["date"]
    now_str = datetime.utcnow().strftime("%H%M%S")

    # DA not available → Draft = Final (Spec Teil 1 §1.2, Teil 6 §6.1)
    if devils_advocate is None:
        logger.info("DA not available — promoting Draft to Final")
        return _promote_draft_to_final(draft_output, today_str, now_str)

    # DA available — full Final run
    preprocessor_output = draft_output["preprocessor_output"]

    # ===== LLM CALL =====
    logger.info("LLM: Building Final prompt (with DA challenges)")
    system_prompt = build_system_prompt()
    user_prompt = build_final_user_prompt(
        preprocessor_output, inputs,
        draft_output["briefing_text"],
        devils_advocate,
    )

    logger.info("LLM: Calling Claude Sonnet (Final)")
    llm_result = call_cio_llm(system_prompt, user_prompt, config.get("llm", {}))

    if not llm_result["success"]:
        logger.warning("Final LLM failed — promoting Draft to Final")
        return _promote_draft_to_final(draft_output, today_str, now_str)

    briefing_text = llm_result["briefing_text"]

    # ===== POST-PROCESSOR =====
    logger.info("POST-PROCESSOR (Final): Validation")
    valid, errors = validate_output(briefing_text, is_final=True)
    if not valid:
        logger.warning(f"Final validation failed: {errors} — using Draft")
        return _promote_draft_to_final(draft_output, today_str, now_str)

    logger.info("POST-PROCESSOR (Final): Fact-Check")
    fact_flags = fact_check_briefing(briefing_text, inputs, preprocessor_output)
    fact_result = handle_fact_check_flags(fact_flags)
    if fact_result["action"] == "RETRY_OR_FALLBACK":
        logger.warning("Final fact-check failed — using Draft")
        return _promote_draft_to_final(draft_output, today_str, now_str)

    logger.info("POST-PROCESSOR (Final): Action Items + DA Resolution")
    action_items = extract_action_items(briefing_text, preprocessor_output)
    da_resolution = extract_da_resolution(briefing_text)

    # DA Resolution Write-back: match resolutions to DA challenge IDs
    # and update da_history effectiveness tracking
    da_history_updated = None
    if devils_advocate:
        try:
            cio_resolutions = build_cio_resolutions(da_resolution, devils_advocate)
            if cio_resolutions:
                from step5_devils_advocate.postprocessor import update_effectiveness_after_cio_final
                da_history_raw = inputs.get("da_history")
                if da_history_raw:
                    import copy
                    da_history_copy = copy.deepcopy(da_history_raw)
                    da_history_updated = update_effectiveness_after_cio_final(
                        da_history_copy, cio_resolutions
                    )
                    logger.info(
                        f"DA Write-back: {len(cio_resolutions)} resolutions processed, "
                        f"acceptance_rate={da_history_updated.get('challenge_effectiveness', {}).get('acceptance_rate_overall', 0):.0%}"
                    )
                else:
                    logger.warning("DA Write-back: da_history not available — skipping")
            else:
                logger.info("DA Write-back: No resolutions to write back")
        except Exception as e:
            logger.error(f"DA Write-back failed (non-fatal): {e}")

    section_texts = parse_sections(briefing_text)
    section_word_counts = {k: len(v.split()) for k, v in section_texts.items()}

    final_output = {
        "date": today_str,
        "run_id": f"cio_final_{today_str.replace('-', '')}_{now_str}",
        "generation_model": config.get("llm", {}).get("model", "claude-sonnet-4-5-20250929"),
        "temperature": config.get("llm", {}).get("temperature", 0.3),
        "briefing_type": preprocessor_output["header"]["briefing_type"],
        "system_conviction": preprocessor_output["header"]["system_conviction"],
        "risk_ampel": inputs.get("risk_alerts", {}).get("portfolio_status", "GREEN"),
        "fragility_state": _extract_fragility_string(
            inputs.get("layer_analysis", {}).get("fragility_state", "UNKNOWN")
        ),
        "data_quality": preprocessor_output["header"]["data_quality"],
        "v16_regime": inputs.get("v16_production", {}).get("regime", "UNKNOWN"),
        "briefing_text": briefing_text,
        "section_texts": section_texts,
        "section_word_counts": section_word_counts,
        "da_resolution": da_resolution,
        "da_markers": da_resolution.get("details", []),
        "action_items": action_items,
        "confidence_markers": draft_output.get("confidence_markers", []),
        "fact_check_flags": fact_flags,
        "fact_check_corrections_count": len(fact_flags),
        "is_fallback": False,
        "is_draft_as_final": False,
        "metadata": {
            "sections_present": extract_sections(briefing_text),
            "sections_modified_by_da": da_resolution.get("modified_sections", []),
            "word_count": len(briefing_text.split()),
            "llm_attempts": llm_result.get("attempt", 1),
            "fact_check_flags": fact_flags,
        },
        "preprocessor_output": preprocessor_output,
        "cio_history_digest": draft_output.get("cio_history_digest", DEFAULT_HISTORY),
        "da_history_updated": da_history_updated,
    }

    logger.info(
        f"CIO Final complete. DA challenges={da_resolution.get('total_challenges', 0)}, "
        f"Words={len(briefing_text.split())}"
    )
    return final_output


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _promote_draft_to_final(draft_output: dict, today_str: str, now_str: str) -> dict:
    """
    Promote Draft to Final when DA is unavailable or Final generation fails.
    Spec: Teil 1 §1.2, Teil 6 §6.1
    """
    briefing_text = draft_output["briefing_text"]
    annotation = (
        "\n\n---\n"
        "Devil's Advocate nicht verfuegbar — Draft als Final uebernommen.\n"
        "---"
    )
    briefing_text_final = briefing_text + annotation

    section_texts = parse_sections(briefing_text_final)
    section_word_counts = {k: len(v.split()) for k, v in section_texts.items()}

    return {
        "date": today_str,
        "run_id": f"cio_final_{today_str.replace('-', '')}_{now_str}",
        "generation_model": draft_output.get("generation_model", ""),
        "temperature": draft_output.get("temperature", 0.3),
        "briefing_type": draft_output["briefing_type"],
        "system_conviction": draft_output["system_conviction"],
        "risk_ampel": draft_output["risk_ampel"],
        "fragility_state": draft_output["fragility_state"],
        "data_quality": draft_output["data_quality"],
        "v16_regime": draft_output["v16_regime"],
        "briefing_text": briefing_text_final,
        "section_texts": section_texts,
        "section_word_counts": section_word_counts,
        "da_resolution": {
            "total_challenges": 0,
            "accepted": 0,
            "noted": 0,
            "rejected": 0,
            "details": [],
            "modified_sections": [],
        },
        "da_markers": [],
        "action_items": draft_output.get("action_items", []),
        "confidence_markers": draft_output.get("confidence_markers", []),
        "fact_check_flags": draft_output.get("fact_check_flags", []),
        "fact_check_corrections_count": draft_output.get("fact_check_corrections_count", 0),
        "is_fallback": draft_output.get("is_fallback", False),
        "is_draft_as_final": True,
        "metadata": {
            **draft_output.get("metadata", {}),
            "sections_modified_by_da": [],
            "da_skipped": True,
        },
        "preprocessor_output": draft_output.get("preprocessor_output", {}),
        "cio_history_digest": draft_output.get("cio_history_digest", DEFAULT_HISTORY),
    }


def _build_fallback_output(preprocessor_output: dict | None, inputs: dict,
                           config: dict, today_str: str, now_str: str,
                           mode: str) -> dict:
    """
    Generate fallback output when LLM or validation fails.
    Uses deterministic fallback briefing from postprocessor.
    """
    briefing_text = generate_fallback_briefing(preprocessor_output, inputs)
    section_texts = parse_sections(briefing_text)
    section_word_counts = {k: len(v.split()) for k, v in section_texts.items()}

    header = {}
    if preprocessor_output:
        header = preprocessor_output.get("header", {})

    run_prefix = "cio_draft" if mode == "draft" else "cio_final"

    return {
        "date": today_str,
        "run_id": f"{run_prefix}_{today_str.replace('-', '')}_{now_str}",
        "generation_model": "FALLBACK",
        "temperature": 0.0,
        "briefing_type": header.get("briefing_type", "WATCH"),
        "system_conviction": header.get("system_conviction", "LOW"),
        "risk_ampel": header.get("risk_ampel",
                                  inputs.get("risk_alerts", {}).get("portfolio_status", "GREEN")),
        "fragility_state": _extract_fragility_string(header.get("fragility_state", "UNKNOWN")),
        "data_quality": header.get("data_quality", "DEGRADED"),
        "v16_regime": header.get("v16_regime",
                                  inputs.get("v16_production", {}).get("regime", "UNKNOWN")),
        "briefing_text": briefing_text,
        "section_texts": section_texts,
        "section_word_counts": section_word_counts,
        "da_resolution": {
            "total_challenges": 0, "accepted": 0, "noted": 0, "rejected": 0,
            "details": [], "modified_sections": [],
        },
        "da_markers": [],
        "action_items": [],
        "confidence_markers": [],
        "fact_check_flags": [],
        "fact_check_corrections_count": 0,
        "is_fallback": True,
        "is_draft_as_final": mode == "final",
        "metadata": {
            "sections_present": extract_sections(briefing_text),
            "word_count": len(briefing_text.split()),
            "fallback_reason": "LLM_FAILURE",
        },
        "preprocessor_output": preprocessor_output or {},
        "cio_history_digest": DEFAULT_HISTORY,
    }
