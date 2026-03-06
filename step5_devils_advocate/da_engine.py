"""
step5_devils_advocate/engine.py
Devil's Advocate Engine — Orchestrator

Runs: Pre-Processor (8 phases) → LLM Call → Post-Processor (6 steps)
Source: DA Spec Teil 4 §4.8
"""

import logging
import time
from datetime import date

from .preprocessor import (
    validate_inputs,
    parse_draft,
    run_omission_detection,
    check_internal_consistency,
    detect_drift,
    check_confidence_saturation,
    load_da_history_and_persistent,
    select_focus,
    select_perspective_seed,
    calculate_asymmetry,
    assemble_preprocessor_output,
)
from .llm import (
    build_user_prompt,
    call_da_llm,
)
from .postprocessor import (
    validate_challenges,
    merge_persistent_challenges,
    update_da_history,
    check_seed_rotation,
    format_da_output,
)

logger = logging.getLogger("da_engine")


def run_devils_advocate(inputs: dict, config: dict) -> dict:
    """
    Full Devil's Advocate run.
    Called after CIO Draft (Step 4), before CIO Final (Step 6).

    Returns:
        {"success": True, "da_output": {...}, "da_history": {...}}
        or
        {"success": False, "reason": "...", "action": "Draft als Final"}
    """
    start_time = time.time()
    today = date.today()

    # ================================================================
    # PRE-PROCESSOR
    # ================================================================

    logger.info("=" * 60)
    logger.info("DEVIL'S ADVOCATE (Step 5)")
    logger.info("=" * 60)

    # Phase 1: Input Validation + Draft Parsing
    logger.info("PRE-PROCESSOR: Phase 1 — Input Validation + Draft Parsing")
    validation = validate_inputs(inputs)
    if not validation["can_run"]:
        logger.error(f"DA cannot run: {validation['reason']}")
        return {
            "success": False,
            "reason": validation["reason"],
            "action": "CIO Draft wird als Final uebernommen",
        }

    parsed_draft = parse_draft(inputs["draft_memo"], config)
    logger.info(f"  Draft parsed: {parsed_draft['word_count']} words, "
                f"{len(parsed_draft['sections'])} sections, "
                f"{len(parsed_draft['key_assumptions'])} key assumptions")

    # Phase 2: Omission Detection
    logger.info("PRE-PROCESSOR: Phase 2 — Omission Detection")
    omissions = run_omission_detection(parsed_draft, inputs, config)
    logger.info(f"  {len(omissions)} omissions found")

    # Phase 3: Internal Consistency Check
    logger.info("PRE-PROCESSOR: Phase 3 — Internal Consistency Check")
    consistency_flags = check_internal_consistency(parsed_draft, inputs["draft_memo"], config)
    logger.info(f"  {len(consistency_flags)} consistency flags")

    # Phase 4: Drift Detection
    logger.info("PRE-PROCESSOR: Phase 4 — Drift Detection")
    drift_flags = detect_drift(
        parsed_draft, inputs["draft_memo"],
        inputs.get("yesterday_final"), inputs, config,
    )
    logger.info(f"  {len(drift_flags)} drift flags")

    # Phase 5: Confidence Saturation
    logger.info("PRE-PROCESSOR: Phase 5 — Confidence Saturation")
    confidence_saturation = check_confidence_saturation(
        parsed_draft, inputs["draft_memo"], inputs, config,
    )
    logger.info(f"  Saturation: {confidence_saturation['score']:.0%} "
                f"({'ACTIVE' if confidence_saturation['active'] else 'inactive'})")

    # Phase 6: DA History Load + Persistent Challenges
    logger.info("PRE-PROCESSOR: Phase 6 — DA History + Persistent Challenges")
    da_history_data = load_da_history_and_persistent(inputs.get("da_history"), config)
    logger.info(f"  First run: {da_history_data['is_first_run']}, "
                f"{len(da_history_data['persistent_challenges'])} persistent, "
                f"{len(da_history_data['forced_decision_challenges'])} forced decision")

    # Phase 7: Focus Rotation + Perspective Seed
    logger.info("PRE-PROCESSOR: Phase 7 — Focus Rotation + Perspective Seed")
    focus_selection = select_focus(today, inputs.get("da_history"), config)
    perspective_seed = select_perspective_seed(today, inputs.get("da_history"), config)
    logger.info(f"  Focus: {focus_selection['primary_focus']}, "
                f"Seed: {perspective_seed['seed_label']}")

    # Phase 8: Asymmetry + Output Assembly
    logger.info("PRE-PROCESSOR: Phase 8 — Asymmetry + Output Assembly")
    asymmetry = calculate_asymmetry(parsed_draft, confidence_saturation, config)
    logger.info(f"  Mode: {asymmetry['mode']}, "
                f"max_flags: {asymmetry['max_flags_to_llm']}, "
                f"min_challenges: {asymmetry['min_challenges']}")

    preprocessor_output = assemble_preprocessor_output(
        validation["manifest"], parsed_draft,
        omissions, consistency_flags, drift_flags,
        confidence_saturation, da_history_data,
        focus_selection, perspective_seed, asymmetry,
    )

    total_flags = preprocessor_output["flags"]["total_found"]
    sent_flags = preprocessor_output["flags"]["sent_to_llm"]
    logger.info(f"  Total flags: {total_flags}, sent to LLM: {sent_flags}")

    # ================================================================
    # LLM CALL
    # ================================================================

    logger.info("LLM: Building DA prompt")
    user_prompt = build_user_prompt(preprocessor_output, inputs)
    logger.info(f"  Prompt size: ~{len(user_prompt)} chars")

    logger.info("LLM: Calling Claude Sonnet (DA)")
    llm_result = call_da_llm(user_prompt, config)

    if not llm_result["success"]:
        logger.error(f"DA LLM failed: {llm_result.get('error')}")
        return {
            "success": False,
            "reason": "LLM failed after retries",
            "action": "CIO Draft wird als Final uebernommen",
        }

    raw_challenges = llm_result["challenges"]
    logger.info(f"LLM: {len(raw_challenges)} challenges parsed from output")

    # ================================================================
    # POST-PROCESSOR
    # ================================================================

    # Step 1: Challenge Validation
    logger.info("POST-PROCESSOR: Step 1 — Challenge Validation")
    valid, validated_challenges, val_errors = validate_challenges(
        raw_challenges, preprocessor_output, inputs, config,
    )

    if not valid and any(e.get("action") == "RETRY" for e in val_errors):
        logger.warning(f"DA validation failed: {val_errors}")
        if len(validated_challenges) == 0:
            return {
                "success": False,
                "reason": "No valid challenges after validation",
                "action": "CIO Draft wird als Final uebernommen",
            }
    logger.info(f"  {len(validated_challenges)} valid challenges "
                f"({len(val_errors)} validation notes)")

    # Step 2: Persistent Challenge Merger
    logger.info("POST-PROCESSOR: Step 2 — Persistent Challenge Merger")
    merged_challenges = merge_persistent_challenges(
        validated_challenges,
        da_history_data["persistent_challenges"],
        da_history_data["forced_decision_challenges"],
    )
    logger.info(f"  {len(merged_challenges)} challenges after merge "
                f"(+{len(merged_challenges) - len(validated_challenges)} from history)")

    # Step 3: DA History Update
    logger.info("POST-PROCESSOR: Step 3 — DA History Update")
    da_history_updated = update_da_history(
        inputs.get("da_history"), merged_challenges, preprocessor_output,
    )
    logger.info(f"  {len(da_history_updated['open_challenges'])} open challenges in history")

    # Step 4: Perspective Seed Rotation Check
    logger.info("POST-PROCESSOR: Step 4 — Seed Rotation Check")
    rotation = check_seed_rotation(inputs.get("da_history"), today, config)
    if rotation["rotation_needed"]:
        da_history_updated["perspective_seed_index"] = rotation["new_index"]
        da_history_updated["last_seed_rotation"] = rotation["new_rotation_date"]
        logger.info(f"  Seed rotated to index {rotation['new_index']}")
    else:
        logger.info("  No rotation needed")

    # Step 5: Output Formatting
    logger.info("POST-PROCESSOR: Step 5 — Output Formatting")
    da_output = format_da_output(
        merged_challenges, preprocessor_output,
        da_history_updated, inputs, llm_result,
    )

    elapsed = round(time.time() - start_time, 1)
    da_output["metadata"]["generation_time_seconds"] = elapsed

    # Summary
    meta = da_output["metadata"]
    logger.info("=" * 60)
    logger.info(f"DA COMPLETE in {elapsed}s — "
                f"{meta['total_challenges']} challenges "
                f"(SUBST:{meta['by_severity']['SUBSTANTIVE']} "
                f"MOD:{meta['by_severity']['MODERATE']} "
                f"MIN:{meta['by_severity']['MINOR']})")
    logger.info(f"  Focus: {da_output['primary_focus']}, "
                f"Seed: {da_output['perspective_seed']}")
    logger.info(f"  Persistent: {meta['persistent_renewed']}, "
                f"Forced: {meta['forced_decisions']}")
    logger.info(f"  Saturation: {confidence_saturation['score']:.0%} "
                f"({'ACTIVE' if confidence_saturation['active'] else 'inactive'})")
    logger.info("=" * 60)

    return {
        "success": True,
        "da_output": da_output,
        "da_history": da_history_updated,
    }
