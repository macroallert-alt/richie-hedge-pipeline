"""
step4_cio/dashboard_update.py
CIO → Dashboard Integration
Spec: V87 §8.4, CIO Addendum Dashboard Writer

Identical pattern to IC Pipeline's update_dashboard_json():
  - Read data/dashboard/latest.json
  - Replace briefing block with CIO output
  - Update header fields (briefing_type, conviction, risk_ampel)
  - Update pipeline_health.steps.step_4_cio_draft / step_6_cio_final
  - Update digest lines
  - Remove CIO_BRIEFING from known_unknowns
  - Write back

CIODetail.jsx expects:
  briefing.status = "AVAILABLE"
  briefing.source = "CIO_FINAL" | "CIO_DRAFT"
  briefing.full_text = "..."
  briefing.sections = {S1_delta: "...", S2_catalysts: "...", ...}
  briefing.section_word_counts = {S1_delta: 145, ...}
  briefing.da_markers = [{section, marker_type, challenge_summary, cio_response}, ...]
  briefing.da_resolution_summary = {total, accepted, noted, rejected, details}
  briefing.key_assumptions = [{assumption, basis, vulnerability}, ...]
  briefing.confidence_markers = [{section, claim, confidence, basis}, ...]
"""

import json
import logging
import os
from datetime import datetime

from step4_cio.postprocessor import extract_key_assumptions

logger = logging.getLogger("cio_dashboard")


def build_briefing_block(cio_output: dict) -> dict:
    """
    Map CIO output to dashboard.json briefing block.
    Format matches CIODetail.jsx expectations exactly.
    """
    is_draft_as_final = cio_output.get("is_draft_as_final", False)
    is_fallback = cio_output.get("is_fallback", False)

    if is_fallback:
        source = "FALLBACK"
    elif is_draft_as_final:
        source = "CIO_DRAFT_AS_FINAL"
    elif "cio_final" in cio_output.get("run_id", ""):
        source = "CIO_FINAL"
    else:
        source = "CIO_DRAFT"

    # Sections
    sections = cio_output.get("section_texts", {})
    section_word_counts = cio_output.get("section_word_counts", {})

    # DA markers for CIODetail.jsx
    da_resolution = cio_output.get("da_resolution", {})
    da_markers = []
    for detail in da_resolution.get("details", []):
        da_markers.append({
            "section": detail.get("section", ""),
            "marker_type": detail.get("marker_type", ""),
            "challenge_summary": detail.get("challenge_summary", ""),
            "cio_response": detail.get("cio_response", ""),
        })

    da_resolution_summary = {
        "total": da_resolution.get("total_challenges", 0),
        "accepted": da_resolution.get("accepted", 0),
        "noted": da_resolution.get("noted", 0),
        "rejected": da_resolution.get("rejected", 0),
        "details": [
            {
                "resolution": d.get("marker_type", ""),
                "summary": d.get("challenge_summary", ""),
            }
            for d in da_resolution.get("details", [])
        ],
    }

    # Key assumptions
    briefing_text = cio_output.get("briefing_text", "")
    key_assumptions = extract_key_assumptions(briefing_text)

    # Confidence markers
    confidence_markers = cio_output.get("confidence_markers", [])

    return {
        "status": "AVAILABLE",
        "source": source,
        "full_text": briefing_text,
        "sections": sections,
        "section_word_counts": section_word_counts,
        "da_markers": da_markers,
        "da_resolution_summary": da_resolution_summary,
        "key_assumptions": key_assumptions,
        "confidence_markers": confidence_markers,
    }


def update_dashboard_json(cio_output: dict, dashboard_json_path: str,
                         inputs_raw: dict | None = None) -> None:
    """
    Read data/dashboard/latest.json, replace briefing block,
    update header + pipeline_health + digest + known_unknowns,
    update risk block (from Risk Officer) and layers block (from Market Analyst),
    write back.
    """
    if not os.path.exists(dashboard_json_path):
        logger.warning(
            f"Dashboard JSON not found at {dashboard_json_path} — skipping"
        )
        return

    if inputs_raw is None:
        inputs_raw = {}

    try:
        with open(dashboard_json_path, "r") as f:
            dashboard = json.load(f)

        # 1. Replace briefing block
        dashboard["briefing"] = build_briefing_block(cio_output)

        # 2. Update header fields
        header = dashboard.get("header", {})
        header["briefing_type"] = cio_output.get("briefing_type", header.get("briefing_type"))
        header["system_conviction"] = cio_output.get("system_conviction", "N/A")
        header["risk_ampel"] = cio_output.get("risk_ampel", header.get("risk_ampel"))

        # fragility_state may be a dict from Market Analyst — extract string
        frag_raw = cio_output.get("fragility_state", header.get("fragility_state", "N/A"))
        if isinstance(frag_raw, dict):
            frag_raw = frag_raw.get("state", frag_raw.get("level", "N/A"))
        header["fragility_state"] = frag_raw
        header["is_draft_fallback"] = cio_output.get("is_fallback", False)

        # DA stats in header
        da_res = cio_output.get("da_resolution", {})
        header["da_challenges_total"] = da_res.get("total_challenges", 0)
        header["da_accepted"] = da_res.get("accepted", 0)
        header["da_noted"] = da_res.get("noted", 0)
        header["da_rejected"] = da_res.get("rejected", 0)

        # Fact check
        header["fact_check_corrections"] = cio_output.get("fact_check_corrections_count", 0)

        # Action items counts
        action_items = cio_output.get("action_items", [])
        header["action_items_act_count"] = sum(1 for a in action_items if a.get("type") == "ACT")
        header["action_items_review_count"] = sum(1 for a in action_items if a.get("type") == "REVIEW")
        header["action_items_watch_count"] = sum(1 for a in action_items if a.get("type") == "WATCH")

        dashboard["header"] = header

        # 3. Update digest lines
        digest = dashboard.get("digest", {})
        bt = cio_output.get("briefing_type", "WATCH")
        regime = cio_output.get("v16_regime", "UNKNOWN")
        conv = cio_output.get("system_conviction", "N/A")
        ampel = cio_output.get("risk_ampel", "GREEN")

        digest["line_1_type_and_delta"] = (
            f"{bt} — V16 Regime: {regime}. "
            f"Conviction: {conv}. Risk: {ampel}."
        )

        act_count = header.get("action_items_act_count", 0)
        review_count = header.get("action_items_review_count", 0)
        if act_count > 0:
            digest["line_2_actions"] = f"{act_count} ACT, {review_count} REVIEW Items."
        elif review_count > 0:
            digest["line_2_actions"] = f"{review_count} REVIEW Items. Kein sofortiger Handlungsbedarf."
        else:
            digest["line_2_actions"] = "Keine Action Items. System stabil."

        dq = cio_output.get("data_quality", "DEGRADED")
        digest["line_3_confidence"] = (
            f"Conviction: {conv}. Fragility: {frag_raw}. Data: {dq}."
        )
        dashboard["digest"] = digest

        # 4. Update action_items block
        dashboard["action_items"] = _build_action_items_block(action_items)

        # 5. Update pipeline_health
        now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        steps = dashboard.get("pipeline_health", {}).get("steps", {})

        run_id = cio_output.get("run_id", "")
        word_count = cio_output.get("metadata", {}).get("word_count", 0)

        if "cio_final" in run_id:
            steps["step_6_cio_final"] = {
                "status": "OK",
                "completed_at": now_utc,
                "summary": (
                    f"{bt} briefing, {word_count} words, "
                    f"conviction={conv}"
                ),
            }
            # Also mark step 4 as OK if not already
            if steps.get("step_4_cio_draft", {}).get("status") != "OK":
                steps["step_4_cio_draft"] = {
                    "status": "OK",
                    "completed_at": now_utc,
                    "summary": "Draft completed (promoted to final)",
                }
        else:
            steps["step_4_cio_draft"] = {
                "status": "OK",
                "completed_at": now_utc,
                "summary": f"{bt} draft, {word_count} words",
            }

        dashboard.setdefault("pipeline_health", {})["steps"] = steps

        # 6. Remove CIO_BRIEFING from known_unknowns
        kus = dashboard.get("known_unknowns", [])
        dashboard["known_unknowns"] = [
            ku for ku in kus if ku.get("gap") != "CIO_BRIEFING"
        ]

        # 7. Update risk block from Risk Officer input (CIO has it loaded)
        risk_input = inputs_raw.get("risk_alerts", {})
        if risk_input and risk_input.get("portfolio_status"):
            dashboard["risk"] = _build_risk_block(risk_input)
            dashboard["known_unknowns"] = [
                ku for ku in dashboard["known_unknowns"]
                if ku.get("gap") != "RISK"
            ]
            steps["step_3_risk_officer"] = {
                "status": "OK",
                "completed_at": risk_input.get("date", now_utc),
                "summary": (
                    f"{risk_input.get('portfolio_status', '?')} — "
                    f"{len(risk_input.get('alerts', []))} alerts"
                ),
            }
            dashboard.setdefault("pipeline_health", {})["steps"] = steps
            logger.info(f"Risk block updated: {risk_input.get('portfolio_status')}")

        # 8. Update layers block from Market Analyst input (CIO has it loaded)
        la_input = inputs_raw.get("layer_analysis", {})
        logger.info(f"Layers input check: keys={list(la_input.keys())[:10] if la_input else 'EMPTY'}")

        # Market Analyst JSON structure: top-level 'layers' dict with per-layer objects
        # Each layer has 'score', 'signal', 'freshness' etc.
        layer_scores = la_input.get("layer_scores", {})
        if not layer_scores:
            # Try extracting from 'layers' dict (MA production format)
            layers_dict = la_input.get("layers", {})
            if isinstance(layers_dict, dict):
                for layer_key, layer_data in layers_dict.items():
                    if isinstance(layer_data, dict) and "score" in layer_data:
                        layer_scores[layer_key] = layer_data["score"]

        if layer_scores:
            la_with_scores = {**la_input, "layer_scores": layer_scores}
            dashboard["layers"] = _build_layers_block(la_with_scores)
            dashboard["known_unknowns"] = [
                ku for ku in dashboard["known_unknowns"]
                if ku.get("gap") != "LAYER_SCORES"
            ]
            # Extract strings for summary (may be dicts)
            ma_regime = la_input.get('system_regime', '?')
            if isinstance(ma_regime, dict):
                ma_regime = ma_regime.get('regime', '?')
            ma_frag = la_input.get('fragility_state', '?')
            if isinstance(ma_frag, dict):
                ma_frag = ma_frag.get('state', '?')

            steps["step_1_market_analyst"] = {
                "status": "OK",
                "completed_at": la_input.get("date", now_utc),
                "summary": f"Regime: {ma_regime}, Fragility: {ma_frag}",
            }
            dashboard.setdefault("pipeline_health", {})["steps"] = steps
            logger.info(f"Layers block updated: {la_input.get('system_regime')}, {len(layer_scores)} layers")
        else:
            logger.warning("Layers block NOT updated — no layer_scores found in Market Analyst data")

        # 9. Update signals block from Signal Generator input
        signals_input = inputs_raw.get("signals", {})
        if signals_input and signals_input.get("execution_path"):
            dashboard["signals"] = _build_signals_block(signals_input)
            dashboard["known_unknowns"] = [
                ku for ku in dashboard["known_unknowns"]
                if ku.get("gap") != "SIGNALS"
            ]
            steps["step_2_signal_generator"] = {
                "status": "OK",
                "completed_at": signals_input.get("run_timestamp", now_utc),
                "summary": (
                    f"{signals_input.get('execution_path', '?')} — "
                    f"Router: {signals_input.get('router', {}).get('current_state', '?')}, "
                    f"max prox: {signals_input.get('router', {}).get('max_proximity', 0):.0%}"
                ),
            }
            dashboard.setdefault("pipeline_health", {})["steps"] = steps
            logger.info(f"Signals block updated: {signals_input.get('execution_path')}")
        else:
            logger.info("Signals block NOT updated — no Signal Generator data")

        # Write back
        with open(dashboard_json_path, "w") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)

        logger.info(f"Dashboard JSON updated: {dashboard_json_path}")

    except Exception as e:
        logger.error(f"Dashboard JSON update failed: {e}")


def _build_action_items_block(action_items: list) -> dict:
    """Build the action_items block for dashboard.json."""
    act_items = [a for a in action_items if a.get("type") == "ACT"]
    review_items = [a for a in action_items if a.get("type") == "REVIEW"]
    watch_items = [a for a in action_items if a.get("type") == "WATCH"]

    prominent = []
    for item in (act_items + review_items)[:5]:
        prominent.append({
            "id": item.get("id", ""),
            "type": item.get("type", ""),
            "urgency": item.get("urgency", ""),
            "description": item.get("description", ""),
            "source_alerts": item.get("source_alerts", []),
            "source_patterns": item.get("source_patterns", []),
        })

    return {
        "summary": {
            "act_count": len(act_items),
            "review_count": len(review_items),
            "watch_count": len(watch_items),
            "total": len(action_items),
            "escalated_today": sum(1 for a in action_items if a.get("escalated")),
            "new_today": sum(1 for a in action_items if a.get("days_open", 0) <= 1),
            "resolved_today": 0,
        },
        "prominent": prominent,
        "aggregated": {
            "count": len(action_items),
            "items": action_items,
        },
        "ongoing_conditions": [],
    }


def _build_risk_block(risk_officer: dict) -> dict:
    """
    Map Risk Officer JSON to dashboard.json risk block.
    Matches RiskDetail.jsx expectations.
    """
    alerts = risk_officer.get("alerts", [])

    # Map alerts to dashboard format
    dashboard_alerts = []
    for alert in alerts:
        dashboard_alerts.append({
            "check_id": alert.get("check_id", ""),
            "severity": alert.get("severity", "MONITOR"),
            "trend": alert.get("trend", "STABLE"),
            "days_active": alert.get("days_active", 0),
            "details": alert.get("details", {}),
            "recommendation": alert.get("recommendation", ""),
        })

    # Emergency triggers
    emg = risk_officer.get("emergency_triggers", {})
    emergency_triggers = {
        "max_drawdown_breach": _is_trigger_active(emg, "EMG_PORTFOLIO_DRAWDOWN"),
        "correlation_crisis": _is_trigger_active(emg, "EMG_CORRELATION_CRISIS"),
        "liquidity_crisis": _is_trigger_active(emg, "EMG_LIQUIDITY_CRISIS"),
        "regime_forced": _is_trigger_active(emg, "EMG_REGIME_FORCED"),
    }

    # Ongoing conditions
    ongoing = risk_officer.get("ongoing_conditions", [])

    return {
        "status": "AVAILABLE",
        "portfolio_status": risk_officer.get("portfolio_status", "GREEN"),
        "alerts": dashboard_alerts,
        "emergency_triggers": emergency_triggers,
        "ongoing_conditions_count": len(ongoing),
        "ongoing_conditions": ongoing,
        "sensitivity": risk_officer.get("sensitivity", {}),
        "risk_summary": risk_officer.get("risk_summary", ""),
        "fast_path": risk_officer.get("fast_path", False),
    }


def _is_trigger_active(emg_dict: dict, key: str) -> bool:
    """Check if an emergency trigger is active."""
    trigger = emg_dict.get(key, {})
    if isinstance(trigger, dict):
        return trigger.get("status") == "ACTIVE"
    return bool(trigger)


def _build_layers_block(layer_analysis: dict) -> dict:
    """
    Map Market Analyst JSON to dashboard.json layers block.
    Matches LayersDetail.jsx expectations.
    Writes full layer objects {score, direction, conviction, limiting_factor}
    when available. Frontend handles both flat numbers and objects.
    """
    layer_scores = layer_analysis.get("layer_scores", {})
    layers_dict = layer_analysis.get("layers", {})

    # Build layer_scores: full objects if available, flat numbers as fallback
    clean_scores = {}
    for key, val in layer_scores.items():
        # Check if full layer object exists in 'layers' dict
        full_obj = layers_dict.get(key, {}) if isinstance(layers_dict, dict) else {}
        if isinstance(full_obj, dict) and "score" in full_obj:
            # Write full object with score, direction, conviction, limiting_factor
            clean_scores[key] = {
                "score": _safe_number(full_obj.get("score", 0)),
                "direction": full_obj.get("direction", None),
                "conviction": full_obj.get("conviction", None),
                "limiting_factor": full_obj.get("limiting_factor", None),
            }
        elif isinstance(val, (int, float)):
            clean_scores[key] = val
        elif isinstance(val, dict):
            # layer_scores value is itself a dict
            clean_scores[key] = {
                "score": _safe_number(val.get("score", val.get("score_raw", 0))),
                "direction": val.get("direction", None),
                "conviction": val.get("conviction", None),
                "limiting_factor": val.get("limiting_factor", None),
            }
        else:
            try:
                clean_scores[key] = float(val)
            except (TypeError, ValueError):
                clean_scores[key] = 0.0

    fragility_data = layer_analysis.get("fragility_data", {})

    # system_regime may be a string or a dict
    system_regime = layer_analysis.get("system_regime", "UNKNOWN")
    if isinstance(system_regime, dict):
        system_regime = system_regime.get("regime", system_regime.get("name", "UNKNOWN"))

    # fragility_state may also be dict
    fragility_state = layer_analysis.get("fragility_state", "N/A")
    if isinstance(fragility_state, dict):
        fragility_state = fragility_state.get("state", fragility_state.get("level", "N/A"))

    # Calculate regime stability percentage (if available)
    conv_dynamics = layer_analysis.get("conviction_dynamics", {})
    stability_days = conv_dynamics.get("regime_stability_days", None)
    regime_stability_pct = None
    if stability_days is not None:
        regime_stability_pct = min(100, round(stability_days / 30 * 100))

    return {
        "status": "AVAILABLE",
        "system_regime": system_regime,
        "regime_stability_pct": regime_stability_pct,
        "fragility_state": fragility_state,
        "fragility_data": fragility_data,
        "layer_scores": clean_scores,
    }


def _build_signals_block(signals_input: dict) -> dict:
    """
    Map Signal Generator JSON to dashboard.json signals block.
    Matches SignalsDetail.jsx expectations.
    """
    router = signals_input.get("router", {})
    proximity = router.get("proximity", {})
    projections = signals_input.get("projections", {})
    concentration = projections.get("concentration_check", {}).get("baseline", {})
    recommendations = signals_input.get("recommendations", {})
    trade_list = signals_input.get("trade_list", [])

    # Router status per trigger
    router_status = {}
    for trigger_id in ("em_broad", "china_stimulus", "commodity_super"):
        prox_data = proximity.get(trigger_id, {})
        composite = prox_data.get("composite", 0.0)
        trend = prox_data.get("trend", "STABLE")

        # Determine state label
        current_state = router.get("current_state", "US_DOMESTIC")
        if current_state.lower() == trigger_id:
            state = "ACTIVE"
        elif composite >= 0.7:
            state = "APPROACHING"
        elif composite >= 0.3:
            state = "MONITORING"
        else:
            state = "INACTIVE"

        router_status[trigger_id.upper()] = {
            "proximity": round(composite, 4),
            "trend": trend,
            "state": state,
        }

    # PermOpt status
    perm_opt = signals_input.get("perm_opt", {})
    permopt_status = {
        "budget_pct": perm_opt.get("total_pct", 0),
        "active": perm_opt.get("status") != "UNAVAILABLE",
        "positions_count": 0,
    }

    return {
        "status": "AVAILABLE",
        "execution_path": signals_input.get("execution_path", "UNKNOWN"),
        "trade_count": len(trade_list),
        "router_state": router.get("current_state", "US_DOMESTIC"),
        "router_days_in_state": router.get("days_in_state", 0),
        "router_status": router_status,
        "max_proximity": router.get("max_proximity", 0),
        "max_proximity_trigger": router.get("max_proximity_trigger"),
        "effective_concentration": concentration.get("effective_tech_pct", 0),
        "concentration_warning": concentration.get("warning", False),
        "permopt_status": permopt_status,
        "has_actionable_recommendations": recommendations.get("has_actionable_recommendations", False),
        "summary_for_cio": recommendations.get("summary_for_cio", ""),
        "entry_evaluation": router.get("entry_evaluation", {}),
        "exit_check": router.get("exit_check"),
        "emergency": router.get("emergency"),
        "crisis_override": router.get("crisis_override", False),
    }


def _safe_number(val):
    """Convert value to number, return 0 on failure."""
    if isinstance(val, (int, float)):
        return val
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0

