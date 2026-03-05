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
        header["fragility_state"] = cio_output.get("fragility_state", header.get("fragility_state", "N/A"))
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

        frag = cio_output.get("fragility_state", "N/A")
        dq = cio_output.get("data_quality", "DEGRADED")
        digest["line_3_confidence"] = (
            f"Conviction: {conv}. Fragility: {frag}. Data: {dq}."
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
            steps["step_1_market_analyst"] = {
                "status": "OK",
                "completed_at": la_input.get("date", now_utc),
                "summary": (
                    f"Regime: {la_input.get('system_regime', '?')}, "
                    f"Fragility: {la_input.get('fragility_state', '?')}"
                ),
            }
            dashboard.setdefault("pipeline_health", {})["steps"] = steps
            logger.info(f"Layers block updated: {la_input.get('system_regime')}, {len(layer_scores)} layers")
        else:
            logger.warning("Layers block NOT updated — no layer_scores found in Market Analyst data")

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
    """
    layer_scores = layer_analysis.get("layer_scores", {})
    fragility_data = layer_analysis.get("fragility_data", {})

    # Calculate regime stability percentage (if available)
    conv_dynamics = layer_analysis.get("conviction_dynamics", {})
    stability_days = conv_dynamics.get("regime_stability_days", None)
    regime_stability_pct = None
    if stability_days is not None:
        # Rough heuristic: 30+ days = 100%, scale linearly
        regime_stability_pct = min(100, round(stability_days / 30 * 100))

    return {
        "status": "AVAILABLE",
        "system_regime": layer_analysis.get("system_regime", "UNKNOWN"),
        "regime_stability_pct": regime_stability_pct,
        "fragility_state": layer_analysis.get("fragility_state", "N/A"),
        "fragility_data": fragility_data,
        "layer_scores": layer_scores,
    }

