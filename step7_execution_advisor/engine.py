"""
step7_execution_advisor/engine.py
Execution Advisor Engine — 8-Phase Orchestrator.

Phases:
  0. SETUP       — Extract inputs, load config
  1. EVENT WINDOW — Parse calendar, compute 48h/14d/convergence
  2. DW DATA     — Already loaded by main.py
  3. V16 CONTEXT — Extract weights, regime, trades from Step 2 output
  4. EXEC SCORE  — 6 dimensions + veto (deterministic)
  5. CONFIRM/CONFLICT — Pro/Contra aggregation
  6. LLM TEXT    — Generate briefing via Sonnet (with fallback)
  6b. ROTATION   — Build rotation block + append weights_history
  7. ASSEMBLE    — Build step7_execution_advisor.json

Source: Trading Desk Spec Teil 5 §22, Rotation Circle Spec Teil 4 §18.4
"""

import logging
import os
import time
from datetime import date, datetime

from .scoring import calculate_execution_score
from .confirming_conflicting import build_confirming_conflicting
from .llm import generate_execution_briefing
from .event_reader import compute_event_window

logger = logging.getLogger("execution_advisor.engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_execution_advisor(inputs: dict, config: dict,
                          today: date = None) -> dict:
    """
    Main Execution Advisor orchestrator.

    Args:
        inputs: {
            "signal_generator": dict or None,  # step2_signal_generator.json
            "risk_officer": dict or None,       # step3_risk_officer.json
            "cio_final": dict or None,          # step6_cio_final.json
            "dw_data": dict,                    # DW RAW_MARKET fields
            "dw_degraded": bool,                # True if DW unavailable
            "events": list[dict],               # EVENT_CALENDAR.yaml events
            "dashboard": dict or None,          # dashboard.json (backup)
        }
        config: merged config from all JSON files
        today: override date for testing

    Returns:
        Full output dict (written to step7_execution_advisor.json)
    """
    start_time = time.time()
    today = today or date.today()

    logger.info("=" * 60)
    logger.info("EXECUTION ADVISOR (Step 7)")
    logger.info(f"Date: {today}")
    logger.info("=" * 60)

    degraded_reasons = []

    # ============================================================
    # PHASE 0: SETUP
    # ============================================================
    logger.info("")
    logger.info("PHASE 0: SETUP")

    signal_gen = inputs.get("signal_generator") or {}
    risk_officer = inputs.get("risk_officer") or {}
    cio_final = inputs.get("cio_final") or {}
    dw_data = inputs.get("dw_data") or {}
    dw_degraded = inputs.get("dw_degraded", False)
    events = inputs.get("events") or []
    dashboard = inputs.get("dashboard") or {}

    if dw_degraded:
        degraded_reasons.append(
            "DW Sheet nicht erreichbar — Dimensionen 2-6 auf 0"
        )

    if not events:
        degraded_reasons.append(
            "EVENT_CALENDAR.yaml nicht verfuegbar — Event Risk auf 0"
        )

    if not risk_officer:
        degraded_reasons.append("Risk Officer nicht verfuegbar")

    if not cio_final:
        degraded_reasons.append("CIO Final nicht verfuegbar")

    logger.info(f"  Signal Generator: {'LOADED' if signal_gen else 'MISSING'}")
    logger.info(f"  Risk Officer: {'LOADED' if risk_officer else 'MISSING'}")
    logger.info(f"  CIO Final: {'LOADED' if cio_final else 'MISSING'}")
    logger.info(f"  DW Data: {len(dw_data)} fields {'(DEGRADED)' if dw_degraded else ''}")
    logger.info(f"  Events: {len(events)}")

    # ============================================================
    # PHASE 1: EVENT WINDOW
    # ============================================================
    logger.info("")
    logger.info("PHASE 1: EVENT WINDOW")

    event_config = config.get("event_window", {})
    event_window = compute_event_window(events, today, event_config)

    logger.info(f"  48h: {len(event_window['next_48h'])} events")
    logger.info(f"  14d: {event_window['event_density_14d']} events")
    logger.info(f"  Convergence weeks: {len(event_window['convergence_weeks'])}")

    # ============================================================
    # PHASE 2: DW DATA (already loaded)
    # ============================================================
    logger.info("")
    logger.info("PHASE 2: DW DATA")
    logger.info(f"  Fields available: {len(dw_data)}")

    # ============================================================
    # PHASE 3: V16 CONTEXT
    # ============================================================
    logger.info("")
    logger.info("PHASE 3: V16 CONTEXT")

    v16_data = _extract_v16_context(signal_gen, dashboard)
    v16_weights = v16_data.get("current_weights", {})
    v16_regime = v16_data.get("regime", "UNKNOWN")

    logger.info(f"  Regime: {v16_regime}")
    logger.info(f"  Positions: {len(v16_weights)}")
    logger.info(f"  Top position: {_top_position(v16_weights)}")

    # ============================================================
    # PHASE 4: EXECUTION SCORE
    # ============================================================
    logger.info("")
    logger.info("PHASE 4: EXECUTION SCORE")

    scoring_result = calculate_execution_score(
        events=events,
        v16_weights=v16_weights,
        v16_regime=v16_regime,
        dw_data=dw_data,
        today=today,
    )

    logger.info(f"  Raw Score: {scoring_result['raw_score']}/18")
    logger.info(f"  Adjusted: {scoring_result['total_score']}/18")
    logger.info(f"  Level: {scoring_result['execution_level']}")
    if scoring_result["veto_applied"]:
        logger.info(f"  VETO: {scoring_result['veto_reason']}")

    for dim_name, dim_data in scoring_result["dimensions"].items():
        logger.info(f"    {dim_name}: {dim_data['score']}/3 — {dim_data.get('label', '')}")

    # ============================================================
    # PHASE 5: CONFIRMING / CONFLICTING
    # ============================================================
    logger.info("")
    logger.info("PHASE 5: CONFIRMING / CONFLICTING")

    router_output = signal_gen.get("router", {})
    v16_trades = signal_gen.get("v16_trades", {})

    cc_result = build_confirming_conflicting(
        scoring_result=scoring_result,
        v16_weights=v16_weights,
        v16_regime=v16_regime,
        v16_trades=v16_trades,
        risk_officer=risk_officer,
        cio_final=cio_final,
        router_output=router_output,
        dw_data=dw_data,
    )

    logger.info(f"  Confirming: {cc_result['confirming_count']}")
    logger.info(f"  Conflicting: {cc_result['conflicting_count']}")
    logger.info(f"  Net: {cc_result['net_assessment']}")

    # ============================================================
    # PHASE 6: LLM TEXT
    # ============================================================
    logger.info("")
    logger.info("PHASE 6: LLM TEXT")

    llm_result = generate_execution_briefing(
        scoring_result=scoring_result,
        cc_result=cc_result,
        v16_data=v16_data,
        risk_officer=risk_officer,
        cio_final=cio_final,
        router_output=router_output,
        event_window=event_window,
        today=today,
        config=config,
        dw_data=dw_data,
    )

    if llm_result.get("llm_fallback"):
        degraded_reasons.append("LLM Fallback verwendet")

    logger.info(f"  LLM used: {llm_result.get('llm_used', False)}")
    logger.info(f"  Fallback: {llm_result.get('llm_fallback', False)}")
    logger.info(f"  Text length: {llm_result.get('llm_raw_length', 0)} chars")

    # ============================================================
    # PHASE 6b: ROTATION BLOCK
    # ============================================================
    logger.info("")
    logger.info("PHASE 6b: ROTATION BLOCK")

    from .rotation_builder import build_rotation_block

    WEIGHTS_HISTORY_PATH = os.path.join(
        os.path.dirname(BASE_DIR), "data", "weights_history", "weights_history.json"
    )
    CLUSTER_CONFIG_PATH = os.path.join(
        os.path.dirname(BASE_DIR), "config", "cluster_config.json"
    )

    # Build latest_data dict for rotation_builder
    latest_data_for_rotation = {
        "v16": {
            "regime": v16_regime,
            "macro_state_num": v16_data.get("macro_state_num", 0),
            "macro_state_name": v16_data.get("macro_state_name", "UNKNOWN"),
            "target_weights": v16_weights,
            "growth_signal": v16_data.get("growth_signal", "UNKNOWN"),
            "liq_direction": v16_data.get("liq_direction", "UNKNOWN"),
            "stress_score": v16_data.get("stress_score", 0),
            "dd_protect_status": v16_data.get("dd_protect_status", "INACTIVE"),
            "current_drawdown": v16_data.get("current_drawdown", 0),
            "dd_protect_threshold": v16_data.get("dd_protect_threshold", -15),
        },
        "signals": {
            "router_state": router_output.get("current_state", "UNKNOWN"),
            "router_days_in_state": router_output.get("days_in_state", 0),
        },
        "execution": {
            "execution_level": scoring_result.get("execution_level", "UNKNOWN"),
        },
    }

    try:
        rotation_block = build_rotation_block(
            latest_data=latest_data_for_rotation,
            weights_history_path=WEIGHTS_HISTORY_PATH,
            cluster_config_path=CLUSTER_CONFIG_PATH,
            today=today,
        )
        logger.info(f"  Status: {rotation_block.get('status')}")
        logger.info(f"  Mode: {rotation_block.get('mode')}")
        logger.info(f"  Material Shifts: {rotation_block.get('material_shifts_count')}")
    except Exception as e:
        logger.error(f"  Rotation block build FAILED: {e}")
        rotation_block = None
        degraded_reasons.append(f"Rotation Block fehlgeschlagen: {e}")

    # ============================================================
    # PHASE 7: ASSEMBLE OUTPUT
    # ============================================================
    logger.info("")
    logger.info("PHASE 7: ASSEMBLE OUTPUT")

    engine_time = round(time.time() - start_time, 2)

    # Build recommendation block
    recommendation = {
        "action": _determine_action(scoring_result["execution_level"]),
        "reasoning": llm_result.get("recommendation_short", ""),
        "specific_actions": llm_result.get("specific_actions", []),
        "would_change_my_mind": llm_result.get("would_change_my_mind", {
            "execute_if": [],
            "hold_if": [],
        }),
    }

    # V16 context for output
    top5 = sorted(
        v16_weights.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    weight_deltas = v16_data.get("weight_deltas", {})
    material_count = sum(1 for d in weight_deltas.values() if abs(d) > 0.01)

    output = {
        "date": today.isoformat(),
        "run_timestamp": datetime.utcnow().isoformat() + "Z",

        "execution_assessment": scoring_result,

        "confirming_conflicting": cc_result,

        "recommendation": recommendation,

        "event_window": event_window,

        "briefing_text": llm_result.get("briefing_text", ""),

        "v16_context": {
            "regime": v16_regime,
            "top5": [
                {"asset": a, "weight": w} for a, w in top5
            ],
            "material_rebalance_trades": material_count,
            "total_weight": sum(v16_weights.values()),
        },

        "pipeline_context": {
            "risk_ampel": risk_officer.get("risk_ampel", "UNKNOWN"),
            "cio_conviction": cio_final.get("conviction", "UNKNOWN"),
            "cio_briefing_type": cio_final.get("briefing_type", "UNKNOWN"),
            "fragility_state": risk_officer.get("fragility_state", "UNKNOWN"),
            "router_state": router_output.get("current_state", "UNKNOWN"),
            "router_max_proximity": router_output.get("max_proximity", 0),
        },

        "rotation": rotation_block,

        "meta": {
            "execution_path": "DEGRADED" if degraded_reasons else "FULL",
            "degraded": bool(degraded_reasons),
            "degraded_reasons": degraded_reasons,
            "llm_model": llm_result.get("llm_model", ""),
            "llm_used": llm_result.get("llm_used", False),
            "llm_fallback": llm_result.get("llm_fallback", False),
            "engine_time_seconds": engine_time,
        },
    }

    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = round(time.time() - start_time, 1)
    output["meta"]["total_time_seconds"] = total_time

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"EXECUTION ADVISOR COMPLETE in {total_time}s")
    logger.info(f"  Level: {scoring_result['execution_level']} "
                f"(Score: {scoring_result['total_score']}/18)")
    logger.info(f"  C/C: {cc_result['confirming_count']} vs "
                f"{cc_result['conflicting_count']} — {cc_result['net_assessment']}")
    logger.info(f"  Action: {recommendation['action']}")
    if rotation_block:
        logger.info(f"  Rotation: {rotation_block['status']} ({rotation_block['mode']})")
    if degraded_reasons:
        logger.info(f"  DEGRADED: {', '.join(degraded_reasons)}")
    logger.info("=" * 60)

    return output


# ============================================================
# HELPERS
# ============================================================

def _extract_v16_context(signal_gen: dict, dashboard: dict) -> dict:
    """
    Extract V16 context from Signal Generator output or dashboard fallback.

    Returns dict with: regime, current_weights, weight_deltas, macro_state_name,
                       latest_prices (V96: for LLM market data block)
    Weights are normalized to {asset: float} format.
    """
    # Try Signal Generator first
    v16_trades = signal_gen.get("v16_trades", {})
    if v16_trades and v16_trades.get("weights"):
        return {
            "regime": v16_trades.get("v16_regime", "UNKNOWN"),
            "current_weights": _normalize_weights(v16_trades.get("weights", {})),
            "weight_deltas": _normalize_weights(v16_trades.get("weight_deltas", {})),
            "macro_state_name": v16_trades.get("state_label", "UNKNOWN"),
            "macro_state_num": v16_trades.get("macro_state_num", 0),
            "growth_signal": v16_trades.get("growth_signal", 0),
            "liq_direction": v16_trades.get("liq_direction", 0),
            "stress_score": v16_trades.get("stress_score", 0),
            "dd_protect_status": v16_trades.get("dd_protect_status", "INACTIVE"),
            "current_drawdown": v16_trades.get("current_drawdown", 0),
            "dd_protect_threshold": v16_trades.get("dd_protect_threshold", -15),
            "latest_prices": v16_trades.get("latest_prices", {}),
        }

    # Fallback: dashboard.json v16 block
    v16_block = dashboard.get("v16", {})
    if v16_block:
        return {
            "regime": v16_block.get("regime", "UNKNOWN"),
            "current_weights": _normalize_weights(v16_block.get("current_weights", {})),
            "weight_deltas": _normalize_weights(v16_block.get("weight_deltas", {})),
            "macro_state_name": v16_block.get("macro_state_name", "UNKNOWN"),
            "macro_state_num": v16_block.get("macro_state_num", 0),
            "growth_signal": v16_block.get("growth_signal", 0),
            "liq_direction": v16_block.get("liq_direction", 0),
            "stress_score": v16_block.get("stress_score", 0),
            "dd_protect_status": v16_block.get("dd_protect_status", "INACTIVE"),
            "current_drawdown": v16_block.get("current_drawdown", 0),
            "dd_protect_threshold": v16_block.get("dd_protect_threshold", -15),
            "latest_prices": v16_block.get("latest_prices", {}),
        }

    return {
        "regime": "UNKNOWN",
        "current_weights": {},
        "weight_deltas": {},
        "macro_state_name": "UNKNOWN",
        "macro_state_num": 0,
        "growth_signal": 0,
        "liq_direction": 0,
        "stress_score": 0,
        "dd_protect_status": "INACTIVE",
        "current_drawdown": 0,
        "dd_protect_threshold": -15,
        "latest_prices": {},
    }


def _normalize_weights(weights: dict) -> dict:
    """
    Normalize weights to {asset: float} format.
    Handles both direct floats and nested dicts like {"weight": 0.277, ...}.
    """
    normalized = {}
    for asset, value in weights.items():
        if isinstance(value, (int, float)):
            normalized[asset] = float(value)
        elif isinstance(value, dict):
            # Try common keys: weight, target_weight, value
            for key in ("weight", "target_weight", "value"):
                if key in value:
                    try:
                        normalized[asset] = float(value[key])
                    except (ValueError, TypeError):
                        pass
                    break
        elif isinstance(value, str):
            try:
                normalized[asset] = float(value)
            except ValueError:
                pass
    return normalized


def _top_position(weights: dict) -> str:
    """Get top position string for logging."""
    if not weights:
        return "none"
    top = max(weights.items(), key=lambda x: abs(x[1]))
    return f"{top[0]} {top[1]:.1%}"


def _determine_action(level: str) -> str:
    """Map execution level to action string."""
    return {
        "EXECUTE": "EXECUTE_NORMAL",
        "CAUTION": "EXECUTE_WITH_LIMITS",
        "WAIT": "WAIT_FOR_CLARITY",
        "HOLD": "DO_NOT_REBALANCE",
    }.get(level, "UNKNOWN")
