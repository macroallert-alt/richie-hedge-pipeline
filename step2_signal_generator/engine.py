"""
step2_signal_generator/engine.py
Signal Generator Engine — Main Orchestrator

Two paths:
  FAST_PATH: Router inactive, low proximity, Fragility HEALTHY (~99% of days)
  FULL_PATH: Router active or approaching, or Fragility >= ELEVATED

All deterministic. No LLM. <10 seconds runtime.
Source: Signal Generator Spec Teil 4 §20
"""

import logging
import time
from datetime import date, datetime

from .router_state import default_router_state, update_router_state_history
from .router_engine import run_router_engine, build_fast_path_router_output
from .router_proximity import (
    quick_proximity_check,
    compute_proximity_trends,
    inject_trends_into_proximity,
)
from .compilation import compile_v16_trades, compile_trade_list
from .projection import (
    calculate_baseline_projection,
    calculate_effective_concentration,
    build_projections_output,
)
from .recommendations import build_recommendations
from .outcome_tracker import manage_outcome_tracker, create_outcome_entry

logger = logging.getLogger("signal_generator.engine")


def run_signal_generator(inputs: dict, config: dict, today: date = None) -> dict:
    """
    Main Signal Generator orchestrator.
    Spec Teil 4 §20

    Args:
        inputs: {
            "router_raw_data": dict,      # From step0r_router_data.json
            "market_analyst": dict,        # From step1_market_analyst.json
            "v16_data": dict,              # From dashboard.json v16 block
            "own_history": dict or None,   # From yesterday's step2_signal_generator.json
        }
        config: merged config from all JSON files
        today: override date for testing

    Returns:
        signal_output dict (written to step2_signal_generator.json)
    """
    start_time = time.time()
    today = today or date.today()

    logger.info("=" * 60)
    logger.info("SIGNAL GENERATOR (Step 2)")
    logger.info(f"Date: {today}")
    logger.info("=" * 60)

    # ============================================================
    # PHASE 0: SETUP
    # ============================================================
    router_raw_data = inputs.get("router_raw_data") or {}
    market_analyst = inputs.get("market_analyst") or {}
    v16_data = inputs.get("v16_data") or {}
    own_history = inputs.get("own_history") or {}

    # Extract key fields
    fragility_state = _extract_fragility_state(market_analyst)
    v16_regime = _extract_v16_regime(v16_data)
    router_state = own_history.get("router_state") or default_router_state()
    yesterday_proximity = router_state.get("trigger_proximity_yesterday")

    logger.info(f"  V16 Regime: {v16_regime}")
    logger.info(f"  Fragility: {fragility_state}")
    logger.info(f"  Router State: {router_state.get('current_state', 'US_DOMESTIC')}")

    # ============================================================
    # PHASE 1: V16 TRADE COMPILATION
    # ============================================================
    logger.info("")
    logger.info("PHASE 1: V16 Trade Compilation")
    v16_trades = compile_v16_trades(v16_data)
    logger.info(f"  {len(v16_trades.get('weights', {}))} assets, "
                f"regime={v16_trades.get('v16_regime')}")

    # ============================================================
    # PHASE 2: FAST PATH CHECK
    # ============================================================
    logger.info("")
    logger.info("PHASE 2: Fast Path Check")
    quick_prox = quick_proximity_check(router_raw_data)

    use_fast_path = (
        router_state.get("current_state", "US_DOMESTIC") == "US_DOMESTIC"
        and quick_prox < 0.3
        and fragility_state == "HEALTHY"
    )

    execution_path = "FAST_PATH" if use_fast_path else "FULL_PATH"
    logger.info(f"  Quick Proximity: {quick_prox:.4f}")
    logger.info(f"  Execution Path: {execution_path}")

    # ============================================================
    # PHASE 3: ROUTER ENGINE
    # ============================================================
    logger.info("")
    logger.info(f"PHASE 3: Router Engine ({execution_path})")

    if execution_path == "FAST_PATH":
        router_output = build_fast_path_router_output(
            router_state, quick_prox, fragility_state, today, config,
        )
    else:
        router_output = run_router_engine(
            router_raw_data, v16_regime, fragility_state,
            router_state, None, today, config,  # g7_input=None in V1
        )

    logger.info(f"  State: {router_output.get('current_state')}")
    logger.info(f"  Max Proximity: {router_output.get('max_proximity', 0):.4f} "
                f"({router_output.get('max_proximity_trigger', 'none')})")

    if router_output.get("emergency"):
        logger.warning(f"  EMERGENCY: {router_output['emergency'].get('reason')}")
    if router_output.get("crisis_override"):
        logger.warning(f"  CRISIS OVERRIDE active")

    entry_eval = router_output.get("entry_evaluation", {})
    if entry_eval.get("is_evaluation_day"):
        rec = entry_eval.get("recommendation")
        if rec:
            logger.info(f"  Entry Evaluation: {rec.get('action', 'none')}")

    exit_check = router_output.get("exit_check")
    if exit_check and exit_check.get("exit_triggered"):
        logger.info(f"  Exit Signal: {exit_check.get('exit_id')}")

    # ============================================================
    # PHASE 4: PROXIMITY TRENDS (Dashboard Writer Addendum)
    # ============================================================
    logger.info("")
    logger.info("PHASE 4: Proximity Trends")
    trends = compute_proximity_trends(
        router_output.get("proximity", {}),
        yesterday_proximity,
        config,
    )
    router_output["proximity"] = inject_trends_into_proximity(
        router_output.get("proximity", {}), trends,
    )
    for tid, tdata in trends.items():
        logger.info(f"  {tid}: {tdata['trend']} (delta={tdata['delta']:+.4f})")

    # ============================================================
    # PHASE 5: PORTFOLIO PROJECTION (Baseline only in V1)
    # ============================================================
    logger.info("")
    logger.info("PHASE 5: Portfolio Projection")
    baseline = calculate_baseline_projection(v16_trades)
    concentration = calculate_effective_concentration(baseline, config)
    projections = build_projections_output(baseline, concentration)

    logger.info(f"  Positions: {len(baseline.get('positions', {}))}")
    logger.info(f"  Total Weight: {baseline.get('total_weight', 0):.4f}")
    logger.info(f"  Effective Tech: {concentration.get('effective_tech_pct', 0):.1%}")
    if concentration.get("warning"):
        logger.warning(f"  {concentration.get('warning_message')}")

    # ============================================================
    # PHASE 6: TRADE COMPILATION
    # ============================================================
    logger.info("")
    logger.info("PHASE 6: Trade Compilation")
    trade_list = compile_trade_list(v16_trades, router_output)
    logger.info(f"  {len(trade_list)} trades compiled")

    # ============================================================
    # PHASE 7: RECOMMENDATIONS
    # ============================================================
    logger.info("")
    logger.info("PHASE 7: Recommendations")
    recommendations = build_recommendations(router_output, concentration, config)
    logger.info(f"  Actionable: {recommendations.get('has_actionable_recommendations', False)}")
    logger.info(f"  Summary: {recommendations.get('summary_for_cio', '')[:120]}")

    # ============================================================
    # PHASE 8: OUTCOME TRACKER
    # ============================================================
    logger.info("")
    logger.info("PHASE 8: Outcome Tracker")
    outcome_tracker = own_history.get("outcome_tracker", [])
    outcome_tracker = manage_outcome_tracker(outcome_tracker, router_output, today, config)

    # Add new entry if router recommended entry
    entry_rec = entry_eval.get("recommendation")
    if entry_rec is not None and entry_rec.get("action") == "ENTRY_RECOMMENDATION":
        new_entry = create_outcome_entry(entry_rec, today)
        outcome_tracker.append(new_entry)
        logger.info(f"  New outcome entry: {new_entry.get('signal_id')}")

    logger.info(f"  Total tracked: {len(outcome_tracker)}")

    # ============================================================
    # PHASE 9: ROUTER STATE UPDATE
    # ============================================================
    logger.info("")
    logger.info("PHASE 9: Router State Update")
    updated_router_state = update_router_state_history(router_state, router_output, today)
    logger.info(f"  State: {updated_router_state.get('current_state')}")
    logger.info(f"  History: {len(updated_router_state.get('history_30d', []))} days")

    # ============================================================
    # PHASE 10: ASSEMBLE OUTPUT
    # ============================================================
    logger.info("")
    logger.info("PHASE 10: Assemble Output")

    signal_output = {
        "date": today.isoformat(),
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "execution_path": execution_path,

        "v16_trades": v16_trades,

        "f6_trades": {
            "source": "F6_PRODUCTION",
            "status": "UNAVAILABLE",
            "modified": False,
            "active_positions": [],
            "new_signals": [],
            "exits_today": [],
            "f6_advisory": {
                "router_proximity_warning": False,
                "router_active_warning": False,
                "detail": None,
                "suggestion": None,
                "binding": False,
            },
            "note": "F6 not live. Available in V2.",
        },

        "router": router_output,

        "perm_opt": {
            "status": "UNAVAILABLE",
            "note": "PermOpt available in V2 (after G7 Monitor)",
        },

        "projections": projections,
        "trade_list": trade_list,
        "recommendations": recommendations,
        "router_state": updated_router_state,
        "outcome_tracker": outcome_tracker,

        "context": {
            "system_regime": _extract_system_regime(market_analyst),
            "fragility_state": fragility_state,
            "layer_summary": market_analyst.get("layer_summary", {}),
            "g7_dominant_thesis": "UNAVAILABLE",
            "g7_last_review": "UNAVAILABLE",
        },
    }

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = round(time.time() - start_time, 2)
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"SIGNAL GENERATOR COMPLETE in {elapsed}s")
    logger.info(f"  Path: {execution_path}")
    logger.info(f"  Router: {router_output.get('current_state')} "
                f"(max prox: {router_output.get('max_proximity', 0):.2%})")
    logger.info(f"  V16: {v16_trades.get('v16_regime')} — {len(v16_trades.get('weights', {}))} assets")
    logger.info(f"  Trades: {len(trade_list)}")
    logger.info(f"  Recommendations: {'YES' if recommendations.get('has_actionable_recommendations') else 'NO'}")
    logger.info("=" * 60)

    return signal_output


# ============================================================
# INPUT EXTRACTION HELPERS
# ============================================================

def _extract_fragility_state(market_analyst: dict) -> str:
    """Extract Fragility State from Market Analyst output."""
    # Try nested structure first
    fs = market_analyst.get("fragility_state", {})
    if isinstance(fs, dict):
        return fs.get("state", "HEALTHY")
    if isinstance(fs, str):
        return fs
    return "HEALTHY"


def _extract_v16_regime(v16_data: dict) -> str:
    """Extract V16 regime from dashboard v16 block."""
    # dashboard.json uses "regime", other sources may use different keys
    regime = v16_data.get("regime")
    if regime:
        return regime
    regime = v16_data.get("regime_label")
    if regime:
        return regime
    regime = v16_data.get("v16_regime")
    if regime:
        return regime

    # Try to map state to regime
    state = v16_data.get("state_label", "")
    if "risk_on" in state.lower() or "risk-on" in state.lower():
        return "FULL_EXPANSION"
    if "selective" in state.lower():
        return "SELECTIVE"

    return "UNKNOWN"


def _extract_system_regime(market_analyst: dict) -> str:
    """Extract System Regime from Market Analyst."""
    sr = market_analyst.get("system_regime", {})
    if isinstance(sr, dict):
        return sr.get("regime", "UNKNOWN")
    if isinstance(sr, str):
        return sr
    return "UNKNOWN"
