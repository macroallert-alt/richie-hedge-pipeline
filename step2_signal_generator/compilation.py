"""
step2_signal_generator/compilation.py
Trade Compilation — V16 Trades with Attribution Tags

V1 Scope: Only V16 trades. F6, PermOpt, Fragility are UNAVAILABLE stubs.
Source: Signal Generator Spec Teil 3 §17
"""

import logging
from typing import Optional

logger = logging.getLogger("signal_generator.compilation")


def compile_v16_trades(v16_data: dict) -> dict:
    """
    Compile V16 trades block. Passed through, NEVER modified.
    Spec Teil 1 §1.2: Signal Generator does NOT modify V16 weights.

    Returns:
        {
            "source": "V16_PRODUCTION",
            "modified": False,
            "weights": {asset: {"weight": float, "attribution": "V16"}, ...},
            "rebalance_trades": [...],
            "v16_regime": str,
            "v16_state": str
        }
    """
    weights_raw = v16_data.get("current_weights") or v16_data.get("weights", {})
    regime = (v16_data.get("regime")
              or v16_data.get("regime_label")
              or v16_data.get("v16_regime")
              or "UNKNOWN")
    state = (v16_data.get("macro_state_name")
             or v16_data.get("state_label")
             or v16_data.get("v16_state")
             or "UNKNOWN")

    # Format weights with attribution
    weights = {}
    for asset, weight in weights_raw.items():
        if isinstance(weight, (int, float)):
            weights[asset] = {
                "weight": round(float(weight), 6),
                "attribution": "V16",
            }
        elif isinstance(weight, dict) and "weight" in weight:
            weights[asset] = {
                "weight": round(float(weight["weight"]), 6),
                "attribution": "V16",
            }

    # Rebalance trades (compute from weight changes if available)
    rebalance_trades = _extract_rebalance_trades(v16_data, weights)

    return {
        "source": "V16_PRODUCTION",
        "modified": False,
        "weights": weights,
        "rebalance_trades": rebalance_trades,
        "v16_regime": regime,
        "v16_state": state,
    }


def compile_trade_list(v16_trades: dict, router_output: dict) -> list:
    """
    Build consolidated trade list with attribution.
    Spec Teil 3 §17.2

    V1: Only V16 rebalance trades + Router recommendations (if any).
    F6, PermOpt, Fragility are stubs.
    """
    trades = []

    # V16 Rebalance Trades
    for trade in v16_trades.get("rebalance_trades", []):
        trades.append({
            "asset": trade.get("asset", "UNKNOWN"),
            "action": trade.get("action", "HOLD"),
            "weight_delta": trade.get("delta", 0.0),
            "target_weight": trade.get("target_weight", 0.0),
            "attribution": "V16",
            "status": "EXECUTABLE",
            "confidence": "VALIDATED",
            "expiry_condition": None,
        })

    # Router Entry Recommendation (if present)
    entry_rec = router_output.get("entry_evaluation", {}).get("recommendation")
    if entry_rec is not None and entry_rec.get("action") == "ENTRY_RECOMMENDATION":
        allocation = entry_rec.get("allocation", {})
        for etf, pct in allocation.get("distribution", {}).items():
            trades.append({
                "asset": etf,
                "action": "BUY",
                "weight_target": pct,
                "attribution": "ROUTER",
                "status": "RECOMMENDATION",
                "confidence": "UNVALIDATED",
                "trigger": entry_rec.get("trigger", "UNKNOWN"),
                "requires": "Agent R + Operator Freigabe",
            })

    # Router Exit Recommendation (if present)
    exit_check = router_output.get("exit_check")
    if exit_check is not None and exit_check.get("exit_triggered", False):
        # Determine which ETFs to exit based on current state
        current_state = router_output.get("current_state", "US_DOMESTIC")
        exit_etfs = _get_state_etfs(current_state)
        for etf in exit_etfs:
            trades.append({
                "asset": etf,
                "action": "SELL",
                "attribution": "ROUTER",
                "status": "RECOMMENDATION",
                "confidence": "UNVALIDATED",
                "reason": exit_check.get("reason", "Exit condition met"),
                "requires": "Agent R + Operator Freigabe",
            })

    # Emergency Exit (if present)
    emergency = router_output.get("emergency")
    if emergency is not None and emergency.get("action") == "EMERGENCY_EXIT":
        prev_state = emergency.get("previous_state", "")
        exit_etfs = _get_state_etfs(prev_state)
        for etf in exit_etfs:
            trades.append({
                "asset": etf,
                "action": "SELL",
                "attribution": "ROUTER",
                "status": "EMERGENCY",
                "confidence": "UNVALIDATED",
                "reason": emergency.get("reason", "Emergency exit"),
                "requires": "Immediate — Agent R + Operator",
            })

    return trades


def _extract_rebalance_trades(v16_data: dict, weights: dict) -> list:
    """
    Extract rebalance trades from V16 data.
    If V16 provides explicit trades, use those.
    Otherwise, compute from current vs previous weights.
    """
    # Check if V16 data has explicit trades
    explicit_trades = v16_data.get("rebalance_trades", [])
    if explicit_trades:
        return explicit_trades

    # Use weight_deltas from dashboard.json if available
    weight_deltas = v16_data.get("weight_deltas", {})
    if weight_deltas:
        trades = []
        for asset, delta in weight_deltas.items():
            try:
                delta_val = float(delta)
            except (ValueError, TypeError):
                continue
            target_w = weights.get(asset, {})
            target_weight = target_w.get("weight", 0.0) if isinstance(target_w, dict) else 0.0
            if abs(delta_val) > 0.0001:
                action = "BUY" if delta_val > 0 else "SELL"
                trades.append({
                    "asset": asset,
                    "action": action,
                    "delta": round(delta_val, 6),
                    "target_weight": round(target_weight, 6),
                    "attribution": "V16",
                    "source_system": "V16_PRODUCTION",
                })
        if trades:
            return trades

    # Fallback: mark all current weights as HOLD
    trades = []
    for asset, wdata in weights.items():
        w = wdata.get("weight", 0.0) if isinstance(wdata, dict) else 0.0
        if w > 0.001:
            trades.append({
                "asset": asset,
                "action": "HOLD",
                "delta": 0.0,
                "target_weight": w,
                "attribution": "V16",
                "source_system": "V16_PRODUCTION",
            })

    return trades


def _get_state_etfs(state: str) -> list:
    """Get ETFs associated with a Router state."""
    state_etfs = {
        "EM_BROAD": ["VWO", "INDA"],
        "CHINA_STIMULUS": ["FXI", "KWEB"],
        "COMMODITY_SUPER": ["GLD", "SLV", "DBC", "GDX"],
    }
    return state_etfs.get(state, [])
