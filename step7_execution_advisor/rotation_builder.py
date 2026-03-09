"""
step7_execution_advisor/rotation_builder.py
Rotation Block Builder — Berechnet den kompletten rotation Block fuer latest.json.

Aufgaben:
  1. weights_history.json lesen
  2. Heutigen Eintrag berechnen und appenden (idempotent)
  3. Deltas berechnen (1d, 1W, 1M, 3M, 6M, 9M, 1Y)
  4. Cluster-Deltas aggregieren
  5. Materialitaet bestimmen pro Cluster und Asset
  6. NEU/EXIT/SHIFT/HOLD Labels bestimmen
  7. Status + Mode bestimmen (ALIGNED/SHIFTING/BIG_ROTATION)
  8. Trigger erkennen (STATE_CHANGE/DD_PROTECT/CONFLUENCE/DRIFT/NONE)
  9. Sparkline-Daten extrahieren (letzte 30 Eintraege)
  10. State-Change Events fuer Sparkline-Overlay extrahieren
  11. State-History aufbauen (letzte 5 Wechsel mit Snapshots)
  12. Comparison Snapshots fuer Zeitspalten zusammenstellen
  13. Vollstaendigen rotation Block als Dict zurueckgeben

Source: Rotation Circle Spec Teil 2 + Teil 4 §18.3
"""

import json
import logging
import os
from datetime import date, timedelta

logger = logging.getLogger("execution_advisor.rotation_builder")


# ============================================================
# SIGNAL MAPPING: int → readable string
# ============================================================
GROWTH_SIGNAL_MAP = {1: "POSITIVE", -1: "NEGATIVE", 0: "NEUTRAL"}
LIQ_DIRECTION_MAP = {1: "EASING", -1: "TIGHTENING", 0: "NEUTRAL"}


def _map_growth_signal(value) -> str:
    """Map growth_signal from int or string to readable string."""
    if isinstance(value, str) and value in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        return value
    return GROWTH_SIGNAL_MAP.get(value, "UNKNOWN")


def _map_liq_direction(value) -> str:
    """Map liq_direction from int or string to readable string."""
    if isinstance(value, str) and value in ("EASING", "TIGHTENING", "NEUTRAL"):
        return value
    return LIQ_DIRECTION_MAP.get(value, "UNKNOWN")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def build_rotation_block(
    latest_data: dict,
    weights_history_path: str,
    cluster_config_path: str,
    today: date = None,
) -> dict:
    """
    Berechnet den kompletten rotation Block fuer latest.json.

    Side effect: Appendet heutigen Eintrag an weights_history.json.

    Args:
        latest_data: Dict mit v16, signals, execution Bloecken
        weights_history_path: Pfad zu weights_history.json
        cluster_config_path: Pfad zu cluster_config.json
        today: Override fuer Tests

    Returns:
        rotation Block (dict) fuer latest.json
    """
    today = today or date.today()

    logger.info("Building rotation block...")

    # ── Load cluster config ──
    cluster_config = _load_cluster_config(cluster_config_path)
    asset_to_cluster = cluster_config["asset_to_cluster"]
    all_cluster_keys = list(cluster_config["clusters"].keys())
    cluster_display_names = {
        k: v["display_name"] for k, v in cluster_config["clusters"].items()
    }

    # ── Load weights history ──
    history = _load_weights_history(weights_history_path)

    # ── Extract current V16 data ──
    v16 = latest_data.get("v16", {})
    target_weights = v16.get("target_weights", {})

    # Filter to active weights only (> 0)
    active_weights = {k: round(v, 4) for k, v in target_weights.items() if v > 0}

    # Calculate cluster weights (9-cluster mapping)
    cluster_weights = _calculate_cluster_weights(active_weights, asset_to_cluster)
    active_clusters = {k: v for k, v in cluster_weights.items() if v > 0}

    # ── Build today's history entry ──
    yesterday_entry = history[-1] if history else None

    state_changed = False
    if yesterday_entry:
        prev_state = yesterday_entry.get("macro_state_num")
        curr_state = v16.get("macro_state_num", 0)
        state_changed = (prev_state != curr_state)

    signals = latest_data.get("signals", {})

    today_entry = {
        "date": today.isoformat(),
        "regime": v16.get("regime", "UNKNOWN"),
        "macro_state_num": v16.get("macro_state_num", 0),
        "macro_state_name": v16.get("macro_state_name", "UNKNOWN"),
        "router_state": signals.get("router_state", "UNKNOWN"),
        "router_days_in_state": signals.get("router_days_in_state", 0),
        "dd_protect_active": v16.get("dd_protect_status", "INACTIVE") != "INACTIVE",
        "state_changed": state_changed,
        "growth_signal": _map_growth_signal(v16.get("growth_signal", 0)),
        "liq_direction": _map_liq_direction(v16.get("liq_direction", 0)),
        "stress_score": v16.get("stress_score", 0),
        "weights": active_weights,
        "cluster_weights": active_clusters,
        "total_weight": round(sum(active_weights.values()), 4),
    }

    # ── Append to history (idempotent) ──
    _append_to_history(history, today_entry, weights_history_path)

    # ── Comparison snapshots (1d, 1W, 1M, 3M, 6M, 9M, 1Y) ──
    snapshot_offsets = {
        "1d": 1, "1w": 7, "1m": 30, "3m": 90,
        "6m": 180, "9m": 270, "1y": 365,
    }
    comparison_snapshots = {}
    for key, days_back in snapshot_offsets.items():
        target_date = today - timedelta(days=days_back)
        snap = _find_snapshot(history, target_date)
        if snap:
            comparison_snapshots[key] = {
                "date": snap["date"],
                "weights": snap.get("weights", {}),
                "cluster_weights": snap.get("cluster_weights", {}),
            }
        else:
            comparison_snapshots[key] = None

    # ── Delta calculations per asset ──
    delta_keys = ["1d", "1w", "1m", "3m"]
    asset_deltas = {}
    for dk in delta_keys:
        snap = comparison_snapshots.get(dk)
        if snap:
            snap_weights = snap.get("weights", {})
            asset_deltas[dk] = _calculate_deltas(active_weights, snap_weights)
        else:
            asset_deltas[dk] = {}

    # ── Cluster delta aggregation ──
    cluster_deltas_raw = {}
    for dk in delta_keys:
        cluster_deltas_raw[dk] = _aggregate_to_clusters(
            asset_deltas[dk], asset_to_cluster
        )

    # ── Build cluster_deltas output ──
    cluster_deltas_output = {}
    for ck in sorted(active_clusters.keys(),
                     key=lambda x: active_clusters.get(x, 0), reverse=True):
        d1d = cluster_deltas_raw["1d"].get(ck, 0.0)
        d1w = cluster_deltas_raw["1w"].get(ck) if cluster_deltas_raw["1w"] else None
        d1m = cluster_deltas_raw["1m"].get(ck) if cluster_deltas_raw["1m"] else None
        d3m = cluster_deltas_raw["3m"].get(ck) if cluster_deltas_raw["3m"] else None

        cluster_deltas_output[ck] = {
            "delta_1d": round(d1d, 4),
            "delta_1w": round(d1w, 4) if d1w is not None else None,
            "delta_1m": round(d1m, 4) if d1m is not None else None,
            "delta_3m": round(d3m, 4) if d3m is not None else None,
            "materiality": _get_materiality_label(d1d),
            "direction": _get_direction(d1d),
        }

    # Also include clusters that were active yesterday but not today (EXIT)
    if comparison_snapshots.get("1d"):
        prev_clusters = comparison_snapshots["1d"].get("cluster_weights", {})
        for ck in prev_clusters:
            if ck not in cluster_deltas_output and prev_clusters[ck] > 0:
                d1d = 0.0 - prev_clusters[ck]
                cluster_deltas_output[ck] = {
                    "delta_1d": round(d1d, 4),
                    "delta_1w": None,
                    "delta_1m": None,
                    "delta_3m": None,
                    "materiality": _get_materiality_label(d1d),
                    "direction": "FALLING",
                }

    # ── Build asset_details output ──
    asset_details = {}
    all_assets_today = set(active_weights.keys())
    all_assets_yesterday = set()
    if comparison_snapshots.get("1d"):
        all_assets_yesterday = set(comparison_snapshots["1d"].get("weights", {}).keys())
    all_assets = all_assets_today | all_assets_yesterday

    for asset in all_assets:
        w_today = active_weights.get(asset, 0.0)
        w_yesterday = 0.0
        if comparison_snapshots.get("1d"):
            w_yesterday = comparison_snapshots["1d"].get("weights", {}).get(asset, 0.0)

        d1d = asset_deltas["1d"].get(asset, 0.0)
        d1w = asset_deltas["1w"].get(asset) if asset_deltas["1w"] else None
        d1m = asset_deltas["1m"].get(asset) if asset_deltas["1m"] else None
        d3m = asset_deltas["3m"].get(asset) if asset_deltas["3m"] else None

        cluster_key = asset_to_cluster.get(asset, "UNKNOWN")

        asset_details[asset] = {
            "weight": round(w_today, 4),
            "cluster": cluster_key,
            "cluster_name": cluster_display_names.get(cluster_key, cluster_key),
            "delta_1d": round(d1d, 4),
            "delta_1w": round(d1w, 4) if d1w is not None else None,
            "delta_1m": round(d1m, 4) if d1m is not None else None,
            "delta_3m": round(d3m, 4) if d3m is not None else None,
            "materiality": _get_materiality_label(d1d),
            "direction": _get_direction(d1d),
            "label": _determine_asset_label(w_today, w_yesterday),
        }

    # Sort asset_details by weight descending
    asset_details = dict(sorted(
        asset_details.items(),
        key=lambda x: x[1]["weight"],
        reverse=True,
    ))

    # ── New / Exited positions ──
    new_positions = []
    exited_positions = []
    for asset, detail in asset_details.items():
        if detail["label"] == "NEW":
            new_positions.append({
                "asset": asset,
                "cluster": detail["cluster"],
                "cluster_name": detail["cluster_name"],
                "weight": detail["weight"],
            })
        elif detail["label"] == "EXIT":
            w_prev = 0.0
            if comparison_snapshots.get("1d"):
                w_prev = comparison_snapshots["1d"].get("weights", {}).get(asset, 0.0)
            exited_positions.append({
                "asset": asset,
                "cluster": detail["cluster"],
                "cluster_name": detail["cluster_name"],
                "prev_weight": round(w_prev, 4),
            })

    # ── cluster_current output ──
    cluster_current = {}
    for ck in sorted(active_clusters.keys(),
                     key=lambda x: active_clusters.get(x, 0), reverse=True):
        cluster_current[ck] = {
            "weight": round(active_clusters[ck], 4),
            "display_name": cluster_display_names.get(ck, ck),
        }

    # ── Status + Mode ──
    cluster_deltas_1d_values = {
        ck: cd.get("delta_1d", 0.0)
        for ck, cd in cluster_deltas_output.items()
    }
    total_abs_delta = sum(abs(v) for v in cluster_deltas_1d_values.values())
    material_shifts = sum(
        1 for cd in cluster_deltas_output.values()
        if cd["materiality"] != "GREEN"
    )

    status, mode = _determine_rotation_status(
        cluster_deltas_1d_values, state_changed, total_abs_delta
    )

    # ── Days since material rotation ──
    days_since_material = _days_since_material_rotation(history, today)

    # ── Days since state change ──
    days_since_state_change = _days_since_state_change(history, today)

    # ── Trigger ──
    trigger = _detect_trigger(today_entry, yesterday_entry, latest_data)

    # ── DD-Protect block ──
    dd_protect = {
        "active": today_entry["dd_protect_active"],
        "threshold": v16.get("dd_protect_threshold", -15.0),
        "current_drawdown": v16.get("current_drawdown", 0.0),
    }

    # ── Sparkline data (last 30 entries) ──
    sparkline_data = _build_sparkline_data(history, all_cluster_keys)

    # ── Sparkline state changes ──
    sparkline_state_changes = _build_sparkline_state_changes(history)

    # ── State history (last 5 state changes with snapshots) ──
    state_history = _build_state_history(history, cluster_display_names)

    # ── Assemble rotation block ──
    rotation_block = {
        "date": today.isoformat(),

        "status": status,
        "mode": mode,

        "total_absolute_delta_pp": round(total_abs_delta * 100, 1),
        "material_shifts_count": material_shifts,
        "days_since_material_rotation": days_since_material,
        "days_since_state_change": days_since_state_change,

        "trigger": trigger,

        "dd_protect": dd_protect,

        "cluster_current": cluster_current,

        "cluster_deltas": cluster_deltas_output,

        "asset_details": asset_details,

        "new_positions": new_positions,
        "exited_positions": exited_positions,

        "sparkline_data": sparkline_data,
        "sparkline_state_changes": sparkline_state_changes,

        "state_history": state_history,

        "comparison_snapshots": comparison_snapshots,
    }

    logger.info(f"  Status: {status}")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Material Shifts: {material_shifts}")
    logger.info(f"  Total Abs Delta: {total_abs_delta * 100:.1f}pp")
    logger.info(f"  Trigger: {trigger['type']}")

    return rotation_block


# ============================================================
# FILE I/O
# ============================================================

def _load_cluster_config(path: str) -> dict:
    """Load cluster_config.json."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load cluster_config.json: {e}")
        # Return minimal fallback
        return {"clusters": {}, "asset_to_cluster": {}}


def _load_weights_history(path: str) -> list:
    """Load weights_history.json."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Failed to load weights_history.json: {e}")
        return []


def _append_to_history(history: list, entry: dict, path: str) -> None:
    """Append today's entry to history (idempotent: replace if same date)."""
    today_str = entry["date"]

    # Idempotent: remove existing entry for today
    if history and history[-1]["date"] == today_str:
        history.pop()

    history.append(entry)

    # Write back
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"  weights_history.json: {len(history)} entries")
    except Exception as e:
        logger.error(f"Failed to write weights_history.json: {e}")


# ============================================================
# CLUSTER CALCULATIONS
# ============================================================

def _calculate_cluster_weights(weights: dict, asset_to_cluster: dict) -> dict:
    """Calculate cluster weights from asset weights using 9-cluster mapping."""
    cluster_weights = {}
    for asset, weight in weights.items():
        cluster = asset_to_cluster.get(asset, "UNKNOWN")
        cluster_weights[cluster] = cluster_weights.get(cluster, 0.0) + weight
    return {k: round(v, 4) for k, v in cluster_weights.items()}


# ============================================================
# DELTA CALCULATIONS
# ============================================================

def _find_snapshot(history: list, target_date: date) -> dict | None:
    """
    Find closest history entry to target_date.
    Max 5 days tolerance (weekends/holidays).
    """
    if not history:
        return None

    best = None
    best_dist = float("inf")
    for entry in history:
        entry_date = date.fromisoformat(entry["date"])
        dist = abs((entry_date - target_date).days)
        if dist < best_dist:
            best = entry
            best_dist = dist

    if best_dist > 5:
        return None
    return best


def _calculate_deltas(current_weights: dict, snapshot_weights: dict) -> dict:
    """
    Calculate deltas between current and historical weights.
    Returns {asset: delta} for all assets with material difference.
    """
    all_assets = set(current_weights.keys()) | set(snapshot_weights.keys())
    deltas = {}
    for asset in all_assets:
        curr = current_weights.get(asset, 0.0)
        prev = snapshot_weights.get(asset, 0.0)
        delta = curr - prev
        if abs(delta) > 0.0001:
            deltas[asset] = round(delta, 4)
    return deltas


def _aggregate_to_clusters(asset_deltas: dict, asset_to_cluster: dict) -> dict:
    """Sum asset deltas to cluster deltas."""
    cluster_deltas = {}
    for asset, delta in asset_deltas.items():
        cluster = asset_to_cluster.get(asset, "UNKNOWN")
        cluster_deltas[cluster] = cluster_deltas.get(cluster, 0.0) + delta
    return {k: round(v, 4) for k, v in cluster_deltas.items()}


# ============================================================
# MATERIALITY + LABELS
# ============================================================

def _get_materiality_label(delta: float) -> str:
    """Materiality based on absolute delta (Spec §4.5)."""
    abd = abs(delta)
    if abd < 0.02:
        return "GREEN"
    if abd < 0.05:
        return "YELLOW"
    if abd < 0.10:
        return "ORANGE"
    return "RED"


def _get_direction(delta: float) -> str:
    """Direction arrow logic (Spec §11.3)."""
    if delta > 0.001:
        return "RISING"
    if delta < -0.001:
        return "FALLING"
    return "STABLE"


def _determine_asset_label(weight_today: float, weight_yesterday: float) -> str:
    """NEW/EXIT/SHIFT/HOLD label (Spec §4.6)."""
    if weight_yesterday == 0 and weight_today > 0:
        return "NEW"
    if weight_yesterday > 0 and weight_today == 0:
        return "EXIT"
    if abs(weight_today - weight_yesterday) < 0.001:
        return "HOLD"
    return "SHIFT"


# ============================================================
# STATUS + MODE
# ============================================================

def _determine_rotation_status(
    cluster_deltas_1d: dict, state_changed: bool, total_abs_delta: float
) -> tuple[str, str]:
    """
    Determine rotation status and mode (Spec §4.3).
    Returns: (status, mode)
    """
    if state_changed:
        return "BIG_ROTATION", "STATE_TRANSITION"
    if total_abs_delta > 0.15:
        return "BIG_ROTATION", "STABLE"

    max_delta = max((abs(d) for d in cluster_deltas_1d.values()), default=0)
    if max_delta >= 0.02:
        return "SHIFTING", "STABLE"

    return "ALIGNED", "STABLE"


# ============================================================
# DAYS SINCE CALCULATIONS
# ============================================================

def _days_since_material_rotation(history: list, today: date) -> int:
    """Find days since last entry with any cluster delta >= 2pp."""
    # Walk backwards through history looking for material change
    for i in range(len(history) - 1, 0, -1):
        curr = history[i]
        prev = history[i - 1]
        curr_clusters = curr.get("cluster_weights", {})
        prev_clusters = prev.get("cluster_weights", {})
        all_keys = set(curr_clusters.keys()) | set(prev_clusters.keys())
        for ck in all_keys:
            delta = abs(curr_clusters.get(ck, 0) - prev_clusters.get(ck, 0))
            if delta >= 0.02:
                entry_date = date.fromisoformat(curr["date"])
                return (today - entry_date).days
    return 999  # No material rotation found


def _days_since_state_change(history: list, today: date) -> int:
    """Find days since last state change in history."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("state_changed"):
            entry_date = date.fromisoformat(history[i]["date"])
            return (today - entry_date).days
    # No state change found — use router_days_in_state as fallback
    if history:
        return history[-1].get("router_days_in_state", 999)
    return 999


# ============================================================
# TRIGGER DETECTION
# ============================================================

def _detect_trigger(today_entry: dict, yesterday_entry: dict | None,
                    latest_data: dict) -> dict:
    """Detect rotation trigger (Spec §4.4)."""
    if yesterday_entry is None:
        return {"type": "NONE", "label": "Keine materielle Rotation", "detail": None}

    # 1. State Change
    if today_entry.get("state_changed"):
        return {
            "type": "STATE_CHANGE",
            "label": (
                f"State {yesterday_entry['macro_state_num']} → "
                f"{today_entry['macro_state_num']}: "
                f"{today_entry['macro_state_name']}"
            ),
            "detail": (
                f"Growth: {today_entry['growth_signal']}, "
                f"Liquidity: {today_entry['liq_direction']}"
            ),
        }

    # 2. DD-Protect toggle
    if today_entry.get("dd_protect_active") != yesterday_entry.get("dd_protect_active"):
        action = "AKTIVIERT" if today_entry["dd_protect_active"] else "DEAKTIVIERT"
        v16 = latest_data.get("v16", {})
        return {
            "type": "DD_PROTECT",
            "label": f"DD-Protect {action} bei {v16.get('current_drawdown', 0):.1f}%",
            "detail": f"Threshold: {v16.get('dd_protect_threshold', -15)}%",
        }

    # 3. Growth signal change
    if today_entry.get("growth_signal") != yesterday_entry.get("growth_signal"):
        return {
            "type": "CONFLUENCE",
            "label": (
                f"Growth Signal: {yesterday_entry['growth_signal']} → "
                f"{today_entry['growth_signal']}"
            ),
            "detail": None,
        }

    # 4. Liquidity direction change
    if today_entry.get("liq_direction") != yesterday_entry.get("liq_direction"):
        return {
            "type": "CONFLUENCE",
            "label": (
                f"Liquidity: {yesterday_entry['liq_direction']} → "
                f"{today_entry['liq_direction']}"
            ),
            "detail": None,
        }

    # 5. Check if any material delta exists
    today_clusters = today_entry.get("cluster_weights", {})
    yesterday_clusters = yesterday_entry.get("cluster_weights", {})
    all_keys = set(today_clusters.keys()) | set(yesterday_clusters.keys())
    max_delta = 0
    for ck in all_keys:
        d = abs(today_clusters.get(ck, 0) - yesterday_clusters.get(ck, 0))
        max_delta = max(max_delta, d)

    if max_delta >= 0.02:
        return {
            "type": "DRIFT",
            "label": "Kursbedingte Gewichtsanpassung",
            "detail": None,
        }

    return {"type": "NONE", "label": "Keine materielle Rotation", "detail": None}


# ============================================================
# SPARKLINE DATA
# ============================================================

def _build_sparkline_data(history: list, all_cluster_keys: list) -> dict:
    """Build sparkline data from last 30 history entries (Spec §4.7)."""
    last_30 = history[-30:]
    if len(last_30) < 2:
        return {}

    # Find clusters active in any of the 30 days
    active_clusters = set()
    for entry in last_30:
        for ck, w in entry.get("cluster_weights", {}).items():
            if w > 0:
                active_clusters.add(ck)

    sparklines = {}
    dates = [e["date"] for e in last_30]
    for ck in sorted(active_clusters):
        values = []
        for entry in last_30:
            values.append(round(entry.get("cluster_weights", {}).get(ck, 0.0), 4))
        sparklines[ck] = {
            "dates": dates,
            "values": values,
        }

    return sparklines


def _build_sparkline_state_changes(history: list) -> list:
    """Find state changes within sparkline window (last 30 entries) (Spec §4.7)."""
    last_30 = history[-30:]
    changes = []
    for entry in last_30:
        if entry.get("state_changed"):
            changes.append({
                "date": entry["date"],
                "from_state": 0,  # Not stored directly — would need prev entry
                "to_state": entry.get("macro_state_num", 0),
                "from_name": "",
                "to_name": entry.get("macro_state_name", ""),
            })
    return changes


# ============================================================
# STATE HISTORY
# ============================================================

def _build_state_history(history: list, cluster_display_names: dict) -> list:
    """Build state history — last 5 state changes with snapshots (Spec §4.8)."""
    state_changes = []

    for i in range(1, len(history)):
        entry = history[i]
        if entry.get("state_changed"):
            prev = history[i - 1]

            snap_before = prev.get("cluster_weights", {})
            snap_after = entry.get("cluster_weights", {})

            new_clusters = [
                cluster_display_names.get(ck, ck)
                for ck in snap_after
                if snap_after.get(ck, 0) > 0 and snap_before.get(ck, 0) == 0
            ]
            exited_clusters = [
                cluster_display_names.get(ck, ck)
                for ck in snap_before
                if snap_before.get(ck, 0) > 0 and snap_after.get(ck, 0) == 0
            ]

            # Infer trigger reason
            trigger_reason = _infer_state_change_reason(prev, entry)

            state_changes.append({
                "date": entry["date"],
                "from_state": prev.get("macro_state_num", 0),
                "to_state": entry.get("macro_state_num", 0),
                "from_name": prev.get("macro_state_name", "UNKNOWN"),
                "to_name": entry.get("macro_state_name", "UNKNOWN"),
                "snapshot_before": snap_before,
                "snapshot_after": snap_after,
                "new_clusters": new_clusters,
                "exited_clusters": exited_clusters,
                "trigger_reason": trigger_reason,
            })

    # Return last 5, most recent first
    return list(reversed(state_changes[-5:]))


def _infer_state_change_reason(prev: dict, curr: dict) -> str:
    """Infer why a state change happened based on changed fields."""
    reasons = []
    if prev.get("growth_signal") != curr.get("growth_signal"):
        reasons.append(
            f"Growth Signal: {prev.get('growth_signal')} → {curr.get('growth_signal')}"
        )
    if prev.get("liq_direction") != curr.get("liq_direction"):
        reasons.append(
            f"Liquidity: {prev.get('liq_direction')} → {curr.get('liq_direction')}"
        )
    if prev.get("dd_protect_active") != curr.get("dd_protect_active"):
        action = "AKTIVIERT" if curr.get("dd_protect_active") else "DEAKTIVIERT"
        reasons.append(f"DD-Protect {action}")

    if reasons:
        return " + ".join(reasons)
    return "Macro State Wechsel"
