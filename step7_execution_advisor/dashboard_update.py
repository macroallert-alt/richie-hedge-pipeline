"""
step7_execution_advisor/dashboard_update.py
Dashboard execution + rotation Block Writer.

Reads existing dashboard.json (latest.json), adds/updates:
  - 'execution' block (Trading Desk)
  - 'rotation' block (Rotation Circle)
  - 'v16.cluster_weights' updated to 9-cluster mapping

Source: Trading Desk Spec Teil 4 §18, Rotation Circle Spec Teil 4 §18.5
"""

import json
import logging
import os

logger = logging.getLogger("execution_advisor.dashboard_update")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "data", "dashboard", "latest.json"
)
CLUSTER_CONFIG_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "config", "cluster_config.json"
)


def update_dashboard_json(execution_output: dict) -> None:
    """
    Add execution + rotation blocks to dashboard.json.
    Also updates v16.cluster_weights to 9-cluster mapping.

    Read existing → update blocks → write back.
    """
    # Read existing
    try:
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        logger.info("dashboard.json loaded for update")
    except FileNotFoundError:
        logger.warning(f"dashboard.json not found at {DASHBOARD_PATH} — creating new")
        dashboard = {}
    except Exception as e:
        logger.error(f"Failed to read dashboard.json: {e}")
        return

    # 1. Build and write execution block
    dashboard["execution"] = _build_execution_block(execution_output)

    # 2. Write rotation block (pass-through from engine output)
    rotation_block = execution_output.get("rotation")
    if rotation_block:
        dashboard["rotation"] = rotation_block
        logger.info("dashboard.json: rotation block written")
    else:
        logger.warning("dashboard.json: no rotation block in engine output")

    # 3. Update v16.cluster_weights to 9-cluster mapping
    _update_cluster_weights(dashboard)

    # Write back
    try:
        os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
        with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False, default=str)
        logger.info("dashboard.json updated with execution + rotation blocks")
    except Exception as e:
        logger.error(f"Failed to write dashboard.json: {e}")


def _update_cluster_weights(dashboard: dict) -> None:
    """
    Update v16.cluster_weights to 9-cluster mapping.
    HYG moves from BOND to own cluster CREDIT.

    Source: Rotation Circle Spec Teil 4 §18.5
    """
    v16 = dashboard.get("v16")
    if not v16:
        return

    target_weights = v16.get("target_weights", {})
    if not target_weights:
        return

    # Load cluster config
    cluster_config = _load_cluster_config()
    if not cluster_config:
        return

    asset_to_cluster = cluster_config.get("asset_to_cluster", {})
    all_cluster_keys = list(cluster_config.get("clusters", {}).keys())

    # Recalculate cluster weights
    new_cluster_weights = {}
    for asset, weight in target_weights.items():
        if weight <= 0:
            continue
        cluster = asset_to_cluster.get(asset, "UNKNOWN")
        new_cluster_weights[cluster] = new_cluster_weights.get(cluster, 0.0) + weight

    # Ensure all 9 clusters present (even if 0)
    v16["cluster_weights"] = {
        c: round(new_cluster_weights.get(c, 0.0), 4) for c in all_cluster_keys
    }

    logger.info("v16.cluster_weights updated to 9-cluster mapping")


def _load_cluster_config() -> dict:
    """Load cluster_config.json."""
    try:
        with open(CLUSTER_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load cluster_config.json: {e}")
        return {}


def _build_execution_block(output: dict) -> dict:
    """
    Build the execution block for dashboard.json from the full output.

    Source: Trading Desk Spec Teil 4 §18.2
    """
    assessment = output.get("execution_assessment", {})
    cc = output.get("confirming_conflicting", {})
    rec = output.get("recommendation", {})
    event_win = output.get("event_window", {})
    v16_ctx = output.get("v16_context", {})

    # Dimensions — compact format for dashboard
    dims_full = assessment.get("dimensions", {})
    dims_compact = {}
    for dim_name, dim_data in dims_full.items():
        dims_compact[dim_name] = {
            "score": dim_data.get("score", 0),
            "max": dim_data.get("max", 3),
            "label": dim_data.get("label", ""),
        }

    # Top confirming/conflicting (max 3 each for dashboard)
    top_confirming = [
        f"{c['signal']} — {c['detail']}"
        for c in cc.get("confirming", [])[:3]
    ]
    top_conflicting = [
        f"{c['signal']} — {c['detail']}"
        for c in cc.get("conflicting", [])[:3]
    ]

    # Event window compact
    next_48h = event_win.get("next_48h", [])
    next_48h_events = []
    for e in next_48h:
        hours = e.get("hours_until", "?")
        next_48h_events.append(f"{e.get('event', 'Unknown')} ({hours}h)")

    event_density = event_win.get("event_density_14d", 0)
    if event_density >= 5:
        density_label = "HIGH"
    elif event_density >= 3:
        density_label = "ELEVATED"
    elif event_density >= 1:
        density_label = "NORMAL"
    else:
        density_label = "LOW"

    # Calendar upcoming (top 5)
    calendar_upcoming = []
    for e in event_win.get("calendar_upcoming", [])[:5]:
        entry = {
            "date": e.get("date", ""),
            "event": e.get("event", ""),
            "impact": e.get("impact", "MEDIUM"),
        }
        if e.get("hours_until", 0) <= 48:
            entry["hours_until"] = e["hours_until"]
        else:
            entry["days_until"] = e.get("days_until", 0)
        calendar_upcoming.append(entry)

    # Calendar monthly (top 10)
    calendar_monthly = event_win.get("calendar_monthly", [])[:10]

    return {
        "date": output.get("date", ""),
        "execution_level": assessment.get("execution_level", "UNKNOWN"),
        "total_score": assessment.get("total_score", 0),
        "max_score": assessment.get("max_possible", 18),
        "veto_applied": assessment.get("veto_applied", False),

        "dimensions": dims_compact,

        "confirming_count": cc.get("confirming_count", 0),
        "conflicting_count": cc.get("conflicting_count", 0),
        "net_assessment": cc.get("net_assessment", "UNKNOWN"),

        "top_confirming": top_confirming,
        "top_conflicting": top_conflicting,

        "recommendation_action": rec.get("action", "UNKNOWN"),
        "recommendation_short": rec.get("reasoning", ""),
        "specific_actions": rec.get("specific_actions", []),

        "event_window": {
            "next_48h_count": len(next_48h),
            "next_48h_events": next_48h_events,
            "next_14d_count": event_density,
            "convergence_week": bool(event_win.get("convergence_weeks")),
            "event_density_label": density_label,
        },

        "calendar_upcoming": calendar_upcoming,
        "calendar_monthly": calendar_monthly,

        "briefing_text": output.get("briefing_text", ""),

        "would_change_my_mind": rec.get("would_change_my_mind", {
            "execute_if": [],
            "hold_if": [],
        }),
    }
