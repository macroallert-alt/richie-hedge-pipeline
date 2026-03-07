"""
step7_execution_advisor/dashboard_update.py
Dashboard execution Block Writer.

Reads existing dashboard.json, adds/updates the 'execution' block,
writes back. Does NOT modify any other blocks.

Source: Trading Desk Spec Teil 4 §18, Teil 5 §24.2
"""

import json
import logging
import os

logger = logging.getLogger("execution_advisor.dashboard_update")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "data", "dashboard", "latest.json"
)


def update_dashboard_json(execution_output: dict) -> None:
    """
    Add execution block to dashboard.json.
    Read existing → add execution → write back.

    IMPORTANT: Only touches the 'execution' key. All other blocks untouched.
    """
    # Read existing
    try:
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        logger.info("dashboard.json loaded for execution block update")
    except FileNotFoundError:
        logger.warning(f"dashboard.json not found at {DASHBOARD_PATH} — creating new")
        dashboard = {}
    except Exception as e:
        logger.error(f"Failed to read dashboard.json: {e}")
        return

    # Build execution block
    dashboard["execution"] = _build_execution_block(execution_output)

    # Write back
    try:
        os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
        with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False, default=str)
        logger.info("dashboard.json updated with execution block")
    except Exception as e:
        logger.error(f"Failed to write dashboard.json: {e}")


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
