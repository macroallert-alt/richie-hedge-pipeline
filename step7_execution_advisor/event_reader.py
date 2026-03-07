"""
step7_execution_advisor/event_reader.py
EVENT_CALENDAR.yaml Parser + Event Window Calculator.

Reads EVENT_CALENDAR.yaml from step_0k_event_calendar/ and computes:
  - Events in next 48h (HIGH and MEDIUM)
  - Events in next 14d (HIGH)
  - Convergence weeks (2+ HIGH events within 5 days)
  - Calendar upcoming (next N events for dashboard)
  - Calendar monthly (all events in current + next month)

Source: Trading Desk Spec Teil 3 §8, Teil 5 §22.1
"""

import logging
import os
from datetime import date, datetime, timedelta

import yaml

logger = logging.getLogger("execution_advisor.event_reader")

# Path to EVENT_CALENDAR.yaml (relative to repo root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
YAML_PATH = os.path.join(REPO_ROOT, "step_0k_event_calendar", "EVENT_CALENDAR.yaml")


def load_event_calendar() -> list[dict]:
    """
    Load events from EVENT_CALENDAR.yaml.

    Returns:
        List of event dicts, or empty list if file not found.
    """
    if not os.path.exists(YAML_PATH):
        logger.warning(f"EVENT_CALENDAR.yaml not found at {YAML_PATH}")
        return []

    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        events = data.get("events", [])
        logger.info(f"Loaded EVENT_CALENDAR.yaml: {len(events)} events")
        return events

    except Exception as e:
        logger.error(f"Failed to load EVENT_CALENDAR.yaml: {e}")
        return []


def compute_event_window(events: list[dict], today: date, config: dict = None) -> dict:
    """
    Compute the event window for the Execution Advisor.

    Args:
        events: List of event dicts from EVENT_CALENDAR.yaml
        today: Current date
        config: Optional execution_config.json event_window section

    Returns:
        {
            "next_48h": [...],
            "next_14d": [...],
            "convergence_weeks": [...],
            "event_density_14d": int,
            "calendar_upcoming": [...],
            "calendar_monthly": [...],
        }
    """
    config = config or {}
    upcoming_count = config.get("calendar_upcoming_count", 5)
    monthly_count = config.get("calendar_monthly_count", 10)

    next_48h = []
    next_14d = []
    all_future = []

    for event in events:
        try:
            event_date = date.fromisoformat(event["date"])
        except (ValueError, KeyError):
            continue

        days_until = (event_date - today).days

        if days_until < 0:
            continue  # Past event

        hours_until = days_until * 24  # Approximate

        future_entry = {
            "event": event.get("event", "Unknown"),
            "date": event["date"],
            "days_until": days_until,
            "hours_until": hours_until,
            "impact": event.get("impact", "MEDIUM"),
            "type": event.get("type", "UNKNOWN"),
            "themes": event.get("themes", []),
        }
        all_future.append(future_entry)

        if days_until <= 2:  # 48h window
            next_48h.append(future_entry)

        if days_until <= 14:
            next_14d.append(future_entry)

    # Convergence weeks
    convergence_weeks = _detect_convergence(events, today)

    # Event density in 14d
    event_density_14d = len(next_14d)

    # Calendar upcoming (next N events)
    calendar_upcoming = all_future[:upcoming_count]

    # Calendar monthly (current + next month)
    calendar_monthly = _build_calendar_monthly(events, today, monthly_count)

    return {
        "next_48h": next_48h,
        "next_14d": next_14d,
        "convergence_weeks": convergence_weeks,
        "event_density_14d": event_density_14d,
        "calendar_upcoming": calendar_upcoming,
        "calendar_monthly": calendar_monthly,
    }


def _detect_convergence(events: list[dict], today: date) -> list[dict]:
    """
    Detect weeks with 2+ HIGH-impact events within 5 trading days.

    Source: Trading Desk Spec Teil 3 §8.3
    """
    # Filter to HIGH events in next 21 days
    high_events = []
    for e in events:
        try:
            event_date = date.fromisoformat(e["date"])
        except (ValueError, KeyError):
            continue
        days_until = (event_date - today).days
        if 0 <= days_until <= 21 and e.get("impact") == "HIGH":
            high_events.append(e)

    convergence_weeks = []
    for i, e1 in enumerate(high_events):
        d1 = date.fromisoformat(e1["date"])
        cluster = [e1]
        for e2 in high_events[i + 1:]:
            d2 = date.fromisoformat(e2["date"])
            if abs((d2 - d1).days) <= 5:
                cluster.append(e2)
        if len(cluster) >= 2:
            dates = [date.fromisoformat(e["date"]) for e in cluster]
            cw = {
                "start": (min(dates) - timedelta(days=1)).isoformat(),
                "end": (max(dates) + timedelta(days=1)).isoformat(),
                "events": [e.get("event", "Unknown") for e in cluster],
                "risk_level": "HIGH" if len(cluster) >= 3 else "ELEVATED",
            }
            # Deduplicate by start+end
            if not any(
                existing["start"] == cw["start"] and existing["end"] == cw["end"]
                for existing in convergence_weeks
            ):
                convergence_weeks.append(cw)

    return convergence_weeks


def _build_calendar_monthly(events: list[dict], today: date,
                            max_count: int) -> list[dict]:
    """
    Build calendar entries for current + next month.
    """
    # Current month start and next month end
    current_month_start = today.replace(day=1)
    if today.month == 12:
        next_month_end = date(today.year + 1, 2, 1) - timedelta(days=1)
    elif today.month == 11:
        next_month_end = date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        next_month_end = date(today.year, today.month + 2, 1) - timedelta(days=1)

    monthly = []
    for event in events:
        try:
            event_date = date.fromisoformat(event["date"])
        except (ValueError, KeyError):
            continue

        if current_month_start <= event_date <= next_month_end:
            monthly.append({
                "date": event["date"],
                "event": event.get("event", "Unknown"),
                "type": event.get("type", "UNKNOWN"),
                "impact": event.get("impact", "MEDIUM"),
            })

    # Sort and limit
    monthly.sort(key=lambda x: x["date"])
    return monthly[:max_count]
