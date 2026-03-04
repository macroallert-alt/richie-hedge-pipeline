"""
Market Analyst — Catalysts Module
Maps upcoming events to per-layer catalyst exposure.

Tier 1 (Critical): FOMC, CPI, NFP
Tier 2 (High): Treasury Refunding, OpEx, ECB, BOJ, Earnings Season
Tier 3 (Dynamic): Populated by IC high-novelty claims at runtime

Source: AGENT2_SPEC_TEIL5 Section 17
"""

from datetime import datetime, date, timedelta


def calculate_catalyst_exposure(
    layer_name: str,
    catalyst_config: dict,
    ic_catalysts: list,
    today: date = None,
) -> list:
    """
    Per layer: which catalysts are approaching?

    layer_name: full layer name
    catalyst_config: from catalysts.json
    ic_catalysts: list of IC-identified tier 3 catalysts
    today: current date

    Returns: sorted list of exposures (nearest first)
    """
    if today is None:
        today = datetime.utcnow().date()
    elif isinstance(today, datetime):
        today = today.date()

    exposures = []

    # Tier 1 + Tier 2 from calendar
    for tier_key, tier_num in [("tier_1_critical", 1), ("tier_2_high", 2)]:
        events = catalyst_config.get(tier_key, [])
        for event in events:
            if layer_name not in event.get("affects_layers", []):
                continue

            next_date = _get_next_event_date(event, today)
            if next_date is None:
                continue

            days_until = (next_date - today).days

            # Only events within pre_event_days window
            if days_until > event.get("pre_event_days", 3):
                continue

            # Also include if we're in post_event window (just happened)
            if days_until < -event.get("post_event_days", 1):
                continue

            exposures.append({
                "event": event["name"],
                "tier": tier_num,
                "days_until": days_until,
                "direction": event.get("direction", "UNKNOWN"),
                "impact": event.get("typical_impact", "MEDIUM"),
                "pre_event_action": (
                    "REDUCE_CONVICTION" if event.get("direction") == "BINARY"
                    else "MONITOR"
                ),
                "notes": event.get("notes", ""),
            })

    # Tier 3 from IC
    for ic_cat in (ic_catalysts or []):
        affected = ic_cat.get("affects_layers", [])
        if layer_name in affected:
            exposures.append({
                "event": ic_cat.get("event", "Unknown IC catalyst"),
                "tier": 3,
                "days_until": ic_cat.get("days_until"),
                "direction": ic_cat.get("direction", "UNKNOWN"),
                "impact": ic_cat.get("impact", "MEDIUM"),
                "pre_event_action": "MONITOR",
                "notes": ic_cat.get("description", ""),
            })

    return sorted(exposures, key=lambda x: x.get("days_until") or 999)


def _get_next_event_date(event: dict, today: date) -> date:
    """
    Finds the next occurrence of an event relative to today.
    Uses dates_YYYY list if available, otherwise returns None.
    """
    year = today.year
    dates_key = f"dates_{year}"
    date_strings = event.get(dates_key, [])

    if not date_strings:
        return None

    # Find the nearest future (or very recent past) date
    best = None
    for ds in date_strings:
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        # Consider dates from a few days ago (post-event window) to future
        post_days = event.get("post_event_days", 1)
        if d >= today - timedelta(days=post_days):
            if best is None or d < best:
                best = d

    return best
