"""
step_0k_event_calendar/opex_calculator.py
Rule-based OpEx / Quad Witching date calculator.

No LLM, no web search. Pure arithmetic.
Source: Trading Desk Spec Teil 2 §6.3
"""

from datetime import date, timedelta

QUAD_MONTHS = {3, 6, 9, 12}


def calculate_opex_dates(start_date: date, months_ahead: int = 6) -> list[dict]:
    """
    Calculate Monthly OpEx and Quad Witching dates.

    Rule: 3rd Friday of each month.
    Quad Witching: March, June, September, December.

    Args:
        start_date: Start date (typically today)
        months_ahead: How many months to look ahead (default 6)

    Returns:
        List of event dicts in the standard calendar format
    """
    events = []

    current = start_date.replace(day=1)
    for _ in range(months_ahead):
        third_friday = _third_friday(current.year, current.month)

        # Only include future dates
        if third_friday >= start_date:
            is_quad = current.month in QUAD_MONTHS

            quarter = (current.month - 1) // 3 + 1
            if is_quad:
                notes = (
                    f"Q{quarter} Quarterly Options + Futures Expiry. "
                    f"Auto-calculated."
                )
            else:
                notes = "Monthly Options + Futures Expiry. Auto-calculated."

            events.append({
                "date": third_friday.isoformat(),
                "event": "Quad Witching" if is_quad else "Monthly Options Expiry",
                "type": "OPEX",
                "subtype": "QUAD_WITCHING" if is_quad else "MONTHLY_OPEX",
                "impact": "HIGH" if is_quad else "MEDIUM",
                "themes": ["VOLATILITY", "LIQUIDITY"],
                "time_et": "ALL_DAY",
                "notes": notes,
                "source_verified": True,
                "source_url": None,
                "consensus": {"populated": False},
                "portfolio_sensitivity": {"populated": False},
                "outcome": {"populated": False},
            })

        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return events


def _third_friday(year: int, month: int) -> date:
    """
    Calculate the 3rd Friday of a given month.

    Algorithm:
    1. Find the weekday of the 1st of the month
    2. Calculate days until first Friday
    3. Add 2 weeks = 3rd Friday
    """
    first_day = date(year, month, 1)
    # Monday=0 ... Friday=4
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday
