"""
step_3_risk_officer/utils/helpers.py
Datum-Funktionen, Logging, Severity-Konstanten.
"""

import logging
from datetime import datetime, date, timezone

# ─── LOGGING ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("risk_officer")


def log_info(msg):
    logger.info(msg)


def log_warning(msg):
    logger.warning(msg)


def log_error(msg):
    logger.error(msg)


# ─── SEVERITY ─────────────────────────────────────────────────────

SEVERITY_ORDER = {
    "RESOLVED": 0,
    "MONITOR": 1,
    "WARNING": 2,
    "CRITICAL": 3,
    "EMERGENCY": 4
}

SEVERITY_LADDER = ["MONITOR", "WARNING", "CRITICAL"]


# ─── DATUM ────────────────────────────────────────────────────────

def today():
    """Heutiges Datum als date Objekt."""
    return date.today()


def now_iso():
    """Aktueller Zeitstempel als ISO-String (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date(date_str):
    """Parst Datum-String zu date Objekt. Unterstuetzt mehrere Formate."""
    if isinstance(date_str, date):
        return date_str
    if isinstance(date_str, datetime):
        return date_str.date()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def days_between(d1, d2):
    """Tage zwischen zwei Daten (positiv wenn d2 > d1)."""
    return (parse_date(d2) - parse_date(d1)).days


# ─── ALERT HELPERS ────────────────────────────────────────────────

_alert_counter = 0


def make_alert_id(date_str=None):
    """Generiert eindeutige Alert-ID: RO-YYYYMMDD-NNN."""
    global _alert_counter
    _alert_counter += 1
    d = date_str or today().strftime("%Y%m%d")
    if isinstance(d, date):
        d = d.strftime("%Y%m%d")
    d = d.replace("-", "")
    return f"RO-{d}-{_alert_counter:03d}"


def reset_alert_counter():
    """Reset fuer neuen Run."""
    global _alert_counter
    _alert_counter = 0


def make_alert(severity, message, check_id, affected_positions=None,
               affected_systems=None, trade_class=None, current_value=None,
               threshold=None, recommendation=None):
    """Erzeugt standardisierten Alert-Dict."""
    return {
        "id": make_alert_id(),
        "severity": severity,
        "check_id": check_id,
        "message": message,
        "affected_positions": affected_positions or [],
        "affected_systems": affected_systems or [],
        "trade_class": trade_class or "A",
        "current_value": current_value,
        "threshold": threshold,
        "recommendation": recommendation or "",
        "context": {},
        "previous_severity": None,
        "trend": "NEW",
        "days_active": 1,
        "base_severity": severity,
        "boost_applied": None
    }