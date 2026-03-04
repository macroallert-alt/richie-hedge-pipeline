"""
step_3_risk_officer/utils/mappings.py
Laedt Config-Dateien und stellt Asset-Mapping-Funktionen bereit.
"""

import os
import yaml
from datetime import date
from .helpers import log_info, log_error, parse_date

# ─── CONFIG PFADE ─────────────────────────────────────────────────

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def _load_yaml(filename):
    """Laedt eine YAML-Datei aus dem config/ Verzeichnis."""
    path = os.path.join(_CONFIG_DIR, filename)
    if not os.path.exists(path):
        log_error(f"Config file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── CONFIG LOADER ────────────────────────────────────────────────

def load_risk_config():
    """Laedt RISK_CONFIG.yaml."""
    cfg = _load_yaml("RISK_CONFIG.yaml")
    if cfg is None:
        raise FileNotFoundError("RISK_CONFIG.yaml not found — cannot proceed.")
    log_info(f"Risk config loaded: version {cfg.get('version', 'unknown')}")
    return cfg


def load_event_calendar():
    """Laedt EVENT_CALENDAR.yaml und gibt anstehende Events zurueck."""
    cal = _load_yaml("EVENT_CALENDAR.yaml")
    if cal is None:
        return []
    events = cal.get("events_2026", [])
    return events


def load_asset_mappings():
    """Laedt ASSET_MAPPINGS.yaml."""
    mappings = _load_yaml("ASSET_MAPPINGS.yaml")
    if mappings is None:
        raise FileNotFoundError("ASSET_MAPPINGS.yaml not found — cannot proceed.")
    return mappings


# ─── ASSET MAPPING FUNKTIONEN ────────────────────────────────────

_mappings_cache = None


def _get_mappings():
    global _mappings_cache
    if _mappings_cache is None:
        _mappings_cache = load_asset_mappings()
    return _mappings_cache


def get_sector_breakdown(asset):
    """
    Gibt Sektor-Breakdown fuer ein Asset zurueck.
    Direkte Sektor-ETFs: {"Tech": 1.0}
    Multi-Sektor: {"Tech": 0.30, "Financials": 0.13, ...}
    Unbekannt: {"Other": 1.0}
    """
    m = _get_mappings()
    return m.get("asset_to_sector", {}).get(asset, {"Other": 1.0})


def get_asset_class(asset):
    """
    Gibt Asset-Klasse zurueck: Equity_US, Equity_International,
    Bonds, Commodities, Cash_Equivalent.
    """
    m = _get_mappings()
    return m.get("asset_to_class", {}).get(asset, "Equity_US")


def is_international_asset(asset):
    """Prueft ob ein Asset international ist."""
    m = _get_mappings()
    return asset in m.get("international_assets", [])


def get_upcoming_events(reference_date=None):
    """
    Gibt Events zurueck die nach reference_date liegen,
    mit days_until berechnet.
    """
    ref = reference_date or date.today()
    raw_events = load_event_calendar()
    upcoming = []
    for evt in raw_events:
        try:
            evt_date = parse_date(evt["date"])
        except (ValueError, KeyError):
            continue
        days_until = (evt_date - ref).days
        if days_until >= 0:
            upcoming.append({
                "event": evt["event"],
                "date": str(evt_date),
                "days_until": days_until
            })
    upcoming.sort(key=lambda x: x["days_until"])
    return upcoming