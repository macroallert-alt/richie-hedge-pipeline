"""
Cycles Circle — Main Orchestrator
Baldur Creek Capital | Step 0v (V4.3 — LLM Cycle Narrative)

1. Collect all data (incremental persistence)
2. Run phase detection for all 10 cycles
3. Save results locally (cycle_data.json)
3b. Generate chart data (cycles_chart_data.json) — with smoothed curve, phase zones, cycle position
3c. Lead-Engine V1.1 — calibration + early warning (conditional_returns, regime_interaction, transition_engine)
3d. LLM Cycle Narrative — Claude Sonnet synthesizes all engine results into investor-ready German text
4. Write to Cycles Sheet (DASHBOARD, PHASES, HISTORY)
5. Write cycles block to latest.json (for CyclesCard) — EXTENDED with indicator values
6. Git commit + push data files

Usage:
  python -m step_0v_cycles.main [--force-backfill] [--skip-sheet] [--skip-git]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cycles.main")

from .data_collector import collect_all_data
from .phase_engine import detect_all_phases
from .config import (
    CYCLES_SHEET_ID,
    CYCLES_DASHBOARD_TAB, CYCLES_PHASES_TAB, CYCLES_HISTORY_TAB,
    CYCLE_DEFINITIONS, CYCLE_NAMES,
    DATA_DIR, HISTORY_DIR,
)

# Cycle display order
CYCLE_ORDER = [
    "LIQUIDITY", "CREDIT", "COMMODITY", "CHINA_CREDIT",
    "DOLLAR", "BUSINESS", "FED_RATES", "EARNINGS",
    "TRADE", "POLITICAL",
]

TIER_MAP = {
    "LIQUIDITY": 1, "CREDIT": 1, "COMMODITY": 1, "CHINA_CREDIT": 1,
    "DOLLAR": 2, "BUSINESS": 2, "FED_RATES": 2, "EARNINGS": 2,
    "TRADE": 3, "POLITICAL": 3,
}

QUALITY_MAP = {
    "LIQUIDITY": "HIGH", "CREDIT": "HIGH", "COMMODITY": "LOW",
    "CHINA_CREDIT": "MEDIUM", "DOLLAR": "MEDIUM", "BUSINESS": "HIGH",
    "FED_RATES": "HIGH", "EARNINGS": "HIGH", "TRADE": "HIGH", "POLITICAL": "HIGH",
}

NCYCLES_MAP = {
    "LIQUIDITY": 8, "CREDIT": 5, "COMMODITY": 2, "CHINA_CREDIT": 5,
    "DOLLAR": 3, "BUSINESS": 6, "FED_RATES": 7, "EARNINGS": 7,
    "TRADE": 8, "POLITICAL": 24,
}


# ---------------------------------------------------------------------------
# Sheets write service
# ---------------------------------------------------------------------------

def _get_sheets_write():
    import tempfile
    try:
        from googleapiclient.discovery import build
        from google.oauth2.service_account import Credentials
    except ImportError:
        logger.error("googleapiclient not installed")
        return None

    sa_key = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
    if not sa_key:
        logger.error("No GCP credentials")
        return None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(sa_key)
            tmp = f.name
        creds = Credentials.from_service_account_file(
            tmp, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        os.unlink(tmp)
        return build("sheets", "v4", credentials=creds, cache_discovery=False)
    except Exception as e:
        logger.error(f"Sheets write auth failed: {e}")
        return None


def _write_sheet(service, range_str, values):
    try:
        service.spreadsheets().values().update(
            spreadsheetId=CYCLES_SHEET_ID, range=range_str,
            valueInputOption="RAW", body={"values": values},
        ).execute()
        logger.info(f"  Written: {range_str}")
    except Exception as e:
        logger.error(f"  Write failed [{range_str}]: {e}")


def _append_sheet(service, range_str, values):
    try:
        service.spreadsheets().values().append(
            spreadsheetId=CYCLES_SHEET_ID, range=range_str,
            valueInputOption="RAW", insertDataOption="INSERT_ROWS",
            body={"values": values},
        ).execute()
        logger.info(f"  Appended: {range_str}")
    except Exception as e:
        logger.error(f"  Append failed [{range_str}]: {e}")


# ---------------------------------------------------------------------------
# Write DASHBOARD
# ---------------------------------------------------------------------------

def write_dashboard(service, result):
    logger.info("Writing DASHBOARD...")
    s = result.get("summary", {})
    nt = s.get("next_turn") or {}

    one_liner = (f"Alignment {result['alignment_score']}/10 ({result['alignment_label']}) | "
                 f"{s.get('bullish', 0)} bullish, {s.get('bearish', 0)} bearish, "
                 f"{s.get('neutral', 0)} neutral")
    dz_count = s.get("in_danger_zone", 0)
    if dz_count:
        one_liner += f" | {dz_count} Danger Zone(s)"

    row = [[
        date.today().isoformat(),
        result.get("current_regime", "UNKNOWN"),
        result.get("alignment_score", ""),
        result.get("alignment_label", ""),
        s.get("bullish", ""),
        s.get("bearish", ""),
        s.get("neutral", ""),
        nt.get("cycle", ""),
        nt.get("phase", ""),
        nt.get("months", ""),
        "",
        "HIGH" if dz_count >= 2 else "MEDIUM" if dz_count >= 1 else "LOW",
        one_liner,
    ]]
    _write_sheet(service, f"{CYCLES_DASHBOARD_TAB}!A5", row)


# ---------------------------------------------------------------------------
# Write PHASES
# ---------------------------------------------------------------------------

def write_phases(service, result):
    logger.info("Writing PHASES...")
    rows = []
    for cid in CYCLE_ORDER:
        c = result.get("cycles", {}).get(cid, {})
        dz = c.get("danger_zone", {})
        val = c.get("indicator_value")
        rows.append([
            date.today().isoformat(),
            cid,
            CYCLE_NAMES.get(cid, cid),
            TIER_MAP.get(cid, ""),
            c.get("phase", "UNKNOWN"),
            c.get("phase_confidence", ""),
            c.get("phase_duration_months", ""),
            _fmt(c.get("velocity"), 6),
            _fmt(c.get("acceleration"), 6),
            c.get("velocity_z_score", ""),
            "",  # amplitude
            dz.get("zone_name", ""),
            dz.get("distance_absolute", ""),
            str(c.get("in_danger_zone", False)).upper(),
            c.get("v16_alignment", ""),
            _fmt(val, 2),
            _fmt(c.get("indicator_12m_ma"), 2),
            QUALITY_MAP.get(cid, ""),
            NCYCLES_MAP.get(cid, ""),
        ])
    _write_sheet(service, f"{CYCLES_PHASES_TAB}!A5", rows)


# ---------------------------------------------------------------------------
# Write HISTORY (append)
# ---------------------------------------------------------------------------

def write_history(service, result):
    logger.info("Writing HISTORY...")
    rows = []
    for cid in CYCLE_ORDER:
        c = result.get("cycles", {}).get(cid, {})
        dz = c.get("danger_zone", {})
        val = c.get("indicator_value")
        rows.append([
            date.today().isoformat(),
            cid,
            c.get("phase", "UNKNOWN"),
            c.get("phase_confidence", ""),
            _fmt(c.get("velocity"), 6),
            _fmt(c.get("acceleration"), 6),
            _fmt(val, 2),
            dz.get("distance_absolute", ""),
            result.get("alignment_score", ""),
            result.get("current_regime", ""),
            "",  # phase_changed
            "",  # notes
        ])
    _append_sheet(service, f"{CYCLES_HISTORY_TAB}!A5", rows)


# ---------------------------------------------------------------------------
# Update latest.json — EXTENDED with indicator values per cycle
# ---------------------------------------------------------------------------

def update_latest_json(result):
    paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "dashboard", "latest.json"),
        "data/dashboard/latest.json",
    ]
    latest_path = None
    for p in paths:
        if os.path.exists(p):
            latest_path = p
            break

    if not latest_path:
        logger.warning("latest.json not found — skipping")
        return

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = json.load(f)

        s = result.get("summary", {})
        nt = s.get("next_turn") or {}

        latest["cycles"] = {
            "date": date.today().isoformat(),
            "detected_at": result.get("detected_at", ""),
            "alignment_score": result.get("alignment_score"),
            "alignment_label": result.get("alignment_label"),
            "current_regime": result.get("current_regime"),
            "bullish": s.get("bullish", 0),
            "bearish": s.get("bearish", 0),
            "neutral": s.get("neutral", 0),
            "in_danger_zone": s.get("in_danger_zone", 0),
            "next_turn_cycle": nt.get("cycle"),
            "next_turn_months": nt.get("months"),
            "one_liner": (f"Alignment {result['alignment_score']}/10 ({result['alignment_label']}) | "
                          f"{s.get('bullish', 0)} bullish, {s.get('bearish', 0)} bearish"),
            "cycle_phases": {
                cid: {
                    "phase": c.get("phase"),
                    "confidence": c.get("phase_confidence"),
                    "tier": c.get("tier"),
                    "v16_alignment": c.get("v16_alignment"),
                    "in_danger_zone": c.get("in_danger_zone"),
                    "indicator_value": _fmt(c.get("indicator_value"), 4),
                    "indicator_12m_ma": _fmt(c.get("indicator_12m_ma"), 4),
                    "velocity": _fmt(c.get("velocity"), 6),
                    "percentile": c.get("percentile"),
                    "danger_zone": c.get("danger_zone"),
                }
                for cid, c in result.get("cycles", {}).items()
            },
        }

        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(latest, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"latest.json updated with cycles block (extended)")

    except Exception as e:
        logger.error(f"Failed to update latest.json: {e}")


# ---------------------------------------------------------------------------
# Save phase result locally
# ---------------------------------------------------------------------------

def save_phase_result(result):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "cycle_data.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Phase result saved → {path}")
    except Exception as e:
        logger.error(f"Save failed: {e}")


# ---------------------------------------------------------------------------
# Generate Chart Data (V4.1 — Smoothed + Phase Zones + Cycle Position)
# ---------------------------------------------------------------------------

# Mapping: cycle_id → how to extract the indicator time series from raw data
CHART_INDICATOR_EXTRACTORS = {
    "LIQUIDITY":    {"type": "liquidity", "column": "Fed_Net_Liq"},
    "CREDIT":       {"type": "fred", "key": "HY_OAS_FRED", "multiply": 100},
    "COMMODITY":    {"type": "computed_crb_real"},
    "CHINA_CREDIT": {"type": "computed_copper_gold"},
    "DOLLAR":       {"type": "fred", "key": "DXY"},
    "BUSINESS":     {"type": "fred_yoy", "key": "INDPRO"},
    "FED_RATES":    {"type": "computed_real_ffr"},
    "EARNINGS":     {"type": "fred_qyoy", "key": "CORP_PROFITS"},
    "TRADE":        {"type": "fred_yoy", "key": "CASS"},
    "POLITICAL":    {"type": "none"},
}

# Primary asset overlay per cycle (from config.py CYCLE_DEFINITIONS)
CHART_ASSET_OVERLAY = {
    "LIQUIDITY": "DBC", "CREDIT": "HYG", "COMMODITY": "DBC",
    "CHINA_CREDIT": "DBC", "DOLLAR": "GLD", "BUSINESS": "SPY",
    "FED_RATES": "GLD", "EARNINGS": "SPY", "TRADE": "DBC",
    "POLITICAL": "SPY",
}

# ---------------------------------------------------------------------------
# Phase classification for monthly data points (simplified from phase_engine.py)
# ---------------------------------------------------------------------------

MONTHLY_PHASE_CLASSIFIERS = {
    "CREDIT": lambda v, ma, vel: (
        "DISTRESS" if v > 700 else
        "DETERIORATION" if v > 500 and vel and vel > 0 else
        "DETERIORATION" if v > 400 and vel and vel > 0 else
        "EXPANSION" if v < 350 and vel is not None and vel <= 0 else
        "LATE_EXPANSION" if v < 350 and vel and vel > 0 else
        "RECOVERY" if v < 500 and vel and vel < 0 else
        "REPAIR" if v > 500 and vel and vel < 0 else
        "EXPANSION"
    ),
    "LIQUIDITY": lambda v, ma, vel: (
        "CONTRACTION" if ma and v < ma and vel and vel < 0 else
        "EARLY_RECOVERY" if vel and vel > 0 and ma and v < ma else
        "LATE_EXPANSION" if ma and v > ma and vel and vel < 0 else
        "EXPANSION" if ma and v > ma and vel and vel > 0 else
        "EXPANSION"
    ),
    "COMMODITY": lambda v, ma, vel: (
        "BEAR" if ma and v < ma and vel and vel < 0 else
        "EARLY_BULL" if ma and v < ma and vel and vel > 0 else
        "EUPHORIA" if ma and v > ma and vel and vel > 0.05 else
        "MID_BULL" if ma and v > ma and vel and vel > 0 else
        "MID_BULL"
    ),
    "CHINA_CREDIT": lambda v, ma, vel: (
        "CONTRACTION" if ma and v < ma and vel and vel < 0 else
        "EARLY_STIMULUS" if vel and vel > 0 and ma and v < ma else
        "PEAK" if ma and v > ma and vel and vel < 0 else
        "EXPANSION" if ma and v > ma and vel and vel > 0 else
        "EXPANSION"
    ),
    "DOLLAR": lambda v, ma, vel: (
        "STRENGTHENING" if vel and vel > 0 else
        "WEAKENING" if vel and vel < 0 else
        "PLATEAU"
    ),
    "BUSINESS": lambda v, ma, vel: (
        "RECESSION" if v < -2 else
        "TROUGH" if v < 0 and vel and vel > 0 else
        "EARLY_RECOVERY" if v >= 0 and v < 2 and vel and vel > 0 else
        "EXPANSION" if v >= 2 else
        "LATE_EXPANSION" if v > 0 and vel and vel < 0 else
        "EXPANSION"
    ),
    "FED_RATES": lambda v, ma, vel: (
        "EASING" if v < 0 else
        "NEUTRAL" if abs(v) <= 2 else
        "RESTRICTIVE" if v > 2 else
        "NEUTRAL"
    ),
    "EARNINGS": lambda v, ma, vel: (
        "CONTRACTION" if v < 0 and vel and vel < 0 else
        "TROUGH" if v < 0 and vel and vel > 0 else
        "RECOVERY" if v >= 0 and v < 5 and vel and vel > 0 else
        "EXPANSION" if v >= 5 else
        "LATE_EXPANSION" if v > 0 and vel and vel < 0 else
        "EXPANSION"
    ),
    "TRADE": lambda v, ma, vel: (
        "COLLAPSE" if v < -10 else
        "CONTRACTION" if v < -5 else
        "TROUGH" if v < 0 and vel and vel > 0 else
        "RECOVERY" if v >= 0 and v < 3 else
        "EXPANSION" if v >= 3 else
        "RECOVERY"
    ),
}

# Phase → color category for frontend
PHASE_COLOR_CATEGORY = {
    "EXPANSION": "green", "EARLY_RECOVERY": "green", "RECOVERY": "green",
    "MID_BULL": "green", "EARLY_BULL": "green", "EARLY_STIMULUS": "green",
    "EASING": "green", "NEUTRAL": "green",
    "LATE_EXPANSION": "yellow", "PEAK": "yellow", "PLATEAU": "yellow",
    "LATE": "yellow", "TIGHTENING": "yellow", "RESTRICTIVE": "yellow",
    "WITHDRAWAL": "yellow", "REPAIR": "yellow",
    "CONTRACTION": "orange", "DETERIORATION": "orange",
    "STRENGTHENING": "orange", "WEAKENING": "orange", "BEAR": "orange",
    "TROUGH": "red", "DISTRESS": "red", "RECESSION": "red",
    "COLLAPSE": "red", "EUPHORIA": "red",
}


def _resample_monthly(series):
    """Resample a daily/weekly/mixed series to monthly (last value per month)."""
    if not series:
        return []
    monthly = {}
    for pt in series:
        d = pt.get("date", "")
        v = pt.get("value")
        if not d or v is None:
            continue
        ym = d[:7]
        monthly[ym] = v
    return [{"date": ym, "value": round(v, 6)} for ym, v in sorted(monthly.items())]


def _compute_ma(monthly_series, window=12):
    """Compute moving average over monthly series."""
    result = []
    vals = [pt["value"] for pt in monthly_series]
    for i in range(len(monthly_series)):
        if i < window - 1:
            result.append({"date": monthly_series[i]["date"], "value": None})
        else:
            window_vals = vals[i - window + 1: i + 1]
            avg = sum(window_vals) / len(window_vals)
            result.append({"date": monthly_series[i]["date"], "value": round(avg, 6)})
    return result


def _compute_smoothed(monthly_series, window=12):
    """Compute 2x smoothed curve: MA of MA (the 'sine wave').
    First pass: 12M MA. Second pass: 12M MA of that.
    This removes noise and shows the pure cycle rhythm.
    """
    if len(monthly_series) < window * 2:
        return []
    ma1 = _compute_ma(monthly_series, window)
    ma1_clean = [pt for pt in ma1 if pt["value"] is not None]
    if len(ma1_clean) < window:
        return ma1
    ma2 = _compute_ma(ma1_clean, window)
    return ma2


def _compute_monthly_velocity(monthly_series, lookback=3):
    """Compute velocity (rate of change) for monthly series."""
    vels = {}
    vals = [(pt["date"], pt["value"]) for pt in monthly_series]
    for i in range(lookback, len(vals)):
        cur = vals[i][1]
        prev = vals[i - lookback][1]
        if prev and prev != 0 and cur is not None:
            vels[vals[i][0]] = (cur - prev) / abs(prev)
    return vels


def _compute_phase_zones(cycle_id, monthly_series, ma_12m):
    """Compute historical phase zones for a cycle.
    Returns list of {"start": "YYYY-MM", "end": "YYYY-MM", "phase": "...", "color": "..."}.
    """
    classifier = MONTHLY_PHASE_CLASSIFIERS.get(cycle_id)
    if not classifier or len(monthly_series) < 24:
        return []

    ma_map = {pt["date"]: pt["value"] for pt in ma_12m} if ma_12m else {}
    vel_map = _compute_monthly_velocity(monthly_series, lookback=3)

    phases_by_month = []
    for pt in monthly_series:
        d = pt["date"]
        v = pt["value"]
        ma = ma_map.get(d)
        vel = vel_map.get(d)
        if v is None:
            continue
        try:
            phase = classifier(v, ma, vel)
        except Exception:
            phase = "UNKNOWN"
        phases_by_month.append({"date": d, "phase": phase})

    if not phases_by_month:
        return []

    zones = []
    current_phase = phases_by_month[0]["phase"]
    current_start = phases_by_month[0]["date"]

    for i in range(1, len(phases_by_month)):
        if phases_by_month[i]["phase"] != current_phase:
            zones.append({
                "start": current_start,
                "end": phases_by_month[i - 1]["date"],
                "phase": current_phase,
                "color": PHASE_COLOR_CATEGORY.get(current_phase, "gray"),
            })
            current_phase = phases_by_month[i]["phase"]
            current_start = phases_by_month[i]["date"]

    zones.append({
        "start": current_start,
        "end": phases_by_month[-1]["date"],
        "phase": current_phase,
        "color": PHASE_COLOR_CATEGORY.get(current_phase, "gray"),
    })

    return zones


def _compute_cycle_position(cycle_id, current_phase, phase_zones):
    """Compute where we are in the current cycle."""
    typical = CYCLE_DEFINITIONS.get(cycle_id, {}).get("typical_duration_months", 0)

    phase_start = None
    months_in_phase = 0
    if phase_zones:
        for zone in reversed(phase_zones):
            if zone["phase"] == current_phase:
                phase_start = zone["start"]
                try:
                    sy, sm = int(zone["start"][:4]), int(zone["start"][5:7])
                    ey, em = int(zone["end"][:4]), int(zone["end"][5:7])
                    months_in_phase = (ey - sy) * 12 + (em - sm) + 1
                except (ValueError, IndexError):
                    months_in_phase = 0
                break

    cycle_start = None
    if phase_zones and len(phase_zones) >= 2:
        for i in range(len(phase_zones) - 1, 0, -1):
            prev_color = phase_zones[i - 1].get("color", "")
            curr_color = phase_zones[i].get("color", "")
            if prev_color in ("red", "orange") and curr_color == "green":
                cycle_start = phase_zones[i]["start"]
                break

    months_since_cycle_start = 0
    if cycle_start:
        try:
            sy, sm = int(cycle_start[:4]), int(cycle_start[5:7])
            ny, nm = date.today().year, date.today().month
            months_since_cycle_start = (ny - sy) * 12 + (nm - sm)
        except (ValueError, IndexError):
            pass

    pct = round(months_since_cycle_start / typical * 100, 1) if typical > 0 else 0
    est_remaining = max(0, typical - months_since_cycle_start) if typical > 0 else 0

    return {
        "current_phase": current_phase,
        "phase_start": phase_start,
        "months_in_phase": months_in_phase,
        "cycle_start": cycle_start,
        "months_since_cycle_start": months_since_cycle_start,
        "typical_duration_months": typical,
        "pct_complete": min(pct, 150),
        "estimated_months_remaining": est_remaining,
    }


def _extract_indicator_series(cycle_id, data):
    """Extract the primary indicator time series for a cycle from raw data."""
    cfg = CHART_INDICATOR_EXTRACTORS.get(cycle_id)
    if not cfg or cfg["type"] == "none":
        return []

    if cfg["type"] == "liquidity":
        col = cfg["column"]
        raw = data.get("liquidity", [])
        series = [{"date": r["date"], "value": r.get(col)}
                  for r in raw if r.get(col) is not None and r.get("date")]
        series.sort(key=lambda x: x["date"])
        return series

    if cfg["type"] == "fred":
        key = cfg["key"]
        mult = cfg.get("multiply", 1)
        raw = data.get("fred", {}).get(key, [])
        series = [{"date": r["date"], "value": r["value"] * mult}
                  for r in raw if r.get("value") is not None and r.get("date")]
        series.sort(key=lambda x: x["date"])
        return series

    if cfg["type"] == "fred_yoy":
        key = cfg["key"]
        raw = data.get("fred", {}).get(key, [])
        raw = [r for r in raw if r.get("value") is not None and r.get("date")]
        raw.sort(key=lambda x: x["date"])
        if len(raw) < 13:
            return []
        result = []
        for i in range(12, len(raw)):
            c = raw[i]["value"]
            a = raw[i - 12]["value"]
            if a and a != 0:
                result.append({"date": raw[i]["date"],
                               "value": round((c - a) / abs(a) * 100, 2)})
        return result

    if cfg["type"] == "fred_qyoy":
        key = cfg["key"]
        raw = data.get("fred", {}).get(key, [])
        raw = [r for r in raw if r.get("value") is not None and r.get("date")]
        raw.sort(key=lambda x: x["date"])
        if len(raw) < 5:
            return []
        result = []
        for i in range(4, len(raw)):
            c = raw[i]["value"]
            a = raw[i - 4]["value"]
            if a and a != 0:
                result.append({"date": raw[i]["date"],
                               "value": round((c - a) / abs(a) * 100, 2)})
        return result

    if cfg["type"] == "computed_crb_real":
        dbc = data.get("prices", {}).get("DBC", [])
        cpi_raw = data.get("fred", {}).get("CPI", [])
        if not dbc or not cpi_raw:
            return []
        dbc_sorted = sorted([p for p in dbc if p.get("price") and p.get("date")],
                            key=lambda x: x["date"])
        cpi_map = {}
        for c in cpi_raw:
            if c.get("value") and c.get("date"):
                cpi_map[c["date"][:7]] = c["value"]
        result = []
        for p in dbc_sorted:
            ym = p["date"][:7]
            cv = cpi_map.get(ym)
            if cv is None:
                for k in sorted(cpi_map.keys(), reverse=True):
                    if k <= ym:
                        cv = cpi_map[k]
                        break
            if cv and cv > 0:
                result.append({"date": p["date"], "value": round(p["price"] / cv * 100, 4)})
        return result

    if cfg["type"] == "computed_copper_gold":
        copper = data.get("prices", {}).get("COPPER", [])
        gold = data.get("prices", {}).get("GLD", [])
        if not copper or not gold:
            return []
        copper_sorted = sorted([p for p in copper if p.get("price") and p.get("date")],
                               key=lambda x: x["date"])
        gold_map = {g["date"]: g["price"] for g in gold if g.get("price") and g.get("date")}
        result = []
        for p in copper_sorted:
            gp = gold_map.get(p["date"])
            if gp and gp > 0:
                result.append({"date": p["date"], "value": round(p["price"] / gp, 6)})
        return result

    if cfg["type"] == "computed_real_ffr":
        ff_raw = data.get("fred", {}).get("FEDFUNDS", [])
        cpi_raw = data.get("fred", {}).get("CPI", [])
        if not ff_raw or not cpi_raw:
            return []
        ff_sorted = sorted([r for r in ff_raw if r.get("value") is not None and r.get("date")],
                           key=lambda x: x["date"])
        cpi_sorted = sorted([r for r in cpi_raw if r.get("value") is not None and r.get("date")],
                            key=lambda x: x["date"])
        if len(cpi_sorted) < 13:
            return []
        cpi_yoy_map = {}
        for i in range(12, len(cpi_sorted)):
            c = cpi_sorted[i]["value"]
            a = cpi_sorted[i - 12]["value"]
            if a and a != 0:
                cpi_yoy_map[cpi_sorted[i]["date"][:7]] = round((c - a) / a * 100, 2)
        result = []
        for ff in ff_sorted:
            ym = ff["date"][:7]
            cy = cpi_yoy_map.get(ym)
            if cy is None:
                for k in sorted(cpi_yoy_map.keys(), reverse=True):
                    if k <= ym:
                        cy = cpi_yoy_map[k]
                        break
            if cy is not None:
                result.append({"date": ff["date"], "value": round(ff["value"] - cy, 2)})
        return result

    return []


def _extract_asset_series(ticker, data):
    """Extract an asset price series from raw data."""
    raw = data.get("prices", {}).get(ticker, [])
    series = [{"date": p["date"], "value": p["price"]}
              for p in raw if p.get("price") and p.get("date")]
    series.sort(key=lambda x: x["date"])
    return series


def generate_chart_data(data, result):
    """Generate cycles_chart_data.json with monthly time series for frontend charts.

    Per cycle (V4.1):
      - indicator: monthly resampled primary indicator
      - ma_12m: 12-month moving average
      - smoothed: 2x12M MA (the 'sine wave' — pure cycle rhythm)
      - asset_overlay: monthly resampled primary asset
      - phase_zones: historical phase classification as colored zones
      - cycle_position: where we are in the current cycle (for Cycle Clock)
    """
    logger.info("Generating chart data (V4.1 — smoothed + phases + clock)...")
    chart_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date.today().isoformat(),
        "cycles": {},
    }

    for cid in CYCLE_ORDER:
        cdef = CYCLE_DEFINITIONS.get(cid, {})
        asset_ticker = CHART_ASSET_OVERLAY.get(cid)
        phase_result = result.get("cycles", {}).get(cid, {})
        current_phase = phase_result.get("phase", "UNKNOWN")

        # Extract indicator series
        indicator_raw = _extract_indicator_series(cid, data)
        indicator_monthly = _resample_monthly(indicator_raw)

        # Compute 12M MA
        ma_12m = _compute_ma(indicator_monthly, 12) if indicator_monthly else []

        # Compute smoothed (2x12M MA — the sine wave)
        smoothed = _compute_smoothed(indicator_monthly, 12) if indicator_monthly else []

        # Extract asset overlay
        asset_raw = _extract_asset_series(asset_ticker, data) if asset_ticker else []
        asset_monthly = _resample_monthly(asset_raw)

        # Compute phase zones (historical phase classification)
        phase_zones = _compute_phase_zones(cid, indicator_monthly, ma_12m)

        # Compute cycle position (where we are in the cycle)
        cycle_position = _compute_cycle_position(cid, current_phase, phase_zones)

        n_indicator = len(indicator_monthly)
        n_asset = len(asset_monthly)
        n_smoothed = len([s for s in smoothed if s.get("value") is not None])
        n_zones = len(phase_zones)

        chart_data["cycles"][cid] = {
            "name": CYCLE_NAMES.get(cid, cid),
            "tier": TIER_MAP.get(cid, 0),
            "asset_ticker": asset_ticker,
            "indicator_count": n_indicator,
            "asset_count": n_asset,
            "smoothed_count": n_smoothed,
            "phase_zones_count": n_zones,
            "current_phase": current_phase,
            "current_value": _fmt(phase_result.get("indicator_value"), 4),
            "indicator": indicator_monthly,
            "ma_12m": ma_12m,
            "smoothed": smoothed,
            "asset_overlay": asset_monthly,
            "phase_zones": phase_zones,
            "cycle_position": cycle_position,
        }

        logger.info(f"  {cid}: {n_indicator} ind, {n_smoothed} smooth, "
                     f"{n_zones} zones, pos={cycle_position.get('pct_complete', 0):.0f}%")

    # Save
    path = os.path.join(DATA_DIR, "cycles_chart_data.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chart_data, f, indent=1, ensure_ascii=False, default=str)
        size_kb = os.path.getsize(path) / 1024
        logger.info(f"Chart data saved → {path} ({size_kb:.0f} KB)")
    except Exception as e:
        logger.error(f"Chart data save failed: {e}")

    return chart_data


# ---------------------------------------------------------------------------
# Step 3d: LLM Cycle Narrative (V4.3 — NEW)
# ---------------------------------------------------------------------------

CYCLE_NARRATIVE_SYSTEM_PROMPT = """Du bist der Chief Investment Officer von Baldur Creek Capital, einem systematischen Macro-Hedge-Fund. 
Du schreibst eine wöchentliche Zyklen-Analyse auf Deutsch für erfahrene Investoren.

REGELN:
- Schreibe klar, direkt, ohne Füllwörter. Wie ein Bloomberg-Terminal-Kommentar.
- Verwende konkrete Zahlen aus den Daten (Prozente, Monate, Wahrscheinlichkeiten).
- Nenne Assets beim Ticker (SPY, GLD, TLT, etc.).
- Strukturiere in genau 4 Abschnitte:
  1. GESAMTLAGE (2-3 Sätze): Was sagen die Zyklen zusammen?
  2. ZYKLEN-DETAIL (3-4 Sätze): Welche Zyklen sind kritisch, welche stabil?
  3. RISIKO (2-3 Sätze): V16-Transition, Crash-Wahrscheinlichkeit, was müsste passieren für Eskalation?
  4. POSITIONIERUNG (2-3 Sätze): Welche Assets bevorzugt, welche meiden? Konkrete Implikationen.
- Keine Überschriften, keine Bullet Points. Fließtext in 4 Absätzen.
- Maximal 350 Wörter gesamt.
- Sei ehrlich über Unsicherheiten. Wenn die Daten gemischt sind, sag das."""


def _build_narrative_prompt(lead_results):
    """Build the user prompt with key data for the LLM narrative."""
    assessment = lead_results.get("assessment", {})
    phase_pos = lead_results.get("phase_positions", {})
    cascade = lead_results.get("cascade", {})
    confirmation = lead_results.get("confirmation", {})
    v16_trans = lead_results.get("v16_transitions", {})
    crash_corr = lead_results.get("crash_correction", {})
    analogues = lead_results.get("analogues", {})

    # Phase positions summary
    phase_lines = []
    for cid in ["LIQUIDITY", "CREDIT", "COMMODITY", "CHINA_CREDIT",
                "DOLLAR", "BUSINESS", "FED_RATES", "EARNINGS", "TRADE"]:
        pp = phase_pos.get(cid, {})
        if pp:
            phase_lines.append(
                f"  {cid}: {pp.get('current_phase', '?')} | "
                f"Position: {pp.get('phase_position_pct', '?')}% | "
                f"Status: {pp.get('status', '?')} | "
                f"Remaining: ~{pp.get('remaining_median', '?')} Mo"
            )

    # Cascade speed
    cas_cur = cascade.get("current", {})
    cascade_info = (
        f"Cascade Speed: {cas_cur.get('cascade_speed', '?')} "
        f"({cas_cur.get('severity', '?')}), "
        f"{cas_cur.get('n_transitions', 0)} Transitions in 6 Mo"
    )
    transitioned = [t.get("cycle", "?") for t in cas_cur.get("transitioned_cycles", [])]
    if transitioned:
        cascade_info += f"\n  Gekippt: {', '.join(transitioned)}"

    # Confirmation
    conf_info = (
        f"Confirmation Score: {confirmation.get('confirmation_score', '?')} "
        f"({confirmation.get('bullish_count', 0)} bullish, "
        f"{confirmation.get('bearish_count', 0)} bearish, "
        f"{confirmation.get('neutral_count', 0)} neutral)"
    )

    # V16 Transition - find current dual cluster entry
    v16_info = "V16 Transition: Keine Daten"
    by_dual = v16_trans.get("by_dual_cluster", {})
    if by_dual:
        # Find the entry with the most data or the current state
        for key, val in by_dual.items():
            if isinstance(val, dict) and val.get("n_months", 0) > 0:
                growth_6m = val.get("v16_stays_growth_6m", "?")
                stress_6m = val.get("v16_to_stress_6m", "?")
                crisis_6m = val.get("v16_to_crisis_6m", "?")
                n = val.get("n_months", 0)
                if isinstance(growth_6m, (int, float)):
                    growth_pct = round(growth_6m * 100, 1)
                    stress_pct = round(stress_6m * 100, 1) if isinstance(stress_6m, (int, float)) else "?"
                    crisis_pct = round(crisis_6m * 100, 1) if isinstance(crisis_6m, (int, float)) else "?"
                    v16_info = (
                        f"V16 Transition (aktueller Dual-State {key}, n={n}): "
                        f"Growth {growth_pct}%, Stress {stress_pct}%, Crisis {crisis_pct}%"
                    )
                    break

    # Crash vs Correction
    crash_info = "Crash vs Korrektur: Keine Daten"
    dual_dd = crash_corr.get("dual_state_drawdowns", {})
    if dual_dd:
        for key, val in dual_dd.items():
            if isinstance(val, dict):
                crash_info = (
                    f"Crash/Korrektur ({key}): "
                    f"Avg Return {val.get('avg_return', '?')}, "
                    f"Worst {val.get('worst', '?')}, "
                    f"P10 {val.get('p10', '?')}, n={val.get('n', '?')}"
                )
                break

    # Analogues
    analogue_lines = []
    for a in (analogues.get("analogues", []) or [])[:3]:
        spy = a.get("what_happened_next", {}).get("spy_6m_return")
        gld = a.get("what_happened_next", {}).get("gld_6m_return")
        tlt = a.get("what_happened_next", {}).get("tlt_6m_return")
        spy_s = f"{spy*100:+.1f}%" if isinstance(spy, (int, float)) else "?"
        gld_s = f"{gld*100:+.1f}%" if isinstance(gld, (int, float)) else "?"
        tlt_s = f"{tlt*100:+.1f}%" if isinstance(tlt, (int, float)) else "?"
        analogue_lines.append(
            f"  {a.get('period_start', '?')} (Sim={a.get('similarity_score', '?')}): "
            f"SPY {spy_s}, GLD {gld_s}, TLT {tlt_s}"
        )

    # Extended cycles
    extended = assessment.get("extended_cycles", [])
    extended_info = f"Extended Cycles: {', '.join(extended)}" if extended else "Keine Extended Cycles"

    prompt = f"""Hier sind die aktuellen Ergebnisse der Baldur Creek Capital Zyklen-Engine (Lead-Engine V1.1).
Erstelle daraus die wöchentliche Zyklen-Analyse.

OVERALL ASSESSMENT:
  Verdict: {assessment.get('verdict', 'N/A')}
  Signifikante Returns: {assessment.get('n_significant_returns', '?')} / {assessment.get('n_total_returns', '?')} ({assessment.get('pct_significant', '?')}%)
  {extended_info}

PHASE POSITIONS:
{chr(10).join(phase_lines)}

CASCADE & CONFIRMATION:
  {cascade_info}
  {conf_info}

V16 TRANSITION:
  {v16_info}

CRASH VS KORREKTUR:
  {crash_info}

HISTORISCHE ANALOGIEN:
{chr(10).join(analogue_lines) if analogue_lines else '  Keine Analogien verfügbar'}

Datum: {date.today().isoformat()}
V16 aktueller State: LATE_EXPANSION"""

    return prompt


def generate_cycle_narrative(lead_results):
    """Step 3d: Generate LLM cycle narrative using Anthropic SDK.

    Writes cycle_narrative field into transition_engine.json.
    Non-fatal — if the LLM call fails, the engine continues without narrative.
    """
    logger.info("STEP 3d: LLM Cycle Narrative")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("  No ANTHROPIC_API_KEY — skipping narrative generation")
        return None

    if not lead_results:
        logger.warning("  No lead_results — skipping narrative generation")
        return None

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
    except ImportError:
        logger.error("  anthropic SDK not installed — skipping narrative")
        return None
    except Exception as e:
        logger.error(f"  Anthropic client init failed: {e}")
        return None

    user_prompt = _build_narrative_prompt(lead_results)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=CYCLE_NARRATIVE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        narrative_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                narrative_text += block.text

        narrative_text = narrative_text.strip()

        if not narrative_text:
            logger.warning("  LLM returned empty narrative")
            return None

        logger.info(f"  Narrative generated: {len(narrative_text)} chars, "
                    f"{len(narrative_text.split())} words")

        # Write narrative into transition_engine.json
        te_path = os.path.join(DATA_DIR, "transition_engine.json")
        if os.path.exists(te_path):
            try:
                with open(te_path, "r", encoding="utf-8") as f:
                    te_data = json.load(f)

                te_data["cycle_narrative"] = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": "claude-sonnet-4-6",
                    "text": narrative_text,
                    "word_count": len(narrative_text.split()),
                }

                with open(te_path, "w", encoding="utf-8") as f:
                    json.dump(te_data, f, indent=1, ensure_ascii=False, default=str)
                logger.info(f"  Narrative written to transition_engine.json")
            except Exception as e:
                logger.error(f"  Failed to update transition_engine.json: {e}")
        else:
            logger.warning(f"  transition_engine.json not found at {te_path}")

        return narrative_text

    except Exception as e:
        logger.error(f"  LLM narrative generation failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# Git commit + push
# ---------------------------------------------------------------------------

def git_commit_data():
    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        latest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "data", "dashboard", "latest.json")

        subprocess.run(["git", "add", data_path], check=True, capture_output=True)
        if os.path.exists(latest_path):
            subprocess.run(["git", "add", latest_path], check=True, capture_output=True)

        r = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if r.returncode != 0:
            msg = f"Cycles Phase Detection — {date.today().isoformat()}"
            subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
            subprocess.run(["git", "push"], check=True, capture_output=True, timeout=30)
            logger.info("Git commit + push OK")
        else:
            logger.info("No data changes to commit")
    except subprocess.TimeoutExpired:
        logger.warning("Git push timed out")
    except Exception as e:
        logger.warning(f"Git commit failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, dec=2):
    if val is None:
        return ""
    try:
        return round(float(val), dec)
    except (ValueError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cycles Circle — Phase Detection")
    parser.add_argument("--force-backfill", action="store_true")
    parser.add_argument("--skip-sheet", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BALDUR CREEK CAPITAL — CYCLES CIRCLE")
    logger.info(f"Date: {date.today().isoformat()}")
    logger.info(f"Backfill: {args.force_backfill}")
    logger.info("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)

    # Step 1: Collect
    logger.info("STEP 1: Data Collection")
    data = collect_all_data(force_backfill=args.force_backfill)
    if data is None:
        logger.error("Collection failed — aborting")
        sys.exit(1)

    # Step 2: Detect
    logger.info("STEP 2: Phase Detection")
    result = detect_all_phases(data)

    # Step 3: Save locally
    logger.info("STEP 3: Save Locally")
    save_phase_result(result)

    # Step 3b: Generate chart data
    logger.info("STEP 3b: Generate Chart Data (V4.1)")
    generate_chart_data(data, result)

    # Step 3c: Lead-Engine V1.1 — Calibration + Early Warning
    logger.info("STEP 3c: Cycles Kalibrierung + Fruehwarnung (Lead-Engine V1.1)")
    lead_results = None
    try:
        from .lead_engine import run_lead_engine
        lead_results = run_lead_engine()  # reads own inputs, writes own outputs
        if lead_results:
            logger.info(f"  Lead-Engine verdict: {lead_results.get('assessment', {}).get('verdict', 'N/A')}")
        else:
            logger.warning("  Lead-Engine returned None — check input data")
    except Exception as e:
        logger.error(f"  Lead-Engine failed (non-fatal): {e}")

    # Step 3d: LLM Cycle Narrative (V4.3 — NEW)
    try:
        generate_cycle_narrative(lead_results)
    except Exception as e:
        logger.error(f"  Cycle Narrative failed (non-fatal): {e}")

    # Step 4: Write to Sheet
    if not args.skip_sheet:
        logger.info("STEP 4: Write to Cycles Sheet")
        svc = _get_sheets_write()
        if svc:
            write_dashboard(svc, result)
            write_phases(svc, result)
            write_history(svc, result)
        else:
            logger.warning("No Sheets service — skipping")
    else:
        logger.info("STEP 4: Skipped (--skip-sheet)")

    # Step 5: Update latest.json
    logger.info("STEP 5: Update latest.json")
    update_latest_json(result)

    # Step 6: Git
    if not args.skip_git:
        logger.info("STEP 6: Git Commit + Push")
        git_commit_data()
    else:
        logger.info("STEP 6: Skipped (--skip-git)")

    # Summary
    s = result.get("summary", {})
    logger.info("")
    logger.info("=" * 60)
    logger.info("CYCLES CIRCLE COMPLETE")
    logger.info(f"  Regime: {result.get('current_regime')}")
    logger.info(f"  Alignment: {result.get('alignment_score')}/10 ({result.get('alignment_label')})")
    logger.info(f"  Bull: {s.get('bullish')} | Bear: {s.get('bearish')} | "
                f"Neutral: {s.get('neutral')} | Unknown: {s.get('unknown')}")
    logger.info(f"  Danger Zones: {s.get('in_danger_zone')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
