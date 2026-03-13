"""
Cycles Circle — Main Orchestrator
Baldur Creek Capital | Step 0v (V4.0 — Chart Data)

1. Collect all data (incremental persistence)
2. Run phase detection for all 10 cycles
3. Save results locally (cycle_data.json)
3b. Generate chart data (cycles_chart_data.json) — NEW
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
# Generate Chart Data (NEW — V4.0)
# ---------------------------------------------------------------------------

# Mapping: cycle_id → how to extract the indicator time series from raw data
# Returns list of {"date": "YYYY-MM-DD", "value": float}
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


def _resample_monthly(series):
    """Resample a daily/weekly/mixed series to monthly (last value per month).
    Input: list of {"date": "YYYY-MM-DD", "value": float}, sorted ascending.
    Output: list of {"date": "YYYY-MM", "value": float}, sorted ascending.
    """
    if not series:
        return []
    monthly = {}
    for pt in series:
        d = pt.get("date", "")
        v = pt.get("value")
        if not d or v is None:
            continue
        ym = d[:7]  # "YYYY-MM"
        monthly[ym] = v  # last value wins (series is sorted ascending)
    result = [{"date": ym, "value": round(v, 6)} for ym, v in sorted(monthly.items())]
    return result


def _compute_ma(monthly_series, window=12):
    """Compute moving average over monthly series.
    Returns list of {"date": "YYYY-MM", "value": float} with same length,
    None for first (window-1) entries.
    """
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


def _extract_indicator_series(cycle_id, data):
    """Extract the primary indicator time series for a cycle from raw data.
    Returns list of {"date": "YYYY-MM-DD", "value": float} sorted ascending.
    """
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
        # Build CPI YoY map (monthly)
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
    """Extract an asset price series from raw data.
    Returns list of {"date": "YYYY-MM-DD", "value": float} sorted ascending.
    """
    raw = data.get("prices", {}).get(ticker, [])
    series = [{"date": p["date"], "value": p["price"]}
              for p in raw if p.get("price") and p.get("date")]
    series.sort(key=lambda x: x["date"])
    return series


def generate_chart_data(data, result):
    """Generate cycles_chart_data.json with monthly time series for frontend charts.

    Per cycle:
      - indicator: monthly resampled primary indicator
      - ma_12m: 12-month moving average
      - asset_overlay: monthly resampled primary asset
      - current_phase: current phase info from result
      - meta: name, tier, unit, asset ticker
    """
    logger.info("Generating chart data...")
    chart_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date.today().isoformat(),
        "cycles": {},
    }

    for cid in CYCLE_ORDER:
        cdef = CYCLE_DEFINITIONS.get(cid, {})
        asset_ticker = CHART_ASSET_OVERLAY.get(cid)
        phase_result = result.get("cycles", {}).get(cid, {})

        # Extract indicator series
        indicator_raw = _extract_indicator_series(cid, data)
        indicator_monthly = _resample_monthly(indicator_raw)

        # Compute 12M MA
        ma_12m = _compute_ma(indicator_monthly, 12) if indicator_monthly else []

        # Extract asset overlay
        asset_raw = _extract_asset_series(asset_ticker, data) if asset_ticker else []
        asset_monthly = _resample_monthly(asset_raw)

        n_indicator = len(indicator_monthly)
        n_asset = len(asset_monthly)

        chart_data["cycles"][cid] = {
            "name": CYCLE_NAMES.get(cid, cid),
            "tier": TIER_MAP.get(cid, 0),
            "asset_ticker": asset_ticker,
            "indicator_count": n_indicator,
            "asset_count": n_asset,
            "current_phase": phase_result.get("phase"),
            "current_value": _fmt(phase_result.get("indicator_value"), 4),
            "indicator": indicator_monthly,
            "ma_12m": ma_12m,
            "asset_overlay": asset_monthly,
        }

        logger.info(f"  {cid}: {n_indicator} indicator pts, {n_asset} asset pts")

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
    logger.info("STEP 3b: Generate Chart Data")
    generate_chart_data(data, result)

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
