"""
Cycles Circle — Main Orchestrator
Baldur Creek Capital | Step 0v (V3.4 FINAL)

1. Collect all data (incremental persistence)
2. Run phase detection for all 10 cycles
3. Save results locally (cycle_data.json)
4. Write to Cycles Sheet (DASHBOARD, PHASES, HISTORY)
5. Write cycles block to latest.json (for CyclesCard)
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
# Update latest.json
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
                }
                for cid, c in result.get("cycles", {}).items()
            },
        }

        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(latest, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"latest.json updated with cycles block")

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
