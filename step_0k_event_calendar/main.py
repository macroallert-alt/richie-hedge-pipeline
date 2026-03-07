"""
step_0k_event_calendar/main.py
Event Calendar Updater — Entry Point

Workflow:
  1. Load existing EVENT_CALENDAR.yaml
  2. Remove past events (> 6 months)
  3. Delete future events (will be regenerated)
  4. LLM + Web Search → new events
  5. OpEx/Quad Witching rule-based calculation
  6. Merge (keep past + new future)
  7. Sort, deduplicate
  8. Write EVENT_CALENDAR.yaml
  9. Write local archive

Git commit + push is handled by GH Actions workflow, not Python.

Schedule: 1st of each month, 06:00 UTC (via GH Actions)
Source: Trading Desk Spec Teil 1 §2.2, Teil 2 §6.1
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime

import yaml

from .updater import run_calendar_updater, merge_events
from .opex_calculator import calculate_opex_dates

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("event_calendar")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, "EVENT_CALENDAR.yaml")


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
def load_existing_yaml() -> dict:
    """Load existing EVENT_CALENDAR.yaml or return empty structure."""
    if not os.path.exists(YAML_PATH):
        logger.warning(f"No existing YAML at {YAML_PATH} — starting fresh")
        return {"meta": {}, "events": []}

    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Handle old format (no meta block, just events list)
        if "events" in data and "meta" not in data:
            logger.info("Old YAML format detected — migrating")
            data = {"meta": {}, "events": data["events"]}

        events = data.get("events", [])
        logger.info(f"Loaded existing YAML: {len(events)} events")
        return data

    except Exception as e:
        logger.error(f"Failed to load YAML: {e}")
        return {"meta": {}, "events": []}


def write_yaml(data: dict) -> None:
    """Write EVENT_CALENDAR.yaml with proper formatting."""
    try:
        with open(YAML_PATH, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        logger.info(f"YAML written: {YAML_PATH} ({data['meta'].get('events_total', '?')} events)")
    except Exception as e:
        logger.error(f"YAML write FAILED: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Local Archive
# ---------------------------------------------------------------------------
def write_local_archive(data: dict) -> None:
    """Write archive copy of the calendar."""
    try:
        today_str = date.today().isoformat()
        archive_dir = os.path.join(os.path.dirname(BASE_DIR), "archive", today_str)
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, "EVENT_CALENDAR.yaml")
        with open(archive_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data, f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        logger.info(f"Archive: archive/{today_str}/EVENT_CALENDAR.yaml")
    except Exception as e:
        logger.warning(f"Archive write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Event Calendar Updater — Step 0k"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without writing YAML (print to stdout instead)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EVENT CALENDAR UPDATER (Step 0k) — Starting")
    logger.info(f"dry_run={args.dry_run}")
    logger.info("=" * 60)

    start_time = time.time()
    today = date.today()

    # ----------------------------------------------------------------
    # 1. Load existing YAML
    # ----------------------------------------------------------------
    logger.info("Loading existing EVENT_CALENDAR.yaml...")
    existing_yaml = load_existing_yaml()

    # ----------------------------------------------------------------
    # 2+3+4. LLM + Web Search → new events
    # ----------------------------------------------------------------
    logger.info("Running LLM + Web Search for new events...")
    try:
        llm_events = run_calendar_updater(today)
        logger.info(f"LLM returned {len(llm_events)} events")
    except RuntimeError as e:
        logger.error(f"LLM update failed: {e}")
        logger.error("Keeping existing YAML unchanged.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 5. OpEx/Quad Witching (rule-based)
    # ----------------------------------------------------------------
    logger.info("Calculating OpEx/Quad Witching dates...")
    opex_events = calculate_opex_dates(today, months_ahead=6)
    logger.info(f"OpEx calculator: {len(opex_events)} dates")

    # ----------------------------------------------------------------
    # 6+7. Merge, sort, dedup
    # ----------------------------------------------------------------
    logger.info("Merging events...")
    merged = merge_events(existing_yaml, llm_events, opex_events, today)

    logger.info(
        f"Merge result: {merged['meta']['events_total']} total "
        f"({merged['meta']['past_events_kept']} kept past, "
        f"{merged['meta']['future_events_new']} new future)"
    )

    # ----------------------------------------------------------------
    # 8. Write YAML
    # ----------------------------------------------------------------
    if args.dry_run:
        logger.info("Dry-run — printing to stdout:")
        print(yaml.dump(
            merged,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ))
    else:
        write_yaml(merged)
        write_local_archive(merged)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = round(time.time() - start_time, 1)
    types_found = set(e["type"] for e in merged["events"])

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"EVENT CALENDAR UPDATER COMPLETE in {elapsed}s")
    logger.info(f"  Events: {merged['meta']['events_total']}")
    logger.info(f"  Types: {len(types_found)} ({', '.join(sorted(types_found))})")
    logger.info(f"  Next update: {merged['meta']['next_update']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
