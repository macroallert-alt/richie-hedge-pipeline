#!/usr/bin/env python3
"""
main.py — Data Collector V16 Orchestrator
==========================================
4-Phasen-Pipeline:
  Phase 1: FETCH    — Daten holen (fetchers.py)
  Phase 2: CALC     — Transformieren (transforms.py)
  Phase 3: QUALITY  — Pruefen (quality.py)
  Phase 4: WRITE    — Schreiben (writers.py)

Taeglich via GitHub Actions ~05:30 UTC.
Timing-Budget: 3 Minuten total.
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, date

# ═══════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════

def setup_logging(log_dir: str = "data/logs"):
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f"dc_{today}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger("data_collector")


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

V16_SHEET_ID = os.environ.get(
    'V16_SHEET_ID', '11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE'
)
CONFIG_DIR = os.environ.get('CONFIG_DIR', 'config')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'data')
CACHE_DIR = os.environ.get('CACHE_DIR', 'data/cache')
MAX_RUNTIME_SECONDS = 180  # 3 Minuten Budget


# ═══════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════

def run_pipeline():
    logger = setup_logging()
    start_time = time.time()
    today = datetime.now().date()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  DATA COLLECTOR V16 — Daily Pipeline     ║")
    logger.info(f"║  Date: {today}                        ║")
    logger.info("╚══════════════════════════════════════════╝")

    # ─── Load Registry ───
    registry_path = os.path.join(CONFIG_DIR, 'field_registry.json')
    if not os.path.exists(registry_path):
        logger.error(f"field_registry.json not found at {registry_path}")
        sys.exit(1)

    with open(registry_path) as f:
        registry_data = json.load(f)
    registry = {f['name']: f for f in registry_data['fields']}
    logger.info(f"Registry loaded: {len(registry)} fields")

    # ─── Load Cache ───
    from cache import HistoryCache, FetchCache
    history_cache = HistoryCache(CACHE_DIR)
    history_data = history_cache.load()
    fetch_cache = FetchCache(CACHE_DIR)
    fetch_cache_data = fetch_cache.load()

    # ═════════════════════════════════════════════
    # PHASE 1: FETCH
    # ═════════════════════════════════════════════
    t1 = time.time()
    from fetchers import MasterFetcher
    fetcher = MasterFetcher(config_dir=CONFIG_DIR)
    fetch_results = fetcher.fetch_all(cache=fetch_cache_data)

    # Extract V16-specific data
    v16_prices = None
    v16_liquidity = None
    if '_v16_prices' in fetch_results and fetch_results['_v16_prices'].success:
        v16_prices = fetch_results['_v16_prices'].value
    if '_v16_liquidity' in fetch_results and fetch_results['_v16_liquidity'].success:
        v16_liquidity = fetch_results['_v16_liquidity'].value

    elapsed_p1 = time.time() - t1
    logger.info(f"Phase 1 took {elapsed_p1:.1f}s")

    # Budget check
    if time.time() - start_time > MAX_RUNTIME_SECONDS * 0.8:
        logger.warning("Time budget 80% consumed after Phase 1!")

    # ═════════════════════════════════════════════
    # PHASE 2: TRANSFORM
    # ═════════════════════════════════════════════
    t2 = time.time()
    from transforms import TransformEngine
    holidays_path = os.path.join(CONFIG_DIR, 'us_holidays.json')
    engine = TransformEngine(registry, history_data, holidays_path)
    transformed = engine.transform_all(fetch_results, today)
    elapsed_p2 = time.time() - t2
    logger.info(f"Phase 2 took {elapsed_p2:.1f}s")

    # ═════════════════════════════════════════════
    # PHASE 3: QUALITY
    # ═════════════════════════════════════════════
    t3 = time.time()
    from quality import QualityEngine
    quality = QualityEngine(registry, history_data)
    dq_summary = quality.run_all(transformed, fetch_results)
    elapsed_p3 = time.time() - t3
    logger.info(f"Phase 3 took {elapsed_p3:.1f}s")

    # ─── HALT CHECK ───
    if dq_summary.get('data_quality_level') == "CRITICAL":
        t1_issues = dq_summary.get('t1_issues', [])
        if len(t1_issues) > 5:
            logger.error(f"CRITICAL: {len(t1_issues)} T1 sources down — HALTING WRITE")
            # Still write DQ summary for debugging
            from writers import JSONWriter
            writer = JSONWriter(OUTPUT_DIR)
            writer.write_dq_summary(dq_summary)
            sys.exit(2)

    # ═════════════════════════════════════════════
    # PHASE 4: WRITE
    # ═════════════════════════════════════════════
    t4 = time.time()
    from writers import V16SheetWriter, JSONWriter

    # 4a: V16 Sheet
    sheet_status = {"DATA_Prices": False, "DATA_Liquidity": False, "DATA_K16_K17": False}
    try:
        sheet_writer = V16SheetWriter(V16_SHEET_ID)
        if v16_prices:
            sheet_status = sheet_writer.write_all(v16_prices, v16_liquidity or {}, transformed, today)
        else:
            logger.warning("No V16 prices — skipping sheet write")
    except Exception as e:
        logger.error(f"V16 Sheet write failed: {e}")

    # 4b: JSON Outputs
    json_writer = JSONWriter(OUTPUT_DIR)
    json_paths = json_writer.write_all(transformed, dq_summary)

    elapsed_p4 = time.time() - t4
    logger.info(f"Phase 4 took {elapsed_p4:.1f}s")

    # ─── UPDATE CACHE ───
    engine.update_history(transformed, today)
    history_cache.data = history_data
    history_cache.prune()
    history_cache.save()
    fetch_cache.update_from_results(fetch_results)
    fetch_cache.save()

    # ═════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════
    total_elapsed = time.time() - start_time

    logger.info("")
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  PIPELINE COMPLETE                       ║")
    logger.info("╠══════════════════════════════════════════╣")
    logger.info(f"║  Total time:   {total_elapsed:6.1f}s                    ║")
    logger.info(f"║  Phase 1 FETCH:  {elapsed_p1:5.1f}s                    ║")
    logger.info(f"║  Phase 2 CALC:   {elapsed_p2:5.1f}s                    ║")
    logger.info(f"║  Phase 3 QUAL:   {elapsed_p3:5.1f}s                    ║")
    logger.info(f"║  Phase 4 WRITE:  {elapsed_p4:5.1f}s                    ║")
    logger.info(f"║  Quality: {dq_summary.get('data_quality_level', '?'):10s}                  ║")
    logger.info(f"║  Fields: {dq_summary.get('fields_ok', 0)}/{dq_summary.get('fields_total', 0)} OK"
                f"   Stale: {dq_summary.get('fields_stale', 0)}"
                f"   Failed: {dq_summary.get('fields_failed', 0)}     ║")
    logger.info(f"║  Sheet: {sum(1 for v in sheet_status.values() if v)}/3 tabs          ║")
    logger.info("╚══════════════════════════════════════════╝")

    # Alerts
    for alert in dq_summary.get('alerts', []):
        logger.warning(f"ALERT: {alert}")

    # Exit code
    if dq_summary.get('data_quality_level') == "CRITICAL":
        sys.exit(2)
    elif dq_summary.get('data_quality_level') == "DEGRADED":
        sys.exit(0)  # OK but with warnings
    else:
        sys.exit(0)


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\nPipeline abgebrochen.")
        sys.exit(130)
    except Exception as e:
        logging.getLogger("data_collector").error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
