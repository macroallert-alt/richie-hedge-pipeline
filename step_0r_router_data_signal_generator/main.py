"""
step_0r_router_data_signal_generator/main.py
Router Raw Data Collector — Entry Point

Fetches all data the Conviction Router needs:
  - yfinance: VWO, FXI, SPY, DBC, GLD, DXY, USDCNH (CNH=X)
  - FRED: BAMLEMCBPIOAS (EM Corporate Bond OAS)
  - dashboard.json: V16 Credit Impulse (for z-score)

Computes: MAs, returns, deltas, z-scores, relative performance.

Writes:
  - step0r_router_data.json to Drive CURRENT/
  - archive/DATUM/step0r_router_data.json (local, committed by GH Actions)
  - Updates local Parquet cache (data/cache/router/)

Schedule: 05:10 UTC Mo-Fr (before Signal Generator)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("router_data")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# ---------------------------------------------------------------------------
# Google Infrastructure IDs (consistent with all other pipeline steps)
# ---------------------------------------------------------------------------
DRIVE_ROOT_ID = "1Tng3i4Cly7isKOxIkGqiTmGiZNEtPj3D"
CURRENT_FOLDER_ID = "1JelM_zZgPeX8TluTfaNqQmsTm3tXkG_8"

OUTPUT_FILENAME = "step0r_router_data.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    path = os.path.join(CONFIG_DIR, "router_assets.json")
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Google Drive Helpers (same pattern as all pipeline steps)
# ---------------------------------------------------------------------------
def _get_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
    if not creds_json:
        return None
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


def write_drive_json(service, data: dict, filename: str,
                     folder_id: str = CURRENT_FOLDER_ID) -> None:
    from googleapiclient.http import MediaInMemoryUpload

    content = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    media = MediaInMemoryUpload(content, mimetype="application/json")

    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        service.files().update(fileId=files[0]["id"], media_body=media).execute()
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        service.files().create(body=metadata, media_body=media).execute()

    logger.info(f"Drive: {filename} written to CURRENT/")


# ---------------------------------------------------------------------------
# Local Archive
# ---------------------------------------------------------------------------
def _write_local_archive(data: dict, filename: str) -> None:
    try:
        today_str = date.today().isoformat()
        archive_dir = os.path.join(os.path.dirname(BASE_DIR), "archive", today_str)
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, filename)
        with open(archive_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Local archive: archive/{today_str}/{filename}")
    except Exception as e:
        logger.warning(f"Local archive write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Router Data Collector — Step 0r")
    parser.add_argument("--dry-run", action="store_true", help="No Drive writes")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ROUTER DATA COLLECTOR (Step 0r)")
    logger.info(f"dry_run={args.dry_run}")
    logger.info("=" * 60)

    start_time = time.time()
    today = date.today()

    # Load config
    config = load_config()

    # Initialize cache
    from step_0r_router_data_signal_generator.cache import RouterHistoryCache
    cache = RouterHistoryCache()
    cache.load()

    # ================================================================
    # PHASE 1: FETCH
    # ================================================================
    logger.info("")
    logger.info("PHASE 1: FETCH")
    logger.info("-" * 40)

    from step_0r_router_data_signal_generator.fetcher import RouterFetcher
    fetcher = RouterFetcher(config)

    # 1a: yfinance prices
    price_data = fetcher.fetch_yfinance_prices()

    # 1b: FRED BAMLEM
    bamlem_result = fetcher.fetch_fred_bamlem()

    # 1c: Credit Impulse from V16 Sheet (DATA_K16_K17, column E)
    credit_impulse = fetcher.read_credit_impulse_from_v16_sheet()

    # ================================================================
    # PHASE 2: UPDATE CACHE
    # ================================================================
    logger.info("")
    logger.info("PHASE 2: UPDATE CACHE")
    logger.info("-" * 40)

    # Update cache with yfinance data
    for asset_key, series in price_data.items():
        cache.update_from_df(asset_key, series)
        logger.info(f"  Cache {asset_key}: {cache.days_available(asset_key)} days")

    # Update cache with BAMLEM
    if bamlem_result is not None:
        bamlem_series, bamlem_val, bamlem_date = bamlem_result
        cache.update_from_df("BAMLEM", bamlem_series)
        logger.info(f"  Cache BAMLEM: {cache.days_available('BAMLEM')} days")

    # Update cache with Credit Impulse
    if credit_impulse is not None:
        cache.update("credit_impulse", credit_impulse, datetime.combine(today, datetime.min.time()))
        logger.info(f"  Cache credit_impulse: {cache.days_available('credit_impulse')} days")

    # ================================================================
    # PHASE 3: CALCULATE
    # ================================================================
    logger.info("")
    logger.info("PHASE 3: CALCULATE")
    logger.info("-" * 40)

    from step_0r_router_data_signal_generator.calculator import build_router_raw_data

    # Get BAMLEM series from cache (includes today's fetch)
    bamlem_cached = cache.get("BAMLEM")

    router_data = build_router_raw_data(
        cache_data={key: cache.get(key) for key in cache.data.keys()},
        bamlem_series=bamlem_cached,
        credit_impulse=credit_impulse,
        config=config,
    )

    # Add metadata
    router_data["date"] = today.isoformat()
    router_data["run_timestamp"] = datetime.utcnow().isoformat() + "Z"

    # Log summary
    dq = router_data["data_quality"]
    logger.info(f"  Assets: {dq['assets_ok']}/{dq['assets_fetched']} OK")
    logger.info(f"  FRED: {'OK' if dq['fred_ok'] else 'FAILED'}")
    logger.info(f"  Credit Impulse: {'OK' if dq['credit_impulse_ok'] else 'MISSING'}")
    if dq["stale_fields"]:
        logger.warning(f"  Stale: {dq['stale_fields']}")

    # Log key values
    logger.info("")
    logger.info("KEY VALUES:")
    _log_val("  DXY", router_data["dxy"]["value"], f"d126={router_data['dxy']['delta_126d']}")
    _log_val("  VWO", router_data["vwo"]["price"],
             f"ma50={router_data['vwo']['ma_50d']} ma200={router_data['vwo']['ma_200d']} "
             f"r126={router_data['vwo']['return_126d']}")
    _log_val("  FXI", router_data["fxi"]["price"],
             f"ma50={router_data['fxi']['ma_50d']} r63={router_data['fxi']['return_63d']}")
    _log_val("  BAMLEM", router_data["bamlem"]["value"], f"d63={router_data['bamlem']['delta_63d']}")
    _log_val("  USDCNH", router_data["usdcnh"]["value"], f"d63={router_data['usdcnh']['delta_63d']}")
    _log_val("  DBC", router_data["dbc"]["price"], f"r126={router_data['dbc']['return_126d']}")
    logger.info(f"  Relative VWO-SPY 126d: {router_data['relative']['vwo_spy_126d']}")
    logger.info(f"  Relative FXI-SPY 63d:  {router_data['relative']['fxi_spy_63d']}")

    # ================================================================
    # PHASE 4: WRITE
    # ================================================================
    logger.info("")
    logger.info("PHASE 4: WRITE")
    logger.info("-" * 40)

    if not args.dry_run:
        # Drive write
        try:
            drive_service = _get_drive_service()
            if drive_service:
                write_drive_json(drive_service, router_data, OUTPUT_FILENAME)
            else:
                logger.warning("No Drive service — skipping Drive write")
        except Exception as e:
            logger.error(f"Drive write failed: {e}")

        # Local archive
        _write_local_archive(router_data, OUTPUT_FILENAME)
    else:
        logger.info("Dry-run — skipping Drive write")
        # Write to local file for inspection
        local_path = os.path.join(os.path.dirname(BASE_DIR), "data", "router")
        os.makedirs(local_path, exist_ok=True)
        out_path = os.path.join(local_path, OUTPUT_FILENAME)
        with open(out_path, "w") as f:
            json.dump(router_data, f, indent=2, default=str)
        logger.info(f"Dry-run output: {out_path}")

    # Save cache
    cache.prune()
    cache.save()

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = round(time.time() - start_time, 1)
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"ROUTER DATA COLLECTOR COMPLETE in {elapsed}s")
    logger.info(f"  Date: {today}")
    logger.info(f"  Assets: {dq['assets_ok']}/{dq['assets_fetched']}, "
                f"FRED: {'OK' if dq['fred_ok'] else 'FAIL'}, "
                f"CI: {'OK' if dq['credit_impulse_ok'] else 'MISS'}")
    logger.info("=" * 60)


def _log_val(label: str, value, extra: str = ""):
    if value is not None:
        logger.info(f"{label}: {value} {extra}")
    else:
        logger.warning(f"{label}: MISSING {extra}")


if __name__ == "__main__":
    main()
