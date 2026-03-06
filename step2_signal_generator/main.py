"""
step2_signal_generator/main.py
Signal Generator — Entry Point (Step 2)

Reads:
  - step0r_router_data.json from Drive CURRENT/
  - step1_market_analyst.json from Drive CURRENT/
  - dashboard.json (V16 block, local)
  - step2_signal_generator.json from Drive CURRENT/ (own history)

Writes:
  - step2_signal_generator.json to Drive CURRENT/
  - archive/DATUM/step2_signal_generator.json (local, committed by GH Actions)

Schedule: After Step 1 Market Analyst (via workflow_run), before Step 3 Risk Officer
Runtime: <10 seconds. No LLM. 100% deterministic.
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
logger = logging.getLogger("signal_generator")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DASHBOARD_JSON_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "data", "dashboard", "latest.json"
)

# ---------------------------------------------------------------------------
# Google Infrastructure IDs (consistent with all pipeline steps)
# ---------------------------------------------------------------------------
DRIVE_ROOT_ID = "1Tng3i4Cly7isKOxIkGqiTmGiZNEtPj3D"
CURRENT_FOLDER_ID = "1JelM_zZgPeX8TluTfaNqQmsTm3tXkG_8"

OUTPUT_FILENAME = "step2_signal_generator.json"


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load and merge all config JSON files."""
    config = {}
    config_files = {
        "router_triggers": "router_triggers.json",
        "router_state_machine": "router_state_machine.json",
        "router_timing": "router_timing.json",
        "router_fragility_overlay": "router_fragility_overlay.json",
        "compilation_rules": "compilation_rules.json",
    }

    for key, filename in config_files.items():
        path = os.path.join(CONFIG_DIR, filename)
        try:
            with open(path, "r") as f:
                config[key] = json.load(f)
            logger.info(f"  Config loaded: {filename}")
        except Exception as e:
            logger.error(f"  Config FAILED: {filename} — {e}")
            config[key] = {}

    return config


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


def read_drive_json(service, filename: str, folder_id: str = CURRENT_FOLDER_ID):
    """Read a JSON file from Drive."""
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id,name)").execute()
        files = results.get("files", [])
        if not files:
            logger.warning(f"Drive: {filename} not found in CURRENT/")
            return None
        content = service.files().get_media(fileId=files[0]["id"]).execute()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Drive read failed for {filename}: {e}")
        return None


def write_drive_json(service, data: dict, filename: str,
                     folder_id: str = CURRENT_FOLDER_ID) -> None:
    """Upload or update a JSON file in Drive."""
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
# Input Loading
# ---------------------------------------------------------------------------
def load_inputs(drive_service) -> dict:
    """Load all Signal Generator inputs."""
    inputs = {}

    # 1. Router Raw Data (from Drive CURRENT/)
    if drive_service:
        inputs["router_raw_data"] = read_drive_json(drive_service, "step0r_router_data.json")
    else:
        inputs["router_raw_data"] = None

    # 2. Market Analyst (from Drive CURRENT/)
    if drive_service:
        inputs["market_analyst"] = read_drive_json(drive_service, "step1_market_analyst.json")
    else:
        inputs["market_analyst"] = None

    # 3. V16 Data (from dashboard.json, local)
    inputs["v16_data"] = _read_v16_from_dashboard()

    # 4. Own History (from Drive CURRENT/ — yesterday's output)
    if drive_service:
        inputs["own_history"] = read_drive_json(drive_service, OUTPUT_FILENAME)
    else:
        inputs["own_history"] = None

    return inputs


def _read_v16_from_dashboard() -> dict:
    """Read V16 production data from dashboard.json."""
    try:
        if os.path.exists(DASHBOARD_JSON_PATH):
            with open(DASHBOARD_JSON_PATH, "r", encoding="utf-8") as f:
                dashboard = json.load(f)
            v16 = dashboard.get("v16", {})
            if v16.get("status") == "AVAILABLE":
                logger.info(f"  V16: LOADED from dashboard.json")
                return v16
            else:
                logger.warning(f"  V16: status={v16.get('status', 'MISSING')}")
                return v16
        else:
            logger.warning(f"  dashboard.json not found at {DASHBOARD_JSON_PATH}")
            return {}
    except Exception as e:
        logger.warning(f"  V16 read from dashboard failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Signal Generator — Step 2")
    parser.add_argument("--dry-run", action="store_true", help="No Drive writes")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SIGNAL GENERATOR (Step 2) — Starting")
    logger.info(f"dry_run={args.dry_run}")
    logger.info("=" * 60)

    start_time = time.time()

    # Load config
    logger.info("Loading config...")
    config = load_config()

    # Initialize Drive
    try:
        drive_service = _get_drive_service()
        if drive_service:
            logger.info("  Drive: connected")
        else:
            logger.warning("  Drive: No credentials — running offline")
    except Exception as e:
        logger.warning(f"  Drive init failed: {e}")
        drive_service = None

    # Load inputs
    logger.info("Loading inputs...")
    inputs = load_inputs(drive_service)

    # Log input status
    for key, data in inputs.items():
        if data is None:
            logger.info(f"  {key}: MISSING")
        elif isinstance(data, dict) and not data:
            logger.info(f"  {key}: EMPTY")
        else:
            logger.info(f"  {key}: LOADED")

    # Check minimum inputs
    if not inputs.get("v16_data"):
        logger.error("V16 data not available — Signal Generator cannot run")
        sys.exit(1)

    # Run engine
    from step2_signal_generator.engine import run_signal_generator
    signal_output = run_signal_generator(inputs, config)

    # Write outputs
    if not args.dry_run:
        # Drive write
        if drive_service:
            try:
                write_drive_json(drive_service, signal_output, OUTPUT_FILENAME)
            except Exception as e:
                logger.error(f"Drive write failed: {e}")

        # Local archive
        _write_local_archive(signal_output, OUTPUT_FILENAME)
    else:
        logger.info("Dry-run — skipping writes")
        # Write to local file for inspection
        local_path = os.path.join(os.path.dirname(BASE_DIR), "data", "signal_generator")
        os.makedirs(local_path, exist_ok=True)
        out_path = os.path.join(local_path, OUTPUT_FILENAME)
        with open(out_path, "w") as f:
            json.dump(signal_output, f, indent=2, default=str)
        logger.info(f"Dry-run output: {out_path}")

    # Summary
    elapsed = round(time.time() - start_time, 1)
    logger.info("")
    logger.info(f"Signal Generator complete in {elapsed}s")
    logger.info(f"  Path: {signal_output.get('execution_path')}")
    router = signal_output.get("router", {})
    logger.info(f"  Router: {router.get('current_state')} "
                f"(max prox: {router.get('max_proximity', 0):.2%})")


if __name__ == "__main__":
    main()
