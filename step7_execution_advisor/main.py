"""
step7_execution_advisor/main.py
Execution Advisor — Entry Point (Step 7)

Reads:
  - step2_signal_generator.json from Drive CURRENT/
  - step3_risk_officer.json from Drive CURRENT/
  - step6_cio_final.json from Drive CURRENT/
  - DW Sheet RAW_MARKET (28 fields, L2/L4/L5/L7)
  - EVENT_CALENDAR.yaml (local, from step_0k_event_calendar/)
  - dashboard.json (local, backup for V16 data)

Writes:
  - step7_execution_advisor.json to Drive CURRENT/
  - dashboard.json → execution block (local)
  - archive/DATUM/step7_execution_advisor.json (local)

Schedule: After Step 6 CIO Final (via workflow_run)
Runtime: ~15 seconds (of which ~10s LLM)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("execution_advisor")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DASHBOARD_JSON_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "data", "dashboard", "latest.json"
)

# ---------------------------------------------------------------------------
# Google Infrastructure IDs
# ---------------------------------------------------------------------------
DRIVE_ROOT_ID = "1Tng3i4Cly7isKOxIkGqiTmGiZNEtPj3D"
CURRENT_FOLDER_ID = "1JelM_zZgPeX8TluTfaNqQmsTm3tXkG_8"

OUTPUT_FILENAME = "step7_execution_advisor.json"


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load and merge all config JSON files."""
    config = {}
    config_files = {
        "scoring_thresholds": "scoring_thresholds.json",
        "asset_sensitivity": "asset_sensitivity.json",
        "execution_config": "execution_config.json",
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

    # Flatten execution_config into top-level for easy access
    exec_cfg = config.get("execution_config", {})
    config["llm"] = exec_cfg.get("llm", {})
    config["event_window"] = exec_cfg.get("event_window", {})
    config["dashboard"] = exec_cfg.get("dashboard", {})

    return config


# ---------------------------------------------------------------------------
# Google Drive Helpers
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


def read_drive_json(service, filename: str,
                    folder_id: str = CURRENT_FOLDER_ID):
    """Read a JSON file from Drive CURRENT/."""
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
    """Upload or update a JSON file in Drive CURRENT/."""
    from googleapiclient.http import MediaInMemoryUpload

    content = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")
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
    """Load all Execution Advisor inputs."""
    inputs = {}

    # 1. Signal Generator (V16 Trades, Router)
    if drive_service:
        inputs["signal_generator"] = read_drive_json(
            drive_service, "step2_signal_generator.json"
        )
    else:
        inputs["signal_generator"] = None

    # 2. Risk Officer (Risk Alerts, Ampel, Fragility)
    if drive_service:
        inputs["risk_officer"] = read_drive_json(
            drive_service, "step3_risk_officer.json"
        )
    else:
        inputs["risk_officer"] = None

    # 3. CIO Final (Conviction, Briefing Type)
    if drive_service:
        inputs["cio_final"] = read_drive_json(
            drive_service, "step6_cio_final.json"
        )
    else:
        inputs["cio_final"] = None

    # 4. DW Sheet RAW_MARKET
    try:
        from .dw_reader import read_dw_raw_market
        inputs["dw_data"] = read_dw_raw_market()
        inputs["dw_degraded"] = not bool(inputs["dw_data"])
    except Exception as e:
        logger.warning(f"DW Sheet read failed: {e}")
        inputs["dw_data"] = {}
        inputs["dw_degraded"] = True

    # 5. Event Calendar (YAML, local)
    from .event_reader import load_event_calendar
    inputs["events"] = load_event_calendar()

    # 6. Dashboard (backup for V16 weights)
    inputs["dashboard"] = _read_dashboard_json()

    return inputs


def _read_dashboard_json() -> dict:
    """Read dashboard.json as backup for V16 data."""
    try:
        if os.path.exists(DASHBOARD_JSON_PATH):
            with open(DASHBOARD_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"dashboard.json read failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Execution Advisor — Step 7"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="No Drive writes, no dashboard update"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EXECUTION ADVISOR (Step 7) — Starting")
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
        if key == "dw_degraded":
            continue
        if data is None:
            logger.info(f"  {key}: MISSING")
        elif isinstance(data, dict) and not data:
            logger.info(f"  {key}: EMPTY")
        elif isinstance(data, list):
            logger.info(f"  {key}: {len(data)} items")
        else:
            logger.info(f"  {key}: LOADED")

    # Check minimum inputs — Signal Generator is PFLICHT
    if not inputs.get("signal_generator") and not inputs.get("dashboard"):
        logger.error(
            "Neither Signal Generator nor dashboard available — "
            "Execution Advisor cannot run"
        )
        sys.exit(1)

    # Run engine
    from .engine import run_execution_advisor
    output = run_execution_advisor(inputs, config)

    # Write outputs
    if not args.dry_run:
        # Drive write
        if drive_service:
            try:
                write_drive_json(drive_service, output, OUTPUT_FILENAME)
            except Exception as e:
                logger.error(f"Drive write failed: {e}")

        # Dashboard update
        try:
            from .dashboard_update import update_dashboard_json
            update_dashboard_json(output)
        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")

        # Local archive
        _write_local_archive(output, OUTPUT_FILENAME)
    else:
        logger.info("Dry-run — skipping writes")
        # Write to local file for inspection
        local_path = os.path.join(
            os.path.dirname(BASE_DIR), "data", "execution_advisor"
        )
        os.makedirs(local_path, exist_ok=True)
        out_path = os.path.join(local_path, OUTPUT_FILENAME)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Dry-run output: {out_path}")

    # Summary
    elapsed = round(time.time() - start_time, 1)
    assessment = output.get("execution_assessment", {})
    logger.info("")
    logger.info(f"Execution Advisor complete in {elapsed}s")
    logger.info(f"  Level: {assessment.get('execution_level')} "
                f"(Score: {assessment.get('total_score')}/18)")


if __name__ == "__main__":
    main()
