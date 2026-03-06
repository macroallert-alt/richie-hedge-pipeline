"""
step5_devils_advocate/main.py
Devil's Advocate — Entry Point
Usage: python -m step5_devils_advocate

Reads:
  - step4_cio_draft.json from Drive CURRENT/
  - step1_market_analyst.json from Drive CURRENT/
  - step0b_ic_intelligence.json from Drive CURRENT/
  - step3_risk_officer.json from Drive CURRENT/
  - da_history.json from Drive HISTORY/
  - yesterday's CIO Final from archive/

Writes:
  - step5_devils_advocate.json to Drive CURRENT/
  - da_history.json to Drive HISTORY/ (update)
  - AGENT_SUMMARY tab in DW Sheet
  - archive/DATUM/step5_devils_advocate.json (local, committed by GH Actions)
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta

import yaml

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("da_main")

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
DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    path = os.path.join(CONFIG_DIR, "DA_CONFIG.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Google API Services
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


def _get_sheets_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
    if not creds_json:
        return None
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)


# ---------------------------------------------------------------------------
# Drive Read/Write Helpers
# ---------------------------------------------------------------------------
def read_drive_json(service, filename: str, folder_id: str = CURRENT_FOLDER_ID) -> dict | None:
    """Read a JSON file from a Drive folder."""
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id,name)").execute()
        files = results.get("files", [])
        if not files:
            logger.warning(f"Drive: {filename} not found in folder {folder_id}")
            return None
        content = service.files().get_media(fileId=files[0]["id"]).execute()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Drive read failed for {filename}: {e}")
        return None


def write_drive_json(service, data: dict, filename: str,
                     folder_id: str = CURRENT_FOLDER_ID) -> None:
    """Upload or update a JSON file in a Drive folder."""
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

    logger.info(f"Drive: {filename} written")


def _find_folder(service, name: str, parent_id: str) -> str | None:
    """Find a folder in Drive."""
    try:
        query = (
            f"name='{name}' and '{parent_id}' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Local Archive Helpers
# ---------------------------------------------------------------------------
def _write_local_archive(data: dict, filename: str) -> None:
    """Write JSON to archive/YYYY-MM-DD/ for Git-based archiving."""
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


def _read_yesterday_archive() -> dict | None:
    """Read yesterday's CIO Final from Git archive."""
    try:
        for days_back in range(1, 5):
            check_date = (date.today() - timedelta(days=days_back)).isoformat()
            archive_path = os.path.join(
                os.path.dirname(BASE_DIR), "archive", check_date, "step6_cio_final.json"
            )
            if os.path.exists(archive_path):
                with open(archive_path, "r") as f:
                    data = json.load(f)
                logger.info(f"  yesterday_final: Found archive/{check_date}/step6_cio_final.json")
                return data
        return None
    except Exception as e:
        logger.warning(f"Yesterday archive read failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Load All Inputs
# ---------------------------------------------------------------------------
def load_all_inputs(drive_service) -> dict:
    """Load all DA inputs from Drive CURRENT/ and archive."""
    inputs = {}

    if not drive_service:
        logger.error("No Drive service — DA cannot load inputs")
        return inputs

    # PFLICHT: CIO Draft
    draft = read_drive_json(drive_service, "step4_cio_draft.json")
    inputs["draft_memo"] = draft

    # DEGRADED inputs
    inputs["risk_alerts"] = read_drive_json(drive_service, "step3_risk_officer.json")
    inputs["layer_analysis"] = read_drive_json(drive_service, "step1_market_analyst.json")
    inputs["ic_intelligence"] = read_drive_json(drive_service, "step0b_ic_intelligence.json")
    inputs["signals"] = read_drive_json(drive_service, "step2_signal_generator.json")

    # V16 Production (from dashboard.json)
    inputs["v16_production"] = _read_v16_from_dashboard()

    # F6 Production (UNAVAILABLE)
    inputs["f6_production"] = {"status": "UNAVAILABLE", "active_positions": [], "signals_today": []}

    # DA History (from Drive HISTORY/)
    history_folder_id = _find_folder(drive_service, "HISTORY", DRIVE_ROOT_ID)
    if history_folder_id:
        da_hist = read_drive_json(drive_service, "da_history.json", history_folder_id)
        inputs["da_history"] = da_hist
    else:
        inputs["da_history"] = None

    # Yesterday's CIO Final (from Git archive)
    inputs["yesterday_final"] = _read_yesterday_archive()

    return inputs


def _read_v16_from_dashboard() -> dict:
    """Read V16 production data from dashboard.json."""
    try:
        if os.path.exists(DASHBOARD_JSON_PATH):
            with open(DASHBOARD_JSON_PATH, "r") as f:
                dashboard = json.load(f)
            v16 = dashboard.get("v16", {})
            if v16.get("status") == "AVAILABLE":
                return v16
        return {}
    except Exception as e:
        logger.warning(f"V16 read from dashboard failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Devil's Advocate — Step 5")
    parser.add_argument("--dry-run", action="store_true", help="No writes")
    args = parser.parse_args()

    logger.info(f"Devil's Advocate starting — dry_run={args.dry_run}")
    start_time = datetime.utcnow()

    # Load config
    config = load_config()

    # Initialize Google services
    try:
        drive_service = _get_drive_service()
        sheets_service = _get_sheets_service()
    except ImportError:
        logger.warning("Google API libs not installed — running without Drive/Sheets")
        drive_service = None
        sheets_service = None

    # Load all inputs
    logger.info("Loading inputs...")
    inputs = load_all_inputs(drive_service)

    # Log input status
    for key, data in inputs.items():
        if data is None:
            logger.info(f"  {key}: MISSING (None)")
        elif isinstance(data, dict) and data.get("status") == "UNAVAILABLE":
            logger.info(f"  {key}: UNAVAILABLE")
        elif isinstance(data, dict) and not data:
            logger.info(f"  {key}: EMPTY")
        else:
            logger.info(f"  {key}: LOADED")

    # Check mandatory input
    if inputs.get("draft_memo") is None:
        logger.error("DRAFT_MEMO not found — DA cannot run")
        logger.error("CIO Draft must be in Drive CURRENT/step4_cio_draft.json")
        sys.exit(1)

    # Run DA engine
    from step5_devils_advocate.engine import run_devils_advocate
    result = run_devils_advocate(inputs, config)

    if not result["success"]:
        logger.warning(f"DA failed: {result.get('reason')}")
        logger.warning(f"Action: {result.get('action')}")
        # Not a hard failure — CIO Final will handle Draft-as-Final
        sys.exit(0)

    da_output = result["da_output"]
    da_history = result["da_history"]

    # Write outputs
    if not args.dry_run:
        # 1. Write DA output to Drive CURRENT/
        if drive_service:
            try:
                write_drive_json(drive_service, da_output, "step5_devils_advocate.json")
            except Exception as e:
                logger.error(f"DA Drive write failed: {e}")

        # 2. Write DA History to Drive HISTORY/
        if drive_service:
            try:
                history_folder_id = _find_folder(drive_service, "HISTORY", DRIVE_ROOT_ID)
                if history_folder_id:
                    write_drive_json(drive_service, da_history, "da_history.json", history_folder_id)
                    logger.info("Drive: da_history.json written to HISTORY/")
                else:
                    logger.warning("HISTORY folder not found — skipping DA history write")
            except Exception as e:
                logger.error(f"DA History Drive write failed: {e}")

        # 3. Write AGENT_SUMMARY
        if sheets_service:
            try:
                from step5_devils_advocate.postprocessor import write_agent_summary
                write_agent_summary(sheets_service, da_output)
            except Exception as e:
                logger.error(f"AGENT_SUMMARY write failed: {e}")

        # 4. Archive locally (committed by GitHub Actions)
        _write_local_archive(da_output, "step5_devils_advocate.json")

    else:
        logger.info("Dry-run — skipping all writes")
        print(json.dumps(da_output, indent=2, default=str))

    # Log summary
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    meta = da_output.get("metadata", {})
    logger.info(f"Devil's Advocate complete in {elapsed:.1f}s")
    logger.info(f"  Challenges: {meta.get('total_challenges', 0)}, "
                f"Focus: {da_output.get('primary_focus', '?')}, "
                f"Seed: {da_output.get('perspective_seed', '?')}")


if __name__ == "__main__":
    main()
