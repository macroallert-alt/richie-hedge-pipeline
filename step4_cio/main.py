"""
step4_cio/main.py
CIO Agent — Entry Point
Usage: python -m step4_cio --mode draft
       python -m step4_cio --mode final

Reads:
  - step1_market_analyst.json from Drive CURRENT/
  - step0b_ic_intelligence.json from Drive CURRENT/
  - step3_risk_officer.json from Drive CURRENT/
  - V16 Production Sheet (regime, weights, dd-protect)
  - cio_history_digest.json from Drive HISTORY/ (if exists)

Writes:
  - step4_cio_draft.json to Drive CURRENT/ (mode=draft)
  - step6_cio_final.json + step6_cio_final_memo.md to Drive CURRENT/ (mode=final)
  - AGENT_SUMMARY tab in DW Sheet
  - briefing block in data/dashboard/latest.json
  - cio_history_digest.json to Drive HISTORY/
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime

import yaml

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cio_main")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Dashboard JSON path (written by V16_DAILY_RUNNER, updated by IC, updated by CIO)
DASHBOARD_JSON_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "data", "dashboard", "latest.json"
)

# ---------------------------------------------------------------------------
# Google Infrastructure IDs
# ---------------------------------------------------------------------------
DRIVE_ROOT_ID = "1Tng3i4Cly7isKOxIkGqiTmGiZNEtPj3D"
CURRENT_FOLDER_ID = "1JelM_zZgPeX8TluTfaNqQmsTm3tXkG_8"
DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    path = os.path.join(CONFIG_DIR, "CIO_CONFIG.yaml")
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
# Drive Read: Download JSON from CURRENT/ folder
# ---------------------------------------------------------------------------
def read_drive_json(service, filename: str, folder_id: str = CURRENT_FOLDER_ID) -> dict | None:
    """Read a JSON file from a Drive folder. Returns None if not found."""
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id,name)").execute()
        files = results.get("files", [])
        if not files:
            logger.warning(f"Drive: {filename} not found in folder {folder_id}")
            return None

        file_id = files[0]["id"]
        content = service.files().get_media(fileId=file_id).execute()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Drive read failed for {filename}: {e}")
        return None


# ---------------------------------------------------------------------------
# Drive Write: Upload JSON to CURRENT/ folder
# ---------------------------------------------------------------------------
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

    logger.info(f"Drive: {filename} written to CURRENT/")


def write_drive_text(service, text: str, filename: str,
                     folder_id: str = CURRENT_FOLDER_ID) -> None:
    """Upload or update a text/markdown file in a Drive folder."""
    from googleapiclient.http import MediaInMemoryUpload

    content = text.encode("utf-8")
    media = MediaInMemoryUpload(content, mimetype="text/markdown")

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
# V16 Production Sheet Read
# ---------------------------------------------------------------------------
def read_v16_production(sheets_service) -> dict:
    """Read V16 production data from V16 Sheet."""
    try:
        # Read key fields from V16 sheet
        # The V16_DAILY_RUNNER writes regime info — we read the same data
        # that the dashboard.json v16 block contains
        # For now: read from dashboard.json v16 block as it's the freshest source
        if os.path.exists(DASHBOARD_JSON_PATH):
            with open(DASHBOARD_JSON_PATH, "r") as f:
                dashboard = json.load(f)
            v16 = dashboard.get("v16", {})
            if v16.get("status") == "AVAILABLE":
                return {
                    "date": dashboard.get("date", ""),
                    "regime": v16.get("regime", "UNKNOWN"),
                    "regime_confidence": v16.get("regime_confidence"),
                    "dd_protect_status": v16.get("dd_protect_status", "INACTIVE"),
                    "dd_protect_assets": v16.get("dd_protect_assets", []),
                    "current_drawdown": v16.get("current_drawdown", 0.0),
                    "current_weights": v16.get("current_weights", {}),
                    "target_weights": v16.get("target_weights", {}),
                    "top_5_weights": v16.get("top_5_weights", []),
                    "performance": v16.get("performance", {}),
                    "macro_state_num": v16.get("macro_state_num"),
                    "macro_state_name": v16.get("macro_state_name", ""),
                    "growth_signal": v16.get("growth_signal"),
                    "liq_direction": v16.get("liq_direction"),
                    "stress_score": v16.get("stress_score"),
                }
        logger.warning("V16 data not available from dashboard.json")
        return {}
    except Exception as e:
        logger.error(f"V16 read failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Sheets Write: AGENT_SUMMARY
# ---------------------------------------------------------------------------
def write_agent_summary(sheets_service, cio_output: dict, mode: str) -> None:
    """Write 1 row CIO summary to AGENT_SUMMARY tab."""
    try:
        if sheets_service is None:
            return

        today = date.today().isoformat()
        agent_name = "CIO_DRAFT" if mode == "draft" else "CIO_FINAL"
        meta = cio_output.get("metadata", {})

        # Columns: DATE, AGENT, ROLE, SUMMARY, CONFIDENCE, KEY_ACTION,
        #          DRIVE_LINK, OVERRIDE, OVERRIDE_REASON,
        #          section_texts_json, section_word_counts_json, fact_check_corrections_count
        row = [
            today,
            agent_name,
            "Chief Investment Officer",
            cio_output.get("briefing_text", "")[:300],
            cio_output.get("system_conviction", ""),
            f"{cio_output.get('briefing_type', '')} — "
            f"{sum(1 for a in cio_output.get('action_items', []) if a.get('type') == 'ACT')} ACT items",
            "",  # DRIVE_LINK (filled by Drive write)
            "",  # OVERRIDE
            "",  # OVERRIDE_REASON
            json.dumps(cio_output.get("section_texts", {})),
            json.dumps(cio_output.get("section_word_counts", {})),
            str(cio_output.get("fact_check_corrections_count", 0)),
        ]

        sheets_service.spreadsheets().values().append(
            spreadsheetId=DW_SHEET_ID,
            range="AGENT_SUMMARY!A:L",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
        logger.info(f"Sheet AGENT_SUMMARY: {agent_name} row written")

    except Exception as e:
        logger.error(f"AGENT_SUMMARY write failed: {e}")


# ---------------------------------------------------------------------------
# Load All Inputs
# ---------------------------------------------------------------------------
def load_all_inputs(drive_service, sheets_service) -> dict:
    """Load all CIO inputs from Drive CURRENT/ and V16 Sheet."""
    inputs = {}

    # V16 Production (from dashboard.json)
    inputs["v16_production"] = read_v16_production(sheets_service)

    if drive_service:
        # Market Analyst
        ma = read_drive_json(drive_service, "step1_market_analyst.json")
        inputs["layer_analysis"] = ma if ma else {}
        inputs["beliefs"] = ma.get("conviction_dynamics", {}) if ma else {}

        # IC Intelligence
        ic = read_drive_json(drive_service, "step0b_ic_intelligence.json")
        inputs["ic_intelligence"] = ic if ic else {}

        # Risk Officer
        ro = read_drive_json(drive_service, "step3_risk_officer.json")
        inputs["risk_alerts"] = ro if ro else {}

        # Signal Generator (UNAVAILABLE in V1)
        sg = read_drive_json(drive_service, "step2_signal_generator.json")
        inputs["signals"] = sg if sg else {"status": "UNAVAILABLE"}

        # F6 Production (UNAVAILABLE — no active positions)
        inputs["f6_production"] = {"status": "UNAVAILABLE", "active_positions": [], "signals_today": []}

        # CIO History Digest (from Drive HISTORY/)
        history_folder_id = _find_folder(drive_service, "HISTORY", DRIVE_ROOT_ID)
        if history_folder_id:
            cio_hist = read_drive_json(drive_service, "cio_history_digest.json", history_folder_id)
            inputs["cio_history"] = cio_hist
        else:
            inputs["cio_history"] = None

        # Yesterday's briefing (from Drive CURRENT/ — same file, previous day)
        inputs["yesterday_briefing"] = None  # First run has no yesterday

    else:
        logger.warning("No Drive service — using empty inputs for non-V16 data")
        inputs["layer_analysis"] = {}
        inputs["ic_intelligence"] = {}
        inputs["risk_alerts"] = {}
        inputs["signals"] = {"status": "UNAVAILABLE"}
        inputs["f6_production"] = {"status": "UNAVAILABLE", "active_positions": [], "signals_today": []}
        inputs["cio_history"] = None
        inputs["yesterday_briefing"] = None

    return inputs


def _find_folder(service, name: str, parent_id: str) -> str | None:
    """Find a folder in Drive. Returns folder ID or None."""
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CIO Agent — Step 4 (Draft) / Step 6 (Final)")
    parser.add_argument(
        "--mode",
        choices=["draft", "final"],
        default="draft",
        help="Run mode: 'draft' (Step 4) or 'final' (Step 6, promotes draft if no DA)",
    )
    args = parser.parse_args()

    logger.info(f"CIO Agent starting — mode={args.mode}")
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
    inputs = load_all_inputs(drive_service, sheets_service)

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

    # Import engine
    from step4_cio.engine import run_cio_draft, run_cio_final

    if args.mode == "draft":
        # ========== DRAFT RUN ==========
        logger.info("=" * 60)
        logger.info("CIO DRAFT (Step 4)")
        logger.info("=" * 60)

        cio_output = run_cio_draft(
            inputs=inputs,
            config=config,
            cio_history=inputs.get("cio_history"),
            yesterday_briefing=inputs.get("yesterday_briefing"),
        )

        # Write Draft to Drive CURRENT/
        if drive_service:
            # Remove preprocessor_output from Drive JSON (too large, internal only)
            drive_data = {k: v for k, v in cio_output.items() if k != "preprocessor_output"}
            write_drive_json(drive_service, drive_data, "step4_cio_draft.json")

        # THEN: Promote Draft to Final (DA not built yet → V1 behavior)
        logger.info("=" * 60)
        logger.info("CIO FINAL (Step 6) — DA not available, promoting Draft")
        logger.info("=" * 60)

        final_output = run_cio_final(
            inputs=inputs,
            draft_output=cio_output,
            devils_advocate=None,  # DA not built yet
            config=config,
        )

        # Write Final to Drive CURRENT/
        if drive_service:
            drive_data = {k: v for k, v in final_output.items() if k != "preprocessor_output"}
            write_drive_json(drive_service, drive_data, "step6_cio_final.json")
            write_drive_text(drive_service, final_output["briefing_text"], "step6_cio_final_memo.md")

        # Write CIO History Digest to Drive HISTORY/
        if drive_service:
            history_digest = final_output.get("cio_history_digest", {})
            history_folder_id = _find_folder(drive_service, "HISTORY", DRIVE_ROOT_ID)
            if history_folder_id:
                write_drive_json(drive_service, history_digest, "cio_history_digest.json", history_folder_id)
                logger.info("Drive: cio_history_digest.json written to HISTORY/")
            else:
                logger.warning("HISTORY folder not found — skipping history write")

        # Write AGENT_SUMMARY
        if sheets_service:
            write_agent_summary(sheets_service, final_output, "final")

        # Update dashboard.json
        from step4_cio.dashboard_update import update_dashboard_json
        update_dashboard_json(final_output, DASHBOARD_JSON_PATH)

        output_for_log = final_output

    elif args.mode == "final":
        # ========== FINAL-ONLY RUN (when DA is built) ==========
        # Load draft from Drive CURRENT/
        if not drive_service:
            logger.error("Cannot run final mode without Drive service")
            sys.exit(1)

        draft = read_drive_json(drive_service, "step4_cio_draft.json")
        if not draft:
            logger.error("No draft found in Drive CURRENT/ — run draft first")
            sys.exit(1)

        # Load DA output (when built)
        da = read_drive_json(drive_service, "step5_devils_advocate.json")

        logger.info("=" * 60)
        logger.info("CIO FINAL (Step 6)")
        logger.info("=" * 60)

        final_output = run_cio_final(
            inputs=inputs,
            draft_output=draft,
            devils_advocate=da,
            config=config,
        )

        if drive_service:
            drive_data = {k: v for k, v in final_output.items() if k != "preprocessor_output"}
            write_drive_json(drive_service, drive_data, "step6_cio_final.json")
            write_drive_text(drive_service, final_output["briefing_text"], "step6_cio_final_memo.md")

        if sheets_service:
            write_agent_summary(sheets_service, final_output, "final")

        from step4_cio.dashboard_update import update_dashboard_json
        update_dashboard_json(final_output, DASHBOARD_JSON_PATH)

        output_for_log = final_output

    # Log summary
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    bt = output_for_log.get("briefing_type", "?")
    conv = output_for_log.get("system_conviction", "?")
    words = output_for_log.get("metadata", {}).get("word_count", 0)
    is_fb = output_for_log.get("is_fallback", False)
    is_daf = output_for_log.get("is_draft_as_final", False)

    logger.info(f"CIO Agent complete in {elapsed:.1f}s")
    logger.info(
        f"  Type={bt}, Conviction={conv}, Words={words}, "
        f"Fallback={is_fb}, DraftAsFinal={is_daf}"
    )


if __name__ == "__main__":
    main()
