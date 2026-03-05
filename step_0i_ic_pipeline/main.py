"""
IC Intelligence Pipeline — Main Entry Point
Usage: python -m step_0i_ic_pipeline.main --stage all
Stages: extraction, intelligence, briefing, all
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ic_pipeline")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Google Drive Output
# ---------------------------------------------------------------------------
DRIVE_ROOT_ID = "1Tng3i4Cly7isKOxIkGqiTmGiZNEtPj3D"


def _get_drive_service():
    """Build Google Drive service from env credentials."""
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
    """Build Google Sheets service from env credentials."""
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


def _find_or_create_folder(service, name: str, parent_id: str) -> str:
    """Find or create a folder in Drive."""
    query = (
        f"name='{name}' and '{parent_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {
        "name": name,
        "parents": [parent_id],
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def _upload_to_drive(service, data: dict, filename: str, folder_id: str) -> None:
    """Upload or update a JSON file in a Drive folder."""
    from googleapiclient.http import MediaInMemoryUpload

    content = json.dumps(data, indent=2).encode("utf-8")
    media = MediaInMemoryUpload(content, mimetype="application/json")

    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        service.files().update(fileId=files[0]["id"], media_body=media).execute()
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        service.files().create(body=metadata, media_body=media).execute()


def write_drive_outputs(
    claims_output: dict, intel: dict, briefing: dict
) -> None:
    """Write all outputs to Google Drive CURRENT/ and HISTORY/ic/YYYY-MM-DD/."""
    try:
        service = _get_drive_service()
        if service is None:
            logger.warning("No Drive credentials — skipping Drive writes")
            return

        today_str = date.today().isoformat()

        # Find CURRENT folder
        current_id = _find_or_create_folder(service, "CURRENT", DRIVE_ROOT_ID)

        # Write to CURRENT/
        _upload_to_drive(service, intel, "step0b_ic_intelligence.json", current_id)
        _upload_to_drive(service, claims_output, "step0b_ic_claims.json", current_id)
        _upload_to_drive(service, briefing, "step0b_ic_briefing.json", current_id)
        logger.info("Drive: CURRENT/ updated")

        # Write to HISTORY/ic/YYYY-MM-DD/ — disabled until P1 (SA quota fix)
        # try:
        #     history_id = _find_or_create_folder(service, "HISTORY", DRIVE_ROOT_ID)
        #     ic_history_id = _find_or_create_folder(service, "ic", history_id)
        #     day_id = _find_or_create_folder(service, today_str, ic_history_id)
        #     _upload_to_drive(service, intel, "step0b_ic_intelligence.json", day_id)
        #     _upload_to_drive(service, claims_output, "step0b_ic_claims.json", day_id)
        #     _upload_to_drive(service, briefing, "step0b_ic_briefing.json", day_id)
        #     logger.info(f"Drive: HISTORY/ic/{today_str}/ archived")
        # except Exception as hist_err:
        #     logger.warning(f"Drive HISTORY write skipped: {hist_err}")
        logger.info("Drive: HISTORY write disabled (P1 — SA quota fix pending)")

    except ImportError:
        logger.warning("Google API libraries not installed — Drive writes skipped")
    except Exception as e:
        logger.error(f"Drive write failed: {e}")


# ---------------------------------------------------------------------------
# Google Sheets Output
# ---------------------------------------------------------------------------
DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"


def write_intelligence_tab(claims: list[dict], sources_config: list[dict]) -> None:
    """Write 1 row per source to INTELLIGENCE tab (12 columns A:L)."""
    try:
        service = _get_sheets_service()
        if service is None:
            return

        today = date.today().isoformat()

        # Build source lookup for tier + bias
        source_lookup = {s["source_id"]: s for s in sources_config}

        # Aggregate per source: use highest-novelty claim as core thesis
        source_rows = {}
        for claim in claims:
            sid = claim["source_id"]
            if sid not in source_rows:
                source_rows[sid] = {
                    "source_name": claim.get("source_name", sid),
                    "core_thesis": "",
                    "direction": "",
                    "intensity": 0,
                    "bias_adj": 0,
                    "claim_type": "",
                    "confidence": 0.0,
                    "novelty_max": 0,
                    "topics": set(),
                    "content_date": "",
                }
            row = source_rows[sid]
            if claim.get("novelty_score", 0) >= row["novelty_max"]:
                row["novelty_max"] = claim["novelty_score"]
                row["core_thesis"] = claim.get("claim_text", "")[:300]
                row["direction"] = claim["sentiment"]["direction"]
                row["intensity"] = claim["sentiment"]["intensity"]
                row["claim_type"] = claim.get("claim_type", "")
                row["confidence"] = claim.get("confidence", {}).get(
                    "extraction_confidence", 0.0
                )
                row["content_date"] = claim.get("content_date", "")
                # Bias-adjusted signal
                known_bias = source_lookup.get(sid, {}).get("known_bias", 0)
                signed = row["intensity"] if row["direction"] == "BULLISH" else (
                    -row["intensity"] if row["direction"] == "BEARISH" else 0
                )
                row["bias_adj"] = signed - known_bias
            row["topics"].update(claim.get("topics", []))

        rows = []
        for sid, row in source_rows.items():
            tier = source_lookup.get(sid, {}).get("tier", "")
            rows.append([
                today,                                  # A: DATE
                sid,                                    # B: SOURCE
                tier,                                   # C: TIER
                row["core_thesis"],                     # D: CORE_THESIS
                row["direction"],                       # E: DIRECTION
                str(row["intensity"]),                  # F: INTENSITY
                str(row["bias_adj"]),                   # G: BIAS_ADJ
                row["claim_type"],                      # H: CLAIM_TYPE
                str(row["novelty_max"]),                # I: NOVELTY
                ", ".join(sorted(row["topics"])),        # J: TOPICS
                str(round(row["confidence"], 2)),        # K: CONFIDENCE
                row["content_date"],                    # L: CONTENT_DATE
            ])

        if rows:
            service.spreadsheets().values().append(
                spreadsheetId=DW_SHEET_ID,
                range="INTELLIGENCE!A:L",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": rows},
            ).execute()
            logger.info(f"Sheet INTELLIGENCE: {len(rows)} rows written")

    except ImportError:
        logger.warning("Google API libs missing — Sheet write skipped")
    except Exception as e:
        logger.error(f"INTELLIGENCE tab write failed: {e}")


def write_agent_summary_tab(briefing: dict) -> None:
    """Write 1 row IC summary to AGENT_SUMMARY tab."""
    try:
        service = _get_sheets_service()
        if service is None:
            return

        today = date.today().isoformat()
        meta = briefing.get("metadata", {})

        row = [
            today, "IC_PIPELINE", "Intelligence Collector",
            briefing.get("briefing_text", "")[:300],
            str(meta.get("topics_covered", 0)),
            f"{meta.get('divergences_flagged', 0)} divergences",
            "", "", "",
        ]

        service.spreadsheets().values().append(
            spreadsheetId=DW_SHEET_ID,
            range="AGENT_SUMMARY!A:I",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
        logger.info("Sheet AGENT_SUMMARY: IC row written")

    except ImportError:
        pass
    except Exception as e:
        logger.error(f"AGENT_SUMMARY write failed: {e}")


# ---------------------------------------------------------------------------
# Stage Runners
# ---------------------------------------------------------------------------
def run_extraction(sources: list[dict]) -> tuple[list[dict], dict]:
    """Run Stufe 1: Fetch + Extract."""
    from step_0i_ic_pipeline.src.extraction.fetcher import fetch_all_sources
    from step_0i_ic_pipeline.src.extraction.extractor import extract_claims

    logger.info("=" * 60)
    logger.info("STUFE 1: EXTRACTION")
    logger.info("=" * 60)

    all_content, fetch_state, failed_sources = fetch_all_sources(sources)
    logger.info(f"Fetched {len(all_content)} items, {len(failed_sources)} failed")

    source_lookup = {s["source_id"]: s for s in sources}
    all_claims = []

    for content in all_content:
        sid = content["source_id"]
        src_config = source_lookup.get(sid, {})
        try:
            claims = extract_claims(content, src_config)
            all_claims.extend(claims)
        except Exception as e:
            logger.error(f"[{sid}] Extraction failed: {e}")
            failed_sources.append({
                "source_id": sid,
                "error": f"Extraction: {e}",
                "retry_next_run": True,
            })

    today = date.today().isoformat()
    run_id = f"run_{today.replace('-', '')}_{datetime.utcnow().strftime('%H%M%S')}"

    no_new = [
        s["source_id"] for s in sources
        if s.get("active", True)
        and s["source_id"] not in {c["source_id"] for c in all_claims}
        and s["source_id"] not in {f["source_id"] for f in failed_sources}
    ]

    claims_output = {
        "extraction_date": today,
        "extraction_run_id": run_id,
        "sources_attempted": sum(1 for s in sources if s.get("active", True)),
        "sources_successful": len(set(c["source_id"] for c in all_claims)),
        "sources_failed": len(failed_sources),
        "sources_no_new_content": len(no_new),
        "total_claims_extracted": len(all_claims),
        "failed_sources": failed_sources,
        "no_new_content": no_new,
        "claims": all_claims,
    }

    # Save locally
    claims_path = os.path.join(DATA_DIR, "claims", f"claims_{today}.json")
    _save_json(claims_output, claims_path)

    logger.info(
        f"Extraction complete: {len(all_claims)} claims from "
        f"{claims_output['sources_successful']} sources"
    )
    return all_claims, claims_output


def run_intelligence(
    claims: list[dict],
    sources: list[dict],
    expertise_matrix: dict,
    taxonomy: dict,
) -> dict:
    """Run Stufe 2: Intelligence Engine (deterministic)."""
    from step_0i_ic_pipeline.src.intelligence.engine import run_intelligence_engine

    logger.info("=" * 60)
    logger.info("STUFE 2: INTELLIGENCE ENGINE")
    logger.info("=" * 60)

    intel = run_intelligence_engine(claims, sources, expertise_matrix, taxonomy)

    today = date.today().isoformat()
    intel_path = os.path.join(DATA_DIR, "intelligence", f"intel_{today}.json")
    _save_json(intel, intel_path)

    return intel


def run_briefing(intel: dict) -> dict:
    """Run Stufe 3: Agent 0 Briefing."""
    from step_0i_ic_pipeline.src.briefing.agent0 import generate_briefing

    logger.info("=" * 60)
    logger.info("STUFE 3: AGENT 0 BRIEFING")
    logger.info("=" * 60)

    briefing = generate_briefing(intel)

    today = date.today().isoformat()
    briefing_path = os.path.join(DATA_DIR, "briefings", f"briefing_{today}.json")
    _save_json(briefing, briefing_path)

    return briefing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IC Intelligence Pipeline")
    parser.add_argument(
        "--stage",
        choices=["extraction", "intelligence", "briefing", "all"],
        default="all",
        help="Which stage to run",
    )
    args = parser.parse_args()

    logger.info(f"IC Pipeline starting — stage={args.stage}")
    start_time = datetime.utcnow()

    # Load config
    sources_config = _load_json(os.path.join(CONFIG_DIR, "sources.json"))
    sources = sources_config["sources"]
    expertise_matrix = _load_json(os.path.join(CONFIG_DIR, "expertise_matrix.json"))
    taxonomy = _load_json(os.path.join(CONFIG_DIR, "taxonomy.json"))

    claims = []
    claims_output = {}
    intel = {}
    briefing = {}

    try:
        # Stufe 1
        if args.stage in ("extraction", "all"):
            claims, claims_output = run_extraction(sources)

        # Stufe 2
        if args.stage in ("intelligence", "all"):
            if args.stage == "intelligence" and not claims:
                # Load today's claims from file
                today = date.today().isoformat()
                claims_path = os.path.join(DATA_DIR, "claims", f"claims_{today}.json")
                if os.path.exists(claims_path):
                    claims_data = _load_json(claims_path)
                    claims = claims_data.get("claims", [])
                    claims_output = claims_data
                else:
                    logger.error("No claims file found for today")
                    sys.exit(1)

            intel = run_intelligence(claims, sources, expertise_matrix, taxonomy)

        # Stufe 3
        if args.stage in ("briefing", "all"):
            if args.stage == "briefing" and not intel:
                today = date.today().isoformat()
                intel_path = os.path.join(DATA_DIR, "intelligence", f"intel_{today}.json")
                if os.path.exists(intel_path):
                    intel = _load_json(intel_path)
                else:
                    logger.error("No intel file found for today")
                    sys.exit(1)

            briefing = run_briefing(intel)

        # Write to Google Drive + Sheets
        if args.stage == "all":
            write_drive_outputs(claims_output, intel, briefing)
            write_intelligence_tab(claims, sources)
            write_agent_summary_tab(briefing)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"IC Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()