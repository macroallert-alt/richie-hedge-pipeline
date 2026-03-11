"""
Daily Briefing System — Main Entry Point
Baldur Creek Capital | Step 0u
Usage: python -m step_0u_daily_briefing.main [--skip-news] [--skip-telegram] [--latest PATH]

Orchestrates:
  1. Data Collection (latest.json + DW Sheet)
  2. Composite Score Engine
  3. Breaking News Scanner
  4. Newsletter Assembly (LLM)
  5. Save newsletter JSON
  6. Telegram delivery
  7. Drive upload (HISTORY/newsletter/)

Idempotency: Checks last_run_date.txt — skips if already run today.
Fallback: If LLM fails → degraded newsletter. If Sheets fail → fallback from latest.json.
Retry: Caller (GitHub Actions) handles retries (3x per Spec §2.3).
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daily_briefing")

# ---------------------------------------------------------------------------
# Imports from submodules
# ---------------------------------------------------------------------------
from .config import (
    HISTORY_DIR,
    IDEMPOTENCY_FLAG_FILE,
    NEWSLETTER_FORMATS,
)
from .data_reader import collect_all_data, load_latest_json, extract_pipeline_data
from .composite import compute_composite_scores
from .news_scanner import run_news_scanner
from .assembler import assemble_newsletter, save_newsletter


# ---------------------------------------------------------------------------
# Idempotency check (Spec §2.3)
# ---------------------------------------------------------------------------

def check_already_run():
    """Return True if newsletter was already generated today."""
    if not os.path.exists(IDEMPOTENCY_FLAG_FILE):
        return False
    try:
        with open(IDEMPOTENCY_FLAG_FILE, "r") as f:
            last_date = f.read().strip()
        return last_date == date.today().isoformat()
    except Exception:
        return False


def mark_run_complete():
    """Write today's date to idempotency flag file."""
    os.makedirs(os.path.dirname(IDEMPOTENCY_FLAG_FILE), exist_ok=True)
    with open(IDEMPOTENCY_FLAG_FILE, "w") as f:
        f.write(date.today().isoformat())


# ---------------------------------------------------------------------------
# Telegram delivery (Spec §8)
# ---------------------------------------------------------------------------

def send_telegram_message(newsletter):
    """
    Send compressed newsletter summary to Telegram.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping Telegram")
        return False

    try:
        import requests as http_requests
    except ImportError:
        logger.error("requests not available — cannot send Telegram")
        return False

    # Build message (Spec §8.2)
    tact = newsletter.get("composite_scores", {}).get("tactical", {})
    pos = newsletter.get("composite_scores", {}).get("positional", {})
    struct = newsletter.get("composite_scores", {}).get("structural", {})

    # Zone emoji
    def zone_emoji(zone):
        return {"CALM": "🟢", "ELEVATED": "🟡", "STRESS": "🟠", "PANIC": "🔴"}.get(zone, "⚪")

    one_thing = newsletter.get("one_thing", "")
    regime = newsletter.get("regime_context", {}).get("v16_regime", "?")
    fmt = newsletter.get("format", "DAILY")
    d = newsletter.get("date", "")

    # Warnings
    warnings = newsletter.get("warning_triggers", [])
    warn_str = ""
    if warnings:
        warn_str = "\n⚠ WARNINGS:\n" + "\n".join(
            f"  {w['description']}" for w in warnings[:3]
        )

    # Risk
    risk_critical = len([a for a in newsletter.get("breaking_news", []) if a.get("impact") == "HIGH"])

    msg = f"""📊 BALDUR CREEK CAPITAL — {d}

💡 {one_thing}

COMPOSITE:
  {zone_emoji(tact.get('zone'))} TACTICAL:   {tact.get('score', '?')} {tact.get('zone', '?')} (vel {tact.get('velocity', 0):+.0f})
  {zone_emoji(pos.get('zone'))} POSITIONAL: {pos.get('score', '?')} {pos.get('zone', '?')} (vel {pos.get('velocity', 0):+.0f})
  {zone_emoji(struct.get('zone'))} STRUCTURAL: {struct.get('score', '?')} {struct.get('zone', '?')} (vel {struct.get('velocity', 0):+.0f})

REGIME: {regime} ({newsletter.get('regime_context', {}).get('regime_duration_days', '?')}d)
PIPELINE COHERENCE: {newsletter.get('pipeline_coherence', {}).get('score', '?')}%
{warn_str}"""

    # Breaking news
    breaking = newsletter.get("breaking_news", [])
    if breaking:
        msg += "\n\n📰 BREAKING:\n"
        for b in breaking[:3]:
            msg += f"  [{b.get('impact', '')}] {b.get('title', '')[:60]}\n"

    # Scenarios
    scenarios = newsletter.get("scenarios", [])
    if scenarios:
        msg += "\n📊 SZENARIEN:\n"
        for s in scenarios[:3]:
            msg += f"  {s.get('id', '?')} ({s.get('probability_pct', '?')}%): {s.get('description', '')[:50]}\n"

    # Trim to Telegram limit
    if len(msg) > 4096:
        msg = msg[:4090] + "\n..."

    try:
        resp = http_requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": msg,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def send_telegram_error():
    """Send error notification to Telegram (Spec §2.3 fallback)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    try:
        import requests as http_requests
        http_requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": f"⚠️ Newsletter konnte nicht generiert werden ({date.today().isoformat()}). Manueller Check empfohlen.",
            },
            timeout=15,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Drive upload
# ---------------------------------------------------------------------------

def upload_to_drive(newsletter_path):
    """Upload newsletter JSON to Drive HISTORY/newsletter/YYYY/MM/."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2.service_account import Credentials
        import tempfile
    except ImportError:
        logger.warning("Google API client not available — skipping Drive upload")
        return None

    sa_key = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
    if not sa_key:
        logger.warning("No GCP credentials — skipping Drive upload")
        return None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(sa_key)
            tmp_path = f.name
        creds = Credentials.from_service_account_file(
            tmp_path,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        os.unlink(tmp_path)

        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        # Find or create HISTORY/newsletter folder
        from .config import DRIVE_ROOT_FOLDER
        # Simplified: upload to root HISTORY folder
        # In production, create YYYY/MM subfolders

        filename = os.path.basename(newsletter_path)
        media = MediaFileUpload(newsletter_path, mimetype="application/json")
        file_metadata = {
            "name": filename,
            "parents": [DRIVE_ROOT_FOLDER],
        }

        uploaded = service.files().create(
            body=file_metadata, media_body=media, fields="id"
        ).execute()

        logger.info(f"Newsletter uploaded to Drive: {uploaded.get('id')}")
        return uploaded.get("id")

    except Exception as e:
        logger.error(f"Drive upload failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Update latest.json with newsletter summary
# ---------------------------------------------------------------------------

def update_latest_json(newsletter, latest_json_path):
    """
    Add newsletter summary to latest.json so the Vercel dashboard can read it.
    Adds a 'newsletter' block to the existing latest.json.
    """
    if not os.path.exists(latest_json_path):
        logger.warning(f"latest.json not found at {latest_json_path} — skipping update")
        return

    try:
        with open(latest_json_path, "r", encoding="utf-8") as f:
            latest = json.load(f)

        # Add newsletter summary block
        tact = newsletter.get("composite_scores", {}).get("tactical", {})
        pos = newsletter.get("composite_scores", {}).get("positional", {})
        struct = newsletter.get("composite_scores", {}).get("structural", {})

        latest["newsletter"] = {
            "date": newsletter.get("date"),
            "format": newsletter.get("format"),
            "generated_at": newsletter.get("generated_at"),
            "one_thing": newsletter.get("one_thing", ""),
            "composite_scores": {
                "tactical": {"score": tact.get("score"), "zone": tact.get("zone"),
                             "velocity": tact.get("velocity")},
                "positional": {"score": pos.get("score"), "zone": pos.get("zone"),
                               "velocity": pos.get("velocity")},
                "structural": {"score": struct.get("score"), "zone": struct.get("zone"),
                                "velocity": struct.get("velocity")},
            },
            "data_integrity": newsletter.get("data_integrity", {}).get("score"),
            "pipeline_coherence": newsletter.get("pipeline_coherence", {}).get("score"),
            "anchor_type": newsletter.get("anchor_type"),
            "warning_count": len(newsletter.get("warning_triggers", [])),
            "breaking_news_count": len(newsletter.get("breaking_news", [])),
            "scenarios_count": len(newsletter.get("scenarios", [])),
        }

        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(latest, f, indent=2, ensure_ascii=False, default=str)

        logger.info("latest.json updated with newsletter block")

    except Exception as e:
        logger.error(f"Failed to update latest.json: {e}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Daily Briefing System — Step 0u")
    parser.add_argument("--skip-news", action="store_true", help="Skip Brave Search news scan")
    parser.add_argument("--skip-telegram", action="store_true", help="Skip Telegram delivery")
    parser.add_argument("--skip-drive", action="store_true", help="Skip Drive upload")
    parser.add_argument("--force", action="store_true", help="Ignore idempotency check")
    parser.add_argument("--latest", type=str, default=None, help="Path to latest.json")
    parser.add_argument("--creds", type=str, default=None, help="Path to GCP SA key JSON")
    args = parser.parse_args()

    today = date.today()
    fmt = NEWSLETTER_FORMATS.get(today.weekday(), "DAILY")
    logger.info(f"=== DAILY BRIEFING SYSTEM — {today.isoformat()} ({fmt}) ===")

    # Idempotency check
    if not args.force and check_already_run():
        logger.info("Newsletter already generated today — skipping (use --force to override)")
        return 0

    try:
        # STEP 1: Collect data
        logger.info("STEP 1: Collecting data...")
        indicator_values, pipeline_data = collect_all_data(args.latest, args.creds)
        regime = pipeline_data.get("v16_regime", "UNKNOWN")
        logger.info(f"  Regime: {regime}, Indicators: {len(indicator_values)}, "
                     f"Coherence: {pipeline_data.get('pipeline_coherence_pct')}%")

        # STEP 2: Composite Scores
        logger.info("STEP 2: Computing composite scores...")
        composite = compute_composite_scores(indicator_values, regime, pipeline_data)
        tact = composite["tactical"]["score"]
        pos = composite["positional"]["score"]
        struct = composite["structural"]["score"]
        logger.info(f"  TACTICAL={tact} {composite['tactical']['zone']}, "
                     f"POSITIONAL={pos} {composite['positional']['zone']}, "
                     f"STRUCTURAL={struct} {composite['structural']['zone']}")

        # STEP 3: Breaking News
        news_result = None
        if not args.skip_news:
            logger.info("STEP 3: Scanning breaking news...")
            v16_weights = pipeline_data.get("v16_weights", {})
            news_result = run_news_scanner(v16_weights)
            logger.info(f"  HIGH: {news_result.get('high_impact_count', 0)}, "
                         f"MEDIUM: {news_result.get('medium_impact_count', 0)}")
        else:
            logger.info("STEP 3: News scan skipped")
            news_result = {"hits": [], "high_impact_count": 0, "summary": "News-Scan uebersprungen."}

        # STEP 4: Assemble Newsletter
        logger.info("STEP 4: Assembling newsletter...")
        newsletter = assemble_newsletter(
            composite, pipeline_data, news_result, indicator_values
        )
        logger.info(f"  Format: {newsletter.get('format')}, Anchor: {newsletter.get('anchor_type')}, "
                     f"Blocks: {sum(1 for v in newsletter.values() if v)}")

        # STEP 5: Save
        logger.info("STEP 5: Saving newsletter...")
        newsletter_path = save_newsletter(newsletter)

        # Also save to data/dashboard/ for latest.json merge
        latest_path = args.latest or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "dashboard", "latest.json",
        )
        if os.path.exists(latest_path):
            update_latest_json(newsletter, latest_path)

        # STEP 6: Telegram
        if not args.skip_telegram:
            logger.info("STEP 6: Sending Telegram...")
            send_telegram_message(newsletter)
        else:
            logger.info("STEP 6: Telegram skipped")

        # STEP 7: Drive upload
        if not args.skip_drive:
            logger.info("STEP 7: Uploading to Drive...")
            upload_to_drive(newsletter_path)
        else:
            logger.info("STEP 7: Drive upload skipped")

        # Mark complete
        mark_run_complete()

        logger.info(f"=== DAILY BRIEFING COMPLETE — {newsletter.get('anchor_type')} ===")
        return 0

    except Exception as e:
        logger.error(f"DAILY BRIEFING FAILED: {e}", exc_info=True)
        send_telegram_error()
        return 1


if __name__ == "__main__":
    sys.exit(main())
