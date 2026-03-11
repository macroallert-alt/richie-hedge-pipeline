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
# Telegram delivery — Full Newsletter (Multi-Message)
# ---------------------------------------------------------------------------

def _send_single_telegram(token, chat_id, text):
    """Send a single Telegram message. Returns True on success."""
    try:
        import requests as http_requests
    except ImportError:
        return False

    # Telegram limit is 4096 chars — split if needed
    chunks = []
    while len(text) > 4096:
        # Find last newline before 4096
        split_at = text.rfind("\n", 0, 4096)
        if split_at == -1:
            split_at = 4096
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)

    success = True
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            resp = http_requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Telegram chunk send failed: {e}")
            success = False
    return success


def send_telegram_message(newsletter):
    """
    Send full newsletter to Telegram as multi-message sequence.
    All actionable information included — no truncation.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping Telegram")
        return False

    d = newsletter.get("date", "")
    fmt = newsletter.get("format", "DAILY")

    def zone_emoji(zone):
        return {"CALM": "🟢", "ELEVATED": "🟡", "STRESS": "🟠", "PANIC": "🔴"}.get(zone, "⚪")

    def sev_emoji(sev):
        return {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🔵"}.get(sev, "⚪")

    # ===================================================================
    # MESSAGE 1: Header + One Thing + Composite + Regime + Warnings
    # ===================================================================
    tact = newsletter.get("composite_scores", {}).get("tactical", {})
    pos = newsletter.get("composite_scores", {}).get("positional", {})
    struct = newsletter.get("composite_scores", {}).get("structural", {})

    regime_ctx = newsletter.get("regime_context", {})
    regime = regime_ctx.get("v16_regime", "?")
    regime_days = regime_ctx.get("regime_duration_days", "?")
    fragility = regime_ctx.get("fragility_state", "?")

    coherence = newsletter.get("pipeline_coherence", {})
    coherence_score = coherence.get("score", "?")
    data_int = newsletter.get("data_integrity", {})
    data_int_score = data_int.get("score", "?")

    msg1 = f"""📊 BALDUR CREEK CAPITAL — {d} ({fmt})

💡 {newsletter.get('one_thing', '')}

━━━ COMPOSITE SCORES ━━━
{zone_emoji(tact.get('zone'))} TACTICAL:    {tact.get('score', '?')} {tact.get('zone', '?')}
   Velocity: {tact.get('velocity', 0):+.1f}/d | Acceleration: {tact.get('acceleration', 0):+.1f}
{zone_emoji(pos.get('zone'))} POSITIONAL:  {pos.get('score', '?')} {pos.get('zone', '?')}
   Velocity: {pos.get('velocity', 0):+.1f}/d | Acceleration: {pos.get('acceleration', 0):+.1f}
{zone_emoji(struct.get('zone'))} STRUCTURAL:  {struct.get('score', '?')} {struct.get('zone', '?')}
   Velocity: {struct.get('velocity', 0):+.1f}/d | Acceleration: {struct.get('acceleration', 0):+.1f}

━━━ REGIME & SYSTEM ━━━
Regime: {regime} (seit {regime_days}d)
Fragility: {fragility}
Pipeline Coherence: {coherence_score}%
Data Integrity: {data_int_score}%
Anchor Type: {newsletter.get('anchor_type', '?')}"""

    # Regime Interpretation (LLM)
    regime_interp = newsletter.get("regime_interpretation", "")
    if regime_interp:
        msg1 += f"\n\n📝 {regime_interp}"

    # Warning triggers
    warnings = newsletter.get("warning_triggers", [])
    if warnings:
        msg1 += "\n\n⚠️ WARNING TRIGGERS:"
        for w in warnings:
            msg1 += f"\n  {w.get('id', '?')}: {w.get('description', '')} ({w.get('penalty', 0):+d})"
    else:
        msg1 += "\n\n✅ Keine Warning Triggers aktiv"

    # Pipeline divergences
    divs = coherence.get("divergences", [])
    if divs:
        msg1 += "\n\n⚡ DIVERGENZEN:"
        for dv in divs:
            msg1 += f"\n  {dv.get('type', '')}: {dv.get('detail', '')}"

    _send_single_telegram(token, chat_id, msg1)

    # ===================================================================
    # MESSAGE 2: Against You + If Wrong + Risk Alerts
    # ===================================================================
    msg2 = "━━━ WAS GEGEN DICH LÄUFT ━━━"

    against = newsletter.get("against_you", {})
    positions = against.get("positions", [])
    if positions:
        for p in positions:
            asset = p.get("asset", "?")
            risk = p.get("top_risk", "")
            prob = p.get("probability_pct", "?")
            mech = p.get("mechanism", "")
            msg2 += f"\n\n🎯 {asset}:"
            msg2 += f"\n  Risiko: {risk}"
            msg2 += f"\n  Mechanismus: {mech}"
            msg2 += f"\n  Wahrscheinlichkeit: {prob}%"
    else:
        msg2 += "\n  Keine spezifischen Risiken identifiziert"

    if_wrong = against.get("if_wrong_summary", "")
    if if_wrong:
        msg2 += f"\n\n💀 IF WRONG: {if_wrong}"

    # Risk alerts from pipeline
    risk_heatmap = newsletter.get("risk_heatmap", {})
    hm_positions = risk_heatmap.get("positions", [])
    hm_factors = risk_heatmap.get("risk_factors", [])
    hm_matrix = risk_heatmap.get("matrix", [])
    if hm_positions and hm_factors and hm_matrix:
        msg2 += "\n\n━━━ RISK HEATMAP ━━━"
        for i, pos_name in enumerate(hm_positions):
            if i < len(hm_matrix):
                row_str = " | ".join(
                    f"{hm_factors[j]}: {hm_matrix[i][j]}"
                    for j in range(min(len(hm_factors), len(hm_matrix[i])))
                )
                msg2 += f"\n{pos_name}: {row_str}"

    _send_single_telegram(token, chat_id, msg2)

    # ===================================================================
    # MESSAGE 3: Szenarien (komplett)
    # ===================================================================
    scenarios = newsletter.get("scenarios", [])
    if scenarios:
        msg3 = "━━━ SZENARIEN ━━━"
        for s in scenarios:
            sid = s.get("id", "?")
            prob = s.get("probability_pct", "?")
            desc = s.get("description", "")
            impact = s.get("portfolio_impact", "")
            comp_impact = s.get("composite_impact", "")
            action = s.get("action", "")
            msg3 += f"\n\n📊 Szenario {sid} ({prob}%):"
            msg3 += f"\n  {desc}"
            if impact:
                msg3 += f"\n  Portfolio: {impact}"
            if comp_impact:
                msg3 += f"\n  Composite: {comp_impact}"
            if action:
                msg3 += f"\n  Action: {action}"
        _send_single_telegram(token, chat_id, msg3)

    # ===================================================================
    # MESSAGE 4: Breaking News (alle HIGH + MEDIUM)
    # ===================================================================
    breaking = newsletter.get("breaking_news", [])
    if breaking:
        msg4 = "━━━ BREAKING NEWS ━━━"
        for b in breaking:
            impact = b.get("impact", "?")
            title = b.get("title", "")
            cat = b.get("category", "")
            source = b.get("source", "")
            transmission = b.get("portfolio_transmission", {})

            emoji = "🔴" if impact == "HIGH" else "🟠" if impact == "MEDIUM" else "🔵"
            msg4 += f"\n\n{emoji} [{impact}] {title}"
            msg4 += f"\n  Kategorie: {cat}"
            if source:
                msg4 += f" | Quelle: {source}"

            # Portfolio transmission
            if transmission:
                assets = transmission.get("affected_assets", [])
                if isinstance(assets, list):
                    for a in assets[:3]:
                        if isinstance(a, dict):
                            msg4 += f"\n  → {a.get('asset', '?')}: {a.get('direction', '?')} ({a.get('mechanism', '')})"
                elif isinstance(transmission, dict):
                    for key in ["direction", "mechanism", "exposure_pct"]:
                        val = transmission.get(key)
                        if val:
                            msg4 += f"\n  {key}: {val}"

        news_summary = newsletter.get("breaking_news_summary", "")
        if news_summary:
            msg4 += f"\n\n📋 Zusammenfassung: {news_summary}"

        _send_single_telegram(token, chat_id, msg4)

    # ===================================================================
    # MESSAGE 5: Indikatoren (Core + Regime-Sensitiv)
    # ===================================================================
    indicators = newsletter.get("indicators", {})
    core = indicators.get("core", [])
    regime_sens = indicators.get("regime_sensitive", [])

    if core or regime_sens:
        msg5 = "━━━ INDIKATOREN ━━━"

        if core:
            msg5 += "\n\nCORE:"
            for ind in core:
                name = ind.get("name", "?")
                val = ind.get("value")
                norm = ind.get("normalized")
                weight = ind.get("weight", 0)
                status = ind.get("status", "?")
                alert_flag = "⚠️" if ind.get("alert") else "✅"
                val_str = f"{val}" if val is not None else "N/A"
                norm_str = f"{norm:.0f}/100" if norm is not None else "N/A"
                msg5 += f"\n  {alert_flag} {name}: {val_str} (Score: {norm_str}, Gewicht: {weight:.0%}, {status})"

        if regime_sens:
            msg5 += "\n\nREGIME-SENSITIV ({regime}):"
            for ind in regime_sens:
                name = ind.get("name", "?")
                val = ind.get("value")
                norm = ind.get("normalized")
                weight = ind.get("weight", 0)
                status = ind.get("status", "?")
                alert_flag = "⚠️" if ind.get("alert") else "✅"
                val_str = f"{val}" if val is not None else "N/A"
                norm_str = f"{norm:.0f}/100" if norm is not None else "N/A"
                msg5 += f"\n  {alert_flag} {name}: {val_str} (Score: {norm_str}, Gewicht: {weight:.0%}, {status})"

        watchlist = indicators.get("watchlist_triggered", [])
        if watchlist:
            msg5 += "\n\n🚨 WATCHLIST TRIGGERED:"
            for w in watchlist:
                msg5 += f"\n  {w}"

        _send_single_telegram(token, chat_id, msg5)

    # ===================================================================
    # MESSAGE 6: Intelligence Digest + Catalysts + Epistemic
    # ===================================================================
    intel = newsletter.get("intelligence_digest", {})
    msg6 = "━━━ INTELLIGENCE DIGEST ━━━"
    msg6 += f"\nIC Konsens: {intel.get('ic_net_direction', '?')} (Score: {intel.get('ic_net_score', 0)})"
    msg6 += f"\nAktive Threads: {intel.get('active_threads', 0)} | Bedrohlich: {intel.get('threatening_threads', 0)}"
    msg6 += f"\nPre-Mortems HIGH: {intel.get('pre_mortem_high_count', 0)}"
    msg6 += f"\nCadence-Anomalien: {intel.get('cadence_anomalies', 0)}"
    msg6 += f"\nExperten-Dissens: {intel.get('expert_disagreements', 0)}"

    # Catalysts
    catalysts = newsletter.get("catalysts_48h", [])
    if catalysts:
        msg6 += "\n\n━━━ CATALYSTS 48h ━━━"
        for c in catalysts:
            c_date = c.get("date", "")
            event = c.get("event", "")
            impact = c.get("impact", "")
            hours = c.get("hours_until", "")
            msg6 += f"\n  [{impact}] {c_date} — {event}"
            if hours:
                msg6 += f" (in {hours}h)"
    else:
        msg6 += "\n\nKeine HIGH-Impact Catalysts in den nächsten 48h"

    # Epistemic
    epist = newsletter.get("epistemic_status", {})
    msg6 += f"\n\n━━━ EPISTEMIC STATUS ━━━"
    msg6 += f"\nData Quality: {epist.get('data_quality', '?')}"
    msg6 += f"\nSystem Conviction: {epist.get('system_conviction', '?')}"

    blind_spots = epist.get("blind_spots", [])
    if blind_spots:
        msg6 += "\nBlind Spots:"
        for bs in blind_spots:
            msg6 += f"\n  • {bs}"

    stale = epist.get("stale_sources", [])
    if stale:
        msg6 += "\nStale Sources:"
        for s in stale:
            msg6 += f"\n  • {s.get('name', '?')} ({s.get('type', '')})"

    _send_single_telegram(token, chat_id, msg6)

    # ===================================================================
    # MESSAGE 7: Portfolio + Behavioral Safeguards
    # ===================================================================
    portfolio = newsletter.get("portfolio_attribution", {})
    msg7 = "━━━ PORTFOLIO ━━━"

    total_pnl = portfolio.get("total_pnl_pct")
    if total_pnl is not None:
        msg7 += f"\nTages-P&L: {total_pnl:+.2f}%"

    positions = portfolio.get("positions", [])
    if positions:
        for p in positions:
            asset = p.get("asset", "?")
            weight = p.get("weight_pct", 0)
            pnl = p.get("pnl_pct")
            held = p.get("held_days")
            pnl_str = f" | P&L: {pnl:+.2f}%" if pnl is not None else ""
            held_str = f" | {held}d" if held is not None else ""
            msg7 += f"\n  {asset}: {weight}%{pnl_str}{held_str}"

    # Behavioral
    behavioral = newsletter.get("behavioral", {})
    anchoring = behavioral.get("anchoring_alerts", [])
    inaction = behavioral.get("inaction_tracker", {})
    sys_action = behavioral.get("system_action", "?")

    msg7 += f"\n\n━━━ BEHAVIORAL SAFEGUARDS ━━━"
    msg7 += f"\nSystem Action: {sys_action}"
    msg7 += f"\nInaction Status: {inaction.get('status', '?')}"

    if anchoring:
        msg7 += "\n\n🧠 ANCHORING CHECK:"
        for a in anchoring:
            msg7 += f"\n  {a.get('asset', '?')} ({a.get('weight_pct', 0)}%): {a.get('question', '')}"

    # Contrarian (Fridays)
    contrarian = newsletter.get("contrarian_check")
    if contrarian:
        msg7 += f"\n\n🔄 CONTRARIAN CHECK:\n  {contrarian}"

    msg7 += f"\n\n━━━━━━━━━━━━━━━━━━━━━"
    msg7 += f"\n🏁 Baldur Creek Capital — {newsletter.get('anchor_type', '?')} — {d}"

    _send_single_telegram(token, chat_id, msg7)

    logger.info("Telegram full newsletter sent (7 messages)")
    return True


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
