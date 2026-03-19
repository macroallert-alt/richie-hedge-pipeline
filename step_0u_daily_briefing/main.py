"""
Daily Briefing System — Main Entry Point
Baldur Creek Capital | Step 0u
Usage: python -m step_0u_daily_briefing.main [--skip-news] [--skip-telegram] [--latest PATH]

V1.1: Added MSG 10 (Crypto Circle) in Telegram delivery.
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daily_briefing")

from .config import (
    HISTORY_DIR,
    IDEMPOTENCY_FLAG_FILE,
    NEWSLETTER_FORMATS,
    RISK_HEATMAP_DESCRIPTIONS,
)
from .data_reader import collect_all_data, load_latest_json, extract_pipeline_data
from .composite import compute_composite_scores
from .news_scanner import run_news_scanner
from .assembler import assemble_newsletter, save_newsletter


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def check_already_run():
    if not os.path.exists(IDEMPOTENCY_FLAG_FILE):
        return False
    try:
        with open(IDEMPOTENCY_FLAG_FILE, "r") as f:
            return f.read().strip() == date.today().isoformat()
    except Exception:
        return False

def mark_run_complete():
    os.makedirs(os.path.dirname(IDEMPOTENCY_FLAG_FILE), exist_ok=True)
    with open(IDEMPOTENCY_FLAG_FILE, "w") as f:
        f.write(date.today().isoformat())


# ---------------------------------------------------------------------------
# Telegram — Full Newsletter Multi-Message
# ---------------------------------------------------------------------------

def _send_single_telegram(token, chat_id, text):
    """Send text to Telegram, auto-splitting at 4096 char boundary."""
    try:
        import requests as http_requests
    except ImportError:
        return False

    chunks = []
    while len(text) > 4096:
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
                json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Telegram chunk send failed: {e}")
            success = False
    return success


def send_telegram_message(newsletter):
    """Send full newsletter to Telegram as multi-message sequence."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping")
        return False

    d = newsletter.get("date", "")
    fmt = newsletter.get("format", "DAILY")

    def ze(zone):
        return {"CALM": "🟢", "ELEVATED": "🟡", "STRESS": "🟠", "PANIC": "🔴"}.get(zone, "⚪")

    # =======================================================================
    # MSG 1: Header + One Thing + Composite + Regime + Warnings
    # =======================================================================
    tact = newsletter.get("composite_scores", {}).get("tactical", {})
    pos = newsletter.get("composite_scores", {}).get("positional", {})
    struct = newsletter.get("composite_scores", {}).get("structural", {})
    regime_ctx = newsletter.get("regime_context", {})
    coherence = newsletter.get("pipeline_coherence", {})
    data_int = newsletter.get("data_integrity", {})

    msg1 = f"""📊 BALDUR CREEK CAPITAL — {d} ({fmt})

💡 {newsletter.get('one_thing', '')}

━━━ COMPOSITE SCORES ━━━
{ze(tact.get('zone'))} TACTICAL:    {tact.get('score', '?')} {tact.get('zone', '?')}
   Velocity: {tact.get('velocity', 0):+.1f}/d | Acceleration: {tact.get('acceleration', 0):+.1f}
{ze(pos.get('zone'))} POSITIONAL:  {pos.get('score', '?')} {pos.get('zone', '?')}
   Velocity: {pos.get('velocity', 0):+.1f}/d | Acceleration: {pos.get('acceleration', 0):+.1f}
{ze(struct.get('zone'))} STRUCTURAL:  {struct.get('score', '?')} {struct.get('zone', '?')}
   Velocity: {struct.get('velocity', 0):+.1f}/d | Acceleration: {struct.get('acceleration', 0):+.1f}

━━━ REGIME & SYSTEM ━━━
Regime: {regime_ctx.get('v16_regime', '?')} (seit {regime_ctx.get('regime_duration_days', '?')}d)
Fragility: {regime_ctx.get('fragility_state', '?')}
Pipeline Coherence: {coherence.get('score', '?')}%
Data Integrity: {data_int.get('score', '?')}%
Anchor: {newsletter.get('anchor_type', '?')}"""

    ri = newsletter.get("regime_interpretation", "")
    if ri:
        msg1 += f"\n\n📝 {ri}"

    warnings = newsletter.get("warning_triggers", [])
    if warnings:
        msg1 += "\n\n⚠️ WARNING TRIGGERS:"
        for w in warnings:
            msg1 += f"\n  {w.get('id', '?')}: {w.get('description', '')} ({w.get('penalty', 0):+d})"
    else:
        msg1 += "\n\n✅ Keine Warning Triggers aktiv"

    divs = coherence.get("divergences", [])
    if divs:
        msg1 += "\n\n⚡ DIVERGENZEN:"
        for dv in divs:
            msg1 += f"\n  {dv.get('type', '')}: {dv.get('detail', '')}"

    _send_single_telegram(token, chat_id, msg1)

    # =======================================================================
    # MSG 2: Against You + If Wrong + Risk Heatmap (ausformuliert)
    # =======================================================================
    msg2 = "━━━ WAS GEGEN DICH LÄUFT ━━━"

    against = newsletter.get("against_you", {})
    against_positions = against.get("positions", [])
    if against_positions:
        for p in against_positions:
            msg2 += f"\n\n🎯 {p.get('asset', '?')}:"
            msg2 += f"\n  Risiko: {p.get('top_risk', '')}"
            msg2 += f"\n  Mechanismus: {p.get('mechanism', '')}"
            msg2 += f"\n  Wahrscheinlichkeit: {p.get('probability_pct', '?')}%"
    else:
        msg2 += "\n  Keine spezifischen Risiken identifiziert"

    if_wrong = against.get("if_wrong_summary", "")
    if if_wrong:
        msg2 += f"\n\n💀 IF WRONG: {if_wrong}"

    _send_single_telegram(token, chat_id, msg2)

    # =======================================================================
    # MSG 3: Risk Heatmap — ausformuliert mit Mechanismus + LLM-Kontext
    # =======================================================================
    heatmap = newsletter.get("risk_heatmap", {})
    hm_positions = heatmap.get("positions", [])
    hm_factors = heatmap.get("risk_factors", [])
    hm_assessments = heatmap.get("assessments", {})

    if hm_positions:
        msg3 = "━━━ RISK HEATMAP (DETAILLIERT) ━━━"
        # Get weights for context
        portfolio = newsletter.get("portfolio_attribution", {})
        pos_list = portfolio.get("positions", [])
        weight_map = {p["asset"]: p["weight_pct"] for p in pos_list}

        for ticker in hm_positions:
            weight = weight_map.get(ticker, "?")
            msg3 += f"\n\n📌 {ticker} ({weight}%):"
            hm_desc = RISK_HEATMAP_DESCRIPTIONS.get(ticker, {})

            for factor in hm_factors:
                fd = hm_desc.get(factor, {})
                severity = fd.get("severity", "?")
                mechanism = fd.get("mechanism", "")

                sev_icon = {"DIREKT": "🔴", "INDIREKT": "🟠", "MINIMAL": "🟢", "SAFE_HAVEN": "🛡️"}.get(severity, "⚪")
                msg3 += f"\n\n  {sev_icon} {factor} → {severity}"
                if mechanism:
                    msg3 += f"\n  {mechanism}"

                # LLM current assessment
                assess_key = f"{ticker}__{factor}"
                assessment = hm_assessments.get(assess_key, "")
                if assessment:
                    msg3 += f"\n  📊 Heute: {assessment}"

        _send_single_telegram(token, chat_id, msg3)

    # =======================================================================
    # MSG 4: Szenarien (komplett)
    # =======================================================================
    scenarios = newsletter.get("scenarios", [])
    if scenarios:
        msg4 = "━━━ SZENARIEN ━━━"
        for s in scenarios:
            msg4 += f"\n\n📊 Szenario {s.get('id', '?')} ({s.get('probability_pct', '?')}%):"
            msg4 += f"\n  {s.get('description', '')}"
            pi = s.get('portfolio_impact', '')
            if pi:
                msg4 += f"\n  Portfolio: {pi}"
            ci = s.get('composite_impact', '')
            if ci:
                msg4 += f"\n  Composite: {ci}"
            ac = s.get('action', '')
            if ac:
                msg4 += f"\n  Action: {ac}"
        _send_single_telegram(token, chat_id, msg4)

    # =======================================================================
    # MSG 5: Breaking News
    # =======================================================================
    breaking = newsletter.get("breaking_news", [])
    if breaking:
        msg5 = "━━━ BREAKING NEWS ━━━"
        for b in breaking:
            impact = b.get("impact", "?")
            emoji = "🔴" if impact == "HIGH" else "🟠" if impact == "MEDIUM" else "🔵"
            msg5 += f"\n\n{emoji} [{impact}] {b.get('title', '')}"
            msg5 += f"\n  Kategorie: {b.get('category', '')}"
            src = b.get('source', '')
            if src:
                msg5 += f" | Quelle: {src}"
            trans = b.get("portfolio_transmission", {})
            if isinstance(trans, dict):
                assets = trans.get("affected_assets", [])
                if isinstance(assets, list):
                    for a in assets[:3]:
                        if isinstance(a, dict):
                            msg5 += f"\n  → {a.get('asset', '?')}: {a.get('direction', '?')} ({a.get('mechanism', '')})"

        summary = newsletter.get("breaking_news_summary", "")
        if summary:
            msg5 += f"\n\n📋 {summary}"
        _send_single_telegram(token, chat_id, msg5)

    # =======================================================================
    # MSG 6+7: Indikatoren (ausformuliert, investor-ready)
    # =======================================================================
    indicators = newsletter.get("indicators", {})
    all_inds = indicators.get("core", []) + indicators.get("regime_sensitive", [])

    if all_inds:
        msg_ind = "━━━ INDIKATOREN (DETAILLIERT) ━━━"
        regime = regime_ctx.get("v16_regime", "?")
        msg_ind += f"\nAktives Regime: {regime}\n"

        for ind in all_inds:
            key = ind.get("key", "")
            name = ind.get("name", key)
            val = ind.get("value")
            unit = ind.get("unit_label", "")
            norm = ind.get("normalized")
            weight = ind.get("weight", 0)
            status = ind.get("status", "?")
            alert = ind.get("alert", False)
            what = ind.get("what", "")
            thresholds = ind.get("thresholds", "")
            why = ind.get("why_it_matters", "")
            assessment = ind.get("current_assessment", "")

            alert_icon = "⚠️" if alert else "✅"
            val_str = f"{val} {unit}" if val is not None else "N/A"
            norm_str = f"{norm:.0f}/100" if norm is not None else "N/A"

            msg_ind += f"\n{'─' * 30}"
            msg_ind += f"\n{alert_icon} {name}"
            msg_ind += f"\n   Wert: {val_str} | Score: {norm_str} | Gewicht: {weight:.0%}"

            if status == "MISSING":
                msg_ind += "\n   ⚪ Keine Daten verfuegbar"
                continue

            if what:
                msg_ind += f"\n   Was: {what}"
            if thresholds:
                msg_ind += f"\n   Schwellen: {thresholds}"
            if why:
                msg_ind += f"\n   Portfolio-Relevanz: {why}"
            if assessment:
                msg_ind += f"\n   📊 Aktuell: {assessment}"
            msg_ind += ""

        _send_single_telegram(token, chat_id, msg_ind)

    # =======================================================================
    # MSG 8: Intelligence + Catalysts + Epistemic
    # =======================================================================
    intel = newsletter.get("intelligence_digest", {})
    msg8 = "━━━ INTELLIGENCE DIGEST ━━━"
    msg8 += f"\nIC Konsens: {intel.get('ic_net_direction', '?')} (Score: {intel.get('ic_net_score', 0)})"
    msg8 += f"\nThreads: {intel.get('active_threads', 0)} aktiv | {intel.get('threatening_threads', 0)} bedrohlich"
    msg8 += f"\nPre-Mortems HIGH: {intel.get('pre_mortem_high_count', 0)}"
    msg8 += f"\nCadence-Anomalien: {intel.get('cadence_anomalies', 0)}"
    msg8 += f"\nExperten-Dissens: {intel.get('expert_disagreements', 0)}"

    catalysts = newsletter.get("catalysts_48h", [])
    if catalysts:
        msg8 += "\n\n━━━ CATALYSTS 48h ━━━"
        for c in catalysts:
            msg8 += f"\n  [{c.get('impact', '')}] {c.get('date', '')} — {c.get('event', '')}"
            h = c.get('hours_until', '')
            if h:
                msg8 += f" (in {h}h)"
    else:
        msg8 += "\n\nKeine HIGH-Impact Catalysts in 48h"

    epist = newsletter.get("epistemic_status", {})
    msg8 += f"\n\n━━━ EPISTEMIC STATUS ━━━"
    msg8 += f"\nData Quality: {epist.get('data_quality', '?')}"
    msg8 += f"\nConviction: {epist.get('system_conviction', '?')}"
    blind = epist.get("blind_spots", [])
    if blind:
        msg8 += "\nBlind Spots:"
        for bs in blind:
            msg8 += f"\n  • {bs}"

    _send_single_telegram(token, chat_id, msg8)

    # =======================================================================
    # MSG 9: Portfolio + Behavioral + Closing
    # =======================================================================
    portfolio = newsletter.get("portfolio_attribution", {})
    msg9 = "━━━ PORTFOLIO ━━━"
    total = portfolio.get("total_pnl_pct")
    if total is not None:
        msg9 += f"\nTages-P&L: {total:+.2f}%"
    for p in portfolio.get("positions", []):
        pnl = p.get("pnl_pct")
        pnl_str = f" | P&L: {pnl:+.2f}%" if pnl is not None else ""
        msg9 += f"\n  {p.get('asset', '?')}: {p.get('weight_pct', 0)}%{pnl_str}"

    behav = newsletter.get("behavioral", {})
    msg9 += f"\n\n━━━ BEHAVIORAL SAFEGUARDS ━━━"
    msg9 += f"\nSystem Action: {behav.get('system_action', '?')}"
    msg9 += f"\nInaction: {behav.get('inaction_tracker', {}).get('status', '?')}"
    for a in behav.get("anchoring_alerts", []):
        msg9 += f"\n🧠 {a.get('asset', '?')} ({a.get('weight_pct', 0)}%): {a.get('question', '')}"

    contrarian = newsletter.get("contrarian_check")
    if contrarian:
        msg9 += f"\n\n🔄 CONTRARIAN: {contrarian}"

    _send_single_telegram(token, chat_id, msg9)

    # =======================================================================
    # MSG 10: Crypto Circle (V1.1)
    # =======================================================================
    crypto = newsletter.get("crypto_briefing")
    if crypto and crypto.get("available"):
        c = crypto
        ens_daily = c.get('ensemble_daily')
        ens_weekly = c.get('ensemble_weekly')
        changed_icon = " ← GEAENDERT" if c.get('ensemble_changed') else ""

        msg10 = "━━━ CRYPTO CIRCLE ━━━"
        msg10 += f"\nBTC: ${c.get('btc_price', 0):,.0f}"
        msg10 += f"\nEnsemble: {ens_daily:.2f}" if ens_daily is not None else ""
        msg10 += f" (Weekly: {ens_weekly:.2f})" if ens_weekly is not None else ""
        msg10 += changed_icon
        msg10 += f"\nMomentum: 1M={'✅' if c.get('mom_1M') else '❌'} " \
                 f"3M={'✅' if c.get('mom_3M') else '❌'} " \
                 f"6M={'✅' if c.get('mom_6M') else '❌'} " \
                 f"12M={'✅' if c.get('mom_12M') else '❌'}"

        if c.get('below_200wma'):
            wma = c.get('wma_200', 0)
            msg10 += f"\n⚠️ BTC UNTER 200WMA (${wma:,.0f})"
            if c.get('below_wma_changed'):
                msg10 += " ← NEU"

        phase = c.get('phase')
        phase_name = c.get('phase_name', '?')
        msg10 += f"\nPhase: {phase} ({phase_name})"

        w_alloc = c.get('weekly_alloc_total')
        if w_alloc is not None:
            msg10 += f"\nAllokation: {w_alloc:.0%}"
            msg10 += f"  BTC={c.get('weekly_btc', 0):.1%}" \
                     f" ETH={c.get('weekly_eth', 0):.1%}" \
                     f" SOL={c.get('weekly_sol', 0):.1%}" \
                     f" Cash={c.get('weekly_cash', 0):.1%}"

        alert_count = c.get('alert_count', 0)
        alerts = c.get('alerts', [])
        if alert_count > 0:
            msg10 += f"\n\n🔔 {alert_count} Alert(s):"
            for a in alerts:
                icon = "🔴" if a.get('severity') == 'HIGH' else "🟡"
                msg10 += f"\n  {icon} {a.get('type', '?')}: {a.get('message', '')}"
        else:
            msg10 += "\n\n✅ Keine Crypto Alerts"

        _send_single_telegram(token, chat_id, msg10)

    # =======================================================================
    # MSG 11: Closing
    # =======================================================================
    msg_close = f"━━━━━━━━━━━━━━━━━━━━━"
    msg_close += f"\n🏁 Baldur Creek Capital — {newsletter.get('anchor_type', '?')} — {d}"

    _send_single_telegram(token, chat_id, msg_close)

    msg_count = 10 if (crypto and crypto.get("available")) else 9
    logger.info(f"Telegram full newsletter sent ({msg_count + 1} messages)")
    return True


def send_telegram_error():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        import requests as http_requests
        http_requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": f"⚠️ Newsletter fehlgeschlagen ({date.today().isoformat()}). Manueller Check."},
            timeout=15,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Drive upload
# ---------------------------------------------------------------------------

def upload_to_drive(newsletter_path):
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
            tmp_path, scopes=["https://www.googleapis.com/auth/drive"])
        os.unlink(tmp_path)

        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        from .config import DRIVE_ROOT_FOLDER
        filename = os.path.basename(newsletter_path)
        media = MediaFileUpload(newsletter_path, mimetype="application/json")
        uploaded = service.files().create(
            body={"name": filename, "parents": [DRIVE_ROOT_FOLDER]},
            media_body=media, fields="id"
        ).execute()
        logger.info(f"Newsletter uploaded to Drive: {uploaded.get('id')}")
        return uploaded.get("id")
    except Exception as e:
        logger.error(f"Drive upload failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Update latest.json — FULL newsletter data for Dashboard Briefing Circle
# ---------------------------------------------------------------------------

def update_latest_json(newsletter, latest_json_path):
    """Write the full newsletter dict into latest.json for the Briefing Circle."""
    if not os.path.exists(latest_json_path):
        logger.warning(f"latest.json not found — skipping update")
        return
    try:
        with open(latest_json_path, "r", encoding="utf-8") as f:
            latest = json.load(f)

        # Write the full newsletter — BriefingDetail.jsx reads all these fields.
        # Only strip the hash (not needed in dashboard) to keep it clean.
        nl = dict(newsletter)
        nl.pop("hash", None)
        latest["newsletter"] = nl

        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(latest, f, indent=2, ensure_ascii=False, default=str)
        logger.info("latest.json updated with FULL newsletter block")
    except Exception as e:
        logger.error(f"Failed to update latest.json: {e}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Daily Briefing System — Step 0u")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-telegram", action="store_true")
    parser.add_argument("--skip-drive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--latest", type=str, default=None)
    parser.add_argument("--creds", type=str, default=None)
    args = parser.parse_args()

    today = date.today()
    fmt = NEWSLETTER_FORMATS.get(today.weekday(), "DAILY")
    logger.info(f"=== DAILY BRIEFING SYSTEM — {today.isoformat()} ({fmt}) ===")

    if not args.force and check_already_run():
        logger.info("Already run today — skipping (use --force)")
        return 0

    try:
        logger.info("STEP 1: Collecting data...")
        indicator_values, pipeline_data = collect_all_data(args.latest, args.creds)
        regime = pipeline_data.get("v16_regime", "UNKNOWN")
        logger.info(f"  Regime: {regime}, Indicators: {len(indicator_values)}, Coherence: {pipeline_data.get('pipeline_coherence_pct')}%")

        logger.info("STEP 2: Computing composite scores...")
        composite = compute_composite_scores(indicator_values, regime, pipeline_data)
        logger.info(f"  T={composite['tactical']['score']} {composite['tactical']['zone']}, P={composite['positional']['score']}, S={composite['structural']['score']}")

        news_result = None
        if not args.skip_news:
            logger.info("STEP 3: Scanning breaking news...")
            news_result = run_news_scanner(pipeline_data.get("v16_weights", {}))
            logger.info(f"  HIGH: {news_result.get('high_impact_count', 0)}, MED: {news_result.get('medium_impact_count', 0)}")
        else:
            logger.info("STEP 3: Skipped")
            news_result = {"hits": [], "high_impact_count": 0, "summary": "Skipped."}

        logger.info("STEP 4: Assembling newsletter...")
        newsletter = assemble_newsletter(composite, pipeline_data, news_result, indicator_values)
        logger.info(f"  Format: {newsletter.get('format')}, Anchor: {newsletter.get('anchor_type')}, Blocks: {sum(1 for v in newsletter.values() if v)}")

        logger.info("STEP 5: Saving...")
        newsletter_path = save_newsletter(newsletter)

        latest_path = args.latest or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "dashboard", "latest.json")
        if os.path.exists(latest_path):
            update_latest_json(newsletter, latest_path)

        if not args.skip_telegram:
            logger.info("STEP 6: Telegram...")
            send_telegram_message(newsletter)
        else:
            logger.info("STEP 6: Skipped")

        if not args.skip_drive:
            logger.info("STEP 7: Drive upload...")
            upload_to_drive(newsletter_path)
        else:
            logger.info("STEP 7: Skipped")

        mark_run_complete()
        logger.info(f"=== COMPLETE — {newsletter.get('anchor_type')} ===")
        return 0

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        send_telegram_error()
        return 1

if __name__ == "__main__":
    sys.exit(main())
