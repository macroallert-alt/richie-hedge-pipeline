"""
Daily Briefing System — Newsletter Assembler
Baldur Creek Capital | Step 0u
Based on: DAILY_BRIEFING_SYSTEM_SPEC_TEIL1.md §3.3, TEIL2 §12.1

Builds the full newsletter_YYYY-MM-DD.json with all 15 blocks.
Deterministic blocks are built from data; narrative blocks use LLM.

V1.1: Added build_crypto_block() for Crypto Circle daily check integration.
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime, timezone

logger = logging.getLogger("assembler")

try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .config import (
    CLAUDE_MODEL,
    NEWSLETTER_MAX_TOKENS,
    RISK_HEATMAP_MAPPING,
    RISK_HEATMAP_DESCRIPTIONS,
    RISK_FACTORS,
    NEWSLETTER_FORMATS,
    HISTORY_DIR,
    INDICATOR_DESCRIPTIONS,
    INDICATOR_NORMALIZATION,
    get_composite_zone,
    get_anchor_type,
)


# ---------------------------------------------------------------------------
# Deterministic Blocks
# ---------------------------------------------------------------------------

def build_composite_block(composite_result):
    return {
        "tactical": {
            "score": composite_result["tactical"]["score"],
            "zone": composite_result["tactical"]["zone"],
            "velocity": composite_result["tactical"]["velocity"],
            "acceleration": composite_result["tactical"]["acceleration"],
        },
        "positional": {
            "score": composite_result["positional"]["score"],
            "zone": composite_result["positional"]["zone"],
            "velocity": composite_result["positional"]["velocity"],
            "acceleration": composite_result["positional"]["acceleration"],
        },
        "structural": {
            "score": composite_result["structural"]["score"],
            "zone": composite_result["structural"]["zone"],
            "velocity": composite_result["structural"]["velocity"],
            "acceleration": composite_result["structural"]["acceleration"],
        },
    }


def build_data_integrity_block(composite_result):
    return composite_result.get("data_integrity", {})


def build_regime_context_block(pipeline_data):
    return {
        "v16_regime": pipeline_data.get("v16_regime", "UNKNOWN"),
        "regime_duration_days": pipeline_data.get("regime_duration_days", 0),
        "previous_regime": None,
        "system_regime": pipeline_data.get("system_regime", "UNKNOWN"),
        "fragility_state": pipeline_data.get("fragility_state", "UNKNOWN"),
        "shift_countdown": {"available": False},
    }


def build_portfolio_attribution_block(pipeline_data):
    top5 = pipeline_data.get("v16_top5", [])
    positions = []
    for p in top5:
        ticker = p.get("ticker", "")
        weight = p.get("weight", 0)
        positions.append({
            "asset": ticker,
            "weight_pct": round(weight * 100, 1),
            "pnl_pct": None,
            "pnl_contribution": None,
            "held_days": None,
        })
    return {
        "total_pnl_pct": None,
        "positions": positions,
        "effective_independent_bets": None,
        "peer_comparison": None,
    }


def build_risk_heatmap_block(pipeline_data):
    top5 = pipeline_data.get("v16_top5", [])
    top3_tickers = [p["ticker"] for p in top5[:3]]
    matrix = []
    for ticker in top3_tickers:
        row = []
        mapping = RISK_HEATMAP_MAPPING.get(ticker, {})
        for factor in RISK_FACTORS:
            severity = mapping.get(factor, "UNKNOWN")
            row.append(severity)
        matrix.append(row)
    return {
        "positions": top3_tickers,
        "risk_factors": RISK_FACTORS,
        "matrix": matrix,
    }


def build_indicators_block(indicator_values, composite_result):
    core = []
    regime_sensitive = []

    for detail in composite_result.get("tactical", {}).get("details", []):
        key = detail["indicator"]
        raw = detail["raw_value"]
        normalized = detail["normalized"]
        status_flag = "OK"
        if normalized is not None:
            if normalized < 15:
                status_flag = "CRITICAL"
            elif normalized < 30:
                status_flag = "WARNING"

        # Get static description
        desc = INDICATOR_DESCRIPTIONS.get(key, {})

        entry = {
            "key": key,
            "name": desc.get("name", key),
            "value": raw,
            "unit_label": desc.get("unit_label", ""),
            "normalized": normalized,
            "weight": detail["weight"],
            "status": detail["status"],
            "alert": status_flag in ("WARNING", "CRITICAL"),
            "what": desc.get("what", ""),
            "thresholds": desc.get("thresholds", ""),
            "why_it_matters": desc.get("why_it_matters", ""),
            # LLM fills this:
            "current_assessment": "",
        }

        if len(core) < 5:
            core.append(entry)
        else:
            regime_sensitive.append(entry)

    return {
        "core": core,
        "regime_sensitive": regime_sensitive,
        "watchlist_triggered": [],
    }


def build_pipeline_coherence_block(pipeline_data):
    subsystems = {
        "v16": pipeline_data.get("v16_regime", "UNKNOWN"),
        "market_analyst": pipeline_data.get("system_regime", "UNKNOWN"),
        "ic_consensus": "BEARISH" if pipeline_data.get("ic_net_bearish_score", 0) < -2 else
                        "BULLISH" if pipeline_data.get("ic_net_bearish_score", 0) > 2 else "NEUTRAL",
        "risk_officer": pipeline_data.get("risk_status", "UNKNOWN"),
        "cio": pipeline_data.get("briefing_type", "UNKNOWN"),
        "execution_advisor": pipeline_data.get("execution_level", "UNKNOWN"),
    }
    divergences = []
    if pipeline_data.get("regime_conflict"):
        divergences.append({
            "type": "REGIME_CONFLICT",
            "detail": f"V16 {subsystems['v16']} vs MA {subsystems['market_analyst']}",
        })
    return {
        "score": pipeline_data.get("pipeline_coherence_pct", 0),
        "subsystems": subsystems,
        "divergences": divergences,
    }


def build_epistemic_block(pipeline_data):
    return {
        "data_quality": pipeline_data.get("data_quality", "UNKNOWN"),
        "system_conviction": pipeline_data.get("system_conviction", "UNKNOWN"),
        "stale_sources": [],
        "blind_spots": [],
        "signal_freshness": {
            "regime_age_days": pipeline_data.get("regime_duration_days", 0),
        },
    }


def build_intelligence_digest_block(pipeline_data):
    return {
        "ic_net_direction": "BEARISH" if pipeline_data.get("ic_net_bearish_score", 0) < -2 else
                           "BULLISH" if pipeline_data.get("ic_net_bearish_score", 0) > 2 else "NEUTRAL",
        "ic_net_score": pipeline_data.get("ic_net_bearish_score", 0),
        "pre_mortem_high_count": pipeline_data.get("pre_mortem_high_count", 0),
        "active_threads": pipeline_data.get("active_thread_count", 0),
        "threatening_threads": pipeline_data.get("threatening_thread_count", 0),
        "cadence_anomalies": pipeline_data.get("cadence_anomaly_count", 0),
        "expert_disagreements": pipeline_data.get("expert_disagreement_count", 0),
    }


def build_catalysts_block(pipeline_data):
    cal = pipeline_data.get("calendar_upcoming", [])
    catalysts = []
    for c in cal:
        hours = c.get("hours_until")
        days = c.get("days_until")
        if hours is not None and hours <= 48:
            catalysts.append(c)
        elif days is not None and days <= 2:
            catalysts.append(c)
    return catalysts[:5]


def build_behavioral_block(pipeline_data):
    top5 = pipeline_data.get("v16_top5", [])
    anchoring = []
    for p in top5:
        ticker = p.get("ticker", "")
        weight = round(p.get("weight", 0) * 100, 1)
        if weight > 20:
            anchoring.append({
                "asset": ticker,
                "weight_pct": weight,
                "question": f"Wuerdest du {ticker} heute NEU kaufen bei {weight}%?",
            })
    return {
        "anchoring_alerts": anchoring,
        "inaction_tracker": {
            "status": "WARTEN_IST_STRATEGIE" if pipeline_data.get("execution_level") != "EXECUTE" else "SYSTEM_ACTIVE",
        },
        "system_action": pipeline_data.get("execution_level", "UNKNOWN"),
    }


# ---------------------------------------------------------------------------
# Crypto Circle Block (V1.1) — liest crypto_daily_check.json
# ---------------------------------------------------------------------------

def build_crypto_block():
    """Lese crypto_daily_check.json und baue Crypto-Sektion fuer Newsletter.

    Datei wird von daily_risk_check.py geschrieben (Step 0y Crypto Daily).
    Wenn die Datei nicht existiert oder leer ist: leerer Block.
    """
    # Suche crypto_daily_check.json im step_0y_crypto/data/ Verzeichnis
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base, 'step_0y_crypto', 'data', 'crypto_daily_check.json')

    if not os.path.exists(json_path):
        logger.info("crypto_daily_check.json nicht gefunden — Crypto-Sektion leer")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Prüfe ob Daten von heute sind (oder maximal 1 Tag alt)
        check_date = data.get('date', '')
        today = date.today().isoformat()
        if check_date and check_date != today:
            logger.warning(f"crypto_daily_check.json ist von {check_date}, nicht heute ({today})")
            # Trotzdem einlesen — lieber alte Daten als keine

        ensemble = data.get('ensemble', {})
        bonus = data.get('bottom_bonus', {})
        weekly = data.get('weekly_signal', {})
        alloc = weekly.get('allocation', {})
        alerts = data.get('alerts', [])

        block = {
            'available': True,
            'date': check_date,
            'btc_price': data.get('btc_price'),
            'ensemble_daily': ensemble.get('daily'),
            'ensemble_weekly': ensemble.get('weekly'),
            'ensemble_changed': ensemble.get('changed', False),
            'mom_1M': ensemble.get('mom_1M', False),
            'mom_3M': ensemble.get('mom_3M', False),
            'mom_6M': ensemble.get('mom_6M', False),
            'mom_12M': ensemble.get('mom_12M', False),
            'below_200wma': bonus.get('active', False),
            'wma_200': bonus.get('wma_200'),
            'below_wma_changed': bonus.get('changed', False),
            'btc_dominance': data.get('btc_dominance', {}).get('daily'),
            'phase': weekly.get('phase'),
            'phase_name': weekly.get('phase_name'),
            'weekly_alloc_total': alloc.get('total'),
            'weekly_btc': alloc.get('btc'),
            'weekly_eth': alloc.get('eth'),
            'weekly_sol': alloc.get('sol'),
            'weekly_cash': alloc.get('cash'),
            'alert_count': data.get('alert_count', 0),
            'alerts': alerts,
        }

        logger.info(f"Crypto block: Ensemble={ensemble.get('daily')}, "
                     f"Alerts={len(alerts)}, Changed={ensemble.get('changed')}")
        return block

    except Exception as e:
        logger.error(f"Crypto block ERR: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM Narrative Blocks (extended with indicator assessments + heatmap)
# ---------------------------------------------------------------------------

def _build_indicator_context_for_llm(indicator_values, composite_result):
    """Build a compact indicator summary for the LLM to generate assessments."""
    lines = []
    for detail in composite_result.get("tactical", {}).get("details", []):
        key = detail["indicator"]
        raw = detail["raw_value"]
        normalized = detail["normalized"]
        desc = INDICATOR_DESCRIPTIONS.get(key, {})
        name = desc.get("name", key)
        unit = desc.get("unit_label", "")
        if raw is not None:
            lines.append(f"  {key} ({name}): {raw} {unit} → Score {normalized}/100")
    return "\n".join(lines) if lines else "  Keine Indikatordaten verfuegbar"


def _build_heatmap_context_for_llm(pipeline_data):
    """Build risk heatmap summary for LLM to contextualize."""
    top5 = pipeline_data.get("v16_top5", [])
    top3 = [p["ticker"] for p in top5[:3]]
    lines = []
    for ticker in top3:
        weight = next((round(p["weight"] * 100, 1) for p in top5 if p["ticker"] == ticker), 0)
        hm = RISK_HEATMAP_DESCRIPTIONS.get(ticker, {})
        for factor in RISK_FACTORS:
            entry = hm.get(factor, {})
            sev = entry.get("severity", "?")
            lines.append(f"  {ticker} ({weight}%) × {factor}: {sev}")
    return "\n".join(lines) if lines else "  Keine Heatmap-Daten"


def build_llm_prompt(composite_result, pipeline_data, news_result, indicator_values):
    """
    Build system + user prompt for the LLM Newsletter call.
    Extended: now also generates indicator_assessments and heatmap_assessments.
    """
    tact = composite_result["tactical"]
    pos = composite_result["positional"]
    struct = composite_result["structural"]

    top5 = pipeline_data.get("v16_top5", [])
    positions_str = ", ".join(
        f"{p['ticker']} {round(p['weight']*100,1)}%"
        for p in top5 if p.get("weight", 0) > 0
    )

    warnings_str = "\n".join(
        f"  - {w['description']} ({w['penalty']})"
        for w in composite_result.get("warning_triggers", [])
    ) or "  Keine"

    news_str = "Keine High-Impact News."
    if news_result and news_result.get("hits"):
        news_lines = []
        for h in news_result["hits"][:5]:
            news_lines.append(f"  [{h['impact']}] {h['title'][:80]} — {h.get('source', '')}")
        news_str = "\n".join(news_lines)

    cal = pipeline_data.get("calendar_upcoming", [])
    events_str = "\n".join(
        f"  {c.get('date', '')} {c.get('event', '')} [{c.get('impact', '')}]"
        for c in cal[:5]
    ) or "  Keine Events"

    alerts = pipeline_data.get("risk_alerts", [])
    alerts_str = "\n".join(
        f"  [{a.get('severity', '')}] {a.get('check_id', '')}: {a.get('message', '')[:80]}"
        for a in alerts[:5]
    ) or "  Keine"

    raw_digest = pipeline_data.get("cio_digest") or ""
    if isinstance(raw_digest, dict):
        digest = " ".join(str(v) for v in raw_digest.values())[:500]
    else:
        digest = str(raw_digest)[:500]

    indicator_ctx = _build_indicator_context_for_llm(indicator_values, composite_result)
    heatmap_ctx = _build_heatmap_context_for_llm(pipeline_data)

    system_prompt = """Du bist der Newsletter-Engine von Baldur Creek Capital.
Sprache: Deutsch. Tonalitaet: Sachlich-direkt, operativ, ehrlich.
Jede Zeile muss die Frage beantworten: "Was bedeutet das fuer mein Buch?"
Mindestens 50% der Inhalte muessen Risiken und Gegenargumente zeigen.
Schreibe so dass ein Dritter ohne Vorwissen versteht was gemeint ist.

Antworte NUR als JSON-Objekt mit diesen Feldern:
{
  "one_thing": "Ein Satz — was heute zaehlt, mit Portfolio-Bezug",
  "regime_interpretation": "2-3 Saetze zum Regime-Kontext, verstaendlich fuer Dritte",
  "scenarios": [
    {"id": "A", "probability_pct": 45, "description": "2-3 Saetze was passiert", "portfolio_impact": "Konkret: welche Positionen wie betroffen", "composite_impact": "+/- X Punkte", "action": "NONE/MONITOR/TRADE"},
    {"id": "B", "probability_pct": 35, "description": "...", "portfolio_impact": "...", "composite_impact": "...", "action": "..."},
    {"id": "C", "probability_pct": 20, "description": "...", "portfolio_impact": "...", "composite_impact": "...", "action": "..."}
  ],
  "against_you": [
    {"asset": "HYG", "top_risk": "Konkretes Risiko", "probability_pct": 35, "mechanism": "Wie genau wirkt es auf diese Position"}
  ],
  "if_wrong_summary": "1-2 Saetze: wahrscheinlichster Schaden heute, konkret mit Zahlen",
  "blind_spots": ["Was das System nicht sieht — konkret benennen"],
  "contrarian_note": null,
  "indicator_assessments": {
    "INDIKATOR_KEY": "2-3 Saetze: Wie ist der aktuelle Wert im heutigen Kontext einzuordnen? Was bedeutet er konkret fuer das Portfolio? Gibt es Handlungsbedarf?"
  },
  "heatmap_assessments": {
    "TICKER__RISK_FACTOR": "1-2 Saetze: Wie relevant ist dieser Risikofaktor HEUTE fuer diese Position, gegeben aktuelle Nachrichten und Daten?"
  }
}

Fuer indicator_assessments: Generiere einen Eintrag pro Indikator der Daten hat (nicht fuer MISSING).
Fuer heatmap_assessments: Generiere einen Eintrag pro Position-Risikofaktor-Kombination der Top-3-Positionen. Key-Format: "TICKER__Risikofaktor" (z.B. "HYG__Credit Spreads").
Keine Markdown-Formatierung, keine Erklaerung, NUR das JSON-Objekt."""

    user_prompt = f"""DATEN FUER HEUTE ({date.today().isoformat()}):

COMPOSITE SCORES:
  TACTICAL:   {tact['score']} {tact['zone']} (vel {tact['velocity']:+.0f}, acc {tact['acceleration']:+.0f})
  POSITIONAL: {pos['score']} {pos['zone']} (vel {pos['velocity']:+.0f}, acc {pos['acceleration']:+.0f})
  STRUCTURAL: {struct['score']} {struct['zone']} (vel {struct['velocity']:+.0f}, acc {struct['acceleration']:+.0f})

V16 REGIME: {pipeline_data.get('v16_regime', 'UNKNOWN')} (seit {pipeline_data.get('regime_duration_days', '?')} Tagen)
PORTFOLIO: {positions_str}
RISK AMPEL: {pipeline_data.get('risk_ampel', 'UNKNOWN')}
EXECUTION: {pipeline_data.get('execution_level', 'UNKNOWN')}

INDIKATOREN (aktuell):
{indicator_ctx}

RISK HEATMAP (Position x Faktor: Severity):
{heatmap_ctx}

WARNING TRIGGERS:
{warnings_str}

RISK ALERTS:
{alerts_str}

EVENTS (naechste 48h):
{events_str}

BREAKING NEWS:
{news_str}

IC KONSENS: net bearish score = {pipeline_data.get('ic_net_bearish_score', 0)}
PIPELINE COHERENCE: {pipeline_data.get('pipeline_coherence_pct', 0)}%
PRE-MORTEM HIGH RISK: {pipeline_data.get('pre_mortem_high_count', 0)}

CIO DIGEST (Auszug):
{digest}

Generiere das Newsletter-JSON mit indicator_assessments und heatmap_assessments."""

    return system_prompt, user_prompt


def call_newsletter_llm(system_prompt, user_prompt, api_key=None):
    if not HAS_REQUESTS:
        logger.error("requests not available for LLM call")
        return None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.error("No ANTHROPIC_API_KEY")
        return None

    try:
        resp = http_requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": NEWSLETTER_MAX_TOKENS,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=90,
        )
        resp.raise_for_status()
        result = resp.json()

        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        if text.endswith("```"):
            text = text[:-3]

        parsed = json.loads(text.strip())
        logger.info("LLM newsletter blocks generated successfully")
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"LLM response not valid JSON: {e}")
        logger.debug(f"Raw LLM response: {text[:500]}")
        return None
    except Exception as e:
        logger.error(f"LLM newsletter call failed: {e}")
        return None


def fallback_narrative_blocks(pipeline_data, composite_result):
    tact = composite_result["tactical"]
    regime = pipeline_data.get("v16_regime", "UNKNOWN")
    return {
        "one_thing": f"Composite {tact['score']} {tact['zone']}. Regime {regime}. LLM nicht verfuegbar — manueller Check empfohlen.",
        "regime_interpretation": f"V16 Regime: {regime}. Keine LLM-Interpretation verfuegbar.",
        "scenarios": [],
        "against_you": [],
        "if_wrong_summary": "LLM nicht verfuegbar — keine Szenario-Analyse.",
        "blind_spots": ["LLM-Analyse nicht verfuegbar"],
        "contrarian_note": None,
        "indicator_assessments": {},
        "heatmap_assessments": {},
    }


# ---------------------------------------------------------------------------
# MAIN: Assemble full newsletter JSON
# ---------------------------------------------------------------------------

def assemble_newsletter(composite_result, pipeline_data, news_result,
                        indicator_values, api_key=None):
    today = date.today()
    now = datetime.now(timezone.utc)
    fmt = NEWSLETTER_FORMATS.get(today.weekday(), "DAILY")

    # Deterministic blocks
    composite_block = build_composite_block(composite_result)
    data_integrity = build_data_integrity_block(composite_result)
    regime_context = build_regime_context_block(pipeline_data)
    portfolio_attr = build_portfolio_attribution_block(pipeline_data)
    risk_heatmap = build_risk_heatmap_block(pipeline_data)
    indicators = build_indicators_block(indicator_values, composite_result)
    coherence = build_pipeline_coherence_block(pipeline_data)
    epistemic = build_epistemic_block(pipeline_data)
    intel_digest = build_intelligence_digest_block(pipeline_data)
    catalysts = build_catalysts_block(pipeline_data)
    behavioral = build_behavioral_block(pipeline_data)
    crypto_block = build_crypto_block()

    # LLM narrative blocks
    system_prompt, user_prompt = build_llm_prompt(
        composite_result, pipeline_data, news_result, indicator_values
    )
    llm_blocks = call_newsletter_llm(system_prompt, user_prompt, api_key)
    if llm_blocks is None:
        logger.warning("LLM failed — using fallback narrative blocks")
        llm_blocks = fallback_narrative_blocks(pipeline_data, composite_result)

    # Merge LLM indicator assessments into indicator block
    assessments = llm_blocks.get("indicator_assessments", {})
    for ind in indicators.get("core", []) + indicators.get("regime_sensitive", []):
        key = ind.get("key", "")
        if key in assessments:
            ind["current_assessment"] = assessments[key]

    # Merge LLM heatmap assessments into risk heatmap block
    heatmap_assessments = llm_blocks.get("heatmap_assessments", {})
    risk_heatmap["assessments"] = heatmap_assessments

    # Breaking news
    breaking = []
    if news_result:
        for h in news_result.get("hits", [])[:5]:
            breaking.append({
                "category": h.get("category", ""),
                "title": h.get("title", ""),
                "impact": h.get("impact", "LOW"),
                "source": h.get("source", ""),
                "portfolio_transmission": h.get("portfolio_transmission", {}),
            })

    # Assemble
    newsletter = {
        "schema_version": "1.2",
        "date": today.isoformat(),
        "weekday": today.strftime("%A"),
        "format": fmt,
        "generated_at": now.isoformat(),
        "one_thing": llm_blocks.get("one_thing", ""),
        "composite_scores": composite_block,
        "data_integrity": data_integrity,
        "regime_context": regime_context,
        "regime_interpretation": llm_blocks.get("regime_interpretation", ""),
        "portfolio_attribution": portfolio_attr,
        "risk_heatmap": risk_heatmap,
        "indicators": indicators,
        "pipeline_coherence": coherence,
        "scenarios": llm_blocks.get("scenarios", []),
        "behavioral": behavioral,
        "against_you": {
            "positions": llm_blocks.get("against_you", []),
            "if_wrong_summary": llm_blocks.get("if_wrong_summary", ""),
        },
        "epistemic_status": {
            **epistemic,
            "blind_spots": llm_blocks.get("blind_spots", []),
        },
        "catalysts_48h": catalysts,
        "intelligence_digest": intel_digest,
        "liquidity_pipeline_7d": {
            "available": False,
            "placeholder": "Liquidity Pipeline ab Phase 5 verfuegbar",
        },
        "history_enrichment": {
            "available": False,
            "placeholder": "History Enrichment ab April 2026 verfuegbar",
        },
        "breaking_news": breaking,
        "breaking_news_summary": news_result.get("summary", "") if news_result else "",
        "warning_triggers": composite_result.get("warning_triggers", []),
        "contrarian_check": llm_blocks.get("contrarian_note") if fmt == "DAILY_CONTRARIAN" else None,
        "prediction_log": {
            "todays_predictions": _extract_predictions(llm_blocks),
        },
        "anchor_type": get_anchor_type(
            composite_result["tactical"]["score"],
            (news_result or {}).get("high_impact_count", 0) > 0,
            pipeline_data.get("risk_emergency_active", False),
            False,
        ),
        # V1.1: Crypto Circle Daily Check
        "crypto_briefing": crypto_block,
    }

    content_for_hash = json.dumps(newsletter, sort_keys=True, default=str)
    newsletter["hash"] = "sha256:" + hashlib.sha256(content_for_hash.encode()).hexdigest()

    return newsletter


def _extract_predictions(llm_blocks):
    predictions = []
    for scenario in llm_blocks.get("scenarios", []):
        if scenario.get("probability_pct", 0) >= 30:
            predictions.append({
                "prediction": scenario.get("description", ""),
                "probability_pct": scenario.get("probability_pct", 0),
                "basis": f"Scenario {scenario.get('id', '?')}",
            })
    return predictions


# ---------------------------------------------------------------------------
# Save newsletter
# ---------------------------------------------------------------------------

def save_newsletter(newsletter, output_dir=None):
    if output_dir is None:
        output_dir = HISTORY_DIR
    os.makedirs(output_dir, exist_ok=True)
    d = newsletter.get("date", date.today().isoformat())
    filename = f"newsletter_{d}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(newsletter, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Newsletter saved: {path}")
    return path
