"""
Daily Briefing System — Newsletter Assembler
Baldur Creek Capital | Step 0u
Based on: DAILY_BRIEFING_SYSTEM_SPEC_TEIL1.md §3.3, TEIL2 §12.1

Builds the full newsletter_YYYY-MM-DD.json with all 15 blocks.
Deterministic blocks are built from data; narrative blocks use LLM.
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
    RISK_FACTORS,
    NEWSLETTER_FORMATS,
    HISTORY_DIR,
    get_composite_zone,
    get_anchor_type,
)


# ---------------------------------------------------------------------------
# Deterministic Blocks
# ---------------------------------------------------------------------------

def build_composite_block(composite_result):
    """Block 2: Composite Scores (deterministic)."""
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
    """Block 2 sub: Data Integrity."""
    return composite_result.get("data_integrity", {})


def build_regime_context_block(pipeline_data):
    """Block 3: Regime Context (deterministic)."""
    return {
        "v16_regime": pipeline_data.get("v16_regime", "UNKNOWN"),
        "regime_duration_days": pipeline_data.get("regime_duration_days", 0),
        "previous_regime": None,  # Would need history; placeholder
        "system_regime": pipeline_data.get("system_regime", "UNKNOWN"),
        "fragility_state": pipeline_data.get("fragility_state", "UNKNOWN"),
        "shift_countdown": {"available": False},
    }


def build_portfolio_attribution_block(pipeline_data):
    """Block 4: Portfolio Attribution (deterministic)."""
    top5 = pipeline_data.get("v16_top5", [])
    positions = []
    for p in top5:
        ticker = p.get("ticker", "")
        weight = p.get("weight", 0)
        positions.append({
            "asset": ticker,
            "weight_pct": round(weight * 100, 1),
            "pnl_pct": None,          # Needs price data; placeholder
            "pnl_contribution": None,
            "held_days": None,
        })

    return {
        "total_pnl_pct": None,  # Placeholder — needs daily price data
        "positions": positions,
        "effective_independent_bets": None,  # Phase 5 feature
        "peer_comparison": None,             # Phase 5 feature
    }


def build_risk_heatmap_block(pipeline_data):
    """Block 5: Risk Heatmap (deterministic)."""
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
    """Block 6: Indicator Dashboard (deterministic)."""
    core = []
    regime_sensitive = []

    for detail in composite_result.get("tactical", {}).get("details", []):
        key = detail["indicator"]
        raw = detail["raw_value"]
        normalized = detail["normalized"]
        status_flag = "OK"
        if normalized is not None:
            if normalized < 30:
                status_flag = "WARNING"
            elif normalized < 15:
                status_flag = "CRITICAL"

        entry = {
            "name": key,
            "value": raw,
            "normalized": normalized,
            "weight": detail["weight"],
            "status": detail["status"],
            "alert": status_flag in ("WARNING", "CRITICAL"),
        }

        # First 5 = core, rest = regime sensitive
        if len(core) < 5:
            core.append(entry)
        else:
            regime_sensitive.append(entry)

    return {
        "core": core,
        "regime_sensitive": regime_sensitive,
        "watchlist_triggered": [],  # Filled later if needed
    }


def build_pipeline_coherence_block(pipeline_data):
    """Block 7: Pipeline Coherence (deterministic)."""
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
    """Block 11: Epistemic Status (deterministic)."""
    # Stale sources from IC source cards
    stale = []
    # Would need source_cards from intel; use what we have
    return {
        "data_quality": pipeline_data.get("data_quality", "UNKNOWN"),
        "system_conviction": pipeline_data.get("system_conviction", "UNKNOWN"),
        "stale_sources": stale,
        "blind_spots": [],  # Filled by LLM
        "signal_freshness": {
            "regime_age_days": pipeline_data.get("regime_duration_days", 0),
        },
    }


def build_intelligence_digest_block(pipeline_data):
    """Block 13: Intelligence Digest (deterministic)."""
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
    """Block 12: Catalysts 48h (deterministic)."""
    cal = pipeline_data.get("calendar_upcoming", [])
    # Filter to HIGH impact within 48h
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
    """Block 9: Behavioral Safeguards (deterministic skeleton)."""
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
# LLM Narrative Blocks
# ---------------------------------------------------------------------------

def build_llm_prompt(composite_result, pipeline_data, news_result, indicator_values):
    """
    Build the system prompt + user prompt for the LLM Newsletter call.
    The LLM generates: one_thing, scenarios, against_you, regime_interpretation.
    """
    # Compact data summary for LLM
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

    # Breaking news
    news_str = "Keine High-Impact News."
    if news_result and news_result.get("hits"):
        news_lines = []
        for h in news_result["hits"][:5]:
            news_lines.append(f"  [{h['impact']}] {h['title'][:80]} — {h.get('source', '')}")
        news_str = "\n".join(news_lines)

    # Events
    cal = pipeline_data.get("calendar_upcoming", [])
    events_str = "\n".join(
        f"  {c.get('date', '')} {c.get('event', '')} [{c.get('impact', '')}]"
        for c in cal[:5]
    ) or "  Keine Events"

    # Risk alerts
    alerts = pipeline_data.get("risk_alerts", [])
    alerts_str = "\n".join(
        f"  [{a.get('severity', '')}] {a.get('check_id', '')}: {a.get('message', '')[:80]}"
        for a in alerts[:5]
    ) or "  Keine"

    # CIO digest (first 500 chars)
    raw_digest = pipeline_data.get("cio_digest") or ""
    if isinstance(raw_digest, dict):
        digest = " ".join(str(v) for v in raw_digest.values())[:500]
    else:
        digest = str(raw_digest)[:500]

    system_prompt = """Du bist der Newsletter-Engine von Baldur Creek Capital.
Sprache: Deutsch. Tonalitaet: Sachlich-direkt, operativ, ehrlich.
Jede Zeile muss die Frage beantworten: "Was bedeutet das fuer mein Buch?"
Mindestens 50% der Inhalte muessen Risiken und Gegenargumente zeigen.

Antworte NUR als JSON-Objekt mit diesen Feldern:
{
  "one_thing": "Ein Satz — was heute zaehlt",
  "regime_interpretation": "2-3 Saetze zum Regime-Kontext",
  "scenarios": [
    {"id": "A", "probability_pct": 45, "description": "...", "portfolio_impact": "...", "composite_impact": "...", "action": "NONE/MONITOR/TRADE"},
    {"id": "B", "probability_pct": 35, "description": "...", "portfolio_impact": "...", "composite_impact": "...", "action": "..."},
    {"id": "C", "probability_pct": 20, "description": "...", "portfolio_impact": "...", "composite_impact": "...", "action": "..."}
  ],
  "against_you": [
    {"asset": "HYG", "top_risk": "...", "probability_pct": 35, "mechanism": "..."}
  ],
  "if_wrong_summary": "1 Satz: wahrscheinlichster Schaden heute",
  "blind_spots": ["...", "..."],
  "contrarian_note": null
}

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

Generiere das Newsletter-JSON."""

    return system_prompt, user_prompt


def call_newsletter_llm(system_prompt, user_prompt, api_key=None):
    """
    Call Claude to generate narrative newsletter blocks.
    Returns parsed dict or None on failure.
    """
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
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Parse JSON
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
    """Degraded mode: generate minimal narrative without LLM."""
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
    }


# ---------------------------------------------------------------------------
# MAIN: Assemble full newsletter JSON
# ---------------------------------------------------------------------------

def assemble_newsletter(composite_result, pipeline_data, news_result,
                        indicator_values, api_key=None):
    """
    Assemble the full newsletter JSON with all 15 blocks.

    Args:
        composite_result: from composite.compute_composite_scores()
        pipeline_data: from data_reader.extract_pipeline_data()
        news_result: from news_scanner.run_news_scanner()
        indicator_values: raw indicator dict
        api_key: Anthropic API key

    Returns:
        Complete newsletter dict (ready for JSON serialization).
    """
    today = date.today()
    now = datetime.now(timezone.utc)

    # Newsletter format based on weekday
    fmt = NEWSLETTER_FORMATS.get(today.weekday(), "DAILY")

    # --- Deterministic blocks ---
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

    # --- LLM narrative blocks ---
    system_prompt, user_prompt = build_llm_prompt(
        composite_result, pipeline_data, news_result, indicator_values
    )
    llm_blocks = call_newsletter_llm(system_prompt, user_prompt, api_key)
    if llm_blocks is None:
        logger.warning("LLM failed — using fallback narrative blocks")
        llm_blocks = fallback_narrative_blocks(pipeline_data, composite_result)

    # --- Breaking news ---
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

    # --- Assemble full newsletter ---
    newsletter = {
        "schema_version": "1.0",
        "date": today.isoformat(),
        "weekday": today.strftime("%A"),
        "format": fmt,
        "generated_at": now.isoformat(),

        # Block 1: One Thing (LLM)
        "one_thing": llm_blocks.get("one_thing", ""),

        # Block 2: Composite Scores (deterministic)
        "composite_scores": composite_block,
        "data_integrity": data_integrity,

        # Block 3: Regime Context (deterministic + LLM interpretation)
        "regime_context": regime_context,
        "regime_interpretation": llm_blocks.get("regime_interpretation", ""),

        # Block 4: Portfolio Attribution (deterministic)
        "portfolio_attribution": portfolio_attr,

        # Block 5: Risk Heatmap (deterministic)
        "risk_heatmap": risk_heatmap,

        # Block 6: Indicators (deterministic)
        "indicators": indicators,

        # Block 7: Pipeline Coherence (deterministic)
        "pipeline_coherence": coherence,

        # Block 8: Scenarios (LLM)
        "scenarios": llm_blocks.get("scenarios", []),

        # Block 9: Behavioral Safeguards (deterministic)
        "behavioral": behavioral,

        # Block 10: Against You (LLM)
        "against_you": {
            "positions": llm_blocks.get("against_you", []),
            "if_wrong_summary": llm_blocks.get("if_wrong_summary", ""),
        },

        # Block 11: Epistemic Status (deterministic + LLM blind spots)
        "epistemic_status": {
            **epistemic,
            "blind_spots": llm_blocks.get("blind_spots", []),
        },

        # Block 12: Catalysts 48h (deterministic)
        "catalysts_48h": catalysts,

        # Block 13: Intelligence Digest (deterministic)
        "intelligence_digest": intel_digest,

        # Block 14: Liquidity Pipeline (placeholder for Phase 5)
        "liquidity_pipeline_7d": {
            "available": False,
            "placeholder": "Liquidity Pipeline ab Phase 5 verfuegbar",
        },

        # Block 15: History Enrichment (placeholder)
        "history_enrichment": {
            "available": False,
            "placeholder": "History Enrichment ab April 2026 verfuegbar",
        },

        # Breaking News
        "breaking_news": breaking,
        "breaking_news_summary": news_result.get("summary", "") if news_result else "",

        # Warning triggers
        "warning_triggers": composite_result.get("warning_triggers", []),

        # Contrarian check (Fridays only)
        "contrarian_check": llm_blocks.get("contrarian_note") if fmt == "DAILY_CONTRARIAN" else None,

        # Prediction log entry for today
        "prediction_log": {
            "todays_predictions": _extract_predictions(llm_blocks),
        },

        # Anchor type
        "anchor_type": get_anchor_type(
            composite_result["tactical"]["score"],
            (news_result or {}).get("high_impact_count", 0) > 0,
            pipeline_data.get("risk_emergency_active", False),
            False,  # regime shift — would need yesterday's regime
        ),
    }

    # Add hash (Spec §14.1)
    content_for_hash = json.dumps(newsletter, sort_keys=True, default=str)
    newsletter["hash"] = "sha256:" + hashlib.sha256(content_for_hash.encode()).hexdigest()

    return newsletter


def _extract_predictions(llm_blocks):
    """Extract predictions from scenarios for the prediction log."""
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
    """
    Save newsletter JSON to local file.

    Returns path to saved file.
    """
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
