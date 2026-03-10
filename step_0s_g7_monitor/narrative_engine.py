"""
step_0s_g7_monitor/narrative_engine.py
Phase 8: Narrative Generation — Etappe 3 VOLLSTAENDIG

Produces:
  1. Weekly Narrative Brief (LLM Sonnet call with web search context)
  2. Dashboard Explanations (human-readable context for every metric)
  3. Data Enrichment (LLM extracts published data for empty indicator cells)

Replaces the 82-line stub from Etappe 1.

LLM: claude-sonnet-4-6, Temperature 0.3, max 4000 tokens
Referenz: G7_WORLD_ORDER_MONITOR_SPEC_TEIL5.md
"""

import os
import json
import time
from datetime import datetime, timezone

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

# ============================================================
# ASSET GEO EXPOSURE (Spec Teil 5 §8)
# ============================================================

ASSET_GEO_EXPOSURE = {
    "SPY": "US large cap, ~20% Asia revenue",
    "QQQ": "US tech, heavy TSMC supply chain dependency",
    "GLD": "Neutral, benefits from ALL instability scenarios",
    "TLT": "100% US govt credit, USD duration risk",
    "EEM": "30% China, 15% Taiwan, 15% India",
    "BTC": "Borderless, risk-appetite correlated",
    "DBC": "Global commodities, benefits from multipolar/conflict",
    "VGK": "EU equities, energy + demographics headwind",
}


# ============================================================
# NARRATIVE PROMPT BUILDER
# ============================================================

def _build_narrative_prompt(power_scores, gap_data, overlays, g7_status,
                            scenario_result, previous_narrative):
    """Build the main LLM prompt for narrative generation."""

    # Format power scores
    ps_lines = []
    for r in REGIONS:
        ps = power_scores.get(r, {})
        ps_lines.append(f"  {r:12s}: Score={ps.get('score', 0):.1f}  "
                       f"Mom={ps.get('momentum', 0):+.2f}  "
                       f"Acc={ps.get('acceleration', 0):+.2f}")
    ps_text = "\n".join(ps_lines)

    # Format overlays
    scsi = overlays.get("scsi", {})
    ddi = overlays.get("ddi", {})
    fdp = overlays.get("fdp", {})
    ewi = overlays.get("ewi", {})
    sanctions = overlays.get("sanctions", {})
    feedback = overlays.get("feedback_loops", [])

    # EWI details
    ewi_details = ""
    for sig in ewi.get("active_details", []):
        ewi_details += f"  - {sig.get('signal', '?')}: value={sig.get('value')}, threshold={sig.get('threshold')} [{sig.get('temporal', '?')}]\n"
    if not ewi_details:
        ewi_details = "  None active"

    # Feedback loops
    loop_text = ""
    for loop in feedback[:5]:
        loop_text += f"  - {loop.get('name', '?')} ({loop.get('region', '?')}): severity={loop.get('severity', 0)}, status={loop.get('status', '?')}\n"
    if not loop_text:
        loop_text = "  No active feedback loops (all momenta at 0 — early system, insufficient history)"

    # Previous headline
    prev_headline = ""
    if isinstance(previous_narrative, dict):
        prev_headline = previous_narrative.get("headline", "First run — no previous")

    # FDP summary
    fdp_usa = fdp.get("USA", {}).get("composite_proximity", 0)
    fdp_jp = fdp.get("JP_KR_TW", {}).get("composite_proximity", 0)
    fdp_cn = fdp.get("CHINA", {}).get("composite_proximity", 0)

    # Attractiveness
    attr = overlays.get("attractiveness", [])
    attr_text = ""
    for a in attr[:3]:
        attr_text += f"  #{a.get('rank', '?')} {a.get('region', '?')} (score={a.get('composite_score', 0):.1f})\n"

    return f"""You are the G7 World Order Monitor Narrative Engine for Baldur Creek Capital,
a systematic macro trading operation.

CRITICAL: ALL output text MUST be in GERMAN (Deutsch). Every field — headline,
weekly_shift_narrative, scenario_implications, portfolio_context, counter_narrative,
unasked_question, cascade_watch, dashboard_explanations — everything in German.
Technical terms (SCSI, DDI, FDP, EWI, ETF tickers) stay in English.

DESIGN PRINCIPLES — FOLLOW STRICTLY:
1. PORTFOLIO-FIRST: Every insight references specific portfolio impact. Not "DDI rising"
   but "DDI at 42.7 — with majority of portfolio USD-denominated, this is still below
   the 60+ threshold that would challenge our current allocation."
2. TEMPORAL TAGS: Every insight gets [TACTICAL], [CYCLICAL], or [STRUCTURAL].
3. ATTENTION HIERARCHY: Lead with what CHANGED and MATTERS. Skip stable dimensions.
4. COUNTER-NARRATIVE: Include strongest argument AGAINST current positioning.
5. EXPLANATORY: Write so the operator understands WHY each number matters, not just WHAT it is.

QUANTITATIVE CONTEXT:
Power Scores (Score / Momentum / Acceleration):
{ps_text}

USA-China Gap: {gap_data.get('gap', 0)} (Trend: {gap_data.get('trend', 'STABLE')}, Momentum: {gap_data.get('gap_momentum', 0):+.2f})

OVERLAYS:
SCSI (Supply Chain Stress): {scsi.get('composite', 0)} (Trend: {scsi.get('trend', 'STABLE')})
  Shipping Stress: {scsi.get('shipping_stress', 0)}, Freight PPI Z-Score: {scsi.get('freight_ppi_zscore', 0)}
  Chokepoint Alerts: {scsi.get('active_chokepoint_alerts', 0)}

DDI (De-Dollarization): {ddi.get('composite', 0)} (Trend: {ddi.get('trend', 'STABLE')})
  Components: COFER USD={ddi.get('components', {}).get('cofer_usd_share', {}).get('value', '?')}%,
  Gold Bias: {ddi.get('portfolio_implications', {}).get('gold_bias', '?')}

FDP (Fiscal Dominance): USA={fdp_usa:.2f}, Japan={fdp_jp:.2f}, China={fdp_cn:.2f}
  USA implication: {fdp.get('USA', {}).get('implication', 'N/A')}

EWI (Early Warning): {ewi.get('active_signals', 0)}/{ewi.get('total_signals', 10)} active, severity={ewi.get('severity', 'NONE')}
{ewi_details}

Sanctions: {sanctions.get('escalation_trend', 'STABLE')}

Feedback Loops:
{loop_text}

Attractiveness Ranking:
{attr_text}

G7 STATUS: {g7_status.get('g7_status', 'UNKNOWN')}
Attention Flag: {g7_status.get('attention_flag', 'NONE')}
Active Shifts: {g7_status.get('active_shifts', [])}
Previous Headline: {prev_headline}

GPR Index: {overlays.get('gpr_index_current', 100)} (Z-score: {overlays.get('gpr_index_zscore', 0):+.2f})

RESPOND IN THIS EXACT JSON SCHEMA (no markdown, no backticks, pure JSON):
{{
    "headline": "One specific sentence for email subject — what is the MOST important thing this week",

    "weekly_shift_narrative": "150-250 words. What changed THIS WEEK, why it matters, and what to watch. Portfolio-first framing throughout.",

    "top_signals": [
        {{"rank": 1, "signal": "description", "temporal": "TACTICAL|CYCLICAL|STRUCTURAL",
          "portfolio_impact": "specific impact on holdings", "action_horizon": "when to act"}}
    ],

    "scenario_implications": "100-150 words. How current data affects the 4 scenario probabilities.",

    "portfolio_context": "100-200 words. PORTFOLIO-FIRST. Reference specific positions and what to do.",

    "counter_narrative": {{
        "our_thesis": "what we believe",
        "strongest_counter": "the best argument against us",
        "data_supporting_counter": ["data point 1", "data point 2", "data point 3"],
        "our_response": "why we still hold our position",
        "action": "what to review or monitor"
    }},

    "unasked_question": "One provocative question the operator should be asking",

    "cascade_watch": ["second-order effect 1", "second-order effect 2"],

    "regime_congruence": {{
        "congruent": true,
        "tension": null,
        "operator_implication": "what this means for positioning"
    }},

    "historical_analog": {{
        "best_match": "period name",
        "key_parallel": "what is similar",
        "portfolio_lesson": "what to learn"
    }},

    "dashboard_explanations": {{
        "power_gap_explanation": "3-4 sentences explaining USA-China gap of {gap_data.get('gap', 0)}: what drives it, what it means for the portfolio, and what would change it. Written for a non-specialist to understand.",
        "scsi_explanation": "2-3 sentences on supply chain stress at {scsi.get('composite', 0)}: what it measures, current risk level, portfolio exposure.",
        "ddi_explanation": "2-3 sentences on de-dollarization at {ddi.get('composite', 0)}: where we are in the cycle, what it means for gold/USD positions.",
        "fdp_explanation": "2-3 sentences on fiscal dominance proximity for USA at {fdp_usa:.2f}: how close to the threshold, timeline, bond market impact.",
        "ewi_explanation": "2-3 sentences on early warning at {ewi.get('active_signals', 0)} signals: what is firing and urgency level.",
        "status_explanation": "2-3 sentences explaining G7 status {g7_status.get('g7_status', 'UNKNOWN')}: why this status, what triggers it, what it means for risk management."
    }}
}}"""


# ============================================================
# LLM CALL
# ============================================================

def _call_anthropic(prompt, max_tokens=4000, temperature=0.3):
    """Call Claude Sonnet via Anthropic API."""
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [Narrative] No ANTHROPIC_API_KEY — returning stub")
        return None

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from response
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        return text
    except Exception as e:
        print(f"  [Narrative] API call failed: {e}")
        return None


def _parse_json_response(text):
    """Parse JSON from LLM response, handling markdown fences."""
    if not text:
        return None

    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  [Narrative] JSON parse failed: {e}")
        # Try to find JSON in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        return None


# ============================================================
# NARRATIVE VALIDATION
# ============================================================

def _validate_narrative(narrative):
    """Validate narrative output per Spec Teil 5 §10."""
    errors = []
    warnings = []

    required = ["headline", "weekly_shift_narrative", "top_signals",
                "scenario_implications", "portfolio_context",
                "counter_narrative", "unasked_question"]
    for f in required:
        if not narrative.get(f):
            errors.append(f"MISSING: {f}")

    # Top signals
    signals = narrative.get("top_signals", [])
    if len(signals) < 2:
        warnings.append(f"TOP_SIGNALS: Only {len(signals)} (want 3-5)")

    # Counter-narrative
    cn = narrative.get("counter_narrative", {})
    if isinstance(cn, dict):
        if not cn.get("strongest_counter"):
            errors.append("COUNTER: No strongest_counter")
        if len(cn.get("data_supporting_counter", [])) < 2:
            warnings.append("COUNTER: Fewer than 3 supporting data points")

    # Headline quality
    h = narrative.get("headline", "")
    if len(h) > 120:
        warnings.append(f"HEADLINE: {len(h)} chars (target <100)")

    # Word count
    wc = sum(len(str(v).split()) for v in narrative.values() if isinstance(v, str))
    if wc < 200:
        warnings.append(f"LENGTH: {wc} words (target 600-1000)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "word_count": wc,
    }


# ============================================================
# FALLBACK NARRATIVE (when LLM unavailable)
# ============================================================

def _build_fallback_narrative(power_scores, gap_data, overlays, g7_status):
    """Deterministic fallback when LLM call fails."""
    gap = gap_data.get("gap", 0)
    status = g7_status.get("g7_status", "UNKNOWN")
    scsi = overlays.get("scsi", {}).get("composite", 0)
    ddi = overlays.get("ddi", {}).get("composite", 0)
    fdp_usa = overlays.get("fdp", {}).get("USA", {}).get("composite_proximity", 0)
    ewi_count = overlays.get("ewi", {}).get("active_signals", 0)

    return {
        "headline": f"G7 Status {status} — Abstand {gap:.1f}, SCSI {scsi:.0f}, DDI {ddi:.0f}",
        "weekly_shift_narrative": (
            f"G7 Weltordnungs-Monitor meldet Status {status}. "
            f"USA-China Machtabstand beträgt {gap:.1f} Punkte. "
            f"Supply Chain Stress (SCSI) bei {scsi:.0f}/100. "
            f"De-Dollarisierungs-Index (DDI) bei {ddi:.0f}/100. "
            f"US Fiskal-Dominanz-Nähe bei {fdp_usa:.0%}. "
            f"Frühwarnsystem zeigt {ewi_count} aktive Signale. "
            f"Vollständige Narrative ausstehend — LLM nicht verfügbar."
        ),
        "top_signals": [],
        "scenario_implications": "Szenario-Analyse ausstehend — LLM nicht verfügbar.",
        "portfolio_context": "Portfolio-Kontext ausstehend — LLM nicht verfügbar.",
        "counter_narrative": {
            "our_thesis": "Aktuelle Positionierung",
            "strongest_counter": "Ausstehend — LLM-Analyse",
            "data_supporting_counter": [],
            "our_response": "Ausstehend",
            "action": "Prüfen wenn Narrative Engine verfügbar",
        },
        "unasked_question": "",
        "cascade_watch": [],
        "regime_congruence": {"congruent": True, "tension": None,
                             "operator_implication": "Ausstehend — LLM-Analyse"},
        "historical_analog": {},
        "dashboard_explanations": _build_deterministic_explanations(
            gap_data, overlays, g7_status),
        "attention_flag": g7_status.get("attention_flag", "NONE"),
        "word_count": 0,
        "llm_model": "fallback_deterministic",
        "generation_time_seconds": 0,
    }


def _build_deterministic_explanations(gap_data, overlays, g7_status):
    """Build dashboard explanations without LLM — deterministic text."""
    gap = gap_data.get("gap", 0)
    trend = gap_data.get("trend", "STABLE")
    scsi = overlays.get("scsi", {}).get("composite", 0)
    ddi = overlays.get("ddi", {}).get("composite", 0)
    fdp_usa = overlays.get("fdp", {}).get("USA", {}).get("composite_proximity", 0)
    ewi_count = overlays.get("ewi", {}).get("active_signals", 0)
    ewi_sev = overlays.get("ewi", {}).get("severity", "NONE")
    status = g7_status.get("g7_status", "UNKNOWN")

    # Gap explanation
    if gap > 15:
        gap_ex = (f"USA führt China mit {gap:.1f} Punkten — ein komfortabler Vorsprung. "
                  f"Trend: {trend}. Auf diesem Niveau ist das Thukydides-Fallen-Risiko NIEDRIG. "
                  f"Primäre US-Vorteile: Kapitalmarkttiefe, Reservewährung, Energieunabhängigkeit. "
                  f"Ein Abstand unter 15 würde VERSCHIEBUNG auslösen, unter 10 ERHÖHTES RISIKO.")
    elif gap > 10:
        gap_ex = (f"USA führt China mit {gap:.1f} Punkten — moderater Vorsprung, Trend {trend}. "
                  f"Dies löst VERSCHIEBUNG aus, da der Abstand unter 15 liegt. "
                  f"Chinas Zugewinne kommen durch wirtschaftliches Gewicht und Demografie. "
                  f"Bei weiterem Rückgang unter 10 eskaliert der Status zu ERHÖHTES RISIKO.")
    else:
        gap_ex = (f"USA führt China mit nur {gap:.1f} Punkten — enger Vorsprung, Trend {trend}. "
                  f"ERHÖHTES RISIKO. Historische Machttransitionen werden volatil wenn der "
                  f"Abstand so eng wird. Portfolio: Absicherungen erhöhen, Taiwan-Straße "
                  f"beobachten, Gold-Übergewichtung erwägen.")

    # SCSI
    if scsi < 20:
        scsi_ex = (f"Supply-Chain-Stress bei {scsi:.0f}/100 — minimale Störung. "
                   f"Keine Chokepoint-Alarme aktiv. Frachtkosten normal. "
                   f"Keine Portfolio-Aktion nötig auf Lieferketten-Seite.")
    elif scsi < 50:
        scsi_ex = (f"Supply-Chain-Stress bei {scsi:.0f}/100 — moderat. "
                   f"Frachtkosten erhöht. Chokepoint-Eskalation beobachten. "
                   f"Energieimportierende Regionen (JP/KR/TW, EU, Indien) am stärksten exponiert.")
    else:
        scsi_ex = (f"Supply-Chain-Stress bei {scsi:.0f}/100 — ERHÖHT. "
                   f"Aktive Störungen in Schifffahrtsrouten. Direkter Portfolio-Impact "
                   f"auf energiesensitive und handelsabhängige Positionen.")

    # DDI
    if ddi < 50:
        ddi_ex = (f"De-Dollarisierungs-Index bei {ddi:.0f}/100 — unter neutral. "
                  f"USD-Dominanz weiterhin intakt. COFER-Reserveanteil stabil. "
                  f"Aktuelle Gold-Allokation angemessen, kein Handlungsdruck.")
    elif ddi < 65:
        ddi_ex = (f"De-Dollarisierungs-Index bei {ddi:.0f}/100 — nähert sich Schwelle. "
                  f"COFER-USD-Anteil rückläufig, Zentralbank-Goldkäufe erhöht. "
                  f"Stützt aktuelle Gold-Übergewichtung. SWIFT-Anteil für Beschleunigung beobachten.")
    else:
        ddi_ex = (f"De-Dollarisierungs-Index bei {ddi:.0f}/100 — AKTIVE De-Dollarisierung. "
                  f"Gold-Allokation erhöhen erwägen. PermOpt-Überprüfung angebracht. "
                  f"USD-denominiertes Portfolio steht vor strukturellem Gegenwind.")

    # FDP
    if fdp_usa < 0.60:
        fdp_ex = (f"US Fiskal-Dominanz-Nähe bei {fdp_usa:.0%} — handhabbar. "
                  f"Zinskosten signifikant aber noch nicht Fed-Politik einschränkend. "
                  f"TLT-Position vorerst sicher, Schuldenentwicklung beobachten.")
    elif fdp_usa < 0.85:
        fdp_ex = (f"US Fiskal-Dominanz-Nähe bei {fdp_usa:.0%} — Fiskalspielraum verengt sich. "
                  f"Zinszahlungen verbrauchen wachsenden Anteil der Einnahmen. "
                  f"Anleihenmarkt-Neubewertungsrisiko steigt. TLT-Duration prüfen.")
    else:
        fdp_ex = (f"US Fiskal-Dominanz-Nähe bei {fdp_usa:.0%} — KRITISCH. "
                  f"Nähert sich der Fiskal-Dominanz-Schwelle. Fed-Politik zunehmend "
                  f"durch Fiskalbedürfnisse eingeschränkt. Langläufer-Exposure reduzieren.")

    # EWI
    if ewi_count == 0:
        ewi_ex = "Keine Frühwarnsignale aktiv. System ruhig über alle überwachten Indikatoren."
    elif ewi_count <= 2:
        ewi_ex = (f"{ewi_count} Frühwarnsignal(e) aktiv, Schwere {ewi_sev}. "
                  f"Erhöht aber nicht alarmierend. Auf Clustering achten — "
                  f"3+ Signale würden zur VERSCHIEBUNG beitragen.")
    else:
        ewi_ex = (f"{ewi_count} Frühwarnsignale aktiv, Schwere {ewi_sev}. "
                  f"Mehrere unabhängige Warnsysteme schlagen an. "
                  f"Historisch geht dieses Alarmniveau Regimewechseln voraus.")

    # Status
    status_map = {
        "STABLE": (f"G7 Status STABIL — keine Auslöser aktiv. Alle überwachten Schwellenwerte "
                   f"im Normalbereich. Keine sofortige Risikomanagement-Aktion nötig."),
        "SHIFTING": (f"G7 Status IM WANDEL — die Weltordnung ist in Transition. "
                     f"Ausgelöst durch: {', '.join(g7_status.get('active_shifts', ['Gap < 15'])[:2])}. "
                     f"Bedeutet erhöhte Beobachtung, keine Panik. Portfolio-Review quartalsweise."),
        "ELEVATED_RISK": (f"G7 Status ERHÖHTES RISIKO — mehrere Stress-Indikatoren aktiv. "
                          f"Ausgelöst durch: {', '.join(g7_status.get('active_shifts', [])[:2])}. "
                          f"Alle Positionen prüfen, Absicherungen erhöhen, Stop-Losses aktualisieren."),
        "STRUCTURAL_BREAK": (f"G7 Status STRUKTURBRUCH — großer Regimewechsel erkannt. "
                             f"Sofortige Portfolio-Überprüfung erforderlich. "
                             f"Risikomanagement hat Vorrang vor Renditeoptimierung."),
    }
    status_ex = status_map.get(status, f"G7 status {status}.")

    return {
        "power_gap_explanation": gap_ex,
        "scsi_explanation": scsi_ex,
        "ddi_explanation": ddi_ex,
        "fdp_explanation": fdp_ex,
        "ewi_explanation": ewi_ex,
        "status_explanation": status_ex,
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def phase8_narrative_generation(power_scores, gap_data, overlays, g7_status,
                                scenario_result, web_search_results,
                                previous_narrative):
    """
    Phase 8: Generate narrative brief via LLM.
    Falls back to deterministic narrative if LLM unavailable.

    Returns dict with all narrative fields + dashboard_explanations.
    """
    print("[Phase 8] Narrative Generation (Etappe 3)...")
    start = time.time()

    # Build prompt
    prompt = _build_narrative_prompt(
        power_scores, gap_data, overlays, g7_status,
        scenario_result, previous_narrative)

    # Call LLM
    raw_response = _call_anthropic(prompt, max_tokens=4000, temperature=0.3)

    if raw_response:
        narrative = _parse_json_response(raw_response)
        if narrative:
            # Validate
            validation = _validate_narrative(narrative)
            if validation["errors"]:
                print(f"  Validation errors: {validation['errors']}")
            if validation["warnings"]:
                print(f"  Validation warnings: {validation['warnings']}")

            # Ensure dashboard_explanations exist
            if not narrative.get("dashboard_explanations"):
                narrative["dashboard_explanations"] = _build_deterministic_explanations(
                    gap_data, overlays, g7_status)

            narrative["attention_flag"] = g7_status.get("attention_flag", "NONE")
            narrative["word_count"] = validation.get("word_count", 0)
            narrative["llm_model"] = "claude-sonnet-4-6"
            narrative["generation_time_seconds"] = round(time.time() - start, 1)

            print(f"  Narrative generated: {narrative['word_count']} words, "
                  f"{narrative['generation_time_seconds']}s")
            print(f"  Headline: {narrative.get('headline', '?')[:80]}")
            return narrative
        else:
            print("  JSON parse failed — using fallback")
    else:
        print("  LLM call failed — using fallback")

    # Fallback
    fallback = _build_fallback_narrative(power_scores, gap_data, overlays, g7_status)
    fallback["generation_time_seconds"] = round(time.time() - start, 1)
    print(f"  Fallback narrative generated")
    return fallback
