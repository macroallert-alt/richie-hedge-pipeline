"""
step5_devils_advocate/llm.py
Devil's Advocate LLM — System Prompt, User Prompt Builder, API Call, Challenge Parser

Source: DA Spec Teil 3
"""

import json
import logging
import re
import time
from datetime import date

import anthropic

logger = logging.getLogger("da_llm")


# =============================================================================
# SYSTEM PROMPT (Spec Teil 3 §3.2)
# =============================================================================

SYSTEM_PROMPT = """Du bist der Devil's Advocate eines quantitativen Multi-Agent-Trading-Systems. Du bist KEIN Kritiker und KEIN Contrarian. Du bist ein Anomalie-Detektor — ein Immunsystem das erkennt was anders ist als erwartet.

DER OPERATOR betreibt:
- V16 Global Macro RV System: Regime-Erkennung, 25 US-ETFs, automatisch
- F6 Stock Picker: Einzelaktien, Covered Call Overlay, 21-Tage-Holding
- Eine Pipeline aus deterministischen und LLM-basierten Agents

DU ANALYSIERST den CIO Draft — das taegliche Briefing das der CIO (Chief Investment Officer Agent) geschrieben hat. Dein Output wird dem CIO zurueckgegeben. Er bewertet jede deiner Challenges mit ACCEPTED (aendert den Final), NOTED (auf Watchlist), oder REJECTED (zurueckgewiesen mit Begruendung).

DEINE IDENTITAET:

Du bist der Mensch der um 4 Uhr morgens aufwacht und fragt: "Was wenn wir alle falsch liegen? Nicht in den Details — in der Grundannahme?"

Du hast KEINE eigene Marktmeinung. Du sagst nicht "ich glaube der Markt faellt." Du sagst: "Ist dir aufgefallen dass die Daten auch SO interpretiert werden koennten?"

Du bist nicht pessimistisch. An Tagen wo der CIO panisch ist, fragst du: "Bist du sicher dass es SO schlimm ist? Was sind die stabilisierenden Faktoren?"

DU HAST DREI AUFTRAEGE — in dieser Reihenfolge der Wichtigkeit:

AUFTRAG 1: ALTERNATIVE NARRATIVE
  Lies den CIO Draft. Identifiziere die Geschichte die er erzaehlt — die Verbindungen zwischen den Fakten, nicht die Fakten selbst.
  Erzaehle dieselben Fakten als eine ANDERE, gleichwertig plausible Geschichte.
  Nicht das Gegenteil — eine andere Interpretation.

  SCHLECHT: "Der CIO sagt Risk-On, aber ich sage Risk-Off."
  GUT: "Der CIO sagt Liquidity steigt = bullish. Alternative: Liquidity steigt weil Kapitalflucht aus EM. Das ist nicht bullish, das ist ein Warnsignal."

  Die alternative Narrative muss durch dieselben Daten gestuetzt sein die der CIO verwendet. Keine erfundenen Daten, keine Spekulation ohne Basis.

AUFTRAG 2: DIE UNGESTELLTE FRAGE
  Was ist die Frage die das System heute stellen SOLLTE, aber nicht stellt?
  Welche Dimension der Realitaet wird nicht gemessen?

  Das System fokussiert auf US-Maerkte, Macro-Regime, und Portfolio-Risiko.
  Was liegt AUSSERHALB dieses Rahmens und koennte trotzdem relevant sein?

  Nicht als Feature Request. Als konkretes Risiko fuer HEUTE oder diese Woche.

  SCHLECHT: "Das System sollte auch Japan-Daten tracken." (Feature Request)
  GUT: "BOJ hat ein Emergency Meeting angekuendigt. Das System hat keinen Japan-Input. Wenn der Carry Trade unwinded, trifft es US-Tech in 48h."

AUFTRAG 3: PRAEMISSEN-ANGRIFF
  Wenn der CIO Key Assumptions am Ende seines Drafts gelistet hat, nimm die schwaechste und greife sie an.
  Wenn keine Key Assumptions vorhanden sind, identifiziere die impliziten Annahmen im Draft-Text.

  Greife die ANNAHME an, nicht die Schlussfolgerung.

  SCHLECHT: "Tech 42% ist zu hoch." (Schlussfolgerung)
  GUT: "Der CIO nimmt an Tech 42% sei manageable WEIL Fragility HEALTHY ist. Aber Fragility misst HHI, nicht Korrelation." (Praemisse)

  Nutze die Pre-Processor Flags als Munition: Omissions, Drift, Inkonsistenzen.
  Nutze die Confidence Markers: LOW Confidence = offenes Angriffsfeld.

HARTE REGELN:

1. Greife NIEMALS V16-Gewichte oder V16-Regime an. Sie sind sakrosankt.
2. Greife NIEMALS F6-Signale an. Sie sind sakrosankt.
3. Aendere NIEMALS Risk Officer Severities. Die sind offiziell.
4. Gib KEINE Trade-Empfehlungen. Das ist nicht deine Rolle.
5. Produziere KEINE generischen Challenges. "Was wenn Rezession?" ist wertlos.
6. Jede Challenge MUSS durch spezifische Daten aus den Inputs gestuetzt sein.
7. Qualitaet > Quantitaet. Eine messerscharfe Challenge ist besser als drei mittelmassige.
8. Maximum 5 Challenges. Wenn du mehr als 5 findest, nimm die 5 besten.
9. Minimum 1 Challenge. Wenn du NICHTS findest, erklaere warum — aber das sollte selten sein.
10. Sage NICHT "der CIO irrt sich." Sage "Ist dir aufgefallen dass..."

CHALLENGE SEVERITY:

SUBSTANTIVE — Koennte die Gesamteinschaetzung des CIO aendern wenn korrekt.
MODERATE — Relevanter Punkt, aber aendert wahrscheinlich nicht die Gesamtrichtung.
MINOR — Beobachtung am Rand. Nur aufnehmen wenn weniger als 3 Challenges insgesamt.

OUTPUT-FORMAT:

Jede Challenge hat diese Struktur:
  TYPE: NARRATIVE | UNASKED_QUESTION | PREMISE_ATTACK
  TARGET_SECTION: S1-S7 oder SYSTEM (wenn systemweit)
  TARGET_ASSUMPTION: KA1/KA2/KA3 oder null
  SEVERITY: SUBSTANTIVE | MODERATE | MINOR
  CHALLENGE: [Der eigentliche Text — praezise, datengestuetzt, nicht laenger als noetig]
  EVIDENCE:
  - [Datenpunkt 1]
  - [Datenpunkt 2]
  - [...]

Trenne Challenges mit einer Zeile die nur --- enthaelt.
Beginne direkt mit der ersten Challenge. Kein einleitender Text. Keine Zusammenfassung am Ende.

VERBOTENE FORMULIERUNGEN:

SAGE NICHT:                              SAGE STATTDESSEN:
"Der CIO liegt falsch"                   "Die Daten lassen auch diese Lesart zu..."
"Das ist gefaehrlich"                    "Der Expected Loss bei diesem Szenario ist..."
"Ich empfehle zu verkaufen"              [VERBOTEN — keine Trade-Empfehlungen]
"V16 sollte auf Risk-Off wechseln"       [VERBOTEN — Master-Schutz]
"Was wenn Rezession?"                    [Zu generisch — spezifisches Szenario mit Daten]
"Der Risk Officer uebersieht..."         [VERBOTEN — keine Agent-Kritik]
"Offensichtlich hat der CIO..."          "Ist aufgefallen dass..."
"""


# =============================================================================
# CONFIDENCE SATURATION BONUS INSTRUCTION (Spec Teil 3 §3.3)
# =============================================================================

SATURATION_BONUS_ACTIVE = """
⚠️ CONFIDENCE SATURATION AKTIV (Score: {score:.0%})

Das System ist sich ungewoehnlich einig. Alle Subsysteme zeigen in dieselbe Richtung.
Historisch sind Momente maximaler Einigkeit oft Wendepunkte.

ZUSAETZLICHER AUFTRAG: Stelle eine "Illusion of Safety" Challenge.

Frage: "Was ist das Szenario das ALLE uebersehen — nicht weil es unwahrscheinlich ist,
sondern weil es im aktuellen Konsens-Framework unsichtbar ist?"
"""

SATURATION_BONUS_INACTIVE = """Confidence Saturation nicht aktiv ({score:.0%}, Schwelle 85%). Kein Bonus-Auftrag."""


# =============================================================================
# USER PROMPT BUILDER (Spec Teil 3 §3.3)
# =============================================================================

def build_user_prompt(preprocessor_output: dict, inputs: dict) -> str:
    """Build the dynamic user prompt from preprocessor output + inputs."""
    pp = preprocessor_output
    draft = inputs["draft_memo"]

    # Steering section
    focus = pp["focus"]
    seed = pp["perspective_seed"]
    asymmetry = pp["asymmetry"]
    saturation = pp["confidence_saturation"]

    # Saturation bonus
    if saturation["active"]:
        sat_bonus = SATURATION_BONUS_ACTIVE.format(score=saturation["score"])
    else:
        sat_bonus = SATURATION_BONUS_INACTIVE.format(score=saturation["score"])

    # Persistent challenges
    persistent_text = _format_persistent(pp.get("persistent_challenges", []))
    forced_text = _format_forced_decision(pp.get("forced_decision_challenges", []))

    # Flags
    flags_json = json.dumps(pp["flags"]["for_llm"], indent=2, default=str, ensure_ascii=False)

    # Draft data
    briefing_text = draft.get("briefing_text", "")
    key_assumptions = json.dumps(draft.get("key_assumptions", []), indent=2, ensure_ascii=False)
    confidence_markers = json.dumps(draft.get("confidence_markers", []), indent=2, ensure_ascii=False)

    # Raw data for cross-checking (truncated to keep token count reasonable)
    risk_alerts_json = _truncate_json(inputs.get("risk_alerts", {}), 2000)
    signals_json = _truncate_json(inputs.get("signals", {}), 500)
    layer_json = _truncate_json(_extract_layer_summary(inputs.get("layer_analysis", {})), 1500)
    ic_json = _truncate_json(_extract_ic_summary(inputs.get("ic_intelligence", {})), 2000)
    v16_json = _truncate_json(inputs.get("v16_production", {}), 800)
    f6_json = _truncate_json(inputs.get("f6_production", {}), 500)

    # Yesterday context
    yesterday = inputs.get("yesterday_final")
    if yesterday:
        y_type = yesterday.get("briefing_type", "?")
        y_conv = yesterday.get("system_conviction", "?")
        y_ampel = yesterday.get("risk_ampel", "?")
        y_ka = json.dumps(yesterday.get("key_assumptions", []), indent=2, ensure_ascii=False)
    else:
        y_type = "UNAVAILABLE"
        y_conv = "UNAVAILABLE"
        y_ampel = "UNAVAILABLE"
        y_ka = "Nicht verfuegbar (erster Run oder Archive fehlt)"

    # DA History context
    acc_rate = pp.get("acceptance_rate", {})
    acc_overall = acc_rate.get("acceptance_rate_overall", 0)
    acc_narrative = acc_rate.get("acceptance_rate_narrative", 0)
    acc_unasked = acc_rate.get("acceptance_rate_unasked", 0)
    acc_premise = acc_rate.get("acceptance_rate_premise", 0)
    noted_rate = acc_rate.get("noted_rate_overall", 0)

    prompt = f"""Analysiere den folgenden CIO Draft und produziere deine Challenges.

=== STEUERUNG (vom Pre-Processor) ===

PRIMAERER FOKUS HEUTE: {focus['primary_focus']}
(Widme diesem Auftrag die meiste Aufmerksamkeit. Die anderen zwei bleiben aktiv aber sekundaer.)

PERSPEKTIV-SEED:
{seed['seed_instruction']}

ASYMMETRIE-INSTRUKTION:
{asymmetry['challenge_guidance']}

MINDEST-CHALLENGES: {asymmetry['min_challenges']}

=== PRE-PROCESSOR FLAGS (deine Munition) ===

{flags_json}

=== CONFIDENCE SATURATION ===

Status: {'AKTIV' if saturation['active'] else 'Nicht aktiv'}
Score: {saturation['score']:.0%}
{saturation['interpretation']}

{sat_bonus}

=== PERSISTENT CHALLENGES (muessen erneut gestellt werden) ===

{persistent_text}

=== FORCED DECISION CHALLENGES (CIO muss ACCEPTED oder REJECTED waehlen) ===

{forced_text}

=== CIO DRAFT ===

Header:
  Datum: {draft.get('date', '?')}
  Briefing-Typ: {draft.get('briefing_type', '?')}
  System Conviction: {draft.get('system_conviction', '?')}
  Risk Ampel: {draft.get('risk_ampel', '?')}
  Fragility: {draft.get('fragility_state', '?')}
  V16 Regime: {draft.get('v16_regime', draft.get('preprocessor_output', {}).get('header', {}).get('v16_regime', '?'))}

Briefing:
{briefing_text}

Key Assumptions:
{key_assumptions}

Confidence Markers:
{confidence_markers}

=== ROHDATEN (fuer Gegenpruefung) ===

Risk Officer Alerts:
{risk_alerts_json}

Signal Generator:
{signals_json}

Market Analyst Layer Scores:
{layer_json}

IC Intelligence (Claims, Divergenzen, Konsens):
{ic_json}

V16 Production:
{v16_json}

F6 Production:
{f6_json}

=== GESTRIGES FINAL (fuer Drift-Kontext) ===

Gestriger Briefing-Typ: {y_type}
Gestrige Conviction: {y_conv}
Gestrige Risk Ampel: {y_ampel}
Gestrige Key Assumptions:
{y_ka}

=== DA HISTORY KONTEXT ===

Acceptance Rate (30 Tage): {acc_overall:.0%}
  NARRATIVE: {acc_narrative:.0%}
  UNASKED_QUESTION: {acc_unasked:.0%}
  PREMISE_ATTACK: {acc_premise:.0%}
NOTED Rate (30 Tage): {noted_rate:.0%}

Schreibe jetzt deine Challenges. Beginne direkt mit Challenge 1."""

    return prompt


def _format_persistent(persistent_challenges: list) -> str:
    """Format persistent challenges for prompt."""
    if not persistent_challenges:
        return "Keine offenen Persistent Challenges."
    lines = ["PFLICHT — Diese Challenges muessen erneut gestellt werden:\n"]
    for pc in persistent_challenges:
        responses = [r.get("resolution") for r in pc.get("cio_responses", [])]
        lines.append(f"ID: {pc.get('id')}")
        lines.append(f"Erstmals gestellt: {pc.get('first_raised', '?')}")
        lines.append(f"Tage offen: {pc.get('days_open', 1)}")
        lines.append(f"Bisherige CIO-Antworten: {responses}")
        lines.append(f"Challenge: {pc.get('challenge_text', '')}")
        lines.append(f"Eskalation: {pc.get('escalation_note', '')}")
        lines.append(f"\nErneuere diese Challenge. Du darfst sie umformulieren, "
                     f"erweitern, oder mit neuen Daten anreichern — aber das THEMA "
                     f"muss bestehen bleiben bis der CIO es resolved.\n")
    return "\n".join(lines)


def _format_forced_decision(forced_challenges: list) -> str:
    """Format forced decision challenges for prompt."""
    if not forced_challenges:
        return "Keine Forced Decision Challenges."
    lines = ["⚠️ FORCED DECISION — CIO muss ACCEPTED oder REJECTED waehlen:\n"]
    for fd in forced_challenges:
        lines.append(f"ID: {fd.get('id')}")
        lines.append(f"Tage offen: {fd.get('days_open', 1)}")
        lines.append(f"Mal NOTED: {fd.get('noted_count', 3)}")
        lines.append(f"Challenge: {fd.get('challenge_text', '')}")
        lines.append(f"ESKALATION: {fd.get('escalation_note', '')}")
        lines.append(f"\nStelle diese Challenge PROMINENT und DEUTLICH.\n")
    return "\n".join(lines)


def _extract_layer_summary(layer_analysis: dict) -> dict:
    """Extract compact layer summary for prompt."""
    if not layer_analysis:
        return {"status": "UNAVAILABLE"}
    summary = {
        "system_regime": layer_analysis.get("system_regime"),
        "fragility_state": layer_analysis.get("fragility_state"),
    }
    layers = layer_analysis.get("layers", layer_analysis.get("layer_scores", {}))
    if layers:
        compact = {}
        for name, data in layers.items():
            if isinstance(data, dict):
                compact[name] = {
                    "score": data.get("score"),
                    "direction": data.get("direction"),
                    "regime": data.get("regime"),
                }
            else:
                compact[name] = data
        summary["layers"] = compact
    return summary


def _extract_ic_summary(ic_intel: dict) -> dict:
    """Extract compact IC summary for prompt."""
    if not ic_intel:
        return {"status": "UNAVAILABLE"}
    return {
        "high_novelty_claims": ic_intel.get("high_novelty_claims", [])[:10],
        "divergences": ic_intel.get("divergences", []),
        "consensus": ic_intel.get("consensus", ic_intel.get("ic_consensus", {})),
        "extraction_summary": ic_intel.get("extraction_summary", {}),
    }


def _truncate_json(data, max_chars: int) -> str:
    """JSON-serialize and truncate if needed."""
    try:
        text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text
    except Exception:
        return str(data)[:max_chars]


# =============================================================================
# CHALLENGE PARSER (Spec Teil 3 §3.4)
# =============================================================================

def parse_challenges(llm_output_text: str) -> list:
    """Parse LLM output into structured challenges."""
    challenges = []

    # Split on "CHALLENGE N:" or "---"
    blocks = re.split(r'(?:^|\n)(?:CHALLENGE\s*\d+\s*:|---)\s*\n?', llm_output_text)
    blocks = [b.strip() for b in blocks if b.strip()]

    today_str = date.today().strftime("%Y%m%d")

    for i, block in enumerate(blocks):
        challenge = {
            "id": f"da_{today_str}_{i+1:03d}",
            "type": _extract_field(block, "TYPE"),
            "target_section": _extract_field(block, "TARGET_SECTION"),
            "target_assumption": _extract_field(block, "TARGET_ASSUMPTION"),
            "severity": _extract_field(block, "SEVERITY"),
            "challenge_text": _extract_field(block, "CHALLENGE"),
            "evidence": _extract_list_field(block, "EVIDENCE"),
            "is_persistent": False,
            "persistent_days": 0,
        }

        # Validate type
        if challenge["type"] not in ["NARRATIVE", "UNASKED_QUESTION", "PREMISE_ATTACK"]:
            challenge["type"] = "PREMISE_ATTACK"
        # Validate severity
        if challenge["severity"] not in ["SUBSTANTIVE", "MODERATE", "MINOR"]:
            challenge["severity"] = "MODERATE"
        # Normalize target_assumption
        if challenge["target_assumption"] in ("null", "", "None", "none"):
            challenge["target_assumption"] = None

        # Only include if challenge text exists
        if challenge["challenge_text"]:
            challenges.append(challenge)

    return challenges


def _extract_field(block: str, field_name: str) -> str | None:
    """Extract a named field from a challenge block."""
    pattern = rf'^{field_name}:\s*(.+?)(?=\n[A-Z_]+:|\nEVIDENCE:|\Z)'
    match = re.search(pattern, block, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_list_field(block: str, field_name: str) -> list:
    """Extract a list field (lines starting with -) from a challenge block."""
    pattern = rf'^{field_name}:\s*\n((?:[-•]\s*.+\n?)+)'
    match = re.search(pattern, block, re.MULTILINE)
    if match:
        items = re.findall(r'^[-•]\s*(.+)$', match.group(1), re.MULTILINE)
        return items
    return []


# =============================================================================
# LLM API CALL (Spec Teil 3 §3.5)
# =============================================================================

def call_da_llm(user_prompt: str, config: dict) -> dict:
    """Call Claude API with retry logic."""
    llm_cfg = config.get("llm", {})
    model = llm_cfg.get("model", "claude-sonnet-4-5-20250929")
    temperature = llm_cfg.get("temperature", 0.4)
    max_tokens = llm_cfg.get("max_tokens", 4000)
    retries = llm_cfg.get("retry_on_failure", 2)
    temp_increment = llm_cfg.get("retry_temperature_increment", 0.1)
    timeout = llm_cfg.get("timeout_seconds", 45)

    client = anthropic.Anthropic()

    for attempt in range(retries + 1):
        try:
            logger.info(f"DA LLM call attempt {attempt + 1}/{retries + 1}, "
                        f"temp={temperature}, model={model}")
            start = time.time()

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=timeout,
            )

            output_text = response.content[0].text
            elapsed = time.time() - start
            logger.info(f"DA LLM success on attempt {attempt + 1}. "
                        f"{len(output_text)} chars, {elapsed:.1f}s")

            challenges = parse_challenges(output_text)
            if len(challenges) >= 1:
                return {
                    "success": True,
                    "output_text": output_text,
                    "challenges": challenges,
                    "attempt": attempt + 1,
                    "generation_time_seconds": round(elapsed, 1),
                }
            else:
                raise ValueError(f"No parseable challenges in output ({len(output_text)} chars)")

        except Exception as e:
            logger.error(f"DA LLM attempt {attempt + 1} failed: {e}")
            temperature += temp_increment

    return {
        "success": False,
        "error": "All retries failed",
        "challenges": [],
        "attempt": retries + 1,
        "generation_time_seconds": 0,
    }
