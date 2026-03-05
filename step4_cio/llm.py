"""
step4_cio/llm.py
CIO Agent — LLM Call Management
Spec: CIO Spec Teil 4

- System prompt (identical for draft + final)
- Draft user prompt builder
- Final user prompt builder
- API call with retry logic
- Briefing structure validation
"""

import json
import logging
import os

logger = logging.getLogger("cio_llm")


# ==========================================================================
# SYSTEM PROMPT (Spec Teil 4 §4.2 — identical for both runs)
# ==========================================================================

SYSTEM_PROMPT = """Du bist der CIO (Chief Investment Officer) eines quantitativen Multi-Agent-Trading-Systems. Dein Leser ist ein erfahrener Trader der folgende Systeme betreibt:

- V16 Global Macro RV System: Regime-Erkennung basierend auf Liquidity Cycle, 25 US-ETFs, automatisch via GitHub Actions
- F6 Stock Picker: Einzelaktien via SectorRarity + Heat, mit Covered Call Overlay, 21-Tage-Holding
- Eine Pipeline aus deterministischen und LLM-basierten Agents die das Portfolio ueberwachen

DEINE DREI FUNKTIONEN:

1. SYNTHESE: Was sagen die Systeme und Agents ZUSAMMEN? Du uebersetzt Einzeldatenpunkte in ein kohaerentes Gesamtbild.

2. CROSS-DOMAIN PATTERN RECOGNITION: Du erkennst Muster die kein einzelner Agent sehen kann, weil jeder nur seine Domaene kennt. Definierte Patterns (Klasse A) werden dir vom Pre-Processor geliefert — diese MUESSEN im Briefing erscheinen. Darueber hinaus darfst du eigene Beobachtungen machen (Klasse B), die du KLAR als "CIO OBSERVATION" kennzeichnest.

3. PRIORISIERUNG: Du sagst dem Operator wo er seine begrenzte Aufmerksamkeit investieren soll.

WAS DU BIST:
- Synthese-Agent, Interpretations-Layer, Priorisierungs-Instanz
- Du informierst, kontextualisierst, warnst, empfiehlst Pruefungen
- Du nennst IC-Quellen NAMENTLICH ("Macro Alf warnt...", "Howell verschiebt...")

WAS DU NICHT BIST:
- KEIN Entscheider. Du sagst NICHT "kaufe X" oder "verkaufe Y"
- KEIN Override. Du modifizierst KEINE V16- oder F6-Entscheidungen
- KEIN Risk Officer. Du stufst Alerts weder hoch noch runter ueber die offizielle Severity
- KEIN Zusammenfasser. Du fasst nicht nur auf — du synthetisierst und denkst

MASTER-SCHUTZ:
- V16-Gewichte sind SAKROSANKT. Nie als falsch bezeichnen, nie Aenderungen empfehlen
- F6-Signale sind SAKROSANKT. Nie als falsch bezeichnen
- Risk Officer Severities sind OFFIZIELL. Du darfst Kontext hinzufuegen, aber nicht herunterstufen oder heraufstufen
- Deine Rolle ist Interpretation und Kontext, nicht Korrektur anderer Agents

EPISTEMISCHE REGELN:
- V16 und Market Analyst teilen viele Datenquellen. Ihre Uebereinstimmung ist teilweise zirkulaer und hat BEGRENZTEN Bestaetigungswert
- IC-Intelligence basiert auf unabhaengigen qualitativen Quellen. Uebereinstimmung zwischen IC und V16/Market Analyst hat HOHEN Bestaetigungswert
- Risk Officer Alerts basieren auf Portfolio-Daten die von V16/F6-Gewichten abhaengen — keine unabhaengige Regime-Bestaetigung
- SAGE NICHT: "Drei Systeme bestaetigen Risk-On"
- SAGE: "V16 und Market Analyst zeigen Risk-On (geteilte Datenbasis). IC stuetzt die Richtung unabhaengig."

FUER QUANTITATIVE EINSCHAETZUNGEN: Verwende Market Analyst Layer Scores. Der ist die quantitative Autoritaet.
FUER QUALITATIVE KONTEXTUALISIERUNG: Verwende IC-Intelligence direkt. Das liefert das Narrativ.
BEHAUPTE NIE: Ein Layer Score sei falsch basierend auf deinem eigenen Read der IC-Claims.

DICHTE-REGELN:
1. Jeder Satz muss Information transportieren. Keine Filler. Direkt in medias res.
2. Pflichtinhalte sind Pflicht. Optionale Inhalte nur wenn relevant. Leere optionale Absaetze weglassen.
3. Wiederhole keine Information die in einer anderen Sektion steht. Referenziere ("siehe S3") statt zu wiederholen.
4. Laenge folgt Relevanz. ROUTINE-Tage kurz. ACTION-Tage lang. Das ergibt sich automatisch.
5. Zahlen statt Adjektive. NICHT: "leicht erhoeht." SONDERN: "42%, Schwelle 40%, +2pp."

VERBOTENE FORMULIERUNGEN:
- "Heute war ein interessanter Tag" → Starte direkt mit S1 Inhalt
- "Es gibt einiges zu besprechen" → Starte direkt mit S1 Inhalt
- "Die Maerkte waren volatil" → Nenne konkrete Zahlen
- "Ich empfehle zu kaufen/verkaufen" → "REVIEW: Pruefe mit Agent R ob..."
- "V16 liegt falsch weil..." → VERBOTEN (Master-Schutz)
- "Der Risk Officer uebertreibt" → VERBOTEN (Severities sind offiziell)
- "Leicht erhoeht/moderat/etwas" → Nenne die konkreten Zahlen

KEY ASSUMPTIONS BLOCK (nach S7, vor Schluss):
Benenne 2-3 zentrale Annahmen die dem heutigen Briefing zugrunde liegen.
Format pro Annahme:
KA[N]: [id] — [Annahme in einem Satz]
     Wenn falsch: [Was sich aendern wuerde]

OUTPUT-FORMAT:
Dein Output ist ein Briefing in 7 Sektionen + KEY ASSUMPTIONS. Halte die Reihenfolge exakt ein:
S1: DELTA
S2: CATALYSTS & TIMING
S3: RISK & ALERTS
S4: PATTERNS & SYNTHESIS
S5: INTELLIGENCE DIGEST
S6: PORTFOLIO CONTEXT
S7: ACTION ITEMS & WATCHLIST
KEY ASSUMPTIONS

Jede Sektion beginnt mit der Ueberschrift "## S1: DELTA" etc.
Schreibe KEINEN einleitenden Absatz vor S1. Starte direkt mit dem Header und dann S1."""


# ==========================================================================
# BRIEFING TYPE INSTRUCTIONS (Spec Teil 4 §4.3)
# ==========================================================================

BRIEFING_TYPE_INSTRUCTIONS = {
    "ROUTINE": """ROUTINE-TAG INSTRUKTION:
Heute ist ein ruhiger Tag. Nutze Sektion 4 (Patterns & Synthesis) fuer STRATEGISCHE PERSPEKTIVE.
Statt tagesaktuelle Pattern-Analyse: Welche laengerfristigen Trends sind relevant?
- Wie lange ist das aktuelle Regime aktiv? Ist das ungewoehnlich?
- Welche langsamen Trends akkumulieren sich (Router Proximity, Correlation, etc.)?
- Welche Threads sind seit Tagen aktiv ohne Resolution?
- Was hat sich ueber Wochen veraendert, aber nie einen Alert ausgeloest?
ROUTINE-Tage sollen WERTVOLLER sein, nicht langweiliger.
S1 und S3 koennen sehr kurz sein. S4 und S5 bekommen mehr Raum.
S7: "KEINE AKTION ERFORDERLICH." ist die erwartete Aussage.""",

    "WATCH": """WATCH-TAG INSTRUKTION:
Standard-Briefing. Alle Sektionen mit normaler Tiefe.
Fokus auf die WATCH-Trigger: Was hat den Tag von ROUTINE zu WATCH gehoben?
Benenne die Trigger explizit in S1 oder S4.""",

    "ACTION": """ACTION-TAG INSTRUKTION:
Handlungsbedarf. Fokussiere auf aktive Alerts und Action Items.
S3 (Risk) und S7 (Action Items) sind die wichtigsten Sektionen.
Strategische Perspektive ist sekundaer.
Jedes Action Item bekommt vollen Kontext: Was, Warum, Wie dringend, Naechste Schritte.""",

    "EMERGENCY": """EMERGENCY-TAG INSTRUKTION:
KRISE. Nur das Kritische. Jede Sektion so kurz wie moeglich.
S7 (Action Items) so detailliert wie noetig. Alle Items sind ACT-Level.
Strategische Perspektive entfaellt komplett.
Fokus: Was ist passiert? Was ist die unmittelbare Gefahr? Was muss der Operator JETZT tun?""",
}

MONDAY_INSTRUCTION = """
MONTAGS-SONDERMODUS:
Es ist Montag. Dein letztes Briefing war Freitag.
- Beziehe dich auf FREITAGS-STATUS als Baseline, nicht auf "gestern"
- IC-Claims die ueber das Wochenende eingegangen sind: Potentiell hoeher gewichten weil mehr Content akkumuliert
- Wenn verfuegbar: Relevante Pre-Market-Bewegungen von Freitag Close bis heute erwaehnen
- Formuliere Delta als "Seit Freitag:" nicht "Seit gestern:"
"""


# ==========================================================================
# PROMPT BUILDERS
# ==========================================================================

def build_system_prompt() -> str:
    """Return the system prompt (identical for draft and final)."""
    return SYSTEM_PROMPT


def build_draft_user_prompt(preprocessor_output: dict, inputs: dict,
                            yesterday_briefing: dict | None) -> str:
    """Build the user prompt for CIO Draft (Durchlauf 1). Spec Teil 4 §4.3."""
    header = preprocessor_output.get("header", {})
    history = preprocessor_output.get("history", {})

    # Briefing type instruction
    bt = header.get("briefing_type", "WATCH")
    bt_instruction = BRIEFING_TYPE_INSTRUCTIONS.get(bt, BRIEFING_TYPE_INSTRUCTIONS["WATCH"])

    # Monday extension
    if preprocessor_output.get("is_monday", False):
        bt_instruction += "\n" + MONDAY_INSTRUCTION

    # Yesterday highlights
    yesterday_hl = "Kein gestriges Briefing verfuegbar (erster Run oder nicht geladen)."
    if yesterday_briefing and yesterday_briefing.get("briefing_text"):
        # Extract first 500 chars as highlights
        yt = yesterday_briefing["briefing_text"]
        yesterday_hl = (
            f"Gestriges Briefing ({yesterday_briefing.get('date', '?')}): "
            f"{yesterday_briefing.get('briefing_type', '?')} | "
            f"Conviction: {yesterday_briefing.get('system_conviction', '?')}\n"
            f"{yt[:500]}..."
        )

    # Truncate IC high_novelty_claims to top 20 for token budget
    ic = inputs.get("ic_intelligence", {})
    ic_truncated = {**ic}
    claims = ic_truncated.get("high_novelty_claims", [])
    if len(claims) > 20:
        ic_truncated["high_novelty_claims"] = claims[:20]
        ic_truncated["_claims_truncated"] = f"Showing top 20 of {len(claims)}"

    prompt = f"""Schreibe das heutige CIO Briefing (DRAFT).

Schreibe das bestmoegliche Briefing. Es wird anschliessend einem Stresstest durch den Devil's Advocate unterzogen.

=== HEADER (vom Pre-Processor — uebernimm exakt in der ersten Zeile) ===
Datum: {preprocessor_output.get('date', '')}
Briefing-Typ: {header.get('briefing_type', 'WATCH')}
System Conviction: {header.get('system_conviction', 'MODERATE')}
Risk Ampel: {header.get('risk_ampel', 'GREEN')}
Fragility State: {header.get('fragility_state', 'HEALTHY')}
Data Quality: {header.get('data_quality', 'DEGRADED')}
V16 Regime: {header.get('v16_regime', 'UNKNOWN')}
Referenzdatum (fuer Delta): {preprocessor_output.get('reference_date', '')}
Ist Montag: {preprocessor_output.get('is_monday', False)}

=== PRE-PROCESSOR ERGEBNISSE ===

AKTIVE PATTERNS (Klasse A — MUESSEN im Briefing in S4 erscheinen):
{_to_json(preprocessor_output.get('patterns', {}).get('class_a_active', []))}

ANTI-PATTERNS (Claims die trotz hoher Novelty kein Signal sind):
{_to_json(preprocessor_output.get('patterns', {}).get('anti_patterns', []))}

ABSENZ-FLAGS:
{_to_json(preprocessor_output.get('absence_flags', []))}

ONGOING CONDITIONS (komprimiert — in S3 als Einzeiler behandeln):
{_to_json(preprocessor_output.get('alert_treatment', {}).get('compressed_ongoing', []))}

CONFIDENCE MARKERS (NICHT im Briefing-Text zeigen, aber beruecksichtigen):
{_to_json(preprocessor_output.get('confidence_markers', []))}

OFFENE ACTION ITEMS AUS VORTAGEN:
{_to_json(history.get('open_action_items', []))}

AKTIVE THREADS (Multi-Tage-Themen fuer Trend-Aussagen):
{_to_json(history.get('active_threads', []))}

RESOLVED THREADS LETZTE 7 TAGE:
{_to_json(history.get('resolved_threads_7d', []))}

=== V16 PRODUCTION ===
{_to_json(inputs.get('v16_production', {}))}

=== F6 PRODUCTION ===
{_to_json(inputs.get('f6_production', {"status": "UNAVAILABLE"}))}

=== RISK OFFICER ALERTS ===
{_to_json(inputs.get('risk_alerts', {}))}

=== SIGNAL GENERATOR ===
{_to_json(inputs.get('signals', {"status": "UNAVAILABLE"}))}

=== MARKET ANALYST ===
Layer Analysis:
{_to_json(inputs.get('layer_analysis', {"status": "UNAVAILABLE"}))}

Beliefs:
{_to_json(inputs.get('beliefs', {"status": "UNAVAILABLE"}))}

=== IC INTELLIGENCE ===
{_to_json(ic_truncated)}

=== TEMPORAL CONTEXT ===
{_to_json(preprocessor_output.get('temporal_context', {}))}

=== GESTRIGES BRIEFING (Highlights fuer Kontinuitaet) ===
{yesterday_hl}

=== BRIEFING-TYP-SPEZIFISCHE INSTRUKTION ===
{bt_instruction}

Schreibe jetzt das Briefing. 7 Sektionen + KEY ASSUMPTIONS, Header zuerst. Halte die Pflichtinhalte ein."""

    return prompt


def build_final_user_prompt(preprocessor_output: dict, inputs: dict,
                            draft_briefing_text: str,
                            devils_advocate: dict) -> str:
    """Build the user prompt for CIO Final (Durchlauf 2). Spec Teil 4 §4.4."""
    header = preprocessor_output.get("header", {})
    bt = header.get("briefing_type", "WATCH")
    bt_instruction = BRIEFING_TYPE_INSTRUCTIONS.get(bt, BRIEFING_TYPE_INSTRUCTIONS["WATCH"])

    # Truncate IC for token budget
    ic = inputs.get("ic_intelligence", {})
    ic_truncated = {**ic}
    claims = ic_truncated.get("high_novelty_claims", [])
    if len(claims) > 20:
        ic_truncated["high_novelty_claims"] = claims[:20]

    prompt = f"""Du bist der CIO. Du hast einen DRAFT geschrieben. Ein unabhaengiger Analyst (Devil's Advocate) hat Gegenargumente formuliert.

Deine Aufgabe: Bewerte die Gegenargumente auf ihre Substanz und erstelle das FINAL Briefing.

REGELN:
1. Wenn ein Gegenargument SUBSTANTIELL ist — also durch Daten in deinen Inputs gestuetzt wird — passe den betroffenen Abschnitt an. Markiere die Aenderung mit einem DA-Marker.
2. Wenn ein Gegenargument VALIDE aber nicht stark genug ist um das Briefing zu aendern — setze es auf die Watchlist. Markiere als NOTED.
3. Wenn ein Gegenargument NICHT durch Daten gestuetzt ist — weise es explizit zurueck mit Begruendung. Markiere als REJECTED.
4. Deine Aufgabe ist NICHT Konsens mit dem Devil's Advocate. Deine Aufgabe ist das bestmoegliche Briefing.
5. Unberuehrte Sektionen bleiben IDENTISCH zum Draft. Aendere NUR was der Devil's Advocate substantiell in Frage gestellt hat.

DA-MARKER FORMAT (inline in der betroffenen Sektion):
[DA: {{Zusammenfassung des Einwands}}. {{ACCEPTED/NOTED/REJECTED}} — {{Begruendung/Auswirkung}}. Original Draft: "{{Originaler Text}}"]

Am Ende des Briefings: DA RESOLUTION SUMMARY als Appendix.

=== DEIN DRAFT ===
{draft_briefing_text}

=== DEVIL'S ADVOCATE CHALLENGES ===
{_to_json(devils_advocate.get('challenges', []))}

=== CONFIDENCE MARKERS (vom Pre-Processor — zeigen wo LOW confidence war) ===
{_to_json(preprocessor_output.get('confidence_markers', []))}

=== ALLE ORIGINAL-INPUTS (identisch zum Draft-Durchlauf) ===

--- PRE-PROCESSOR ERGEBNISSE ---
{_to_json(preprocessor_output)}

--- V16 PRODUCTION ---
{_to_json(inputs.get('v16_production', {}))}

--- F6 PRODUCTION ---
{_to_json(inputs.get('f6_production', {"status": "UNAVAILABLE"}))}

--- RISK OFFICER ALERTS ---
{_to_json(inputs.get('risk_alerts', {}))}

--- SIGNAL GENERATOR ---
{_to_json(inputs.get('signals', {"status": "UNAVAILABLE"}))}

--- MARKET ANALYST ---
{_to_json(inputs.get('layer_analysis', {"status": "UNAVAILABLE"}))}
{_to_json(inputs.get('beliefs', {"status": "UNAVAILABLE"}))}

--- IC INTELLIGENCE ---
{_to_json(ic_truncated)}

--- TEMPORAL CONTEXT ---
{_to_json(preprocessor_output.get('temporal_context', {}))}

=== BRIEFING-TYP-SPEZIFISCHE INSTRUKTION ===
{bt_instruction}

Schreibe jetzt das FINAL Briefing. Gleiche 7 Sektionen + KEY ASSUMPTIONS + DA Resolution Summary am Ende.
Header ist identisch zum Draft.
Aendere NUR Sektionen die vom Devil's Advocate substantiell betroffen sind."""

    return prompt


# ==========================================================================
# LLM API CALL WITH RETRY (Spec Teil 4 §4.9)
# ==========================================================================

def call_cio_llm(system_prompt: str, user_prompt: str, llm_config: dict) -> dict:
    """
    LLM call with retry logic.
    On failure: increase temperature and retry.
    After max retries: return failure for fallback.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed")
        return {"success": False, "error": "anthropic not installed"}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return {"success": False, "error": "ANTHROPIC_API_KEY not set"}

    client = anthropic.Anthropic(api_key=api_key)

    model = llm_config.get("model", "claude-sonnet-4-5-20250929")
    temperature = llm_config.get("temperature", 0.3)
    max_tokens = llm_config.get("max_tokens", 8000)
    max_retries = llm_config.get("retry_on_failure", 2)
    temp_increment = llm_config.get("retry_temperature_increment", 0.1)
    timeout = llm_config.get("timeout_seconds", 180)

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                f"LLM call attempt {attempt + 1}/{max_retries + 1}, "
                f"temp={temperature:.1f}, model={model}"
            )

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=timeout,
            )

            briefing_text = response.content[0].text

            # Basic structure validation
            if _validate_briefing_structure(briefing_text):
                logger.info(
                    f"LLM success on attempt {attempt + 1}. "
                    f"Output: {len(briefing_text)} chars, "
                    f"{len(briefing_text.split())} words"
                )
                return {
                    "success": True,
                    "briefing_text": briefing_text,
                    "attempt": attempt + 1,
                    "model": model,
                    "temperature": temperature,
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                }
            else:
                missing = _get_missing_sections(briefing_text)
                logger.warning(
                    f"Attempt {attempt + 1}: structure invalid, "
                    f"missing sections: {missing}"
                )
                raise ValueError(f"Missing sections: {missing}")

        except Exception as e:
            logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
            temperature += temp_increment

    return {
        "success": False,
        "error": "All retries failed",
        "attempts": max_retries + 1,
    }


# ==========================================================================
# HELPERS
# ==========================================================================

def _validate_briefing_structure(text: str) -> bool:
    """Check if all 7 sections are present."""
    required = ["## S1:", "## S2:", "## S3:", "## S4:", "## S5:", "## S6:", "## S7:"]
    return all(section in text for section in required)


def _get_missing_sections(text: str) -> list:
    """Return list of missing section markers."""
    required = ["## S1:", "## S2:", "## S3:", "## S4:", "## S5:", "## S6:", "## S7:"]
    return [s for s in required if s not in text]


def _to_json(obj) -> str:
    """Safe JSON serialization for prompt insertion."""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)
