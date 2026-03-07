"""
step7_execution_advisor/llm.py
Execution Advisor LLM Text Generation.

Generates the human-readable Execution Briefing via Sonnet.
Score and Confirming/Conflicting are already computed deterministically.
The LLM only writes the text summary + recommendation.

Includes: Prompt building, LLM call with retry, response parsing, fallback.

Source: Trading Desk Spec Teil 6 §31
"""

import logging
import re
from datetime import date, timedelta

from shared.llm import call_anthropic

logger = logging.getLogger("execution_advisor.llm")


# =============================================================================
# SYSTEM PROMPT (Spec Teil 6 §31.2 — verbatim + V96 Ergänzungen)
# =============================================================================

SYSTEM_PROMPT_ADVISOR = """Du bist der Execution Advisor von Baldur Creek Capital, einem systematischen Macro Hedge Fund.

DEINE ROLLE:
Du schreibst ein taegliches Execution Briefing. Dein Job ist NICHT die Marktlage zu analysieren (das macht der CIO). Dein Job ist die Frage zu beantworten: "Soll ich V16's Rebalance HEUTE ausfuehren?"

Du bekommst:
- Den Execution Score (deterministisch berechnet, 0-18)
- Confirming Signale (was FÜR Ausfuehrung spricht)
- Conflicting Signale (was GEGEN Ausfuehrung spricht)
- V16 Positionen und Trades
- Pipeline-Kontext (Risk Officer, CIO, Router)
- Event-Kalender (was in 48h/14d kommt)
- Aktuelle Marktdaten (DW Sheet Werte + ETF-Preise)

REGELN — STRIKT EINHALTEN:

1. IMMER Confirming UND Conflicting zeigen. BEIDE Seiten. Nie einseitig. Auch wenn Score 0 ist: zeige trotzdem die positiven Signale. Auch wenn Score 15 ist: zeige trotzdem was gut laeuft.

2. Schreibe auf Deutsch. Der Operator (Richie) ist deutschsprachig.

3. Maximal 600 Woerter gesamt. Kein Fuelltext. Jeder Satz muss Information enthalten.

4. Du overrulst V16 NICHT. V16 entscheidet WAS gehalten wird. Du empfiehlst nur:
   - WANN ausfuehren (heute, morgen, nach Event?)
   - WIE ausfuehren (Market Orders, Limit Orders, Position-Size?)
   - OB der Zeitpunkt optimal ist

5. Verwende die EXAKTEN Zahlen aus den Daten. Keine Schaetzungen, keine Rundungen die du selbst machst. Wenn COT Gold -47.8% ist, schreib -47.8%, nicht "fast -50%".

6. "Would Change My Mind" Abschnitt MUSS konkrete, messbare Bedingungen enthalten die auf den bereitgestellten AKTUELLEN MARKTDATEN basieren. Verwende NUR Werte und Levels die in den Daten stehen. ERFINDE KEINE Preisniveaus aus deinem Training. Beispiel: Wenn VIX aktuell bei 19.5 ist, schreibe "VIX > 28" (sinnvoller Schwellenwert relativ zum aktuellen Wert). Wenn du einen Spot-Preis NICHT in den Daten findest, verwende stattdessen relative Bedingungen ("Gold faellt >5%") oder die bereitgestellten ETF-Preise ("GLD faellt unter X").

7. Wenn V16 heute KEINE materiellen Trades hat (alle HOLD, Delta < 1%): Sage das explizit. "Kein materielles Rebalancing noetig. Die Empfehlung bezieht sich auf den Fall dass V16 morgen Regime wechselt."

8. Bei EXECUTE (Score 0-3): Halte dich KURZ. 2-3 Saetze Bestaetigung, 1-2 Saetze was beobachten. Kein Drama wenn alles gruen ist.

9. Bei WAIT oder HOLD (Score 7+): Sei SPEZIFISCH warum und wie lange. "Warte bis FOMC-Ergebnis (morgen 14:00 ET)" nicht "warte auf Klarheit".

10. Nenne in der Empfehlung IMMER spezifische Assets aus dem Portfolio. Nicht "vorsichtig sein" sondern "HYG: Limit-Orders setzen, DBC: Slippage-Risiko bei hohem Volumen".

11. Bei "Would Change My Mind" Triggern: Verwende die bereitgestellten ETF-Preise (GLD, DXY-Return, SPY, HYG etc.) und DW-Indikatoren (VIX, HY OAS, MOVE etc.) als Referenz. NIEMALS Spot-Preise erfinden die nicht in den Daten stehen. Wenn ein bestimmter Spot-Preis nicht verfuegbar ist, formuliere die Bedingung relativ ("GLD faellt >5% vom aktuellen Kurs {aktueller_kurs}") oder ueber verfuegbare Indikatoren ("HY OAS steigt ueber 350bps").

FORMAT — EXAKT EINHALTEN:

=== EXECUTION ASSESSMENT: {LEVEL} (Score: {score}/{max}) ===

CONFIRMING:
• [Signal]: [1-Satz Detail mit konkreten Zahlen]
• [Signal]: [1-Satz Detail]
• ...
(Alle relevanten Confirming-Signale, mindestens 3, maximal 8)

CONFLICTING:
• [Signal]: [1-Satz Detail mit konkreten Zahlen]
• [Signal]: [1-Satz Detail]
• ...
(Alle relevanten Conflicting-Signale, mindestens 1, maximal 8. Wenn keine: "• Keine Conflicting-Signale identifiziert.")

NETTO: [EXAKT 1 Satz. Format: "X Confirming vs Y Conflicting. {Kernaussage}."]

EMPFEHLUNG:
[2-5 Saetze. Konkret. Spezifische Assets nennen. Timing angeben.]

WOULD CHANGE MY MIND:
• Execute sofort wenn: [messbare Bedingung basierend auf bereitgestellten Daten]
• Execute sofort wenn: [messbare Bedingung] (optional)
• Eskaliere zu Wait/Hold wenn: [messbare Bedingung basierend auf bereitgestellten Daten]
• Eskaliere zu Wait/Hold wenn: [messbare Bedingung] (optional)"""


# =============================================================================
# USER PROMPT TEMPLATE (Spec Teil 6 §31.3 + V96 Marktdaten-Block)
# =============================================================================

USER_PROMPT_TEMPLATE_ADVISOR = """Heute: {today} ({weekday})

=== V16 KONTEXT ===
Regime: {v16_regime} (State: {v16_state})
Top-5 Positionen:
{top5_formatted}

Rebalance-Trades heute:
{rebalance_formatted}
Materielles Rebalancing: {material_rebalance}

=== AKTUELLE MARKTDATEN (fuer "Would Change My Mind" Referenzwerte) ===
{market_data_formatted}

=== EXECUTION SCORE: {total_score}/{max_score} → {execution_level} ===
Veto: {veto_info}

DIMENSIONEN:
1. Event Risk:        {event_score}/3 — {event_label}
2. Positioning:       {pos_score}/3 — {pos_label}
3. Liquidity:         {liq_score}/3 — {liq_label}
4. Cross-Asset:       {cross_score}/3 — {cross_label}
5. GEX Regime:        {gex_score}/3 — {gex_label}
6. Sentiment:         {sent_score}/3 — {sent_label}

=== CONFIRMING SIGNALE ({confirming_count}) ===
{confirming_formatted}

=== CONFLICTING SIGNALE ({conflicting_count}) ===
{conflicting_formatted}

=== PIPELINE-KONTEXT ===
Risk Officer Ampel: {risk_ampel}
Risk Alerts aktiv: {risk_alert_count}
Fragility State: {fragility_state}
CIO Conviction: {cio_conviction}
CIO Briefing Typ: {briefing_type}
Router State: {router_state} (Max Proximity: {max_proximity})

=== EVENT WINDOW ===
Naechste 48h: {events_48h}
Naechste 14d: {events_14d_count} Events
Convergence Week: {convergence}

WICHTIG: Verwende bei "Would Change My Mind" NUR die oben bereitgestellten Marktdaten als Referenzwerte. Erfinde KEINE Preisniveaus.

Schreibe das Execution Briefing im vorgegebenen Format."""


# =============================================================================
# MARKET DATA FORMATTER (V96 NEU)
# =============================================================================

def _format_market_data(dw_data: dict, v16_data: dict) -> str:
    """
    Format current market data for LLM prompt.
    Provides actual values so LLM doesn't hallucinate price levels.
    """
    lines = []

    # --- ETF Prices from V16 Top-5 ---
    weights = v16_data.get("current_weights", {})
    top5 = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    if top5:
        lines.append("ETF-Preise (Top-5 Portfolio-Positionen):")
        # Prices come from signal generator v16_trades
        prices = v16_data.get("latest_prices", {})
        for asset, weight in top5:
            price = prices.get(asset)
            if price is not None:
                lines.append(f"  {asset}: ${price:.2f} (Gewicht: {weight:.1%})")
            else:
                lines.append(f"  {asset}: Preis nicht verfuegbar (Gewicht: {weight:.1%})")

    # --- DW Indicators ---
    # L2 Sentiment
    dw_lines = []
    _add_dw_line(dw_lines, dw_data, "VIX_LEVEL", "VIX")
    _add_dw_line(dw_lines, dw_data, "HY_OAS_SPREAD", "HY OAS Spread (bps)")
    _add_dw_line(dw_lines, dw_data, "CNN_FEAR_GREED", "CNN Fear & Greed")
    _add_dw_line(dw_lines, dw_data, "PUT_CALL_RATIO", "Put/Call Ratio")
    _add_dw_line(dw_lines, dw_data, "MOVE_INDEX", "MOVE Index")
    _add_dw_line(dw_lines, dw_data, "INSIDER_BUY_SELL", "Insider Buy/Sell")

    # L4 Positioning
    _add_dw_line(dw_lines, dw_data, "COT_SP500_COMM_NET", "COT S&P500 Commercial Net (%)")
    _add_dw_line(dw_lines, dw_data, "COT_GOLD_COMM_NET", "COT Gold Commercial Net (%)")
    _add_dw_line(dw_lines, dw_data, "COT_TREASURY_COMM_NET", "COT Treasury Commercial Net (%)")
    _add_dw_line(dw_lines, dw_data, "OPTIONS_GEX", "Options GEX ($B)")
    _add_dw_line(dw_lines, dw_data, "FUND_FLOWS_EQUITY", "Fund Flows Equity")

    # L5 Fragility
    _add_dw_line(dw_lines, dw_data, "FIN_STRESS_INDEX", "OFR Financial Stress Index")
    _add_dw_line(dw_lines, dw_data, "SOFR_FFR_SPREAD", "SOFR-FFR Spread")
    _add_dw_line(dw_lines, dw_data, "LIQUIDITY_AMIHUD", "Liquidity (Amihud)")
    _add_dw_line(dw_lines, dw_data, "VIX_TERM_STRUCTURE", "VIX Term Structure")

    # L7 Cross-Asset
    _add_dw_line(dw_lines, dw_data, "GOLD_RETURN_20D", "Gold Return 20d (%)")
    _add_dw_line(dw_lines, dw_data, "DXY_RETURN_20D", "DXY Return 20d (%)")
    _add_dw_line(dw_lines, dw_data, "COPPER_SMA50_TREND", "Copper vs SMA50 Trend")
    _add_dw_line(dw_lines, dw_data, "SPY_SMA50_TREND", "SPY vs SMA50 Trend")
    _add_dw_line(dw_lines, dw_data, "REAL_YIELD_10Y_TREND", "Real Yield 10Y Trend")
    _add_dw_line(dw_lines, dw_data, "YIELD_CURVE_10Y2Y", "Yield Curve 10Y-2Y")

    if dw_lines:
        lines.append("")
        lines.append("DW-Indikatoren (aktuell):")
        lines.extend(dw_lines)

    if not lines:
        return "(Keine Marktdaten verfuegbar — verwende relative Bedingungen)"

    return "\n".join(lines)


def _add_dw_line(lines: list, dw_data: dict, key: str, label: str):
    """Add a DW indicator line if available."""
    value = dw_data.get(key)
    if value is not None and value != "":
        lines.append(f"  {label}: {value}")


# =============================================================================
# PROMPT BUILDER (Spec Teil 6 §31.4)
# =============================================================================

def build_advisor_user_prompt(
    scoring_result: dict,
    cc_result: dict,
    v16_data: dict,
    risk_officer: dict,
    cio_final: dict,
    router_output: dict,
    event_window: dict,
    today: date,
    dw_data: dict = None,
) -> str:
    """Build the complete user prompt for the Execution Advisor LLM call."""

    dw_data = dw_data or {}

    # V16 Top-5
    weights = v16_data.get("current_weights", {})
    top5 = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top5_formatted = "\n".join(
        f"  {asset}: {weight:.1%}" for asset, weight in top5
    ) or "  (keine Daten)"

    # Rebalance Trades
    trades = v16_data.get("weight_deltas", {})
    material_trades = {a: d for a, d in trades.items() if abs(d) > 0.01}
    if material_trades:
        rebalance_formatted = "\n".join(
            f"  {asset}: {'+' if delta > 0 else ''}{delta:.2%} → "
            f"{'BUY' if delta > 0 else 'SELL'}"
            for asset, delta in material_trades.items()
        )
        material_rebalance = f"JA — {len(material_trades)} materielle Trades"
    else:
        rebalance_formatted = "  Keine materiellen Trades (alle HOLD, Delta < 1%)"
        material_rebalance = "NEIN — alle Positionen HOLD"

    # Dimensions
    dims = scoring_result["dimensions"]

    # Confirming/Conflicting formatted
    confirming_formatted = "\n".join(
        f"  • {c['signal']}: {c['detail']}"
        for c in cc_result["confirming"]
    ) or "  (keine)"

    conflicting_formatted = "\n".join(
        f"  • {c['signal']}: {c['detail']}"
        for c in cc_result["conflicting"]
    ) or "  (keine)"

    # Events 48h
    events_48h = event_window.get("next_48h", [])
    if events_48h:
        events_48h_str = ", ".join(
            f"{e['event']} (in {e['hours_until']}h, {e['impact']})"
            for e in events_48h
        )
    else:
        events_48h_str = "Keine HIGH-Impact Events"

    # Veto
    if scoring_result.get("veto_applied"):
        veto_info = f"JA — {scoring_result['veto_reason']}"
    else:
        veto_info = "Nein"

    # Market data block (V96 NEU)
    market_data_formatted = _format_market_data(dw_data, v16_data)

    return USER_PROMPT_TEMPLATE_ADVISOR.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
        v16_regime=v16_data.get("regime", "UNKNOWN"),
        v16_state=v16_data.get("macro_state_name", "UNKNOWN"),
        top5_formatted=top5_formatted,
        rebalance_formatted=rebalance_formatted,
        material_rebalance=material_rebalance,
        market_data_formatted=market_data_formatted,
        total_score=scoring_result["total_score"],
        max_score=scoring_result["max_possible"],
        execution_level=scoring_result["execution_level"],
        veto_info=veto_info,
        event_score=dims["event_risk"]["score"],
        event_label=dims["event_risk"].get("label", ""),
        pos_score=dims["positioning_conflict"]["score"],
        pos_label=dims["positioning_conflict"].get("label", ""),
        liq_score=dims["liquidity_risk"]["score"],
        liq_label=dims["liquidity_risk"].get("label", ""),
        cross_score=dims["cross_asset_confirmation"]["score"],
        cross_label=dims["cross_asset_confirmation"].get("label", ""),
        gex_score=dims["gex_regime"]["score"],
        gex_label=dims["gex_regime"].get("label", ""),
        sent_score=dims["sentiment_extreme"]["score"],
        sent_label=dims["sentiment_extreme"].get("label", ""),
        confirming_count=cc_result["confirming_count"],
        confirming_formatted=confirming_formatted,
        conflicting_count=cc_result["conflicting_count"],
        conflicting_formatted=conflicting_formatted,
        risk_ampel=risk_officer.get("risk_ampel", "UNKNOWN"),
        risk_alert_count=risk_officer.get("active_alert_count", 0),
        fragility_state=risk_officer.get("fragility_state", "UNKNOWN"),
        cio_conviction=cio_final.get("conviction", "UNKNOWN"),
        briefing_type=cio_final.get("briefing_type", "UNKNOWN"),
        router_state=router_output.get("current_state", "UNKNOWN"),
        max_proximity=f"{router_output.get('max_proximity', 0):.0%}",
        events_48h=events_48h_str,
        events_14d_count=event_window.get("event_density_14d", 0),
        convergence="JA" if event_window.get("convergence_weeks") else "NEIN",
    )


# =============================================================================
# RESPONSE PARSING (Spec Teil 6 §31.5)
# =============================================================================

def parse_advisor_llm_response(response: dict) -> dict:
    """
    Parse LLM Execution Briefing response.

    Extracts:
    - briefing_text: Full text
    - recommendation_short: First sentence of EMPFEHLUNG
    - specific_actions: List of specific actions
    - would_change_my_mind: Execute-if / Hold-if
    """
    # Extract text
    text_parts = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])

    briefing_text = "\n".join(text_parts).strip()

    if not briefing_text:
        return {
            "briefing_text": "",
            "recommendation_short": "",
            "specific_actions": [],
            "would_change_my_mind": {"execute_if": [], "hold_if": []},
            "llm_raw_length": 0,
            "parse_success": False,
        }

    # Extract recommendation
    recommendation_short = _extract_section_first_sentence(
        briefing_text, "EMPFEHLUNG:"
    )

    # Extract specific actions (bullet points under EMPFEHLUNG)
    specific_actions = _extract_bullet_points(
        briefing_text, "EMPFEHLUNG:"
    )

    # Would Change My Mind
    execute_if = _extract_conditional_triggers(
        briefing_text, "Execute sofort wenn:"
    )
    hold_if = _extract_conditional_triggers(
        briefing_text, "Eskaliere zu Wait/Hold wenn:"
    )
    # Fallback patterns
    if not hold_if:
        hold_if = _extract_conditional_triggers(
            briefing_text, "Wait/Hold wenn:"
        )
    if not hold_if:
        hold_if = _extract_conditional_triggers(
            briefing_text, "Eskaliere zu Hold wenn:"
        )

    return {
        "briefing_text": briefing_text,
        "recommendation_short": recommendation_short,
        "specific_actions": specific_actions,
        "would_change_my_mind": {
            "execute_if": execute_if,
            "hold_if": hold_if,
        },
        "llm_raw_length": len(briefing_text),
        "parse_success": True,
    }


def _extract_section_first_sentence(text: str, section_header: str) -> str:
    """Extract first sentence after a section header."""
    idx = text.find(section_header)
    if idx < 0:
        return ""
    after = text[idx + len(section_header):].strip()
    # First sentence: until first period followed by space or newline
    match = re.match(r'^(.*?[.!])\s', after)
    if match:
        return match.group(1).strip()
    # Fallback: first line
    first_line = after.split("\n")[0].strip()
    return first_line


def _extract_bullet_points(text: str, section_header: str) -> list[str]:
    """Extract bullet points from a section."""
    idx = text.find(section_header)
    if idx < 0:
        return []

    after = text[idx + len(section_header):]
    # Find next section header
    next_section = re.search(r'\n(?:===|[A-ZÄÖÜ]{3,}:|\nWOULD)', after)
    if next_section:
        section_text = after[:next_section.start()]
    else:
        section_text = after

    bullets = []
    for line in section_text.split("\n"):
        line = line.strip()
        if line.startswith("•") or line.startswith("-") or line.startswith("●"):
            bullet = re.sub(r'^[•\-●]\s*', '', line).strip()
            if bullet:
                bullets.append(bullet)

    return bullets


def _extract_conditional_triggers(text: str, prefix: str) -> list[str]:
    """Extract conditional triggers (Execute wenn / Hold wenn)."""
    triggers = []
    for line in text.split("\n"):
        line = line.strip()
        clean = re.sub(r'^[•\-●]\s*', '', line)
        if clean.lower().startswith(prefix.lower()):
            condition = clean[len(prefix):].strip()
            if condition:
                triggers.append(condition)
    return triggers


# =============================================================================
# LLM CALL WITH RETRY + FALLBACK (Spec Teil 6 §31.6)
# =============================================================================

def generate_execution_briefing(
    scoring_result: dict,
    cc_result: dict,
    v16_data: dict,
    risk_officer: dict,
    cio_final: dict,
    router_output: dict,
    event_window: dict,
    today: date,
    config: dict = None,
    dw_data: dict = None,
) -> dict:
    """
    Generate Execution Briefing via LLM with retry and fallback.

    Returns:
        {
            "briefing_text": str,
            "recommendation_short": str,
            "specific_actions": list[str],
            "would_change_my_mind": dict,
            "llm_used": bool,
            "llm_fallback": bool,
            "llm_model": str,
        }
    """
    config = config or {}
    llm_config = config.get("llm", {})

    model = llm_config.get("model", "claude-sonnet-4-5-20250929")
    temperature = llm_config.get("temperature", 0.2)
    max_tokens = llm_config.get("max_tokens", 3000)
    retry_count = llm_config.get("retry_on_failure", 1)
    retry_temp_inc = llm_config.get("retry_temperature_increment", 0.1)
    timeout = llm_config.get("timeout_seconds", 120)

    # Build user prompt (now includes dw_data for market data block)
    user_prompt = build_advisor_user_prompt(
        scoring_result, cc_result, v16_data, risk_officer,
        cio_final, router_output, event_window, today,
        dw_data=dw_data,
    )

    for attempt in range(1, retry_count + 2):
        try:
            logger.info(
                f"Execution Advisor LLM call attempt {attempt} "
                f"(temp={temperature})"
            )

            response = call_anthropic(
                system_prompt=SYSTEM_PROMPT_ADVISOR,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            parsed = parse_advisor_llm_response(response)

            if not parsed["parse_success"]:
                logger.warning(f"LLM response parse failed, attempt {attempt}")
                temperature += retry_temp_inc
                continue

            # Validation: briefing must be at least 100 chars
            if len(parsed["briefing_text"]) < 100:
                logger.warning(
                    f"Briefing too short ({len(parsed['briefing_text'])} chars)"
                )
                temperature += retry_temp_inc
                continue

            # Validation: must contain CONFIRMING and CONFLICTING sections
            text = parsed["briefing_text"]
            if "CONFIRMING" not in text or "CONFLICTING" not in text:
                logger.warning("Briefing missing CONFIRMING/CONFLICTING sections")
                temperature += retry_temp_inc
                continue

            logger.info(
                f"Execution Briefing generated: {len(text)} chars, "
                f"{len(parsed['specific_actions'])} actions"
            )

            return {
                **parsed,
                "llm_used": True,
                "llm_fallback": False,
                "llm_model": model,
            }

        except Exception as e:
            logger.error(f"LLM attempt {attempt} failed: {e}")
            temperature += retry_temp_inc

    # Fallback
    logger.warning("All LLM attempts failed — using fallback text")
    fallback_text = generate_fallback_text(scoring_result, cc_result)

    return {
        "briefing_text": fallback_text,
        "recommendation_short": _fallback_recommendation(
            scoring_result["execution_level"]
        ),
        "specific_actions": [],
        "would_change_my_mind": {"execute_if": [], "hold_if": []},
        "llm_used": False,
        "llm_fallback": True,
        "llm_model": model,
        "llm_raw_length": len(fallback_text),
        "parse_success": True,
    }


# =============================================================================
# FALLBACK TEXT (Spec Teil 4 §16.5)
# =============================================================================

def generate_fallback_text(scoring_result: dict, cc_result: dict) -> str:
    """Fallback Execution Briefing when LLM call fails. Template-based."""
    level = scoring_result["execution_level"]
    score = scoring_result["total_score"]
    max_s = scoring_result["max_possible"]
    cc = cc_result["confirming_count"]
    cf = cc_result["conflicting_count"]

    lines = [
        f"=== EXECUTION ASSESSMENT: {level} (Score: {score}/{max_s}) ===",
        "",
        f"CONFIRMING ({cc}):",
    ]
    for c in cc_result["confirming"][:5]:
        lines.append(f"  • {c['signal']}: {c['detail']}")

    lines.append(f"\nCONFLICTING ({cf}):")
    for c in cc_result["conflicting"][:5]:
        lines.append(f"  • {c['signal']}: {c['detail']}")

    lines.append(f"\nNETTO: {cc} Confirming vs {cf} Conflicting.")

    rec = _fallback_recommendation(level)
    lines.append(f"\nEMPFEHLUNG: {rec}")

    lines.append("\n(Fallback-Text — LLM nicht verfügbar)")

    return "\n".join(lines)


def _fallback_recommendation(level: str) -> str:
    """Fallback recommendation text per execution level."""
    return {
        "EXECUTE": "V16 Rebalance normal ausfuehren. Keine besonderen Vorkehrungen noetig.",
        "CAUTION": "Ausfuehren mit Limit-Orders. Position-Size pruefen vor Execution.",
        "WAIT": "Nicht heute rebalancen. Warte auf Event-Aufloesung oder Positioning-Shift.",
        "HOLD": "Aktiv NICHT rebalancen. Mehrere Risikofaktoren aktiv.",
    }.get(level, "Execution Level unbekannt.")
