"""
step_0k_event_calendar/updater.py
LLM + Web Search based event calendar updater.

Handles:
  - LLM call with web_search tool
  - Robust response parsing (markdown, trailing commas, validation)
  - YAML merge logic (keep past, replace future, dedup)

Source: Trading Desk Spec Teil 2 §6.2, Teil 6 §30
"""

import json
import logging
import re
from datetime import date, datetime, timedelta

from shared.llm import call_anthropic

logger = logging.getLogger("event_calendar.updater")


# =============================================================================
# LLM CONFIG (Spec Teil 6 §30.1)
# =============================================================================

LLM_CONFIG_CALENDAR = {
    "model": "claude-sonnet-4-5-20250929",
    "temperature": 0.1,
    "max_tokens": 12000,
    "tools": [
        {
            "type": "web_search_20250305",
            "name": "web_search",
        }
    ],
    "retry_on_failure": 2,
    "retry_temperature_increment": 0.05,
    "timeout_seconds": 300,
}


# =============================================================================
# SYSTEM PROMPT (Spec Teil 6 §30.2 — verbatim)
# =============================================================================

SYSTEM_PROMPT_CALENDAR = """Du bist ein Daten-Extraktor fuer den Baldur Creek Capital Event Calendar. Du arbeitest fuer einen systematischen Macro Hedge Fund.

DEINE AUFGABE:
Suche die offiziellen Termine fuer 11 Event-Typen fuer die naechsten 6 Monate ab heute. Verwende Web Search um die aktuellen Termine von den offiziellen Quellen zu holen.

REGELN — STRIKT EINHALTEN:
1. Verwende NUR offizielle Quellen:
   - FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
   - CPI: bls.gov/cpi/ oder bls.gov/schedule/
   - NFP: bls.gov/schedule/ (Employment Situation)
   - PCE: bea.gov/news/schedule
   - ISM: ismworld.org
   - GDP: bea.gov/news/schedule
   - ECB: ecb.europa.eu (Governing Council dates)
   - BoJ: boj.or.jp/en/mopo/mpmdeci/
   - PBoC: 20. jeden Monats (fest), verifiziere bei pbc.gov.cn
   - OPEC: opec.org/opec_web/en/press_room/
   - China Data: stats.gov.cn oder NBS calendar

2. Wenn eine offizielle Quelle nicht erreichbar ist oder keine Termine zeigt:
   - Suche alternative zuverlaessige Quellen (Reuters, Bloomberg Calendar, Investing.com Economic Calendar)
   - Markiere als source_verified: false
   - Schreibe in notes welche Quelle du verwendet hast

3. Fuer FOMC: Unterscheide FOMC_STANDARD und FOMC_SEP (mit Dot Plot). SEP-Meetings sind: Maerz, Juni, September, Dezember.

4. Fuer GDP: Unterscheide GDP_ADVANCE (wichtigst), GDP_PRELIMINARY, GDP_FINAL.

5. Fuer China Data: Unterscheide CHINA_GDP (quartalsweise, mit IP+Retail) und CHINA_MONTHLY (nur IP+Retail).

6. PBoC LPR: Immer am 20. des Monats (oder naechster Werktag wenn 20. auf Wochenende faellt). Verifiziere ob es Aenderungen gibt.

7. OPEC+: Termine sind weniger vorhersagbar. Suche nach den aktuell geplanten Meetings. Wenn keine geplanten gefunden werden, notiere das.

8. NICHT suchen: OpEx/Quad Witching. Das wird regelbasiert berechnet.

OUTPUT-FORMAT:
Antworte AUSSCHLIESSLICH mit einem JSON-Array. KEIN anderer Text davor oder danach. Keine Markdown-Codeblocks. Nur reines JSON.

Format pro Event:
{
  "date": "YYYY-MM-DD",
  "event": "Menschenlesbarer Name auf Englisch",
  "type": "FOMC|CPI|NFP|PCE|ISM_MFG|GDP|ECB|BOJ|PBOC|OPEC|CHINA_DATA",
  "subtype": "FOMC_STANDARD|FOMC_SEP|GDP_ADVANCE|GDP_PRELIMINARY|GDP_FINAL|CHINA_GDP|CHINA_MONTHLY|null",
  "impact": "HIGH|MEDIUM",
  "themes": ["THEME1", "THEME2"],
  "time_et": "HH:MM|VARIABLE|ALL_DAY",
  "notes": "Kontext, welche Quelle, Besonderheiten",
  "source_verified": true|false,
  "source_url": "URL der Quelle oder null"
}

Erlaubte Themes: FED_POLICY, INFLATION, RECESSION, LIQUIDITY, COMMODITIES, CHINA_EM, DOLLAR, VOLATILITY, ENERGY

Impact-Regeln:
- FOMC: immer HIGH (FOMC_SEP = besonders wichtig, trotzdem HIGH)
- CPI: immer HIGH
- NFP: immer HIGH
- PCE: immer HIGH
- ISM_MFG: MEDIUM
- GDP: GDP_ADVANCE = MEDIUM, GDP_PRELIMINARY = MEDIUM, GDP_FINAL = MEDIUM
- ECB: MEDIUM
- BoJ: MEDIUM (aber bei ueberraschenden Entscheidungen historisch HIGH-Impact)
- PBoC: MEDIUM
- OPEC: MEDIUM
- China Data: CHINA_GDP = MEDIUM, CHINA_MONTHLY = MEDIUM

WICHTIG:
- Sortiere die Events chronologisch nach Datum
- Keine Duplikate (ein Event pro Datum+Typ Kombination)
- Wenn du unsicher bist ob ein Termin stimmt: source_verified: false setzen und in notes erklaeren
- Lieber einen Termin mit source_verified: false als gar keinen Termin"""


# =============================================================================
# USER PROMPT TEMPLATE (Spec Teil 6 §30.3)
# =============================================================================

USER_PROMPT_TEMPLATE_CALENDAR = """Heute ist {today} ({weekday}).

Suche alle offiziellen Event-Termine fuer die naechsten 6 Monate (bis {end_date}).

Beginne mit den wichtigsten Event-Typen:
1. FOMC (federalreserve.gov) — alle Meetings bis {end_date}, markiere welche SEP haben
2. CPI (bls.gov) — monatliche Releases bis {end_date}
3. NFP (bls.gov) — monatliche Employment Situation bis {end_date}
4. PCE (bea.gov) — monatliche Releases bis {end_date}

Dann die restlichen:
5. ISM Manufacturing — monatlich
6. GDP — quartalsweise (Advance, Preliminary, Final)
7. ECB — alle geplanten Meetings
8. BoJ — alle geplanten Meetings
9. PBoC LPR — 20. jeden Monats
10. OPEC+ — geplante Meetings (wenn bekannt)
11. China Data — GDP quartalsweise + monatliche IP/Retail

Suche JEDEN Event-Typ einzeln ueber Web Search. Verwende die offiziellen Quellen.

Antworte NUR mit dem JSON-Array. Kein anderer Text."""


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

VALID_TYPES = {
    "FOMC", "CPI", "NFP", "PCE", "ISM_MFG", "GDP",
    "ECB", "BOJ", "PBOC", "OPEC", "CHINA_DATA",
}

VALID_THEMES = {
    "FED_POLICY", "INFLATION", "RECESSION", "LIQUIDITY",
    "COMMODITIES", "CHINA_EM", "DOLLAR", "VOLATILITY", "ENERGY",
}

VALID_IMPACTS = {"HIGH", "MEDIUM"}

DEFAULT_THEMES = {
    "FOMC": ["FED_POLICY", "LIQUIDITY"],
    "CPI": ["INFLATION", "FED_POLICY"],
    "NFP": ["RECESSION", "FED_POLICY"],
    "PCE": ["INFLATION", "FED_POLICY"],
    "ISM_MFG": ["RECESSION", "COMMODITIES"],
    "GDP": ["RECESSION", "FED_POLICY"],
    "ECB": ["FED_POLICY", "DOLLAR"],
    "BOJ": ["LIQUIDITY", "FED_POLICY"],
    "PBOC": ["CHINA_EM", "LIQUIDITY"],
    "OPEC": ["COMMODITIES", "ENERGY"],
    "CHINA_DATA": ["CHINA_EM", "COMMODITIES"],
}

HIGH_IMPACT_TYPES = {"FOMC", "CPI", "NFP", "PCE"}


# =============================================================================
# LLM CALL WITH RETRY (Spec Teil 6 §30.6)
# =============================================================================

def run_calendar_updater(today: date) -> list[dict]:
    """
    Main function: Call LLM, parse response, retry on failure.

    Args:
        today: Current date

    Returns:
        List of validated event dicts

    Raises:
        RuntimeError: If all retries fail
    """
    end_date = today + timedelta(days=183)
    weekday = today.strftime("%A")

    user_prompt = USER_PROMPT_TEMPLATE_CALENDAR.format(
        today=today.isoformat(),
        weekday=weekday,
        end_date=end_date.isoformat(),
    )

    config = LLM_CONFIG_CALENDAR.copy()

    for attempt in range(1, config["retry_on_failure"] + 2):
        logger.info(
            f"LLM Calendar Update attempt {attempt} "
            f"(temp={config['temperature']})"
        )

        try:
            response = call_anthropic(
                system_prompt=SYSTEM_PROMPT_CALENDAR,
                user_prompt=user_prompt,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                tools=config["tools"],
                timeout=config["timeout_seconds"],
            )

            events = parse_calendar_llm_response(response)

            # Min validation: expect at least 20 events
            if len(events) < 20:
                logger.warning(
                    f"Only {len(events)} events parsed — expected >= 20. "
                    f"Retrying..."
                )
                config["temperature"] += config["retry_temperature_increment"]
                continue

            # Type coverage: at least 8 of 11 types
            types_found = set(e["type"] for e in events)
            if len(types_found) < 8:
                missing = VALID_TYPES - types_found
                logger.warning(f"Missing event types: {missing}. Retrying...")
                config["temperature"] += config["retry_temperature_increment"]
                continue

            logger.info(
                f"Calendar update successful: {len(events)} events, "
                f"{len(types_found)} types"
            )
            return events

        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            config["temperature"] += config["retry_temperature_increment"]

    # All attempts failed
    raise RuntimeError("Calendar LLM update failed after all retries")


# =============================================================================
# RESPONSE PARSING (Spec Teil 6 §30.4)
# =============================================================================

def parse_calendar_llm_response(response: dict) -> list[dict]:
    """
    Parse LLM response. Robust against:
    - JSON in Markdown codeblocks
    - Trailing commas
    - Missing fields
    - Text before/after JSON
    - Tool-use blocks (Web Search)

    Args:
        response: Normalized Anthropic API response dict

    Returns:
        List of validated event dicts
    """
    # 1. Extract text content from response
    text_parts = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])

    raw_text = "\n".join(text_parts).strip()

    if not raw_text:
        raise ValueError("No text content in LLM response")

    # 2. Extract JSON
    json_text = _extract_json_array(raw_text)

    # 3. Parse JSON
    try:
        events_raw = json.loads(json_text)
    except json.JSONDecodeError as e:
        # Attempt: remove trailing commas
        cleaned = re.sub(r',\s*([}\]])', r'\1', json_text)
        try:
            events_raw = json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"JSON parse failed after cleanup: {e}")

    if not isinstance(events_raw, list):
        raise ValueError(f"Expected JSON array, got {type(events_raw)}")

    # 4. Validate and fill defaults
    validated = []
    seen = set()  # Dedup: (date, type)

    for i, event in enumerate(events_raw):
        # Required fields
        date_str = event.get("date")
        event_type = event.get("type", "").upper()
        event_name = event.get("event", "")

        if not date_str:
            logger.warning(f"Event {i}: missing date, skipping")
            continue

        # Validate date
        try:
            parsed_date = date.fromisoformat(date_str)
        except ValueError:
            logger.warning(f"Event {i}: invalid date '{date_str}', skipping")
            continue

        if event_type not in VALID_TYPES:
            logger.warning(f"Event {i}: invalid type '{event_type}', skipping")
            continue

        # Dedup
        dedup_key = (date_str, event_type)
        if dedup_key in seen:
            logger.warning(f"Event {i}: duplicate {dedup_key}, skipping")
            continue
        seen.add(dedup_key)

        # Validate themes
        themes = event.get("themes", [])
        themes = [t for t in themes if t in VALID_THEMES]
        if not themes:
            themes = DEFAULT_THEMES.get(event_type, ["FED_POLICY"])

        # Validate impact
        impact = event.get("impact", "MEDIUM").upper()
        if impact not in VALID_IMPACTS:
            impact = "HIGH" if event_type in HIGH_IMPACT_TYPES else "MEDIUM"

        validated.append({
            "date": date_str,
            "event": event_name or f"{event_type} Event",
            "type": event_type,
            "subtype": event.get("subtype"),
            "impact": impact,
            "themes": themes,
            "time_et": event.get("time_et", "UNKNOWN"),
            "notes": event.get("notes", ""),
            "source_verified": event.get("source_verified", False),
            "source_url": event.get("source_url"),
            # Future layers (always empty in V1)
            "consensus": {"populated": False},
            "portfolio_sensitivity": {"populated": False},
            "outcome": {"populated": False},
        })

    # Sort chronologically
    validated.sort(key=lambda x: x["date"])

    logger.info(
        f"Parsed {len(validated)} events from LLM response "
        f"({len(events_raw)} raw, {len(events_raw) - len(validated)} rejected)"
    )

    return validated


def _extract_json_array(text: str) -> str:
    """Extract JSON array from text that may contain markdown or other content."""
    # Attempt 1: Markdown codeblock
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Attempt 2: Find [ ... ] directly
    start = text.find('[')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    # Attempt 3: Entire text is JSON
    return text


# =============================================================================
# YAML MERGE LOGIC (Spec Teil 6 §30.5)
# =============================================================================

def merge_events(
    existing_yaml: dict,
    new_events: list[dict],
    opex_events: list[dict],
    today: date,
) -> dict:
    """
    Merge existing YAML with new LLM events and OpEx events.

    Rules:
    1. Remove past events > 6 months old
    2. Keep past events < 6 months (for Post-Event Tracking V3)
    3. Replace future events with new data
    4. Add OpEx events
    5. Sort, deduplicate

    Args:
        existing_yaml: Current YAML dict (may be empty)
        new_events: Events from LLM
        opex_events: Events from OpEx calculator
        today: Current date

    Returns:
        Complete YAML dict with meta + events
    """
    cutoff_past = today - timedelta(days=183)  # 6 months back

    existing_events = existing_yaml.get("events", [])

    # Keep past events (between cutoff and today)
    kept_past = []
    for event in existing_events:
        try:
            event_date = date.fromisoformat(event["date"])
        except (ValueError, KeyError):
            continue
        if cutoff_past <= event_date < today:
            kept_past.append(event)

    # Future = entirely new (LLM + OpEx)
    future_events = new_events + opex_events

    # Merge
    all_events = kept_past + future_events

    # Deduplicate (date + type)
    seen = set()
    deduped = []
    for event in all_events:
        key = (event["date"], event["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(event)

    # Sort chronologically
    deduped.sort(key=lambda x: x["date"])

    # Calculate next update date (1st of next month)
    if today.month == 12:
        next_update = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_update = today.replace(month=today.month + 1, day=1)

    return {
        "meta": {
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "next_update": next_update.isoformat() + "T06:00:00Z",
            "updater_version": "1.0",
            "event_types_count": len(set(e["type"] for e in deduped)),
            "events_total": len(deduped),
            "horizon_months": 6,
            "past_events_kept": len(kept_past),
            "future_events_new": len(future_events),
        },
        "events": deduped,
    }
