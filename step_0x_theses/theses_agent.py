"""
Thesen Circle — Main Script
Baldur Creek Capital | Step 0x (V1.0.8)

V1.0.8: Sheet-Read für ETF-Preise, Relative-Value-Ketten, Convergence-Berechnung

Pipeline:
  1. System-Synthese (7 interne JSONs → kompakte Zusammenfassung)
  2a. Offene Suche (Web Search, kategorie-offen, OHNE interne Daten)
  2b. Adversarial / Red Team (Web Search, was tötet unser Portfolio)
  3a. Thesen-Kandidaten (10-15 kompakte Liste)
  3b. Vollständige Kausalketten + Relative-Value-Ketten (größter Call)
  4. Gegenthese (separater Call pro Top-5-These, Web Search)
  5. Bewertung + Priorisierung (Conviction, Asymmetrie, Tier, Retrospektive)
  → Assemblierung (deterministisch, kein LLM)
  → JSON schreiben + Archiv + Git Push

Usage:
  python -m step_0x_theses.theses_agent [--skip-git] [--skip-llm]
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("theses")

from .config import (
    BASE_DIR, DATA_DIR, HISTORY_DIR, OUTPUT_FILE,
    PIPELINE_ROOT, SYSTEM_INPUTS, PREVIOUS_THESES_FILE,
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, WEB_SEARCH_TOOL,
    TIER_1_MIN_SCORE, TIER_2_MIN_SCORE, CONVICTION_CHANGE_FLAG_THRESHOLD,
    LIFECYCLE_CONFIRMED_FOR_EMERGING, LIFECYCLE_CONFIRMED_RATIO_FOR_MATURE,
    LIFECYCLE_WEEKS_FOR_MATURE, LIFECYCLE_REFUTED_FOR_CHALLENGED,
    LIFECYCLE_REFUTED_FOR_DEAD,
    WATCHLIST_SEED, V16_STATES,
    HALLUCINATION_GUARD, JSON_INSTRUCTION,
    STEP1_SYSTEM_PROMPT, STEP2A_SYSTEM_PROMPT, STEP2B_SYSTEM_PROMPT,
    STEP3A_SYSTEM_PROMPT, STEP3B_SYSTEM_PROMPT, STEP3_JSON_SCHEMA,
    STEP4_SYSTEM_PROMPT, STEP5_SYSTEM_PROMPT,
    DW_SHEET_ID, DW_PRICES_TAB, V16_ETF_MAP, RATIO_PAIRS,
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_json_safe(path, label=""):
    """Load a JSON file, return None on any error."""
    if not os.path.exists(path):
        logger.warning(f"JSON nicht gefunden ({label}): {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Geladen ({label}): {path}")
        return data
    except Exception as e:
        logger.warning(f"JSON Parse-Fehler ({label}): {e}")
        return None


def parse_llm_response(resp):
    """Parse LLM response, robustly extracting JSON even if preceded by prose.
    Handles: prose before JSON, ```json blocks, truncated JSON from max_tokens.
    Bewährtes Pattern aus secular_trends.py V1.4+ mit Erweiterungen V1.0.1."""
    # Sammle alle Text-Blöcke (Web Search Responses haben mehrere content blocks)
    parts = []
    for block in resp.content:
        if block.type == "text":
            parts.append(block.text)
    txt = "\n".join(parts).strip()

    if not txt:
        logger.error("LLM returned no text content")
        return None

    # Strategy 1: ```json ... ``` Block extrahieren (häufigstes Pattern mit Web Search)
    match = re.search(r'```json\s*(.*?)```', txt, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _try_repair_json(candidate)
            if repaired is not None:
                logger.warning("JSON aus ```json Block repariert (abgeschnitten)")
                return repaired

    # Strategy 2: ```json ohne schließendes ``` (abgeschnitten bei max_tokens)
    match2 = re.search(r'```json\s*(.*)', txt, re.DOTALL)
    if match2:
        candidate = match2.group(1).strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _try_repair_json(candidate)
            if repaired is not None:
                logger.warning("JSON aus offenem ```json Block repariert")
                return repaired

    # Strategy 3: Finde erstes { und letztes } im gesamten Text
    first_brace = txt.find("{")
    last_brace = txt.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidate = txt[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _try_repair_json(candidate)
            if repaired is not None:
                logger.warning("JSON aus {}-Extraktion repariert")
                return repaired

    # Strategy 4: Nur { gefunden, kein schließendes } (komplett abgeschnitten)
    if first_brace >= 0:
        candidate = txt[first_brace:]
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
        repaired = _try_repair_json(candidate)
        if repaired is not None:
            logger.warning("JSON aus abgeschnittenem Output repariert")
            return repaired

    logger.error(f"LLM parse fail — first 500 chars: {txt[:500]}")
    return None


def _try_repair_json(text):
    """Versuche abgeschnittenes JSON zu reparieren durch Schließen offener Klammern."""
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    if open_braces == 0 and open_brackets == 0:
        return None  # Schon balanced aber trotzdem nicht parsebar

    repaired = text.rstrip()

    # Entferne unvollständige letzte Zeile (oft mitten im Wort abgeschnitten)
    last_newline = repaired.rfind('\n')
    if last_newline > len(repaired) * 0.5:
        last_line = repaired[last_newline + 1:]
        if last_line.count('"') % 2 != 0:
            repaired = repaired[:last_newline]

    # Zähle nochmal nach dem Trimmen
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False
    for ch in repaired:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    if in_string:
        repaired += '"'

    repaired += ']' * max(open_brackets, 0)
    repaired += '}' * max(open_braces, 0)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def call_llm(system_prompt, user_message, use_web_search=False, max_tokens=None):
    """Generic LLM call. Returns parsed JSON or None.
    Uses streaming for large max_tokens to avoid SDK 10-minute timeout."""
    import anthropic
    client = anthropic.Anthropic()

    effective_max_tokens = max_tokens or LLM_MAX_TOKENS

    kwargs = {
        "model": LLM_MODEL,
        "max_tokens": effective_max_tokens,
        "temperature": LLM_TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if use_web_search:
        kwargs["tools"] = [WEB_SEARCH_TOOL]

    logger.info(f"LLM Call ({'mit Web Search' if use_web_search else 'ohne Web Search'}, "
                f"max_tokens={effective_max_tokens})...")

    # Use streaming for large requests to avoid SDK 10-min timeout
    if effective_max_tokens > 16000 or use_web_search:
        # Stream and collect into a final message
        collected_text = []
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                pass  # Just consume the stream
            resp = stream.get_final_message()
    else:
        resp = client.messages.create(**kwargs)

    parsed = parse_llm_response(resp)
    if parsed:
        logger.info("LLM Call OK — JSON geparsed")
    else:
        logger.error("LLM Call — JSON Parse fehlgeschlagen")
    return parsed


# ═══════════════════════════════════════════════════════════════
# GOOGLE SHEET — ETF PREISE LESEN
# ═══════════════════════════════════════════════════════════════

def read_prices_from_sheet():
    """Liest aktuelle ETF-Preise aus dem V16 DW Sheet Prices Tab.
    Returns dict: {ticker: price} oder None bei Fehler.
    Graceful degradation: Wenn Sheet nicht erreichbar → None → Agent fährt ohne Relative-Value fort."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # GCP Service Account Key aus Environment (wie IC Pipeline, G7 Monitor)
        sa_key_json = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
        if not sa_key_json:
            logger.warning("PRICES: Kein GCP_SA_KEY/GOOGLE_CREDENTIALS — Sheet-Read übersprungen")
            return None

        sa_info = json.loads(sa_key_json)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)

        sheet = gc.open_by_key(DW_SHEET_ID)
        ws = sheet.worksheet(DW_PRICES_TAB)

        # Alle Werte lesen
        all_values = ws.get_all_values()
        if len(all_values) < 2:
            logger.warning("PRICES: Sheet hat weniger als 2 Zeilen")
            return None

        # Header = erste Zeile (Ticker), letzte Zeile = aktuellste Preise
        headers = all_values[0]
        last_row = all_values[-1]

        prices = {}
        for i, header in enumerate(headers):
            ticker = header.strip().upper()
            if ticker and ticker in V16_ETF_MAP and i < len(last_row):
                val = last_row[i].strip()
                if val:
                    try:
                        # Komma → Punkt falls nötig, dann float
                        price = float(val.replace(",", "."))
                        if price > 0:
                            prices[ticker] = price
                    except ValueError:
                        pass

        if prices:
            logger.info(f"PRICES: {len(prices)} ETF-Preise aus Sheet gelesen "
                        f"(z.B. SPY={prices.get('SPY', '?')}, GLD={prices.get('GLD', '?')})")
        else:
            logger.warning("PRICES: Keine gültigen Preise im Sheet gefunden")
            return None

        return prices

    except ImportError:
        logger.warning("PRICES: gspread/google-auth nicht installiert — Sheet-Read übersprungen")
        return None
    except Exception as e:
        logger.warning(f"PRICES: Sheet-Read fehlgeschlagen: {e}")
        return None


def compute_relative_values(prices):
    """Berechnet Ratio-Tabelle aus ETF-Preisen.
    Returns Liste von Ratio-Dicts oder leere Liste bei Fehler.
    Nur Ratios wo beide Ticker Preise haben."""
    if not prices:
        return []

    ratios = []
    for numerator, denominator, description in RATIO_PAIRS:
        if numerator in prices and denominator in prices:
            num_price = prices[numerator]
            den_price = prices[denominator]
            if den_price > 0:
                ratio_value = round(num_price / den_price, 4)
                ratios.append({
                    "numerator": numerator,
                    "denominator": denominator,
                    "description": description,
                    "ratio_value": ratio_value,
                    "numerator_price": num_price,
                    "denominator_price": den_price,
                    "numerator_name": V16_ETF_MAP.get(numerator, numerator),
                    "denominator_name": V16_ETF_MAP.get(denominator, denominator),
                })

    if ratios:
        logger.info(f"RATIOS: {len(ratios)} Ratio-Paare berechnet")
    return ratios


def format_prices_for_llm(prices, ratios):
    """Formatiert Preise und Ratios als Text-Block für den LLM-Prompt."""
    if not prices:
        return "Keine ETF-Preis-Daten verfügbar. Baue Relative-Value-Ketten NUR mit per Web Search verifizierten Preisen."

    lines = []
    lines.append("=== AKTUELLE ETF-PREISE (V16 System, Quelle: V16_DATA) ===")
    for ticker in sorted(prices.keys()):
        name = V16_ETF_MAP.get(ticker, ticker)
        lines.append(f"  {ticker} ({name}): ${prices[ticker]:.2f}")

    if ratios:
        lines.append("")
        lines.append("=== BERECHNETE RATIOS (V16 System, Quelle: V16_DATA) ===")
        for r in ratios:
            lines.append(f"  {r['description']}: {r['numerator']}/{r['denominator']} = {r['ratio_value']:.4f} "
                         f"({r['numerator_name']} ${r['numerator_price']:.2f} / "
                         f"{r['denominator_name']} ${r['denominator_price']:.2f})")

    lines.append("")
    lines.append("REGEL: Nutze diese Daten für Relative-Value-Ketten. Setze source='V16_DATA'. "
                 "Für Assets die NICHT in dieser Liste sind: Nutze NUR per Web Search verifizierte Preise mit Quellenangabe. "
                 "ERFINDE NIEMALS Preise.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SYSTEM INPUTS LADEN
# ═══════════════════════════════════════════════════════════════

def summarize_transition_engine(data):
    """Kompakte Zusammenfassung der Cycles Transition Engine."""
    if not data:
        return "Keine Cycles-Daten verfügbar."

    lines = []
    # Overall Assessment
    oa = data.get("overall_assessment", {})
    lines.append(f"Verdict: {oa.get('verdict', 'UNBEKANNT')}")
    lines.append(f"Cascade Severity: {oa.get('cascade_severity', '?')}")

    # Confirmation Counter
    cc = data.get("confirmation_counter", {})
    lines.append(f"Confirmation: {cc.get('bullish_count', 0)} bullish, "
                 f"{cc.get('bearish_count', 0)} bearish, "
                 f"{cc.get('neutral_count', 0)} neutral")
    lines.append(f"Score: {cc.get('confirmation_score', '?')} — {cc.get('interpretation', '')}")

    # Cascade Speed
    cs = data.get("cascade_speed", {}).get("current", {})
    lines.append(f"Cascade Speed: {cs.get('cascade_speed', '?')} "
                 f"({cs.get('n_transitions', 0)} Transitions in 6 Mo)")
    for t in cs.get("transitioned_cycles", []):
        lines.append(f"  {t['cycle']}: {t['from']} → {t['to']} ({t['month']})")

    # Phase Positions (aktuelle Phasen)
    pp = data.get("phase_positions", {})
    for cycle, info in pp.items():
        phase = info.get("current_phase", "?")
        lines.append(f"  {cycle}: {phase}")

    # Extended cycles
    ext = oa.get("extended_cycles", [])
    if ext:
        lines.append(f"Extended beyond median: {', '.join(ext)}")

    return "\n".join(lines)


def summarize_secular_trends(data):
    """Kompakte Zusammenfassung der Säkularen Trends."""
    if not data:
        return "Keine Säkulare-Trends-Daten verfügbar."

    lines = []
    cs = data.get("conviction_summary", {})
    lines.append(f"Aktive Regimes: {cs.get('active_regimes', '?')}/{cs.get('total_regimes', '?')}")
    lines.append(f"Weighted Activation: {cs.get('weighted_activation', '?')}")
    lines.append(f"Richtung: {cs.get('convergence_direction', '?')}")

    # Regime Status
    rs = cs.get("regime_status", {})
    for key, info in rs.items():
        act = info.get("activation", "?")
        active = "AKTIV" if info.get("active") else "INAKTIV"
        rob = info.get("robustness", "?")
        frag = info.get("fragility_status", "?")
        lines.append(f"  {info.get('name', key)}: {act:.2f} ({active}), "
                     f"Robustheit: {rob}, Fragilität: {frag}")

    # Tailwind Scores
    tw = cs.get("tailwind_scores", {})
    if tw:
        lines.append("Tailwinds:")
        for asset, score in tw.items():
            lines.append(f"  {asset}: {score:+d}")

    # Narrative (gekürzt)
    narr = cs.get("narrative", "")
    if narr:
        lines.append(f"Narrativ: {narr[:300]}...")

    return "\n".join(lines)


def summarize_g7(data):
    """Kompakte Zusammenfassung des G7 Monitors."""
    if not data:
        return "Keine G7-Daten verfügbar."

    lines = []
    if isinstance(data, dict):
        for key in ["scenarios", "active_scenarios", "heatmap", "dashboard"]:
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    for item in val[:5]:
                        if isinstance(item, dict):
                            name = item.get("name", item.get("scenario", "?"))
                            prob = item.get("probability", item.get("prob", "?"))
                            lines.append(f"  {name}: {prob}")
                elif isinstance(val, dict):
                    lines.append(f"  {key}: {json.dumps(val, ensure_ascii=False)[:200]}")

        meta = data.get("metadata", {})
        if meta:
            lines.append(f"Stand: {meta.get('generated_at', '?')}")

    if not lines:
        lines.append(f"G7 Daten geladen, Top-Level Keys: {list(data.keys())[:10]}")

    return "\n".join(lines)


def summarize_disruptions(data):
    """Kompakte Zusammenfassung der Disruptions."""
    if not data:
        return "Keine Disruptions-Daten verfügbar."

    lines = []
    if isinstance(data, dict):
        for key in ["disruptions", "active_disruptions", "intelligence_briefing", "threats"]:
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    for item in val[:5]:
                        if isinstance(item, dict):
                            name = item.get("name", item.get("title", "?"))
                            sev = item.get("severity", item.get("level", "?"))
                            lines.append(f"  {name}: Severity {sev}")
                elif isinstance(val, str):
                    lines.append(f"  {key}: {val[:200]}")

        meta = data.get("metadata", {})
        if meta:
            lines.append(f"Stand: {meta.get('generated_at', '?')}")

    if not lines:
        lines.append(f"Disruptions Daten geladen, Top-Level Keys: {list(data.keys())[:10]}")

    return "\n".join(lines)


def load_system_inputs():
    """Alle 7 System-JSONs laden und zusammenfassen."""
    logger.info("=" * 50)
    logger.info("SYSTEM INPUTS LADEN")
    logger.info("=" * 50)

    inputs = {}

    # 1. Cycles Transition Engine
    te = load_json_safe(SYSTEM_INPUTS["cycles_transition"], "Cycles Transition")
    inputs["cycles_summary"] = summarize_transition_engine(te)

    # 2. Cycles Conditional Returns (nur Zusammenfassung — File ist 3.5MB)
    cr = load_json_safe(SYSTEM_INPUTS["cycles_conditional"], "Cycles Conditional")
    if cr:
        es = cr.get("executive_summary", cr.get("metadata", {}))
        inputs["cycles_conditional_summary"] = json.dumps(es, ensure_ascii=False)[:500] if es else "Conditional Returns geladen, keine Summary."
    else:
        inputs["cycles_conditional_summary"] = "Nicht verfügbar."

    # 3. Säkulare Trends
    st = load_json_safe(SYSTEM_INPUTS["secular_trends"], "Säkulare Trends")
    inputs["secular_summary"] = summarize_secular_trends(st)

    # 4. G7 Monitor
    g7 = load_json_safe(SYSTEM_INPUTS["g7_monitor"], "G7 Monitor")
    inputs["g7_summary"] = summarize_g7(g7)

    # 5. Disruptions
    dis = load_json_safe(SYSTEM_INPUTS["disruptions"], "Disruptions")
    inputs["disruptions_summary"] = summarize_disruptions(dis)

    # 6. V16 State — aus Transition Engine extrahieren
    if te:
        oa = te.get("overall_assessment", {})
        cc = te.get("confirmation_counter", {})
        inputs["v16_state"] = oa.get("verdict", "UNBEKANNT")
        inputs["v16_weights_summary"] = (
            f"Bullish: {', '.join(cc.get('bullish_cycles', []))} | "
            f"Bearish: {', '.join(cc.get('bearish_cycles', []))} | "
            f"Neutral: {', '.join(cc.get('neutral_cycles', []))}"
        )
    else:
        inputs["v16_state"] = "Nicht verfügbar"
        inputs["v16_weights_summary"] = "Nicht verfügbar"

    # 7. IC Claims — Pfad noch nicht in SYSTEM_INPUTS, optional
    ic_path = os.path.join(PIPELINE_ROOT, "step_0i", "data", "ic_claims.json")
    ic = load_json_safe(ic_path, "IC Claims")
    if ic:
        if isinstance(ic, dict):
            claims = ic.get("claims", ic.get("active_claims", []))
            if isinstance(claims, list):
                top = sorted(claims, key=lambda c: c.get("conviction", 0), reverse=True)[:5]
                ic_lines = []
                for c in top:
                    ic_lines.append(f"  {c.get('claim', c.get('title', '?'))} "
                                    f"(Conv: {c.get('conviction', '?')}, "
                                    f"Quelle: {c.get('source', c.get('source_id', '?'))})")
                inputs["ic_summary"] = "\n".join(ic_lines) if ic_lines else "IC Claims geladen, keine Top-Claims."
            else:
                inputs["ic_summary"] = f"IC Daten geladen, Keys: {list(ic.keys())[:8]}"
        else:
            inputs["ic_summary"] = "IC Daten in unerwartetem Format."
    else:
        inputs["ic_summary"] = "Nicht verfügbar."

    return inputs


def load_previous_theses():
    """Vorwoche-Thesen laden."""
    prev = load_json_safe(PREVIOUS_THESES_FILE, "Vorwoche Thesen")
    if prev and "theses" in prev:
        logger.info(f"Vorwoche: {len(prev['theses'])} Thesen geladen")
        return prev
    logger.info("Keine Vorwoche-Thesen gefunden (erster Run)")
    return None


def build_previous_theses_summary(prev):
    """Kompakte Zusammenfassung der Vorwoche-Thesen für LLM-Input."""
    if not prev:
        return "Erster Run — keine Vorwoche-Thesen."

    lines = []
    theses = prev.get("theses", [])
    for t in theses:
        lines.append(
            f"- {t.get('id', '?')}: {t.get('title', '?')} "
            f"[{t.get('lifecycle', '?')}] "
            f"Conviction: {t.get('conviction', '?')}, "
            f"Tier: {t.get('tier', '?')}, "
            f"Horizon: {t.get('horizon', '?')}"
        )

    wl = prev.get("watchlist", [])
    if wl:
        lines.append(f"\nWatchlist: {', '.join(wl)}")

    return "\n".join(lines) if lines else "Keine Thesen in Vorwoche."


def build_market_moves_summary(prev):
    """Top-3 Asset-Bewegungen aus Vorwoche-Retrospektive oder Placeholder.
    Der Agent fetcht KEINE eigenen Marktdaten — er nutzt was die Pipeline liefert."""
    if prev:
        retro = prev.get("retrospective", {})
        moves = retro.get("top_3_moves", [])
        if moves:
            lines = []
            for m in moves:
                lines.append(f"  {m.get('asset', '?')}: {m.get('move', '?')}")
            return "\n".join(lines)

    return ("Erster Run — keine Vorwochen-Daten für Retrospektive. "
            "Nutze dein Wissen und Web Search um die wichtigsten "
            "Marktbewegungen der letzten 7 Tage zu identifizieren.")


# ═══════════════════════════════════════════════════════════════
# STEP 1: SYSTEM-SYNTHESE
# ═══════════════════════════════════════════════════════════════

def step1_system_synthesis(system_data, prev_theses):
    """Step 1: Interne Systeme zusammenfassen. Kein Web Search."""
    logger.info("=" * 50)
    logger.info("STEP 1: SYSTEM-SYNTHESE")
    logger.info("=" * 50)

    prev_summary = build_previous_theses_summary(prev_theses)

    user_msg = f"""=== V16 STATE ===
{system_data['v16_state']}

=== V16 POSITIONING ===
{system_data['v16_weights_summary']}

=== CYCLES (Transition Engine) ===
{system_data['cycles_summary']}

=== CYCLES (Conditional Returns) ===
{system_data['cycles_conditional_summary']}

=== SÄKULARE TRENDS ===
{system_data['secular_summary']}

=== IC CLAIMS ===
{system_data['ic_summary']}

=== G7 SZENARIEN ===
{system_data['g7_summary']}

=== DISRUPTIONS ===
{system_data['disruptions_summary']}

=== VORWOCHE THESEN ===
{prev_summary}"""

    result = call_llm(STEP1_SYSTEM_PROMPT, user_msg, use_web_search=False)
    if result:
        logger.info("Step 1 OK")
    else:
        logger.error("Step 1 FEHLGESCHLAGEN")
    return result


# ═══════════════════════════════════════════════════════════════
# STEP 2a: OFFENE SUCHE
# ═══════════════════════════════════════════════════════════════

def step2a_open_search(today):
    """Step 2a: Kategorie-offene Web-Suche. KEIN interner Input (Anchoring vermeiden)."""
    logger.info("=" * 50)
    logger.info("STEP 2a: OFFENE SUCHE (Web Search)")
    logger.info("=" * 50)

    user_msg = f"""Datum: {today}
Durchsuche das Web systematisch in 4 Runden nach makroökonomisch relevanten Entwicklungen der letzten 7 Tage:

RUNDE 1: Was dominiert die Schlagzeilen? Die großen Stories die jeder kennt.
RUNDE 2: Was passiert in Bereichen die NICHT in den Schlagzeilen sind? Suche gezielt abseits der Mainstream-Narrative.
RUNDE 3: Welche institutionellen oder strukturellen Risiken bauen sich LEISE auf? Dinge die noch kein Headline sind aber es werden könnten.
RUNDE 4: Was hat sich VERÄNDERT gegenüber dem Normalzustand? Welche Daten, Flows, oder Verhaltensweisen weichen vom historischen Muster ab?

Mindestens 15-20 verschiedene Suchbegriffe über alle 4 Runden. Du entscheidest selbst welche Kategorien relevant sind. Gib KEINE Kategorie vor die du aus früheren Runs kennst — denke jede Woche frisch."""

    result = call_llm(STEP2A_SYSTEM_PROMPT, user_msg, use_web_search=True, max_tokens=32000)
    if result:
        findings = result.get("open_search_findings", [])
        logger.info(f"Step 2a OK — {len(findings)} Findings")
    else:
        logger.error("Step 2a FEHLGESCHLAGEN")
    return result


# ═══════════════════════════════════════════════════════════════
# STEP 2b: ADVERSARIAL / RED TEAM
# ═══════════════════════════════════════════════════════════════

def step2b_adversarial(system_data, prev_theses):
    """Step 2b: Red Team — was tötet unser Portfolio? Web Search."""
    logger.info("=" * 50)
    logger.info("STEP 2b: ADVERSARIAL / RED TEAM (Web Search)")
    logger.info("=" * 50)

    # Aktive Thesen-Titel für Kontext
    active_titles = "Keine aktiven Thesen (erster Run)."
    if prev_theses and prev_theses.get("theses"):
        active = [t for t in prev_theses["theses"]
                  if t.get("lifecycle") in ("ACTIVE", "EMERGING", "MATURE")]
        if active:
            active_titles = "\n".join(
                f"- {t.get('title', '?')} [{t.get('lifecycle')}] "
                f"Richtung: {t.get('direction', '?')}"
                for t in active
            )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_msg = f"""Datum: {today}

UNSER AKTUELLES POSITIONING:
V16 State: {system_data['v16_state']}
V16 Gewichte: {system_data['v16_weights_summary']}

UNSERE AKTIVEN THESEN:
{active_titles}

Dein Job in 3 Schritten:
1. Finde was uns tötet. Suche im Web nach Risiken die GEGEN unsere Positionierung laufen.
2. Finde unsere BLINDEN FLECKEN: Wo sind wir EXPOSED ohne es zu wissen? Welche Risiko-Kategorien decken unsere Thesen NICHT ab? Denke an: Bereiche die wir komplett ignorieren, Korrelationen die wir nicht sehen, Annahmen die wir unbewusst machen.
3. Pre-Mortem: Wir verlieren 25% in 3 Monaten. Was ist passiert? Schreibe die Nachricht."""

    result = call_llm(STEP2B_SYSTEM_PROMPT, user_msg, use_web_search=True)
    if result:
        threats = result.get("adversarial_findings", [])
        logger.info(f"Step 2b OK — {len(threats)} Threats")
    else:
        logger.error("Step 2b FEHLGESCHLAGEN")
    return result


# ═══════════════════════════════════════════════════════════════
# STEP 3a: THESEN-KANDIDATEN GENERIEREN
# ═══════════════════════════════════════════════════════════════

def step3a_candidates(step1_out, step2a_out, step2b_out, prev_theses, today):
    """Step 3a: Kompakte Kandidaten-Liste (10-15 Thesen). Kein Web Search."""
    logger.info("=" * 50)
    logger.info("STEP 3a: THESEN-KANDIDATEN")
    logger.info("=" * 50)

    prev_summary = build_previous_theses_summary(prev_theses)

    user_msg = f"""Datum: {today}

=== INTERNE SYSTEM-SYNTHESE (Step 1) ===
{json.dumps(step1_out, ensure_ascii=False, indent=2) if step1_out else "Step 1 fehlgeschlagen."}

=== OFFENE RECHERCHE (Step 2a) ===
{json.dumps(step2a_out, ensure_ascii=False, indent=2) if step2a_out else "Step 2a fehlgeschlagen."}

=== ADVERSARIAL / RED TEAM (Step 2b) ===
{json.dumps(step2b_out, ensure_ascii=False, indent=2) if step2b_out else "Step 2b fehlgeschlagen."}

=== VORWOCHE-THESEN ===
{prev_summary}

Generiere 10-15 Thesen-Kandidaten. Bestehende Thesen updaten UND neue generieren.
MINDESTENS 10 Kandidaten. Decke alle drei Zeithorizonte ab."""

    result = call_llm(STEP3A_SYSTEM_PROMPT, user_msg, use_web_search=False)
    if result:
        candidates = result.get("candidates", [])
        logger.info(f"Step 3a OK — {len(candidates)} Kandidaten")
    else:
        logger.error("Step 3a FEHLGESCHLAGEN")
    return result


# ═══════════════════════════════════════════════════════════════
# STEP 3b: VOLLSTÄNDIGE KAUSALKETTEN BAUEN
# ═══════════════════════════════════════════════════════════════

def step3b_build_theses(step3a_out, step1_out, prev_theses, today, prices_text):
    """Step 3b: Vollständige Kausalketten + Relative-Value-Ketten. Kein Web Search."""
    logger.info("=" * 50)
    logger.info("STEP 3b: KAUSALKETTEN + RELATIVE VALUE BAUEN")
    logger.info("=" * 50)

    if not step3a_out or "candidates" not in step3a_out:
        logger.error("Step 3b übersprungen — keine Kandidaten aus Step 3a")
        return None

    candidates = step3a_out["candidates"]
    logger.info(f"Baue Kausalketten für {len(candidates)} Kandidaten")

    # Kandidaten-Zusammenfassung für den Prompt
    candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)

    prev_summary = build_previous_theses_summary(prev_theses)

    user_msg = f"""Datum: {today}

=== THESEN-KANDIDATEN (aus Step 3a) ===
{candidates_text}

=== VORWOCHE-THESEN (für Lifecycle-Updates) ===
{prev_summary}

=== SYSTEM-KONTEXT (kompakt) ===
{json.dumps(step1_out.get("system_summary", {}), ensure_ascii=False, indent=2) if step1_out else "Nicht verfügbar."}

=== ETF-PREISE UND RATIOS FÜR RELATIVE-VALUE-KETTEN ===
{prices_text}

Baue für JEDEN Kandidaten die vollständige Kausalkette mit allen Details.
Baue für JEDEN Kandidaten eine Relative-Value-Kette (wenn Preis-Daten verfügbar).

Antworte in folgendem JSON-Schema:
{STEP3_JSON_SCHEMA}"""

    result = call_llm(STEP3B_SYSTEM_PROMPT, user_msg, use_web_search=False, max_tokens=64000)
    if result:
        theses = result.get("theses", [])
        logger.info(f"Step 3b OK — {len(theses)} Thesen mit Kausalketten")
        # Open questions und watchlist aus Step 3a übernehmen falls Step 3b sie nicht hat
        if "open_questions" not in result and "open_questions" in step3a_out:
            result["open_questions"] = step3a_out["open_questions"]
        if "silence_alerts_investigated" not in result and "silence_alerts_investigated" in step3a_out:
            result["silence_alerts_investigated"] = step3a_out["silence_alerts_investigated"]
        if "watchlist_updates" not in result and "watchlist_updates" in step3a_out:
            result["watchlist_updates"] = step3a_out["watchlist_updates"]
    else:
        logger.error("Step 3b FEHLGESCHLAGEN")
    return result


# ═══════════════════════════════════════════════════════════════
# STEP 4: GEGENTHESE (pro Top-5 Kandidat)
# ═══════════════════════════════════════════════════════════════

def estimate_tier(thesis):
    """Schnelle Tier-Schätzung basierend auf Step 3 Output."""
    conv = thesis.get("conviction", 50)
    asym = thesis.get("asymmetry", 3)
    score = conv * asym
    if score >= TIER_1_MIN_SCORE:
        return 1
    elif score >= TIER_2_MIN_SCORE:
        return 2
    return 3


def summarize_thesis_for_counter(thesis):
    """Kompakte Zusammenfassung einer These für den Gegenthese-Call."""
    chain = thesis.get("causal_chain", {})
    root = chain.get("root", {})

    # Linearisiere den Hauptpfad (tiefster Pfad)
    chain_parts = []
    node = root
    while node:
        chain_parts.append(node.get("claim", "?"))
        children = node.get("children", [])
        node = children[0] if children else None

    chain_str = " → ".join(chain_parts) if chain_parts else "Keine Kausalkette."

    # Stärkstes Argument
    strongest = ""
    if root.get("status") == "BESTÄTIGT":
        strongest = f"Root-Glied bestätigt: {root.get('claim', '?')}"

    return chain_str, strongest


def step4_counter_theses(step3_out):
    """Step 4: Gegenthese für Top-5 Thesen nach Score. Web Search."""
    logger.info("=" * 50)
    logger.info("STEP 4: GEGENTHESEN (Web Search)")
    logger.info("=" * 50)

    if not step3_out or "theses" not in step3_out:
        logger.warning("Step 4 übersprungen — keine Thesen aus Step 3")
        return []

    theses = step3_out["theses"]

    # Top-5 nach geschätztem Score — nicht alle, sonst dauert Step 4 zu lange
    MAX_COUNTER_THESES = 5
    scored = sorted(theses,
                    key=lambda t: t.get("conviction", 50) * t.get("asymmetry", 3),
                    reverse=True)
    tier1_candidates = scored[:MAX_COUNTER_THESES]

    logger.info(f"{len(tier1_candidates)} Thesen bekommen Gegenthesen (Top {MAX_COUNTER_THESES} nach Score)")

    counter_results = []
    for thesis in tier1_candidates:
        tid = thesis.get("id", "unknown")
        title = thesis.get("title", "Unbekannt")
        chain_str, strongest = summarize_thesis_for_counter(thesis)
        conv = thesis.get("conviction", 50)

        user_msg = f"""THESE DIE DU ZERSTÖREN SOLLST:
Titel: {title}
ID: {tid}
Kausalkette: {chain_str}
Aktuelle Conviction: {conv}
Stärkstes Argument dafür: {strongest}

Finde den stärksten Grund warum diese These FALSCH ist."""

        result = call_llm(STEP4_SYSTEM_PROMPT, user_msg, use_web_search=True)
        if result:
            # Sicherstellen dass thesis_id gesetzt ist
            if "thesis_id" not in result:
                result["thesis_id"] = tid
            counter_results.append(result)
            logger.info(f"  Gegenthese für '{title}' OK")
        else:
            logger.warning(f"  Gegenthese für '{title}' FEHLGESCHLAGEN")

    logger.info(f"Step 4 OK — {len(counter_results)} Gegenthesen")
    return counter_results


# ═══════════════════════════════════════════════════════════════
# STEP 5: BEWERTUNG + PRIORISIERUNG
# ═══════════════════════════════════════════════════════════════

def build_previous_convictions(prev_theses):
    """Vorwochen-Conviction Scores für Vergleich NACH der frischen Bewertung."""
    if not prev_theses:
        return "Erster Run — keine Vorwochen-Scores."

    lines = []
    for t in prev_theses.get("theses", []):
        lines.append(f"  {t.get('id', '?')}: {t.get('conviction', '?')}")
    return "\n".join(lines) if lines else "Keine Scores."


def step5_assess(step3_out, step4_out, prev_theses, today):
    """Step 5: Bewertung, Tier-Einteilung, Retrospektive, Epistemic Health."""
    logger.info("=" * 50)
    logger.info("STEP 5: BEWERTUNG + PRIORISIERUNG")
    logger.info("=" * 50)

    if not step3_out:
        logger.error("Step 5 übersprungen — kein Step-3-Output")
        return None

    market_moves = build_market_moves_summary(prev_theses)
    prev_convictions = build_previous_convictions(prev_theses)

    # Step 4 Counter-Thesen als kompakter Text
    counter_text = "Keine Gegenthesen generiert."
    if step4_out:
        counter_parts = []
        for ct in step4_out:
            tid = ct.get("thesis_id", "?")
            cth = ct.get("counter_thesis", {})
            counter_parts.append(
                f"These {tid}: Gegenthese '{cth.get('title', '?')}' — "
                f"Kill: {cth.get('kill_probability', '?')}"
            )
        counter_text = "\n".join(counter_parts)

    user_msg = f"""Datum: {today}

=== THESEN AUS STEP 3 ===
{json.dumps(step3_out.get('theses', []), ensure_ascii=False, indent=2)}

=== GEGENTHESEN AUS STEP 4 ===
{counter_text}

=== TOP-3 ASSET-BEWEGUNGEN LETZTE WOCHE ===
{market_moves}

=== VORWOCHE CONVICTION SCORES ===
{prev_convictions}

Bewerte, priorisiere, und erstelle die Retrospektive."""

    result = call_llm(STEP5_SYSTEM_PROMPT, user_msg, use_web_search=False)
    if result:
        assessed = result.get("assessed_theses", [])
        logger.info(f"Step 5 OK — {len(assessed)} Thesen bewertet")
    else:
        logger.error("Step 5 FEHLGESCHLAGEN")
    return result


# ═══════════════════════════════════════════════════════════════
# LIFECYCLE — Deterministische Python-Logik
# ═══════════════════════════════════════════════════════════════

def count_node_stats(node):
    """Rekursiv Knoten-Statistiken zählen: confirmed, open, refuted, speculation, total."""
    if not node:
        return {"confirmed": 0, "open": 0, "refuted": 0, "speculation": 0, "total": 0}

    stats = {"confirmed": 0, "open": 0, "refuted": 0, "speculation": 0, "total": 1}

    status = node.get("status", "OFFEN")
    if status == "BESTÄTIGT":
        stats["confirmed"] += 1
    elif status == "WIDERLEGT":
        stats["refuted"] += 1
    else:
        stats["open"] += 1

    if node.get("epistemic_type") == "SPECULATION":
        stats["speculation"] += 1

    for child in node.get("children", []):
        child_stats = count_node_stats(child)
        for k in stats:
            stats[k] += child_stats[k]

    return stats


def update_lifecycle(thesis):
    """Deterministische Lifecycle-Transitions. Exakt aus Spec Teil 3, Kap. 6.3."""
    current = thesis.get("lifecycle", "SEED")

    # Knoten-Stats aus der Kausalkette
    chain = thesis.get("causal_chain", {})
    root = chain.get("root", {})
    stats = count_node_stats(root)

    confirmed = stats["confirmed"]
    refuted = stats["refuted"]
    total = stats["total"]
    catalysts = thesis.get("catalysts", [])
    catalysts_triggered = sum(1 for c in catalysts if c.get("status") == "TRIGGERED")
    counter_kill = thesis.get("counter_thesis", {}).get("kill_probability", "LOW")
    weeks_active = thesis.get("weeks_active", 0)
    independent_sources = thesis.get("independent_source_count", 0)

    # DEAD check — höchste Priorität
    if refuted >= LIFECYCLE_REFUTED_FOR_DEAD:
        return "DEAD"

    # CHALLENGED check
    if refuted >= LIFECYCLE_REFUTED_FOR_CHALLENGED or counter_kill == "HIGH":
        if current in ("ACTIVE", "MATURE", "EMERGING"):
            return "CHALLENGED"

    # MATURE check
    if current == "ACTIVE":
        ratio = confirmed / max(total, 1)
        if ratio >= LIFECYCLE_CONFIRMED_RATIO_FOR_MATURE or weeks_active > LIFECYCLE_WEEKS_FOR_MATURE:
            return "MATURE"

    # ACTIVE check — Katalysator eingetreten
    if catalysts_triggered > 0 and current in ("SEED", "EMERGING"):
        return "ACTIVE"

    # EMERGING check
    if current == "SEED":
        if confirmed >= LIFECYCLE_CONFIRMED_FOR_EMERGING or independent_sources >= 2:
            return "EMERGING"

    # Recovery: CHALLENGED → ACTIVE
    if current == "CHALLENGED" and refuted == 0 and counter_kill != "HIGH":
        return "ACTIVE"

    return current  # Keine Veränderung


# ═══════════════════════════════════════════════════════════════
# RELATIVE VALUE CONVERGENCE (deterministisch, kein LLM)
# ═══════════════════════════════════════════════════════════════

def compute_rv_convergence(theses):
    """Berechnet welche Assets am häufigsten als 'cheapest_asset' in Relative-Value-Ketten auftauchen.
    Returns: Liste von {asset, count, thesis_ids, thesis_titles}."""
    cheapest_counts = {}

    for t in theses:
        rv = t.get("relative_value_chain")
        if not rv or not isinstance(rv, dict):
            continue
        cheapest = rv.get("cheapest_asset")
        if not cheapest:
            continue

        if cheapest not in cheapest_counts:
            cheapest_counts[cheapest] = {
                "asset": cheapest,
                "display_name": rv.get("cheapest_asset_display", cheapest),
                "count": 0,
                "thesis_ids": [],
                "thesis_titles": [],
            }
        cheapest_counts[cheapest]["count"] += 1
        cheapest_counts[cheapest]["thesis_ids"].append(t.get("id", "?"))
        cheapest_counts[cheapest]["thesis_titles"].append(t.get("title_short", t.get("title", "?")))

    # Sortiere nach Häufigkeit absteigend
    convergence = sorted(cheapest_counts.values(), key=lambda x: -x["count"])

    # Nur Assets die in >1 These als billigstes auftauchen
    convergence_filtered = [c for c in convergence if c["count"] >= 2]

    if convergence_filtered:
        logger.info(f"RV CONVERGENCE: {len(convergence_filtered)} Assets in ≥2 Thesen als billigster Hebel")
        for c in convergence_filtered:
            logger.info(f"  {c['asset']}: {c['count']}x ({', '.join(c['thesis_ids'])})")

    return convergence


# ═══════════════════════════════════════════════════════════════
# ASSEMBLIERUNG (deterministisch, kein LLM)
# ═══════════════════════════════════════════════════════════════

def assemble_output(step1_out, step2a_out, step2b_out, step3_out, step4_out, step5_out,
                    call_count, search_count, prices_available):
    """Alle Step-Outputs zu einem theses.json zusammensetzen."""
    logger.info("=" * 50)
    logger.info("ASSEMBLIERUNG")
    logger.info("=" * 50)

    now = datetime.now(timezone.utc)
    theses = step3_out.get("theses", []) if step3_out else []

    # Step 5 Assessments mergen
    assessments = {}
    if step5_out:
        for a in step5_out.get("assessed_theses", []):
            assessments[a.get("id")] = a

    # Step 4 Counter-Thesen mergen
    counters = {}
    if step4_out:
        for ct in step4_out:
            tid = ct.get("thesis_id")
            if tid:
                counters[tid] = ct.get("counter_thesis", ct)

    # Merge und Lifecycle-Update
    for thesis in theses:
        tid = thesis.get("id", "")

        # Assessment mergen
        assessment = assessments.get(tid, {})
        thesis["conviction"] = assessment.get("conviction", thesis.get("conviction", 50))
        thesis["conviction_previous"] = assessment.get("conviction_previous")
        thesis["conviction_change"] = assessment.get("conviction_change")
        thesis["conviction_change_flagged"] = assessment.get("conviction_change_flagged", False)
        thesis["conviction_change_reason"] = assessment.get("conviction_change_reason")
        thesis["asymmetry"] = assessment.get("asymmetry", thesis.get("asymmetry", 3))
        thesis["assessment_notes"] = assessment.get("assessment_notes", "")

        # Score berechnen
        thesis["score"] = thesis["conviction"] * thesis["asymmetry"]

        # Tier berechnen
        score = thesis["score"]
        if score >= TIER_1_MIN_SCORE:
            thesis["tier"] = 1
        elif score >= TIER_2_MIN_SCORE:
            thesis["tier"] = 2
        else:
            thesis["tier"] = 3

        # Gegenthese mergen
        if tid in counters:
            thesis["counter_thesis"] = counters[tid]

        # Knoten-Stats berechnen
        chain = thesis.get("causal_chain", {})
        root = chain.get("root", {})
        stats = count_node_stats(root)
        thesis["total_nodes"] = stats["total"]
        thesis["confirmed_links"] = stats["confirmed"]
        thesis["open_links"] = stats["open"]
        thesis["refuted_links"] = stats["refuted"]
        thesis["speculation_count"] = stats["speculation"]

        # Lifecycle deterministische Update
        new_lifecycle = update_lifecycle(thesis)
        old_lifecycle = thesis.get("lifecycle", "SEED")
        if new_lifecycle != old_lifecycle:
            thesis["lifecycle_change"] = {
                "from": old_lifecycle,
                "to": new_lifecycle,
                "reason": f"Automatische Transition: {old_lifecycle} → {new_lifecycle}"
            }
            thesis["lifecycle"] = new_lifecycle
            logger.info(f"  Lifecycle: {tid} {old_lifecycle} → {new_lifecycle}")

        # weeks_active inkrementieren
        thesis["weeks_active"] = thesis.get("weeks_active", 0) + 1

        # last_updated
        thesis["last_updated"] = now.strftime("%Y-%m-%d")
        if "created_at" not in thesis:
            thesis["created_at"] = now.strftime("%Y-%m-%d")

    # Sortieren: Tier aufsteigend, Score absteigend
    theses.sort(key=lambda t: (t.get("tier", 3), -t.get("score", 0)))

    # Tier Summary
    tier_1_ids = [t["id"] for t in theses if t.get("tier") == 1]
    tier_2_ids = [t["id"] for t in theses if t.get("tier") == 2]
    tier_3_ids = [t["id"] for t in theses if t.get("tier") == 3]

    # Conviction Changes
    conviction_changes = step5_out.get("conviction_changes", []) if step5_out else []

    # Watchlist
    watchlist = step5_out.get("final_watchlist", WATCHLIST_SEED) if step5_out else WATCHLIST_SEED

    # Epistemic Health
    epistemic_health = step5_out.get("epistemic_health", {
        "overall": "LOW",
        "web_search_quality": "Nicht bewertet",
        "data_gaps": ["Step 5 fehlgeschlagen"],
        "contradictory_info": [],
        "confidence_notes": "Bewertung nicht durchgeführt"
    }) if step5_out else {
        "overall": "LOW",
        "web_search_quality": "Nicht bewertet",
        "data_gaps": ["Step 5 fehlgeschlagen"],
        "contradictory_info": [],
        "confidence_notes": "Bewertung nicht durchgeführt"
    }

    # Lifecycle Changes zählen
    lifecycle_changes = sum(1 for t in theses if t.get("lifecycle_change"))

    # Relative Value Convergence
    rv_convergence = compute_rv_convergence(theses)

    output = {
        "metadata": {
            "generated_at": now.isoformat(),
            "pipeline_version": "1.0.8",
            "llm_model": LLM_MODEL,
            "total_llm_calls": call_count,
            "web_search_calls": search_count,
            "total_theses": len(theses),
            "tier_1_count": len(tier_1_ids),
            "tier_2_count": len(tier_2_ids),
            "tier_3_count": len(tier_3_ids),
            "new_theses_this_week": sum(1 for t in theses if t.get("weeks_active", 0) <= 1),
            "lifecycle_changes_this_week": lifecycle_changes,
            "epistemic_health": epistemic_health.get("overall", "LOW"),
            "prices_available": prices_available,
        },
        "theses": theses,
        "tier_summary": {
            "tier_1_ids": tier_1_ids,
            "tier_2_ids": tier_2_ids,
            "tier_3_ids": tier_3_ids,
        },
        "relative_value_convergence": rv_convergence,
        "retrospective": step5_out.get("retrospective", {
            "top_3_moves": [],
            "blind_spots": [],
            "batting_average": {"total_moves_tracked": 0, "had_thesis": 0, "hit_rate": 0}
        }) if step5_out else {
            "top_3_moves": [],
            "blind_spots": [],
            "batting_average": {"total_moves_tracked": 0, "had_thesis": 0, "hit_rate": 0}
        },
        "conviction_changes": conviction_changes,
        "open_questions": step3_out.get("open_questions", []) if step3_out else [],
        "silence_alerts": step3_out.get("silence_alerts_investigated", []) if step3_out else [],
        "watchlist": watchlist,
        "adversarial_summary": {
            "worst_case": step2b_out.get("worst_case_scenario", {}) if step2b_out else {},
            "premortem": step2b_out.get("premortem_narrative", "") if step2b_out else "",
        },
        "epistemic_health": epistemic_health,
        "system_synthesis": step1_out.get("system_summary", {}) if step1_out else {},
    }

    logger.info(f"Assemblierung OK — {len(theses)} Thesen, "
                f"Tier 1: {len(tier_1_ids)}, Tier 2: {len(tier_2_ids)}, Tier 3: {len(tier_3_ids)}")
    if rv_convergence:
        top_conv = rv_convergence[0] if rv_convergence else None
        if top_conv:
            logger.info(f"RV Convergence Top: {top_conv['asset']} ({top_conv['count']}x)")

    return output


# ═══════════════════════════════════════════════════════════════
# JSON + ARCHIV + GIT
# ═══════════════════════════════════════════════════════════════

def write_json(output):
    """theses.json schreiben."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(OUTPUT_FILE)
    logger.info(f"Written: {OUTPUT_FILE} ({size:,} bytes)")


def archive_history():
    """Kopie in theses_history/ archivieren."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_file = os.path.join(HISTORY_DIR, f"theses_{today}.json")
    shutil.copy2(OUTPUT_FILE, archive_file)
    logger.info(f"Archiviert: {archive_file}")


def git_push():
    """Git add, commit, push für theses.json + history."""
    try:
        subprocess.run(
            ["git", "add",
             "step_0x_theses/data/theses.json",
             "step_0x_theses/data/theses_history/"],
            check=True
        )
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
            logger.info("Git: Keine Änderungen")
            return
        subprocess.run(
            ["git", "commit", "-m", "Thesen Agent weekly update"],
            check=True
        )
        subprocess.run(["git", "pull", "--rebase"], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info("Git push OK")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git Fehler: {e}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Thesen Agent V1.0.8 — Baldur Creek Capital")
    parser.add_argument("--skip-git", action="store_true", help="Git push überspringen")
    parser.add_argument("--skip-llm", action="store_true", help="LLM-Calls überspringen (Dry Run)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("THESEN AGENT PIPELINE — V1.0.8")
    logger.info("=" * 60)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info(f"Datum: {today}")

    # ── System-Inputs laden ──
    system_data = load_system_inputs()
    prev_theses = load_previous_theses()

    # ── ETF-Preise aus Sheet lesen ──
    logger.info("=" * 50)
    logger.info("ETF-PREISE LADEN (Google Sheet)")
    logger.info("=" * 50)
    prices = read_prices_from_sheet()
    ratios = compute_relative_values(prices) if prices else []
    prices_text = format_prices_for_llm(prices, ratios)
    prices_available = prices is not None and len(prices) > 0

    if args.skip_llm:
        logger.info("--skip-llm: LLM-Calls übersprungen. Dry Run beendet.")
        return

    # ── Call Counter ──
    call_count = 0
    search_count = 0

    # ── Step 1: System-Synthese ──
    step1_out = step1_system_synthesis(system_data, prev_theses)
    call_count += 1

    # ── Step 2a: Offene Suche ──
    step2a_out = step2a_open_search(today)
    call_count += 1
    search_count += 1

    # ── Step 2b: Adversarial ──
    step2b_out = step2b_adversarial(system_data, prev_theses)
    call_count += 1
    search_count += 1

    # ── Step 3a: Thesen-Kandidaten ──
    step3a_out = step3a_candidates(step1_out, step2a_out, step2b_out, prev_theses, today)
    call_count += 1

    # ── Step 3b: Vollständige Kausalketten + Relative Value ──
    step3_out = step3b_build_theses(step3a_out, step1_out, prev_theses, today, prices_text)
    call_count += 1

    # ── Step 4: Gegenthesen ──
    step4_out = step4_counter_theses(step3_out)
    n_counter = len(step4_out)
    call_count += n_counter
    search_count += n_counter

    # ── Step 5: Bewertung ──
    step5_out = step5_assess(step3_out, step4_out, prev_theses, today)
    call_count += 1

    logger.info(f"Total LLM Calls: {call_count}, davon Web Search: {search_count}")

    # ── Assemblierung ──
    output = assemble_output(
        step1_out, step2a_out, step2b_out,
        step3_out, step4_out, step5_out,
        call_count, search_count, prices_available
    )

    # ── JSON schreiben + Archiv ──
    write_json(output)
    archive_history()

    # ── Git Push ──
    if not args.skip_git:
        git_push()

    logger.info("=" * 60)
    logger.info("THESEN AGENT COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
