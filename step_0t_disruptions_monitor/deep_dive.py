#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/deep_dive.py
Stufe 2 — LLM Deep Dive (Top 5 Kategorien nach Screening-Score)
Spec: DISRUPTIONS_AGENT_SPEC TEIL 1 §2.2, TEIL 2 §13-14

Pipeline pro Kategorie:
  1. Brave Search: 5-8 tiefere Queries
  2. Source Quality Scoring
  3. LLM-Analyse via Sonnet: Scores, S-Kurve, Bull/Bear, Second Order, Analogy
"""

import os
import json
import time
import requests

# ===== API KEYS =====
BRAVE_API_KEY = os.environ.get('BRAVE_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

LLM_MODEL = 'claude-sonnet-4-20250514'
BRAVE_DELAY_S = 1.0


def run_deep_dive(category, screening_data, etf_universe, source_quality_weights):
    """
    Fuehre Stufe 2 Deep Dive fuer eine einzelne Kategorie durch.

    Args:
        category: Kategorie-Dict aus Config
        screening_data: Screening-Ergebnis dieser Kategorie (Stufe 1)
        etf_universe: Alle thematischen ETFs
        source_quality_weights: Quellengewichtung aus Config

    Returns:
        Dict mit allen Deep Dive Ergebnissen (Scores, Analyse, etc.)
    """
    cat_id = category['id']
    cat_name = category['name']
    cat_scope = category.get('scope', '')

    # --- Schritt 1: Tiefere Brave Search (5-8 Queries) ---
    search_results = _deep_brave_search(category)
    sources_scanned = len(search_results)

    # --- Schritt 2: Source Quality Scoring ---
    scored_sources = _score_sources(search_results, source_quality_weights)
    tier1_count = sum(1 for s in scored_sources if s['quality_tier'] in ('peer_reviewed', 'government_central_bank', 'tier1_media'))
    avg_quality = sum(s['quality_weight'] for s in scored_sources) / len(scored_sources) if scored_sources else 1.0

    # --- Schritt 3: Relevante ETFs fuer diese Kategorie ---
    cat_etfs = [
        etf for etf in etf_universe
        if cat_id in etf.get('category_ids', [])
    ]
    etf_tickers = [e['ticker'] for e in cat_etfs]

    # --- Schritt 4: LLM Deep Dive Analyse ---
    source_context = _build_source_context(scored_sources[:20])  # Top 20 nach Qualitaet

    llm_result = _llm_deep_dive(
        cat_id=cat_id,
        cat_name=cat_name,
        cat_scope=cat_scope,
        source_context=source_context,
        screening_data=screening_data,
        etf_tickers=etf_tickers,
        source_quality_weights=source_quality_weights
    )

    # --- Zusammenbauen ---
    result = {
        'category_id': cat_id,
        'category_name': cat_name,
        'maturity': llm_result.get('maturity', 50),
        'momentum': llm_result.get('momentum', 50),
        'relevance': llm_result.get('relevance', 50),
        'hype': llm_result.get('hype', 50),
        'headline': llm_result.get('headline', ''),
        'bull_case': llm_result.get('bull_case', ''),
        'bear_case': llm_result.get('bear_case', ''),
        'top_etf': llm_result.get('top_etf', etf_tickers[0] if etf_tickers else ''),
        'top_short': llm_result.get('top_short', ''),
        'historical_analogy': llm_result.get('historical_analogy', ''),
        'model_risk': llm_result.get('model_risk', 'NONE'),
        'second_order_effects': llm_result.get('second_order_effects', []),
        'trigger_events': llm_result.get('trigger_events', []),
        'multi_signal_count': llm_result.get('multi_signal_count', 0),
        'sources_scanned': sources_scanned,
        'source_quality': {
            'total_sources_scanned': sources_scanned,
            'tier1_sources': tier1_count,
            'avg_source_weight': round(avg_quality, 1),
            'confidence_note': llm_result.get('confidence_note', '')
        }
    }

    return result


# ===== BRAVE SEARCH (DEEP) =====

def _deep_brave_search(category):
    """
    Tiefere Brave Search: 5-8 Queries mit spezifischeren Suchbegriffen.
    """
    if not BRAVE_API_KEY:
        print("    [SKIP] Kein BRAVE_API_KEY — Deep Search uebersprungen")
        return []

    cat_name = category['name']
    base_keywords = category.get('keywords_brave', [])

    # Generiere erweiterte Queries
    queries = list(base_keywords)  # Basis-Keywords
    queries.extend([
        f"{cat_name} investment thesis 2026",
        f"{cat_name} commercial deployment milestone",
        f"{cat_name} market size growth forecast",
        f"{cat_name} breakthrough latest news",
        f"{cat_name} ETF fund flows analysis",
    ])
    queries = queries[:8]  # max 8

    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            results = _brave_search(query, count=10)
            for r in results:
                url = r.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
            time.sleep(BRAVE_DELAY_S)
        except Exception as e:
            print(f"    [WARN] Deep Brave '{query}': {e}")

    print(f"    [BRAVE] {len(all_results)} unique Ergebnisse aus {len(queries)} Queries")
    return all_results


def _brave_search(query, count=10):
    """Einzelne Brave Search API Abfrage."""
    url = 'https://api.search.brave.com/res/v1/web/search'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': BRAVE_API_KEY
    }
    params = {
        'q': query,
        'count': count,
        'freshness': 'pm'  # past month fuer Deep Dive
    }
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get('web', {}).get('results', [])


# ===== SOURCE QUALITY SCORING =====

def _score_sources(search_results, quality_weights):
    """
    Bewerte Quellenqualitaet basierend auf Domain.
    Spec §14.1: 8-stufige Gewichtung.
    """
    # Domain → Tier Mapping
    TIER_MAP = {
        # Peer-reviewed
        'nature.com': 'peer_reviewed', 'science.org': 'peer_reviewed',
        'thelancet.com': 'peer_reviewed', 'cell.com': 'peer_reviewed',
        'pnas.org': 'peer_reviewed', 'arxiv.org': 'peer_reviewed',
        # Government / Central Banks
        'federalreserve.gov': 'government_central_bank', 'ecb.europa.eu': 'government_central_bank',
        'iea.org': 'government_central_bank', 'energy.gov': 'government_central_bank',
        'sec.gov': 'government_central_bank', 'bls.gov': 'government_central_bank',
        'imf.org': 'government_central_bank', 'worldbank.org': 'government_central_bank',
        'whitehouse.gov': 'government_central_bank', 'congress.gov': 'government_central_bank',
        # Tier-1 Media
        'ft.com': 'tier1_media', 'wsj.com': 'tier1_media',
        'bloomberg.com': 'tier1_media', 'reuters.com': 'tier1_media',
        'economist.com': 'tier1_media', 'nytimes.com': 'tier1_media',
        'barrons.com': 'tier1_media',
        # Industry Reports
        'mckinsey.com': 'industry_reports', 'gartner.com': 'industry_reports',
        'bcg.com': 'industry_reports', 'bain.com': 'industry_reports',
        'deloitte.com': 'industry_reports', 'pwc.com': 'industry_reports',
        'idc.com': 'industry_reports',
        # Tech Media
        'techcrunch.com': 'tech_media', 'arstechnica.com': 'tech_media',
        'wired.com': 'tech_media', 'theverge.com': 'tech_media',
        'venturebeat.com': 'tech_media', 'semafor.com': 'tech_media',
        'protocol.com': 'tech_media', 'technologyreview.com': 'tech_media',
        # Social Media
        'reddit.com': 'social_media', 'twitter.com': 'social_media',
        'x.com': 'social_media', 'youtube.com': 'social_media',
    }

    scored = []
    for r in search_results:
        url = r.get('url', '')
        domain = _extract_domain(url)

        # Finde Tier
        tier = 'general_news'  # default
        for domain_key, tier_value in TIER_MAP.items():
            if domain_key in domain:
                tier = tier_value
                break

        # PR Detection
        if any(kw in url.lower() for kw in ['prnewswire', 'businesswire', 'globenewswire', 'prs.', '/press-release']):
            tier = 'pr_press_release'

        weight = quality_weights.get(tier, 1.0)

        scored.append({
            'title': r.get('title', ''),
            'url': url,
            'description': r.get('description', ''),
            'domain': domain,
            'quality_tier': tier,
            'quality_weight': weight,
        })

    # Sortiere nach Qualitaet (beste zuerst)
    scored.sort(key=lambda x: x['quality_weight'], reverse=True)
    return scored


def _extract_domain(url):
    """Extrahiere Domain aus URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return ''


def _build_source_context(scored_sources):
    """
    Baue einen kompakten Text-Kontext aus den besten Quellen fuer den LLM-Prompt.
    """
    lines = []
    for i, s in enumerate(scored_sources[:15], 1):  # max 15 Quellen im Prompt
        tier_label = s['quality_tier'].replace('_', ' ').title()
        lines.append(f"[{i}] ({tier_label}, Gewicht {s['quality_weight']}x) {s['title']}")
        if s.get('description'):
            lines.append(f"    {s['description'][:200]}")
    return '\n'.join(lines)


# ===== LLM DEEP DIVE =====

def _llm_deep_dive(cat_id, cat_name, cat_scope, source_context, screening_data, etf_tickers, source_quality_weights):
    """
    LLM-Analyse via Anthropic Sonnet.
    Generiert: Scores, S-Kurve, Bull/Bear, Second Order Effects, Historical Analogy.
    """
    if not ANTHROPIC_API_KEY:
        print("    [SKIP] Kein ANTHROPIC_API_KEY — LLM-Analyse uebersprungen")
        return _default_llm_result(cat_id, cat_name)

    system_prompt = """Du bist ein Disruptions-Analyst fuer einen systematischen Macro-Hedge-Fund.
Deine Aufgabe: Technologische und strukturelle Trends analysieren und in praezise, quantitative Scores uebersetzen.
Antworte AUSSCHLIESSLICH mit validem JSON. Kein Markdown, kein erklaerenderText, nur das JSON-Objekt."""

    user_prompt = f"""Analysiere den Trend "{cat_name}" ({cat_id}).
Scope: {cat_scope}

Screening-Daten dieser Woche:
- Brave Search Hits: {screening_data.get('brave_hit_count', 0)}
- Brave Sentiment: {screening_data.get('brave_sentiment', 50)}/100
- Google Trends: {screening_data.get('google_trends_value', 50)}/100 (1M Change: {screening_data.get('google_trends_1m_change', 0)}%)
- ETF Flow Score: {screening_data.get('etf_flow_1w', 50)}/100

Relevante Nachrichten und Quellen (nach Qualitaet sortiert):
{source_context}

Quellengewichtung beachten: Peer-reviewed 5x, Government 4x, Tier-1 Media 3x, Industry Reports 3x, Tech Media 2x, General 1x, Social 0.5x, PR 0.3x.

Verfuegbare thematische ETFs fuer diesen Trend: {', '.join(etf_tickers)}

Antworte mit folgendem JSON-Schema:
{{
  "maturity": <int 0-100, wo auf der S-Kurve steht der Trend? 0=reine Forschung, 100=Mainstream>,
  "momentum": <int 0-100, wie stark bewegt sich der Trend JETZT?>,
  "relevance": <int 0-100, wie relevant fuer ein V16 Macro Portfolio? Beruecksichtige: Gibt es handelbare ETFs? Beeinflusst es bestehende V16 Assets?>,
  "hype": <int 0-100, wie ueberbewertet/gehypt? Hoher Wert = viel Hype, Vorsicht>,
  "headline": "<einzeilige Zusammenfassung der wichtigsten Entwicklung diese Woche>",
  "bull_case": "<2-3 Saetze Bull Case>",
  "bear_case": "<2-3 Saetze Bear Case>",
  "top_etf": "<bester einzelner ETF-Ticker fuer Long-Exposure auf diesen Trend>",
  "top_short": "<bester einzelner Aktien-Ticker als Short-Kandidat, oder leer wenn keiner offensichtlich>",
  "historical_analogy": "<z.B. 'Internet 1997' oder 'EV 2015' — welche historische Phase passt am besten?>",
  "model_risk": "<NONE | PARAMETER | STRUCTURAL | PARADIGM — kann dieser Trend V16-Modellannahmen brechen?>",
  "multi_signal_count": <int 0-5, wie viele der 5 Inflection-Signale feuern gleichzeitig? (1=Tech Breakthrough, 2=Kapitalzufluss, 3=Regulierung klar, 4=Adoption-Knick, 5=Mainstream-Medien)>,
  "second_order_effects": [
    {{
      "order": <1|2|3>,
      "effect": "<Beschreibung>",
      "sectors_long": ["<ETF oder Sektor>"],
      "sectors_short": ["<ETF oder Sektor>"]
    }}
  ],
  "trigger_events": [
    {{
      "id": "T1",
      "description": "<konkretes, ueberpruefbares Event>",
      "check_method": "<brave_search | etf_data | google_trends>",
      "triggered": false
    }}
  ],
  "confidence_note": "<1-2 Saetze zur Quellenqualitaet und Konfidenz der Analyse>"
}}"""

    try:
        result = _call_anthropic(system_prompt, user_prompt)
        parsed = _parse_json_response(result)
        if parsed:
            return parsed
        else:
            print(f"    [WARN] LLM Response konnte nicht geparst werden — verwende Defaults")
            return _default_llm_result(cat_id, cat_name)
    except Exception as e:
        print(f"    [ERROR] LLM-Analyse fehlgeschlagen: {e}")
        return _default_llm_result(cat_id, cat_name)


def _call_anthropic(system_prompt, user_prompt):
    """Anthropic API Call via REST."""
    url = 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }
    payload = {
        'model': LLM_MODEL,
        'max_tokens': 2000,
        'system': system_prompt,
        'messages': [
            {'role': 'user', 'content': user_prompt}
        ]
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Extrahiere Text aus Response
    content = data.get('content', [])
    text_parts = [c.get('text', '') for c in content if c.get('type') == 'text']
    return '\n'.join(text_parts)


def _parse_json_response(text):
    """Parse JSON aus LLM Response, mit Cleanup."""
    # Entferne Markdown Code-Block wenn vorhanden
    cleaned = text.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Versuche den ersten { ... } Block zu extrahieren
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _default_llm_result(cat_id, cat_name):
    """Fallback wenn LLM nicht verfuegbar."""
    return {
        'maturity': 50,
        'momentum': 50,
        'relevance': 50,
        'hype': 50,
        'headline': f'{cat_name}: LLM-Analyse nicht verfuegbar — Screening-Daten als Basis.',
        'bull_case': 'Keine LLM-Analyse verfuegbar.',
        'bear_case': 'Keine LLM-Analyse verfuegbar.',
        'top_etf': '',
        'top_short': '',
        'historical_analogy': '',
        'model_risk': 'NONE',
        'multi_signal_count': 0,
        'second_order_effects': [],
        'trigger_events': [],
        'confidence_note': 'Fallback — LLM-Analyse fehlgeschlagen oder API Key fehlt.'
    }
