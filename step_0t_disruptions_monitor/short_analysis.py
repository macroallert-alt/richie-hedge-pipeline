#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/short_analysis.py
Drei-Stufen Short-Identifikation.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 2 §10

Stufe 1: Bedrohte Sektoren identifizieren (aus Threat Map + Dependencies)
Stufe 2: Fundamentale Schwaechen finden (EODHD/FMP Screening)
Stufe 3: Disruptions-Overlay (LLM-Analyse: fundamental schwach + Disruptions-Exposure)

Output: Short Candidates mit Thesis, Confidence, Pair Trades
"""

import os
import time
import json
import requests

EODHD_API_KEY = os.environ.get('EODHD_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
LLM_MODEL = 'claude-sonnet-4-20250514'
EODHD_DELAY_S = 0.5


def run_short_analysis(trends, exposure_map, etf_universe):
    """
    Fuehre 3-Stufen Short-Analyse durch.

    Args:
        trends: Alle bewerteten Trends (mit Phase, Watchlist-Status)
        exposure_map: V16 Exposure Map
        etf_universe: Thematische ETFs

    Returns:
        Liste von Short-Candidate-Dicts
    """
    active_watch = [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')]
    if not active_watch:
        print("    [SHORTS] Keine ACTIVE/WATCH Trends — ueberspringe Short-Analyse")
        return []

    # --- Stufe 1: Bedrohte Sektoren ---
    threatened_sectors = _identify_threatened_sectors(active_watch, exposure_map)
    if not threatened_sectors:
        print("    [SHORTS] Keine bedrohten Sektoren identifiziert")
        return []

    print(f"    [SHORTS] Stufe 1: {len(threatened_sectors)} bedrohte Sektoren")

    # --- Stufe 2: Fundamentale Schwaechen ---
    weak_candidates = _screen_fundamental_weakness(threatened_sectors)
    print(f"    [SHORTS] Stufe 2: {len(weak_candidates)} fundamental schwache Kandidaten")

    if not weak_candidates:
        return []

    # --- Stufe 3: Disruptions-Overlay (LLM) ---
    short_results = _disruptions_overlay(weak_candidates, active_watch, etf_universe)
    print(f"    [SHORTS] Stufe 3: {len(short_results)} finale Short-Kandidaten")

    return short_results


# ===== STUFE 1: BEDROHTE SEKTOREN =====

# Mapping: Disruptions-Kategorie → bedrohte Sektoren + Proxy-Tickers fuer Screening
SECTOR_THREAT_MAP = {
    'D1': {
        'sectors': ['Traditional Media', 'Legacy IT Services', 'Content Production'],
        'screen_tickers': ['PARA', 'WBD', 'NWSA', 'IBM', 'HPQ'],
    },
    'D2': {
        'sectors': ['Low-Skill Labor Intensive', 'Traditional Logistics'],
        'screen_tickers': ['UBER', 'LYFT', 'XPO'],
    },
    'D3': {
        'sectors': ['Fossil Energy', 'Traditional Utilities', 'Oil Services'],
        'screen_tickers': ['DVN', 'OXY', 'HAL', 'SLB', 'RIG'],
    },
    'D4': {
        'sectors': ['Traditional Pharma (non-biotech)', 'Medical Devices Legacy'],
        'screen_tickers': ['PFE', 'BMY', 'VTRS', 'OGN'],
    },
    'D5': {
        'sectors': ['Legacy Telecom', 'Traditional Satellite'],
        'screen_tickers': ['LUMN', 'VZ', 'DISH'],
    },
    'D6': {
        'sectors': ['Encryption-dependent (pre-quantum)'],
        'screen_tickers': [],
    },
    'D7': {
        'sectors': ['Traditional Banking', 'Legacy Payment'],
        'screen_tickers': ['ALLY', 'KEY', 'WU'],
    },
    'D8': {
        'sectors': ['China-dependent Manufacturing', 'Long-chain Logistics'],
        'screen_tickers': [],
    },
    'D9': {
        'sectors': ['Fossil Fuel Producers', 'Non-adaptive Mining'],
        'screen_tickers': ['CTRA', 'RRC', 'AR'],
    },
    'D10': {
        'sectors': ['Legacy Security', 'Unpatched Enterprise'],
        'screen_tickers': [],
    },
    'D11': {
        'sectors': ['Youth-focused Consumer', 'Traditional Education'],
        'screen_tickers': ['PTON', 'CHGG'],
    },
    'D12': {
        'sectors': ['Regulation-vulnerable Big Tech', 'Non-compliant Sectors'],
        'screen_tickers': [],
    },
}


def _identify_threatened_sectors(active_trends, exposure_map):
    """
    Identifiziere bedrohte Sektoren basierend auf ACTIVE/WATCH Trends.
    """
    threatened = []

    for t in active_trends:
        cat_id = t['id']
        threat_info = SECTOR_THREAT_MAP.get(cat_id, {})
        sectors = threat_info.get('sectors', [])
        tickers = threat_info.get('screen_tickers', [])

        if not tickers:
            continue

        threatened.append({
            'category_id': cat_id,
            'category_name': t['name'],
            'phase': t.get('phase', 'EMERGING'),
            'velocity': t.get('velocity_label', 'LOW'),
            'sectors': sectors,
            'screen_tickers': tickers,
        })

    return threatened


# ===== STUFE 2: FUNDAMENTALE SCHWAECHEN =====

def _screen_fundamental_weakness(threatened_sectors):
    """
    Spec §10.2: Screening nach fundamental schwachen Unternehmen.
    Kriterien: Debt/Equity > 2.0, Revenue Growth < -5%, Negatives FCF, Short Interest > 10%.
    """
    if not EODHD_API_KEY:
        print("    [SKIP] Kein EODHD_API_KEY — Fundamental-Screening uebersprungen")
        return _fallback_fundamental_screen(threatened_sectors)

    candidates = []

    for sector in threatened_sectors:
        for ticker in sector['screen_tickers']:
            try:
                fundamentals = _get_fundamentals(ticker)
                if not fundamentals:
                    continue

                weakness_flags = []
                debt_equity = fundamentals.get('debt_equity', 0)
                revenue_growth = fundamentals.get('revenue_growth_yoy', 0)
                fcf = fundamentals.get('free_cash_flow', 0)
                short_interest = fundamentals.get('short_interest', 0)
                market_cap = fundamentals.get('market_cap', 0)

                if debt_equity > 2.0:
                    weakness_flags.append(f"High Debt/Equity: {debt_equity:.1f}")
                if revenue_growth < -5:
                    weakness_flags.append(f"Falling Revenue: {revenue_growth:.1f}%")
                if fcf < 0:
                    weakness_flags.append(f"Negative FCF")
                if short_interest > 10:
                    weakness_flags.append(f"High Short Interest: {short_interest:.1f}%")

                # Mindestens 2 Schwaeche-Signale
                if len(weakness_flags) >= 2:
                    candidates.append({
                        'ticker': ticker,
                        'name': fundamentals.get('name', ticker),
                        'market_cap': market_cap,
                        'debt_equity': debt_equity,
                        'revenue_growth_yoy': revenue_growth,
                        'free_cash_flow': fcf,
                        'short_interest': short_interest,
                        'weakness_flags': weakness_flags,
                        'threat_source_id': sector['category_id'],
                        'threat_source_name': sector['category_name'],
                        'threat_phase': sector['phase'],
                        'threat_velocity': sector['velocity'],
                    })

                time.sleep(EODHD_DELAY_S)

            except Exception as e:
                print(f"    [WARN] Fundamentals {ticker}: {e}")

    return candidates


def _get_fundamentals(ticker):
    """Hole Fundamentaldaten via EODHD."""
    url = f'https://eodhd.com/api/fundamentals/{ticker}.US'
    params = {
        'api_token': EODHD_API_KEY,
        'fmt': 'json',
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data or isinstance(data, list):
        return None

    general = data.get('General', {})
    highlights = data.get('Highlights', {})
    balance = data.get('Financials', {}).get('Balance_Sheet', {}).get('quarterly', {})
    valuation = data.get('Valuation', {})

    # Extrahiere relevante Felder
    result = {
        'name': general.get('Name', ''),
        'market_cap': highlights.get('MarketCapitalization', 0),
        'revenue_growth_yoy': _safe_float(highlights.get('QuarterlyRevenueGrowthYOY', 0)) * 100,
        'free_cash_flow': _safe_float(highlights.get('FreeCashFlow', 0)),
        'debt_equity': 0,
        'short_interest': _safe_float(highlights.get('ShortPercentFloat', 0)),
    }

    # Debt/Equity aus Balance Sheet
    if balance:
        latest_q = list(balance.values())[0] if balance else {}
        total_debt = _safe_float(latest_q.get('shortLongTermDebtTotal', 0))
        equity = _safe_float(latest_q.get('totalStockholderEquity', 1))
        if equity > 0:
            result['debt_equity'] = round(total_debt / equity, 2)

    return result


def _fallback_fundamental_screen(threatened_sectors):
    """Fallback ohne API: Verwende bekannte schwache Kandidaten."""
    # Hartcodierte bekannte schwache Unternehmen als Seed
    KNOWN_WEAK = {
        'PARA': {'name': 'Paramount Global', 'weakness_flags': ['High Debt', 'Falling Revenue', 'Legacy Media']},
        'WBD': {'name': 'Warner Bros Discovery', 'weakness_flags': ['High Debt', 'Streaming Losses']},
        'LUMN': {'name': 'Lumen Technologies', 'weakness_flags': ['High Debt', 'Falling Revenue', 'Legacy Telecom']},
        'PFE': {'name': 'Pfizer Inc', 'weakness_flags': ['Falling Revenue', 'Patent Cliffs']},
        'CHGG': {'name': 'Chegg Inc', 'weakness_flags': ['AI Disruption', 'Falling Revenue']},
    }

    candidates = []
    for sector in threatened_sectors:
        for ticker in sector['screen_tickers']:
            if ticker in KNOWN_WEAK:
                info = KNOWN_WEAK[ticker]
                candidates.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'market_cap': 0,
                    'debt_equity': 0,
                    'revenue_growth_yoy': 0,
                    'free_cash_flow': 0,
                    'short_interest': 0,
                    'weakness_flags': info['weakness_flags'],
                    'threat_source_id': sector['category_id'],
                    'threat_source_name': sector['category_name'],
                    'threat_phase': sector['phase'],
                    'threat_velocity': sector['velocity'],
                    '_note': 'Fallback — keine Live-Fundamentaldaten verfuegbar',
                })

    return candidates


# ===== STUFE 3: DISRUPTIONS-OVERLAY (LLM) =====

def _disruptions_overlay(candidates, active_trends, etf_universe):
    """
    LLM analysiert die Kombination: fundamental schwach + Disruptions-bedroht.
    Generiert: combined_thesis, confidence, pair_trade, squeeze_risk.
    """
    if not ANTHROPIC_API_KEY:
        print("    [SKIP] Kein ANTHROPIC_API_KEY — verwende vereinfachte Short-Analyse")
        return _simplified_short_results(candidates)

    if not candidates:
        return []

    # Batch: Max 10 Kandidaten an LLM
    batch = candidates[:10]

    candidates_text = ""
    for c in batch:
        candidates_text += f"\n- {c['ticker']} ({c['name']}): "
        candidates_text += f"Weakness: {', '.join(c['weakness_flags'])}. "
        candidates_text += f"Bedroht von: {c['threat_source_name']} ({c['threat_source_id']}, Phase: {c['threat_phase']}, Velocity: {c['threat_velocity']})"

    # Relevante Long-ETFs fuer Pair Trades
    trend_etfs = {}
    for t in active_trends:
        cat_etfs = [e['ticker'] for e in etf_universe if t['id'] in e.get('category_ids', [])]
        if cat_etfs:
            trend_etfs[t['id']] = cat_etfs[:3]

    etfs_text = json.dumps(trend_etfs, indent=2)

    system_prompt = """Du bist ein Short-Analyst fuer einen systematischen Macro-Hedge-Fund.
Antworte AUSSCHLIESSLICH mit validem JSON. Kein Markdown, nur JSON."""

    user_prompt = f"""Analysiere diese Short-Kandidaten. Jeder ist fundamental schwach UND von einem Disruptions-Trend bedroht.

Kandidaten:{candidates_text}

Verfuegbare Long-ETFs pro Trend (fuer Pair Trades):
{etfs_text}

Fuer JEDEN Kandidaten, antworte mit diesem JSON-Array:
[
  {{
    "ticker": "<Ticker>",
    "combined_thesis": "<1-2 Saetze: Warum fundamental schwach + Disruptions-bedroht>",
    "confidence": "<HIGH | MEDIUM | LOW>",
    "squeeze_risk": <true wenn Short Interest > 20% oder Market Cap < $5B>,
    "takeover_risk": <true wenn guenstig bewertet + attraktive Assets>,
    "turnaround_risk": <true wenn neues Management oder Restrukturierung>,
    "pair_trade": {{
      "long": "<ETF Ticker fuer Long-Seite>",
      "short": "<dieser Ticker>",
      "thesis": "<1 Satz Pair-Trade These>"
    }}
  }}
]

Wenn ein Kandidat kein guter Short ist (zu viel Squeeze/Takeover/Turnaround Risiko), setze confidence auf LOW."""

    try:
        result = _call_anthropic(system_prompt, user_prompt)
        parsed = _parse_json_response(result)
        if parsed and isinstance(parsed, list):
            # Merge LLM-Ergebnisse mit Fundamental-Daten
            return _merge_short_results(candidates, parsed)
        else:
            return _simplified_short_results(candidates)
    except Exception as e:
        print(f"    [ERROR] Short LLM-Analyse fehlgeschlagen: {e}")
        return _simplified_short_results(candidates)


def _merge_short_results(candidates, llm_results):
    """Merge Fundamental-Daten mit LLM-Analyse."""
    llm_map = {r['ticker']: r for r in llm_results if 'ticker' in r}
    merged = []

    for c in candidates:
        ticker = c['ticker']
        llm = llm_map.get(ticker, {})

        merged.append({
            'ticker': ticker,
            'name': c.get('name', ''),
            'market_cap': c.get('market_cap', 0),
            'debt_equity': c.get('debt_equity', 0),
            'revenue_growth_yoy': c.get('revenue_growth_yoy', 0),
            'short_interest': c.get('short_interest', 0),
            'weakness_flags': c.get('weakness_flags', []),
            'threat_source_id': c.get('threat_source_id', ''),
            'threat_source_name': c.get('threat_source_name', ''),
            'combined_thesis': llm.get('combined_thesis', f"Fundamental schwach + bedroht von {c.get('threat_source_name', '?')}"),
            'confidence': llm.get('confidence', 'MEDIUM'),
            'squeeze_risk': llm.get('squeeze_risk', False),
            'takeover_risk': llm.get('takeover_risk', False),
            'turnaround_risk': llm.get('turnaround_risk', False),
            'pair_trade': llm.get('pair_trade', {}),
        })

    return merged


def _simplified_short_results(candidates):
    """Vereinfachte Short-Ergebnisse ohne LLM."""
    results = []
    for c in candidates:
        results.append({
            'ticker': c['ticker'],
            'name': c.get('name', ''),
            'market_cap': c.get('market_cap', 0),
            'debt_equity': c.get('debt_equity', 0),
            'revenue_growth_yoy': c.get('revenue_growth_yoy', 0),
            'short_interest': c.get('short_interest', 0),
            'weakness_flags': c.get('weakness_flags', []),
            'threat_source_id': c.get('threat_source_id', ''),
            'threat_source_name': c.get('threat_source_name', ''),
            'combined_thesis': f"Fundamental schwach + bedroht von {c.get('threat_source_name', '?')}. LLM-Analyse nicht verfuegbar.",
            'confidence': 'LOW',
            'squeeze_risk': False,
            'takeover_risk': False,
            'turnaround_risk': False,
            'pair_trade': {},
        })
    return results


# ===== HELPERS =====

def _call_anthropic(system_prompt, user_prompt):
    """Anthropic API Call."""
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
        'messages': [{'role': 'user', 'content': user_prompt}]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data.get('content', [])
    return '\n'.join(c.get('text', '') for c in content if c.get('type') == 'text')


def _parse_json_response(text):
    """Parse JSON aus LLM Response."""
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
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start >= 0 and end > start:
            try:
                return [json.loads(cleaned[start:end + 1])]
            except json.JSONDecodeError:
                pass
    return None


def _safe_float(val):
    """Sicher zu float konvertieren."""
    try:
        if val is None or val == '':
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0
