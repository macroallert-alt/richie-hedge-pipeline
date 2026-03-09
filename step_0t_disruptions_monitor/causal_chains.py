#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/causal_chains.py
Second Order Effects + Causal Chain Timeline.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 2 §11

Pro ACTIVE/WATCH Trend:
  - Geordnete Sequenz von Effects mit Timing
  - Effect Status: REALIZED / IN_PROGRESS / EMERGING / PROJECTED
  - current_position_marker: Wo stehen wir auf der Chain?
  - next_actionable: Naechster investierbarer Schritt
  - Instrumente (Long + Short) pro Effect
"""

import os
import json
import requests

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
LLM_MODEL = 'claude-sonnet-4-20250514'


def run_causal_chains(trends, etf_universe):
    """
    Generiere Causal Chains fuer alle ACTIVE/WATCH Trends.

    Args:
        trends: Nur ACTIVE/WATCH Trends
        etf_universe: Thematische ETFs fuer Instrument-Zuordnung

    Returns:
        Liste von Causal-Chain-Dicts
    """
    if not trends:
        return []

    if not ANTHROPIC_API_KEY:
        print("    [SKIP] Kein ANTHROPIC_API_KEY — verwende Seed Causal Chains")
        return _seed_causal_chains(trends, etf_universe)

    chains = []

    for t in trends:
        cat_id = t['id']
        cat_name = t['name']

        # Relevante ETFs
        cat_etfs = [e['ticker'] for e in etf_universe if cat_id in e.get('category_ids', [])]

        try:
            print(f"    [CAUSAL] {cat_id} {cat_name}...")
            chain = _generate_causal_chain(t, cat_etfs)
            if chain:
                chains.append(chain)
        except Exception as e:
            print(f"    [WARN] Causal Chain {cat_id}: {e}")
            # Fallback auf Seed
            seed = _get_seed_chain(cat_id, cat_etfs)
            if seed:
                chains.append(seed)

    return chains


def _generate_causal_chain(trend, cat_etfs):
    """LLM-generierte Causal Chain fuer einen Trend."""
    cat_id = trend['id']
    cat_name = trend['name']
    maturity = trend.get('maturity', 50)
    phase = trend.get('phase', 'EMERGING')
    momentum = trend.get('momentum', 50)

    system_prompt = """Du bist ein Investment-Analyst spezialisiert auf kausale Analyse von Technologie-Trends.
Antworte AUSSCHLIESSLICH mit validem JSON. Kein Markdown, nur JSON."""

    user_prompt = f"""Erstelle eine Causal Chain fuer den Trend "{cat_name}" ({cat_id}).
Aktueller Stand: Maturity {maturity}/100, Phase {phase}, Momentum {momentum}/100.

Verfuegbare thematische ETFs: {', '.join(cat_etfs)}

Denke in kausalen Ketten: Was passiert zuerst? Was folgt daraus? Was sind die 2nd und 3rd Order Effects?

Antworte mit folgendem JSON-Schema:
{{
  "trend": "{cat_id} — {cat_name}",
  "causal_chain": [
    {{
      "order": 1,
      "effect": "<Was passiert>",
      "timing": "<Zeitraum, z.B. '2024-2025'>",
      "status": "<REALIZED | IN_PROGRESS | EMERGING | PROJECTED>",
      "instruments_long": ["<ETF oder Sektor>"],
      "instruments_short": ["<ETF oder Sektor, oder leer>"],
      "maturity_on_chain": <int 0-100, wie weit ist dieser spezifische Effect>
    }}
  ],
  "current_position_marker": <float, z.B. 2.5 = zwischen Schritt 2 und 3>,
  "next_actionable": {{
    "effect": "<Naechster investierbarer Effect>",
    "timing": "<z.B. '6-12 Monate'>",
    "instruments": ["<ETF-Ticker>"],
    "thesis": "<1-2 Saetze Investment-These>"
  }}
}}

Regeln:
- Mindestens 4, maximal 7 Effects pro Chain
- Mindestens 1 REALIZED, mindestens 1 PROJECTED
- current_position_marker muss zwischen dem letzten IN_PROGRESS und dem naechsten EMERGING liegen
- instruments_long: Bevorzuge ETFs aus der verfuegbaren Liste
- instruments_short: Nur wenn klar erkennbar wer verliert"""

    result = _call_anthropic(system_prompt, user_prompt)
    parsed = _parse_json_response(result)

    if parsed and 'causal_chain' in parsed:
        return parsed
    return None


# ===== SEED CAUSAL CHAINS =====

def _seed_causal_chains(trends, etf_universe):
    """Vordefinierte Causal Chains als Fallback (kein LLM noetig)."""
    chains = []
    for t in trends:
        cat_etfs = [e['ticker'] for e in etf_universe if t['id'] in e.get('category_ids', [])]
        seed = _get_seed_chain(t['id'], cat_etfs)
        if seed:
            chains.append(seed)
    return chains


def _get_seed_chain(cat_id, cat_etfs):
    """Hartcodierte Seed Chains fuer die wichtigsten Kategorien."""
    seeds = {
        'D1': {
            'trend': 'D1 — Artificial Intelligence',
            'causal_chain': [
                {
                    'order': 1, 'effect': 'Chip-Nachfrage explodiert',
                    'timing': '2023-2024', 'status': 'REALIZED',
                    'instruments_long': ['SMH', 'SOXX'], 'instruments_short': [],
                    'maturity_on_chain': 80
                },
                {
                    'order': 2, 'effect': 'Rechenzentrum-Ausbau beschleunigt',
                    'timing': '2024-2025', 'status': 'IN_PROGRESS',
                    'instruments_long': ['PAVE', 'NFRA'], 'instruments_short': [],
                    'maturity_on_chain': 60
                },
                {
                    'order': 3, 'effect': 'Strom-Nachfrage fuer AI explodiert',
                    'timing': '2025-2026', 'status': 'EMERGING',
                    'instruments_long': ['URA', 'URNM'], 'instruments_short': [],
                    'maturity_on_chain': 35
                },
                {
                    'order': 4, 'effect': 'Grid-Infrastruktur Engpass',
                    'timing': '2026-2027', 'status': 'PROJECTED',
                    'instruments_long': ['PAVE', 'IFRA', 'COPX'], 'instruments_short': [],
                    'maturity_on_chain': 15
                },
                {
                    'order': 5, 'effect': 'Kernkraft-Renaissance fuer Baseload',
                    'timing': '2027-2028', 'status': 'PROJECTED',
                    'instruments_long': ['URA', 'URNM'], 'instruments_short': [],
                    'maturity_on_chain': 10
                },
                {
                    'order': 6, 'effect': 'AI verdraengt traditionelle Content-Produktion',
                    'timing': '2026-2028', 'status': 'EMERGING',
                    'instruments_long': ['ARKW'], 'instruments_short': ['PARA'],
                    'maturity_on_chain': 25
                },
            ],
            'current_position_marker': 2.5,
            'next_actionable': {
                'effect': 'Strom-Nachfrage fuer AI explodiert',
                'timing': '6-12 Monate',
                'instruments': ['URA', 'URNM'],
                'thesis': 'AI-Rechenzentren treiben Baseload-Nachfrage. Kernkraft einzige skalierbare Loesung. Uran-Nachfrage steigt bevor Fusion reif ist.'
            }
        },
        'D2': {
            'trend': 'D2 — Robotics & Automation',
            'causal_chain': [
                {
                    'order': 1, 'effect': 'Funktionierende Prototypen (Tesla Optimus, Figure)',
                    'timing': '2023-2025', 'status': 'IN_PROGRESS',
                    'instruments_long': ['BOTZ', 'ROBO'], 'instruments_short': [],
                    'maturity_on_chain': 45
                },
                {
                    'order': 2, 'effect': 'Erste Warehouse/Retail Deployments',
                    'timing': '2025-2026', 'status': 'EMERGING',
                    'instruments_long': ['BOTZ', 'ROBO', 'IRBO'], 'instruments_short': [],
                    'maturity_on_chain': 25
                },
                {
                    'order': 3, 'effect': 'Arbeitskraefte-Substitution beginnt',
                    'timing': '2026-2028', 'status': 'PROJECTED',
                    'instruments_long': ['ROBO'], 'instruments_short': [],
                    'maturity_on_chain': 15
                },
                {
                    'order': 4, 'effect': 'Reshoring beschleunigt durch Robotik-Kostenparitaet',
                    'timing': '2027-2029', 'status': 'PROJECTED',
                    'instruments_long': ['PAVE', 'IFRA'], 'instruments_short': [],
                    'maturity_on_chain': 10
                },
            ],
            'current_position_marker': 1.5,
            'next_actionable': {
                'effect': 'Erste Warehouse/Retail Deployments',
                'timing': '6-18 Monate',
                'instruments': ['BOTZ', 'ROBO'],
                'thesis': 'Humanoide Roboter erreichen kommerzielle Deployments. Sektor-ETFs bieten breite Exposure ohne Einzelwetten.'
            }
        },
        'D3': {
            'trend': 'D3 — Energy Transition',
            'causal_chain': [
                {
                    'order': 1, 'effect': 'Solar/Wind Kostenparitaet erreicht',
                    'timing': '2020-2024', 'status': 'REALIZED',
                    'instruments_long': ['TAN', 'ICLN'], 'instruments_short': [],
                    'maturity_on_chain': 75
                },
                {
                    'order': 2, 'effect': 'Batterie-Kosten fallen unter $100/kWh',
                    'timing': '2024-2026', 'status': 'IN_PROGRESS',
                    'instruments_long': ['LIT'], 'instruments_short': [],
                    'maturity_on_chain': 55
                },
                {
                    'order': 3, 'effect': 'SMR erste kommerzielle Deployments',
                    'timing': '2026-2028', 'status': 'EMERGING',
                    'instruments_long': ['URA', 'URNM'], 'instruments_short': [],
                    'maturity_on_chain': 20
                },
                {
                    'order': 4, 'effect': 'Wasserstoff-Infrastruktur skaliert',
                    'timing': '2027-2030', 'status': 'PROJECTED',
                    'instruments_long': ['HYDR'], 'instruments_short': [],
                    'maturity_on_chain': 10
                },
                {
                    'order': 5, 'effect': 'Fossile Energie strukturell ruecklaeufig',
                    'timing': '2028-2035', 'status': 'PROJECTED',
                    'instruments_long': ['ICLN'], 'instruments_short': ['XLE'],
                    'maturity_on_chain': 5
                },
            ],
            'current_position_marker': 2.3,
            'next_actionable': {
                'effect': 'SMR erste kommerzielle Deployments',
                'timing': '12-24 Monate',
                'instruments': ['URA', 'URNM'],
                'thesis': 'Small Modular Reactors naehern sich kommerzieller Reife. Uran-Nachfrage steigt strukturell.'
            }
        },
        'D4': {
            'trend': 'D4 — Biotech & Longevity',
            'causal_chain': [
                {
                    'order': 1, 'effect': 'GLP-1 Revolution (Ozempic/Wegovy)',
                    'timing': '2023-2025', 'status': 'REALIZED',
                    'instruments_long': ['XBI'], 'instruments_short': [],
                    'maturity_on_chain': 70
                },
                {
                    'order': 2, 'effect': 'Gene Editing erste Therapien zugelassen',
                    'timing': '2024-2026', 'status': 'IN_PROGRESS',
                    'instruments_long': ['ARKG', 'GNOM'], 'instruments_short': [],
                    'maturity_on_chain': 40
                },
                {
                    'order': 3, 'effect': 'Longevity-Biotech kommerzialisiert',
                    'timing': '2026-2030', 'status': 'EMERGING',
                    'instruments_long': ['ARKG'], 'instruments_short': [],
                    'maturity_on_chain': 15
                },
                {
                    'order': 4, 'effect': 'Gesundheitssystem-Kosten transformiert',
                    'timing': '2028-2035', 'status': 'PROJECTED',
                    'instruments_long': ['XBI', 'EDOC'], 'instruments_short': ['PFE'],
                    'maturity_on_chain': 5
                },
            ],
            'current_position_marker': 1.8,
            'next_actionable': {
                'effect': 'Gene Editing erste Therapien zugelassen',
                'timing': '6-18 Monate',
                'instruments': ['ARKG', 'GNOM'],
                'thesis': 'CRISPR/Base Editing naehert sich breiterer Zulassung. Genomics-ETFs bieten diversifizierte Exposure.'
            }
        },
    }

    chain = seeds.get(cat_id)
    if chain:
        # ETFs aktualisieren falls vorhanden
        if cat_etfs and chain.get('next_actionable', {}).get('instruments'):
            pass  # Behalte Seed-Instrumente
        return chain
    return None


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
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None
