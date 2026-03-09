#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/dependencies.py
Dependency-Netzwerk und Convergence Zone Detection.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 1 §6

Kanten-Typen:
  ACCELERATES (gruen): Trend A beschleunigt Trend B
  THREATENS (rot): Trend A bedroht Trend B
  REQUIRES (blau): Trend A braucht Trend B
  ENABLES (gruen gestrichelt): Trend A ermoeglicht Trend B

Convergence Zone: 2+ verbundene Trends gleichzeitig in Inflection Zone (Maturity 25-50, Momentum > 50)
"""

import os
import json
import requests

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
LLM_MODEL = 'claude-sonnet-4-20250514'


def update_dependencies(trends, seed_dependencies):
    """
    Aktualisiere das Dependency-Netzwerk.

    Beim ersten Lauf: Nutze Seed-Dependencies aus Config.
    Spaeter (wenn LLM verfuegbar): LLM aktualisiert/ergaenzt die Kanten.

    Args:
        trends: Alle bewerteten Trends
        seed_dependencies: Seed-Kanten aus Config

    Returns:
        Aktualisierte Liste von Dependency-Dicts
    """
    active_ids = {t['id'] for t in trends}

    # Filtere Seed-Dependencies auf aktive Trends
    active_deps = [
        d for d in seed_dependencies
        if d['from'] in active_ids and d['to'] in active_ids
    ]

    print(f"    [DEPS] {len(active_deps)} aktive Kanten (von {len(seed_dependencies)} Seeds)")

    # Optional: LLM-Update der Dependencies
    if ANTHROPIC_API_KEY and len(trends) >= 4:
        try:
            updated = _llm_update_dependencies(trends, active_deps)
            if updated:
                # Merge: Behalte alle Seed-Kanten, fuege neue hinzu
                merged = _merge_dependencies(active_deps, updated)
                print(f"    [DEPS] LLM-Update: {len(merged)} Kanten (vorher {len(active_deps)})")
                return merged
        except Exception as e:
            print(f"    [WARN] LLM Dependency-Update fehlgeschlagen: {e}")

    return active_deps


def detect_convergence_zones(trends, dependencies, thresholds):
    """
    Spec §6.5: Finde Cluster von verbundenen Trends nahe am Inflection Point.

    Bedingungen:
      - Maturity 25-50 (Inflection Zone)
      - Momentum > 50
      - Verbunden via ACCELERATES oder ENABLES mit strength > 0.5

    Returns:
        Liste von Convergence-Zone-Dicts
    """
    mat_min = thresholds.get('convergence_maturity_min', 25)
    mat_max = thresholds.get('convergence_maturity_max', 50)
    mom_min = thresholds.get('convergence_momentum_min', 50)
    str_min = thresholds.get('convergence_strength_min', 0.5)

    # Finde Trends in der Inflection Zone mit hohem Momentum
    inflection_trends = [
        t for t in trends
        if mat_min <= t.get('maturity', 0) <= mat_max
        and t.get('momentum', 0) > mom_min
    ]

    if len(inflection_trends) < 2:
        return []

    inflection_ids = {t['id'] for t in inflection_trends}

    # Finde Verbindungen zwischen Inflection-Trends
    zones = []
    seen_clusters = set()

    for trend_a in inflection_trends:
        a_id = trend_a['id']

        # Finde verbundene Inflection-Trends
        connected_ids = set()
        for dep in dependencies:
            if dep.get('type') not in ('ACCELERATES', 'ENABLES'):
                continue
            if dep.get('strength', 0) < str_min:
                continue

            if dep['from'] == a_id and dep['to'] in inflection_ids:
                connected_ids.add(dep['to'])
            elif dep['to'] == a_id and dep['from'] in inflection_ids:
                connected_ids.add(dep['from'])

        if not connected_ids:
            continue

        # Baue Cluster
        cluster = frozenset([a_id] + list(connected_ids))
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)

        cluster_trends = [t for t in inflection_trends if t['id'] in cluster]
        trend_names = [t['name'] for t in cluster_trends]

        zone = {
            'trends': sorted(list(cluster)),
            'trend_names': trend_names,
            'alert_level': 'CONVERGENCE',
            'description': f"Convergence Zone: {' + '.join(trend_names)}",
            'avg_maturity': round(
                sum(t.get('maturity', 0) for t in cluster_trends) / len(cluster_trends)
            ),
            'avg_momentum': round(
                sum(t.get('momentum', 0) for t in cluster_trends) / len(cluster_trends)
            ),
        }
        zones.append(zone)

    # Update Trends mit convergence_member
    for t in trends:
        members = []
        for z in zones:
            if t['id'] in z['trends']:
                members.extend(tid for tid in z['trends'] if tid != t['id'])
        t['convergence_member'] = ','.join(sorted(set(members))) if members else ''

    if zones:
        print(f"    [CONVERGENCE] {len(zones)} Zones erkannt:")
        for z in zones:
            print(f"      {z['description']} (Maturity ~{z['avg_maturity']}, Momentum ~{z['avg_momentum']})")

    return zones


# ===== LLM DEPENDENCY UPDATE =====

def _llm_update_dependencies(trends, current_deps):
    """
    LLM analysiert aktive Trends und schlaegt neue/geaenderte Verbindungen vor.
    """
    trends_text = ""
    for t in trends:
        trends_text += f"\n- {t['id']} {t['name']}: Phase={t.get('phase','?')}, Maturity={t.get('maturity',0)}, Momentum={t.get('momentum',0)}"

    current_deps_text = ""
    for d in current_deps:
        current_deps_text += f"\n- {d['from']} → {d['to']}: {d['type']} (Strength {d.get('strength', 0)})"

    system_prompt = """Du bist ein Netzwerk-Analyst fuer Technologie-Trends.
Antworte AUSSCHLIESSLICH mit validem JSON. Kein Markdown, nur JSON."""

    user_prompt = f"""Analysiere die folgenden aktiven Disruptions-Trends und ihre Verbindungen.

Aktive Trends:{trends_text}

Bestehende Verbindungen:{current_deps_text}

Aufgabe:
1. Pruefe ob bestehende Verbindungen noch korrekt sind (Strength anpassen wenn noetig)
2. Identifiziere NEUE Verbindungen die fehlen
3. Entferne Verbindungen die nicht mehr relevant sind

Kanten-Typen:
- ACCELERATES: Trend A beschleunigt Trend B
- THREATENS: Trend A bedroht Trend B
- REQUIRES: Trend A braucht Trend B
- ENABLES: Trend A ermoeglicht Trend B (indirekt)

Antworte mit:
{{
  "updated_dependencies": [
    {{
      "from": "<Trend ID>",
      "to": "<Trend ID>",
      "type": "<ACCELERATES|THREATENS|REQUIRES|ENABLES>",
      "strength": <float 0-1>,
      "description": "<1 Satz Begruendung>"
    }}
  ],
  "removed": ["<from>-<to> Paare die entfernt werden sollten"],
  "new_count": <int, Anzahl neuer Kanten>
}}"""

    result = _call_anthropic(system_prompt, user_prompt)
    parsed = _parse_json_response(result)

    if parsed and 'updated_dependencies' in parsed:
        return parsed['updated_dependencies']
    return None


def _merge_dependencies(seed_deps, llm_deps):
    """
    Merge Seed-Dependencies mit LLM-Updates.
    LLM kann Strength aendern und neue Kanten hinzufuegen.
    Seed-Kanten werden nie geloescht (Stabilitaet).
    """
    # Index bestehende Kanten
    existing = {}
    for d in seed_deps:
        key = f"{d['from']}-{d['to']}"
        existing[key] = d.copy()

    # LLM-Updates anwenden
    for d in llm_deps:
        key = f"{d['from']}-{d['to']}"
        if key in existing:
            # Update Strength wenn LLM es aendert
            if 'strength' in d:
                existing[key]['strength'] = d['strength']
            if 'description' in d:
                existing[key]['description'] = d['description']
        else:
            # Neue Kante hinzufuegen
            # Validiere Pflichtfelder
            if all(k in d for k in ('from', 'to', 'type')):
                if d.get('type') in ('ACCELERATES', 'THREATENS', 'REQUIRES', 'ENABLES'):
                    existing[key] = {
                        'from': d['from'],
                        'to': d['to'],
                        'type': d['type'],
                        'strength': d.get('strength', 0.5),
                        'description': d.get('description', ''),
                        '_source': 'llm'
                    }

    return list(existing.values())


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
