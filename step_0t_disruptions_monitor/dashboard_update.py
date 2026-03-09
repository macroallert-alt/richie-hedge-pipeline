#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/dashboard_update.py
Schreibt den "disruptions" Block in latest.json.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 3 §16

Neuer Top-Level Key "disruptions" neben "rotation", "g7", etc.
Kompakt fuer Dashboard Card + Circle.
"""

import os
import json
from datetime import datetime, timedelta, timezone


def update_dashboard_json(dashboard_path, trends, exposure_result, short_results,
                          contrarian_alerts, causal_chains, dependencies,
                          convergence_zones, readiness_score, readiness_label, meta):
    """
    Aktualisiere latest.json mit dem disruptions Block.

    Args:
        dashboard_path: Pfad zu latest.json
        trends: Alle bewerteten Trends
        exposure_result: Exposure Check Ergebnis
        short_results: Short-Kandidaten
        contrarian_alerts: Contrarian Alerts
        causal_chains: Causal Chain Daten
        dependencies: Dependency-Kanten
        convergence_zones: Convergence Zones
        readiness_score: Disruption Readiness Score (0-100)
        readiness_label: FUTURE-READY / GAPS_PRESENT / BACKWARD-LOOKING
        meta: Run-Metadaten
    """
    # Lade bestehendes Dashboard
    dashboard = {}
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            dashboard = json.load(f)

    run_date = meta.get('run_date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))

    # Naechster Run (1 Woche spaeter)
    try:
        next_run = (datetime.strptime(run_date, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d')
    except ValueError:
        next_run = ''

    # --- Trends kompakt ---
    blind_spots = exposure_result.get('blind_spots', [])
    blind_spot_ids = {bs['category'] for bs in blind_spots}
    threats = exposure_result.get('threats', [])

    trends_compact = []
    for t in trends:
        trends_compact.append({
            'id': t['id'],
            'name': t['name'],
            'status': t.get('watchlist_status', 'WATCH'),
            'phase': t.get('phase', 'EMERGING'),
            'maturity': t.get('maturity', 0),
            'momentum': t.get('momentum', 0),
            'acceleration': t.get('acceleration', 0),
            'relevance': t.get('relevance', 0),
            'hype': t.get('hype', 0),
            'inflection_score': t.get('inflection_score', 0),
            'velocity': t.get('velocity_label', 'LOW'),
            'crowding': t.get('crowding', 0),
            'model_risk': t.get('model_risk', 'NONE'),
            'headline': t.get('headline', ''),
            'bull_case': t.get('bull_case', ''),
            'bear_case': t.get('bear_case', ''),
            'top_etf': t.get('top_etf', ''),
            'historical_analogy': t.get('historical_analogy', ''),
            'portfolio_exposure_pct': t.get('portfolio_exposure_pct', 0.0),
            'is_blind_spot': t['id'] in blind_spot_ids,
            's_curve_x': t.get('maturity', 0),
        })

    # --- Blind Spots kompakt ---
    blind_spots_compact = []
    for bs in blind_spots:
        blind_spots_compact.append({
            'category': bs.get('category', ''),
            'name': bs.get('name', ''),
            'exposure_pct': bs.get('exposure_pct', 0.0),
            'recommended_etfs': bs.get('recommended_etfs', []),
            'urgency': bs.get('urgency', 'LOW'),
        })

    # --- Threats kompakt ---
    threats_compact = []
    for th in threats:
        for asset in th.get('threatened_assets', []):
            threats_compact.append({
                'category': th.get('category', ''),
                'name': th.get('name', ''),
                'threatened_asset': asset.get('asset', ''),
                'v16_weight': asset.get('v16_weight', 0),
                'threat_level': asset.get('threat_level', 'LOW'),
                'reason': '',
            })

    # --- Contrarian Alerts kompakt ---
    contrarian_compact = []
    for c in contrarian_alerts:
        contrarian_compact.append({
            'sector': c.get('sector', ''),
            'etf': c.get('etf', ''),
            'tailwind_source': f"{c.get('tailwind_source', {}).get('category', '')} — {c.get('tailwind_source', {}).get('category_name', '')}",
            'alert_level': c.get('alert_level', 'WEAK'),
            'thesis_short': c.get('thesis_short', ''),
        })

    # --- Short Candidates kompakt ---
    short_compact = []
    for s in short_results:
        if s.get('confidence') in ('HIGH', 'MEDIUM'):
            short_compact.append({
                'ticker': s.get('ticker', ''),
                'name': s.get('name', ''),
                'threat_source': f"{s.get('threat_source_id', '')} — {s.get('threat_source_name', '')}",
                'fundamental_score': 'WEAK',
                'confidence': s.get('confidence', 'LOW'),
            })

    # --- Causal Chains Highlights ---
    chain_highlights = []
    for chain in causal_chains:
        next_act = chain.get('next_actionable', {})
        if next_act:
            chain_highlights.append({
                'trend': chain.get('trend', ''),
                'current_step': _get_current_step_name(chain),
                'next_step': next_act.get('effect', ''),
                'timing': next_act.get('timing', ''),
                'instruments': next_act.get('instruments', []),
            })

    # --- Dependencies Highlight ---
    deps_highlight = []
    for d in dependencies[:5]:  # Top 5 staerkste
        deps_highlight.append({
            'from': d.get('from', ''),
            'to': d.get('to', ''),
            'type': d.get('type', ''),
            'description': d.get('description', ''),
        })

    # --- Model Risk Alerts ---
    model_risk_alerts = [
        {
            'category': t['id'],
            'name': t['name'],
            'model_risk': t.get('model_risk', 'NONE'),
        }
        for t in trends
        if t.get('model_risk') in ('STRUCTURAL', 'PARADIGM')
    ]

    # --- Convergence Zones ---
    convergence_compact = []
    for z in convergence_zones:
        convergence_compact.append({
            'trends': z.get('trends', []),
            'description': z.get('description', ''),
        })

    # Counts
    active_count = sum(1 for t in trends if t.get('watchlist_status') == 'ACTIVE')
    watch_count = sum(1 for t in trends if t.get('watchlist_status') == 'WATCH')

    # --- Disruptions Block ---
    disruptions_block = {
        'date': run_date,
        'run_type': 'WEEKLY',

        'readiness_score': readiness_score,
        'readiness_label': readiness_label,
        'blind_spots_count': len(blind_spots_compact),
        'threats_count': len(threats_compact),
        'active_trends_count': active_count,
        'watch_trends_count': watch_count,
        'convergence_active': len(convergence_zones) > 0,

        'trends': trends_compact,
        'blind_spots': blind_spots_compact,
        'threats': threats_compact,
        'convergence_zones': convergence_compact,
        'contrarian_alerts': contrarian_compact,
        'top_short_candidates': short_compact,
        'causal_chains_highlights': chain_highlights,
        'dependencies_highlight': deps_highlight,
        'model_risk_alerts': model_risk_alerts,

        'meta': {
            'categories_scanned': meta.get('categories_scanned', 0),
            'deep_dive_count': meta.get('deep_dive_count', 0),
            'sources_total': meta.get('sources_total', 0),
            'avg_source_quality': meta.get('avg_source_quality', 0),
            'llm_model': meta.get('llm_model', ''),
            'run_duration_s': meta.get('run_duration_s', 0),
            'next_run': next_run,
        }
    }

    # --- Agent R Context erweitern ---
    agent_r_context = dashboard.get('agent_r_context', {})
    agent_r_context['disruptions_summary'] = {
        'readiness_score': readiness_score,
        'active_trends': [
            f"{t['id']} {t['name']}" for t in trends
            if t.get('watchlist_status') == 'ACTIVE'
        ],
        'blind_spots': [
            f"{bs['category']} {bs['name']} ({bs['exposure_pct']:.1%} Exposure)"
            for bs in blind_spots_compact
        ],
        'threats': [
            f"{th['name']} → {th['threatened_asset']} ({th['v16_weight']:.1%})"
            for th in threats_compact
        ],
        'convergence_active': len(convergence_zones) > 0,
        'top_contrarian': contrarian_compact[0]['thesis_short'] if contrarian_compact else '',
    }

    # --- Schreiben ---
    dashboard['disruptions'] = disruptions_block
    dashboard['agent_r_context'] = agent_r_context

    # Sicherstellen dass Verzeichnis existiert
    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)

    with open(dashboard_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)

    trend_count = len(trends_compact)
    print(f"    [DASHBOARD] disruptions Block geschrieben: {trend_count} Trends, Readiness {readiness_score}")


def _get_current_step_name(chain):
    """Finde den Namen des aktuellen Schritts auf der Causal Chain."""
    marker = chain.get('current_position_marker', 0)
    steps = chain.get('causal_chain', [])

    # Finde den Schritt der dem Marker am naechsten ist
    current_order = int(marker)
    for step in steps:
        if step.get('order', 0) == current_order:
            return step.get('effect', '')

    # Fallback: letzter IN_PROGRESS
    for step in reversed(steps):
        if step.get('status') == 'IN_PROGRESS':
            return step.get('effect', '')

    return steps[0].get('effect', '') if steps else ''
