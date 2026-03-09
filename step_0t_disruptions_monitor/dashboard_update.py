#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/dashboard_update.py
Schreibt den "disruptions" Block in latest.json.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 3 §16 + DISRUPTIONS_FRONTEND_ERWEITERUNG_SPEC

Phase A Erweiterung:
  NEU: briefing, decision_matrix, regime_context, conviction_scores,
       asymmetric_payoffs, crowding_alerts, vulnerability_watchlist,
       convergence_zones (erweitert um second_order_effects),
       regime_heatmap, g7_cross_references
  ENTFERNT: top_short_candidates (ersetzt durch vulnerability_watchlist)
"""

import os
import json
from datetime import datetime, timedelta, timezone


def update_dashboard_json(dashboard_path, trends, exposure_result, short_results,
                          contrarian_alerts, causal_chains, dependencies,
                          convergence_zones, readiness_score, readiness_label, meta,
                          intelligence_result=None):
    """
    Aktualisiere latest.json mit dem disruptions Block.

    Args:
        dashboard_path: Pfad zu latest.json
        trends: Alle bewerteten Trends
        exposure_result: Exposure Check Ergebnis
        short_results: Short-Kandidaten (Legacy, wird ignoriert wenn intelligence_result vorhanden)
        contrarian_alerts: Contrarian Alerts
        causal_chains: Causal Chain Daten
        dependencies: Dependency-Kanten
        convergence_zones: Convergence Zones (Original ODER erweitert)
        readiness_score: Disruption Readiness Score (0-100)
        readiness_label: FUTURE-READY / GAPS_PRESENT / BACKWARD-LOOKING
        meta: Run-Metadaten
        intelligence_result: Phase A Intelligence Briefing Ergebnis (optional, NEU)
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

    # Conviction Scores aus Intelligence Result (wenn vorhanden)
    conviction_scores = {}
    if intelligence_result:
        conviction_scores = intelligence_result.get('conviction_scores', {})

    trends_compact = []
    for t in trends:
        conv_data = conviction_scores.get(t['id'], {})
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
            # NEU: Conviction Score
            'conviction': conv_data.get('conviction', 0),
            'conviction_label': conv_data.get('label', ''),
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
    # Verwende erweiterte Zones aus Intelligence Result wenn vorhanden
    if intelligence_result and intelligence_result.get('second_order_effects_zones'):
        zones_to_write = intelligence_result['second_order_effects_zones']
    else:
        zones_to_write = convergence_zones

    convergence_compact = []
    for z in zones_to_write:
        zone_data = {
            'trends': z.get('trends', []),
            'description': z.get('description', ''),
        }
        # Phase A: Second Order Effects wenn vorhanden
        if z.get('primary_effects'):
            zone_data['primary_effects'] = z['primary_effects']
        if z.get('second_order_effects'):
            zone_data['second_order_effects'] = z['second_order_effects']
        if z.get('net_portfolio_impact'):
            zone_data['net_portfolio_impact'] = z['net_portfolio_impact']
        convergence_compact.append(zone_data)

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
        'convergence_active': len(zones_to_write) > 0,

        'trends': trends_compact,
        'blind_spots': blind_spots_compact,
        'threats': threats_compact,
        'convergence_zones': convergence_compact,
        'contrarian_alerts': contrarian_compact,
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

    # --- Phase A: Neue Bloecke aus Intelligence Result ---
    if intelligence_result:
        # Briefing Card (Spec §2)
        if intelligence_result.get('briefing'):
            disruptions_block['briefing'] = intelligence_result['briefing']

        # Decision Matrix (Spec §3)
        if intelligence_result.get('decision_matrix'):
            # Conviction Scores aus Python-Berechnung in decision_matrix einfuegen
            dm_list = intelligence_result['decision_matrix']
            for entry in dm_list:
                tid = entry.get('trend_id', '')
                conv_data = conviction_scores.get(tid, {})
                if conv_data:
                    entry['conviction'] = conv_data.get('conviction', entry.get('conviction', 0))
                    entry['conviction_label'] = conv_data.get('label', entry.get('conviction_label', 'LOW'))
            disruptions_block['decision_matrix'] = dm_list

        # Regime Context (Spec §4)
        if intelligence_result.get('regime_context'):
            disruptions_block['regime_context'] = intelligence_result['regime_context']

        # Conviction Scores (Spec §7)
        if intelligence_result.get('conviction_scores'):
            disruptions_block['conviction_scores'] = intelligence_result['conviction_scores']

        # Asymmetric Payoffs (Spec §9)
        if intelligence_result.get('asymmetric_payoffs'):
            disruptions_block['asymmetric_payoffs'] = intelligence_result['asymmetric_payoffs']

        # Crowding Alerts (Spec §6)
        if intelligence_result.get('crowding_alerts'):
            disruptions_block['crowding_alerts'] = intelligence_result['crowding_alerts']

        # Vulnerability Watchlist (Spec §8 — ersetzt top_short_candidates)
        if intelligence_result.get('vulnerability_watchlist'):
            disruptions_block['vulnerability_watchlist'] = intelligence_result['vulnerability_watchlist']

        # Regime Heatmap (Spec §16)
        if intelligence_result.get('regime_heatmap'):
            disruptions_block['regime_heatmap'] = intelligence_result['regime_heatmap']

        # G7 Cross-References (Spec §23)
        if intelligence_result.get('g7_cross_references'):
            disruptions_block['g7_cross_references'] = intelligence_result['g7_cross_references']

    # --- Legacy: top_short_candidates nur wenn KEIN intelligence_result ---
    if not intelligence_result:
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
        disruptions_block['top_short_candidates'] = short_compact

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
        'convergence_active': len(zones_to_write) > 0,
        'top_contrarian': contrarian_compact[0]['thesis_short'] if contrarian_compact else '',
    }

    # Phase A: Erweiterte Agent R Summary
    if intelligence_result:
        regime_ctx = intelligence_result.get('regime_context', {})
        agent_r_context['disruptions_summary']['regime'] = regime_ctx.get('current_regime', 'NEUTRAL')
        agent_r_context['disruptions_summary']['regime_sizing_multiplier'] = (
            regime_ctx.get('regime_rules', {}).get('sizing_multiplier', 1.0)
        )
        vuln_list = intelligence_result.get('vulnerability_watchlist', [])
        if vuln_list:
            agent_r_context['disruptions_summary']['vulnerable_assets'] = [
                f"{v['asset']} ({v['aggregate_threat_level']})"
                for v in vuln_list[:5]
            ]

    # --- Schreiben ---
    dashboard['disruptions'] = disruptions_block
    dashboard['agent_r_context'] = agent_r_context

    # Sicherstellen dass Verzeichnis existiert
    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)

    with open(dashboard_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)

    trend_count = len(trends_compact)
    intel_str = " + Intelligence Briefing" if intelligence_result else ""
    print(f"    [DASHBOARD] disruptions Block geschrieben: {trend_count} Trends, Readiness {readiness_score}{intel_str}")


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
