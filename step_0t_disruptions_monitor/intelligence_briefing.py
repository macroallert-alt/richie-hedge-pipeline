#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/intelligence_briefing.py
Intelligence Briefing — Phase A Erweiterung
Spec: DISRUPTIONS_FRONTEND_ERWEITERUNG_SPEC TEIL 1-2, TEIL 4-5

Hybrid-Ansatz:
  Python berechnet: conviction_scores, crowding_alerts, regime_context (Rules),
                    regime_heatmap (statische Matrix)
  LLM generiert:   briefing Text, decision_matrix (Trigger/Regime-Impact Texte),
                    asymmetric_payoffs (Szenarien), vulnerability recommendations,
                    second_order_effects, g7_cross_references, regime_fit pro Trend

Wird NACH Dependencies und Vulnerability Analysis aufgerufen, VOR Sheet Write.
"""

import os
import json
import requests

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
LLM_MODEL = 'claude-sonnet-4-20250514'


# =====================================================================
# HAUPTFUNKTION
# =====================================================================

def generate_intelligence_briefing(trends, exposure_result, contrarian_alerts,
                                   causal_chains, convergence_zones,
                                   vulnerability_watchlist, v16_weights,
                                   v16_regime, config, run_date):
    """
    Generiere alle Phase-A Erweiterungs-Bloecke.

    Returns:
        Dict mit allen neuen Bloecken fuer latest.json:
        {
            'briefing': {},
            'decision_matrix': [],
            'regime_context': {},
            'conviction_scores': {},
            'asymmetric_payoffs': {},
            'crowding_alerts': [],
            'vulnerability_watchlist': [],  # angereichert mit LLM-Texten
            'second_order_effects_zones': [],  # erweiterte convergence_zones
            'regime_heatmap': {},
            'g7_cross_references': [],
        }
    """
    print("\n--- INTELLIGENCE BRIEFING ---")

    result = {}

    # ===== 1. REGIME CONTEXT (Python — deterministisch) =====
    regime_context = _build_regime_context(v16_regime)
    result['regime_context'] = regime_context
    print(f"    [REGIME] {regime_context['current_regime']} — Sizing ×{regime_context['regime_rules']['sizing_multiplier']}")

    # ===== 2. CONVICTION SCORES (Python — Formel, OHNE regime_fit) =====
    # regime_fit kommt vom LLM spaeter, wird dann nachtraeglich eingerechnet
    conviction_partial = _calculate_conviction_scores_partial(trends)

    # ===== 3. CROWDING ALERTS (Python — regelbasiert) =====
    crowding_alerts = _detect_crowding_alerts(trends)
    result['crowding_alerts'] = crowding_alerts
    print(f"    [CROWDING] {len(crowding_alerts)} Alerts")

    # ===== 4. REGIME HEATMAP (Python — statische Matrix) =====
    regime_heatmap = _build_regime_heatmap(trends, v16_regime)
    result['regime_heatmap'] = regime_heatmap
    print(f"    [HEATMAP] {len(regime_heatmap.get('matrix', {}))} Trends in Matrix")

    # ===== 5. LLM CALL — Texte und Einschaetzungen =====
    if ANTHROPIC_API_KEY:
        try:
            llm_result = _run_llm_briefing(
                trends=trends,
                exposure_result=exposure_result,
                contrarian_alerts=contrarian_alerts,
                causal_chains=causal_chains,
                convergence_zones=convergence_zones,
                vulnerability_watchlist=vulnerability_watchlist,
                v16_weights=v16_weights,
                regime_context=regime_context,
                config=config,
                run_date=run_date,
            )
            print(f"    [LLM] Briefing generiert")
        except Exception as e:
            print(f"    [ERROR] LLM Briefing fehlgeschlagen: {e}")
            llm_result = _fallback_llm_result(trends, regime_context, run_date)
    else:
        print("    [SKIP] Kein ANTHROPIC_API_KEY — verwende Fallback")
        llm_result = _fallback_llm_result(trends, regime_context, run_date)

    # ===== 6. CONVICTION SCORES FINALISIEREN (Python + LLM regime_fit) =====
    regime_fits = llm_result.get('regime_fits', {})
    conviction_scores = _finalize_conviction_scores(conviction_partial, regime_fits, regime_context)
    result['conviction_scores'] = conviction_scores
    print(f"    [CONVICTION] {len(conviction_scores)} Scores berechnet")

    # ===== 7. VULNERABILITY WATCHLIST ANREICHERN =====
    vuln_enriched = _enrich_vulnerability_watchlist(
        vulnerability_watchlist, llm_result.get('vulnerability_texts', {})
    )
    result['vulnerability_watchlist'] = vuln_enriched

    # ===== 8. CONVERGENCE ZONES MIT SECOND ORDER EFFECTS =====
    enriched_zones = _enrich_convergence_zones(
        convergence_zones, llm_result.get('second_order_effects', [])
    )
    result['second_order_effects_zones'] = enriched_zones

    # ===== 9. RESTLICHE LLM-BLOECKE UEBERNEHMEN =====
    result['briefing'] = llm_result.get('briefing', {})
    result['decision_matrix'] = llm_result.get('decision_matrix', [])
    result['asymmetric_payoffs'] = llm_result.get('asymmetric_payoffs', {})
    result['g7_cross_references'] = llm_result.get('g7_cross_references', [])

    print(f"    [BRIEFING] Komplett: {len(result)} Bloecke generiert")
    return result


# =====================================================================
# PYTHON-BERECHNUNGEN (deterministisch)
# =====================================================================

# ----- REGIME CONTEXT -----

REGIME_RULES = {
    'EXPANSION':        {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    'RISK_ON':          {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    'BROAD_RISK_ON':    {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    'SELECTIVE':        {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    'TRANSITION':       {'sizing_multiplier': 0.75, 'min_conviction': 50, 'min_asymmetry': 2.0, 'trigger_strictness': 'MODERATE'},
    'NEUTRAL':          {'sizing_multiplier': 0.75, 'min_conviction': 50, 'min_asymmetry': 2.0, 'trigger_strictness': 'MODERATE'},
    'CONFLICTED':       {'sizing_multiplier': 0.75, 'min_conviction': 50, 'min_asymmetry': 2.0, 'trigger_strictness': 'MODERATE'},
    'CONTRACTION':      {'sizing_multiplier': 0.5,  'min_conviction': 70, 'min_asymmetry': 3.0, 'trigger_strictness': 'STRICT'},
    'RISK_OFF':         {'sizing_multiplier': 0.5,  'min_conviction': 70, 'min_asymmetry': 3.0, 'trigger_strictness': 'STRICT'},
    'BROAD_RISK_OFF':   {'sizing_multiplier': 0.5,  'min_conviction': 70, 'min_asymmetry': 3.0, 'trigger_strictness': 'STRICT'},
    'CRISIS':           {'sizing_multiplier': 0.0,  'min_conviction': 999, 'min_asymmetry': 999, 'trigger_strictness': 'BLOCKED'},
    'RISK_OFF_FORCED':  {'sizing_multiplier': 0.0,  'min_conviction': 999, 'min_asymmetry': 999, 'trigger_strictness': 'BLOCKED'},
}

REGIME_COLORS = {
    'EXPANSION': '#22C55E', 'RISK_ON': '#22C55E', 'BROAD_RISK_ON': '#22C55E', 'SELECTIVE': '#22C55E',
    'TRANSITION': '#EAB308', 'NEUTRAL': '#EAB308', 'CONFLICTED': '#EAB308',
    'CONTRACTION': '#F97316', 'RISK_OFF': '#F97316', 'BROAD_RISK_OFF': '#F97316',
    'CRISIS': '#EF4444', 'RISK_OFF_FORCED': '#EF4444',
}


def _build_regime_context(v16_regime):
    """Spec §4.2: Baue regime_context Block."""
    regime = v16_regime if v16_regime in REGIME_RULES else 'NEUTRAL'
    rules = REGIME_RULES[regime]
    color = REGIME_COLORS.get(regime, '#EAB308')

    # Impact Summary Text
    mult = rules['sizing_multiplier']
    if mult == 0:
        summary = "CRISIS MODUS — Keine neuen Disruption-Trades. Alle Positionen auf HOLD/AVOID."
    elif mult < 1.0:
        summary = (f"Sizing auf {int(mult * 100)}% reduziert. "
                   f"Nur Trends mit Conviction >{rules['min_conviction']} "
                   f"und Asymmetrie >{rules['min_asymmetry']}:1 kommen in Frage. "
                   f"Trigger Events verschaerft.")
    else:
        summary = "Volle Auslastung. Standard-Thresholds aktiv."

    return {
        'current_regime': regime,
        'regime_color': color,
        'regime_impact_summary': summary,
        'regime_rules': {
            'sizing_multiplier': mult,
            'min_conviction_threshold': rules['min_conviction'],
            'min_asymmetry_threshold': rules['min_asymmetry'],
            'trigger_strictness': rules['trigger_strictness'],
        },
        'regime_history_4w': [],  # Wird von main.py aus History befuellt
    }


# ----- CONVICTION SCORES (Partial — ohne regime_fit) -----

def _calculate_conviction_scores_partial(trends):
    """
    Spec §7.1: Berechne Conviction-Komponenten OHNE regime_fit.
    Formel: Conviction = 0.30×Mom + 0.25×Inflection + 0.25×(100-Crowd) + 0.20×Regime_Fit
    regime_fit kommt vom LLM und wird spaeter eingerechnet.
    """
    partial = {}
    for t in trends:
        if t.get('watchlist_status') not in ('ACTIVE', 'WATCH'):
            continue

        tid = t['id']
        momentum = t.get('momentum', 0)
        inflection = t.get('inflection_score', 0)
        crowding = t.get('crowding', 0)

        # Inflection Proximity: min(100, inflection_score × 1.3)
        inflection_proximity = min(100, inflection * 1.3)

        momentum_contrib = 0.30 * momentum
        inflection_contrib = 0.25 * inflection_proximity
        crowding_inv_contrib = 0.25 * (100 - crowding)
        # regime_fit_contrib wird spaeter ergaenzt

        partial[tid] = {
            'momentum': momentum,
            'inflection_proximity': round(inflection_proximity, 1),
            'crowding': crowding,
            'momentum_contrib': round(momentum_contrib, 1),
            'inflection_contrib': round(inflection_contrib, 1),
            'crowding_inv_contrib': round(crowding_inv_contrib, 1),
            'partial_sum': round(momentum_contrib + inflection_contrib + crowding_inv_contrib, 1),
        }

    return partial


def _finalize_conviction_scores(partial, regime_fits, regime_context):
    """Finalisiere Conviction Scores mit regime_fit vom LLM."""
    scores = {}
    for tid, p in partial.items():
        # regime_fit: 0-100, default 50 wenn LLM nichts liefert
        regime_fit = regime_fits.get(tid, {}).get('score', 50)
        regime_fit_reason = regime_fits.get(tid, {}).get('reason', 'Keine LLM-Einschaetzung verfuegbar')

        regime_fit_contrib = 0.20 * regime_fit
        conviction = round(p['partial_sum'] + regime_fit_contrib)
        conviction = max(0, min(100, conviction))

        if conviction > 70:
            label = 'HIGH'
        elif conviction >= 40:
            label = 'MEDIUM'
        else:
            label = 'LOW'

        scores[tid] = {
            'conviction': conviction,
            'label': label,
            'components': {
                'momentum_contrib': p['momentum_contrib'],
                'inflection_contrib': p['inflection_contrib'],
                'crowding_inv_contrib': p['crowding_inv_contrib'],
                'regime_fit_contrib': round(regime_fit_contrib, 1),
            },
            'regime_fit': regime_fit,
            'regime_fit_reason': regime_fit_reason,
        }

    return scores


# ----- CROWDING ALERTS -----

def _detect_crowding_alerts(trends):
    """
    Spec §6.2: Regelbasierte Crowding Alerts.
    DANGER:   Crowding >75 UND Momentum 4W Delta < -5
    WARNING:  Crowding >75 UND Momentum stagnierend (4W Delta -5 bis 0)
    ELEVATED: Crowding >65 UND Momentum 4W Delta < -10
    """
    alerts = []
    for t in trends:
        if t.get('watchlist_status') not in ('ACTIVE', 'WATCH'):
            continue

        crowding = t.get('crowding', 0)
        momentum = t.get('momentum', 0)
        # acceleration dient als Proxy fuer 4W Delta (positiv = steigend)
        momentum_delta = t.get('acceleration', 0)

        alert_level = None

        if crowding > 75 and momentum_delta < -5:
            alert_level = 'DANGER'
        elif crowding > 75 and -5 <= momentum_delta <= 0:
            alert_level = 'WARNING'
        elif crowding > 65 and momentum_delta < -10:
            alert_level = 'ELEVATED'

        if alert_level:
            alerts.append({
                'trend_id': t['id'],
                'trend_name': t['name'],
                'crowding': crowding,
                'momentum': momentum,
                'momentum_4w_delta': momentum_delta,
                'alert_level': alert_level,
                'description': f"Crowding {crowding} bei "
                               f"{'fallender' if momentum_delta < 0 else 'stagnierender'} "
                               f"Momentum ({momentum_delta:+d} in 4W). "
                               f"{'Klassisches Distribution-Muster.' if alert_level == 'DANGER' else 'Aufmerksamkeit erhoehen.'}",
                'recommendation': '',  # Wird vom LLM ergaenzt
                'top_etf': t.get('top_etf', ''),
                'historical_parallel': '',  # Wird vom LLM ergaenzt
            })

    return alerts


# ----- REGIME HEATMAP (statische Matrix aus Spec §16.2) -----

REGIME_HEATMAP_SCORES = {
    'D1':  {'name': 'AI',           'EXPANSION': 90, 'TRANSITION': 70, 'CONTRACTION': 40, 'CRISIS': 20},
    'D2':  {'name': 'Robotics',     'EXPANSION': 80, 'TRANSITION': 55, 'CONTRACTION': 35, 'CRISIS': 15},
    'D3':  {'name': 'Energy',       'EXPANSION': 65, 'TRANSITION': 60, 'CONTRACTION': 50, 'CRISIS': 70},
    'D4':  {'name': 'Biotech',      'EXPANSION': 60, 'TRANSITION': 55, 'CONTRACTION': 50, 'CRISIS': 40},
    'D5':  {'name': 'Space',        'EXPANSION': 70, 'TRANSITION': 65, 'CONTRACTION': 55, 'CRISIS': 60},
    'D6':  {'name': 'Quantum',      'EXPANSION': 75, 'TRANSITION': 50, 'CONTRACTION': 30, 'CRISIS': 15},
    'D7':  {'name': 'Fintech',      'EXPANSION': 85, 'TRANSITION': 60, 'CONTRACTION': 45, 'CRISIS': 20},
    'D8':  {'name': 'Supply Chain', 'EXPANSION': 70, 'TRANSITION': 75, 'CONTRACTION': 60, 'CRISIS': 45},
    'D9':  {'name': 'Climate',      'EXPANSION': 75, 'TRANSITION': 70, 'CONTRACTION': 40, 'CRISIS': 25},
    'D10': {'name': 'Cyber',        'EXPANSION': 65, 'TRANSITION': 70, 'CONTRACTION': 75, 'CRISIS': 85},
    'D11': {'name': 'Demographics', 'EXPANSION': 50, 'TRANSITION': 55, 'CONTRACTION': 55, 'CRISIS': 50},
    'D12': {'name': 'Regulatory',   'EXPANSION': 40, 'TRANSITION': 55, 'CONTRACTION': 65, 'CRISIS': 80},
}


def _build_regime_heatmap(trends, v16_regime):
    """Spec §16.2: Baue regime_heatmap Block."""
    # Mappe V16 Regime auf Heatmap-Regime (4 Buckets)
    regime_bucket_map = {
        'EXPANSION': 'EXPANSION', 'RISK_ON': 'EXPANSION', 'BROAD_RISK_ON': 'EXPANSION', 'SELECTIVE': 'EXPANSION',
        'TRANSITION': 'TRANSITION', 'NEUTRAL': 'TRANSITION', 'CONFLICTED': 'TRANSITION',
        'CONTRACTION': 'CONTRACTION', 'RISK_OFF': 'CONTRACTION', 'BROAD_RISK_OFF': 'CONTRACTION',
        'CRISIS': 'CRISIS', 'RISK_OFF_FORCED': 'CRISIS',
    }
    current_bucket = regime_bucket_map.get(v16_regime, 'TRANSITION')

    # Nur aktive Trends in die Matrix
    active_ids = {t['id'] for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')}

    matrix = {}
    for tid, data in REGIME_HEATMAP_SCORES.items():
        # Alle 12 Trends aufnehmen (auch ARCHIVED fuer vollstaendige Matrix)
        matrix[tid] = {
            'name': data['name'],
            'scores': {
                'EXPANSION': data['EXPANSION'],
                'TRANSITION': data['TRANSITION'],
                'CONTRACTION': data['CONTRACTION'],
                'CRISIS': data['CRISIS'],
            }
        }

    return {
        'regimes': ['EXPANSION', 'TRANSITION', 'CONTRACTION', 'CRISIS'],
        'current_regime': current_bucket,
        'matrix': matrix,
    }


# =====================================================================
# LLM CALL
# =====================================================================

def _run_llm_briefing(trends, exposure_result, contrarian_alerts,
                      causal_chains, convergence_zones, vulnerability_watchlist,
                      v16_weights, regime_context, config, run_date):
    """
    EIN grosser Sonnet-Call der alle textbasierten Bloecke generiert.
    """
    # --- Input zusammenbauen ---
    active_watch = [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')]
    archived = [t for t in trends if t.get('watchlist_status') == 'ARCHIVED']

    trends_text = ""
    for t in active_watch:
        trends_text += (
            f"\n- {t['id']} {t['name']}: Status={t.get('watchlist_status')}, "
            f"Phase={t.get('phase')}, Maturity={t.get('maturity', 0)}, "
            f"Momentum={t.get('momentum', 0)}, Acceleration={t.get('acceleration', 0)}, "
            f"Inflection={t.get('inflection_score', 0)}, Crowding={t.get('crowding', 0)}, "
            f"TopETF={t.get('top_etf', '')}, "
            f"Headline={t.get('headline', '')}"
        )

    blind_spots = exposure_result.get('blind_spots', [])
    bs_text = ', '.join(f"{bs['category']} {bs['name']} ({bs.get('exposure_pct', 0):.1%})" for bs in blind_spots)

    convergence_text = ""
    for z in convergence_zones:
        convergence_text += f"\n- {' + '.join(z.get('trend_names', z.get('trends', [])))}: {z.get('description', '')}"

    contrarian_text = ""
    for c in contrarian_alerts[:3]:
        contrarian_text += f"\n- {c.get('etf', '')} ({c.get('sector', '')}): {c.get('thesis_short', '')}"

    chains_text = ""
    for ch in causal_chains[:5]:
        chains_text += f"\n- {ch.get('trend', '')}: Naechster Schritt = {ch.get('next_actionable', {}).get('effect', '?')}"

    vuln_text = ""
    for v in vulnerability_watchlist[:5]:
        threat_ids = ', '.join(t['trend_id'] for t in v.get('threatening_trends', []))
        vuln_text += f"\n- {v['asset']} ({v['v16_weight_pct']:.1f}%): bedroht von {threat_ids}"

    v16_text = ', '.join(f"{k}:{v:.1f}%" for k, v in v16_weights.items() if v != 0) if v16_weights else 'Keine Gewichte'

    g7_crossref = config.get('g7_disruptions_crossref', {})
    g7_text = json.dumps(g7_crossref, indent=2)

    regime = regime_context['current_regime']
    regime_mult = regime_context['regime_rules']['sizing_multiplier']

    system_prompt = """Du bist der Chief Intelligence Officer eines systematischen Macro Hedge Fund (Baldur Creek Capital).
Du schreibst auf Deutsch. Alle Analysen sind regime-konditioniert.
Antworte AUSSCHLIESSLICH mit validem JSON. Kein Markdown, keine Erklaerungen ausserhalb des JSON."""

    user_prompt = f"""Generiere den woechentlichen Intelligence Briefing fuer {run_date}.

AKTUELLES V16 REGIME: {regime} (Sizing-Multiplier: {regime_mult})

AKTIVE/WATCH TRENDS:{trends_text}

ARCHIVED TRENDS: {', '.join(f"{t['id']} {t['name']}" for t in archived)}

BLIND SPOTS: {bs_text if bs_text else 'Keine'}

CONVERGENCE ZONES:{convergence_text if convergence_text else ' Keine'}

CONTRARIAN ALERTS:{contrarian_text if contrarian_text else ' Keine'}

CAUSAL CHAINS:{chains_text if chains_text else ' Keine'}

VULNERABLE ASSETS:{vuln_text if vuln_text else ' Keine'}

V16 PORTFOLIO: {v16_text}

G7-DISRUPTIONS CROSS-REF MAP:
{g7_text}

Generiere folgendes JSON-Objekt (ALLE Felder sind Pflicht):

{{
  "briefing": {{
    "date": "{run_date}",
    "regime_context": "{regime}",
    "headline": "<Knackige Headline, 1 Satz, Deutsch>",
    "body": "<800-1200 Woerter Fliesstext. Kein Markdown, nur Fliesstext. Deutsch.>",
    "sections": [
      {{"title": "Strukturelle Verschiebungen", "content": "<Analyse der wichtigsten Trend-Bewegungen>"}},
      {{"title": "Convergence-Analyse", "content": "<Convergence Zones und deren Bedeutung>"}},
      {{"title": "Regime-Konditionierung", "content": "<Was bedeutet {regime} fuer Disruption-Trades?>"}},
      {{"title": "Blind Spots & Contrarian", "content": "<Blind Spots und Contrarian Opportunities>"}},
      {{"title": "Causal Chain Fruehwarnung", "content": "<Welche Chains stehen vor dem naechsten Schritt?>"}},
      {{"title": "Model Risk Bewertung", "content": "<Wo koennten unsere Modelle falsch liegen?>"}}
    ],
    "key_changes_this_week": ["<Aenderung 1>", "<Aenderung 2>", "<Aenderung 3>"]
  }},

  "decision_matrix": [
    {{
      "trend_id": "<ID>",
      "trend_name": "<Name>",
      "conviction": 0,
      "conviction_label": "<HIGH/MEDIUM/LOW>",
      "asymmetry": 0.0,
      "asymmetry_bull_pct": 0,
      "asymmetry_bear_pct": 0,
      "timeframe": "<3M/6M/12M/18M/24M>",
      "instrument": "<ETF Ticker>",
      "sizing_hint_pct": 0.0,
      "sizing_regime_adjusted_pct": 0.0,
      "trigger_event": "<Konkretes Event das den Einstieg triggert>",
      "regime_impact": "<Text: Was bedeutet {regime} fuer diesen Trade?>",
      "status": "<EXECUTE/WATCH_FOR_TRIGGER/HOLD/AVOID>"
    }}
  ],

  "asymmetric_payoffs": {{
    "<Trend-ID>": {{
      "bull_return_pct": 0,
      "bear_return_pct": 0,
      "ratio": 0.0,
      "ratio_label": "<EXCEPTIONAL/EXCELLENT/GOOD/MARGINAL/UNFAVORABLE>",
      "instrument": "<ETF>",
      "bull_scenario": "<1-2 Saetze Bull Case>",
      "bear_scenario": "<1-2 Saetze Bear Case>",
      "base_scenario": "<1-2 Saetze Base Case>",
      "probability_bull": 0,
      "probability_bear": 0,
      "probability_base": 0
    }}
  }},

  "regime_fits": {{
    "<Trend-ID>": {{
      "score": 0,
      "reason": "<1 Satz: Warum passt/passt nicht dieser Trend zum aktuellen Regime?>"
    }}
  }},

  "vulnerability_texts": {{
    "<Asset-Ticker>": {{
      "recommendation": "<1-2 Saetze: Was soll Richie mit dieser Position tun?>",
      "hedge_instrument": "<Konkreter Hedge-Vorschlag mit Instrument>"
    }}
  }},

  "second_order_effects": [
    {{
      "zone_trends": ["<ID1>", "<ID2>"],
      "primary_effects": [
        {{"asset": "<Ticker>", "direction": "BULLISH", "mechanism": "<1 Satz>"}}
      ],
      "second_order_effects": [
        {{
          "asset": "<Ticker>",
          "direction": "<BULLISH/BEARISH>",
          "mechanism": "<1 Satz: Wie genau wirkt der indirekte Effekt?>",
          "confidence": "<HIGH/MEDIUM/LOW>",
          "timeframe": "<6-12M/12-18M/12-24M/18-36M>"
        }}
      ],
      "net_portfolio_impact": "<1-2 Saetze: Netto-Effekt auf V16 Portfolio>"
    }}
  ],

  "g7_cross_references": [
    {{
      "disruption_trend_id": "<D-ID>",
      "disruption_trend_name": "<Name>",
      "g7_event_id": "<evt_id>",
      "g7_event_title": "<Titel des G7 Events>",
      "g7_country": "<Laenderkuerzel>",
      "relationship": "<ACCELERATES/AMPLIFIES/ENABLES/DECELERATES/DISRUPTS>",
      "description": "<1-2 Saetze>",
      "impact_on_portfolio": "<1 Satz Portfolio-Impact>"
    }}
  ]
}}

REGELN:
- decision_matrix: NUR fuer ACTIVE und WATCH Trends (keine ARCHIVED). sizing_hint_pct = sinnvolle Allocation (1-5%). sizing_regime_adjusted_pct = sizing_hint_pct × {regime_mult}. In CRISIS: alle Status = AVOID.
- asymmetric_payoffs: NUR fuer ACTIVE und WATCH Trends. ratio = abs(bull_return_pct / bear_return_pct). probability_bull + probability_bear + probability_base = 100.
- ratio_label: >4.0=EXCEPTIONAL, 3.0-4.0=EXCELLENT, 2.0-3.0=GOOD, 1.5-2.0=MARGINAL, <1.5=UNFAVORABLE
- regime_fits: Fuer JEDEN ACTIVE/WATCH Trend. Score 0-100 (100=perfekt zum Regime passend).
- vulnerability_texts: Fuer JEDES Asset in der Vulnerability Watchlist.
- second_order_effects: Fuer JEDE Convergence Zone. Mindestens 2 Second Order Effects pro Zone.
- g7_cross_references: Basierend auf der G7-Disruptions Cross-Ref Map. Nur relevante aktuelle Verbindungen.
- Alle Texte auf Deutsch.
- conviction und conviction_label in decision_matrix werden spaeter von Python ueberschrieben — trotzdem schaetzen.
"""

    response_text = _call_anthropic(system_prompt, user_prompt, max_tokens=4096)
    parsed = _parse_json_response(response_text)

    if not parsed:
        print("    [WARN] LLM Response konnte nicht geparst werden")
        return _fallback_llm_result(trends, regime_context, run_date)

    return parsed


# =====================================================================
# ENRICHMENT FUNCTIONS
# =====================================================================

def _enrich_vulnerability_watchlist(vulnerability_watchlist, vulnerability_texts):
    """Ergaenze Vulnerability Watchlist mit LLM-Texten."""
    for v in vulnerability_watchlist:
        asset = v['asset']
        texts = vulnerability_texts.get(asset, {})
        if texts:
            v['recommendation'] = texts.get('recommendation', v.get('recommendation', ''))
            v['hedge_instrument'] = texts.get('hedge_instrument', v.get('hedge_instrument', ''))
    return vulnerability_watchlist


def _enrich_convergence_zones(convergence_zones, second_order_effects):
    """Ergaenze Convergence Zones mit Second Order Effects vom LLM."""
    # Baue Lookup: frozenset(trend_ids) → effects
    effects_map = {}
    for soe in second_order_effects:
        zone_key = frozenset(soe.get('zone_trends', []))
        effects_map[zone_key] = soe

    enriched = []
    for z in convergence_zones:
        zone_copy = dict(z)
        zone_key = frozenset(z.get('trends', []))

        effects = effects_map.get(zone_key)
        if effects:
            zone_copy['primary_effects'] = effects.get('primary_effects', [])
            zone_copy['second_order_effects'] = effects.get('second_order_effects', [])
            zone_copy['net_portfolio_impact'] = effects.get('net_portfolio_impact', '')
        else:
            zone_copy['primary_effects'] = []
            zone_copy['second_order_effects'] = []
            zone_copy['net_portfolio_impact'] = ''

        enriched.append(zone_copy)

    return enriched


# =====================================================================
# FALLBACK (wenn kein API Key oder LLM-Fehler)
# =====================================================================

def _fallback_llm_result(trends, regime_context, run_date):
    """Minimale Fallback-Daten wenn LLM nicht verfuegbar."""
    active_watch = [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')]
    regime = regime_context['current_regime']
    mult = regime_context['regime_rules']['sizing_multiplier']

    # Fallback Briefing
    briefing = {
        'date': run_date,
        'regime_context': regime,
        'headline': f'Woechentlicher Disruptions-Scan — {regime} Regime',
        'body': f'LLM-Briefing nicht verfuegbar. {len(active_watch)} aktive Trends werden ueberwacht.',
        'sections': [],
        'key_changes_this_week': ['LLM-Analyse nicht verfuegbar — nur regelbasierte Daten'],
    }

    # Fallback Decision Matrix
    decision_matrix = []
    for t in active_watch:
        status = 'AVOID' if mult == 0 else 'WATCH_FOR_TRIGGER'
        decision_matrix.append({
            'trend_id': t['id'],
            'trend_name': t['name'],
            'conviction': 0,
            'conviction_label': 'LOW',
            'asymmetry': 0.0,
            'asymmetry_bull_pct': 0,
            'asymmetry_bear_pct': 0,
            'timeframe': '12M',
            'instrument': t.get('top_etf', ''),
            'sizing_hint_pct': 0.0,
            'sizing_regime_adjusted_pct': 0.0,
            'trigger_event': 'LLM nicht verfuegbar',
            'regime_impact': f'{regime}: LLM-Analyse nicht verfuegbar',
            'status': status,
        })

    return {
        'briefing': briefing,
        'decision_matrix': decision_matrix,
        'asymmetric_payoffs': {},
        'regime_fits': {},
        'vulnerability_texts': {},
        'second_order_effects': [],
        'g7_cross_references': [],
    }


# =====================================================================
# HELPERS
# =====================================================================

def _call_anthropic(system_prompt, user_prompt, max_tokens=4096):
    """Anthropic API Call."""
    url = 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }
    payload = {
        'model': LLM_MODEL,
        'max_tokens': max_tokens,
        'system': system_prompt,
        'messages': [{'role': 'user', 'content': user_prompt}],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    content = data.get('content', [])
    return '\n'.join(c.get('text', '') for c in content if c.get('type') == 'text')


def _parse_json_response(text):
    """Parse JSON aus LLM Response."""
    cleaned = text.strip()
    # Markdown Fences entfernen
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
        # Versuche erstes { bis letztes } zu finden
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None
