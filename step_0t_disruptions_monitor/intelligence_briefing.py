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

LLM-Calls aufgeteilt in 3 kleinere Calls fuer robustes JSON-Parsing:
  Call 1: briefing + regime_fits + decision_matrix
  Call 2: asymmetric_payoffs + vulnerability_texts
  Call 3: second_order_effects + g7_cross_references

Wird NACH Dependencies und Vulnerability Analysis aufgerufen, VOR Sheet Write.
"""

import os
import json
import re
import requests

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
LLM_MODEL = 'claude-sonnet-4-6'


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
        Dict mit allen neuen Bloecken fuer latest.json.
    """
    print("\n--- INTELLIGENCE BRIEFING ---")

    result = {}

    # ===== 1. REGIME CONTEXT (Python) =====
    regime_context = _build_regime_context(v16_regime)
    result['regime_context'] = regime_context
    print(f"    [REGIME] {regime_context['current_regime']} — Sizing x{regime_context['regime_rules']['sizing_multiplier']}")

    # ===== 2. CONVICTION SCORES PARTIAL (Python) =====
    conviction_partial = _calculate_conviction_scores_partial(trends)

    # ===== 3. CROWDING ALERTS (Python) =====
    crowding_alerts = _detect_crowding_alerts(trends)
    result['crowding_alerts'] = crowding_alerts
    print(f"    [CROWDING] {len(crowding_alerts)} Alerts")

    # ===== 4. REGIME HEATMAP (Python) =====
    regime_heatmap = _build_regime_heatmap(trends, v16_regime)
    result['regime_heatmap'] = regime_heatmap
    print(f"    [HEATMAP] {len(regime_heatmap.get('matrix', {}))} Trends in Matrix")

    # ===== 5. LLM CALLS — 3 separate Calls =====
    ctx = _build_shared_context(trends, exposure_result, contrarian_alerts,
                                causal_chains, convergence_zones,
                                vulnerability_watchlist, v16_weights,
                                regime_context, config, run_date)

    call1_result = {}
    call2_result = {}
    call3_result = {}

    if ANTHROPIC_API_KEY:
        try:
            call1_result = _llm_call_1_briefing(ctx)
            if call1_result.get('briefing'):
                print(f"    [LLM-1] Briefing + Decision Matrix OK")
            else:
                print(f"    [LLM-1] FALLBACK — Parse lieferte kein briefing")
        except Exception as e:
            print(f"    [ERROR] LLM Call 1 fehlgeschlagen: {e}")

        try:
            call2_result = _llm_call_2_payoffs(ctx)
            if call2_result.get('asymmetric_payoffs'):
                print(f"    [LLM-2] Payoffs + Vulnerability OK")
            else:
                print(f"    [LLM-2] FALLBACK — Parse lieferte keine payoffs")
        except Exception as e:
            print(f"    [ERROR] LLM Call 2 fehlgeschlagen: {e}")

        try:
            call3_result = _llm_call_3_effects(ctx)
            if call3_result.get('second_order_effects'):
                print(f"    [LLM-3] Second Order + G7 OK")
            else:
                print(f"    [LLM-3] FALLBACK — Parse lieferte keine effects")
        except Exception as e:
            print(f"    [ERROR] LLM Call 3 fehlgeschlagen: {e}")
    else:
        print("    [SKIP] Kein ANTHROPIC_API_KEY")

    # ===== 6. CONVICTION SCORES FINALISIEREN =====
    regime_fits = call1_result.get('regime_fits', {})
    conviction_scores = _finalize_conviction_scores(conviction_partial, regime_fits, regime_context)
    result['conviction_scores'] = conviction_scores
    print(f"    [CONVICTION] {len(conviction_scores)} Scores berechnet")

    # ===== 7. VULNERABILITY WATCHLIST ANREICHERN =====
    vuln_enriched = _enrich_vulnerability_watchlist(
        vulnerability_watchlist, call2_result.get('vulnerability_texts', {})
    )
    result['vulnerability_watchlist'] = vuln_enriched

    # ===== 8. CONVERGENCE ZONES MIT SECOND ORDER EFFECTS =====
    enriched_zones = _enrich_convergence_zones(
        convergence_zones, call3_result.get('second_order_effects', [])
    )
    result['second_order_effects_zones'] = enriched_zones

    # ===== 9. RESTLICHE BLOECKE =====
    result['briefing'] = call1_result.get('briefing', _fallback_briefing(trends, regime_context, run_date))
    result['decision_matrix'] = call1_result.get('decision_matrix', _fallback_decision_matrix(trends, regime_context))
    result['asymmetric_payoffs'] = call2_result.get('asymmetric_payoffs', {})
    result['g7_cross_references'] = call3_result.get('g7_cross_references', [])

    print(f"    [BRIEFING] Komplett: {len(result)} Bloecke generiert")
    return result


# =====================================================================
# SHARED CONTEXT BUILDER
# =====================================================================

def _build_shared_context(trends, exposure_result, contrarian_alerts,
                          causal_chains, convergence_zones,
                          vulnerability_watchlist, v16_weights,
                          regime_context, config, run_date):
    """Baue gemeinsamen Input-Kontext fuer alle 3 LLM-Calls."""
    active_watch = [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')]
    archived = [t for t in trends if t.get('watchlist_status') == 'ARCHIVED']

    trends_text = ""
    for t in active_watch:
        headline = (t.get('headline', '') or '')[:100]  # Limit headline length
        trends_text += (
            f"\n- {t['id']} {t['name']}: Status={t.get('watchlist_status')}, "
            f"Phase={t.get('phase')}, Maturity={t.get('maturity', 0)}, "
            f"Momentum={t.get('momentum', 0)}, Acceleration={t.get('acceleration', 0)}, "
            f"Inflection={t.get('inflection_score', 0)}, Crowding={t.get('crowding', 0)}, "
            f"TopETF={t.get('top_etf', '')}"
            f"{f', Headline={headline}' if headline else ''}"
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

    regime = regime_context['current_regime']
    regime_mult = regime_context['regime_rules']['sizing_multiplier']

    return {
        'run_date': run_date,
        'regime': regime,
        'regime_mult': regime_mult,
        'trends_text': trends_text,
        'archived_text': ', '.join(f"{t['id']} {t['name']}" for t in archived),
        'bs_text': bs_text or 'Keine',
        'convergence_text': convergence_text or ' Keine',
        'contrarian_text': contrarian_text or ' Keine',
        'chains_text': chains_text or ' Keine',
        'vuln_text': vuln_text or ' Keine',
        'v16_text': v16_text,
        'g7_crossref': config.get('g7_disruptions_crossref', {}),
        'active_watch': [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')],
        'convergence_zones': convergence_zones,
        'vulnerability_watchlist': vulnerability_watchlist,
    }


SYSTEM_PROMPT = """Du bist der Chief Intelligence Officer eines systematischen Macro Hedge Fund (Baldur Creek Capital).
Du schreibst auf Deutsch. Alle Analysen sind regime-konditioniert.
Antworte AUSSCHLIESSLICH mit validem JSON. Kein Markdown, keine Erklaerungen, kein Text vor oder nach dem JSON.
Das JSON muss mit { beginnen und mit } enden. Keine ```json Bloecke."""


# =====================================================================
# LLM CALL 1: Briefing + Regime Fits + Decision Matrix
# =====================================================================

def _llm_call_1_briefing(ctx):
    """Call 1: Briefing Text + Regime Fits + Decision Matrix."""

    active_ids = [t['id'] for t in ctx['active_watch']]
    regime_fits_template = ', '.join(f'"{tid}": {{"score": <0-100>, "reason": "<1 Satz>"}}' for tid in active_ids)

    user_prompt = f"""Generiere den woechentlichen Intelligence Briefing fuer {ctx['run_date']}.

REGIME: {ctx['regime']} (Sizing x{ctx['regime_mult']})
TRENDS:{ctx['trends_text']}
ARCHIVED: {ctx['archived_text']}
BLIND SPOTS: {ctx['bs_text']}
CONVERGENCE:{ctx['convergence_text']}
CONTRARIAN:{ctx['contrarian_text']}
CAUSAL CHAINS:{ctx['chains_text']}

Generiere dieses JSON (EXAKT diese Struktur):

{{
  "briefing": {{
    "date": "{ctx['run_date']}",
    "regime_context": "{ctx['regime']}",
    "headline": "<Knackige Headline, 1 Satz>",
    "body": "<300-500 Woerter Fliesstext, kein Markdown>",
    "sections": [
      {{"title": "Strukturelle Verschiebungen", "content": "<2-3 Saetze>"}},
      {{"title": "Convergence-Analyse", "content": "<2-3 Saetze>"}},
      {{"title": "Regime-Konditionierung", "content": "<2-3 Saetze>"}},
      {{"title": "Blind Spots & Contrarian", "content": "<2-3 Saetze>"}},
      {{"title": "Causal Chain Fruehwarnung", "content": "<2-3 Saetze>"}},
      {{"title": "Model Risk Bewertung", "content": "<2-3 Saetze>"}}
    ],
    "key_changes_this_week": ["<Aenderung 1>", "<Aenderung 2>", "<Aenderung 3>"]
  }},
  "regime_fits": {{
    {regime_fits_template}
  }},
  "decision_matrix": [
    <Ein Objekt pro ACTIVE/WATCH Trend mit: "trend_id", "trend_name", "conviction" (int), "conviction_label" (HIGH/MEDIUM/LOW), "asymmetry" (float), "asymmetry_bull_pct" (int), "asymmetry_bear_pct" (negative int), "timeframe" (6M/12M/18M), "instrument" (ETF), "sizing_hint_pct" (float 1-5), "sizing_regime_adjusted_pct" (= sizing_hint_pct x {ctx['regime_mult']}), "trigger_event" (1 Satz), "regime_impact" (1 Satz), "status" (EXECUTE/WATCH_FOR_TRIGGER/HOLD/AVOID)>
  ]
}}"""

    text = _call_anthropic(SYSTEM_PROMPT, user_prompt, max_tokens=4500)
    parsed = _parse_json_response(text)
    if not parsed:
        print(f"    [WARN] Call 1 Parse fehlgeschlagen. Response-Laenge: {len(text)} chars")
        print(f"    [WARN] Erste 500 Zeichen: {text[:500]}")
        print(f"    [WARN] Letzte 200 Zeichen: {text[-200:]}")
        return {}
    return parsed


# =====================================================================
# LLM CALL 2: Asymmetric Payoffs + Vulnerability Texts
# =====================================================================

def _llm_call_2_payoffs(ctx):
    """Call 2: Asymmetric Payoffs + Vulnerability Texts."""

    user_prompt = f"""Generiere Payoff-Analysen fuer Disruption-Trends.

REGIME: {ctx['regime']}
TRENDS:{ctx['trends_text']}
VULNERABLE ASSETS:{ctx['vuln_text']}
V16 PORTFOLIO: {ctx['v16_text']}

Generiere dieses JSON:

{{
  "asymmetric_payoffs": {{
    "<Trend-ID fuer JEDEN ACTIVE/WATCH Trend>": {{
      "bull_return_pct": <positiv, z.B. 35>,
      "bear_return_pct": <negativ, z.B. -12>,
      "ratio": <bull/abs(bear), z.B. 2.9>,
      "ratio_label": "<EXCEPTIONAL(>4)/EXCELLENT(3-4)/GOOD(2-3)/MARGINAL(1.5-2)/UNFAVORABLE(<1.5)>",
      "instrument": "<ETF Ticker>",
      "bull_scenario": "<1 Satz>",
      "bear_scenario": "<1 Satz>",
      "base_scenario": "<1 Satz>",
      "probability_bull": <int>,
      "probability_bear": <int>,
      "probability_base": <int>
    }}
  }},
  "vulnerability_texts": {{
    "<Asset-Ticker fuer JEDES vulnerable Asset>": {{
      "recommendation": "<1-2 Saetze>",
      "hedge_instrument": "<Konkreter Hedge-Vorschlag>"
    }}
  }}
}}

probability_bull + probability_bear + probability_base = 100.
ratio = abs(bull_return_pct / bear_return_pct). Alle Texte Deutsch."""

    text = _call_anthropic(SYSTEM_PROMPT, user_prompt, max_tokens=2500)
    parsed = _parse_json_response(text)
    if not parsed:
        print(f"    [WARN] Call 2 Parse fehlgeschlagen. Response-Laenge: {len(text)} chars")
        print(f"    [WARN] Erste 500 Zeichen: {text[:500]}")
        print(f"    [WARN] Letzte 200 Zeichen: {text[-200:]}")
        return {}
    return parsed


# =====================================================================
# LLM CALL 3: Second Order Effects + G7 Cross-References
# =====================================================================

def _llm_call_3_effects(ctx):
    """Call 3: Second Order Effects + G7 Cross-References."""

    g7_text = json.dumps(ctx['g7_crossref'], indent=2)
    # Limit g7 cross-ref text to prevent prompt overflow
    if len(g7_text) > 2000:
        g7_text = g7_text[:2000] + '\n  ... [TRUNCATED]'

    convergence_detail = ""
    for z in ctx['convergence_zones']:
        trends_str = ' + '.join(z.get('trend_names', z.get('trends', [])))
        convergence_detail += f"\n- {trends_str}: {z.get('description', '')}"

    user_prompt = f"""Generiere Second Order Effects und G7 Cross-References.

REGIME: {ctx['regime']}
TRENDS:{ctx['trends_text']}
V16 PORTFOLIO: {ctx['v16_text']}

CONVERGENCE ZONES:{convergence_detail if convergence_detail else ' Keine'}

G7-DISRUPTIONS CROSS-REF MAP:
{g7_text}

Generiere dieses JSON:

{{
  "second_order_effects": [
    {{
      "zone_trends": ["<ID1>", "<ID2>"],
      "primary_effects": [
        {{"asset": "<ETF>", "direction": "BULLISH", "mechanism": "<1 Satz>"}}
      ],
      "second_order_effects": [
        {{
          "asset": "<ETF/Asset>",
          "direction": "<BULLISH/BEARISH>",
          "mechanism": "<1 Satz>",
          "confidence": "<HIGH/MEDIUM/LOW>",
          "timeframe": "<6-12M/12-18M/12-24M>"
        }}
      ],
      "net_portfolio_impact": "<1 Satz>"
    }}
  ],
  "g7_cross_references": [
    {{
      "disruption_trend_id": "<D-ID>",
      "disruption_trend_name": "<n>",
      "g7_event_id": "<evt_id>",
      "g7_event_title": "<Titel>",
      "g7_country": "<Kuerzel>",
      "relationship": "<ACCELERATES/AMPLIFIES/ENABLES/DECELERATES/DISRUPTS>",
      "description": "<1 Satz>",
      "impact_on_portfolio": "<1 Satz>"
    }}
  ]
}}

second_order_effects: Mindestens 2 Effects pro Zone. Alle Texte Deutsch."""

    text = _call_anthropic(SYSTEM_PROMPT, user_prompt, max_tokens=2500)
    parsed = _parse_json_response(text)
    if not parsed:
        print(f"    [WARN] Call 3 Parse fehlgeschlagen. Response-Laenge: {len(text)} chars")
        print(f"    [WARN] Erste 500 Zeichen: {text[:500]}")
        print(f"    [WARN] Letzte 200 Zeichen: {text[-200:]}")
        return {}
    return parsed


# =====================================================================
# PYTHON-BERECHNUNGEN (deterministisch)
# =====================================================================

REGIME_RULES = {
    # --- Expansive (Sizing 1.0) ---
    'FULL_EXPANSION':     {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    'STEADY_GROWTH':      {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    'REFLATION':          {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    'EARLY_RECOVERY':     {'sizing_multiplier': 1.0,  'min_conviction': 40, 'min_asymmetry': 1.5, 'trigger_strictness': 'NORMAL'},
    # --- Selektiv (Sizing 0.75) ---
    'FRAGILE_EXPANSION':  {'sizing_multiplier': 0.75, 'min_conviction': 50, 'min_asymmetry': 2.0, 'trigger_strictness': 'MODERATE'},
    'LATE_EXPANSION':     {'sizing_multiplier': 0.75, 'min_conviction': 50, 'min_asymmetry': 2.0, 'trigger_strictness': 'MODERATE'},
    'NEUTRAL':            {'sizing_multiplier': 0.75, 'min_conviction': 50, 'min_asymmetry': 2.0, 'trigger_strictness': 'MODERATE'},
    'SOFT_LANDING':       {'sizing_multiplier': 0.75, 'min_conviction': 50, 'min_asymmetry': 2.0, 'trigger_strictness': 'MODERATE'},
    # --- Defensiv (Sizing 0.5) ---
    'STRESS_ELEVATED':    {'sizing_multiplier': 0.5,  'min_conviction': 70, 'min_asymmetry': 3.0, 'trigger_strictness': 'STRICT'},
    'CONTRACTION':        {'sizing_multiplier': 0.5,  'min_conviction': 70, 'min_asymmetry': 3.0, 'trigger_strictness': 'STRICT'},
    'DEEP_CONTRACTION':   {'sizing_multiplier': 0.5,  'min_conviction': 70, 'min_asymmetry': 3.0, 'trigger_strictness': 'STRICT'},
    # --- Krise (Sizing 0.0) ---
    'FINANCIAL_CRISIS':   {'sizing_multiplier': 0.0,  'min_conviction': 999, 'min_asymmetry': 999, 'trigger_strictness': 'BLOCKED'},
}

REGIME_COLORS = {
    'FULL_EXPANSION':    '#22C55E',
    'STEADY_GROWTH':     '#22C55E',
    'REFLATION':         '#22C55E',
    'EARLY_RECOVERY':    '#22C55E',
    'FRAGILE_EXPANSION': '#EAB308',
    'LATE_EXPANSION':    '#EAB308',
    'NEUTRAL':           '#EAB308',
    'SOFT_LANDING':      '#EAB308',
    'STRESS_ELEVATED':   '#F97316',
    'CONTRACTION':       '#F97316',
    'DEEP_CONTRACTION':  '#F97316',
    'FINANCIAL_CRISIS':  '#EF4444',
}


def _build_regime_context(v16_regime):
    regime = v16_regime if v16_regime in REGIME_RULES else 'NEUTRAL'
    rules = REGIME_RULES[regime]
    color = REGIME_COLORS.get(regime, '#EAB308')

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
        'regime_history_4w': [],
    }


def _calculate_conviction_scores_partial(trends):
    partial = {}
    for t in trends:
        if t.get('watchlist_status') not in ('ACTIVE', 'WATCH'):
            continue
        tid = t['id']
        momentum = t.get('momentum', 0)
        inflection = t.get('inflection_score', 0)
        crowding = t.get('crowding', 0)
        inflection_proximity = min(100, inflection * 1.3)
        momentum_contrib = 0.30 * momentum
        inflection_contrib = 0.25 * inflection_proximity
        crowding_inv_contrib = 0.25 * (100 - crowding)
        partial[tid] = {
            'momentum_contrib': round(momentum_contrib, 1),
            'inflection_contrib': round(inflection_contrib, 1),
            'crowding_inv_contrib': round(crowding_inv_contrib, 1),
            'partial_sum': round(momentum_contrib + inflection_contrib + crowding_inv_contrib, 1),
        }
    return partial


def _finalize_conviction_scores(partial, regime_fits, regime_context):
    scores = {}
    for tid, p in partial.items():
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


def _detect_crowding_alerts(trends):
    alerts = []
    for t in trends:
        if t.get('watchlist_status') not in ('ACTIVE', 'WATCH'):
            continue
        crowding = t.get('crowding', 0)
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
                'momentum': t.get('momentum', 0),
                'momentum_4w_delta': momentum_delta,
                'alert_level': alert_level,
                'description': f"Crowding {crowding} bei "
                               f"{'fallender' if momentum_delta < 0 else 'stagnierender'} "
                               f"Momentum ({momentum_delta:+d} in 4W). "
                               f"{'Klassisches Distribution-Muster.' if alert_level == 'DANGER' else 'Aufmerksamkeit erhoehen.'}",
                'recommendation': '',
                'top_etf': t.get('top_etf', ''),
                'historical_parallel': '',
            })
    return alerts


# Heatmap-Spalten in Zyklusreihenfolge:
# EARLY_RECOVERY → REFLATION → FULL_EXPANSION → STEADY_GROWTH → FRAGILE_EXPANSION →
# LATE_EXPANSION → STRESS_ELEVATED → CONTRACTION → DEEP_CONTRACTION → FINANCIAL_CRISIS →
# SOFT_LANDING → NEUTRAL
REGIME_HEATMAP_ORDER = [
    'EARLY_RECOVERY', 'REFLATION', 'FULL_EXPANSION', 'STEADY_GROWTH',
    'FRAGILE_EXPANSION', 'LATE_EXPANSION', 'STRESS_ELEVATED', 'CONTRACTION',
    'DEEP_CONTRACTION', 'FINANCIAL_CRISIS', 'SOFT_LANDING', 'NEUTRAL',
]

REGIME_HEATMAP_SCORES = {
    #                       ER   REFL  FULL  STDY  FRAG  LATE  STRS  CONT  DEEP  FNCR  SOFT  NEUT
    'D1':  {'name': 'AI',
            'EARLY_RECOVERY': 80, 'REFLATION': 85, 'FULL_EXPANSION': 90, 'STEADY_GROWTH': 85,
            'FRAGILE_EXPANSION': 75, 'LATE_EXPANSION': 65, 'STRESS_ELEVATED': 50, 'CONTRACTION': 40,
            'DEEP_CONTRACTION': 25, 'FINANCIAL_CRISIS': 20, 'SOFT_LANDING': 70, 'NEUTRAL': 65},
    'D2':  {'name': 'Robotics',
            'EARLY_RECOVERY': 70, 'REFLATION': 75, 'FULL_EXPANSION': 80, 'STEADY_GROWTH': 70,
            'FRAGILE_EXPANSION': 55, 'LATE_EXPANSION': 50, 'STRESS_ELEVATED': 40, 'CONTRACTION': 35,
            'DEEP_CONTRACTION': 20, 'FINANCIAL_CRISIS': 15, 'SOFT_LANDING': 55, 'NEUTRAL': 55},
    'D3':  {'name': 'Energy',
            'EARLY_RECOVERY': 60, 'REFLATION': 70, 'FULL_EXPANSION': 65, 'STEADY_GROWTH': 60,
            'FRAGILE_EXPANSION': 55, 'LATE_EXPANSION': 60, 'STRESS_ELEVATED': 55, 'CONTRACTION': 50,
            'DEEP_CONTRACTION': 55, 'FINANCIAL_CRISIS': 70, 'SOFT_LANDING': 55, 'NEUTRAL': 60},
    'D4':  {'name': 'Biotech',
            'EARLY_RECOVERY': 65, 'REFLATION': 60, 'FULL_EXPANSION': 60, 'STEADY_GROWTH': 55,
            'FRAGILE_EXPANSION': 50, 'LATE_EXPANSION': 50, 'STRESS_ELEVATED': 45, 'CONTRACTION': 50,
            'DEEP_CONTRACTION': 45, 'FINANCIAL_CRISIS': 40, 'SOFT_LANDING': 55, 'NEUTRAL': 55},
    'D5':  {'name': 'Space',
            'EARLY_RECOVERY': 65, 'REFLATION': 70, 'FULL_EXPANSION': 70, 'STEADY_GROWTH': 65,
            'FRAGILE_EXPANSION': 60, 'LATE_EXPANSION': 60, 'STRESS_ELEVATED': 55, 'CONTRACTION': 55,
            'DEEP_CONTRACTION': 50, 'FINANCIAL_CRISIS': 60, 'SOFT_LANDING': 60, 'NEUTRAL': 65},
    'D6':  {'name': 'Quantum',
            'EARLY_RECOVERY': 65, 'REFLATION': 70, 'FULL_EXPANSION': 75, 'STEADY_GROWTH': 65,
            'FRAGILE_EXPANSION': 50, 'LATE_EXPANSION': 45, 'STRESS_ELEVATED': 35, 'CONTRACTION': 30,
            'DEEP_CONTRACTION': 20, 'FINANCIAL_CRISIS': 15, 'SOFT_LANDING': 50, 'NEUTRAL': 50},
    'D7':  {'name': 'Fintech',
            'EARLY_RECOVERY': 75, 'REFLATION': 80, 'FULL_EXPANSION': 85, 'STEADY_GROWTH': 75,
            'FRAGILE_EXPANSION': 60, 'LATE_EXPANSION': 55, 'STRESS_ELEVATED': 45, 'CONTRACTION': 45,
            'DEEP_CONTRACTION': 25, 'FINANCIAL_CRISIS': 20, 'SOFT_LANDING': 60, 'NEUTRAL': 60},
    'D8':  {'name': 'Supply Chain',
            'EARLY_RECOVERY': 70, 'REFLATION': 75, 'FULL_EXPANSION': 70, 'STEADY_GROWTH': 75,
            'FRAGILE_EXPANSION': 70, 'LATE_EXPANSION': 70, 'STRESS_ELEVATED': 60, 'CONTRACTION': 60,
            'DEEP_CONTRACTION': 50, 'FINANCIAL_CRISIS': 45, 'SOFT_LANDING': 70, 'NEUTRAL': 75},
    'D9':  {'name': 'Climate',
            'EARLY_RECOVERY': 70, 'REFLATION': 75, 'FULL_EXPANSION': 75, 'STEADY_GROWTH': 70,
            'FRAGILE_EXPANSION': 65, 'LATE_EXPANSION': 60, 'STRESS_ELEVATED': 40, 'CONTRACTION': 40,
            'DEEP_CONTRACTION': 30, 'FINANCIAL_CRISIS': 25, 'SOFT_LANDING': 65, 'NEUTRAL': 70},
    'D10': {'name': 'Cyber',
            'EARLY_RECOVERY': 65, 'REFLATION': 65, 'FULL_EXPANSION': 65, 'STEADY_GROWTH': 70,
            'FRAGILE_EXPANSION': 70, 'LATE_EXPANSION': 70, 'STRESS_ELEVATED': 75, 'CONTRACTION': 75,
            'DEEP_CONTRACTION': 80, 'FINANCIAL_CRISIS': 85, 'SOFT_LANDING': 70, 'NEUTRAL': 70},
    'D11': {'name': 'Demographics',
            'EARLY_RECOVERY': 50, 'REFLATION': 50, 'FULL_EXPANSION': 50, 'STEADY_GROWTH': 55,
            'FRAGILE_EXPANSION': 55, 'LATE_EXPANSION': 55, 'STRESS_ELEVATED': 50, 'CONTRACTION': 55,
            'DEEP_CONTRACTION': 50, 'FINANCIAL_CRISIS': 50, 'SOFT_LANDING': 55, 'NEUTRAL': 55},
    'D12': {'name': 'Regulatory',
            'EARLY_RECOVERY': 45, 'REFLATION': 40, 'FULL_EXPANSION': 40, 'STEADY_GROWTH': 50,
            'FRAGILE_EXPANSION': 55, 'LATE_EXPANSION': 60, 'STRESS_ELEVATED': 65, 'CONTRACTION': 65,
            'DEEP_CONTRACTION': 70, 'FINANCIAL_CRISIS': 80, 'SOFT_LANDING': 55, 'NEUTRAL': 55},
}


def _build_regime_heatmap(trends, v16_regime):
    current_regime = v16_regime if v16_regime in REGIME_HEATMAP_ORDER else 'NEUTRAL'
    matrix = {}
    for tid, data in REGIME_HEATMAP_SCORES.items():
        matrix[tid] = {
            'name': data['name'],
            'scores': {r: data[r] for r in REGIME_HEATMAP_ORDER},
        }
    return {
        'regimes': REGIME_HEATMAP_ORDER,
        'current_regime': current_regime,
        'matrix': matrix,
    }


# =====================================================================
# ENRICHMENT
# =====================================================================

def _enrich_vulnerability_watchlist(vulnerability_watchlist, vulnerability_texts):
    for v in vulnerability_watchlist:
        texts = vulnerability_texts.get(v['asset'], {})
        if texts:
            v['recommendation'] = texts.get('recommendation', v.get('recommendation', ''))
            v['hedge_instrument'] = texts.get('hedge_instrument', v.get('hedge_instrument', ''))
    return vulnerability_watchlist


def _enrich_convergence_zones(convergence_zones, second_order_effects):
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
# FALLBACKS
# =====================================================================

def _fallback_briefing(trends, regime_context, run_date):
    active_watch = [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')]
    return {
        'date': run_date,
        'regime_context': regime_context['current_regime'],
        'headline': f'Woechentlicher Disruptions-Scan — {regime_context["current_regime"]} Regime',
        'body': f'LLM-Briefing nicht verfuegbar. {len(active_watch)} aktive Trends werden ueberwacht.',
        'sections': [],
        'key_changes_this_week': ['LLM-Analyse nicht verfuegbar — nur regelbasierte Daten'],
    }


def _fallback_decision_matrix(trends, regime_context):
    active_watch = [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')]
    mult = regime_context['regime_rules']['sizing_multiplier']
    dm = []
    for t in active_watch:
        status = 'AVOID' if mult == 0 else 'WATCH_FOR_TRIGGER'
        dm.append({
            'trend_id': t['id'], 'trend_name': t['name'],
            'conviction': 0, 'conviction_label': 'LOW',
            'asymmetry': 0.0, 'asymmetry_bull_pct': 0, 'asymmetry_bear_pct': 0,
            'timeframe': '12M', 'instrument': t.get('top_etf', ''),
            'sizing_hint_pct': 0.0, 'sizing_regime_adjusted_pct': 0.0,
            'trigger_event': 'LLM nicht verfuegbar',
            'regime_impact': f'{regime_context["current_regime"]}: LLM-Analyse nicht verfuegbar',
            'status': status,
        })
    return dm


# =====================================================================
# HELPERS
# =====================================================================

def _call_anthropic(system_prompt, user_prompt, max_tokens=3000):
    import anthropic

    # Clean unicode from prompts
    system_prompt = _clean_unicode(system_prompt)
    user_prompt = _clean_unicode(user_prompt)

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        return '\n'.join(b.text for b in response.content if b.type == 'text')
    except anthropic.BadRequestError as e:
        print(f"    [API] 400 Error: {e}")
        print(f"    [API] Prompt length: system={len(system_prompt)}, user={len(user_prompt)}")

        # If prompt is very long, retry with truncated prompt
        if len(user_prompt) > 8000:
            print(f"    [API] Retrying with truncated prompt...")
            user_prompt = user_prompt[:8000] + "\n\n[TRUNCATED — bitte mit verfuegbaren Daten antworten]"
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_prompt}],
            )
            return '\n'.join(b.text for b in response.content if b.type == 'text')
        raise


def _clean_unicode(text):
    """Remove problematic unicode characters that can cause API 400 errors."""
    if not text:
        return text
    # Remove BOM, zero-width chars, and other problematic unicode
    text = text.replace('\ufeff', '')   # BOM
    text = text.replace('\ufffe', '')   # BOM reversed
    text = text.replace('\u200b', '')   # Zero-Width Space
    text = text.replace('\u200c', '')   # Zero-Width Non-Joiner
    text = text.replace('\u200d', '')   # Zero-Width Joiner
    text = text.replace('\u2028', '\n') # Line Separator
    text = text.replace('\u2029', '\n') # Paragraph Separator
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    # Replace curly quotes that can break JSON templates in prompts
    text = text.replace('\u201c', '"')  # Left double quote
    text = text.replace('\u201d', '"')  # Right double quote
    text = text.replace('\u2018', "'")  # Left single quote
    text = text.replace('\u2019', "'")  # Right single quote
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2013', '-')  # En dash
    return text


def _parse_json_response(text):
    """Parse JSON aus LLM Response — robust mit mehreren Fallbacks."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Aggressives Cleaning: BOM, Zero-Width Spaces, Unicode-Whitespace
    cleaned = cleaned.lstrip('\ufeff')  # BOM
    cleaned = cleaned.replace('\u200b', '')  # Zero-Width Space
    cleaned = cleaned.replace('\u200c', '')  # Zero-Width Non-Joiner
    cleaned = cleaned.replace('\u200d', '')  # Zero-Width Joiner
    cleaned = cleaned.replace('\ufffe', '')  # BOM reversed
    cleaned = cleaned.replace('\r\n', '\n')  # Windows line endings
    cleaned = cleaned.replace('\r', '\n')

    # Markdown Fences entfernen
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Versuch 1: Direktes Parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"    [PARSE] Versuch 1 (direkt) fehlgeschlagen: {e}")

    # Versuch 2: Erstes { bis letztes }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as e:
            print(f"    [PARSE] Versuch 2 (bracket-extract) fehlgeschlagen: {e}")

    # Versuch 3: Reparatur abgeschnittener JSON
    fragment = cleaned[start:] if start >= 0 else cleaned
    open_braces = fragment.count('{') - fragment.count('}')
    open_brackets = fragment.count('[') - fragment.count(']')
    if open_braces > 0 or open_brackets > 0:
        print(f"    [PARSE] Versuch 3: {open_braces} offene Braces, {open_brackets} offene Brackets — versuche Reparatur")
        repair = fragment.rstrip()
        if repair.endswith(','):
            repair = repair[:-1]
        elif repair.endswith('"'):
            repair += '"'
        repair += ']' * open_brackets
        repair += '}' * open_braces
        try:
            return json.loads(repair)
        except json.JSONDecodeError as e:
            print(f"    [PARSE] Versuch 3 (repair) fehlgeschlagen: {e}")

    return None
