#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/exposure_check.py
V16 Portfolio Exposure Check, Blind Spot Alerts, Threat Map.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 2 §8-9

Berechnet:
  - Look-Through Exposure: V16-Gewichte × Exposure Map → Gesamt-Exposure pro Kategorie
  - Blind Spots: ACTIVE Trends mit ≤1% Portfolio-Exposure + handelbare ETFs vorhanden
  - Threat Map: Negativ exponierte V16-Assets pro ACTIVE/WATCH Trend
  - Model Risk Assessment: PARAMETER/STRUCTURAL/PARADIGM pro Trend
"""


def run_exposure_check(trends, v16_weights, exposure_map, thresholds):
    """
    Fuehre vollstaendigen Exposure Check durch.

    Args:
        trends: Liste aller bewerteten Trends
        v16_weights: Aktuelle V16-Gewichte {ticker: weight_pct}
        exposure_map: V16 Asset → Disruptions-Kategorie Exposure {ticker: {cat_id: pct}}
        thresholds: Schwellenwerte aus Config

    Returns:
        Dict mit portfolio_exposure, blind_spots, threats, model_risk_alerts
    """
    blind_spot_threshold = thresholds.get('blind_spot_exposure_pct', 0.01)

    # --- Portfolio Exposure berechnen ---
    portfolio_exposure = _calculate_portfolio_exposure(v16_weights, exposure_map)

    # --- Blind Spots ---
    blind_spots = _detect_blind_spots(trends, portfolio_exposure, blind_spot_threshold)

    # --- Threat Map ---
    threats = _build_threat_map(trends, v16_weights, exposure_map)

    # --- Model Risk Alerts ---
    model_risk_alerts = _collect_model_risk_alerts(trends)

    # --- Exposure pro Trend anreichern ---
    for t in trends:
        cat_id = t['id']
        exp = portfolio_exposure.get(cat_id, {})
        t['portfolio_exposure_pct'] = exp.get('exposure_pct', 0.0)
        t['is_blind_spot'] = any(bs['category'] == cat_id for bs in blind_spots)

    return {
        'portfolio_exposure': portfolio_exposure,
        'blind_spots': blind_spots,
        'threats': threats,
        'model_risk_alerts': model_risk_alerts,
    }


# ===== PORTFOLIO EXPOSURE (LOOK-THROUGH) =====

def _calculate_portfolio_exposure(v16_weights, exposure_map):
    """
    Spec §8.2: Berechne Gesamt-Portfolio-Exposure auf jede Disruptions-Kategorie.

    Fuer jedes V16-Asset: weight × category_exposure_pct = Contribution.
    Aggregiere ueber alle Assets.

    Returns:
        {cat_id: {"exposure_pct": float, "sources": [{"asset", "weight", "category_pct", "contribution"}]}}
    """
    category_exposure = {}

    for asset, weight in v16_weights.items():
        if weight == 0:
            continue

        # Gewicht normalisieren: V16 liefert als % (z.B. 28.8 fuer 28.8%)
        # Exposure Map nutzt Dezimal (z.B. 0.15 fuer 15%)
        weight_decimal = weight / 100.0 if abs(weight) > 1 else weight

        asset_exposures = exposure_map.get(asset, {})

        for category_id, category_pct in asset_exposures.items():
            contribution = weight_decimal * category_pct

            if category_id not in category_exposure:
                category_exposure[category_id] = {
                    'exposure_pct': 0.0,
                    'sources': []
                }

            category_exposure[category_id]['exposure_pct'] += contribution

            if abs(contribution) > 0.001:
                category_exposure[category_id]['sources'].append({
                    'asset': asset,
                    'weight': round(weight_decimal, 4),
                    'category_pct': category_pct,
                    'contribution': round(contribution, 4)
                })

    # Runden
    for cat_id in category_exposure:
        category_exposure[cat_id]['exposure_pct'] = round(
            category_exposure[cat_id]['exposure_pct'], 4
        )

    return category_exposure


# ===== BLIND SPOT DETECTION =====

def _detect_blind_spots(trends, portfolio_exposure, threshold):
    """
    Spec §8.3: Blind Spot wenn:
    1. Trend ist ACTIVE
    2. Portfolio-Exposure ≤ threshold (default 1%)
    3. Handelbare ETFs existieren

    Returns:
        Liste von Blind-Spot-Dicts
    """
    blind_spots = []

    for t in trends:
        if t.get('watchlist_status') != 'ACTIVE':
            continue

        cat_id = t['id']
        exp = portfolio_exposure.get(cat_id, {})
        exposure_pct = exp.get('exposure_pct', 0.0)

        if abs(exposure_pct) <= threshold:
            # Pruefe ob ETFs verfuegbar (top_etf aus Deep Dive)
            top_etf = t.get('top_etf', '')
            recommended_etfs = []
            if top_etf:
                recommended_etfs.append(top_etf)

            # Urgency basierend auf Phase + Inflection
            inflection = t.get('inflection_score', 0)
            phase = t.get('phase', 'EMERGING')
            if inflection > 75 or phase == 'ACCELERATING':
                urgency = 'HIGH'
            elif phase == 'MATURING':
                urgency = 'MEDIUM'
            else:
                urgency = 'LOW'

            blind_spots.append({
                'category': cat_id,
                'name': t['name'],
                'exposure_pct': round(exposure_pct, 4),
                'trend_status': t.get('watchlist_status', 'ACTIVE'),
                'trend_phase': phase,
                'inflection_score': inflection,
                'recommended_etfs': recommended_etfs,
                'urgency': urgency,
                'alert_text': f"KEIN {t['name']}-Exposure im Portfolio ({exposure_pct:.1%}). "
                              f"Trend ist {t.get('watchlist_status', '?')} bei Maturity {t.get('maturity', '?')}."
            })

    return blind_spots


# ===== THREAT MAP =====

def _build_threat_map(trends, v16_weights, exposure_map):
    """
    Spec §9: Fuer jeden ACTIVE/WATCH Trend: Welche V16-Assets sind NEGATIV exponiert?

    Threat Level:
      LOW:      |portfolio_impact| < 2% UND Velocity LOW
      MEDIUM:   |portfolio_impact| 2-5% ODER Velocity MEDIUM
      HIGH:     |portfolio_impact| > 5% ODER Velocity HIGH
      CRITICAL: |portfolio_impact| > 5% UND Velocity HIGH UND Phase ACCELERATING
    """
    threats = []

    active_watch = [t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')]

    for t in active_watch:
        cat_id = t['id']
        velocity_label = t.get('velocity_label', 'LOW')
        phase = t.get('phase', 'EMERGING')
        threatened_assets = []

        for asset, weight in v16_weights.items():
            if weight == 0:
                continue

            weight_decimal = weight / 100.0 if abs(weight) > 1 else weight
            asset_exposures = exposure_map.get(asset, {})
            threat_pct = asset_exposures.get(cat_id, 0)

            if threat_pct < 0:
                # Negatives Exposure = Threat
                portfolio_impact = weight_decimal * threat_pct

                threat_level = _determine_threat_level(
                    portfolio_impact, velocity_label, phase
                )

                threatened_assets.append({
                    'asset': asset,
                    'v16_weight': round(weight_decimal, 4),
                    'threat_pct': threat_pct,
                    'portfolio_impact': round(portfolio_impact, 4),
                    'threat_level': threat_level,
                })

        if threatened_assets:
            # Sortiere nach Impact (schlimmster zuerst)
            threatened_assets.sort(key=lambda x: x['portfolio_impact'])

            # Gesamt-Impact
            total_impact = sum(a['portfolio_impact'] for a in threatened_assets)
            worst_level = _worst_threat_level([a['threat_level'] for a in threatened_assets])

            threats.append({
                'category': cat_id,
                'name': t['name'],
                'threatened_assets': threatened_assets,
                'total_portfolio_impact': round(total_impact, 4),
                'threat_level': worst_level,
                'trend_velocity': velocity_label,
                'trend_phase': phase,
            })

    return threats


def _determine_threat_level(portfolio_impact, velocity, phase):
    """Spec §9.3: Threat Level bestimmen."""
    abs_impact = abs(portfolio_impact)

    if abs_impact > 0.05 and velocity == 'HIGH' and phase == 'ACCELERATING':
        return 'CRITICAL'
    elif abs_impact > 0.05 or velocity == 'HIGH':
        return 'HIGH'
    elif abs_impact >= 0.02 or velocity == 'MEDIUM':
        return 'MEDIUM'
    else:
        return 'LOW'


def _worst_threat_level(levels):
    """Finde das schlimmste Threat Level aus einer Liste."""
    order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    if not levels:
        return 'LOW'
    return max(levels, key=lambda x: order.get(x, 0))


# ===== MODEL RISK =====

def _collect_model_risk_alerts(trends):
    """
    Spec §9.4: Sammle alle Model Risk Alerts (STRUCTURAL und PARADIGM).
    PARAMETER ist informativ — kein Alert.
    """
    alerts = []

    for t in trends:
        model_risk = t.get('model_risk', 'NONE')
        if model_risk in ('STRUCTURAL', 'PARADIGM'):
            alerts.append({
                'category': t['id'],
                'name': t['name'],
                'model_risk': model_risk,
                'phase': t.get('phase', 'EMERGING'),
                'velocity': t.get('velocity_label', 'LOW'),
            })

    return alerts
