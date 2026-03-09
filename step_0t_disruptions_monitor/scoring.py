#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/scoring.py
Score-Berechnung fuer alle Trends.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 1 §4

Berechnet:
  - 5 Basis-Scores: Maturity, Momentum, Acceleration, Relevance, Hype
  - Abgeleitete Metriken: Inflection Score, Velocity, Crowding
  - Phase (EMERGING/ACCELERATING/MATURING/MAINSTREAM/DEAD_ZONE)
  - Watchlist-Status (ACTIVE/WATCH/PARKED/ARCHIVED)
"""


def calculate_scores(deep_dive_results, screening_results, previous_history, config):
    """
    Berechne alle Scores fuer jeden Trend.

    Args:
        deep_dive_results: Dict {cat_id: deep_dive_result} fuer Top-5
        screening_results: Liste aller Screening-Ergebnisse
        previous_history: Liste historischer Snapshots fuer Velocity/Acceleration
        config: Gesamt-Config

    Returns:
        Liste von Trend-Dicts mit allen Scores
    """
    score_weights = config.get('score_weights', {})
    trends = []

    for sr in screening_results:
        cat_id = sr['category_id']
        cat_name = sr['category_name']

        # Deep Dive Daten (nur fuer Top 5 vorhanden)
        dd = deep_dive_results.get(cat_id, {})
        has_deep_dive = bool(dd)

        # --- Basis-Scores ---
        if has_deep_dive:
            maturity = _clamp(dd.get('maturity', 50), 0, 100)
            momentum = _clamp(dd.get('momentum', 50), 0, 100)
            relevance = _clamp(dd.get('relevance', 50), 0, 100)
            hype = _clamp(dd.get('hype', 50), 0, 100)
            multi_signal_count = _clamp(dd.get('multi_signal_count', 0), 0, 5)
        else:
            # Fuer Nicht-Deep-Dive Kategorien: Scores aus Screening ableiten
            maturity = _estimate_maturity_from_screening(sr)
            momentum = _estimate_momentum_from_screening(sr)
            relevance = 40  # Konservativ ohne LLM
            hype = 50  # Neutral
            multi_signal_count = 0

        # --- Acceleration (2. Ableitung) ---
        acceleration = _calculate_acceleration(cat_id, momentum, previous_history)

        # --- Velocity (Maturity-Aenderung pro Quartal) ---
        velocity = _calculate_velocity(cat_id, maturity, previous_history)
        velocity_label = _velocity_to_label(velocity)

        # --- Inflection Score ---
        inflection_score = _calculate_inflection_score(
            maturity=maturity,
            momentum=momentum,
            acceleration=acceleration,
            multi_signal_count=multi_signal_count,
            weights=score_weights.get('inflection_score', {})
        )

        # --- Crowding Indicator ---
        crowding = _calculate_crowding(sr, hype)

        # --- Dead Zone Detection ---
        dead_zone = _is_dead_zone(cat_id, maturity, acceleration, previous_history)

        # --- Trend-Dict zusammenbauen ---
        trend = {
            'id': cat_id,
            'name': cat_name,
            'maturity': maturity,
            'momentum': momentum,
            'acceleration': acceleration,
            'relevance': relevance,
            'hype': hype,
            'inflection_score': inflection_score,
            'velocity': velocity,
            'velocity_label': velocity_label,
            'crowding': crowding,
            'dead_zone': dead_zone,
            'multi_signal_count': multi_signal_count,
            'screening_score': sr.get('screening_score', 0),
            'has_deep_dive': has_deep_dive,
            # Deep Dive Felder (nur wenn vorhanden)
            'headline': dd.get('headline', ''),
            'bull_case': dd.get('bull_case', ''),
            'bear_case': dd.get('bear_case', ''),
            'top_etf': dd.get('top_etf', ''),
            'top_short': dd.get('top_short', ''),
            'historical_analogy': dd.get('historical_analogy', ''),
            'model_risk': dd.get('model_risk', 'NONE'),
            'second_order_effects': dd.get('second_order_effects', []),
            'trigger_events': dd.get('trigger_events', []),
            'source_quality': dd.get('source_quality', {}),
        }
        trends.append(trend)

    return trends


# ===== INFLECTION SCORE =====

def _calculate_inflection_score(maturity, momentum, acceleration, multi_signal_count, weights):
    """
    Spec §4.2:
    inflection = maturity_proximity × 0.30 + momentum × 0.25
                 + acceleration × 0.25 + multi_signal × 0.20

    maturity_proximity: Max wenn Maturity zwischen 25-45.
    """
    w_prox = weights.get('maturity_proximity', 0.30)
    w_mom = weights.get('momentum', 0.25)
    w_acc = weights.get('acceleration', 0.25)
    w_sig = weights.get('multi_signal_count', 0.20)

    # Maturity proximity to inflection zone (25-45)
    if 25 <= maturity <= 45:
        # Peak bei 35
        distance_from_peak = abs(maturity - 35)
        proximity = 100 - (distance_from_peak * 5)  # 0 distance = 100, 10 distance = 50
    elif maturity < 25:
        proximity = max(0, maturity * 2)  # Linear 0-50 fuer 0-25
    else:
        # > 45: fallend
        proximity = max(0, 100 - (maturity - 45) * 2.5)  # 45=100, 85=0

    # Acceleration normalisieren: -50..+50 → 0..100
    acc_normalized = _clamp((acceleration + 50) * 1.0, 0, 100)

    # Multi-Signal: 0-5 → 0-100
    signal_normalized = multi_signal_count * 20

    inflection = (
        proximity * w_prox +
        momentum * w_mom +
        acc_normalized * w_acc +
        signal_normalized * w_sig
    )

    return round(_clamp(inflection, 0, 100))


# ===== VELOCITY =====

def _calculate_velocity(cat_id, current_maturity, history):
    """
    Spec §4.2: velocity = maturity_change_per_quarter (Punkte/Quartal).
    Braucht min 12 Wochen Daten.
    """
    cat_history = _get_category_history(cat_id, history)

    if len(cat_history) < 12:
        # Nicht genug Daten — schaetze aus aktuellem Momentum
        return 0.0

    # Maturity vor 12 Wochen
    old_maturity = cat_history[-12].get('maturity', current_maturity)
    velocity = current_maturity - old_maturity

    return round(velocity, 1)


def _velocity_to_label(velocity):
    """Spec §4.2: LOW/MEDIUM/HIGH."""
    if velocity > 5:
        return 'HIGH'
    elif velocity >= 2:
        return 'MEDIUM'
    else:
        return 'LOW'


# ===== ACCELERATION =====

def _calculate_acceleration(cat_id, current_momentum, history):
    """
    Spec §4.1: Acceleration = Veraenderung des Momentum ueber 4+ Wochen.
    Range: -50 bis +50.
    """
    cat_history = _get_category_history(cat_id, history)

    if len(cat_history) < 4:
        return 0

    # Momentum vor 4 Wochen
    old_momentum = cat_history[-4].get('momentum', current_momentum)
    raw_acceleration = current_momentum - old_momentum

    return round(_clamp(raw_acceleration, -50, 50))


# ===== CROWDING =====

def _calculate_crowding(screening_data, hype):
    """
    Spec §4.2:
    crowding = etf_inflow_velocity × 0.35 + short_interest_collapse × 0.25
               + sentiment_uniformity × 0.25 + media_saturation × 0.15

    Vereinfachte Version mit verfuegbaren Daten:
    - etf_inflow_velocity ≈ ETF Flow Score (aus Screening)
    - sentiment_uniformity ≈ Brave Sentiment Extremitaet
    - media_saturation ≈ Brave Hit Count normalisiert
    - short_interest_collapse: nicht verfuegbar in Stufe 1, nutze Hype als Proxy
    """
    etf_flow = screening_data.get('etf_flow_1w', 50)
    brave_sentiment = screening_data.get('brave_sentiment', 50)
    brave_hits = screening_data.get('brave_hit_count', 0)

    # ETF Inflow Velocity: Hohe Flows = mehr Crowding
    inflow_score = _clamp(etf_flow, 0, 100)

    # Sentiment Uniformity: Je extremer (nahe 0 oder 100), desto uniformer
    sentiment_extremity = abs(brave_sentiment - 50) * 2
    uniformity_score = _clamp(sentiment_extremity, 0, 100)

    # Media Saturation: Viele Hits = viel Berichterstattung
    media_score = _clamp(brave_hits * 5, 0, 100)

    # Short Interest Collapse Proxy: Hype Score
    short_collapse_proxy = hype

    crowding = (
        inflow_score * 0.35 +
        short_collapse_proxy * 0.25 +
        uniformity_score * 0.25 +
        media_score * 0.15
    )

    return round(_clamp(crowding, 0, 100))


# ===== DEAD ZONE DETECTION =====

def _is_dead_zone(cat_id, current_maturity, current_acceleration, history):
    """
    Spec §4.3: Dead Zone wenn Maturity-Aenderung < 3 UND
    avg Acceleration <= 0 fuer 26+ Wochen.
    """
    cat_history = _get_category_history(cat_id, history)

    if len(cat_history) < 26:
        return False

    recent = cat_history[-26:]
    old_maturity = recent[0].get('maturity', current_maturity)
    maturity_change = current_maturity - old_maturity

    accelerations = [e.get('acceleration', 0) for e in recent]
    avg_acceleration = sum(accelerations) / len(accelerations)

    return maturity_change < 3 and avg_acceleration <= 0


# ===== PHASE DETERMINATION =====

def determine_phases(trends):
    """
    Spec §4.3: Phase basierend auf Maturity + Dead Zone.
    EMERGING (0-20), ACCELERATING (20-50), MATURING (50-75), MAINSTREAM (75-100), DEAD_ZONE.
    """
    for t in trends:
        if t.get('dead_zone', False):
            t['phase'] = 'DEAD_ZONE'
        elif t['maturity'] < 20:
            t['phase'] = 'EMERGING'
        elif t['maturity'] < 50:
            t['phase'] = 'ACCELERATING'
        elif t['maturity'] < 75:
            t['phase'] = 'MATURING'
        else:
            t['phase'] = 'MAINSTREAM'

    return trends


# ===== WATCHLIST STATUS =====

def determine_watchlist_status(trends, thresholds):
    """
    Spec §4.4:
    ACTIVE: ACCELERATING/MATURING + Relevance > 60 + kein Dead Zone
    WATCH: EMERGING mit Momentum > 40, ODER ACCELERATING mit Relevance < 40
    PARKED: Momentum < 20, ODER Dead Zone, ODER Hype > 80
    ARCHIVED: MAINSTREAM oder manuell
    """
    for t in trends:
        phase = t.get('phase', 'EMERGING')
        relevance = t.get('relevance', 50)
        momentum = t.get('momentum', 50)
        hype = t.get('hype', 50)
        dead_zone = t.get('dead_zone', False)

        # PARKED Bedingungen zuerst (Ausschluss)
        if dead_zone:
            t['watchlist_status'] = 'PARKED'
        elif momentum < 20:
            t['watchlist_status'] = 'PARKED'
        elif hype > 80:
            t['watchlist_status'] = 'PARKED'
        # ARCHIVED
        elif phase == 'MAINSTREAM':
            t['watchlist_status'] = 'ARCHIVED'
        # ACTIVE
        elif phase in ('ACCELERATING', 'MATURING') and relevance > 60:
            t['watchlist_status'] = 'ACTIVE'
        # WATCH
        elif phase == 'EMERGING' and momentum > 40:
            t['watchlist_status'] = 'WATCH'
        elif phase == 'ACCELERATING' and relevance <= 60:
            t['watchlist_status'] = 'WATCH'
        elif phase == 'MATURING' and relevance <= 60:
            t['watchlist_status'] = 'WATCH'
        else:
            t['watchlist_status'] = 'WATCH'

        # Convergence member (wird spaeter in dependencies.py gesetzt)
        if 'convergence_member' not in t:
            t['convergence_member'] = ''

        # Last state change (wird in Sheet Writer aus History bestimmt)
        if 'last_state_change' not in t:
            t['last_state_change'] = ''

    return trends


# ===== HELPER =====

def _get_category_history(cat_id, history):
    """Extrahiere History-Eintraege fuer eine bestimmte Kategorie."""
    entries = []
    for snapshot in history:
        for trend in snapshot.get('trends', []):
            if trend.get('id') == cat_id:
                entries.append(trend)
                break
    return entries


def _estimate_maturity_from_screening(screening_data):
    """Schaetze Maturity aus Screening-Daten wenn kein Deep Dive vorhanden."""
    # Konservativer Schaetzer: Hohe Trends-Werte + hohe ETF-Flows deuten auf hoehere Maturity
    trends_val = screening_data.get('google_trends_value', 50)
    etf_flow = screening_data.get('etf_flow_1w', 50)
    # Grobe Schaetzung: 30-70 Range
    estimate = 30 + (trends_val / 100 * 20) + (etf_flow / 100 * 20)
    return round(_clamp(estimate, 10, 80))


def _estimate_momentum_from_screening(screening_data):
    """Schaetze Momentum aus Screening-Daten wenn kein Deep Dive vorhanden."""
    brave_hits = screening_data.get('brave_hit_count', 0)
    brave_sent = screening_data.get('brave_sentiment', 50)
    trends_change = screening_data.get('google_trends_1m_change', 0)

    hit_component = min(100, brave_hits * 5) * 0.3
    sent_component = brave_sent * 0.3
    trends_component = _clamp(50 + trends_change, 0, 100) * 0.4

    estimate = hit_component + sent_component + trends_component
    return round(_clamp(estimate, 10, 90))


def _clamp(value, min_val, max_val):
    """Begrenze Wert auf Range."""
    return max(min_val, min(max_val, value))
