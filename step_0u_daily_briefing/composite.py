"""
Daily Briefing System — Composite Score Engine
Baldur Creek Capital | Step 0u
Based on: DAILY_BRIEFING_SYSTEM_SPEC_TEIL1.md §3.2

Three timeframe scores (TACTICAL / POSITIONAL / STRUCTURAL), each 0-100.
Regime-conditional indicator weighting.
Velocity + Acceleration layer.
Warning trigger penalties.
Data Integrity meta-score.
"""

import json
import logging
import os
from datetime import date, datetime

from .config import (
    COMPOSITE_HISTORY_FILE,
    DATA_INTEGRITY_GREEN,
    DATA_INTEGRITY_YELLOW,
    HISTORY_DIR,
    VELOCITY_RAPID_DETERIORATION,
    VELOCITY_RAPID_IMPROVEMENT,
    ACCELERATION_STRESS_THRESHOLD,
    WARNING_TRIGGERS,
    get_composite_zone,
    get_regime_weights,
    normalize_indicator,
)

logger = logging.getLogger("composite")

# ---------------------------------------------------------------------------
# Composite History (persistent, for velocity/acceleration)
# ---------------------------------------------------------------------------

def load_composite_history():
    """Load composite_history.json (list of daily entries)."""
    if not os.path.exists(COMPOSITE_HISTORY_FILE):
        return []
    try:
        with open(COMPOSITE_HISTORY_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("history", [])
    except Exception as e:
        logger.warning(f"Could not load composite history: {e}")
        return []


def save_composite_history(history):
    """Save composite_history.json. Keep last 365 entries."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    trimmed = history[-365:]
    with open(COMPOSITE_HISTORY_FILE, "w") as f:
        json.dump({"last_updated": date.today().isoformat(), "history": trimmed}, f, indent=2)
    logger.info(f"Composite history saved: {len(trimmed)} entries")


# ---------------------------------------------------------------------------
# Raw Score Calculation (regime-weighted average of normalized indicators)
# ---------------------------------------------------------------------------

def compute_raw_score(indicator_values, regime):
    """
    Compute a single raw composite score (0-100) from indicator values.

    Args:
        indicator_values: dict of {indicator_key: raw_value} e.g. {"HY_OAS": 342, "BREADTH": 0.62, ...}
        regime: V16 regime string e.g. "LATE_EXPANSION"

    Returns:
        (score, details) where score is 0-100 float and details is a list of per-indicator dicts.
    """
    weights = get_regime_weights(regime)
    weighted_sum = 0.0
    total_weight = 0.0
    details = []

    for key, weight in weights.items():
        raw = indicator_values.get(key)
        if raw is None:
            details.append({
                "indicator": key,
                "raw_value": None,
                "normalized": None,
                "weight": weight,
                "contribution": 0.0,
                "status": "MISSING",
            })
            continue

        normalized = normalize_indicator(key, raw)
        if normalized is None:
            details.append({
                "indicator": key,
                "raw_value": raw,
                "normalized": None,
                "weight": weight,
                "contribution": 0.0,
                "status": "NORM_FAILED",
            })
            continue

        contribution = normalized * weight
        weighted_sum += contribution
        total_weight += weight
        details.append({
            "indicator": key,
            "raw_value": raw,
            "normalized": round(normalized, 1),
            "weight": weight,
            "contribution": round(contribution, 2),
            "status": "OK",
        })

    # Re-normalize if some indicators are missing (proportional scaling)
    if total_weight > 0 and total_weight < 0.99:
        score = weighted_sum / total_weight
    elif total_weight >= 0.99:
        score = weighted_sum
    else:
        score = 50.0  # No data at all → neutral

    return round(max(0.0, min(100.0, score)), 1), details


# ---------------------------------------------------------------------------
# Warning Trigger Penalties (Spec §3.2.5)
# ---------------------------------------------------------------------------

def evaluate_warning_triggers(indicator_values, pipeline_data):
    """
    Evaluate all warning triggers and return list of active triggers with penalties.

    Args:
        indicator_values: dict of raw indicator values
        pipeline_data: dict with pipeline context (risk alerts, IC data, regime conflict, etc.)

    Returns:
        list of {"id": str, "description": str, "penalty": int}
    """
    active = []

    # VIX Term Structure Inversion
    vix = indicator_values.get("VIX")
    vix3m = indicator_values.get("VIX3M")
    if vix is not None and vix3m is not None and vix3m > 0:
        vix_ratio = vix / vix3m
        if vix_ratio > 1.0:
            active.append({
                "id": "VIX_INVERSION",
                "description": f"VIX Term Structure invertiert ({vix_ratio:.2f})",
                "penalty": -15,
            })

    # HY Spread Spike (simplified: use zscore if available, else skip)
    hy_zscore = indicator_values.get("HY_OAS_ZSCORE_90D")
    if hy_zscore is not None and hy_zscore > 2.0:
        active.append({
            "id": "HY_SPIKE",
            "description": f"HY OAS Spike (z-score {hy_zscore:.1f})",
            "penalty": -10,
        })

    # Breadth Collapse
    breadth = indicator_values.get("BREADTH")
    breadth_days = indicator_values.get("BREADTH_BELOW_DAYS", 0)
    if breadth is not None and breadth < 0.5 and breadth_days >= 3:
        active.append({
            "id": "BREADTH_COLLAPSE",
            "description": f"Breadth Collapse ({breadth:.2f}, {breadth_days}d)",
            "penalty": -10,
        })

    # Net Liquidity Drain
    net_liq_7d = indicator_values.get("NET_LIQ_7D_CHANGE")
    if net_liq_7d is not None and net_liq_7d < -50_000_000_000:
        active.append({
            "id": "NET_LIQ_DRAIN",
            "description": f"Net Liq Drain 7d (${net_liq_7d/1e9:.0f}B)",
            "penalty": -8,
        })

    # Cross-Source Temperature Spike (from IC data in pipeline_data)
    ic_temp_count = pipeline_data.get("ic_temp_elevated_count", 0)
    if ic_temp_count >= 3:
        active.append({
            "id": "CROSS_SOURCE_TEMP",
            "description": f"Cross-Source Temp Spike ({ic_temp_count} Quellen)",
            "penalty": -5,
        })

    # Risk Officer Emergency
    risk_emergency = pipeline_data.get("risk_emergency_active", False)
    if risk_emergency:
        active.append({
            "id": "RO_EMERGENCY",
            "description": "Risk Officer EMERGENCY aktiv",
            "penalty": -20,
        })

    # Regime Conflict (V16 vs Market Analyst)
    regime_conflict = pipeline_data.get("regime_conflict", False)
    if regime_conflict:
        active.append({
            "id": "REGIME_CONFLICT",
            "description": "V16 vs Market Analyst Regime-Konflikt",
            "penalty": -5,
        })

    return active


# ---------------------------------------------------------------------------
# Velocity & Acceleration (Spec §3.2.3)
# ---------------------------------------------------------------------------

def compute_velocity_acceleration(current_score, history, timeframe_key):
    """
    Compute velocity and acceleration for a given timeframe.

    Args:
        current_score: today's score (0-100)
        history: list of composite history entries (sorted by date ascending)
        timeframe_key: "tactical", "positional", or "structural"

    Returns:
        (velocity, acceleration) — both float, can be negative
    """
    if not history:
        return 0.0, 0.0

    # Get yesterday's score
    yesterday_score = None
    day_before_score = None

    if len(history) >= 1:
        last = history[-1]
        yesterday_entry = last.get(timeframe_key, {})
        yesterday_score = yesterday_entry.get("score")

    if len(history) >= 2:
        prev = history[-2]
        prev_entry = prev.get(timeframe_key, {})
        day_before_score = prev_entry.get("score")

    # Velocity = today - yesterday
    if yesterday_score is not None:
        velocity = round(current_score - yesterday_score, 1)
    else:
        velocity = 0.0

    # Acceleration = velocity_today - velocity_yesterday
    if yesterday_score is not None and day_before_score is not None:
        yesterday_velocity = yesterday_score - day_before_score
        acceleration = round(velocity - yesterday_velocity, 1)
    else:
        acceleration = 0.0

    return velocity, acceleration


def get_velocity_alerts(score, velocity, acceleration):
    """Return list of velocity/acceleration alert strings."""
    alerts = []
    if velocity < VELOCITY_RAPID_DETERIORATION:
        alerts.append(f"RAPID DETERIORATION (vel {velocity:+.1f}/d)")
    elif velocity > VELOCITY_RAPID_IMPROVEMENT:
        alerts.append(f"RAPID IMPROVEMENT (vel {velocity:+.1f}/d)")

    if acceleration < ACCELERATION_STRESS_THRESHOLD and score < 50:
        alerts.append(f"ACCELERATING STRESS (acc {acceleration:+.1f}, score {score})")

    return alerts


# ---------------------------------------------------------------------------
# Data Integrity Score (Spec §3.2.4)
# ---------------------------------------------------------------------------

def compute_data_integrity(indicator_values, expected_indicators, pipeline_feeds):
    """
    Compute data integrity meta-score.

    Args:
        indicator_values: dict of raw values (None = missing/stale)
        expected_indicators: list of indicator keys expected for current regime
        pipeline_feeds: dict of {"feed_name": bool_available}

    Returns:
        dict with score, total_feeds, active_feeds, stale_feeds list
    """
    # Count indicator feeds
    indicator_available = sum(1 for k in expected_indicators if indicator_values.get(k) is not None)
    indicator_total = len(expected_indicators)

    # Count pipeline feeds
    pipeline_available = sum(1 for v in pipeline_feeds.values() if v)
    pipeline_total = len(pipeline_feeds)

    total = indicator_total + pipeline_total
    active = indicator_available + pipeline_available

    score = round((active / total) * 100, 0) if total > 0 else 0

    # Identify stale/missing
    stale = []
    for k in expected_indicators:
        if indicator_values.get(k) is None:
            stale.append({"name": k, "type": "indicator"})
    for name, avail in pipeline_feeds.items():
        if not avail:
            stale.append({"name": name, "type": "pipeline"})

    return {
        "score": int(score),
        "total_feeds": total,
        "active_feeds": active,
        "stale_feeds": stale,
    }


# ---------------------------------------------------------------------------
# TACTICAL Score (Spec §3.2.1 — 0-24h)
#
# Heavily weights: Breaking News, Event Calendar, VIX, Spreads, Risk alerts
# Uses same indicator base but with tactical adjustments:
#   - Events today → penalty
#   - Breaking news → penalty
#   - Risk CRITICAL → penalty (via warning triggers)
# ---------------------------------------------------------------------------

def compute_tactical_adjustments(pipeline_data):
    """
    Extra adjustments for TACTICAL score beyond indicator-based score.
    Returns total adjustment (can be negative).
    """
    adj = 0

    # HIGH impact event today
    events_today = pipeline_data.get("events_today_high_impact", 0)
    if events_today > 0:
        adj -= 5 * events_today  # -5 per HIGH event

    # Breaking news HIGH
    breaking_high = pipeline_data.get("breaking_news_high_count", 0)
    if breaking_high > 0:
        adj -= 8 * breaking_high  # -8 per HIGH breaking news

    # Risk Officer CRITICAL count
    critical_alerts = pipeline_data.get("risk_critical_count", 0)
    if critical_alerts > 0:
        adj -= 5 * critical_alerts

    return adj


# ---------------------------------------------------------------------------
# POSITIONAL Score (Spec §3.2.1 — 1-2 weeks)
#
# Core indicators + IC Belief State + Regime Duration + Pre-Mortems
# ---------------------------------------------------------------------------

def compute_positional_adjustments(pipeline_data):
    """
    Extra adjustments for POSITIONAL score.
    """
    adj = 0

    # IC Belief State — if net bearish across topics
    ic_net_bearish = pipeline_data.get("ic_net_bearish_score", 0)
    if ic_net_bearish < -3:
        adj -= 5
    elif ic_net_bearish < -6:
        adj -= 10

    # Pre-Mortem HIGH risk count
    high_risk_pms = pipeline_data.get("pre_mortem_high_count", 0)
    if high_risk_pms >= 3:
        adj -= 8
    elif high_risk_pms >= 1:
        adj -= 3

    # Regime duration — very fresh regime = uncertainty
    regime_days = pipeline_data.get("regime_duration_days", 0)
    if regime_days <= 2:
        adj -= 5  # Fresh regime = higher uncertainty

    return adj


# ---------------------------------------------------------------------------
# STRUCTURAL Score (Spec §3.2.1 — 1-3 months)
#
# Yield Curve, Liquidity Pipeline, Regime Trajectory
# Heavier weight on slow-moving indicators (2Y10Y, 3M10Y, Real Yield, Net Liq)
# ---------------------------------------------------------------------------

STRUCTURAL_WEIGHT_OVERRIDE = {
    # For STRUCTURAL, override regime weights to emphasize slow-moving indicators
    "2Y10Y":        0.20,
    "3M10Y":        0.15,
    "REAL_YIELD":   0.15,
    "NET_LIQ":      0.20,
    "HY_OAS":       0.12,
    "BREADTH":      0.08,
    "VIX_TERM":     0.05,
    "MOVE":         0.05,
}


def compute_structural_adjustments(pipeline_data):
    """
    Extra adjustments for STRUCTURAL score.
    """
    adj = 0

    # Pipeline coherence — low coherence = structural uncertainty
    coherence = pipeline_data.get("pipeline_coherence_pct", 100)
    if coherence < 50:
        adj -= 10
    elif coherence < 70:
        adj -= 5

    # G7 EWI (if available)
    g7_ewi = pipeline_data.get("g7_ewi_score")
    if g7_ewi is not None and g7_ewi > 6:
        adj -= 5  # Elevated geopolitical early warning

    return adj


# ---------------------------------------------------------------------------
# MAIN: Compute All Three Scores
# ---------------------------------------------------------------------------

def compute_composite_scores(indicator_values, regime, pipeline_data):
    """
    Compute all three composite scores + velocity/acceleration + warnings + data integrity.

    Args:
        indicator_values: dict of {indicator_key: raw_value}
                         Must include VIX, VIX3M as separate keys for VIX_TERM calc.
                         Also include derived keys: VIX_TERM (ratio), NET_LIQ (computed),
                         CU_AU (ratio), etc.
        regime: V16 regime string
        pipeline_data: dict with pipeline context keys:
            - events_today_high_impact: int
            - breaking_news_high_count: int
            - risk_critical_count: int
            - risk_emergency_active: bool
            - regime_conflict: bool
            - ic_temp_elevated_count: int
            - ic_net_bearish_score: float
            - pre_mortem_high_count: int
            - regime_duration_days: int
            - pipeline_coherence_pct: float
            - g7_ewi_score: float or None

    Returns:
        dict with tactical, positional, structural scores + metadata
    """
    # Load history for velocity/acceleration
    history = load_composite_history()

    # Warning triggers (applied to all three scores)
    warnings = evaluate_warning_triggers(indicator_values, pipeline_data)
    total_penalty = sum(w["penalty"] for w in warnings)

    # --- TACTICAL ---
    tact_raw, tact_details = compute_raw_score(indicator_values, regime)
    tact_adj = compute_tactical_adjustments(pipeline_data)
    tact_score = max(0, min(100, round(tact_raw + total_penalty + tact_adj, 1)))
    tact_vel, tact_acc = compute_velocity_acceleration(tact_score, history, "tactical")
    tact_alerts = get_velocity_alerts(tact_score, tact_vel, tact_acc)

    # --- POSITIONAL ---
    pos_raw, pos_details = compute_raw_score(indicator_values, regime)
    pos_adj = compute_positional_adjustments(pipeline_data)
    pos_score = max(0, min(100, round(pos_raw + total_penalty + pos_adj, 1)))
    pos_vel, pos_acc = compute_velocity_acceleration(pos_score, history, "positional")
    pos_alerts = get_velocity_alerts(pos_score, pos_vel, pos_acc)

    # --- STRUCTURAL (uses different weights) ---
    struct_raw, struct_details = compute_raw_score(indicator_values, "STRUCTURAL_OVERRIDE")
    struct_adj = compute_structural_adjustments(pipeline_data)
    struct_score = max(0, min(100, round(struct_raw + total_penalty + struct_adj, 1)))
    struct_vel, struct_acc = compute_velocity_acceleration(struct_score, history, "structural")
    struct_alerts = get_velocity_alerts(struct_score, struct_vel, struct_acc)

    # --- Data Integrity ---
    regime_weights = get_regime_weights(regime)
    expected = list(regime_weights.keys())
    pipeline_feeds = {
        "V16": pipeline_data.get("v16_available", True),
        "IC": pipeline_data.get("ic_available", True),
        "CIO": pipeline_data.get("cio_available", True),
        "Risk": pipeline_data.get("risk_available", True),
        "Execution": pipeline_data.get("execution_available", True),
    }
    data_integrity = compute_data_integrity(indicator_values, expected, pipeline_feeds)

    # Build result
    result = {
        "date": date.today().isoformat(),
        "regime": regime,
        "tactical": {
            "score": tact_score,
            "zone": get_composite_zone(tact_score),
            "velocity": tact_vel,
            "acceleration": tact_acc,
            "raw_score": tact_raw,
            "adjustments": tact_adj,
            "penalty": total_penalty,
            "alerts": tact_alerts,
            "details": tact_details,
        },
        "positional": {
            "score": pos_score,
            "zone": get_composite_zone(pos_score),
            "velocity": pos_vel,
            "acceleration": pos_acc,
            "raw_score": pos_raw,
            "adjustments": pos_adj,
            "penalty": total_penalty,
            "alerts": pos_alerts,
            "details": pos_details,
        },
        "structural": {
            "score": struct_score,
            "zone": get_composite_zone(struct_score),
            "velocity": struct_vel,
            "acceleration": struct_acc,
            "raw_score": struct_raw,
            "adjustments": struct_adj,
            "penalty": total_penalty,
            "alerts": struct_alerts,
            "details": struct_details,
        },
        "warning_triggers": warnings,
        "data_integrity": data_integrity,
    }

    # Update history
    history_entry = {
        "date": date.today().isoformat(),
        "regime": regime,
        "tactical": {"score": tact_score, "velocity": tact_vel, "acceleration": tact_acc},
        "positional": {"score": pos_score, "velocity": pos_vel, "acceleration": pos_acc},
        "structural": {"score": struct_score, "velocity": struct_vel, "acceleration": struct_acc},
        "data_integrity": data_integrity["score"],
    }

    # Avoid duplicate for same date
    if history and history[-1].get("date") == date.today().isoformat():
        history[-1] = history_entry
    else:
        history.append(history_entry)

    save_composite_history(history)

    return result
