"""
step4_cio/preprocessor.py
CIO Pre-Processor — 9 Deterministic Phases (no LLM)
Spec: CIO Spec Teil 3

Phase 1: Input Validation + Completeness Manifest
Phase 2: (handled in engine.py — history load)
Phase 3: Temporal Context
Phase 4: Absence Checks
Phase 5: Pattern Matching (Class A)
Phase 6: Briefing Type Determination
Phase 7: Ongoing Conditions Compression
Phase 8: Confidence Markers + System Conviction
Phase 9: Output Assembly
"""

import logging
from datetime import date, datetime, timedelta

import yaml
import os

logger = logging.getLogger("cio_preprocessor")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


# ==========================================================================
# HELPERS: Market Analyst returns nested dicts — extract strings
# ==========================================================================

def _extract_fragility_string(val) -> str:
    """Market Analyst returns fragility_state as dict — extract string."""
    if isinstance(val, dict):
        return val.get("state", val.get("level", "HEALTHY"))
    return val if val else "HEALTHY"


def _extract_regime_string(val) -> str:
    """Market Analyst returns system_regime as dict — extract string."""
    if isinstance(val, dict):
        return val.get("regime", "UNKNOWN")
    return val if val else "UNKNOWN"


# ==========================================================================
# PHASE 1: INPUT VALIDATION + COMPLETENESS MANIFEST (Spec Teil 3 §3.2)
# ==========================================================================

def validate_inputs(inputs: dict, config: dict) -> dict:
    """
    Check each expected input: present? Has data? Date correct?
    Returns completeness manifest.
    """
    today_str = date.today().isoformat()
    completeness = {}

    # All expected input keys
    expected = [
        "v16_production", "risk_alerts", "signals",
        "layer_analysis", "ic_intelligence",
        "f6_production", "beliefs",
        "cio_history", "yesterday_briefing",
    ]

    for key in expected:
        data = inputs.get(key)
        if data is None or data == {}:
            completeness[key] = {
                "status": "MISSING",
                "date": None,
                "notes": "Not provided",
            }
        else:
            # Check date freshness if available
            data_date = data.get("date", None)
            if data_date and data_date != today_str:
                # Stale but present — still usable
                completeness[key] = {
                    "status": "STALE",
                    "date": data_date,
                    "notes": f"Data from {data_date}, not today ({today_str})",
                }
            else:
                completeness[key] = {
                    "status": "COMPLETE",
                    "date": data_date or today_str,
                    "notes": None,
                }

    # IC Intelligence: check source count for DEGRADED
    ic = inputs.get("ic_intelligence", {})
    if ic and completeness.get("ic_intelligence", {}).get("status") != "MISSING":
        es = ic.get("extraction_summary", {})
        successful = es.get("sources_successful", es.get("sources_processed", 0))
        attempted = es.get("sources_attempted", 0)
        if attempted > 0 and successful < attempted * 0.5:
            completeness["ic_intelligence"]["status"] = "DEGRADED"
            completeness["ic_intelligence"]["notes"] = (
                f"{successful}/{attempted} sources successful"
            )

    return completeness


def calculate_data_quality(completeness: dict, config: dict) -> str:
    """
    FULL: All inputs COMPLETE
    DEGRADED: Pflicht OK, but at least one non-pflicht missing/stale
    CRITICAL_GAPS: At least one pflicht input missing/stale/invalid
    """
    pflicht = config.get("inputs", {}).get("pflicht", ["v16_production", "risk_alerts"])

    for p in pflicht:
        status = completeness.get(p, {}).get("status", "MISSING")
        if status not in ("COMPLETE", "STALE"):
            return "CRITICAL_GAPS"

    all_complete = all(
        completeness.get(k, {}).get("status") == "COMPLETE"
        for k in completeness
    )
    return "FULL" if all_complete else "DEGRADED"


def can_run(completeness: dict, config: dict) -> bool:
    """CIO can only run if all Pflicht inputs are present (COMPLETE or STALE)."""
    pflicht = config.get("inputs", {}).get("pflicht", ["v16_production", "risk_alerts"])
    for p in pflicht:
        status = completeness.get(p, {}).get("status", "MISSING")
        if status == "MISSING":
            return False
    return True


# ==========================================================================
# PHASE 3: TEMPORAL CONTEXT (Spec Teil 3 §3.4)
# ==========================================================================

def build_temporal_context(inputs: dict, config: dict, today_date: date) -> dict:
    """Build temporal context: events, CC expiry, rebalance, router proximity."""

    # Load event calendar
    events = _load_event_calendar()
    events_48h = []
    events_7d = []

    for event in events:
        try:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
            days_until = (event_date - today_date).days
            hours_until = days_until * 24  # approximate

            if 0 <= days_until <= 2:
                events_48h.append({
                    **event,
                    "days_until": days_until,
                    "hours_until": hours_until,
                })
            elif 0 <= days_until <= 7:
                events_7d.append({
                    **event,
                    "days_until": days_until,
                })
        except (ValueError, TypeError):
            continue

    # F6 CC Expiry
    f6 = inputs.get("f6_production", {})
    f6_cc_expiry = []
    for pos in f6.get("active_positions", []):
        cc = pos.get("covered_call", {})
        dte = cc.get("dte")
        if dte is not None:
            f6_cc_expiry.append({
                "ticker": pos.get("ticker", "?"),
                "strike": cc.get("strike", 0),
                "dte": dte,
                "critical": dte <= 3,
            })

    # V16 Rebalance timing (approximate — no exact proximity data in V1)
    v16 = inputs.get("v16_production", {})
    v16_rebalance = {
        "next_expected": None,
        "days_until": None,
        "near_miss_yesterday": False,
        "proximity_to_trigger": 0.0,
    }

    # Router proximity from Signal Generator (UNAVAILABLE in V1)
    signals = inputs.get("signals", {})
    router_proximity = {}
    router_status = signals.get("router_status", {})
    for target, data in router_status.get("targets", {}).items():
        router_proximity[target] = {
            "value": data.get("proximity", 0.0),
            "trend": data.get("trend", "STABLE"),
            "days_trending": 0,
        }

    return {
        "events_48h": events_48h,
        "events_7d": events_7d,
        "f6_cc_expiry": f6_cc_expiry,
        "v16_rebalance": v16_rebalance,
        "router_proximity": router_proximity,
        "is_monday": today_date.weekday() == 0,
    }


def _load_event_calendar() -> list:
    """Load EVENT_CALENDAR.yaml."""
    path = os.path.join(CONFIG_DIR, "EVENT_CALENDAR.yaml")
    if not os.path.exists(path):
        logger.warning(f"Event calendar not found: {path}")
        return []
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("events", [])
    except Exception as e:
        logger.error(f"Failed to load event calendar: {e}")
        return []


# ==========================================================================
# PHASE 4: ABSENCE CHECKS (Spec Teil 3 §3.5)
# ==========================================================================

def check_ic_blind_spot(ic_intelligence: dict, layer_analysis: dict,
                        config: dict) -> list:
    """
    IC_BLIND_SPOT: A theme has no IC claims, but market data moves significantly.
    V1 adaptation: We don't have raw market data from DC. We use layer_analysis
    scores as a proxy — if a layer score is extreme (<=2 or >=8) but IC has
    no coverage, that's a potential blind spot.
    """
    blind_spots = []
    if not ic_intelligence:
        return blind_spots

    ic_consensus = ic_intelligence.get("consensus", {})
    min_sources = (
        config.get("absence_checks", {})
        .get("ic_blind_spot", {})
        .get("min_source_count", 1)
    )

    # Theme → Layer mapping for proxy check
    theme_layer_map = {
        "CREDIT": "L3_credit_spreads",
        "VOLATILITY": "L5_volatility_regime",
        "EQUITY_VALUATION": "L4_equity_internals",
        "ENERGY": "L7_intermarket",
        "DOLLAR": "L7_intermarket",
        "CHINA_EM": "L4_cross_border_flows",
    }

    layer_scores = layer_analysis.get("layer_scores", {})

    for theme, layer_key in theme_layer_map.items():
        ic_data = ic_consensus.get(theme, {})
        source_count = ic_data.get("source_count", ic_data.get("sources", 0))

        if source_count <= min_sources:
            # Check if corresponding layer score is extreme
            score = layer_scores.get(layer_key, 5.0)
            if score <= 2.0 or score >= 8.0:
                blind_spots.append({
                    "type": "IC_BLIND_SPOT",
                    "theme": theme,
                    "ic_source_count": source_count,
                    "layer_key": layer_key,
                    "layer_score": score,
                    "interpretation": (
                        f"{theme}: Layer {layer_key} at {score:.1f} (extreme), "
                        f"but only {source_count} IC source(s) cover this theme."
                    ),
                })

    return blind_spots


def check_near_miss(v16_production: dict, temporal_context: dict,
                    config: dict) -> dict | None:
    """NEAR_MISS: V16 was close to rebalance trigger but didn't fire."""
    threshold = (
        config.get("absence_checks", {})
        .get("near_miss", {})
        .get("proximity_threshold", 0.8)
    )
    proximity = temporal_context.get("v16_rebalance", {}).get("proximity_to_trigger", 0.0)

    if proximity > threshold and not v16_production.get("rebalanced_today", False):
        return {
            "type": "NEAR_MISS",
            "proximity": proximity,
            "interpretation": (
                f"V16 was {proximity:.0%} proximity to rebalance trigger. "
                f"Near miss — system close to changing its mind."
            ),
        }
    return None


def check_extended_calm(cio_history: dict, risk_alerts: dict,
                        config: dict) -> dict | None:
    """EXTENDED_CALM: Risk Officer fast path for ≥N consecutive days."""
    min_days = (
        config.get("absence_checks", {})
        .get("extended_calm", {})
        .get("min_consecutive_days", 5)
    )

    fast_path = risk_alerts.get("fast_path", False)
    consecutive = cio_history.get("consecutive_routine_days", 0)
    if fast_path:
        consecutive += 1

    if consecutive >= min_days:
        return {
            "type": "EXTENDED_CALM",
            "consecutive_days": consecutive,
            "interpretation": (
                f"Risk Officer fast path for {consecutive} consecutive days. "
                f"Unusually long calm. Statistically correlated with complacency."
            ),
        }
    return None


# ==========================================================================
# PHASE 5: PATTERN MATCHING — CLASS A (Spec Teil 3 §3.6)
# ==========================================================================

def match_patterns(inputs: dict, temporal_context: dict,
                   absence_flags: list, cio_history: dict,
                   config: dict) -> list:
    """Check all 9 Class A patterns deterministically."""
    active = []
    la = inputs.get("layer_analysis", {})
    ra = inputs.get("risk_alerts", {})
    ic = inputs.get("ic_intelligence", {})
    v16 = inputs.get("v16_production", {})
    f6 = inputs.get("f6_production", {})
    signals = inputs.get("signals", {})
    layer_scores = la.get("layer_scores", {})
    alerts = ra.get("alerts", [])

    # --- Pattern 1: COMPLACENCY_TRAP ---
    vol_score = layer_scores.get("L5_volatility_regime",
                                  layer_scores.get("L6_volatility_regime", 5.0))
    liq_score = layer_scores.get("L1_global_liquidity", 5.0)
    has_warning_plus = any(
        a.get("severity") in ("WARNING", "CRITICAL", "EMERGENCY")
        for a in alerts
    )
    extended_calm_active = any(f["type"] == "EXTENDED_CALM" for f in absence_flags)
    routine_streak = cio_history.get("consecutive_routine_days", 0)

    if (vol_score <= 3.0
            and liq_score >= 6.0
            and ra.get("portfolio_status") == "GREEN"
            and not has_warning_plus
            and (extended_calm_active or routine_streak >= 5)):
        active.append({
            "pattern": "COMPLACENCY_TRAP",
            "trigger_data": {
                "vol_score": vol_score,
                "liq_score": liq_score,
                "risk_ampel": "GREEN",
                "routine_streak": routine_streak,
            },
            "urgency_impact": "WATCH",
        })

    # --- Pattern 2: REGIME_DIVERGENCE ---
    v16_regime = v16.get("regime", "UNKNOWN")
    ma_regime = _extract_regime_string(la.get("system_regime", "UNKNOWN"))
    if _is_regime_incompatible(v16_regime, ma_regime):
        active.append({
            "pattern": "REGIME_DIVERGENCE",
            "trigger_data": {
                "v16_regime": v16_regime,
                "ma_regime": ma_regime,
            },
            "urgency_impact": None,
            "conviction_impact": "OVERRIDE_LOW",
        })

    # --- Pattern 3: CROWDED_CONSENSUS ---
    ic_consensus = ic.get("consensus", {})
    for theme, data in ic_consensus.items():
        score = data.get("consensus_score", data.get("score", 0.0))
        source_count = data.get("source_count", data.get("sources", 0))
        if abs(score) > 6.0 and source_count >= 4:
            # Check MA alignment
            theme_layers = _get_theme_layers(theme)
            aligned = False
            for lk in theme_layers:
                ls = layer_scores.get(lk, 5.0)
                if (score > 0 and ls > 6.0) or (score < 0 and ls < 4.0):
                    aligned = True
                    break
            if aligned:
                active.append({
                    "pattern": "CROWDED_CONSENSUS",
                    "trigger_data": {
                        "theme": theme,
                        "ic_score": score,
                        "source_count": source_count,
                    },
                    "urgency_impact": "WATCH",
                })

    # --- Pattern 4: TEMPORAL_CONVERGENCE ---
    temporal_triggers = 0
    events_48h = temporal_context.get("events_48h", [])
    high_events = [e for e in events_48h if e.get("impact") == "HIGH"]
    if high_events:
        temporal_triggers += 1
    max_router = max(
        (d["value"] for d in temporal_context.get("router_proximity", {}).values()),
        default=0.0,
    )
    if max_router > 0.6:
        temporal_triggers += 1
    cc_critical = [c for c in temporal_context.get("f6_cc_expiry", []) if c.get("dte", 99) <= 5]
    if cc_critical:
        temporal_triggers += 1

    fragility = _extract_fragility_string(la.get("fragility_state", "HEALTHY"))
    has_risk_or_fragility = (
        has_warning_plus or fragility in ("ELEVATED", "EXTREME", "CRISIS")
    )

    if temporal_triggers >= 2 and has_risk_or_fragility:
        active.append({
            "pattern": "TEMPORAL_CONVERGENCE",
            "trigger_data": {
                "temporal_triggers": temporal_triggers,
                "events_48h": len(high_events),
                "max_router_proximity": max_router,
                "cc_critical": len(cc_critical),
                "fragility": fragility,
            },
            "urgency_impact": "REVIEW",
        })

    # --- Pattern 5: FRAGILITY_ESCALATION ---
    sector_conc_alert = any(
        "SECTOR" in a.get("check_id", "").upper() or "CONCENTRATION" in a.get("check_id", "").upper()
        for a in alerts
    )
    ic_bearish_tech = False
    for theme in ("TECH_AI", "EQUITY_VALUATION"):
        td = ic_consensus.get(theme, {})
        ts = td.get("consensus_score", td.get("score", 0))
        if ts < -2.0:
            ic_bearish_tech = True
        # Also check high novelty claims
        for claim in ic.get("high_novelty_claims", []):
            if (claim.get("novelty", claim.get("novelty_score", 0)) >= 5
                    and claim.get("theme", claim.get("primary_topic", "")) == theme
                    and claim.get("signal", 0) < 0):
                ic_bearish_tech = True

    if (fragility in ("ELEVATED", "EXTREME", "CRISIS")
            and sector_conc_alert and ic_bearish_tech):
        active.append({
            "pattern": "FRAGILITY_ESCALATION",
            "trigger_data": {
                "fragility": fragility,
                "sector_concentration_alert": True,
                "ic_bearish_tech": True,
            },
            "urgency_impact": "ACT" if fragility == "EXTREME" else "REVIEW",
        })

    # --- Pattern 6: SILENT_RISK ---
    # Single system flags risk, no corroboration
    # Check: IC divergence only, OR risk alert MONITOR only, OR blind spot only
    ic_divs = ic.get("divergences", [])
    blind_spot_themes = {f["theme"] for f in absence_flags if f["type"] == "IC_BLIND_SPOT"}
    monitor_only = [a for a in alerts if a.get("severity") == "MONITOR"]

    for div in ic_divs:
        topic = div.get("topic", "")
        if (not any(a.get("check_id", "").upper().find(topic[:4]) >= 0 for a in alerts)
                and topic not in blind_spot_themes):
            active.append({
                "pattern": "SILENT_RISK",
                "trigger_data": {
                    "source": "IC_DIVERGENCE",
                    "topic": topic,
                    "severity": div.get("severity", 0),
                },
                "urgency_impact": "WATCH",
            })
            break  # One is enough

    for theme in blind_spot_themes:
        if not any(p["pattern"] == "SILENT_RISK" for p in active):
            active.append({
                "pattern": "SILENT_RISK",
                "trigger_data": {
                    "source": "IC_BLIND_SPOT",
                    "theme": theme,
                },
                "urgency_impact": "WATCH",
            })
            break

    # --- Pattern 7: OPPORTUNITY_WINDOW ---
    v16_is_risk_on = v16_regime in ("RISK_ON", "SELECTIVE")
    ampel_ok = ra.get("portfolio_status") in ("GREEN", "YELLOW")
    frag_ok = fragility in ("HEALTHY", "ELEVATED")

    for target, prox_data in temporal_context.get("router_proximity", {}).items():
        if prox_data["value"] > 0.7 and v16_is_risk_on and ampel_ok and frag_ok:
            # Check IC bullish on corresponding theme
            theme_map = {"EM_BROAD": "CHINA_EM", "COMMODITY_SUPER": "COMMODITIES"}
            theme = theme_map.get(target, "")
            td = ic_consensus.get(theme, {})
            ic_score = td.get("consensus_score", td.get("score", 0))
            if ic_score > 3.0:
                active.append({
                    "pattern": "OPPORTUNITY_WINDOW",
                    "trigger_data": {
                        "target": target,
                        "proximity": prox_data["value"],
                        "ic_theme": theme,
                        "ic_score": ic_score,
                    },
                    "urgency_impact": "WATCH",
                })

    # --- Pattern 8: F6_POSITION_UNDER_PRESSURE ---
    sector_map = _load_sector_theme_mapping(config)
    for pos in f6.get("active_positions", []):
        sector = pos.get("sector", "")
        theme = sector_map.get(sector, "")
        if theme:
            td = ic_consensus.get(theme, {})
            ic_score = td.get("consensus_score", td.get("score", 0))
            if ic_score < -2.0:
                active.append({
                    "pattern": "F6_POSITION_UNDER_PRESSURE",
                    "trigger_data": {
                        "ticker": pos.get("ticker"),
                        "sector": sector,
                        "theme": theme,
                        "ic_score": ic_score,
                    },
                    "urgency_impact": "REVIEW",
                })

    # --- Pattern 9: F6_SIGNAL_IC_CONFIRMATION ---
    for sig in f6.get("signals_today", []):
        sector = sig.get("sector", "")
        theme = sector_map.get(sector, "")
        if theme:
            td = ic_consensus.get(theme, {})
            ic_score = td.get("consensus_score", td.get("score", 0))
            has_sector_alert = any(
                sector.lower() in a.get("check_id", "").lower()
                for a in alerts
            )
            if ic_score > 3.0 and not has_sector_alert:
                active.append({
                    "pattern": "F6_SIGNAL_IC_CONFIRMATION",
                    "trigger_data": {
                        "ticker": sig.get("ticker"),
                        "sector": sector,
                        "theme": theme,
                        "ic_score": ic_score,
                    },
                    "urgency_impact": None,
                })

    return active


def detect_anti_patterns(ic_intelligence: dict, config: dict) -> list:
    """Detect HIGH_NOVELTY_LOW_SIGNAL anti-pattern."""
    anti = []
    ap_config = config.get("anti_patterns", {}).get("HIGH_NOVELTY_LOW_SIGNAL", {})
    novelty_min = ap_config.get("triggers", {}).get("novelty_min", 5)
    signal_max = ap_config.get("triggers", {}).get("bias_adjusted_signal_abs_max", 2.0)

    for claim in ic_intelligence.get("high_novelty_claims", []):
        novelty = claim.get("novelty", claim.get("novelty_score", 0))
        signal = abs(claim.get("bias_adjusted_signal", claim.get("signal", 0)))
        if novelty >= novelty_min and signal < signal_max:
            anti.append({
                "type": "HIGH_NOVELTY_LOW_SIGNAL",
                "source": claim.get("source_id", claim.get("source", "")),
                "claim": claim.get("claim_text", claim.get("claim", ""))[:100],
                "novelty": novelty,
                "signal": signal,
            })

    return anti


# ==========================================================================
# PHASE 6: BRIEFING TYPE DETERMINATION (Spec Teil 2 §2.2, Teil 3 §3.7)
# ==========================================================================

def determine_briefing_type(risk_alerts: dict, active_patterns: list,
                            ic_intelligence: dict, temporal_context: dict,
                            fragility_state: str, cio_history: dict,
                            config: dict) -> str:
    """
    Deterministic. Highest matching level wins.
    EMERGENCY > ACTION > ROUTINE > WATCH (default)
    """
    alerts = risk_alerts.get("alerts", [])
    emg_triggers = risk_alerts.get("emergency_triggers", {})

    # EMERGENCY
    if (risk_alerts.get("portfolio_status") == "BLACK"
            or any(
                (v.get("status") == "ACTIVE" if isinstance(v, dict) else v)
                for v in emg_triggers.values()
            )
            or fragility_state == "CRISIS"
            or risk_alerts.get("dd_protect_active", False)):
        return "EMERGENCY"

    # ACTION
    if (risk_alerts.get("portfolio_status") in ("YELLOW", "RED")
            or fragility_state == "EXTREME"
            or any(
                a.get("days_open", 0) > 1 and a.get("type") == "ACT"
                for a in cio_history.get("open_action_items", [])
            )
            or sum(1 for a in alerts if a.get("severity") == "WARNING") >= 2
            or any(
                a.get("trend") == "ESCALATING"
                and a.get("severity") in ("WARNING", "CRITICAL")
                for a in alerts
            )):
        return "ACTION"

    # ROUTINE — all conditions must be true
    high_novelty_7plus = [
        c for c in ic_intelligence.get("high_novelty_claims", [])
        if c.get("novelty", c.get("novelty_score", 0)) >= 7
    ]
    events_48h = temporal_context.get("events_48h", [])
    max_router = max(
        (d["value"] for d in temporal_context.get("router_proximity", {}).values()),
        default=0.0,
    )
    open_act = [
        a for a in cio_history.get("open_action_items", [])
        if a.get("type") == "ACT"
    ]

    if (risk_alerts.get("portfolio_status") == "GREEN"
            and len(active_patterns) == 0
            and len(high_novelty_7plus) == 0
            and len(events_48h) == 0
            and max_router < 0.6
            and fragility_state == "HEALTHY"
            and len(open_act) == 0):
        return "ROUTINE"

    # WATCH (default)
    return "WATCH"


# ==========================================================================
# PHASE 7: ONGOING CONDITIONS COMPRESSION (Spec Teil 3 §3.8)
# ==========================================================================

def compress_ongoing_conditions(risk_alerts: dict, config: dict) -> dict:
    """
    Alerts > N days STABLE → one-liner format.
    Only NEW, ESCALATING, DEESCALATING get full treatment.
    """
    threshold = config.get("ongoing_compression", {}).get("stable_days_threshold", 3)
    full_treatment = []
    compressed = []

    for alert in risk_alerts.get("alerts", []):
        trend = alert.get("trend", "STABLE")
        days_active = alert.get("days_active", 0)

        if trend in ("NEW", "ESCALATING", "DEESCALATING"):
            full_treatment.append(alert)
        elif trend == "STABLE" and days_active > threshold:
            compressed.append({
                "check_id": alert.get("check_id", ""),
                "severity": alert.get("severity", ""),
                "days_active": days_active,
                "one_liner": (
                    f"{alert.get('check_id', '')} {alert.get('severity', '')} "
                    f"(Tag {days_active}, stabil)"
                ),
            })
        else:
            full_treatment.append(alert)

    return {
        "full_treatment_alerts": full_treatment,
        "compressed_ongoing": compressed,
    }


# ==========================================================================
# PHASE 8: CONFIDENCE MARKERS + SYSTEM CONVICTION (Spec Teil 3 §3.9)
# ==========================================================================

def calculate_confidence_markers(inputs: dict, ic_intelligence: dict,
                                 active_patterns: list,
                                 absence_flags: list) -> list:
    """Deterministic confidence levels for key claims."""
    markers = []

    # Risk assessment confidence
    risk_conf = "HIGH"
    if any(f["type"] == "IC_BLIND_SPOT" for f in absence_flags):
        risk_conf = "MEDIUM"
    if len(ic_intelligence.get("divergences", [])) >= 2:
        risk_conf = "LOW"

    basis_parts = []
    if risk_conf == "HIGH":
        basis_parts.append("No blind spots, few divergences")
    elif risk_conf == "MEDIUM":
        basis_parts.append("IC blind spot detected")
    else:
        basis_parts.append(f"{len(ic_intelligence.get('divergences', []))} IC divergences")

    markers.append({
        "section": "S3_RISK",
        "claim": "Overall risk assessment",
        "confidence": risk_conf,
        "basis": "; ".join(basis_parts),
    })

    # Intelligence confidence
    es = ic_intelligence.get("extraction_summary", {})
    ic_sources = es.get("sources_processed", es.get("sources_successful", 0))
    if ic_sources >= 7:
        intel_conf = "HIGH"
    elif ic_sources >= 4:
        intel_conf = "MEDIUM"
    else:
        intel_conf = "LOW"
    markers.append({
        "section": "S5_INTELLIGENCE",
        "claim": "IC consensus reliability",
        "confidence": intel_conf,
        "basis": f"{ic_sources} sources processed",
    })

    # Pattern confidence
    for pattern in active_patterns:
        trigger_count = len(pattern.get("trigger_data", {}))
        p_conf = "HIGH" if trigger_count >= 3 else "MEDIUM"
        markers.append({
            "section": "S4_PATTERNS",
            "claim": f"Pattern {pattern['pattern']}",
            "confidence": p_conf,
            "basis": f"{trigger_count} trigger conditions met",
        })

    return markers


def calculate_system_conviction(inputs: dict, ic_intelligence: dict,
                                active_patterns: list,
                                confidence_markers: list,
                                config: dict) -> str:
    """HIGH / MODERATE / LOW — measures subsystem coherence."""

    # Hard override: REGIME_DIVERGENCE → LOW
    if any(p["pattern"] == "REGIME_DIVERGENCE" for p in active_patterns):
        return "LOW"

    # Regime compatibility
    v16_regime = inputs.get("v16_production", {}).get("regime", "UNKNOWN")
    ma_regime = _extract_regime_string(inputs.get("layer_analysis", {}).get("system_regime", "UNKNOWN"))
    compat_map = config.get("regime_compatibility", {})
    regime_compatible = ma_regime in compat_map.get(v16_regime, [])

    # If layer_analysis unavailable, assume compatible (degraded but not blocking)
    if not inputs.get("layer_analysis"):
        regime_compatible = True

    # IC divergences
    divergences = ic_intelligence.get("divergences", [])
    high_sev = [d for d in divergences if d.get("severity", 0) >= 6.0]

    # Confidence markers
    low_markers = sum(1 for m in confidence_markers if m["confidence"] == "LOW")

    # IC consensus scatter
    consensus = ic_intelligence.get("consensus", {})
    no_data_themes = sum(
        1 for v in consensus.values()
        if v.get("confidence") in ("LOW", "NO_DATA", 0.4)
    )

    # CROWDED_CONSENSUS active?
    crowded = any(p["pattern"] == "CROWDED_CONSENSUS" for p in active_patterns)

    # LOW
    if (not regime_compatible
            or len(divergences) >= 3
            or len(high_sev) >= 1
            or low_markers >= 4
            or no_data_themes >= 5):
        return "LOW"

    # HIGH
    if (regime_compatible
            and len(divergences) <= 1
            and all(d.get("severity", 0) < 4.0 for d in divergences)
            and low_markers <= 1
            and no_data_themes <= 3
            and not crowded):
        return "HIGH"

    return "MODERATE"


# ==========================================================================
# PHASE 9: OUTPUT ASSEMBLY (Spec Teil 3 §3.10)
# ==========================================================================

def assemble_preprocessor_output(
    completeness: dict, data_quality: str,
    temporal_context: dict, absence_flags: list,
    active_patterns: list, anti_patterns: list,
    briefing_type: str, ongoing_conditions: dict,
    confidence_markers: list, system_conviction: str,
    cio_history: dict, reference_date: str,
    is_monday: bool, inputs: dict, today_str: str,
) -> dict:
    """Pack all pre-processor outputs into structured JSON for LLM call."""
    return {
        "preprocessor_version": "1.0",
        "date": today_str,
        "reference_date": reference_date,
        "is_monday": is_monday,

        "header": {
            "briefing_type": briefing_type,
            "system_conviction": system_conviction,
            "risk_ampel": inputs.get("risk_alerts", {}).get("portfolio_status", "GREEN"),
            "fragility_state": _extract_fragility_string(inputs.get("layer_analysis", {}).get(
                "fragility_state", "HEALTHY"
            )),
            "data_quality": data_quality,
            "v16_regime": inputs.get("v16_production", {}).get("regime", "UNKNOWN"),
        },

        "data_quality_manifest": completeness,
        "temporal_context": temporal_context,
        "absence_flags": absence_flags,

        "patterns": {
            "class_a_active": active_patterns,
            "anti_patterns": anti_patterns,
        },

        "alert_treatment": {
            "full_treatment": ongoing_conditions.get("full_treatment_alerts", []),
            "compressed_ongoing": ongoing_conditions.get("compressed_ongoing", []),
        },

        "confidence_markers": confidence_markers,

        "history": {
            "active_threads": cio_history.get("active_threads", []),
            "resolved_threads_7d": cio_history.get("resolved_threads_last_7d", []),
            "open_action_items": cio_history.get("open_action_items", []),
            "consecutive_routine_days": cio_history.get("consecutive_routine_days", 0),
            "patterns_last_7d": cio_history.get("patterns_last_7d", {}),
        },
    }


# ==========================================================================
# HELPERS
# ==========================================================================

def _is_regime_incompatible(v16: str, ma: str) -> bool:
    """Check if V16 and Market Analyst regimes fundamentally diverge."""
    incompatible = [
        ("RISK_ON", "BROAD_RISK_OFF"),
        ("RISK_ON", "RISK_OFF_FORCED"),
        ("RISK_OFF", "BROAD_RISK_ON"),
    ]
    return (v16, ma) in incompatible or (ma, v16) in incompatible


def _get_theme_layers(theme: str) -> list:
    """Map IC theme to Market Analyst layer keys (Spec Teil 0b §8.2)."""
    mapping = {
        "LIQUIDITY": ["L1_global_liquidity"],
        "FED_POLICY": ["L1_global_liquidity", "L2_us_monetary_policy"],
        "CREDIT": ["L3_credit_spreads"],
        "RECESSION": ["L2_us_monetary_policy", "L5_macro_leading"],
        "INFLATION": ["L2_us_monetary_policy"],
        "EQUITY_VALUATION": ["L4_equity_internals"],
        "CHINA_EM": ["L4_cross_border_flows"],
        "GEOPOLITICS": ["L7_geopolitical_structural"],
        "ENERGY": ["L7_intermarket"],
        "COMMODITIES": ["L7_intermarket"],
        "TECH_AI": ["L4_equity_internals"],
        "CRYPTO": ["L8_sentiment_positioning"],
        "DOLLAR": ["L7_intermarket"],
        "VOLATILITY": ["L5_volatility_regime", "L6_volatility_regime"],
        "POSITIONING": ["L8_sentiment_positioning"],
    }
    return mapping.get(theme, [])


def _load_sector_theme_mapping(config: dict) -> dict:
    """Load sector → theme mapping from config."""
    return config.get("sector_theme_mapping", {})
