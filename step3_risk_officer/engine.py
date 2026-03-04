"""
step_3_risk_officer/engine.py
Haupt-Orchestrierer: Phase 0-11.
Spec: Risk Officer Teil 4 §26
"""

import time
import json

from .utils.helpers import (
    today, now_iso, reset_alert_counter, log_info, log_warning, log_error
)
from .utils.mappings import load_risk_config

from .checks.exposure import (
    check_sector_concentration, check_geography,
    check_single_name, check_asset_class
)
from .checks.interaction import check_regime_conflict
from .checks.temporal import check_event_calendar
from .checks.emergency import evaluate_emergency_triggers

from .alerts.severity import determine_severity, enrich_alert_context
from .alerts.dedup import deduplicate_and_trend
from .alerts.summary import generate_risk_summary

from .portfolio.status import determine_portfolio_status
from .portfolio.fast_path import evaluate_fast_path


def run_risk_officer(inputs, risk_history=None, run_date=None):
    """
    MAIN ENGINE — Vollstaendiger Risk Officer Run.

    Args:
        inputs: dict mit allen verfuegbaren Daten:
            - v16_production: V16 State + Weights (PFLICHT)
            - layer_analysis: Market Analyst Output (PFLICHT)
            - f6_production: F6 State (OPTIONAL, V1 nicht live)
            - signals: Signal Generator Output (OPTIONAL, V1 nicht live)
            - g7_status: G7 Monitor Output (OPTIONAL)
        risk_history: dict mit gestrigem Risk Officer Output (oder None)
        run_date: date Objekt (default: heute)

    Returns: dict — vollstaendiger Risk Officer Output
    """
    start_time = time.time()
    run_date = run_date or today()
    reset_alert_counter()

    log_info(f"Risk Officer run starting for {run_date}")

    # ═════════════════════════════════════════════════════════════
    # PHASE 0: INPUT VALIDATION
    # ═════════════════════════════════════════════════════════════
    config = load_risk_config()

    v16 = inputs.get("v16_production")
    layer = inputs.get("layer_analysis")
    f6 = inputs.get("f6_production")
    signals = inputs.get("signals")
    g7 = inputs.get("g7_status")

    available = {
        "v16_production": v16 is not None,
        "layer_analysis": layer is not None,
        "f6_production": f6 is not None,
        "signals": signals is not None,
        "g7_status": g7 is not None,
        "risk_history": risk_history is not None
    }

    input_errors = []
    if not v16:
        input_errors.append("MANDATORY: v16_production unavailable")
    if not layer:
        input_errors.append("MANDATORY: layer_analysis unavailable")

    log_info(f"  Inputs: {', '.join(k for k, v in available.items() if v)}")
    if input_errors:
        log_warning(f"  Input errors: {input_errors}")

    # V16 Gewichte extrahieren
    v16_weights = {}
    v16_state = {}
    if v16:
        v16_weights = v16.get("weights", {})
        v16_state = v16

    # Fragility State
    fragility_state = "HEALTHY"
    system_regime = {}
    if layer:
        fragility_state = layer.get("fragility_state", {}).get("state", "HEALTHY")
        system_regime = layer.get("system_regime", {})

    # F6 Positionen
    f6_positions = f6.get("active_positions", []) if f6 else []

    # G7 Status
    g7_status = "UNAVAILABLE"
    if g7 and g7.get("available", False):
        g7_status = g7.get("g7_status", "STABLE")

    # ═════════════════════════════════════════════════════════════
    # PHASE 1: EMERGENCY TRIGGERS (IMMER, vor allem anderen)
    # ═════════════════════════════════════════════════════════════
    log_info("  Phase 1: Emergency Triggers")
    emergencies = evaluate_emergency_triggers(
        v16_state=v16_state if v16 else None,
        f6_available=available["f6_production"],
        signal_gen_available=available["signals"],
        v16_weights=v16_weights,
        config=config.get("emergency_triggers", {})
    )
    if emergencies:
        log_warning(f"  EMERGENCY TRIGGERS ACTIVE: {len(emergencies)}")

    # ═════════════════════════════════════════════════════════════
    # PHASE 2: FAST PATH EVALUATION
    # ═════════════════════════════════════════════════════════════
    # Event Calendar vorab pruefen (brauchen wir fuer Fast Path UND Boost)
    event_alert, event_context = check_event_calendar(
        run_date, config.get("event_calendar", {})
    )

    is_fast_path = False
    if not emergencies and risk_history:
        fp_result = evaluate_fast_path(
            v16_state=v16_state,
            fragility_state=fragility_state,
            risk_history=risk_history,
            event_context=event_context,
            config=config
        )
        is_fast_path = fp_result["fast_path"]
        if is_fast_path:
            log_info("  Phase 2: FAST PATH — all conditions met")
        else:
            log_info(f"  Phase 2: FULL PATH — {fp_result['reason']}")
    else:
        reason = "EMERGENCY" if emergencies else "FIRST_RUN"
        log_info(f"  Phase 2: FULL PATH — {reason}")

    execution_path = "FAST_PATH" if is_fast_path else "FULL_PATH"

    # ═════════════════════════════════════════════════════════════
    # PHASE 3: SENSITIVITY (V1: Stub — kein SPY Beta / Korrelation)
    # ═════════════════════════════════════════════════════════════
    sensitivity = {
        "spy_beta": None,
        "spy_beta_interpretation": None,
        "effective_positions": None,
        "effective_positions_source": "UNAVAILABLE",
        "last_correlation_update": None
    }

    # ═════════════════════════════════════════════════════════════
    # PHASE 4-6: CHECKS
    # ═════════════════════════════════════════════════════════════
    all_alerts = []
    checks_run = 0
    checks_skipped = 0

    if is_fast_path:
        # Fast Path: Nur Event-Calendar Alert (wenn vorhanden)
        if event_alert:
            all_alerts.append(event_alert)
            checks_run += 1
        checks_skipped += 6  # Alle anderen uebersprungen
        log_info("  Phases 4-6: Fast Path — checks skipped")
    else:
        # Full Path: Alle Checks
        log_info("  Phase 4: Exposure Checks")

        # EXP_SECTOR_CONCENTRATION
        if v16_weights:
            sector_exp, sector_alerts = check_sector_concentration(
                v16_weights, f6_positions,
                config.get("sector_concentration", {})
            )
            all_alerts.extend(sector_alerts)
            checks_run += 1
        else:
            checks_skipped += 1

        # EXP_GEOGRAPHY
        if v16_weights:
            geo_alert, geo_breakdown = check_geography(
                v16_weights, f6_positions,
                config.get("geography", {})
            )
            if geo_alert:
                all_alerts.append(geo_alert)
            checks_run += 1
        else:
            checks_skipped += 1

        # EXP_SINGLE_NAME
        if v16_weights:
            name_alerts = check_single_name(
                v16_weights, f6_positions,
                config.get("single_name", {})
            )
            all_alerts.extend(name_alerts)
            checks_run += 1
        else:
            checks_skipped += 1

        # EXP_ASSET_CLASS
        if v16_weights:
            asset_classes, ac_alerts = check_asset_class(
                v16_weights, f6_positions,
                config.get("asset_class", {})
            )
            all_alerts.extend(ac_alerts)
            checks_run += 1
        else:
            checks_skipped += 1

        log_info("  Phase 5: Interaction Checks")

        # INT_REGIME_CONFLICT
        if v16 and layer:
            conflict = check_regime_conflict(
                v16_state, system_regime,
                config.get("regime_conflict", {})
            )
            if conflict:
                all_alerts.append(conflict)
            checks_run += 1
        else:
            checks_skipped += 1

        log_info("  Phase 6: Temporal Checks")

        # TMP_EVENT_CALENDAR (bereits oben berechnet)
        if event_alert:
            all_alerts.append(event_alert)
        checks_run += 1

    # ═════════════════════════════════════════════════════════════
    # PHASE 7: SEVERITY FINALIZATION
    # ═════════════════════════════════════════════════════════════
    log_info("  Phase 7: Severity Finalization")

    finalized_alerts = []
    for alert in all_alerts:
        # Severity-Boost
        sev_result = determine_severity(
            alert, fragility_state, event_context, g7_status, config
        )
        alert["severity"] = sev_result["severity"]
        alert["base_severity"] = sev_result["base_severity"]
        alert["boost_applied"] = sev_result["boost_applied"]

        # Kontext-Enrichment
        alert = enrich_alert_context(
            alert, fragility_state, event_context, g7_status, v16_state
        )
        finalized_alerts.append(alert)

    # Emergency-Alerts hinzufuegen
    for emg in emergencies:
        emg["category"] = "EMERGENCY"
        emg["trend"] = "NEW"
        finalized_alerts.append(emg)

    # ═════════════════════════════════════════════════════════════
    # PHASE 8: ALERT MANAGEMENT
    # ═════════════════════════════════════════════════════════════
    log_info("  Phase 8: Alert Management (dedup + trends)")

    yesterday_alerts = []
    if risk_history:
        yesterday_alerts = risk_history.get("alerts", [])

    active_alerts, ongoing_conditions = deduplicate_and_trend(
        finalized_alerts, yesterday_alerts
    )

    # Alert-Volumen pruefen
    active_count = len([
        a for a in active_alerts if a.get("severity") != "RESOLVED"
    ])
    high_vol = config.get("alert_management", {}).get("high_volume_warning", 5)
    if active_count > high_vol:
        log_warning(f"  High alert volume: {active_count} active alerts")

    # ═════════════════════════════════════════════════════════════
    # PHASE 9: PORTFOLIO STATUS
    # ═════════════════════════════════════════════════════════════
    log_info("  Phase 9: Portfolio Status")

    portfolio_status = determine_portfolio_status(
        active_alerts, emergencies, config
    )
    log_info(f"  → {portfolio_status['status']}: {portfolio_status['reason']}")

    # ═════════════════════════════════════════════════════════════
    # PHASE 10: OUTPUT GENERATION
    # ═════════════════════════════════════════════════════════════
    log_info("  Phase 10: Output Generation")

    g7_context = {
        "status": g7_status,
        "last_update": g7.get("last_update") if g7 else None,
        "severity_impact": "NONE"
    }

    risk_summary = generate_risk_summary(
        portfolio_status, active_alerts, ongoing_conditions,
        sensitivity, g7_context, event_context
    )

    execution_time = int((time.time() - start_time) * 1000)

    metadata = {
        "execution_path": execution_path,
        "checks_run": checks_run,
        "checks_skipped": checks_skipped,
        "execution_time_ms": execution_time,
        "config_version": config.get("version", "unknown"),
        "fast_path_taken": is_fast_path,
        "signal_generator_available": available["signals"],
        "g7_available": available["g7_status"],
        "input_errors": input_errors,
        "v16_state": v16_state.get("v16_state"),
        "router_state": "US_DOMESTIC",  # V1: immer
        "fragility_state": fragility_state,
        "alerts_count": len([a for a in active_alerts if a.get("severity") != "RESOLVED"]),
        "ongoing_conditions_count": len(ongoing_conditions)
    }

    output = {
        "date": str(run_date),
        "run_timestamp": now_iso(),
        "execution_path": execution_path,
        "portfolio_status": portfolio_status["status"],
        "portfolio_status_reason": portfolio_status["reason"],
        "alerts": active_alerts,
        "ongoing_conditions": ongoing_conditions,
        "emergency_triggers": {
            "max_drawdown_breach": any(
                e.get("check_id") == "EMG_PORTFOLIO_DRAWDOWN" for e in emergencies
            ),
            "correlation_crisis": False,
            "liquidity_crisis": False,
            "regime_forced": False
        },
        "sensitivity": sensitivity,
        "g7_context": g7_context,
        "risk_summary": risk_summary,
        "metadata": metadata
    }

    # ═════════════════════════════════════════════════════════════
    # PHASE 11: VALIDATION (Definition of Done)
    # ═════════════════════════════════════════════════════════════
    dod = {
        "portfolio_status_set": output["portfolio_status"] is not None,
        "checks_executed": checks_run > 0 or is_fast_path,
        "alerts_deduplicated": True,
        "risk_summary_generated": output["risk_summary"] is not None,
        "metadata_logged": True
    }
    dod_passed = all(dod.values())
    if not dod_passed:
        failed = [k for k, v in dod.items() if not v]
        log_error(f"  DoD INCOMPLETE: {failed}")
        output["portfolio_status"] = "YELLOW"
        output["portfolio_status_reason"] = f"Run incomplete: {', '.join(failed)}"

    output["metadata"]["dod_checklist"] = dod
    output["metadata"]["dod_passed"] = dod_passed

    log_info(
        f"Risk Officer run complete. Status: {output['portfolio_status']}. "
        f"Path: {execution_path}. Time: {execution_time}ms. "
        f"Alerts: {metadata['alerts_count']}."
    )

    return output