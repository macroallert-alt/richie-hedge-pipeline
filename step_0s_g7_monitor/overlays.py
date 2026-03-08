"""
step_0s_g7_monitor/overlays.py
Phase 4: Overlay Computation — STUB (Etappe 1)

Etappe 2 wird hier implementieren:
  1. Feedback Loop Quantifizierung (7 Loops)
  2. Supply Chain Stress Index (SCSI)
  3. De-Dollarization Index (DDI)
  4. Fiscal Dominance Proximity Score (FDP)
  5. Sanctions Intensity Tracker (SIT)
  6. Early Warning Index (EWI, 10 Canary Signals)
  7. Geopolitical Attractiveness Ranking
  8. Liquidity Distribution Map
  9. Correlation Regime Monitor

Fuer Etappe 1: Returns neutrale Defaults damit Phase 5 (Status Determination)
und Phase 10 (Sheet Writing) funktionieren.
"""

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]


def phase4_overlay_computation(scores, validated_data, previous_overlays):
    """
    Phase 4: Compute all 9 overlays.
    STUB in Etappe 1 — returns neutral defaults.
    """
    print("[Phase 4] Overlay Computation (STUB — Etappe 2)...")

    return {
        # Feedback Loops (Etappe 2)
        "feedback_loops": [],

        # SCSI — Supply Chain Stress Index (Etappe 2)
        "scsi": {
            "composite": 0,
            "trend": "STABLE",
            "active_chokepoint_alerts": 0,
            "chokepoints": {
                "suez": 0, "malacca": 0, "hormuz": 0,
                "bab_el_mandeb": 0, "panama": 0, "taiwan_strait": 0,
            },
            "shipping_stress": 0,
            "bdi_zscore": 0,
            "cascade_score": 0,
        },

        # DDI — De-Dollarization Index (Etappe 2)
        "ddi": {
            "composite": 0,
            "trend": "STABLE",
            "acceleration": 0,
            "components": {},
        },

        # FDP — Fiscal Dominance Proximity (Etappe 2)
        "fdp": {
            region: {"composite_proximity": 0, "trend": "STABLE"}
            for region in REGIONS
        },

        # SIT — Sanctions Intensity Tracker (Etappe 2)
        "sanctions": {
            "escalation_trend": "STABLE",
            "highlight": "Sanctions tracking pending (Etappe 2).",
        },

        # EWI — Early Warning Index (Etappe 2)
        "ewi": {
            "active_signals": 0,
            "total_signals": 10,
            "severity": "NONE",
            "active_details": [],
        },

        # Attractiveness Ranking (Etappe 2)
        "attractiveness": [],

        # Liquidity Distribution Map (Etappe 2)
        "liquidity_map": {},

        # Correlation Regime (Etappe 2)
        "correlation_regime": {"current": "NORMAL"},

        # GPR Index (wird in Etappe 2 aus data_collection gezogen)
        "gpr_index_current": 100,
        "gpr_index_trend": "STABLE",
        "gpr_index_zscore": 0,

        # Scenario shift (Etappe 2)
        "max_scenario_shift": 0,
    }
