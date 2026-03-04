"""
step_3_risk_officer/checks/emergency.py
Emergency Triggers — nur diese koennen EMERGENCY (BLACK) ausloesen.
Spec: Risk Officer Teil 3 §18
"""

from ..utils.helpers import make_alert
from ..utils.mappings import is_international_asset


# ═══════════════════════════════════════════════════════════════════
# EMERGENCY TRIGGER EVALUATION
# Spec Teil 3 §18.2
# Laufen IMMER, auch im Fast Path. VOR allen anderen Checks.
# ═══════════════════════════════════════════════════════════════════

def evaluate_emergency_triggers(v16_state, f6_available, signal_gen_available,
                                 v16_weights, config):
    """
    Prueft alle Emergency Triggers.

    Args:
        v16_state: V16 Production State (dict oder None)
        f6_available: bool — ob F6 Production Tab vorhanden
        signal_gen_available: bool — ob SIGNALS Tab vorhanden
        v16_weights: dict {asset: weight} fuer International Exposure Check
        config: emergency_triggers Section aus RISK_CONFIG

    Returns: list of emergency alert dicts
    """
    emergencies = []

    # ─── EMG_PORTFOLIO_DRAWDOWN ───────────────────────────────────
    # Portfolio DD > -15% (V16 DD-Protect sollte bei -12% aktiv sein)
    emg_dd = config.get("emg_portfolio_drawdown", {})
    if emg_dd.get("enabled", True) and v16_state:
        dd = v16_state.get("current_drawdown_from_peak", 0)
        threshold = emg_dd.get("threshold", -0.15)
        dd_active = v16_state.get("dd_protect_active", False)

        if dd <= threshold:  # DD ist negativ
            emergencies.append(make_alert(
                severity="EMERGENCY",
                message=(
                    f"Portfolio drawdown {dd:.1%} exceeds maximum tolerance "
                    f"({threshold:.0%}). V16 DD-Protect status: "
                    f"{'ACTIVE' if dd_active else 'INACTIVE'}. "
                    f"IMMEDIATE risk reduction recommended."
                ),
                check_id="EMG_PORTFOLIO_DRAWDOWN",
                affected_systems=["V16"],
                trade_class="SYSTEM",
                current_value=round(dd, 4),
                threshold=threshold,
                recommendation=(
                    "Verify V16 DD-Protect is executing correctly. "
                    "If DD-Protect active but insufficient: manual risk reduction via Agent R. "
                    "If DD-Protect NOT active: investigate V16 system failure."
                )
            ))

    # ─── EMG_PRODUCTION_SYSTEMS_DOWN ──────────────────────────────
    # 2 von 3 Systemen nicht verfuegbar
    emg_sys = config.get("emg_production_systems_down", {})
    if emg_sys.get("enabled", True):
        unavailable = []
        if v16_state is None:
            unavailable.append("V16_PRODUCTION")
        if not f6_available:
            unavailable.append("F6_PRODUCTION")
        if not signal_gen_available:
            unavailable.append("SIGNAL_GENERATOR")

        threshold = emg_sys.get("threshold", 2)
        if len(unavailable) >= threshold:
            emergencies.append(make_alert(
                severity="EMERGENCY",
                message=(
                    f"Multiple production systems unavailable: "
                    f"{', '.join(unavailable)}. "
                    f"Portfolio state cannot be fully verified."
                ),
                check_id="EMG_PRODUCTION_SYSTEMS_DOWN",
                affected_systems=unavailable,
                trade_class="SYSTEM",
                current_value=len(unavailable),
                threshold=threshold,
                recommendation=(
                    "Investigate system failures immediately. "
                    "Consider reducing risk positions until systems are restored. "
                    "Do NOT rely on stale data for trading decisions."
                )
            ))

    # ─── EMG_V16_DDPROTECT_FAILURE ────────────────────────────────
    # DD unter Trigger-Level aber DD-Protect nicht aktiv
    emg_ddp = config.get("emg_v16_ddprotect_failure", {})
    if emg_ddp.get("enabled", True) and v16_state:
        dd = v16_state.get("current_drawdown_from_peak", 0)
        trigger_level = v16_state.get("dd_protect_trigger_level", -0.12)
        dd_active = v16_state.get("dd_protect_active", False)

        if dd <= trigger_level and not dd_active:
            emergencies.append(make_alert(
                severity="EMERGENCY",
                message=(
                    f"V16 drawdown at {dd:.1%} has breached DD-Protect trigger "
                    f"({trigger_level:.0%}), but DD-Protect is NOT ACTIVE. "
                    f"Possible V16 system malfunction."
                ),
                check_id="EMG_V16_DDPROTECT_FAILURE",
                affected_systems=["V16"],
                trade_class="SYSTEM",
                current_value=round(dd, 4),
                threshold=trigger_level,
                recommendation=(
                    "CRITICAL: Investigate V16 system immediately. "
                    "Manual DD-Protect execution via Agent R recommended "
                    "until V16 is verified."
                )
            ))

    # ─── EMG_HARD_CAP_EXTREME_VIOLATION ───────────────────────────
    # International > 35% (25% Hard Cap + 10pp Buffer)
    emg_cap = config.get("emg_hard_cap_extreme", {})
    if emg_cap.get("enabled", True) and v16_weights:
        international_exposure = sum(
            w for asset, w in v16_weights.items()
            if is_international_asset(asset)
        )
        buffer = emg_cap.get("buffer_pp", 0.10)
        extreme_threshold = 0.25 + buffer  # Hard Cap + Buffer

        if international_exposure > extreme_threshold:
            emergencies.append(make_alert(
                severity="EMERGENCY",
                message=(
                    f"International exposure {international_exposure:.1%} massively "
                    f"exceeds Hard Cap (25%) by "
                    f"{(international_exposure - 0.25):.1%}pp. "
                    f"This should not occur during normal operations."
                ),
                check_id="EMG_HARD_CAP_EXTREME_VIOLATION",
                trade_class="SYSTEM",
                current_value=round(international_exposure, 4),
                threshold=extreme_threshold,
                recommendation=(
                    "Investigate source of excessive international exposure. "
                    "Immediate position review via Agent R."
                )
            ))

    return emergencies