"""
step_3_risk_officer/alerts/summary.py
Risk Summary Textblock fuer CIO Briefing.
Spec: Risk Officer Teil 3 §22
"""

from ..utils.helpers import SEVERITY_ORDER


def generate_risk_summary(portfolio_status, alerts, ongoing_conditions,
                           sensitivity, g7_context, event_context):
    """
    Generiert kompakten Risk Summary Textblock.

    FORMAT:
    Zeile 1: Status + Alert-Counts
    Zeile 2: Sensitivity Metriken
    Zeile 3+: Wichtigste Alerts (hoechste Severity zuerst)
    Letzte Zeile: G7 Status + naechstes Event
    """
    lines = []

    # ─── Zeile 1: Status ──────────────────────────────────────────
    active = [a for a in alerts if a.get("severity") not in ("RESOLVED",)]
    resolved = [a for a in alerts if a.get("severity") == "RESOLVED"]

    alert_parts = []
    for sev in ("CRITICAL", "WARNING", "MONITOR"):
        sev_alerts = [a for a in active if a.get("severity") == sev]
        if not sev_alerts:
            continue
        escalating = sum(1 for a in sev_alerts if a.get("trend") == "ESCALATING")
        arrow = " ↑" if escalating > 0 else ""
        alert_parts.append(f"{len(sev_alerts)} {sev}{arrow}")

    status_str = portfolio_status.get("status", "GREEN")
    if alert_parts:
        lines.append(f"PORTFOLIO STATUS: {status_str}. {', '.join(alert_parts)}.")
    else:
        lines.append(f"PORTFOLIO STATUS: {status_str}. No active alerts.")

    # ─── Zeile 2: Sensitivity ─────────────────────────────────────
    beta = sensitivity.get("spy_beta")
    eff_pos = sensitivity.get("effective_positions")
    cached = " (cached)" if sensitivity.get("effective_positions_source") == "CACHED" else ""

    sens_parts = []
    if beta is not None:
        sens_parts.append(f"SPY Beta: {beta}")
    if eff_pos is not None:
        sens_parts.append(f"Effective Positions: {eff_pos}{cached}")

    if sens_parts:
        lines.append(". ".join(sens_parts) + ".")
    else:
        lines.append("Sensitivity: not available (V1).")

    # ─── Zeile 3+: Top Alerts ─────────────────────────────────────
    sorted_alerts = sorted(
        active,
        key=lambda a: SEVERITY_ORDER.get(a.get("severity", "MONITOR"), 1),
        reverse=True
    )

    trend_symbols = {
        "NEW": "●", "ESCALATING": "↑", "STABLE": "→",
        "DEESCALATING": "↓", "RESOLVED": "✓"
    }

    for alert in sorted_alerts[:3]:
        trend = trend_symbols.get(alert.get("trend", ""), "")
        msg = alert.get("message", "")[:150]
        lines.append(f"{alert.get('severity', '?')}{trend}: {msg}")

    if len(sorted_alerts) > 3:
        lines.append(f"(+{len(sorted_alerts) - 3} more alerts, see full report)")

    # ─── Ongoing Conditions ───────────────────────────────────────
    if ongoing_conditions:
        oc_parts = []
        for oc in ongoing_conditions[:3]:
            oc_parts.append(
                f"{oc.get('check_id', '?')} ({oc.get('severity', '?')}, "
                f"day {oc.get('days_active', '?')})"
            )
        lines.append(f"Ongoing: {', '.join(oc_parts)}")

    # ─── Resolved ─────────────────────────────────────────────────
    if resolved:
        res_parts = []
        for r in resolved[:2]:
            res_parts.append(
                f"{r.get('check_id', '?')} (was {r.get('previous_severity', '?')})"
            )
        lines.append(f"Resolved: {', '.join(res_parts)}")

    # ─── Letzte Zeile: G7 + Event ─────────────────────────────────
    context_parts = []
    g7_stat = g7_context.get("status", "UNAVAILABLE") if g7_context else "UNAVAILABLE"
    if g7_stat != "UNAVAILABLE":
        context_parts.append(f"G7: {g7_stat}")

    if event_context and event_context.get("next_event_name"):
        context_parts.append(
            f"Next event: {event_context['next_event_name']} "
            f"in {event_context['next_event_days']}d"
        )

    if context_parts:
        lines.append(" | ".join(context_parts))

    # ─── Emergency Sonderfall ─────────────────────────────────────
    if status_str == "BLACK":
        lines.append("IMMEDIATE ACTION REQUIRED.")

    return "\n".join(lines)