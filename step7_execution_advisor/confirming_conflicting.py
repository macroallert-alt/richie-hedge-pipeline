"""
step7_execution_advisor/confirming_conflicting.py
Confirming / Conflicting Engine.

Aggregates all Pro and Contra signals from:
  1. Scoring dimensions (6x)
  2. Pipeline outputs (Risk Officer, CIO, Router)
  3. V16 trade context

ALWAYS shows BOTH sides. Never one-sided.

Source: Trading Desk Spec Teil 4 §15
"""

import logging

logger = logging.getLogger("execution_advisor.confirming_conflicting")


def build_confirming_conflicting(
    scoring_result: dict,
    v16_weights: dict,
    v16_regime: str,
    v16_trades: dict,
    risk_officer: dict,
    cio_final: dict,
    router_output: dict,
    dw_data: dict,
) -> dict:
    """
    Aggregate all Pro and Contra signals.

    Returns:
        {
            "confirming": [{"signal": str, "source": str, "detail": str}, ...],
            "conflicting": [{"signal": str, "source": str, "detail": str}, ...],
            "confirming_count": int,
            "conflicting_count": int,
            "net_assessment": str,
        }
    """
    confirming = []
    conflicting = []

    # ====================================================
    # 1. EVENT RISK
    # ====================================================
    event_dim = scoring_result["dimensions"]["event_risk"]
    event_detail = event_dim.get("detail", {})
    event_score = event_dim.get("score", 0)

    if event_score == 0:
        confirming.append({
            "signal": "Keine HIGH-Impact Events in 48h",
            "source": "EVENT_CALENDAR",
            "detail": f"Nächste 14d: {event_detail.get('events_14d_high_count', 0)} HIGH Events",
        })
    else:
        for ev in event_detail.get("events_48h_high", []):
            conflicting.append({
                "signal": f"{ev} in 48h",
                "source": "EVENT_CALENDAR",
                "detail": "HIGH-Impact Event kann Richtung und Liquidität verändern",
            })
        if event_detail.get("in_convergence_week"):
            cw_events = []
            for cw in event_detail.get("convergence_weeks", []):
                cw_events.extend(cw.get("events", []))
            conflicting.append({
                "signal": "Convergence Week aktiv",
                "source": "EVENT_CALENDAR",
                "detail": f"Mehrere HIGH Events: {', '.join(cw_events[:3])}",
            })

    # ====================================================
    # 2. POSITIONING
    # ====================================================
    pos_detail = scoring_result["dimensions"]["positioning_conflict"].get("detail", {})

    for conflict in pos_detail.get("conflicts", []):
        conflicting.append({
            "signal": f"COT {conflict['asset']} Conflict ({conflict['severity']})",
            "source": "DW_L4_COT",
            "detail": (
                f"{conflict['cot_field']} = {conflict['cot_value']:.1f}% — "
                f"Smart Money gegen {conflict['asset']} "
                f"({conflict['weight']:.0%} im Portfolio)"
            ),
        })

    for confirm in pos_detail.get("confirmations", []):
        confirming.append({
            "signal": f"COT {confirm['asset']} bestätigt",
            "source": "DW_L4_COT",
            "detail": f"{confirm['cot_field']} = {confirm['cot_value']:.1f}% — kein Conflict",
        })

    fund_flows = pos_detail.get("fund_flows")
    if pos_detail.get("flow_conflict"):
        conflicting.append({
            "signal": f"Fund Flows negativ ({fund_flows:+.2f}% AUM)",
            "source": "DW_L4_FLOWS",
            "detail": "Equity-Outflow während V16 Equity-long",
        })
    elif fund_flows is not None and fund_flows > 0:
        confirming.append({
            "signal": f"Fund Flows positiv ({fund_flows:+.2f}% AUM)",
            "source": "DW_L4_FLOWS",
            "detail": "Geld fließt in Aktien — bestätigt V16 Equity-Exposure",
        })

    # ====================================================
    # 3. LIQUIDITY
    # ====================================================
    liq_detail = scoring_result["dimensions"]["liquidity_risk"].get("detail", {})

    for sig_name, sig_data in liq_detail.get("signals", {}).items():
        if sig_data["score"] == 0:
            confirming.append({
                "signal": f"{sig_name}: {sig_data['assessment']}",
                "source": "DW_L5",
                "detail": f"Wert: {sig_data['value']}",
            })
        else:
            conflicting.append({
                "signal": f"{sig_name}: {sig_data['assessment']}",
                "source": "DW_L5",
                "detail": f"Wert: {sig_data['value']} — Liquiditäts-Risiko",
            })

    # ====================================================
    # 4. CROSS-ASSET
    # ====================================================
    cross_detail = scoring_result["dimensions"]["cross_asset_confirmation"].get("detail", {})

    for div in cross_detail.get("divergences", []):
        conflicting.append({
            "signal": div["type"],
            "source": "DW_L7",
            "detail": div["detail"],
        })

    for conf in cross_detail.get("confirmations", []):
        confirming.append({
            "signal": conf["type"],
            "source": "DW_L7",
            "detail": conf["detail"],
        })

    # ====================================================
    # 5. GEX
    # ====================================================
    gex_detail = scoring_result["dimensions"]["gex_regime"].get("detail", {})
    gex_val = gex_detail.get("gex")

    if gex_val is not None:
        if gex_val >= 0:
            confirming.append({
                "signal": f"GEX positiv (${gex_val:.2f}B)",
                "source": "DW_L4_GEX",
                "detail": "Dealer-Hedging dämpft Moves — stabilisierend",
            })
        else:
            conflicting.append({
                "signal": f"GEX negativ (${gex_val:.2f}B) — {gex_detail.get('assessment', '')}",
                "source": "DW_L4_GEX",
                "detail": "Dealer-Hedging verstärkt Moves — erhöhte Volatilität",
            })

    # ====================================================
    # 6. SENTIMENT
    # ====================================================
    sent_detail = scoring_result["dimensions"]["sentiment_extreme"].get("detail", {})

    for sig_name, sig_data in sent_detail.get("signals", {}).items():
        if sig_data.get("warning", False):
            conflicting.append({
                "signal": f"{sig_name}: {sig_data['assessment']}",
                "source": "DW_L2",
                "detail": f"Wert: {sig_data['value']} — Extreme Warnung",
            })
        else:
            confirming.append({
                "signal": f"{sig_name}: {sig_data['assessment']}",
                "source": "DW_L2",
                "detail": f"Wert: {sig_data['value']}",
            })

    # ====================================================
    # 7. PIPELINE OUTPUTS
    # ====================================================

    # Risk Officer
    risk_ampel = risk_officer.get("risk_ampel", "UNKNOWN")
    if risk_ampel == "GREEN":
        confirming.append({
            "signal": "Risk Officer: GREEN",
            "source": "STEP3_RISK_OFFICER",
            "detail": "Keine aktiven Risk-Alerts",
        })
    elif risk_ampel in ("YELLOW", "RED", "BLACK"):
        conflicting.append({
            "signal": f"Risk Officer: {risk_ampel}",
            "source": "STEP3_RISK_OFFICER",
            "detail": f"Aktive Alerts: {risk_officer.get('active_alert_count', 'UNKNOWN')}",
        })

    # CIO Conviction
    cio_conviction = cio_final.get("conviction", "UNKNOWN")
    if cio_conviction == "HIGH":
        confirming.append({
            "signal": "CIO Conviction: HIGH",
            "source": "STEP6_CIO_FINAL",
            "detail": "CIO hat hohes Vertrauen in aktuelle Positionierung",
        })
    elif cio_conviction == "LOW":
        conflicting.append({
            "signal": "CIO Conviction: LOW",
            "source": "STEP6_CIO_FINAL",
            "detail": "CIO hat niedriges Vertrauen — Unsicherheit",
        })

    # Router Proximity
    max_prox = router_output.get("max_proximity", 0)
    if max_prox >= 0.5:
        conflicting.append({
            "signal": f"Router Proximity {max_prox:.0%}",
            "source": "STEP2_SIGNAL_GENERATOR",
            "detail": f"Router nähert sich Trigger ({router_output.get('max_proximity_trigger', 'unknown')})",
        })
    elif max_prox < 0.1:
        confirming.append({
            "signal": "Router inaktiv (Proximity < 10%)",
            "source": "STEP2_SIGNAL_GENERATOR",
            "detail": "Keine internationalen Rotations-Signale",
        })

    # V16 Rebalance scope
    trade_list = v16_trades.get("rebalance_trades", [])
    material_trades = [t for t in trade_list if abs(t.get("delta", 0)) > 0.01]
    if len(material_trades) == 0:
        confirming.append({
            "signal": "V16: Kein materielles Rebalancing",
            "source": "STEP2_V16",
            "detail": "Alle Positionen HOLD — kein Execution-Risiko",
        })
    elif len(material_trades) > 3:
        conflicting.append({
            "signal": f"V16: {len(material_trades)} materielle Trades",
            "source": "STEP2_V16",
            "detail": "Umfangreiches Rebalancing — erhöhtes Slippage-Risiko bei schlechten Bedingungen",
        })

    # ====================================================
    # NET ASSESSMENT
    # ====================================================
    net = len(confirming) - len(conflicting)
    if net >= 3:
        net_assessment = "STRONGLY_CONFIRMING"
    elif net >= 1:
        net_assessment = "MILDLY_CONFIRMING"
    elif net >= -1:
        net_assessment = "BALANCED"
    elif net >= -3:
        net_assessment = "MILDLY_CONFLICTING"
    else:
        net_assessment = "STRONGLY_CONFLICTING"

    return {
        "confirming": confirming,
        "conflicting": conflicting,
        "confirming_count": len(confirming),
        "conflicting_count": len(conflicting),
        "net_assessment": net_assessment,
    }
