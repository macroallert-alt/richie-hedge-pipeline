"""
step7_execution_advisor/scoring.py
Execution Score — 6 Dimensions + Veto System.

100% deterministic. No LLM. All thresholds from config.

Dimensions:
  1. EVENT_RISK         — Events in 48h/14d, Convergence
  2. POSITIONING_CONFLICT — COT/Fund Flows vs V16 direction
  3. LIQUIDITY_RISK     — Market liquidity, Funding stress
  4. CROSS_ASSET_CONFIRMATION — Do other assets confirm V16?
  5. GEX_REGIME         — Gamma Exposure regime
  6. SENTIMENT_EXTREME  — Sentiment at extremes

Source: Trading Desk Spec Teil 3 §7-14
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger("execution_advisor.scoring")


# =============================================================================
# HELPER
# =============================================================================

def _safe_float(value) -> float | None:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# =============================================================================
# ASSET → COT MAPPING (Spec Teil 3 §9.3)
# =============================================================================

ASSET_COT_MAPPING = {
    # Equity ETFs → COT_SP500_COMM_NET
    "SPY": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "QQQ": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "XLK": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "RSP": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "IWM": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "XLP": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "XLU": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "XLE": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "XLF": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},
    "HYG": {"cot_field": "COT_SP500_COMM_NET", "direction": "long"},

    # Gold/Commodities → COT_GOLD_COMM_NET
    "GLD": {"cot_field": "COT_GOLD_COMM_NET", "direction": "long"},
    "SLV": {"cot_field": "COT_GOLD_COMM_NET", "direction": "long"},
    "GDX": {"cot_field": "COT_GOLD_COMM_NET", "direction": "long"},
    "DBC": {"cot_field": "COT_GOLD_COMM_NET", "direction": "long"},

    # EM → no direct COT, use Fund Flows
    "VWO": {"cot_field": None, "flow_field": "FUND_FLOWS_EQUITY"},
    "INDA": {"cot_field": None, "flow_field": "FUND_FLOWS_EQUITY"},
    "FXI": {"cot_field": None, "flow_field": "FUND_FLOWS_EQUITY"},
    "KWEB": {"cot_field": None, "flow_field": "FUND_FLOWS_EQUITY"},

    # Bonds → COT_TREASURY_COMM_NET
    "TLT": {"cot_field": "COT_TREASURY_COMM_NET", "direction": "long"},
    "SHY": {"cot_field": "COT_TREASURY_COMM_NET", "direction": "long"},
}

# Equity assets for total equity weight calculation
EQUITY_ASSETS = {"SPY", "QQQ", "XLK", "RSP", "IWM", "XLP", "XLU", "XLE", "XLF"}

# Gold/Silver assets
GOLD_ASSETS = {"GLD", "SLV", "GDX"}

# Commodity-broad assets
COMMODITY_ASSETS = {"DBC", "GLD", "SLV", "XLE", "GDX"}


# =============================================================================
# DIMENSION 1: EVENT RISK (Spec Teil 3 §8)
# =============================================================================

def score_event_risk(events: list[dict], today: date) -> tuple[int, dict]:
    """
    Score Event Risk 0-3.

    0 = No relevant events
    1 = MEDIUM in 48h or multiple HIGH in 14d
    2 = HIGH event in 48h
    3 = Convergence + imminent HIGH event
    """
    events_48h_high = []
    events_48h_medium = []
    events_14d_high = []

    for event in events:
        try:
            event_date = date.fromisoformat(event["date"])
        except (ValueError, KeyError):
            continue

        days_until = (event_date - today).days

        if days_until < 0:
            continue

        if days_until <= 2:  # 48h window
            if event.get("impact") == "HIGH":
                events_48h_high.append(event)
            else:
                events_48h_medium.append(event)
        elif days_until <= 14:
            if event.get("impact") == "HIGH":
                events_14d_high.append(event)

    # Convergence check
    convergence_weeks = _detect_convergence(events, today)
    in_convergence_week = any(
        date.fromisoformat(cw["start"]) <= today <= date.fromisoformat(cw["end"])
        for cw in convergence_weeks
    )

    # Scoring
    if len(events_48h_high) >= 2 or (
        len(events_48h_high) >= 1 and in_convergence_week
    ):
        score = 3
    elif len(events_48h_high) >= 1:
        score = 2
    elif len(events_48h_medium) >= 1 or len(events_14d_high) >= 2:
        score = 1
    else:
        score = 0

    # Label for dashboard
    if events_48h_high:
        label = events_48h_high[0].get("event", "HIGH Event in 48h")
    elif events_48h_medium:
        label = events_48h_medium[0].get("event", "MEDIUM Event in 48h")
    elif events_14d_high:
        label = f"{len(events_14d_high)} HIGH events in 14d"
    else:
        label = "No events"

    detail = {
        "events_48h_high": [e.get("event", "") for e in events_48h_high],
        "events_48h_medium": [e.get("event", "") for e in events_48h_medium],
        "events_14d_high_count": len(events_14d_high),
        "convergence_weeks": convergence_weeks,
        "in_convergence_week": in_convergence_week,
    }

    return score, {"score": score, "max": 3, "label": label, "detail": detail}


def _detect_convergence(events: list[dict], today: date) -> list[dict]:
    """Detect weeks with 2+ HIGH-impact events within 5 days."""
    high_events = []
    for e in events:
        try:
            event_date = date.fromisoformat(e["date"])
        except (ValueError, KeyError):
            continue
        days_until = (event_date - today).days
        if 0 <= days_until <= 21 and e.get("impact") == "HIGH":
            high_events.append(e)

    convergence_weeks = []
    for i, e1 in enumerate(high_events):
        d1 = date.fromisoformat(e1["date"])
        cluster = [e1]
        for e2 in high_events[i + 1:]:
            d2 = date.fromisoformat(e2["date"])
            if abs((d2 - d1).days) <= 5:
                cluster.append(e2)
        if len(cluster) >= 2:
            dates = [date.fromisoformat(e["date"]) for e in cluster]
            cw = {
                "start": (min(dates) - timedelta(days=1)).isoformat(),
                "end": (max(dates) + timedelta(days=1)).isoformat(),
                "events": [e.get("event", "") for e in cluster],
                "risk_level": "HIGH" if len(cluster) >= 3 else "ELEVATED",
            }
            if not any(
                existing["start"] == cw["start"] and existing["end"] == cw["end"]
                for existing in convergence_weeks
            ):
                convergence_weeks.append(cw)

    return convergence_weeks


# =============================================================================
# DIMENSION 2: POSITIONING CONFLICT (Spec Teil 3 §9)
# =============================================================================

def score_positioning_conflict(v16_weights: dict,
                               dw_data: dict) -> tuple[int, dict]:
    """
    Score Positioning Conflict 0-3.

    Checks: Are Commercials/Smart Money positioned against V16's top positions?
    """
    conflicts = []
    confirmations = []

    # Only check top-5 V16 positions
    top_positions = sorted(
        v16_weights.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    for asset, weight in top_positions:
        mapping = ASSET_COT_MAPPING.get(asset)
        if mapping is None or mapping.get("cot_field") is None:
            continue

        cot_value = _safe_float(dw_data.get(mapping["cot_field"]))
        if cot_value is None:
            continue

        asset_direction = mapping["direction"]
        is_conflict = False
        severity = "NONE"

        if asset_direction == "long":
            if cot_value < -40:
                is_conflict = True
                severity = "EXTREME"
            elif cot_value < -25:
                is_conflict = True
                severity = "ELEVATED"
        elif asset_direction == "short":
            if cot_value > 40:
                is_conflict = True
                severity = "EXTREME"
            elif cot_value > 25:
                is_conflict = True
                severity = "ELEVATED"

        entry = {
            "asset": asset,
            "weight": weight,
            "cot_field": mapping["cot_field"],
            "cot_value": cot_value,
            "direction": asset_direction,
            "severity": severity,
        }

        if is_conflict:
            conflicts.append(entry)
        else:
            confirmations.append(entry)

    # Fund Flows check
    fund_flows = _safe_float(dw_data.get("FUND_FLOWS_EQUITY"))
    flow_conflict = False
    if fund_flows is not None:
        total_equity_weight = sum(
            v16_weights.get(a, 0) for a in EQUITY_ASSETS
        )
        if total_equity_weight > 0.3 and fund_flows < -0.5:
            flow_conflict = True

    # Scoring
    extreme_conflicts = [c for c in conflicts if c["severity"] == "EXTREME"]
    elevated_conflicts = [c for c in conflicts if c["severity"] == "ELEVATED"]

    if len(extreme_conflicts) >= 2 or (
        len(extreme_conflicts) >= 1 and flow_conflict
    ):
        score = 3
    elif len(extreme_conflicts) >= 1:
        score = 2
    elif len(elevated_conflicts) >= 1 or flow_conflict:
        score = 1
    else:
        score = 0

    # Label
    if extreme_conflicts:
        c = extreme_conflicts[0]
        label = f"COT {c['asset']} {c['cot_value']:.1f}%"
    elif elevated_conflicts:
        c = elevated_conflicts[0]
        label = f"COT {c['asset']} {c['cot_value']:.1f}%"
    elif flow_conflict:
        label = f"Fund Flows {fund_flows:+.2f}%"
    else:
        label = "No conflicts"

    detail = {
        "conflicts": conflicts,
        "confirmations": confirmations,
        "fund_flows": fund_flows,
        "flow_conflict": flow_conflict,
        "extreme_count": len(extreme_conflicts),
        "elevated_count": len(elevated_conflicts),
    }

    return score, {"score": score, "max": 3, "label": label, "detail": detail}


# =============================================================================
# DIMENSION 3: LIQUIDITY RISK (Spec Teil 3 §10)
# =============================================================================

def score_liquidity_risk(dw_data: dict) -> tuple[int, dict]:
    """Score Liquidity Risk 0-3."""
    signals = {}

    # Amihud Illiquidity
    amihud = _safe_float(dw_data.get("LIQUIDITY_AMIHUD"))
    if amihud is not None:
        if amihud > 80:
            signals["amihud"] = {"value": amihud, "assessment": "ILLIQUID", "score": 2}
        elif amihud > 60:
            signals["amihud"] = {"value": amihud, "assessment": "MODERATE", "score": 1}
        else:
            signals["amihud"] = {"value": amihud, "assessment": "LIQUID", "score": 0}

    # SOFR-FFR Spread
    sofr_ffr = _safe_float(dw_data.get("SOFR_FFR_SPREAD"))
    if sofr_ffr is not None:
        if sofr_ffr > 15:
            signals["sofr_ffr"] = {"value": sofr_ffr, "assessment": "STRESSED", "score": 2}
        elif sofr_ffr > 8:
            signals["sofr_ffr"] = {"value": sofr_ffr, "assessment": "ELEVATED", "score": 1}
        else:
            signals["sofr_ffr"] = {"value": sofr_ffr, "assessment": "NORMAL", "score": 0}

    # Financial Stress Index
    fin_stress = _safe_float(dw_data.get("FIN_STRESS_INDEX"))
    if fin_stress is not None:
        if fin_stress > 1.0:
            signals["fin_stress"] = {"value": fin_stress, "assessment": "HIGH_STRESS", "score": 2}
        elif fin_stress > 0:
            signals["fin_stress"] = {"value": fin_stress, "assessment": "MILD_STRESS", "score": 1}
        else:
            signals["fin_stress"] = {"value": fin_stress, "assessment": "RELAXED", "score": 0}

    # VIX Term Structure
    vix_ts = _safe_float(dw_data.get("VIX_TERM_STRUCTURE"))
    if vix_ts is not None:
        if vix_ts < 0.90:
            signals["vix_term"] = {"value": vix_ts, "assessment": "BACKWARDATION", "score": 2}
        elif vix_ts < 0.95:
            signals["vix_term"] = {"value": vix_ts, "assessment": "FLAT", "score": 1}
        else:
            signals["vix_term"] = {"value": vix_ts, "assessment": "CONTANGO", "score": 0}

    # Avg Pairwise Correlation
    avg_corr = _safe_float(dw_data.get("AVG_PAIRWISE_CORR"))
    if avg_corr is not None:
        if avg_corr > 0.6:
            signals["avg_corr"] = {"value": avg_corr, "assessment": "HIGH_CORRELATION", "score": 1}
        else:
            signals["avg_corr"] = {"value": avg_corr, "assessment": "NORMAL", "score": 0}

    # Composite score
    individual_scores = [s["score"] for s in signals.values()]

    if not individual_scores:
        return 0, {"score": 0, "max": 3, "label": "No L5 data",
                    "detail": {"signals": signals, "note": "No L5 data available"}}

    max_score = max(individual_scores)
    stressed_count = sum(1 for s in individual_scores if s >= 1)

    if max_score >= 2 and stressed_count >= 2:
        score = 3
    elif max_score >= 2:
        score = 2
    elif stressed_count >= 2:
        score = 1
    elif stressed_count >= 1:
        score = 1
    else:
        score = 0

    score = min(score, 3)

    # Label
    stressed_signals = [k for k, v in signals.items() if v["score"] >= 1]
    if stressed_signals:
        label = ", ".join(f"{s}: {signals[s]['assessment']}" for s in stressed_signals[:2])
    else:
        label = "All clear"

    return score, {"score": score, "max": 3, "label": label, "detail": {"signals": signals}}


# =============================================================================
# DIMENSION 4: CROSS-ASSET CONFIRMATION (Spec Teil 3 §11)
# =============================================================================

def score_cross_asset_confirmation(v16_regime: str, v16_weights: dict,
                                   dw_data: dict) -> tuple[int, dict]:
    """
    Score Cross-Asset Confirmation 0-3.

    0 = all confirmed
    1 = mixed
    2 = one strong divergence
    3 = multiple strong divergences

    INVERTED: high score = bad (divergence).
    """
    divergences = []
    confirmations = []

    gold_ret = _safe_float(dw_data.get("GOLD_RETURN_20D"))
    dxy_ret = _safe_float(dw_data.get("DXY_RETURN_20D"))

    # Gold vs DXY divergence
    if gold_ret is not None and dxy_ret is not None:
        if gold_ret > 1.0 and dxy_ret > 1.0:
            divergences.append({
                "type": "GOLD_DXY_DIVERGENCE",
                "detail": f"Gold +{gold_ret:.1f}% UND DXY +{dxy_ret:.1f}% — beide steigen, unsustainable",
                "severity": "HIGH",
            })
        else:
            confirmations.append({
                "type": "GOLD_DXY_NORMAL",
                "detail": f"Gold {gold_ret:+.1f}%, DXY {dxy_ret:+.1f}% — normale Relation",
            })

    # V16 Gold exposure + DXY rising
    gold_weight = sum(v16_weights.get(a, 0) for a in GOLD_ASSETS)
    if gold_weight > 0.10 and dxy_ret is not None and dxy_ret > 2.0:
        divergences.append({
            "type": "GOLD_POSITION_VS_DXY",
            "detail": f"V16 hat {gold_weight:.0%} Gold/Silber aber DXY steigt +{dxy_ret:.1f}%",
            "severity": "ELEVATED",
        })

    # Commodity exposure + Copper falling
    commodity_weight = sum(v16_weights.get(a, 0) for a in COMMODITY_ASSETS)
    copper_trend = dw_data.get("COPPER_SMA50_TREND")
    if commodity_weight > 0.15 and copper_trend == "Falling":
        divergences.append({
            "type": "COMMODITY_VS_COPPER",
            "detail": f"V16 hat {commodity_weight:.0%} Commodity-Exposure aber Copper SMA50 Falling",
            "severity": "ELEVATED",
        })
    elif commodity_weight > 0.15 and copper_trend == "Rising":
        confirmations.append({
            "type": "COMMODITY_COPPER_CONFIRMED",
            "detail": f"V16 {commodity_weight:.0%} Commodity + Copper Rising — bestätigt",
        })

    # SPY Trend vs V16 Equity exposure
    spy_trend = dw_data.get("SPY_SMA50_TREND")
    equity_weight = sum(v16_weights.get(a, 0) for a in EQUITY_ASSETS)
    if equity_weight > 0.30 and spy_trend == "Falling":
        divergences.append({
            "type": "EQUITY_TREND_DIVERGENCE",
            "detail": f"V16 hat {equity_weight:.0%} Equity aber SPY SMA50 Falling",
            "severity": "HIGH",
        })
    elif equity_weight > 0.30 and spy_trend == "Rising":
        confirmations.append({
            "type": "EQUITY_TREND_CONFIRMED",
            "detail": f"V16 {equity_weight:.0%} Equity + SPY Rising — bestätigt",
        })

    # Real Yield Trend vs Gold
    real_yield_trend = dw_data.get("REAL_YIELD_10Y_TREND")
    if gold_weight > 0.10 and real_yield_trend == "Falling":
        confirmations.append({
            "type": "GOLD_REAL_YIELD_CONFIRMED",
            "detail": f"Gold {gold_weight:.0%} + Real Yields Falling — historisch positiv für Gold",
        })
    elif gold_weight > 0.10 and real_yield_trend == "Rising":
        divergences.append({
            "type": "GOLD_REAL_YIELD_DIVERGENCE",
            "detail": f"V16 hat {gold_weight:.0%} Gold aber Real Yields Rising — Gegenwind",
            "severity": "ELEVATED",
        })

    # Yield Curve
    yield_curve = _safe_float(dw_data.get("YIELD_CURVE_10Y2Y"))
    if yield_curve is not None and yield_curve < 0:
        divergences.append({
            "type": "YIELD_CURVE_INVERTED",
            "detail": f"Yield Curve invertiert ({yield_curve:.0f}bps) — Recession-Signal",
            "severity": "HIGH",
        })

    # Scoring
    high_divs = [d for d in divergences if d["severity"] == "HIGH"]
    elevated_divs = [d for d in divergences if d["severity"] == "ELEVATED"]

    if len(high_divs) >= 2:
        score = 3
    elif len(high_divs) >= 1 and len(elevated_divs) >= 1:
        score = 2
    elif len(high_divs) >= 1 or len(elevated_divs) >= 2:
        score = 2
    elif len(elevated_divs) >= 1:
        score = 1
    else:
        score = 0

    score = min(score, 3)

    # Label
    if divergences:
        label = divergences[0]["type"]
    else:
        label = "All confirmed"

    detail = {
        "divergences": divergences,
        "confirmations": confirmations,
        "high_count": len(high_divs),
        "elevated_count": len(elevated_divs),
    }

    return score, {"score": score, "max": 3, "label": label, "detail": detail}


# =============================================================================
# DIMENSION 5: GEX REGIME (Spec Teil 3 §12)
# =============================================================================

def score_gex_regime(dw_data: dict,
                     event_risk_score: int) -> tuple[int, dict]:
    """
    Score GEX Regime 0-3.

    Negative GEX alone = 1.
    Negative GEX + Event in 48h = 2.
    Strongly negative GEX + Event = 3.
    """
    gex = _safe_float(dw_data.get("OPTIONS_GEX"))

    if gex is None:
        return 0, {"score": 0, "max": 3, "label": "No GEX data",
                    "detail": {"gex": None, "note": "No GEX data available"}}

    has_event = event_risk_score >= 2  # HIGH event in 48h

    if gex < -1.0 and has_event:
        score = 3
        assessment = "STRONGLY_NEGATIVE_PLUS_EVENT"
    elif gex < -0.5 and has_event:
        score = 2
        assessment = "NEGATIVE_PLUS_EVENT"
    elif gex < -0.5:
        score = 1
        assessment = "NEGATIVE"
    elif gex < 0:
        score = 1 if has_event else 0
        assessment = "SLIGHTLY_NEGATIVE"
    else:
        score = 0
        assessment = "POSITIVE_STABILIZING"

    label = f"GEX ${gex:.2f}B — {assessment}"

    detail = {
        "gex": gex,
        "assessment": assessment,
        "has_imminent_event": has_event,
        "note": "Positive GEX = Dealer dampen moves. Negative = Dealer amplify moves.",
    }

    return score, {"score": score, "max": 3, "label": label, "detail": detail}


# =============================================================================
# DIMENSION 6: SENTIMENT EXTREME (Spec Teil 3 §13)
# =============================================================================

def score_sentiment_extreme(dw_data: dict) -> tuple[int, dict]:
    """
    Score Sentiment Extreme 0-3.

    Warns at BOTH extremes:
    - Extreme bullishness = complacency
    - Extreme bearishness = panic / execution risk
    """
    signals = {}

    # AAII Bullish Percentile
    aaii_bull = _safe_float(dw_data.get("AAII_BULL_PCTL"))
    if aaii_bull is not None:
        if aaii_bull > 80:
            signals["aaii"] = {"value": aaii_bull, "assessment": "EXTREME_BULLISH", "warning": True}
        elif aaii_bull < 20:
            signals["aaii"] = {"value": aaii_bull, "assessment": "EXTREME_BEARISH", "warning": True}
        else:
            signals["aaii"] = {"value": aaii_bull, "assessment": "NEUTRAL", "warning": False}

    # CNN Fear & Greed
    cnn_fg = _safe_float(dw_data.get("CNN_FEAR_GREED"))
    if cnn_fg is not None:
        if cnn_fg > 80:
            signals["cnn_fg"] = {"value": cnn_fg, "assessment": "EXTREME_GREED", "warning": True}
        elif cnn_fg < 20:
            signals["cnn_fg"] = {"value": cnn_fg, "assessment": "EXTREME_FEAR", "warning": True}
        else:
            signals["cnn_fg"] = {"value": cnn_fg, "assessment": "NEUTRAL", "warning": False}

    # VIX Level
    vix = _safe_float(dw_data.get("VIX_LEVEL"))
    if vix is not None:
        if vix > 30:
            signals["vix"] = {"value": vix, "assessment": "HIGH_FEAR", "warning": True}
        elif vix < 13:
            signals["vix"] = {"value": vix, "assessment": "COMPLACENT", "warning": True}
        else:
            signals["vix"] = {"value": vix, "assessment": "NORMAL", "warning": False}

    # Put/Call Ratio
    pcr = _safe_float(dw_data.get("PUT_CALL_RATIO"))
    if pcr is not None:
        if pcr < 0.6:
            signals["pcr"] = {"value": pcr, "assessment": "EXTREME_COMPLACENT", "warning": True}
        elif pcr > 1.2:
            signals["pcr"] = {"value": pcr, "assessment": "EXTREME_FEAR", "warning": True}
        else:
            signals["pcr"] = {"value": pcr, "assessment": "NORMAL", "warning": False}

    # HY OAS Spread
    hy_oas = _safe_float(dw_data.get("HY_OAS_SPREAD"))
    if hy_oas is not None:
        if hy_oas > 500:
            signals["hy_oas"] = {"value": hy_oas, "assessment": "CREDIT_STRESS", "warning": True}
        elif hy_oas > 400:
            signals["hy_oas"] = {"value": hy_oas, "assessment": "ELEVATED", "warning": True}
        else:
            signals["hy_oas"] = {"value": hy_oas, "assessment": "NORMAL", "warning": False}

    # Insider Buy/Sell
    insider = _safe_float(dw_data.get("INSIDER_BUY_SELL"))
    if insider is not None:
        if insider > 5.0:
            signals["insider"] = {"value": insider, "assessment": "HEAVY_BUYING", "warning": False}
        elif insider < 0.5:
            signals["insider"] = {"value": insider, "assessment": "HEAVY_SELLING", "warning": True}
        else:
            signals["insider"] = {"value": insider, "assessment": "NORMAL", "warning": False}

    # MOVE Index
    move = _safe_float(dw_data.get("MOVE_INDEX"))
    if move is not None:
        if move > 120:
            signals["move"] = {"value": move, "assessment": "HIGH_BOND_VOL", "warning": True}
        else:
            signals["move"] = {"value": move, "assessment": "NORMAL", "warning": False}

    # Scoring
    warning_count = sum(1 for s in signals.values() if s.get("warning", False))

    if warning_count >= 4:
        score = 3
    elif warning_count >= 3:
        score = 2
    elif warning_count >= 1:
        score = 1
    else:
        score = 0

    score = min(score, 3)

    # Label
    warnings = [k for k, v in signals.items() if v.get("warning")]
    if warnings:
        label = ", ".join(f"{w}: {signals[w]['assessment']}" for w in warnings[:2])
    else:
        label = "Neutral"

    return score, {"score": score, "max": 3, "label": label,
                   "detail": {"signals": signals, "warning_count": warning_count}}


# =============================================================================
# VETO SYSTEM (Spec Teil 3 §7.4)
# =============================================================================

def apply_veto_rules(dimension_scores: dict[str, int],
                     raw_sum: int) -> tuple[int, str | None]:
    """
    Apply veto rules to raw score sum.

    Returns:
        (adjusted_score, veto_reason or None)
    """
    max_dim = max(dimension_scores.values())

    # No single alarm → EXECUTE
    if max_dim == 0:
        return 0, None

    # All relaxed → Cap at EXECUTE range
    if max_dim <= 1 and raw_sum <= 4:
        return min(raw_sum, 3), None

    # Single dimension maximum → Minimum CAUTION
    if max_dim >= 3 and raw_sum < 4:
        veto_dim = [k for k, v in dimension_scores.items() if v == 3][0]
        return 4, f"VETO: {veto_dim} at maximum (3/3)"

    # Double warning + high sum → Minimum WAIT
    if max_dim >= 2 and raw_sum >= 7:
        return max(raw_sum, 7), None

    return raw_sum, None


# =============================================================================
# SCORING ORCHESTRATOR (Spec Teil 3 §14)
# =============================================================================

def calculate_execution_score(
    events: list[dict],
    v16_weights: dict,
    v16_regime: str,
    dw_data: dict,
    today: date,
) -> dict:
    """
    Main scoring function. Computes all 6 dimensions + veto.

    Returns:
        {
            "total_score": int,
            "raw_score": int,
            "max_possible": 18,
            "execution_level": str,
            "veto_applied": bool,
            "veto_reason": str or None,
            "dimensions": { ... },
        }
    """
    # Score each dimension
    event_score, event_result = score_event_risk(events, today)
    pos_score, pos_result = score_positioning_conflict(v16_weights, dw_data)
    liq_score, liq_result = score_liquidity_risk(dw_data)
    cross_score, cross_result = score_cross_asset_confirmation(
        v16_regime, v16_weights, dw_data
    )
    gex_score, gex_result = score_gex_regime(dw_data, event_score)
    sent_score, sent_result = score_sentiment_extreme(dw_data)

    dimension_scores = {
        "event_risk": event_score,
        "positioning_conflict": pos_score,
        "liquidity_risk": liq_score,
        "cross_asset_confirmation": cross_score,
        "gex_regime": gex_score,
        "sentiment_extreme": sent_score,
    }

    raw_sum = sum(dimension_scores.values())

    # Apply veto rules
    adjusted_score, veto_reason = apply_veto_rules(dimension_scores, raw_sum)

    # Determine execution level
    if adjusted_score <= 3:
        level = "EXECUTE"
    elif adjusted_score <= 6:
        level = "CAUTION"
    elif adjusted_score <= 9:
        level = "WAIT"
    else:
        level = "HOLD"

    return {
        "total_score": adjusted_score,
        "raw_score": raw_sum,
        "max_possible": 18,
        "execution_level": level,
        "veto_applied": veto_reason is not None,
        "veto_reason": veto_reason,
        "dimensions": {
            "event_risk": event_result,
            "positioning_conflict": pos_result,
            "liquidity_risk": liq_result,
            "cross_asset_confirmation": cross_result,
            "gex_regime": gex_result,
            "sentiment_extreme": sent_result,
        },
    }
