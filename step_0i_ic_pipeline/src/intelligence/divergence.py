"""
IC Pipeline — Stufe 2: Divergence Calculator
IC Consensus vs. Market Data divergences. Deterministic, no LLM.

Implements: AD.1, AD.4, AD.5, AD.7, AD.8, AD.11 from Addendum V2.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AD.1: Theme to Data Fields Mapping
# ---------------------------------------------------------------------------
THEME_TO_DATA_FIELDS = {
    "LIQUIDITY": {
        "primary": ["net_liquidity"],
        "transforms_needed": ["direction", "pctl_1y"],
    },
    "FED_POLICY": {
        "primary": ["fedwatch_cut_prob"],
        "secondary": ["fed_funds_rate"],
    },
    "CREDIT": {
        "primary": ["hy_oas"],
        "transforms_needed": ["value", "delta_5d", "pctl_1y"],
    },
    "RECESSION": {
        "primary": ["initial_claims"],
        "secondary": ["ism_mfg"],
    },
    "INFLATION": {
        "primary": ["breakeven_5y5y", "cpi_yoy"],
    },
    "VOLATILITY": {
        "primary": ["vix", "vix_term_ratio"],
        "secondary": ["move_index"],
    },
    "POSITIONING": {
        "primary": ["pc_ratio_equity", "aaii_bull_bear"],
    },
    "CHINA_EM": {
        "primary": ["china_10y", "usdcnh"],
    },
    "DOLLAR": {
        "primary": ["dxy"],
        "transforms_needed": ["value", "pctl_1y"],
    },
    "ENERGY": {
        "primary": ["wti_curve"],
        "secondary": ["wti_level", "wti_delta_5d"],
    },
    "COMMODITIES": {
        "primary": ["cu_au_ratio"],
        "transforms_needed": ["value", "direction"],
    },
}

# AD.7: Timing classes per field
FIELD_TIMING_CLASS = {
    "net_liquidity": "DELAYED",
    "fedwatch_cut_prob": "PREV_CLOSE",
    "hy_oas": "PREV_CLOSE",
    "initial_claims": "WEEKLY",
    "ism_mfg": "MONTHLY",
    "breakeven_5y5y": "PREV_CLOSE",
    "cpi_yoy": "MONTHLY",
    "vix": "PREV_CLOSE",
    "vix_term_ratio": "PREV_CLOSE",
    "move_index": "PREV_CLOSE",
    "pc_ratio_equity": "PREV_CLOSE",
    "aaii_bull_bear": "WEEKLY",
    "china_10y": "PREV_CLOSE",
    "usdcnh": "LIVE",
    "dxy": "PREV_CLOSE",
    "wti_curve": "PREV_CLOSE",
    "cu_au_ratio": "PREV_CLOSE",
}


# ---------------------------------------------------------------------------
# AD.1.4: evaluate_market_signal per theme
# ---------------------------------------------------------------------------
def evaluate_market_signal(
    theme: str, dc_data: Optional[dict], mapping: dict
) -> Optional[dict]:
    """
    Evaluate DC data for a theme and return market signal direction.
    Returns {direction, strength, details, dc_fields_used} or None.
    """
    if dc_data is None:
        return None

    theme_data = dc_data.get(theme)
    if theme_data is None:
        return None

    primary_field = mapping["primary"][0]

    if theme == "LIQUIDITY":
        direction_val = theme_data.get("net_liquidity_direction")
        pctl = theme_data.get("net_liquidity_pctl_1y")
        if direction_val == "UP" and pctl and pctl > 50:
            direction, strength = "BULLISH", min(1.0, pctl / 100)
        elif direction_val == "DOWN" and pctl and pctl < 50:
            direction, strength = "BEARISH", min(1.0, (100 - pctl) / 100)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "FED_POLICY":
        cut_prob = theme_data.get("fedwatch_cut_prob")
        if cut_prob is None:
            return None
        if cut_prob > 60:
            direction, strength = "BULLISH", min(1.0, cut_prob / 100)
        elif cut_prob < 30:
            direction, strength = "BEARISH", min(1.0, (100 - cut_prob) / 100)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "CREDIT":
        oas_delta = theme_data.get("hy_oas_delta_5d")
        oas_pctl = theme_data.get("hy_oas_pctl_1y")
        if oas_delta is not None and oas_delta > 10 and oas_pctl and oas_pctl > 60:
            direction, strength = "BEARISH", min(1.0, oas_pctl / 100)
        elif oas_delta is not None and oas_delta < -10 and oas_pctl and oas_pctl < 40:
            direction, strength = "BULLISH", min(1.0, (100 - oas_pctl) / 100)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "RECESSION":
        ism = theme_data.get("ism_mfg")
        if ism is not None and ism > 50:
            direction, strength = "BULLISH", min(1.0, (ism - 50) / 10)
        elif ism is not None and ism < 48:
            direction, strength = "BEARISH", min(1.0, (50 - ism) / 10)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "INFLATION":
        be_5y5y = theme_data.get("breakeven_5y5y")
        if be_5y5y is not None and be_5y5y > 2.5:
            direction, strength = "BEARISH", min(1.0, (be_5y5y - 2.0) / 1.5)
        elif be_5y5y is not None and be_5y5y < 2.0:
            direction, strength = "BULLISH", min(1.0, (2.5 - be_5y5y) / 1.5)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "VOLATILITY":
        vix = theme_data.get("vix_level")
        term_ratio = theme_data.get("vix_term_ratio")
        if vix is not None and vix > 25:
            direction, strength = "BEARISH", min(1.0, vix / 40)
        elif vix is not None and vix < 15 and term_ratio and term_ratio > 1.0:
            direction, strength = "BULLISH", 0.6
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "POSITIONING":
        aaii = theme_data.get("aaii_bull_bear")
        if aaii is not None and aaii > 25:
            direction, strength = "BULLISH", min(1.0, aaii / 40)
        elif aaii is not None and aaii < -15:
            direction, strength = "BEARISH", min(1.0, abs(aaii) / 30)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "CHINA_EM":
        usdcnh = theme_data.get("usdcnh")
        if usdcnh is not None and usdcnh > 7.3:
            direction, strength = "BEARISH", min(1.0, (usdcnh - 7.0) / 0.5)
        elif usdcnh is not None and usdcnh < 7.0:
            direction, strength = "BULLISH", min(1.0, (7.3 - usdcnh) / 0.5)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "DOLLAR":
        dxy_pctl = theme_data.get("dxy_pctl_1y")
        if dxy_pctl is not None and dxy_pctl > 70:
            direction, strength = "BULLISH", min(1.0, dxy_pctl / 100)
        elif dxy_pctl is not None and dxy_pctl < 30:
            direction, strength = "BEARISH", min(1.0, (100 - dxy_pctl) / 100)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "ENERGY":
        wti_delta = theme_data.get("wti_delta_5d")
        if wti_delta is not None and wti_delta > 3:
            direction, strength = "BULLISH", min(1.0, wti_delta / 10)
        elif wti_delta is not None and wti_delta < -3:
            direction, strength = "BEARISH", min(1.0, abs(wti_delta) / 10)
        else:
            direction, strength = "NEUTRAL", 0.3

    elif theme == "COMMODITIES":
        cu_au_dir = theme_data.get("cu_au_direction")
        if cu_au_dir == "UP":
            direction, strength = "BULLISH", 0.6
        elif cu_au_dir == "DOWN":
            direction, strength = "BEARISH", 0.6
        else:
            direction, strength = "NEUTRAL", 0.3
    else:
        return None

    return {
        "direction": direction,
        "strength": strength,
        "details": f"{primary_field}: {theme_data}",
        "dc_fields_used": list(theme_data.keys()),
    }


# ---------------------------------------------------------------------------
# AD.4: Divergence Scoring
# ---------------------------------------------------------------------------
def compute_divergence_score(
    ic_consensus_score: float,
    ic_direction: str,
    market_signal: dict,
) -> Optional[dict]:
    """Compute divergence score between IC consensus and DC market data."""
    market_dir = market_signal["direction"]
    market_strength = market_signal["strength"]

    if ic_direction == market_dir:
        return {"divergence_type": "ALIGNED", "magnitude": 0.0}

    if ic_direction == "NEUTRAL" or market_dir == "NEUTRAL":
        magnitude = abs(ic_consensus_score) / 10 * market_strength * 0.5
        dtype = "PARTIAL"
    elif ic_direction == "BULLISH" and market_dir == "BEARISH":
        magnitude = abs(ic_consensus_score) / 10 * market_strength
        dtype = "IC_BULLISH_DC_BEARISH"
    elif ic_direction == "BEARISH" and market_dir == "BULLISH":
        magnitude = abs(ic_consensus_score) / 10 * market_strength
        dtype = "IC_BEARISH_DC_BULLISH"
    else:
        magnitude = abs(ic_consensus_score) / 10 * market_strength * 0.5
        dtype = "PARTIAL"

    return {
        "divergence_type": dtype,
        "magnitude": round(min(1.0, magnitude), 3),
    }


# ---------------------------------------------------------------------------
# AD.7: Timing Discount
# ---------------------------------------------------------------------------
def _apply_timing_discount(divergence: dict, primary_field: str) -> dict:
    """Apply timing discount for PREV_CLOSE fields when IC claim is from today."""
    timing = FIELD_TIMING_CLASS.get(primary_field, "PREV_CLOSE")
    if timing == "LIVE":
        pass  # no discount
    elif timing == "PREV_CLOSE":
        divergence["magnitude"] = round(divergence["magnitude"] * 0.8, 3)
        divergence["timing_note"] = f"{primary_field} is PREV_CLOSE — possible timing mismatch"
    elif timing in ("WEEKLY", "MONTHLY"):
        divergence["timing_note"] = f"{primary_field} is {timing} — delay expected"
    return divergence


# ---------------------------------------------------------------------------
# Divergence Classification (from Spec Teil 4 §4.7)
# ---------------------------------------------------------------------------
def _classify_divergence(ic_dir: str, market_dir: str) -> str:
    """
    BLIND_SPOT: Market shows stress but barely discussed
    CROWDED: Everyone bullish but market weak
    CONTRARIAN: Big contradiction between consensus and market
    """
    if ic_dir == "NEUTRAL" and market_dir == "BEARISH":
        return "BLIND_SPOT"
    elif ic_dir == "BULLISH" and market_dir == "BEARISH":
        return "CROWDED"
    elif ic_dir == "BEARISH" and market_dir == "BULLISH":
        return "CONTRARIAN"
    return "DIVERGENCE"


# ---------------------------------------------------------------------------
# Main Divergence Calculation
# ---------------------------------------------------------------------------
def calculate_divergences(
    ic_consensus: dict,
    dc_data: Optional[dict],
    divergence_config: dict,
) -> list[dict]:
    """
    Identify divergences between IC consensus and market data.
    Returns list of divergences sorted by severity.
    """
    if dc_data is None:
        logger.info("No DC data available — divergence calculation skipped")
        return []

    divergences = []

    for topic, consensus in ic_consensus.items():
        if consensus.get("confidence") == "NO_DATA":
            continue

        # AD.8: Check if divergence is possible for this topic
        if not divergence_config.get(topic, {}).get("divergence_possible", False):
            continue

        mapping = THEME_TO_DATA_FIELDS.get(topic)
        if not mapping:
            continue

        # Evaluate market signal
        market_signal = evaluate_market_signal(topic, dc_data, mapping)
        if market_signal is None:
            continue

        # Determine IC direction
        score = consensus["consensus_score"]
        if score > 1.0:
            ic_direction = "BULLISH"
        elif score < -1.0:
            ic_direction = "BEARISH"
        else:
            ic_direction = "NEUTRAL"

        # AD.4: Compute divergence score
        div_score = compute_divergence_score(score, ic_direction, market_signal)
        if div_score is None or div_score["divergence_type"] == "ALIGNED":
            continue

        # AD.7: Timing discount
        primary_field = mapping["primary"][0]
        div_score = _apply_timing_discount(div_score, primary_field)

        severity = div_score["magnitude"] * 10  # scale 0-10 for compat

        # AD.11: Source attribution
        top_contributors = [
            c["source_id"]
            for c in sorted(
                consensus.get("contributors", []),
                key=lambda x: abs(x.get("avg_bias_adjusted_signal", 0))
                * x.get("expertise_weight", 1),
                reverse=True,
            )[:3]
        ]

        divergences.append({
            "topic": topic,
            "ic_consensus": score,
            "ic_direction": ic_direction,
            "ic_source_count": consensus.get("source_count", 0),
            "ic_top_contributors": top_contributors,
            "market_direction": market_signal["direction"],
            "market_data": market_signal["details"],
            "dc_source_field": primary_field,
            "dc_source_value": None,  # populated if specific value available
            "dc_timing_class": FIELD_TIMING_CLASS.get(primary_field, "UNKNOWN"),
            "severity": round(severity, 2),
            "magnitude": div_score["magnitude"],
            "divergence_type": _classify_divergence(ic_direction, market_signal["direction"]),
            "divergence_detail": div_score["divergence_type"],
            "timing_note": div_score.get("timing_note"),
            "interpretation": (
                f"{len(top_contributors)} sources say {ic_direction} "
                f"(score {score:.1f}), market data says {market_signal['direction']}"
            ),
        })

    return sorted(divergences, key=lambda x: -x["severity"])