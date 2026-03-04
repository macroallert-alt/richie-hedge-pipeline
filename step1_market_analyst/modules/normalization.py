"""
Market Analyst — Normalization Module
Converts raw field data to sub-scores on integer scale -10 to +10.

Four methods:
  pctl             — Percentile-based linear mapping
  threshold        — Score relative to known neutral point
  nonlinear_extreme — Dead zone in middle, only extremes count
  direction_score  — Direction (UP/DOWN/FLAT) + momentum bonus

Source: AGENT2_SPEC_TEIL3 Sections 5.1-5.4
"""


def pctl_to_score(pctl_1y: float, invert: bool = False) -> int:
    """
    Percentile (0-100) mapped linearly to -10..+10.
    pctl 0 -> -10, pctl 50 -> 0, pctl 100 -> +10.

    Used for: net_liquidity, rrp, tga, mmf_assets, hy_oas, ig_oas,
              nh_nl, dxy, usdcnh, cu_au_ratio, usdjpy
    """
    if pctl_1y is None:
        return 0
    score = (pctl_1y - 50) / 5
    if invert:
        score = -score
    return int(max(-10, min(10, round(score))))


def threshold_to_score(
    value: float,
    neutral: float,
    bullish_extreme: float,
    bearish_extreme: float,
    invert: bool = False,
) -> int:
    """
    Score based on known threshold/neutral point.
    neutral = 0-point. bullish_extreme = +10, bearish_extreme = -10.

    Handles both normal ranges (bullish > neutral > bearish)
    and inverted ranges (bullish < neutral < bearish, e.g., WTI curve
    where backwardation < 1.0 is bullish).

    Used for: spread_2y10y, spread_3m10y, nfci, anfci, vix_term_struct,
              iv_rv_spread, pct_above_200dma, spy_tlt_corr, wti_curve
    """
    if value is None:
        return 0

    # Detect inverted range: bullish_extreme is on the opposite side of neutral
    # than expected (bullish < neutral or bearish > neutral)
    bullish_above = bullish_extreme >= neutral
    bearish_below = bearish_extreme <= neutral

    if bullish_above and bearish_below:
        # Normal range: bullish > neutral > bearish
        if value >= neutral:
            max_dist = bullish_extreme - neutral
            score = min(10, ((value - neutral) / max_dist) * 10) if max_dist > 0 else 0
        else:
            max_dist = neutral - bearish_extreme
            score = -min(10, ((neutral - value) / max_dist) * 10) if max_dist > 0 else 0
    else:
        # Inverted range: bullish < neutral < bearish (e.g., WTI, spy_tlt_corr)
        # Map value->bullish as +10, value->bearish as -10
        bullish_dist = abs(bullish_extreme - neutral)
        bearish_dist = abs(bearish_extreme - neutral)

        if (bullish_extreme < neutral and value <= neutral) or \
           (bullish_extreme > neutral and value >= neutral):
            # Moving toward bullish extreme
            dist = abs(value - neutral)
            max_dist = bullish_dist
            score = min(10, (dist / max_dist) * 10) if max_dist > 0 else 0
        else:
            # Moving toward bearish extreme
            dist = abs(value - neutral)
            max_dist = bearish_dist
            score = -min(10, (dist / max_dist) * 10) if max_dist > 0 else 0

    if invert:
        score = -score
    return int(max(-10, min(10, round(score))))


def nonlinear_extreme_score(
    pctl_1y: float,
    dead_zone_low: float = 25,
    dead_zone_high: float = 75,
    contrarian: bool = False,
    invert: bool = False,
) -> int:
    """
    Signal only at extremes. Dead zone in middle = score 0.
    contrarian=True: high values = negative (crowding), low = positive (capitulation).
    invert=True: applied AFTER contrarian logic (for fields like VIX where high = bad).

    Used for: naaim_exposure, aaii_bull_bear, cot_es_leveraged, cot_zn_leveraged,
              pc_ratio_equity, vix, disc_window
    """
    if pctl_1y is None:
        return 0

    if dead_zone_low <= pctl_1y <= dead_zone_high:
        return 0  # Middle — no signal

    if pctl_1y > dead_zone_high:
        raw = ((pctl_1y - dead_zone_high) / (100 - dead_zone_high)) * 10
        score = -raw if contrarian else raw
    else:
        raw = ((dead_zone_low - pctl_1y) / dead_zone_low) * 10
        score = raw if contrarian else -raw

    if invert:
        score = -score
    return int(max(-10, min(10, round(score))))


def direction_to_score(direction: str, delta_5d_norm: float = 0) -> int:
    """
    For fields where DIRECTION matters more than level.
    direction: "UP" / "DOWN" / "FLAT"
    delta_5d_norm: normalized 5-day delta for momentum bonus (0-10 scale).

    Used for: walcl, china_10y
    """
    base_map = {"UP": 3, "DOWN": -3, "FLAT": 0}
    base = base_map.get(direction, 0)

    # Momentum bonus: up to ±3 additional points
    momentum = min(3, abs(delta_5d_norm))
    if base > 0:
        score = base + momentum
    elif base < 0:
        score = base - momentum
    else:
        score = 0

    return int(max(-10, min(10, round(score))))


def normalize_field(field_data: dict, norm_params: dict) -> int:
    """
    Master dispatcher. Reads normalization config and calls the right method.

    field_data: dict with keys like value, pctl_1y, direction, delta_5d, confidence, etc.
    norm_params: dict from normalization.json for this field.

    Returns: integer sub-score -10 to +10.
    """
    method = norm_params.get("method")
    invert = norm_params.get("invert", False)

    # Handle string invert values from config (contrarian, contextual)
    if isinstance(invert, str):
        invert = False  # contextual/contrarian handled by specific methods

    if method == "pctl":
        return pctl_to_score(
            field_data.get("pctl_1y"),
            invert=invert,
        )

    elif method == "threshold":
        return threshold_to_score(
            value=field_data.get("value"),
            neutral=norm_params["neutral"],
            bullish_extreme=norm_params["bullish_extreme"],
            bearish_extreme=norm_params["bearish_extreme"],
            invert=invert,
        )

    elif method == "nonlinear_extreme":
        return nonlinear_extreme_score(
            pctl_1y=field_data.get("pctl_1y"),
            dead_zone_low=norm_params.get("dead_zone_low", 25),
            dead_zone_high=norm_params.get("dead_zone_high", 75),
            contrarian=norm_params.get("contrarian", False),
            invert=invert,
        )

    elif method == "direction_score":
        return direction_to_score(
            direction=field_data.get("direction", "FLAT"),
            delta_5d_norm=field_data.get("delta_5d_norm", 0),
        )

    else:
        # Unknown method — return 0 (safe default)
        return 0
