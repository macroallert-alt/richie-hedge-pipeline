"""
Market Analyst — Templates Module
Generates Key Driver and Tension strings from templates. NO LLM.

Deterministic: same data = same text. Always.
Agent 0 (CIO, which IS an LLM) gets FACTS with NUMBERS
and builds its own narrative from them.

Source: AGENT2_SPEC_TEIL6 Sections 19.1-19.4
"""


def select_key_driver(
    layer_name: str,
    sub_scores: dict,
    raw_data: dict,
    templates_config: dict,
    field_weights: dict = None,
) -> str:
    """
    Selects the template for the STRONGEST driver of the score.
    The sub-score with the largest absolute value AND PRIMARY weight wins.

    Returns: filled template string
    """
    layer_templates = templates_config.get(layer_name, {})
    kd_templates = layer_templates.get("key_driver_templates", {})

    if not kd_templates or not sub_scores:
        return _fallback_key_driver(layer_name, sub_scores)

    # Filter to PRIMARY fields if weight info available
    if field_weights:
        primary_scores = {
            k: v for k, v in sub_scores.items()
            if not k.startswith("ic_") and
            _is_primary(k, field_weights)
        }
    else:
        primary_scores = {k: v for k, v in sub_scores.items() if not k.startswith("ic_")}

    if not primary_scores:
        primary_scores = {k: v for k, v in sub_scores.items() if not k.startswith("ic_")}

    # Find strongest driver
    strongest = max(primary_scores, key=lambda k: abs(primary_scores[k]))
    score = primary_scores[strongest]

    # Find matching template group
    template_group = _find_template_group(strongest, kd_templates)
    if not template_group:
        return _fallback_key_driver(layer_name, sub_scores)

    # Select template by score magnitude
    template_key = _score_to_template_key(score)
    template_str = _get_template(template_group, template_key, score)

    # Fill placeholders with actual data — pass the strongest field as context
    context_field = _template_group_to_field(strongest, kd_templates)
    filled = _fill_template(template_str, raw_data, context_field=context_field)

    # Add sub-driver detail if available
    drivers = template_group.get("drivers", {})
    if drivers:
        driver_detail = _select_sub_driver(sub_scores, raw_data, drivers)
        if driver_detail:
            filled += " — " + driver_detail

    return filled


def generate_tension(
    layer_name: str,
    sub_scores: dict,
    raw_data: dict,
    templates_config: dict,
) -> str:
    """
    Tension = contradiction WITHIN a layer.
    Exists only when significant sub-scores point in both directions.

    Returns: tension string or None if no contradiction exists.
    """
    # Find significant positive and negative sub-scores
    positives = {k: v for k, v in sub_scores.items()
                 if v >= 3 and not k.startswith("ic_")}
    negatives = {k: v for k, v in sub_scores.items()
                 if v <= -3 and not k.startswith("ic_")}

    if not positives or not negatives:
        return None  # No contradiction

    strongest_pos = max(positives, key=positives.get)
    strongest_neg = min(negatives, key=negatives.get)

    # Check for pre-built tension templates
    layer_templates = templates_config.get(layer_name, {})
    tension_templates = layer_templates.get("tension_templates", {})

    # Try to find a matching template
    template_key = f"{strongest_pos}_vs_{strongest_neg}"
    if template_key in tension_templates:
        return _fill_template(tension_templates[template_key], raw_data, context_field=strongest_pos)

    # Fallback: generic tension string
    pos_label = _field_label(strongest_pos, positives[strongest_pos])
    neg_label = _field_label(strongest_neg, negatives[strongest_neg])

    return f"{pos_label} BUT {neg_label}"


# --- Internal helpers ---


def _find_template_group(field_name: str, kd_templates: dict) -> dict:
    """Finds the template group that matches the field name."""
    # Direct match
    if field_name in kd_templates:
        return kd_templates[field_name]

    # Partial match (e.g., "net_liquidity" matches "net_liq_trend")
    for key, group in kd_templates.items():
        if field_name.startswith(key) or key.startswith(field_name):
            return group

    # Category-based mapping
    field_category_map = {
        "spread_2y10y": "yield_curve",
        "spread_3m10y": "yield_curve",
        "hy_oas": "credit",
        "ig_oas": "credit",
        "nfci": "financial_conditions",
        "anfci": "financial_conditions",
        "pct_above_200dma": "breadth",
        "nh_nl": "new_highs",
        "dxy": "dollar",
        "usdcnh": "china",
        "china_10y": "china",
        "naaim_exposure": "positioning",
        "aaii_bull_bear": "positioning",
        "cot_es_leveraged": "positioning",
        "cu_au_ratio": "cu_au",
        "wti_curve": "oil_curve",
        "vix": "vol_regime",
        "vix_term_struct": "vol_regime",
        "ic_FED_POLICY": "policy_stance",
        "ic_LIQUIDITY": "net_liquidity",
        "disc_window": "policy_stance",
        "real_10y_yield": "yield_curve",
        "usdjpy": "dollar",
    }
    category = field_category_map.get(field_name)
    if category and category in kd_templates:
        return kd_templates[category]

    return None




def _template_group_to_field(field_name: str, kd_templates: dict) -> str:
    """Returns the actual raw_data field name for a given strongest driver."""
    return field_name

def _score_to_template_key(score: int) -> str:
    """Maps score to template key name."""
    if score >= 5:
        return "strong_positive"
    elif score >= 2:
        return "positive"
    elif score <= -5:
        return "strong_negative"
    elif score <= -2:
        return "negative"
    else:
        return "neutral"


def _get_template(template_group: dict, preferred_key: str, score: int) -> str:
    """Gets template with fallback chain."""
    # Try preferred key
    if preferred_key in template_group:
        return template_group[preferred_key]

    # Fallback: positive/negative
    fallback = "positive" if score > 0 else "negative"
    if fallback in template_group:
        return template_group[fallback]

    # Special keys for specific fields
    special_keys = {
        "inverted": score < 0,
        "tight": score > 0,
        "widening": score < 0,
        "stress": score < -4,
        "risk_on": score > 0,
        "defensive": score < 0,
        "backwardation": score > 0,
        "contango": score < 0,
        "calm": score > 0,
        "elevated": -5 < score < 0,
        "acute": score <= -5,
        "easing": score > 0,
        "tightening": score < 0,
        "capitulation": score > 4,
        "euphoria": score < -4,
        "fear": 0 < score <= 4,
    }
    for key, condition in special_keys.items():
        if condition and key in template_group:
            return template_group[key]

    # Last resort
    return list(template_group.values())[0] if template_group else ""


def _fill_template(template_str: str, raw_data: dict, context_field: str = None) -> str:
    """Fills {placeholder} with actual values from raw data.
    
    context_field: the field that triggered this template (e.g. "spread_2y10y").
    Generic placeholders like {value}, {pctl}, {delta_5d}, {direction} resolve
    to the context_field's data first before scanning all fields.
    """
    import re

    GENERIC_KEYS = {"value", "pctl", "delta_5d", "direction", "pctl_1y"}

    ctx_data = raw_data.get(context_field, {}) if context_field else {}
    if not isinstance(ctx_data, dict):
        ctx_data = {"value": ctx_data}

    GENERIC_MAP = {
        "value": "value",
        "pctl": "pctl_1y",
        "delta_5d": "delta_5d",
        "direction": "direction",
        "pctl_1y": "pctl_1y",
    }

    def replacer(match):
        key = match.group(1)

        if key in GENERIC_KEYS and ctx_data:
            mapped = GENERIC_MAP.get(key, key)
            val = ctx_data.get(mapped)
            if val is not None:
                return str(val)

        if key.endswith("_pctl"):
            prefix = key[:-5]
            for field_name, field_data in raw_data.items():
                if not isinstance(field_data, dict):
                    continue
                if prefix.replace("_", "") in field_name.replace("_", ""):
                    return str(field_data.get("pctl_1y", "?"))

        if key.endswith("_direction"):
            prefix = key[:-10]
            for field_name, field_data in raw_data.items():
                if not isinstance(field_data, dict):
                    continue
                if prefix.replace("_", "") in field_name.replace("_", ""):
                    return str(field_data.get("direction", "?"))

        if key.endswith("_value"):
            prefix = key[:-6]
            for field_name, field_data in raw_data.items():
                if not isinstance(field_data, dict):
                    continue
                if prefix.replace("_", "") in field_name.replace("_", ""):
                    return str(field_data.get("value", "?"))

        if key.endswith("_delta_5d"):
            prefix = key[:-9]
            for field_name, field_data in raw_data.items():
                if not isinstance(field_data, dict):
                    continue
                if prefix.replace("_", "") in field_name.replace("_", ""):
                    return str(field_data.get("delta_5d", "?"))

        if ctx_data and key in ctx_data:
            return str(ctx_data[key])

        for field_name, field_data in raw_data.items():
            if not isinstance(field_data, dict):
                continue
            if key in field_data:
                return str(field_data[key])

        return f"{{{key}}}"

    return re.sub(r"\{(\w+)\}", replacer, template_str)


def _select_sub_driver(sub_scores: dict, raw_data: dict, drivers: dict) -> str:
    """Selects the most relevant sub-driver detail."""
    for driver_key, template in drivers.items():
        # Match driver key to field state
        field_name = driver_key.replace("_falling", "").replace("_rising", "")
        field_data = raw_data.get(field_name, {})
        if not isinstance(field_data, dict):
            continue

        direction = field_data.get("direction", "FLAT")

        if "falling" in driver_key and direction == "DOWN":
            return _fill_template(template, raw_data, context_field=field_name)
        elif "rising" in driver_key and direction == "UP":
            return _fill_template(template, raw_data, context_field=field_name)

    return None


def _is_primary(field_name: str, field_weights: dict) -> bool:
    """Checks if field has PRIMARY weight in any regime."""
    w = field_weights.get(field_name, {})
    return w.get("risk_on") == "PRIMARY" or w.get("risk_off") == "PRIMARY"


def _field_label(field_name: str, score: int) -> str:
    """Generates a readable label for a field + score."""
    direction = "bullish" if score > 0 else "bearish"
    clean_name = field_name.replace("_", " ").title()
    return f"{clean_name} ({direction}, score {score})"


def _fallback_key_driver(layer_name: str, sub_scores: dict) -> str:
    """Generates a simple fallback key driver string."""
    if not sub_scores:
        return f"{layer_name}: no data"

    strongest = max(sub_scores, key=lambda k: abs(sub_scores[k]))
    score = sub_scores[strongest]
    return f"{strongest}: score {score}"
