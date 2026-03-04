"""
Market Analyst — Signal Quality Module
Checks whether scores are trustworthy.

Three checks:
  1. Suppression Detection — scores artificially suppressed (e.g., VIX by dealer gamma)
  2. Data Freshness — stale fields degrade sub-scores
  3. Anomaly Pass-Through — anomaly flags from Data Collector

Quality statuses:
  CONFIRMED  — All consistent, no suppression, data fresh
  SUSPICIOUS — Suppression detection triggered
  DEGRADED   — >30% fields stale OR >1 anomaly

Source: AGENT2_SPEC_TEIL3 Section 8
"""


def check_signal_suppression(
    layer_name: str,
    raw_data: dict,
    suppression_rules: list,
) -> list:
    """
    Checks suppression rules against raw field data for a specific layer.

    layer_name: full layer name
    raw_data: {field_name: {value, pctl_1y, direction, ...}}
    suppression_rules: list from signal_quality_rules.json["suppression_checks"]

    Returns: list of triggered suppression checks
    """
    triggered = []

    for rule in suppression_rules:
        if rule.get("layer") != layer_name:
            continue

        all_match = True
        for condition_key, condition in rule.get("conditions", {}).items():
            field_name = condition["field"]
            test = condition["test"]

            field_data = raw_data.get(field_name)
            if field_data is None:
                all_match = False
                break

            if not _evaluate_test(field_data, test):
                all_match = False
                break

        if all_match:
            triggered.append(
                {
                    "check_id": rule["id"],
                    "layer": rule["layer"],
                    "quality": rule["quality"],
                    "reason": rule["reason"],
                    "true_risk": rule["true_risk"],
                    "affected_sub_scores": rule["affected_sub_scores"],
                }
            )

    return triggered


def check_data_freshness(
    layer_fields: list,
    raw_data: dict,
    stale_threshold: float = 0.5,
) -> list:
    """
    Identifies fields with low confidence (stale data).

    layer_fields: list of field names in this layer
    raw_data: {field_name: {confidence, ...}}
    stale_threshold: confidence below this = stale

    Returns: list of stale field dicts
    """
    stale_fields = []

    for field_name in layer_fields:
        if field_name.startswith("ic_"):
            continue

        field_data = raw_data.get(field_name, {})
        confidence = field_data.get("confidence", 1.0)

        if confidence < stale_threshold:
            stale_fields.append(
                {
                    "field": field_name,
                    "confidence": confidence,
                    "impact": "Sub-score halved due to data staleness",
                }
            )

    return stale_fields


def check_anomalies(
    layer_fields: list,
    raw_data: dict,
) -> tuple:
    """
    Passes through anomaly flags from Data Collector.

    Returns: (anomaly_list, divergence_list)
    """
    anomaly_list = []
    divergence_list = []

    for field_name in layer_fields:
        if field_name.startswith("ic_"):
            continue

        field_data = raw_data.get(field_name, {})
        anomaly_flag = field_data.get("anomaly_flag", "OK")

        if anomaly_flag == "ANOMALY":
            anomaly_list.append(field_name)
        elif anomaly_flag == "DIVERGENT":
            divergence_list.append(field_name)

    return anomaly_list, divergence_list


def determine_signal_quality(
    layer_name: str,
    layer_fields: list,
    raw_data: dict,
    signal_quality_config: dict,
    data_clarity: float,
) -> dict:
    """
    Master function: runs all quality checks and returns status.

    Returns: {
        "status": "CONFIRMED" | "SUSPICIOUS" | "DEGRADED",
        "reason": str | None,
        "suppression_checks": [...],
        "stale_fields": [...],
        "anomaly_fields": [...],
        "data_clarity": float
    }
    """
    suppression_rules = signal_quality_config.get("suppression_checks", [])
    stale_threshold = signal_quality_config.get("data_freshness", {}).get(
        "stale_threshold", 0.5
    )
    degraded_config = signal_quality_config.get("degraded_threshold", {})

    # Run checks
    suppressions = check_signal_suppression(layer_name, raw_data, suppression_rules)
    stale_fields = check_data_freshness(layer_fields, raw_data, stale_threshold)
    anomaly_fields, divergence_fields = check_anomalies(layer_fields, raw_data)

    # Determine status
    status = "CONFIRMED"
    reason = None

    # Check for SUSPICIOUS (suppression detected)
    if suppressions:
        status = "SUSPICIOUS"
        reason = suppressions[0]["reason"]

    # Check for DEGRADED (too many stale or anomalous fields)
    data_field_count = len([f for f in layer_fields if not f.startswith("ic_")])
    stale_pct = len(stale_fields) / data_field_count if data_field_count > 0 else 0
    max_stale_pct = degraded_config.get("stale_pct", 0.3)
    max_anomaly_count = degraded_config.get("anomaly_count", 1)

    if stale_pct > max_stale_pct or len(anomaly_fields) > max_anomaly_count:
        status = "DEGRADED"
        reason = (
            f"{len(stale_fields)} stale fields ({stale_pct:.0%}), "
            f"{len(anomaly_fields)} anomalies"
        )

    return {
        "status": status,
        "reason": reason,
        "suppression_checks": suppressions,
        "stale_fields": stale_fields,
        "anomaly_fields": anomaly_fields,
        "data_clarity": data_clarity,
    }


def apply_staleness_penalty(sub_score: int, confidence: float, threshold: float = 0.5) -> int:
    """
    Halves a sub-score if field confidence is below threshold.
    Called during normalization phase.

    Source: AGENT2_SPEC_TEIL3 Section 8.2
    """
    if confidence < threshold:
        return sub_score // 2  # Integer division
    return sub_score


# --- Internal helpers ---


def _evaluate_test(field_data: dict, test: str) -> bool:
    """
    Evaluates a simple test expression against field data.
    Supports: "pctl < 30", "value > 60", "direction == 'UP'"
    """
    parts = test.split()
    if len(parts) != 3:
        return False

    field_key, operator, value_str = parts

    actual = field_data.get(field_key)
    if actual is None:
        # Try common aliases
        if field_key == "pctl":
            actual = field_data.get("pctl_1y")
        if actual is None:
            return False

    # Parse expected value
    value_str = value_str.strip("'\"")
    try:
        expected = float(value_str)
        is_numeric = True
    except ValueError:
        expected = value_str
        is_numeric = False

    # Evaluate
    if is_numeric:
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            return False

        if operator == "<":
            return actual < expected
        elif operator == ">":
            return actual > expected
        elif operator == "<=":
            return actual <= expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "==":
            return actual == expected
    else:
        if operator == "==":
            return str(actual) == expected
        elif operator == "!=":
            return str(actual) != expected

    return False
