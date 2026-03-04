"""
Market Analyst — Cross-Layer Checks Module
Detects divergences between layers. Where alpha lives.

If all layers agree, you don't need a system — everyone sees the same thing.
Agent 2 actively searches for CONTRADICTIONS.

Source: AGENT2_SPEC_TEIL5 Section 15
"""

import re


def run_cross_checks(layer_results: dict, cross_check_config: list) -> list:
    """
    Evaluates all cross-check rules against current layer results.

    layer_results: {layer_name: {"score": int, "regime": str, "direction": str, ...}}
    cross_check_config: list from cross_checks.json

    Returns: list of active flags with tension descriptions and consequences.
    """
    active_flags = []

    for check in cross_check_config:
        cond = check["condition"]
        layer_a_name = cond["layer_a"]
        layer_b_name = cond["layer_b"]

        layer_a = layer_results.get(layer_a_name)
        layer_b = layer_results.get(layer_b_name)

        if layer_a is None or layer_b is None:
            continue

        if _evaluate_condition(layer_a, cond["layer_a_test"]) and \
           _evaluate_condition(layer_b, cond["layer_b_test"]):

            flag = {
                "check_id": check["id"],
                "name": check["name"],
                "tension": check["tension"],
                "layers_involved": [layer_a_name, layer_b_name],
                "consequence": check["consequence"],
                "precedent": check.get("historical_precedent"),
            }
            active_flags.append(flag)

            # Execute conviction downgrade if defined
            if "downgrade_conviction" in check["consequence"]:
                target_layer = check["consequence"]["downgrade_conviction"]
                if target_layer in layer_results:
                    conviction = layer_results[target_layer].get("conviction", {})
                    conviction["composite"] = check["consequence"]["to"]
                    conviction["limiting_factor"] = {
                        "factor": "cross_layer_check",
                        "value": 0.0,
                        "label": check["tension"],
                    }

    return active_flags


def _evaluate_condition(layer_data: dict, test_str: str) -> bool:
    """
    Evaluates a condition string against layer data.
    Supports: "score < -5", "regime == 'EUPHORIA'",
              "score < -4 AND direction == 'DETERIORATING'",
              "regime == 'OUTFLOW' OR regime == 'SQUEEZE'"
    """
    # Handle AND
    if " AND " in test_str:
        parts = test_str.split(" AND ")
        return all(_evaluate_single(layer_data, p.strip()) for p in parts)

    # Handle OR
    if " OR " in test_str:
        parts = test_str.split(" OR ")
        return any(_evaluate_single(layer_data, p.strip()) for p in parts)

    return _evaluate_single(layer_data, test_str)


def _evaluate_single(layer_data: dict, test: str) -> bool:
    """Evaluates a single comparison: 'field op value'."""
    # Parse: field operator value
    match = re.match(r"(\w+)\s*(==|!=|<|>|<=|>=)\s*(.+)", test.strip())
    if not match:
        return False

    field, op, value_str = match.groups()
    value_str = value_str.strip().strip("'\"")

    actual = layer_data.get(field)
    if actual is None:
        return False

    # Try numeric comparison
    try:
        expected = float(value_str)
        actual_num = float(actual)
        if op == "<":
            return actual_num < expected
        elif op == ">":
            return actual_num > expected
        elif op == "<=":
            return actual_num <= expected
        elif op == ">=":
            return actual_num >= expected
        elif op == "==":
            return actual_num == expected
        elif op == "!=":
            return actual_num != expected
    except (ValueError, TypeError):
        pass

    # String comparison
    if op == "==":
        return str(actual) == value_str
    elif op == "!=":
        return str(actual) != value_str

    return False
