"""
Unit tests for Market Analyst Etappe 3 modules.
Tests dynamics, surprise, transitions, conviction.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dynamics import (
    calculate_velocity,
    calculate_acceleration,
    calculate_direction,
    get_score_n_days_ago,
    get_historical_daily_deltas,
    extract_layer_history,
)
from modules.surprise import calculate_surprise, apply_surprise_to_conviction
from modules.transitions import (
    calculate_regime_history,
    calculate_regime_duration_score,
    calculate_transition_proximity,
)
from modules.conviction import (
    calculate_conviction,
    categorize_conviction,
    find_limiting_factor,
    calculate_narrative_alignment,
    calculate_catalyst_fragility,
)

import json


# ============================================================
# MOCK DATA HELPERS
# ============================================================

def make_history(scores, regime="EXPANSION", layer_name="Global Liquidity Cycle (L1)"):
    """Build mock 30-day history from a list of scores."""
    return [
        {
            "date": f"2026-02-{i+1:02d}",
            "layers": {
                layer_name: {"score": s, "regime": regime}
            },
        }
        for i, s in enumerate(scores)
    ]


def make_history_with_regimes(score_regime_pairs, layer_name="Global Liquidity Cycle (L1)"):
    """Build mock history with varying regimes."""
    return [
        {
            "date": f"2026-02-{i+1:02d}",
            "layers": {
                layer_name: {"score": s, "regime": r}
            },
        }
        for i, (s, r) in enumerate(score_regime_pairs)
    ]


# ============================================================
# DYNAMICS TESTS
# ============================================================

def test_velocity():
    assert calculate_velocity(5, 4, 1) == "ACCELERATING", "delta_5d=4 should be ACCELERATING"
    assert calculate_velocity(1, 2, 5) == "DECELERATING", "delta_5d=-4 should be DECELERATING"
    assert calculate_velocity(5, 4, 3) == "MOVING", "delta_5d=2 should be MOVING"
    assert calculate_velocity(5, 5, 5) == "STEADY", "delta_5d=0 should be STEADY"
    assert calculate_velocity(5, None, None) == "STEADY", "None history = STEADY"
    print("  velocity: PASS")


def test_acceleration():
    assert calculate_acceleration(4, 1) == "STRONGLY_ACCELERATING", "accel=3 should be STRONGLY_ACCELERATING"
    assert calculate_acceleration(-4, -1) == "STRONGLY_DECELERATING", "accel=-3 should be STRONGLY_DECELERATING"
    assert calculate_acceleration(3, 2) == "MILDLY_ACCELERATING", "accel=1 should be MILDLY_ACCELERATING"
    assert calculate_acceleration(2, 2) == "FLAT", "accel=0 should be FLAT"
    assert calculate_acceleration(None, None) == "FLAT", "None = FLAT"
    print("  acceleration: PASS")


def test_direction():
    assert calculate_direction(5, 3, 1) == "IMPROVING", "both trends up = IMPROVING"
    assert calculate_direction(1, 3, 5) == "DETERIORATING", "both trends down = DETERIORATING"
    assert calculate_direction(3, 1, 5) == "RECOVERING", "5d up, 21d down = RECOVERING"
    assert calculate_direction(3, 5, 1) == "WEAKENING", "5d down, 21d up = WEAKENING"
    assert calculate_direction(3, 3, 3) == "STABLE", "no change = STABLE"
    print("  direction: PASS")


def test_history_helpers():
    history = [{"score": i} for i in range(10)]  # 0,1,2,...,9
    assert get_score_n_days_ago(history, 0) == 9, "0 days ago = last"
    assert get_score_n_days_ago(history, 1) == 8, "1 day ago"
    assert get_score_n_days_ago(history, 9) == 0, "9 days ago = first"
    assert get_score_n_days_ago(history, 10) is None, "not enough history"

    deltas = get_historical_daily_deltas(history)
    assert len(deltas) == 9, "9 deltas from 10 entries"
    assert all(d == 1 for d in deltas), "all deltas should be 1"
    print("  history_helpers: PASS")


def test_extract_layer_history():
    ln = "Global Liquidity Cycle (L1)"
    hist = make_history([3, 4, 5], layer_name=ln)
    extracted = extract_layer_history(hist, ln)
    assert len(extracted) == 3
    assert extracted[0]["score"] == 3
    assert extracted[2]["score"] == 5
    print("  extract_layer_history: PASS")


# ============================================================
# SURPRISE TESTS
# ============================================================

def test_surprise():
    # Normal delta in stable history
    deltas = [1, -1, 0, 1, -1, 0, 1, -1, 0, 1]  # mean ~0.1, std ~0.83
    result = calculate_surprise(5, 4, deltas)  # delta=1, close to mean
    assert result["category"] == "NORMAL", f"Expected NORMAL, got {result}"

    # Extreme delta
    deltas = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # mean=0.1, std~0.3
    result = calculate_surprise(5, 0, deltas)  # delta=5, way above
    assert result["category"] == "EXTREME", f"Expected EXTREME, got {result}"

    # Not enough history
    result = calculate_surprise(5, 4, [1, 2])
    assert result["category"] == "NORMAL", "Not enough history = NORMAL"

    # None yesterday
    result = calculate_surprise(5, None, [1, 2, 3, 4, 5])
    assert result["category"] == "NORMAL", "None yesterday = NORMAL"
    print("  surprise: PASS")


def test_surprise_conviction_cap():
    assert apply_surprise_to_conviction("EXTREME", 0.8) == 0.4
    assert apply_surprise_to_conviction("HIGH", 0.8) == 0.6
    assert apply_surprise_to_conviction("HIGH", 0.3) == 0.3, "Don't raise, only cap"
    assert apply_surprise_to_conviction("NORMAL", 0.9) == 0.9, "No cap for NORMAL"
    print("  surprise_conviction_cap: PASS")


# ============================================================
# TRANSITIONS TESTS
# ============================================================

def test_regime_history():
    ln = "Global Liquidity Cycle (L1)"
    # Stable regime for 10 days
    hist = make_history([3, 4, 5, 4, 5, 4, 5, 4, 5, 4], regime="EXPANSION", layer_name=ln)
    rh = calculate_regime_history(hist, ln)
    assert rh["current_regime"] == "EXPANSION"
    assert rh["duration_days"] == 10
    assert rh["regime_changes_30d"] == 0
    assert rh["oscillation_flag"] is False
    assert rh["chaotic_flag"] is False

    # Oscillating regime
    pairs = [
        (3, "EXPANSION"), (-3, "TIGHTENING"),
        (3, "EXPANSION"), (-3, "TIGHTENING"),
        (3, "EXPANSION"), (-3, "TIGHTENING"),
    ]
    hist = make_history_with_regimes(pairs, layer_name=ln)
    rh = calculate_regime_history(hist, ln)
    assert rh["oscillation_flag"] is True, "Should detect oscillation"
    assert rh["regime_changes_30d"] == 5

    # Chaotic
    pairs = [
        (3, "EXPANSION"), (-3, "TIGHTENING"), (0, "TRANSITION"),
        (3, "EXPANSION"), (-8, "DRAIN"), (0, "TRANSITION"),
        (3, "EXPANSION"),
    ]
    hist = make_history_with_regimes(pairs, layer_name=ln)
    rh = calculate_regime_history(hist, ln)
    assert rh["chaotic_flag"] is True, f"6 changes should be CHAOTIC, got {rh['regime_changes_30d']}"
    print("  regime_history: PASS")


def test_regime_duration_score():
    # Stable: 60 days, no changes
    assert calculate_regime_duration_score(
        {"duration_days": 60, "regime_changes_30d": 0, "oscillation_flag": False}
    ) == 1.0

    # Young: 3 days
    score = calculate_regime_duration_score(
        {"duration_days": 3, "regime_changes_30d": 0, "oscillation_flag": False}
    )
    assert score == 0.2, f"3 days should be 0.2, got {score}"

    # Chaotic: 15 days but 4 changes
    score = calculate_regime_duration_score(
        {"duration_days": 15, "regime_changes_30d": 4, "oscillation_flag": False}
    )
    assert score == 0.3, f"15d + 4 changes = 0.6*0.5=0.3, got {score}"

    # Oscillating
    score = calculate_regime_duration_score(
        {"duration_days": 30, "regime_changes_30d": 2, "oscillation_flag": True}
    )
    expected = round(0.8 * 0.7 * 0.6, 2)  # 30d * 2changes * oscillation
    assert score == expected, f"Expected {expected}, got {score}"
    print("  regime_duration_score: PASS")


def test_transition_proximity():
    config = {
        "regimes": {
            "EXPANSION": {"score_min": 3},
            "TRANSITION": {"score_min": -2, "score_max": 3},
            "TIGHTENING": {"score_min": -6, "score_max": -2},
            "DRAIN": {"score_max": -6},
        },
        "regime_order": ["DRAIN", "TIGHTENING", "TRANSITION", "EXPANSION"],
    }

    # Score 3, right at boundary of EXPANSION
    result = calculate_transition_proximity(3, "EXPANSION", config, "STEADY", "FLAT")
    assert result["proximity"] > 0.5, f"Score at boundary should have high proximity, got {result['proximity']}"
    assert result["target_regime"] == "TRANSITION"
    assert result["target_direction"] == "DOWN"

    # Score 7, far from boundary
    result = calculate_transition_proximity(7, "EXPANSION", config, "STEADY", "FLAT")
    assert result["proximity"] < 0.5, f"Score far from boundary should have low proximity, got {result['proximity']}"

    # Moving toward boundary boosts proximity
    result_steady = calculate_transition_proximity(4, "EXPANSION", config, "STEADY", "FLAT")
    result_moving = calculate_transition_proximity(4, "EXPANSION", config, "DECELERATING", "FLAT")
    assert result_moving["proximity"] >= result_steady["proximity"], "Moving toward boundary should boost proximity"
    print("  transition_proximity: PASS")


# ============================================================
# CONVICTION TESTS
# ============================================================

def test_categorize_conviction():
    assert categorize_conviction(0.8, 0.9, 0.7, 0.8) == "HIGH", "all >= 0.7 = HIGH"
    assert categorize_conviction(0.8, 0.9, 0.5, 0.8) == "MEDIUM", "min 0.5 = MEDIUM"
    assert categorize_conviction(0.8, 0.3, 0.7, 0.8) == "LOW", "min 0.3 = LOW"
    assert categorize_conviction(0.8, 0.1, 0.7, 0.8) == "CONFLICTED", "min 0.1 = CONFLICTED"
    print("  categorize_conviction: PASS")


def test_limiting_factor():
    result = find_limiting_factor(0.8, 0.3, 0.9, 0.7)
    assert result["factor"] == "narrative_alignment", f"Weakest should be narrative, got {result['factor']}"
    assert result["value"] == 0.3

    result = find_limiting_factor(0.8, 0.9, 0.2, 0.7)
    assert result["factor"] == "catalyst_fragility"
    print("  limiting_factor: PASS")


def test_narrative_alignment():
    # IC confirms, signal confirmed
    assert calculate_narrative_alignment("CONFIRMING", False, "CONFIRMED") == 0.9
    # IC contradicts
    assert calculate_narrative_alignment("CONTRADICTING", False, "CONFIRMED") == 0.2
    # IC confirms but with dissent
    na = calculate_narrative_alignment("CONFIRMING", True, "CONFIRMED")
    assert na == round(0.9 * 0.7, 2), f"Expected {round(0.9*0.7, 2)}, got {na}"
    # Early signal reduces
    na = calculate_narrative_alignment("CONFIRMING", False, "EARLY_SIGNAL")
    assert na == round(0.9 * 0.8, 2), f"Expected {round(0.9*0.8, 2)}, got {na}"
    # No IC data
    assert calculate_narrative_alignment("NO_DATA", False, "CONFIRMED") == 0.5
    print("  narrative_alignment: PASS")


def test_catalyst_fragility():
    # No catalysts = robust
    assert calculate_catalyst_fragility([]) == 1.0
    # FOMC tomorrow
    cf = calculate_catalyst_fragility([{"days_until": 1, "direction": "BINARY"}])
    assert cf == 0.1, f"FOMC tomorrow should be 0.1, got {cf}"
    # FOMC in 3 days
    cf = calculate_catalyst_fragility([{"days_until": 3, "direction": "BINARY"}])
    assert cf == 0.3, f"FOMC in 3d should be 0.3, got {cf}"
    # OpEx tomorrow (DIRECTIONAL)
    cf = calculate_catalyst_fragility([{"days_until": 1, "direction": "DIRECTIONAL"}])
    assert cf == 0.3, f"OpEx tomorrow should be 0.3, got {cf}"
    # Multiple: worst wins
    cf = calculate_catalyst_fragility([
        {"days_until": 3, "direction": "BINARY"},     # 0.3
        {"days_until": 1, "direction": "DIRECTIONAL"},  # 0.3
    ])
    assert cf == 0.3
    print("  catalyst_fragility: PASS")


def test_full_conviction():
    """Integration test: full conviction calculation with mock layer data."""
    layer_data = {
        "raw_data_clarity": 0.85,
        "ic_confirmation": "CONFIRMING",
        "ic_dissent": False,
        "signal_phase": "CONFIRMED",
        "catalyst_exposure": [{"days_until": 3, "direction": "BINARY"}],
        "regime_history": {
            "duration_days": 25,
            "regime_changes_30d": 1,
            "oscillation_flag": False,
        },
        "surprise": {"category": "NORMAL"},
    }
    result = calculate_conviction(layer_data)

    assert result["data_clarity"] == 0.85
    assert result["narrative_alignment"] == 0.9
    assert result["catalyst_fragility"] == 0.3  # FOMC in 3d
    assert result["composite"] == "LOW"  # min is 0.3 (catalyst)
    assert result["limiting_factor"]["factor"] == "catalyst_fragility"
    print("  full_conviction: PASS")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("=== Dynamics Tests ===")
    test_velocity()
    test_acceleration()
    test_direction()
    test_history_helpers()
    test_extract_layer_history()

    print("\n=== Surprise Tests ===")
    test_surprise()
    test_surprise_conviction_cap()

    print("\n=== Transitions Tests ===")
    test_regime_history()
    test_regime_duration_score()
    test_transition_proximity()

    print("\n=== Conviction Tests ===")
    test_categorize_conviction()
    test_limiting_factor()
    test_narrative_alignment()
    test_catalyst_fragility()
    test_full_conviction()

    print("\n=== ALL ETAPPE 3 TESTS PASSED ===")
