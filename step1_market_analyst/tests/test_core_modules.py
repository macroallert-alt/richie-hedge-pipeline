"""
Unit tests for Market Analyst core modules (Etappe 2).
Tests normalization, layer_calculator, signal_phase, signal_quality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.normalization import (
    pctl_to_score,
    threshold_to_score,
    nonlinear_extreme_score,
    direction_to_score,
    normalize_field,
)
from modules.layer_calculator import (
    calculate_layer_score,
    assign_regime,
    calculate_data_clarity,
    get_layer_id,
)
from modules.signal_phase import detect_signal_phase
from modules.signal_quality import (
    apply_staleness_penalty,
    check_signal_suppression,
    check_data_freshness,
    determine_signal_quality,
    _evaluate_test,
)

import json

# ============================================================
# NORMALIZATION TESTS
# ============================================================

def test_pctl_to_score():
    assert pctl_to_score(50) == 0, "pctl 50 should be 0"
    assert pctl_to_score(100) == 10, "pctl 100 should be +10"
    assert pctl_to_score(0) == -10, "pctl 0 should be -10"
    assert pctl_to_score(75) == 5, "pctl 75 should be +5"
    assert pctl_to_score(25) == -5, "pctl 25 should be -5"
    # Inverted
    assert pctl_to_score(75, invert=True) == -5, "pctl 75 inverted should be -5"
    assert pctl_to_score(25, invert=True) == 5, "pctl 25 inverted should be +5"
    # None
    assert pctl_to_score(None) == 0, "None should return 0"
    print("  pctl_to_score: PASS")


def test_threshold_to_score():
    # spread_2y10y: neutral=0, bullish=1.5, bearish=-1.0
    assert threshold_to_score(0, 0, 1.5, -1.0) == 0, "At neutral should be 0"
    assert threshold_to_score(1.5, 0, 1.5, -1.0) == 10, "At bullish extreme should be +10"
    assert threshold_to_score(-1.0, 0, 1.5, -1.0) == -10, "At bearish extreme should be -10"
    assert threshold_to_score(0.75, 0, 1.5, -1.0) == 5, "Halfway to bullish should be +5"
    # pct_above_200dma: neutral=50, bullish=80, bearish=20
    assert threshold_to_score(50, 50, 80, 20) == 0, "200dma at 50 should be 0"
    assert threshold_to_score(80, 50, 80, 20) == 10, "200dma at 80 should be +10"
    assert threshold_to_score(20, 50, 80, 20) == -10, "200dma at 20 should be -10"
    # wti_curve: neutral=1.0, bullish=0.9, bearish=1.1 (backwardation bullish)
    # Inverted range: bullish < neutral < bearish
    assert threshold_to_score(0.9, 1.0, 0.9, 1.1) == 10, "WTI 0.9 (backwardation = bullish extreme) should be +10"
    assert threshold_to_score(1.1, 1.0, 0.9, 1.1) == -10, "WTI 1.1 (contango = bearish extreme) should be -10"
    assert threshold_to_score(1.0, 1.0, 0.9, 1.1) == 0, "WTI 1.0 (neutral) should be 0"
    # None
    assert threshold_to_score(None, 0, 1.5, -1.0) == 0, "None should return 0"
    print("  threshold_to_score: PASS")


def test_nonlinear_extreme():
    # NAAIM contrarian: high pctl = negative (crowding)
    assert nonlinear_extreme_score(95, 25, 75, contrarian=True) == -8, "NAAIM 95th contrarian should be ~-8"
    assert nonlinear_extreme_score(5, 25, 75, contrarian=True) == 8, "NAAIM 5th contrarian should be ~+8"
    assert nonlinear_extreme_score(55, 25, 75, contrarian=True) == 0, "NAAIM 55th should be 0 (dead zone)"
    assert nonlinear_extreme_score(25, 25, 75, contrarian=True) == 0, "At dead zone boundary = 0"
    assert nonlinear_extreme_score(75, 25, 75, contrarian=True) == 0, "At dead zone boundary = 0"
    # VIX inverted (not contrarian, but inverted): high = bad
    assert nonlinear_extreme_score(90, 20, 70, contrarian=False, invert=True) < 0, "VIX 90th inverted should be negative"
    assert nonlinear_extreme_score(10, 20, 70, contrarian=False, invert=True) > 0, "VIX 10th inverted should be positive"
    assert nonlinear_extreme_score(50, 20, 70) == 0, "VIX 50th in dead zone = 0"
    # None
    assert nonlinear_extreme_score(None) == 0, "None should return 0"
    print("  nonlinear_extreme_score: PASS")


def test_direction_to_score():
    assert direction_to_score("UP", 0) == 3, "UP with no momentum = 3"
    assert direction_to_score("DOWN", 0) == -3, "DOWN with no momentum = -3"
    assert direction_to_score("FLAT", 0) == 0, "FLAT = 0"
    assert direction_to_score("UP", 3) == 6, "UP with momentum 3 = 6"
    assert direction_to_score("DOWN", 5) == -6, "DOWN with capped momentum = -6"
    print("  direction_to_score: PASS")


def test_normalize_field_dispatcher():
    # pctl method
    assert normalize_field({"pctl_1y": 75}, {"method": "pctl", "invert": False}) == 5
    # threshold method
    result = normalize_field(
        {"value": 0.75},
        {"method": "threshold", "neutral": 0, "bullish_extreme": 1.5, "bearish_extreme": -1.0, "invert": False},
    )
    assert result == 5, f"threshold dispatch: expected 5, got {result}"
    # nonlinear method
    result = normalize_field(
        {"pctl_1y": 55},
        {"method": "nonlinear_extreme", "dead_zone_low": 25, "dead_zone_high": 75, "contrarian": True, "invert": False},
    )
    assert result == 0, f"nonlinear dead zone: expected 0, got {result}"
    print("  normalize_field dispatcher: PASS")


# ============================================================
# LAYER CALCULATOR TESTS
# ============================================================

def test_calculate_layer_score():
    sub_scores = {"net_liquidity": 7, "walcl": 2, "rrp": 6, "tga": 1, "mmf_assets": -2}
    weights = {
        "net_liquidity": {"risk_on": "PRIMARY", "risk_off": "PRIMARY"},
        "walcl": {"risk_on": "SECONDARY", "risk_off": "SECONDARY"},
        "rrp": {"risk_on": "SECONDARY", "risk_off": "SECONDARY"},
        "tga": {"risk_on": "CONTEXTUAL", "risk_off": "CONTEXTUAL"},
        "mmf_assets": {"risk_on": "CONTEXTUAL", "risk_off": "CONTEXTUAL"},
    }
    score = calculate_layer_score(sub_scores, weights, "Risk-On")
    # Manual: (7*3 + 2*2 + 6*2 + 1*1 + (-2)*1) / (3+2+2+1+1) = (21+4+12+1-2)/9 = 36/9 = 4
    assert score == 4, f"L1 Risk-On score: expected 4, got {score}"

    score_off = calculate_layer_score(sub_scores, weights, "Risk-Off")
    # Same weights for L1, so same result
    assert score_off == 4, f"L1 Risk-Off score: expected 4, got {score_off}"
    print("  calculate_layer_score: PASS")


def test_assign_regime():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "layer_regimes.json")
    with open(config_path) as f:
        regimes = json.load(f)

    # L1 tests
    assert assign_regime(5, "Global Liquidity Cycle (L1)", regimes) == "EXPANSION"
    assert assign_regime(0, "Global Liquidity Cycle (L1)", regimes) == "TRANSITION"
    assert assign_regime(-4, "Global Liquidity Cycle (L1)", regimes) == "TIGHTENING"
    assert assign_regime(-8, "Global Liquidity Cycle (L1)", regimes) == "DRAIN"

    # L5 contrarian tests
    assert assign_regime(7, "Risk Appetite & Sentiment (L5)", regimes) == "CAPITULATION"
    assert assign_regime(-7, "Risk Appetite & Sentiment (L5)", regimes) == "EUPHORIA"
    assert assign_regime(0, "Risk Appetite & Sentiment (L5)", regimes) == "NEUTRAL"

    # L8 inverted tests
    assert assign_regime(5, "Tail Risk & Black Swan (L8)", regimes) == "CALM"
    assert assign_regime(-5, "Tail Risk & Black Swan (L8)", regimes) == "ACUTE"
    assert assign_regime(-9, "Tail Risk & Black Swan (L8)", regimes) == "CRISIS"

    # L2 RECOVERY special case
    assert assign_regime(-3, "Macro Regime (L2)", regimes, direction="IMPROVING") == "RECOVERY"
    assert assign_regime(-3, "Macro Regime (L2)", regimes, direction="DETERIORATING") == "RECESSION"

    print("  assign_regime: PASS")


def test_data_clarity():
    # All positive
    assert calculate_data_clarity({"a": 5, "b": 3, "c": 7}) == 1.0
    # All negative
    assert calculate_data_clarity({"a": -5, "b": -3}) == 1.0
    # Mixed: 2 positive, 2 negative = 0.0 agreement
    assert calculate_data_clarity({"a": 5, "b": -3, "c": 7, "d": -1}) == 0.0
    # Mostly positive: 3 pos, 1 neg = 0.5 agreement
    clarity = calculate_data_clarity({"a": 5, "b": -3, "c": 7, "d": 2})
    assert 0.0 < clarity < 1.0, f"3 pos 1 neg should be between 0 and 1, got {clarity}"
    # All zero
    assert calculate_data_clarity({"a": 0, "b": 0}) == 0.5
    print("  data_clarity: PASS")


def test_get_layer_id():
    assert get_layer_id("Global Liquidity Cycle (L1)") == "L1"
    assert get_layer_id("Tail Risk & Black Swan (L8)") == "L8"
    print("  get_layer_id: PASS")


# ============================================================
# SIGNAL PHASE TESTS
# ============================================================

def test_signal_phase():
    fp = {
        "spread_2y10y": {"timing": "LEADING"},
        "hy_oas": {"timing": "LEADING"},
        "naaim_exposure": {"timing": "LAGGING"},
        "aaii_bull_bear": {"timing": "LAGGING"},
        "vix": {"timing": "COINCIDENT"},
    }
    # EARLY_SIGNAL: leading turned, lagging hasn't
    scores = {"spread_2y10y": -5, "hy_oas": -4, "naaim_exposure": 0, "aaii_bull_bear": 0}
    assert detect_signal_phase(scores, {}, fp) == "EARLY_SIGNAL"

    # CONFIRMED: both agree
    scores = {"spread_2y10y": -5, "hy_oas": -4, "naaim_exposure": -3, "aaii_bull_bear": -2}
    assert detect_signal_phase(scores, {}, fp) == "CONFIRMED"

    # CONFLICTED: leading says down, lagging says up
    scores = {"spread_2y10y": -5, "hy_oas": -4, "naaim_exposure": 5, "aaii_bull_bear": 4}
    assert detect_signal_phase(scores, {}, fp) == "CONFLICTED"

    # NO_SIGNAL: no leading scores
    scores = {"naaim_exposure": 3, "aaii_bull_bear": 2}
    assert detect_signal_phase(scores, {}, fp) == "NO_SIGNAL"

    print("  signal_phase: PASS")


# ============================================================
# SIGNAL QUALITY TESTS
# ============================================================

def test_staleness_penalty():
    assert apply_staleness_penalty(6, 0.3) == 3, "Score 6 at conf 0.3 should halve to 3"
    assert apply_staleness_penalty(7, 0.8) == 7, "Score 7 at conf 0.8 should stay 7"
    assert apply_staleness_penalty(-5, 0.4) == -3, "Score -5 at conf 0.4 should halve to -3"
    print("  staleness_penalty: PASS")


def test_evaluate_test():
    field = {"pctl_1y": 25, "value": -0.5, "direction": "UP"}
    assert _evaluate_test(field, "pctl_1y < 30") is True
    assert _evaluate_test(field, "pctl_1y > 30") is False
    assert _evaluate_test(field, "value < 0") is True
    assert _evaluate_test(field, "direction == UP") is True
    assert _evaluate_test(field, "direction == DOWN") is False
    print("  _evaluate_test: PASS")


def test_suppression_detection():
    rules = [
        {
            "id": "VIX_SUPPRESSION",
            "layer": "Tail Risk & Black Swan (L8)",
            "conditions": {
                "vix": {"field": "vix", "test": "pctl_1y < 30"},
                "pc_ratio": {"field": "pc_ratio_equity", "test": "pctl_1y < 20"},
                "iv_rv": {"field": "iv_rv_spread", "test": "value < 0"},
            },
            "all_must_match": True,
            "quality": "SUSPICIOUS",
            "reason": "VIX suppressed",
            "true_risk": "ELEVATED",
            "affected_sub_scores": ["vix", "pc_ratio_equity", "iv_rv_spread"],
        }
    ]
    # All conditions met
    raw = {
        "vix": {"pctl_1y": 15, "value": 13},
        "pc_ratio_equity": {"pctl_1y": 10},
        "iv_rv_spread": {"value": -2},
    }
    result = check_signal_suppression("Tail Risk & Black Swan (L8)", raw, rules)
    assert len(result) == 1, "Should detect VIX suppression"

    # One condition NOT met
    raw["vix"]["pctl_1y"] = 50
    result = check_signal_suppression("Tail Risk & Black Swan (L8)", raw, rules)
    assert len(result) == 0, "Should NOT detect suppression"

    print("  suppression_detection: PASS")


def test_data_freshness():
    fields = ["net_liquidity", "walcl", "rrp"]
    raw = {
        "net_liquidity": {"confidence": 1.0},
        "walcl": {"confidence": 0.3},
        "rrp": {"confidence": 0.8},
    }
    stale = check_data_freshness(fields, raw, 0.5)
    assert len(stale) == 1, f"Expected 1 stale field, got {len(stale)}"
    assert stale[0]["field"] == "walcl"
    print("  data_freshness: PASS")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("=== Normalization Tests ===")
    test_pctl_to_score()
    test_threshold_to_score()
    test_nonlinear_extreme()
    test_direction_to_score()
    test_normalize_field_dispatcher()

    print("\n=== Layer Calculator Tests ===")
    test_calculate_layer_score()
    test_assign_regime()
    test_data_clarity()
    test_get_layer_id()

    print("\n=== Signal Phase Tests ===")
    test_signal_phase()

    print("\n=== Signal Quality Tests ===")
    test_staleness_penalty()
    test_evaluate_test()
    test_suppression_detection()
    test_data_freshness()

    print("\n=== ALL TESTS PASSED ===")
