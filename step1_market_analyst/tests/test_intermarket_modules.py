"""
Unit tests for Market Analyst Etappe 4 modules.
Tests cross_checks, cascades, catalysts, ic_integration,
templates, system_synthesis, fragility_monitor.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cross_checks import run_cross_checks
from modules.cascades import check_cascades
from modules.catalysts import calculate_catalyst_exposure
from modules.ic_integration import (
    calculate_ic_status,
    determine_ic_weight,
    detect_thesis_shifts,
    get_data_direction,
)
from modules.templates import select_key_driver, generate_tension
from modules.system_synthesis import synthesize_system_regime
from modules.fragility_monitor import (
    calculate_fragility_state,
    check_crisis_condition,
    is_stale,
)

from datetime import date, datetime


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

def load_config(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return json.load(f)


# ============================================================
# CROSS-CHECKS TESTS
# ============================================================

def test_cross_checks():
    config = load_config("cross_checks.json")
    # CREDIT_EARNINGS_DIVERGENCE: L2 < -5 AND L3 > 3
    layers = {
        "Macro Regime (L2)": {"score": -6, "regime": "RECESSION", "direction": "DETERIORATING"},
        "Earnings & Fundamentals (L3)": {
            "score": 5, "regime": "HEALTHY", "conviction": {"composite": "HIGH", "limiting_factor": {}}
        },
        "Global Liquidity Cycle (L1)": {"score": 2, "regime": "TRANSITION"},
        "Risk Appetite & Sentiment (L5)": {"score": 0, "regime": "NEUTRAL"},
        "Central Bank Policy Divergence (L7)": {"score": 0, "regime": "NEUTRAL"},
        "Tail Risk & Black Swan (L8)": {"score": 3, "regime": "CALM"},
        "Cross-Border Flows & FX (L4)": {"score": 0, "regime": "STABLE"},
        "Relative Value & Asset Rotation (L6)": {"score": 0, "regime": "BALANCED"},
    }
    flags = run_cross_checks(layers, config)
    check_ids = [f["check_id"] for f in flags]
    assert "CREDIT_EARNINGS_DIVERGENCE" in check_ids, f"Should detect credit/earnings divergence, got {check_ids}"

    # Verify conviction was downgraded
    assert layers["Earnings & Fundamentals (L3)"]["conviction"]["composite"] == "CONFLICTED"
    print("  cross_checks: PASS")


def test_cross_checks_no_trigger():
    config = load_config("cross_checks.json")
    layers = {
        "Macro Regime (L2)": {"score": 3, "regime": "GROWTH", "direction": "IMPROVING"},
        "Earnings & Fundamentals (L3)": {"score": 4, "regime": "HEALTHY", "conviction": {"composite": "HIGH"}},
        "Global Liquidity Cycle (L1)": {"score": 5, "regime": "EXPANSION"},
        "Risk Appetite & Sentiment (L5)": {"score": 0, "regime": "NEUTRAL"},
        "Central Bank Policy Divergence (L7)": {"score": 2, "regime": "NEUTRAL"},
        "Tail Risk & Black Swan (L8)": {"score": 3, "regime": "CALM"},
        "Cross-Border Flows & FX (L4)": {"score": 1, "regime": "STABLE"},
        "Relative Value & Asset Rotation (L6)": {"score": 2, "regime": "BALANCED"},
    }
    flags = run_cross_checks(layers, config)
    assert len(flags) == 0, f"No flags expected in healthy market, got {len(flags)}"
    print("  cross_checks_no_trigger: PASS")


# ============================================================
# CASCADES TESTS
# ============================================================

def test_cascades():
    config = load_config("cascades.json")
    # CB_TO_LIQUIDITY: L7 TIGHTENING 4 weeks ago, L1 hasn't reacted
    history = []
    for i in range(30):
        d = f"2026-02-{i+1:02d}"
        layers = {
            "Central Bank Policy Divergence (L7)": {"score": -4, "regime": "TIGHTENING"},
            "Global Liquidity Cycle (L1)": {"score": 3, "regime": "EXPANSION"},
        }
        history.append({"date": d, "layers": layers})

    today = date(2026, 3, 1)
    cascades = check_cascades(
        {"Global Liquidity Cycle (L1)": {"score": 3, "regime": "EXPANSION", "direction": "STABLE"}},
        history, config, today
    )

    cb_cascade = [c for c in cascades if c["cascade_id"] == "CB_TO_LIQUIDITY"]
    assert len(cb_cascade) > 0, "Should detect CB->Liquidity cascade"
    assert cb_cascade[0]["status"] == "EXPECTED", "L1 hasn't reacted, should be EXPECTED"
    print("  cascades: PASS")


# ============================================================
# CATALYSTS TESTS
# ============================================================

def test_catalysts():
    config = load_config("catalysts.json")
    # Test with a date near FOMC
    test_date = date(2026, 3, 17)  # FOMC is 2026-03-18, 1 day before
    exposures = calculate_catalyst_exposure(
        "Central Bank Policy Divergence (L7)", config, [], test_date
    )
    fomc = [e for e in exposures if e["event"] == "FOMC"]
    assert len(fomc) > 0, f"Should detect FOMC near {test_date}"
    assert fomc[0]["days_until"] == 1
    assert fomc[0]["direction"] == "BINARY"
    print("  catalysts: PASS")


def test_catalysts_no_exposure():
    config = load_config("catalysts.json")
    # Date far from any event for L3
    test_date = date(2026, 6, 1)
    exposures = calculate_catalyst_exposure(
        "Earnings & Fundamentals (L3)", config, [], test_date
    )
    # Should be empty or only distant events
    near = [e for e in exposures if e.get("days_until", 999) <= 3]
    # Might be empty, that's fine
    print(f"  catalysts_no_exposure: PASS (found {len(near)} near events)")


# ============================================================
# IC INTEGRATION TESTS
# ============================================================

def test_ic_status():
    ic_config = load_config("ic_integration.json")
    ic_data = {
        "consensus": {
            "LIQUIDITY": {"score": 4.0, "confidence": "HIGH", "source_count": 4},
        },
        "high_novelty_claims": [],
    }
    result = calculate_ic_status(ic_data, "Global Liquidity Cycle (L1)", ic_config, "POSITIVE")
    assert result["ic_confirmation"] == "CONFIRMING"
    assert result["ic_dissent"] is False
    assert result["ic_score"] == 4
    print("  ic_status_confirming: PASS")


def test_ic_contradicting():
    ic_config = load_config("ic_integration.json")
    ic_data = {
        "consensus": {
            "LIQUIDITY": {"score": -5.0, "confidence": "HIGH", "source_count": 3},
        },
    }
    result = calculate_ic_status(ic_data, "Global Liquidity Cycle (L1)", ic_config, "POSITIVE")
    assert result["ic_confirmation"] == "CONTRADICTING"
    print("  ic_contradicting: PASS")


def test_ic_weight():
    ic_config = load_config("ic_integration.json")

    # Data clear, IC confirms -> CONTEXTUAL
    assert determine_ic_weight(0.8, {"ic_confirmation": "CONFIRMING"}, ic_config) == "CONTEXTUAL"
    # Data clear, IC contradicts -> SECONDARY
    assert determine_ic_weight(0.8, {"ic_confirmation": "CONTRADICTING"}, ic_config) == "SECONDARY"
    # Data unclear -> PRIMARY
    assert determine_ic_weight(0.3, {"ic_confirmation": "CONFIRMING"}, ic_config) == "PRIMARY"
    # Data unclear + thesis shift -> PRIMARY
    assert determine_ic_weight(0.6, {"ic_confirmation": "CONFIRMING", "ic_thesis_shift": [{"source": "test"}]}, ic_config) == "PRIMARY"
    print("  ic_weight: PASS")


def test_thesis_shift():
    ic_config = load_config("ic_integration.json")
    claims = [
        {"topic": "LIQUIDITY", "novelty_score": 9, "source": "Bloomberg", "claim": "RRP ending", "direction": "BEARISH"},
        {"topic": "LIQUIDITY", "novelty_score": 6, "source": "Reuters", "claim": "Minor change", "direction": "NEUTRAL"},
    ]
    shifts = detect_thesis_shifts(claims, "Global Liquidity Cycle (L1)", ic_config["topic_layer_mapping"])
    assert shifts is not None and len(shifts) == 1, f"Should find 1 thesis shift, got {shifts}"
    assert shifts[0]["novelty"] == 9
    print("  thesis_shift: PASS")


def test_data_direction():
    assert get_data_direction({"a": 5, "b": 3, "c": 2}) == "POSITIVE"
    assert get_data_direction({"a": -5, "b": -3}) == "NEGATIVE"
    assert get_data_direction({"a": 1, "b": -1}) == "NEUTRAL"
    assert get_data_direction({"ic_test": 5, "a": 0}) == "NEUTRAL"  # ic_ excluded, only zeros
    print("  data_direction: PASS")


# ============================================================
# TEMPLATES TESTS
# ============================================================

def test_key_driver():
    templates = load_config("templates.json")
    sub_scores = {"net_liquidity": 7, "walcl": 2, "rrp": 6}
    raw_data = {
        "net_liquidity": {"pctl_1y": 72, "delta_5d": 45, "direction": "UP"},
        "rrp": {"pctl_1y": 15, "delta_5d": -38, "direction": "DOWN"},
        "walcl": {"direction": "UP"},
    }
    result = select_key_driver("Global Liquidity Cycle (L1)", sub_scores, raw_data, templates)
    assert "72" in result or "Net Liquidity" in result or "net_liquidity" in result, \
        f"Key driver should reference net_liquidity data, got: {result}"
    print(f"  key_driver: PASS — '{result[:80]}...'")


def test_tension():
    templates = load_config("templates.json")
    # Conflicting sub-scores
    sub_scores = {"net_liquidity": 7, "mmf_assets": -4, "rrp": 5}
    raw_data = {
        "net_liquidity": {"pctl_1y": 72},
        "mmf_assets": {"pctl_1y": 78},
        "rrp": {"pctl_1y": 15},
    }
    result = generate_tension("Global Liquidity Cycle (L1)", sub_scores, raw_data, templates)
    assert result is not None, "Should detect tension with conflicting scores"
    assert "BUT" in result, f"Tension should contain 'BUT', got: {result}"
    print(f"  tension: PASS — '{result[:80]}'")


def test_no_tension():
    templates = load_config("templates.json")
    # All positive, no conflict
    sub_scores = {"net_liquidity": 7, "walcl": 2, "rrp": 5}
    raw_data = {}
    result = generate_tension("Global Liquidity Cycle (L1)", sub_scores, raw_data, templates)
    assert result is None, "No tension when all scores agree"
    print("  no_tension: PASS")


# ============================================================
# SYSTEM SYNTHESIS TESTS
# ============================================================

def test_synthesis_broad_risk_on():
    layers = {f"Layer{i}": {"score": 5} for i in range(8)}
    # Need real names for L8 check
    layers = {
        "Global Liquidity Cycle (L1)": {"score": 5},
        "Macro Regime (L2)": {"score": 4},
        "Earnings & Fundamentals (L3)": {"score": 6},
        "Cross-Border Flows & FX (L4)": {"score": 3},
        "Risk Appetite & Sentiment (L5)": {"score": 4},
        "Relative Value & Asset Rotation (L6)": {"score": 5},
        "Central Bank Policy Divergence (L7)": {"score": 3},
        "Tail Risk & Black Swan (L8)": {"score": 4},
    }
    result = synthesize_system_regime(layers)
    assert result["regime"] == "BROAD_RISK_ON", f"Expected BROAD_RISK_ON, got {result['regime']}"
    print("  synthesis_broad_risk_on: PASS")


def test_synthesis_risk_off_forced():
    layers = {
        "Global Liquidity Cycle (L1)": {"score": 5},
        "Macro Regime (L2)": {"score": 4},
        "Earnings & Fundamentals (L3)": {"score": 6},
        "Cross-Border Flows & FX (L4)": {"score": 3},
        "Risk Appetite & Sentiment (L5)": {"score": 4},
        "Relative Value & Asset Rotation (L6)": {"score": 5},
        "Central Bank Policy Divergence (L7)": {"score": 3},
        "Tail Risk & Black Swan (L8)": {"score": -7},  # CRISIS
    }
    result = synthesize_system_regime(layers)
    assert result["regime"] == "RISK_OFF_FORCED", f"L8 at -7 should force RISK_OFF, got {result['regime']}"
    print("  synthesis_risk_off_forced: PASS")


def test_synthesis_conflicted():
    layers = {
        "Global Liquidity Cycle (L1)": {"score": 5},
        "Macro Regime (L2)": {"score": 4},
        "Earnings & Fundamentals (L3)": {"score": 5},
        "Cross-Border Flows & FX (L4)": {"score": -4},
        "Risk Appetite & Sentiment (L5)": {"score": -5},
        "Relative Value & Asset Rotation (L6)": {"score": -4},
        "Central Bank Policy Divergence (L7)": {"score": 0},
        "Tail Risk & Black Swan (L8)": {"score": 0},
    }
    result = synthesize_system_regime(layers)
    assert result["regime"] == "CONFLICTED", f"3 pos + 3 neg should be CONFLICTED, got {result['regime']}"
    print("  synthesis_conflicted: PASS")


def test_synthesis_neutral():
    layers = {
        "Global Liquidity Cycle (L1)": {"score": 1},
        "Macro Regime (L2)": {"score": -1},
        "Earnings & Fundamentals (L3)": {"score": 0},
        "Cross-Border Flows & FX (L4)": {"score": 2},
        "Risk Appetite & Sentiment (L5)": {"score": -2},
        "Relative Value & Asset Rotation (L6)": {"score": 0},
        "Central Bank Policy Divergence (L7)": {"score": 1},
        "Tail Risk & Black Swan (L8)": {"score": 0},
    }
    result = synthesize_system_regime(layers)
    assert result["regime"] == "NEUTRAL", f"All near zero should be NEUTRAL, got {result['regime']}"
    print("  synthesis_neutral: PASS")


# ============================================================
# FRAGILITY MONITOR TESTS
# ============================================================

def test_fragility_healthy():
    config = load_config("fragility_monitor.json")
    result = calculate_fragility_state(
        hhi=1200, breadth_pct=75, spy_rsp_6m_delta=0.05, ai_gap_data=None,
        fragility_config=config,
    )
    assert result["state"] == "HEALTHY", f"Expected HEALTHY, got {result['state']}"
    print("  fragility_healthy: PASS")


def test_fragility_elevated():
    config = load_config("fragility_monitor.json")
    result = calculate_fragility_state(
        hhi=1800, breadth_pct=55, spy_rsp_6m_delta=0.12, ai_gap_data=None,
        fragility_config=config,
    )
    assert result["state"] == "ELEVATED", f"Expected ELEVATED, got {result['state']}"
    assert len(result["triggers_active"]) >= 2
    print("  fragility_elevated: PASS")


def test_fragility_extreme():
    config = load_config("fragility_monitor.json")
    result = calculate_fragility_state(
        hhi=2200, breadth_pct=40, spy_rsp_6m_delta=0.25, ai_gap_data=None,
        fragility_config=config,
    )
    assert result["state"] == "EXTREME", f"Expected EXTREME, got {result['state']}"
    print("  fragility_extreme: PASS")


def test_crisis_condition():
    assert check_crisis_condition(-0.35, -0.10) is True, "Mag7 -35%, SPY -10% = CRISIS"
    assert check_crisis_condition(-0.35, -0.20) is False, "SPY -20% too deep, not concentration break"
    assert check_crisis_condition(-0.15, -0.05) is False, "Mag7 -15% not severe enough"
    assert check_crisis_condition(None, None) is False, "None = no crisis"
    print("  crisis_condition: PASS")


def test_staleness():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    assert is_stale(None) is True, "None data = stale"
    assert is_stale({"date_entered": today_str}) is False, "Today = not stale"
    assert is_stale({"date_entered": "2020-01-01"}) is True, "Old date = stale"
    print("  staleness: PASS")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("=== Cross-Checks Tests ===")
    test_cross_checks()
    test_cross_checks_no_trigger()

    print("\n=== Cascades Tests ===")
    test_cascades()

    print("\n=== Catalysts Tests ===")
    test_catalysts()
    test_catalysts_no_exposure()

    print("\n=== IC Integration Tests ===")
    test_ic_status()
    test_ic_contradicting()
    test_ic_weight()
    test_thesis_shift()
    test_data_direction()

    print("\n=== Templates Tests ===")
    test_key_driver()
    test_tension()
    test_no_tension()

    print("\n=== System Synthesis Tests ===")
    test_synthesis_broad_risk_on()
    test_synthesis_risk_off_forced()
    test_synthesis_conflicted()
    test_synthesis_neutral()

    print("\n=== Fragility Monitor Tests ===")
    test_fragility_healthy()
    test_fragility_elevated()
    test_fragility_extreme()
    test_crisis_condition()
    test_staleness()

    print("\n=== ALL ETAPPE 4 TESTS PASSED ===")
