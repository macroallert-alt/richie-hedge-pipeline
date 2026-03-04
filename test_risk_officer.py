"""
Test: Risk Officer V1 mit realistischen V16-Daten.
Basiert auf heutigem V16 Run Output:
  Regime: SELECTIVE (State 3: LATE_EXPANSION)
  Growth: 1 | Liq: -1 | Stress: 0
  Top 5: HYG=27.4%, DBC=20.4%, GLD=18.6%, XLU=18.2%, XLP=15.4%
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from step3_risk_officer.engine import run_risk_officer
from datetime import date


def test_normal_day():
    """Test 1: Normaler Tag mit heutigen V16-Gewichten."""
    print("=" * 60)
    print("TEST 1: Normaler Tag (heutige V16-Gewichte)")
    print("=" * 60)

    inputs = {
        "v16_production": {
            "date": "2026-03-03",
            "v16_state": "Risk-On",
            "v16_regime": "LATE_EXPANSION",
            "v16_confluence": 7,
            "dd_protect_active": False,
            "dd_protect_trigger_level": -0.12,
            "current_drawdown_from_peak": -0.02,
            "weights": {
                "HYG": 0.274, "DBC": 0.204, "GLD": 0.186,
                "XLU": 0.182, "XLP": 0.154,
            }
        },
        "layer_analysis": {
            "system_regime": {
                "regime": "SELECTIVE",
                "lean": "POSITIVE"
            },
            "fragility_state": {
                "state": "HEALTHY"
            }
        }
    }

    result = run_risk_officer(inputs, risk_history=None, run_date=date(2026, 3, 4))

    print(f"\nPortfolio Status: {result['portfolio_status']}")
    print(f"Reason: {result['portfolio_status_reason']}")
    print(f"Alerts: {result['metadata']['alerts_count']}")
    print(f"Execution: {result['execution_path']} ({result['metadata']['execution_time_ms']}ms)")
    print(f"\nRisk Summary:\n{result['risk_summary']}")

    if result['alerts']:
        print(f"\nAlerts Detail:")
        for a in result['alerts']:
            if a['severity'] != 'RESOLVED':
                print(f"  [{a['severity']}] {a['check_id']}: {a['message'][:100]}")

    return result


def test_high_tech_exposure():
    """Test 2: Hohe Tech-Konzentration (simuliert)."""
    print("\n" + "=" * 60)
    print("TEST 2: Hohe Tech-Konzentration")
    print("=" * 60)

    inputs = {
        "v16_production": {
            "date": "2026-03-03",
            "v16_state": "Risk-On",
            "v16_regime": "FULL_EXPANSION",
            "v16_confluence": 8,
            "dd_protect_active": False,
            "dd_protect_trigger_level": -0.12,
            "current_drawdown_from_peak": -0.03,
            "weights": {
                "XLK": 0.18, "QQQ": 0.12, "SPY": 0.15,
                "XLF": 0.08, "GLD": 0.10, "TLT": 0.07,
                "XLE": 0.06, "XLV": 0.05, "XLY": 0.05,
                "IWM": 0.04, "XLI": 0.04, "XLP": 0.03,
                "XLU": 0.03
            }
        },
        "layer_analysis": {
            "system_regime": {
                "regime": "BROAD_RISK_ON",
                "lean": "POSITIVE"
            },
            "fragility_state": {
                "state": "ELEVATED"
            }
        }
    }

    result = run_risk_officer(inputs, risk_history=None, run_date=date(2026, 3, 4))

    print(f"\nPortfolio Status: {result['portfolio_status']}")
    print(f"Reason: {result['portfolio_status_reason']}")
    print(f"\nRisk Summary:\n{result['risk_summary']}")

    if result['alerts']:
        print(f"\nAlerts Detail:")
        for a in result['alerts']:
            if a['severity'] != 'RESOLVED':
                print(f"  [{a['severity']}] {a['check_id']}: {a['message'][:120]}")

    return result


def test_regime_conflict():
    """Test 3: V16 Risk-On vs Market Analyst BROAD_RISK_OFF."""
    print("\n" + "=" * 60)
    print("TEST 3: Regime Conflict")
    print("=" * 60)

    inputs = {
        "v16_production": {
            "date": "2026-03-03",
            "v16_state": "Risk-On",
            "v16_regime": "LATE_EXPANSION",
            "v16_confluence": 6,
            "dd_protect_active": False,
            "dd_protect_trigger_level": -0.12,
            "current_drawdown_from_peak": -0.05,
            "weights": {
                "XLK": 0.12, "SPY": 0.10, "GLD": 0.15,
                "TLT": 0.10, "XLF": 0.08, "DBC": 0.08,
                "XLE": 0.07, "XLP": 0.06, "XLU": 0.06,
                "XLV": 0.05, "HYG": 0.05, "IWM": 0.04,
                "XLY": 0.04
            }
        },
        "layer_analysis": {
            "system_regime": {
                "regime": "BROAD_RISK_OFF",
                "lean": "NEGATIVE"
            },
            "fragility_state": {
                "state": "EXTREME"
            }
        }
    }

    result = run_risk_officer(inputs, risk_history=None, run_date=date(2026, 3, 4))

    print(f"\nPortfolio Status: {result['portfolio_status']}")
    print(f"\nRisk Summary:\n{result['risk_summary']}")

    if result['alerts']:
        print(f"\nAlerts Detail:")
        for a in result['alerts']:
            if a['severity'] != 'RESOLVED':
                print(f"  [{a['severity']}] {a['check_id']}: {a['message'][:120]}")

    return result


def test_emergency_dd():
    """Test 4: Emergency — DD > -15%."""
    print("\n" + "=" * 60)
    print("TEST 4: Emergency — Portfolio Drawdown")
    print("=" * 60)

    inputs = {
        "v16_production": {
            "date": "2026-03-03",
            "v16_state": "DD-Protect",
            "v16_regime": "FINANCIAL_CRISIS",
            "v16_confluence": 2,
            "dd_protect_active": True,
            "dd_protect_trigger_level": -0.12,
            "current_drawdown_from_peak": -0.16,
            "weights": {
                "TLT": 0.25, "GLD": 0.25, "SHY": 0.20,
                "XLP": 0.10, "XLU": 0.10, "SPY": 0.05,
                "HYG": 0.05
            }
        },
        "layer_analysis": {
            "system_regime": {
                "regime": "RISK_OFF_FORCED",
                "lean": "NEGATIVE"
            },
            "fragility_state": {
                "state": "CRISIS"
            }
        }
    }

    result = run_risk_officer(inputs, risk_history=None, run_date=date(2026, 3, 4))

    print(f"\nPortfolio Status: {result['portfolio_status']}")
    print(f"\nRisk Summary:\n{result['risk_summary']}")

    return result


def test_fast_path():
    """Test 5: Fast Path — identische Inputs wie gestern."""
    print("\n" + "=" * 60)
    print("TEST 5: Fast Path (Day 2, unchanged)")
    print("=" * 60)

    inputs = {
        "v16_production": {
            "v16_state": "Risk-On",
            "v16_regime": "LATE_EXPANSION",
            "dd_protect_active": False,
            "current_drawdown_from_peak": -0.02,
            "weights": {
                "HYG": 0.274, "DBC": 0.204, "GLD": 0.186,
                "XLU": 0.182, "XLP": 0.154
            }
        },
        "layer_analysis": {
            "system_regime": {"regime": "SELECTIVE", "lean": "POSITIVE"},
            "fragility_state": {"state": "HEALTHY"}
        }
    }

    day1 = run_risk_officer(inputs, risk_history=None, run_date=date(2026, 3, 3))

    history = {
        "portfolio_status": day1["portfolio_status"],
        "v16_state": "Risk-On",
        "fragility_state": "HEALTHY",
        "alerts": day1["alerts"]
    }

    day2 = run_risk_officer(inputs, risk_history=history, run_date=date(2026, 3, 4))

    print(f"\nDay 1: {day1['portfolio_status']} ({day1['execution_path']})")
    print(f"Day 2: {day2['portfolio_status']} ({day2['execution_path']})")
    print(f"\nDay 2 Risk Summary:\n{day2['risk_summary']}")

    return day2


if __name__ == "__main__":
    r1 = test_normal_day()
    r2 = test_high_tech_exposure()
    r3 = test_regime_conflict()
    r4 = test_emergency_dd()
    r5 = test_fast_path()

    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"  Test 1 (Normal):        {r1['portfolio_status']}")
    print(f"  Test 2 (High Tech):     {r2['portfolio_status']}")
    print(f"  Test 3 (Regime Conf):   {r3['portfolio_status']}")
    print(f"  Test 4 (Emergency DD):  {r4['portfolio_status']}")
    print(f"  Test 5 (Fast Path):     {r5['portfolio_status']} ({r5['execution_path']})")