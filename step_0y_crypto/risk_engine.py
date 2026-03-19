#!/usr/bin/env python3
"""
risk_engine.py — Crypto Circle Risk Engine V1.0
=================================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

V8+Warn KOMPONENTE 3: Phase 4 Altseason Warning + Finale Allokation
  - Phase 4 → Gesamt-Allokation × 0.60
  - NO-ACTION Band: ±10pp Gesamt, ±5pp pro Asset
  - Finale Positionen: BTC/ETH/SOL/Cash

Kombiniert Output von cycle_engine (Ensemble → Gesamt-Allokation)
und signal_engine (Trickle-Down → Phase + Tier-Gewichte)
zur finalen Allokation.

Logik ist 1:1 aus CRYPTO_CIRCLE_V8_WARN_PRODUCTION.py run_engine()
übernommen (Zeilen 360-383).

Input:
  - cycle_result: dict von run_cycle_engine()
  - signal_result: dict von run_signal_engine() oder run_signal_engine_single()
  - current_holdings: dict (optional, für NO-ACTION Band)

Output:
  - Finale Allokation: BTC%, ETH%, SOL%, Cash%
  - Action: REBALANCE oder HOLD (NO-ACTION Band)

Usage:
  from step_0y_crypto.risk_engine import run_risk_engine
  result = run_risk_engine(cycle_result, signal_result)

  # Mit aktuellem Portfolio (für NO-ACTION Band):
  result = run_risk_engine(cycle_result, signal_result, current_holdings)
"""
import os
import sys
import json
import numpy as np
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════
# CONFIG IMPORTS
# ═══════════════════════════════════════════════════════
try:
    from step_0y_crypto.config import (
        V8_PHASE4_WARNING, V8_NO_ACTION_BAND, V8_EXECUTION,
        CONFIG_VERSION,
    )
except (ImportError, ModuleNotFoundError):
    _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/content/step_0y_crypto'
    sys.path.insert(0, _dir)
    sys.path.insert(0, os.path.dirname(_dir))
    from config import (
        V8_PHASE4_WARNING, V8_NO_ACTION_BAND, V8_EXECUTION,
        CONFIG_VERSION,
    )


# ═══════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════
def log(msg):
    print(f"  [RISK] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# PHASE 4 WARNING
# Exakt wie in V8+Warn Production Script Zeilen 364-367
# ═══════════════════════════════════════════════════════

def apply_phase4_warning(alloc, phase):
    """Phase 4 Altseason Warning: Gesamt-Allokation × 0.60.

    Args:
        alloc: float 0.0-1.0 — Gesamt-Allokation aus cycle_engine
        phase: int 1-4 — Phase aus signal_engine

    Returns:
        modified_alloc: float 0.0-1.0
        warning_active: bool
    """
    if phase == 4:
        mult = V8_PHASE4_WARNING['multiplier']
        return round(alloc * mult, 4), True
    return alloc, False


# ═══════════════════════════════════════════════════════
# FINALE POSITIONEN
# Exakt wie in V8+Warn Production Script Zeilen 370-383
# ═══════════════════════════════════════════════════════

def calc_positions(alloc, weights):
    """Berechne finale Positionen aus Gesamt-Allokation und Tier-Gewichten.

    Args:
        alloc:   float 0.0-1.0 — Gesamt-Allokation (nach Phase 4 Warning)
        weights: dict {'BTC': float, 'ETH': float, 'SOL': float}

    Returns:
        dict {'btc': float, 'eth': float, 'sol': float, 'cash': float}
        Alle Werte 0.0-1.0, summieren sich zu 1.0
    """
    btc = round(alloc * weights['BTC'], 4)
    eth = round(alloc * weights['ETH'], 4)
    sol = round(alloc * weights['SOL'], 4)
    cash = round(1.0 - alloc, 4)

    return {
        'btc': btc,
        'eth': eth,
        'sol': sol,
        'cash': cash,
        'total_invested': round(alloc, 4),
    }


# ═══════════════════════════════════════════════════════
# NO-ACTION BAND
# V8+Warn Produktionsspezifikation Abschnitt 12.3
# ═══════════════════════════════════════════════════════

def check_no_action_band(target_positions, current_holdings):
    """Prüfe ob Rebalancing nötig oder innerhalb NO-ACTION Band.

    Args:
        target_positions: dict {'btc': float, 'eth': float, 'sol': float, 'cash': float}
        current_holdings: dict {'btc': float, 'eth': float, 'sol': float, 'cash': float}
                          Alle Werte als Anteil 0.0-1.0

    Returns:
        action: 'REBALANCE' oder 'HOLD'
        deltas: dict pro Asset (Ziel - Aktuell in pp)
        reason: str — Begründung
    """
    if current_holdings is None:
        return 'REBALANCE', {}, 'Keine aktuellen Holdings — erstes Rebalancing'

    total_band = V8_NO_ACTION_BAND['total_pp'] / 100.0  # pp → Anteil
    asset_band = V8_NO_ACTION_BAND['asset_pp'] / 100.0

    deltas = {}
    max_asset_delta = 0.0
    total_delta = 0.0

    for asset in ['btc', 'eth', 'sol']:
        target = target_positions.get(asset, 0.0)
        current = current_holdings.get(asset, 0.0)
        delta = target - current
        deltas[asset] = round(delta * 100, 1)  # In pp
        abs_delta = abs(delta)
        max_asset_delta = max(max_asset_delta, abs_delta)
        total_delta += abs_delta

    # Total Invested Check
    target_total = target_positions.get('total_invested', 0.0)
    current_total = current_holdings.get('total_invested',
                     sum(current_holdings.get(a, 0.0) for a in ['btc', 'eth', 'sol']))
    total_alloc_delta = abs(target_total - current_total)
    deltas['total'] = round((target_total - current_total) * 100, 1)

    # NO-ACTION Band Prüfung
    if total_alloc_delta > total_band:
        return 'REBALANCE', deltas, f'Gesamt-Allokation Δ {deltas["total"]:+.1f}pp > ±{V8_NO_ACTION_BAND["total_pp"]}pp'

    if max_asset_delta > asset_band:
        worst = max(deltas.items(), key=lambda x: abs(x[1]) if x[0] != 'total' else 0)
        return 'REBALANCE', deltas, f'{worst[0].upper()} Δ {worst[1]:+.1f}pp > ±{V8_NO_ACTION_BAND["asset_pp"]}pp'

    return 'HOLD', deltas, f'Alle Deltas innerhalb Band (max {max_asset_delta*100:.1f}pp)'


# ═══════════════════════════════════════════════════════
# ASYMMETRISCHER AUFBAU (Live-Execution Empfehlung)
# V8+Warn Produktionsspezifikation Abschnitt 12.2
# ═══════════════════════════════════════════════════════

def calc_execution_schedule(target_positions, current_holdings):
    """Berechne empfohlenen Execution-Plan (asymmetrisch).
    Reduzierung: sofort. Aufbau: über 3 Wochen.

    Nur Empfehlung — nicht im Backtest genutzt.

    Args:
        target_positions: dict {'btc': float, ...}
        current_holdings: dict {'btc': float, ...} oder None

    Returns:
        dict mit Wochen-Plan oder 'IMMEDIATE'
    """
    if current_holdings is None:
        return {'type': 'INITIAL', 'action': 'Sofort auf Ziel'}

    target_total = target_positions.get('total_invested', 0.0)
    current_total = current_holdings.get('total_invested',
                     sum(current_holdings.get(a, 0.0) for a in ['btc', 'eth', 'sol']))

    delta = target_total - current_total

    if delta < -0.01:
        # Reduzierung → sofort
        return {
            'type': 'REDUCE',
            'action': 'Sofort auf Ziel reduzieren',
            'week1': target_positions,
        }
    elif delta > 0.01:
        # Aufbau → über 3 Wochen
        w1_pct = V8_EXECUTION['build_week1_pct']
        w2_pct = V8_EXECUTION['build_week2_pct']
        w3_pct = V8_EXECUTION['build_week3_pct']

        weeks = {}
        for week, pct in [(1, w1_pct), (2, w1_pct + w2_pct), (3, 1.0)]:
            interim_total = current_total + delta * pct
            weeks[f'week{week}'] = {
                'total_invested': round(interim_total, 4),
                'btc': round(interim_total * target_positions.get('btc', 0) / max(target_total, 0.01), 4),
                'eth': round(interim_total * target_positions.get('eth', 0) / max(target_total, 0.01), 4),
                'sol': round(interim_total * target_positions.get('sol', 0) / max(target_total, 0.01), 4),
            }

        return {
            'type': 'BUILD',
            'action': f'Aufbau über {V8_EXECUTION["build_weeks"]} Wochen',
            **weeks,
        }
    else:
        return {'type': 'NO_CHANGE', 'action': 'Keine Änderung nötig'}


# ═══════════════════════════════════════════════════════
# HAUPTFUNKTION
# ═══════════════════════════════════════════════════════

def run_risk_engine(cycle_result, signal_result, current_holdings=None):
    """V8+Warn Risk Engine — Komponente 3 + Finale Allokation.

    Args:
        cycle_result:     dict von run_cycle_engine()
        signal_result:    dict von run_signal_engine() oder run_signal_engine_single()
        current_holdings: dict {'btc': float, 'eth': float, 'sol': float, 'cash': float}
                          optional, für NO-ACTION Band

    Returns:
        dict mit finaler Allokation, Action, Details
    """
    log("=" * 50)
    log("RISK ENGINE V1.0 — Phase 4 Warning + Finale Allokation")
    log("=" * 50)

    # Inputs aus vorgelagerten Engines
    raw_alloc = cycle_result.get('alloc', 0.0)
    ensemble = cycle_result.get('ensemble', 0.0)
    mom = cycle_result.get('mom', {})
    below_wma = cycle_result.get('below_wma', False)
    wma_200 = cycle_result.get('wma_200')

    phase = signal_result.get('phase', 2)
    phase_name = signal_result.get('phase_name', 'BALANCED')
    weights = signal_result.get('weights', {'BTC': 0.45, 'ETH': 0.35, 'SOL': 0.20})
    btc_d = signal_result.get('btc_dominance')
    btc_d_chg = signal_result.get('btc_d_change', 0.0)

    # ─── Schritt 1: Phase 4 Warning ───
    final_alloc, p4_warning = apply_phase4_warning(raw_alloc, phase)
    if p4_warning:
        log(f"  ⚠️  PHASE 4 WARNING: {raw_alloc:.0%} × {V8_PHASE4_WARNING['multiplier']} = {final_alloc:.0%}")
    else:
        log(f"  Allokation: {final_alloc:.0%} (keine Phase 4 Warning)")

    # ─── Schritt 2: Finale Positionen ───
    positions = calc_positions(final_alloc, weights)
    log(f"  Positionen: BTC={positions['btc']:.1%} ETH={positions['eth']:.1%} "
        f"SOL={positions['sol']:.1%} Cash={positions['cash']:.1%}")

    # ─── Schritt 3: NO-ACTION Band ───
    action, deltas, reason = check_no_action_band(positions, current_holdings)
    if current_holdings is not None:
        log(f"  NO-ACTION:  {action} — {reason}")
        if deltas:
            delta_str = " ".join(f"{k.upper()}={v:+.1f}pp" for k, v in deltas.items())
            log(f"  Deltas:     {delta_str}")
    else:
        log(f"  NO-ACTION:  Nicht geprüft (keine aktuellen Holdings)")

    # ─── Schritt 4: Execution-Empfehlung ───
    execution = calc_execution_schedule(positions, current_holdings)

    # ─── Output ───
    result = {
        'status': 'OK',
        'timestamp': datetime.now(timezone.utc).isoformat(),

        # Finale Allokation
        'allocation': positions,
        'action': action,
        'action_reason': reason,
        'deltas': deltas,

        # Zusammenfassung der 3 Komponenten
        'components': {
            'ensemble': {
                'value': ensemble,
                'raw_alloc': raw_alloc,
                'mom_1M': mom.get('1M', False),
                'mom_3M': mom.get('3M', False),
                'mom_6M': mom.get('6M', False),
                'mom_12M': mom.get('12M', False),
                'below_200wma': below_wma,
                'wma_200': wma_200,
            },
            'trickle_down': {
                'phase': phase,
                'phase_name': phase_name,
                'btc_dominance': btc_d,
                'btc_d_30d_change': btc_d_chg,
                'weights': weights,
            },
            'phase4_warning': {
                'active': p4_warning,
                'multiplier': V8_PHASE4_WARNING['multiplier'],
                'alloc_before': raw_alloc,
                'alloc_after': final_alloc,
            },
        },

        # Execution
        'execution': execution,

        # Meta
        'engine_version': 'risk_engine V1.0',
        'config_version': CONFIG_VERSION,
    }

    return result


# ═══════════════════════════════════════════════════════
# STANDALONE EXECUTION (Test)
# ═══════════════════════════════════════════════════════

def main():
    """Standalone Test — alle 5 Referenzpunkte aus V8+Warn Spec."""
    t0 = datetime.now(timezone.utc)
    print("=" * 60)
    print("RISK ENGINE V1.0 — Standalone Test")
    print(f"  {t0.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    all_ok = True

    # ─── Referenzpunkte aus V8+Warn Produktionsspezifikation Abschnitt 7 ───

    tests = [
        {
            'label': '2017 Top',
            'cycle': {'alloc': 1.00, 'ensemble': 1.00,
                      'mom': {'1M': True, '3M': True, '6M': True, '12M': True},
                      'below_wma': False, 'wma_200': 2500},
            'signal': {'phase': 2, 'phase_name': 'BALANCED',
                       'weights': {'BTC': 0.45, 'ETH': 0.35, 'SOL': 0.20},
                       'btc_dominance': 49.3, 'btc_d_change': 1.1},
            'expected': {'total': 1.00, 'btc': 0.45, 'eth': 0.35, 'sol': 0.20, 'cash': 0.00},
        },
        {
            'label': 'COVID Crash',
            'cycle': {'alloc': 0.50, 'ensemble': 0.00,
                      'mom': {'1M': False, '3M': False, '6M': False, '12M': False},
                      'below_wma': True, 'wma_200': 6500},
            'signal': {'phase': 1, 'phase_name': 'BTC_DOMINANT',
                       'weights': {'BTC': 0.70, 'ETH': 0.25, 'SOL': 0.05},
                       'btc_dominance': 64.0, 'btc_d_change': 4.5},
            'expected': {'total': 0.50, 'btc': 0.35, 'eth': 0.125, 'sol': 0.025, 'cash': 0.50},
        },
        {
            'label': '2025 ATH',
            'cycle': {'alloc': 1.00, 'ensemble': 1.00,
                      'mom': {'1M': True, '3M': True, '6M': True, '12M': True},
                      'below_wma': False, 'wma_200': 50176},
            'signal': {'phase': 3, 'phase_name': 'ALTSEASON',
                       'weights': {'BTC': 0.25, 'ETH': 0.35, 'SOL': 0.40},
                       'btc_dominance': 54.4, 'btc_d_change': -2.5},
            'expected': {'total': 1.00, 'btc': 0.25, 'eth': 0.35, 'sol': 0.40, 'cash': 0.00},
        },
        {
            'label': '2025 Nov Drop',
            'cycle': {'alloc': 0.25, 'ensemble': 0.25,
                      'mom': {'1M': False, '3M': False, '6M': False, '12M': True},
                      'below_wma': False, 'wma_200': 55294},
            'signal': {'phase': 1, 'phase_name': 'BTC_DOMINANT',
                       'weights': {'BTC': 0.70, 'ETH': 0.25, 'SOL': 0.05},
                       'btc_dominance': 62.6, 'btc_d_change': 2.4},
            'expected': {'total': 0.25, 'btc': 0.175, 'eth': 0.0625, 'sol': 0.0125, 'cash': 0.75},
        },
        {
            'label': 'Heute (März 2026)',
            'cycle': {'alloc': 0.25, 'ensemble': 0.25,
                      'mom': {'1M': True, '3M': False, '6M': False, '12M': False},
                      'below_wma': False, 'wma_200': 58903},
            'signal': {'phase': 2, 'phase_name': 'BALANCED',
                       'weights': {'BTC': 0.45, 'ETH': 0.35, 'SOL': 0.20},
                       'btc_dominance': 55.0, 'btc_d_change': 0.7},
            'expected': {'total': 0.25, 'btc': 0.1125, 'eth': 0.0875, 'sol': 0.05, 'cash': 0.75},
        },
    ]

    for t in tests:
        r = run_risk_engine(t['cycle'], t['signal'])
        a = r['allocation']
        e = t['expected']

        ok_total = abs(a['total_invested'] - e['total']) < 0.01
        ok_btc = abs(a['btc'] - e['btc']) < 0.01
        ok_eth = abs(a['eth'] - e['eth']) < 0.01
        ok_sol = abs(a['sol'] - e['sol']) < 0.01
        ok_cash = abs(a['cash'] - e['cash']) < 0.01
        ok = ok_total and ok_btc and ok_eth and ok_sol and ok_cash

        if not ok:
            all_ok = False

        status = "✅" if ok else "❌"
        print(f"\n  {status} {t['label']}")
        print(f"      Total:  {a['total_invested']:.2f} (erwartet {e['total']:.2f}) {'✅' if ok_total else '❌'}")
        print(f"      BTC:    {a['btc']:.4f} (erwartet {e['btc']:.4f}) {'✅' if ok_btc else '❌'}")
        print(f"      ETH:    {a['eth']:.4f} (erwartet {e['eth']:.4f}) {'✅' if ok_eth else '❌'}")
        print(f"      SOL:    {a['sol']:.4f} (erwartet {e['sol']:.4f}) {'✅' if ok_sol else '❌'}")
        print(f"      Cash:   {a['cash']:.4f} (erwartet {e['cash']:.4f}) {'✅' if ok_cash else '❌'}")

    # ─── Phase 4 Warning Test ───
    print("\n--- Phase 4 Warning Test ---")
    cycle_p4 = {'alloc': 1.00, 'ensemble': 1.00,
                'mom': {'1M': True, '3M': True, '6M': True, '12M': True},
                'below_wma': False, 'wma_200': 50000}
    signal_p4 = {'phase': 4, 'phase_name': 'ALTSEASON_WARN',
                 'weights': {'BTC': 0.25, 'ETH': 0.35, 'SOL': 0.40},
                 'btc_dominance': 39.5, 'btc_d_change': -6.7}
    r_p4 = run_risk_engine(cycle_p4, signal_p4)
    expected_total = 0.60  # 1.00 × 0.60
    ok_p4 = abs(r_p4['allocation']['total_invested'] - expected_total) < 0.01
    ok_p4_flag = r_p4['components']['phase4_warning']['active'] == True
    if not (ok_p4 and ok_p4_flag):
        all_ok = False
    print(f"  {'✅' if ok_p4 and ok_p4_flag else '❌'} Phase 4: "
          f"Allok={r_p4['allocation']['total_invested']:.2f} (erwartet 0.60), "
          f"Warning={'aktiv' if r_p4['components']['phase4_warning']['active'] else 'inaktiv'}")

    # ─── NO-ACTION Band Test ───
    print("\n--- NO-ACTION Band Test ---")
    target = {'btc': 0.45, 'eth': 0.35, 'sol': 0.20, 'cash': 0.00, 'total_invested': 1.00}

    # Innerhalb Band
    current_close = {'btc': 0.43, 'eth': 0.34, 'sol': 0.19, 'cash': 0.04, 'total_invested': 0.96}
    action, deltas, reason = check_no_action_band(target, current_close)
    ok_hold = action == 'HOLD'
    if not ok_hold:
        all_ok = False
    print(f"  {'✅' if ok_hold else '❌'} Kleine Deltas → {action} ({reason})")

    # Außerhalb Band (Gesamt)
    current_far = {'btc': 0.20, 'eth': 0.15, 'sol': 0.10, 'cash': 0.55, 'total_invested': 0.45}
    action, deltas, reason = check_no_action_band(target, current_far)
    ok_reb = action == 'REBALANCE'
    if not ok_reb:
        all_ok = False
    print(f"  {'✅' if ok_reb else '❌'} Großes Delta → {action} ({reason})")

    # Außerhalb Band (einzelnes Asset)
    current_asset = {'btc': 0.38, 'eth': 0.35, 'sol': 0.20, 'cash': 0.07, 'total_invested': 0.93}
    action, deltas, reason = check_no_action_band(target, current_asset)
    ok_asset = action == 'REBALANCE'
    if not ok_asset:
        all_ok = False
    print(f"  {'✅' if ok_asset else '❌'} BTC Δ > 5pp → {action} ({reason})")

    # Kein Portfolio
    action, deltas, reason = check_no_action_band(target, None)
    ok_none = action == 'REBALANCE'
    if not ok_none:
        all_ok = False
    print(f"  {'✅' if ok_none else '❌'} Kein Portfolio → {action}")

    # ─── Zusammenfassung ───
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"\n{'='*60}")
    if all_ok:
        print(f"ALLE TESTS BESTANDEN ✅ ({elapsed:.1f}s)")
    else:
        print(f"TESTS FEHLGESCHLAGEN ❌ ({elapsed:.1f}s)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
