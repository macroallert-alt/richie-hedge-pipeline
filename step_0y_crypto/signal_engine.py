#!/usr/bin/env python3
"""
signal_engine.py — Crypto Circle Signal Engine V1.0
=====================================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

V8+Warn KOMPONENTE 2: BTC Dominance Trickle-Down Rotation
  - BTC.D 30d-Veränderung → Phase 1-4
  - Phase → Tier-Gewichte (BTC/ETH/SOL Verteilung)

Logik ist 1:1 aus CRYPTO_CIRCLE_V8_WARN_PRODUCTION.py calc_trickle_down()
übernommen.

Datenquellen für BTC Dominance:
  1. CoinGecko /global (aktueller Wert) — im data_collector
  2. Historisch: CoinMetrics BTC MCap + ETH MCap → Proxy mit Korrekturfaktor
  3. Fallback: Phase 2 (Default, 77% der Zeit)

Input:  BTC Dominance Zeitreihe (oder aktueller Wert + 30d-ago Wert)
Output: phase (1-4), tier_weights dict, btc_d_change

Usage:
  from step_0y_crypto.signal_engine import run_signal_engine
  result = run_signal_engine(btc_d_values)

  # Oder mit Einzelwerten (Produktion):
  result = run_signal_engine_single(btc_d_today, btc_d_30d_ago)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════
# CONFIG IMPORTS
# ═══════════════════════════════════════════════════════
try:
    from step_0y_crypto.config import (
        V8_TRICKLE_DOWN, V8_BTC_D_PROXY,
        API_ENDPOINTS, CONFIG_VERSION,
    )
except (ImportError, ModuleNotFoundError):
    _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/content/step_0y_crypto'
    sys.path.insert(0, _dir)
    sys.path.insert(0, os.path.dirname(_dir))
    from config import (
        V8_TRICKLE_DOWN, V8_BTC_D_PROXY,
        API_ENDPOINTS, CONFIG_VERSION,
    )


# ═══════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════
def log(msg):
    print(f"  [SIGNAL] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# PHASE NAMES
# ═══════════════════════════════════════════════════════
PHASE_NAMES = {
    1: 'BTC_DOMINANT',
    2: 'BALANCED',
    3: 'ALTSEASON',
    4: 'ALTSEASON_WARN',
}


# ═══════════════════════════════════════════════════════
# TRICKLE-DOWN BERECHNUNG (Serie)
# Exakt wie in CRYPTO_CIRCLE_V8_WARN_PRODUCTION.py calc_trickle_down()
# ═══════════════════════════════════════════════════════

def calc_trickle_down(btc_d_vals):
    """Berechne Trickle-Down Phase aus BTC.D 30d Change.

    Args:
        btc_d_vals: numpy array of BTC Dominance values (ascending by date)

    Returns:
        phase:   numpy array of int (1-4)
        chg:     numpy array of float (30d change in pp)
    """
    thresholds = V8_TRICKLE_DOWN['phase_thresholds']
    lookback = V8_TRICKLE_DOWN['dominance_lookback_days']
    default = V8_TRICKLE_DOWN['default_phase']

    p1 = thresholds['phase1_above']
    p3 = thresholds['phase3_below']
    p4 = thresholds['phase4_below']

    n = len(btc_d_vals)
    chg = np.full(n, np.nan)

    for i in range(lookback, n):
        if not np.isnan(btc_d_vals[i]) and not np.isnan(btc_d_vals[i - lookback]):
            chg[i] = btc_d_vals[i] - btc_d_vals[i - lookback]

    phase = np.full(n, default, dtype=int)
    for i in range(n):
        c = chg[i]
        if np.isnan(c):
            phase[i] = default
        elif c < p4:
            phase[i] = 4
        elif c < p3:
            phase[i] = 3
        elif c > p1:
            phase[i] = 1
        else:
            phase[i] = 2

    return phase, chg


# ═══════════════════════════════════════════════════════
# TRICKLE-DOWN BERECHNUNG (Einzelwerte — Produktion)
# ═══════════════════════════════════════════════════════

def calc_phase_single(btc_d_today, btc_d_30d_ago):
    """Berechne Phase aus zwei BTC.D Werten (Produktionsmodus).

    Args:
        btc_d_today:  float — BTC Dominance heute (%)
        btc_d_30d_ago: float — BTC Dominance vor 30 Tagen (%)

    Returns:
        phase: int (1-4)
        change: float (Veränderung in pp)
    """
    if btc_d_today is None or btc_d_30d_ago is None:
        return V8_TRICKLE_DOWN['default_phase'], 0.0

    thresholds = V8_TRICKLE_DOWN['phase_thresholds']
    change = btc_d_today - btc_d_30d_ago

    if change < thresholds['phase4_below']:
        phase = 4
    elif change < thresholds['phase3_below']:
        phase = 3
    elif change > thresholds['phase1_above']:
        phase = 1
    else:
        phase = 2

    return phase, round(change, 2)


# ═══════════════════════════════════════════════════════
# TIER-GEWICHTE
# ═══════════════════════════════════════════════════════

def get_tier_weights(phase):
    """Tier-Gewichte für eine gegebene Phase.

    Returns:
        dict {'BTC': float, 'ETH': float, 'SOL': float}
    """
    weights = V8_TRICKLE_DOWN['phase_weights']
    return weights.get(phase, weights[V8_TRICKLE_DOWN['default_phase']])


# ═══════════════════════════════════════════════════════
# BTC DOMINANCE HISTORY (für Backtest/Validierung)
# ═══════════════════════════════════════════════════════

def build_btc_dominance_proxy(btc_mcap_series, eth_mcap_series, years):
    """BTC.D Proxy aus MarketCap-Daten mit Korrekturfaktoren.
    Exakt wie in V8+Warn Production Script build_btc_dominance().

    Args:
        btc_mcap_series: numpy array BTC Market Cap
        eth_mcap_series: numpy array ETH Market Cap (gleiche Länge, 0 wo fehlt)
        years:           numpy array of int (Jahr pro Datenpunkt)

    Returns:
        btc_d: numpy array of BTC Dominance (%)
    """
    cfs = V8_BTC_D_PROXY['correction_factors']
    default_cf = V8_BTC_D_PROXY['default_cf']
    min_d = V8_BTC_D_PROXY['min_dominance']
    max_d = V8_BTC_D_PROXY['max_dominance']

    n = len(btc_mcap_series)
    btc_d = np.full(n, np.nan)

    for i in range(n):
        bm = btc_mcap_series[i]
        em = eth_mcap_series[i] if i < len(eth_mcap_series) else 0
        yr = years[i] if i < len(years) else 2026

        if np.isnan(bm) or bm <= 0:
            continue

        em = em if not np.isnan(em) else 0
        cf = cfs.get(int(yr), default_cf)

        total = (bm + em) / cf
        if total > 0:
            d = bm / total * 100
            if min_d <= d <= max_d:
                btc_d[i] = d

    return btc_d


# ═══════════════════════════════════════════════════════
# HAUPTFUNKTION (Serie — für Backtest/Validierung)
# ═══════════════════════════════════════════════════════

def run_signal_engine(btc_d_values):
    """V8+Warn Signal Engine — Komponente 2 (Serie).

    Args:
        btc_d_values: numpy array of BTC Dominance (%) ascending by date

    Returns:
        dict mit Phase, Gewichte, Statistiken
    """
    log("=" * 50)
    log("SIGNAL ENGINE V1.0 — Trickle-Down Rotation")
    log("=" * 50)

    n = len(btc_d_values)
    if n < 31:
        log(f"WARNUNG: Nur {n} Tage — brauche mindestens 31 für 30d Change")
        return {
            'status': 'ERROR',
            'error': f'Zu wenig BTC.D Daten: {n} Tage (min 31)',
            'phase': V8_TRICKLE_DOWN['default_phase'],
            'phase_name': PHASE_NAMES[V8_TRICKLE_DOWN['default_phase']],
            'weights': get_tier_weights(V8_TRICKLE_DOWN['default_phase']),
            'btc_d_change': 0.0,
        }

    # Phasen berechnen
    phase_series, chg_series = calc_trickle_down(btc_d_values)

    # Aktuellste Werte
    current_phase = int(phase_series[-1])
    current_chg = round(chg_series[-1], 2) if not np.isnan(chg_series[-1]) else 0.0
    current_btc_d = round(btc_d_values[-1], 2) if not np.isnan(btc_d_values[-1]) else None
    current_weights = get_tier_weights(current_phase)

    # Phase-Verteilung
    phase_dist = {}
    for p in [1, 2, 3, 4]:
        cnt = int((phase_series == p).sum())
        phase_dist[p] = {'days': cnt, 'pct': round(cnt / n * 100, 1)}

    # Logging
    log(f"  BTC.D:     {current_btc_d}%" if current_btc_d else "  BTC.D:     N/A")
    log(f"  30d Δ:     {current_chg:+.1f}pp")
    log(f"  Phase:     {current_phase} ({PHASE_NAMES[current_phase]})")
    log(f"  Gewichte:  BTC={current_weights['BTC']:.0%} "
        f"ETH={current_weights['ETH']:.0%} SOL={current_weights['SOL']:.0%}")
    for p in [1, 2, 3, 4]:
        d = phase_dist[p]
        log(f"    P{p} ({PHASE_NAMES[p]:<16}): {d['days']:>5}d ({d['pct']:>5.1f}%)")

    return {
        'status': 'OK',
        'phase': current_phase,
        'phase_name': PHASE_NAMES[current_phase],
        'weights': current_weights,
        'btc_dominance': current_btc_d,
        'btc_d_change': current_chg,
        'phase_distribution': phase_dist,
        # Serien für risk_engine
        'phase_series': phase_series,
        'chg_series': chg_series,
        'engine_version': 'signal_engine V1.0',
        'config_version': CONFIG_VERSION,
    }


# ═══════════════════════════════════════════════════════
# HAUPTFUNKTION (Einzelwerte — Produktion)
# ═══════════════════════════════════════════════════════

def run_signal_engine_single(btc_d_today, btc_d_30d_ago):
    """V8+Warn Signal Engine — Komponente 2 (Einzelwerte).
    Für wöchentliche Produktion: braucht nur 2 BTC.D Werte.

    Args:
        btc_d_today:   float — BTC Dominance heute (%)
        btc_d_30d_ago: float — BTC Dominance vor 30 Tagen (%)

    Returns:
        dict mit Phase, Gewichte, Change
    """
    phase, change = calc_phase_single(btc_d_today, btc_d_30d_ago)
    weights = get_tier_weights(phase)

    log(f"  BTC.D:     {btc_d_today:.1f}% (30d ago: {btc_d_30d_ago:.1f}%)")
    log(f"  30d Δ:     {change:+.1f}pp")
    log(f"  Phase:     {phase} ({PHASE_NAMES[phase]})")
    log(f"  Gewichte:  BTC={weights['BTC']:.0%} ETH={weights['ETH']:.0%} SOL={weights['SOL']:.0%}")

    return {
        'status': 'OK',
        'phase': phase,
        'phase_name': PHASE_NAMES[phase],
        'weights': weights,
        'btc_dominance': round(btc_d_today, 2),
        'btc_d_30d_ago': round(btc_d_30d_ago, 2),
        'btc_d_change': change,
        'engine_version': 'signal_engine V1.0',
        'config_version': CONFIG_VERSION,
    }


# ═══════════════════════════════════════════════════════
# STANDALONE EXECUTION (Test)
# ═══════════════════════════════════════════════════════

def main():
    """Standalone Test — Einzelwerte und synthetische Serie."""
    import argparse
    parser = argparse.ArgumentParser(description='Crypto Circle Signal Engine V1.0')
    parser.add_argument('--skip-write', action='store_true', help='Kein Sheet Write')
    args = parser.parse_args()

    t0 = datetime.now(timezone.utc)
    print("=" * 60)
    print("SIGNAL ENGINE V1.0 — Standalone Test")
    print(f"  {t0.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Test 1: Einzelwerte (Produktionsmodus)
    print("\n--- Test 1: Einzelwerte ---")
    # Heute: BTC.D ~55%, 30d ago ~54.3% → Δ +0.7pp → Phase 2
    r1 = run_signal_engine_single(55.0, 54.3)
    assert r1['phase'] == 2, f"Erwartet Phase 2, got {r1['phase']}"
    print(f"  Phase 2 (Balanced): ✅")

    # Phase 1: BTC.D steigt stark
    r2 = run_signal_engine_single(62.6, 60.0)
    assert r2['phase'] == 1, f"Erwartet Phase 1, got {r2['phase']}"
    print(f"  Phase 1 (BTC dominant): ✅")

    # Phase 3: BTC.D fällt moderat
    r3 = run_signal_engine_single(54.4, 57.0)
    assert r3['phase'] == 3, f"Erwartet Phase 3, got {r3['phase']}"
    print(f"  Phase 3 (Altseason): ✅")

    # Phase 4: BTC.D crasht
    r4 = run_signal_engine_single(39.5, 46.2)
    assert r4['phase'] == 4, f"Erwartet Phase 4, got {r4['phase']}"
    print(f"  Phase 4 (Altseason Warn): ✅")

    # Test 2: Grenzen
    print("\n--- Test 2: Grenzwerte ---")
    # Exakt +2.0pp → Phase 2 (nicht Phase 1, braucht >+2.0)
    r5 = run_signal_engine_single(52.0, 50.0)
    assert r5['phase'] == 2, f"Erwartet Phase 2 bei exakt +2.0pp, got {r5['phase']}"
    print(f"  +2.0pp exakt → Phase 2: ✅")

    # Exakt -2.0pp → Phase 2 (nicht Phase 3, braucht <-2.0)
    r6 = run_signal_engine_single(48.0, 50.0)
    assert r6['phase'] == 2, f"Erwartet Phase 2 bei exakt -2.0pp, got {r6['phase']}"
    print(f"  -2.0pp exakt → Phase 2: ✅")

    # Knapp über +2.0pp → Phase 1
    r7 = run_signal_engine_single(52.1, 50.0)
    assert r7['phase'] == 1, f"Erwartet Phase 1 bei +2.1pp, got {r7['phase']}"
    print(f"  +2.1pp → Phase 1: ✅")

    # Knapp unter -2.0pp → Phase 3
    r8 = run_signal_engine_single(47.9, 50.0)
    assert r8['phase'] == 3, f"Erwartet Phase 3 bei -2.1pp, got {r8['phase']}"
    print(f"  -2.1pp → Phase 3: ✅")

    # Exakt -5.0pp → Phase 3 (nicht Phase 4, braucht <-5.0)
    r9 = run_signal_engine_single(45.0, 50.0)
    assert r9['phase'] == 3, f"Erwartet Phase 3 bei exakt -5.0pp, got {r9['phase']}"
    print(f"  -5.0pp exakt → Phase 3: ✅")

    # Knapp unter -5.0pp → Phase 4
    r10 = run_signal_engine_single(44.9, 50.0)
    assert r10['phase'] == 4, f"Erwartet Phase 4 bei -5.1pp, got {r10['phase']}"
    print(f"  -5.1pp → Phase 4: ✅")

    # Test 3: Gewichte-Summe
    print("\n--- Test 3: Gewichte-Summen ---")
    for p in [1, 2, 3, 4]:
        w = get_tier_weights(p)
        s = w['BTC'] + w['ETH'] + w['SOL']
        assert abs(s - 1.0) < 0.001, f"Phase {p} Gewichte summieren zu {s}"
        print(f"  Phase {p}: BTC={w['BTC']:.0%} ETH={w['ETH']:.0%} SOL={w['SOL']:.0%} = {s:.0%} ✅")

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"\n{'='*60}")
    print(f"ALLE TESTS BESTANDEN ✅ ({elapsed:.1f}s)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
