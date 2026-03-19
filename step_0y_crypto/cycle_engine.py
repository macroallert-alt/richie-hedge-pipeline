#!/usr/bin/env python3
"""
cycle_engine.py — Crypto Circle Cycle Engine V1.0
===================================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

V8+Warn KOMPONENTE 1: BTC Momentum Ensemble + 200WMA Bottom Bonus
  - 4 Momentum-Signale: 1M smooth(5d) / 3M / 6M / 12M
  - Ensemble = Durchschnitt → Gesamt-Allokation (0.00 - 1.00)
  - 200-Wochen-MA Bottom Bonus: +0.50 wenn BTC < 200WMA

Logik ist 1:1 aus CRYPTO_CIRCLE_V8_WARN_PRODUCTION.py übernommen.
Liest BTC-Historie via:
  1. V16 Sheet (gspread, europäisches Zahlenformat, absteigend)
  2. Fallback: CoinMetrics CSV (GitHub Raw)

Input:  crypto_raw_data.json (aktueller BTC-Preis vom data_collector)
Output: cycle_state dict → wird von signal_engine.py weiterverarbeitet

Usage:
  # Import als Modul:
  from step_0y_crypto.cycle_engine import run_cycle_engine
  result = run_cycle_engine(btc_history, current_btc_price)

  # Standalone Test:
  python cycle_engine.py --skip-write
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
        V8_ENSEMBLE, V8_BOTTOM_BONUS,
        V16_SHEET_ID, V16_TABS,
        API_ENDPOINTS, CONFIG_VERSION,
    )
except (ImportError, ModuleNotFoundError):
    _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/content/step_0y_crypto'
    sys.path.insert(0, _dir)
    sys.path.insert(0, os.path.dirname(_dir))
    from config import (
        V8_ENSEMBLE, V8_BOTTOM_BONUS,
        V16_SHEET_ID, V16_TABS,
        API_ENDPOINTS, CONFIG_VERSION,
    )


# ═══════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════
def log(msg):
    print(f"  [CYCLE] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# BTC HISTORY LOADING
# ═══════════════════════════════════════════════════════

def load_btc_history_v16(gc):
    """BTC-Historie aus V16 Sheet (PRIMARY).
    V16 Sheet hat europäisches Zahlenformat (. = Tausender, , = Dezimal)
    und ist absteigend sortiert (neuestes Datum oben).
    Returns: list of (date_str, price) sorted ascending, or empty list."""
    log("BTC Historie: V16 Sheet...")
    try:
        sh = gc.open_by_key(V16_SHEET_ID)
        ws = sh.worksheet(V16_TABS['prices'])
        vals = ws.get_all_values()
        if len(vals) < 100:
            log("  Zu wenig Zeilen")
            return []
        hdrs = vals[0]
        bc = next((i for i, h in enumerate(hdrs) if 'BTC' in h.upper()), None)
        if bc is None:
            log("  BTC Spalte nicht gefunden")
            return []
        hist = []
        for row in vals[1:]:
            if not row[0].strip():
                continue
            try:
                # Europäisches Format: 71.103 → 71103, 71.103,50 → 71103.50
                s = row[bc].replace('.', '').replace(',', '.') if isinstance(row[bc], str) else str(row[bc])
                v = float(s)
                if v > 0:
                    hist.append((row[0].strip(), v))
            except (ValueError, IndexError):
                continue
        hist.sort(key=lambda x: x[0])
        log(f"  V16: {len(hist)} Tage")
        return hist
    except Exception as e:
        log(f"  V16 ERR: {e}")
        return []


def load_btc_history_coinmetrics():
    """BTC-Historie aus CoinMetrics CSV (FALLBACK).
    BTC ab 2010-07-18 ($0.0858). Gratis, kein Key.
    Returns: list of (date_str, price) sorted ascending, or empty list."""
    log("BTC Historie: CoinMetrics Fallback...")
    try:
        import requests
        url = API_ENDPOINTS.get('coinmetrics_btc',
              'https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv')
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # Parse CSV
        lines = resp.text.strip().split('\n')
        hdrs = lines[0].split(',')
        date_col = next((i for i, h in enumerate(hdrs) if h.strip().lower() == 'time'), 0)
        price_col = next((i for i, h in enumerate(hdrs) if h.strip().lower() == 'priceusd'), None)
        if price_col is None:
            log("  PriceUSD Spalte nicht gefunden")
            return []
        hist = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) <= price_col:
                continue
            try:
                d = parts[date_col].strip()
                p = float(parts[price_col].strip())
                if p > 0 and d:
                    hist.append((d, p))
            except (ValueError, IndexError):
                continue
        hist.sort(key=lambda x: x[0])
        log(f"  CoinMetrics: {len(hist)} Tage")
        return hist
    except Exception as e:
        log(f"  CoinMetrics ERR: {e}")
        return []


def load_btc_history(gc=None):
    """Lade BTC-Historie: V16 Sheet → CoinMetrics Fallback.
    Braucht mindestens 1400 Tage für 200WMA, 252 für 12M Lookback.
    Returns: numpy array of prices (ascending by date), list of date strings."""
    hist = []
    if gc is not None:
        hist = load_btc_history_v16(gc)
    if len(hist) < 252:
        hist = load_btc_history_coinmetrics()
    if len(hist) < 252:
        log(f"WARNUNG: Nur {len(hist)} Tage — Minimum 252 für Ensemble")
        if len(hist) == 0:
            return np.array([]), []

    dates = [d for d, _ in hist]
    prices = np.array([p for _, p in hist])
    log(f"BTC Historie: {len(prices)} Tage, {dates[0]} → {dates[-1]}, "
        f"${prices[0]:,.0f} → ${prices[-1]:,.0f}")
    return prices, dates


# ═══════════════════════════════════════════════════════
# ENSEMBLE BERECHNUNG
# Exakt wie in CRYPTO_CIRCLE_V8_WARN_PRODUCTION.py calc_ensemble()
# ═══════════════════════════════════════════════════════

def calc_ensemble(btc_prices):
    """Berechne BTC Momentum Ensemble + 200WMA Bottom Bonus.

    Args:
        btc_prices: numpy array of BTC daily close prices (ascending)

    Returns:
        dict mit:
          alloc:       float 0.0-1.0 — Gesamt-Allokation (aktuellster Wert)
          ensemble:    float 0.0-1.0 — Ensemble-Wert vor Bottom Bonus
          mom:         dict {1M: bool, 3M: bool, 6M: bool, 12M: bool}
          below_wma:   bool — BTC unter 200WMA
          wma_200:     float — 200WMA Wert
          alloc_series: numpy array — komplette Allokations-Serie
          ensemble_series: numpy array — komplette Ensemble-Serie
    """
    n = len(btc_prices)
    s = pd.Series(btc_prices)

    lookbacks = V8_ENSEMBLE['lookbacks']
    smooth_window = V8_ENSEMBLE['smooth_1m_window']
    wma_days = V8_BOTTOM_BONUS['wma_days']
    wma_min = V8_BOTTOM_BONUS['min_periods']
    bonus = V8_BOTTOM_BONUS['bonus']

    # ─── 4 Momentum-Signale ───
    mom_signals = {}
    for label, lb in lookbacks.items():
        ret = s.pct_change(lb)
        if label == '1M':
            # 5d-Glättung des 21d-Returns (Noise -52%)
            smooth = ret.rolling(smooth_window, min_periods=1).mean()
            mom_signals[label] = (smooth > 0).astype(float).values
        else:
            mom_signals[label] = (ret > 0).astype(float).values

    # ─── Ensemble = Durchschnitt der 4 Signale ───
    ensemble = np.zeros(n)
    for i in range(n):
        vals = [mom_signals[lb][i] for lb in lookbacks]
        if any(np.isnan(v) for v in vals):
            ensemble[i] = np.nan
        else:
            ensemble[i] = np.mean(vals)

    # ─── 200-Wochen-MA Bottom Bonus ───
    wma = s.rolling(wma_days, min_periods=wma_min).mean()
    below_wma = (s < wma).values

    alloc = ensemble.copy()
    for i in range(n):
        if np.isnan(alloc[i]):
            alloc[i] = 0.0
        if below_wma[i]:
            alloc[i] = min(alloc[i] + bonus, 1.0)

    # ─── Aktuellste Werte extrahieren ───
    current_ensemble = ensemble[-1] if not np.isnan(ensemble[-1]) else 0.0
    current_alloc = alloc[-1]
    current_wma = wma.iloc[-1] if pd.notna(wma.iloc[-1]) else None
    current_below = bool(below_wma[-1])

    current_mom = {}
    for label in lookbacks:
        v = mom_signals[label][-1]
        current_mom[label] = bool(v == 1.0) if not np.isnan(v) else False

    return {
        'alloc': round(current_alloc, 4),
        'ensemble': round(current_ensemble, 4),
        'mom': current_mom,
        'below_wma': current_below,
        'wma_200': round(current_wma, 2) if current_wma is not None else None,
        # Serien für weitere Verarbeitung (signal_engine, risk_engine)
        'alloc_series': alloc,
        'ensemble_series': ensemble,
        'mom_series': mom_signals,
        'below_wma_series': below_wma,
        'wma_series': wma.values,
    }


# ═══════════════════════════════════════════════════════
# HAUPTFUNKTION
# ═══════════════════════════════════════════════════════

def run_cycle_engine(btc_prices, dates=None, current_btc_price=None):
    """V8+Warn Cycle Engine — Komponente 1.

    Args:
        btc_prices:       numpy array of BTC daily close prices (ascending)
        dates:            list of date strings (optional, für Logging)
        current_btc_price: float (optional, für Logging)

    Returns:
        dict mit Ensemble-Ergebnis + Metadaten
    """
    log("=" * 50)
    log("CYCLE ENGINE V1.0 — BTC Momentum Ensemble")
    log("=" * 50)

    n = len(btc_prices)
    if n < 252:
        log(f"FEHLER: Nur {n} Tage — brauche mindestens 252 für 12M Lookback")
        return {
            'status': 'ERROR',
            'error': f'Zu wenig BTC-Historie: {n} Tage (min 252)',
            'alloc': 0.0,
            'ensemble': 0.0,
            'mom': {'1M': False, '3M': False, '6M': False, '12M': False},
            'below_wma': False,
            'wma_200': None,
        }

    # Ensemble berechnen
    result = calc_ensemble(btc_prices)

    # Logging
    btc = current_btc_price or btc_prices[-1]
    log(f"  BTC:       ${btc:>10,.0f}")
    log(f"  Ensemble:  {result['ensemble']:.2f} "
        f"(1M={'✅' if result['mom']['1M'] else '❌'} "
        f"3M={'✅' if result['mom']['3M'] else '❌'} "
        f"6M={'✅' if result['mom']['6M'] else '❌'} "
        f"12M={'✅' if result['mom']['12M'] else '❌'})")
    wma_str = f"${result['wma_200']:,.0f}" if result['wma_200'] else 'N/A'
    bonus_str = " ← Bottom Bonus aktiv" if result['below_wma'] else ""
    log(f"  200WMA:    {wma_str}{bonus_str}")
    log(f"  Allokation: {result['alloc']:.0%}")

    # Metadaten
    result['status'] = 'OK'
    result['btc_price'] = round(btc, 2)
    result['btc_history_days'] = n
    result['has_200wma'] = n >= V8_BOTTOM_BONUS['min_periods']
    result['engine_version'] = 'cycle_engine V1.0'
    result['config_version'] = CONFIG_VERSION

    return result


# ═══════════════════════════════════════════════════════
# STANDALONE EXECUTION (Test)
# ═══════════════════════════════════════════════════════

def main():
    """Standalone Test — lädt BTC-Historie und berechnet Ensemble."""
    import argparse
    parser = argparse.ArgumentParser(description='Crypto Circle Cycle Engine V1.0')
    parser.add_argument('--skip-write', action='store_true', help='Kein Sheet Write')
    parser.add_argument('--skip-v16', action='store_true', help='Kein V16 Sheet Read')
    args = parser.parse_args()

    t0 = datetime.now(timezone.utc)
    print("=" * 60)
    print("CYCLE ENGINE V1.0 — Standalone Test")
    print(f"  {t0.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # GCP Auth
    gc = None
    if not args.skip_v16:
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            creds_json = os.environ.get('GOOGLE_CREDENTIALS', '')
            if creds_json:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(creds_json)
                    creds_path = f.name
                creds = Credentials.from_service_account_file(
                    creds_path,
                    scopes=['https://www.googleapis.com/auth/spreadsheets',
                            'https://www.googleapis.com/auth/drive']
                )
                gc = gspread.authorize(creds)
                log("GCP Auth OK")
            else:
                log("GOOGLE_CREDENTIALS nicht gesetzt → CoinMetrics Fallback")
        except Exception as e:
            log(f"GCP Auth ERR: {e} → CoinMetrics Fallback")

    # BTC Historie laden
    btc_prices, dates = load_btc_history(gc)
    if len(btc_prices) == 0:
        log("ABBRUCH: Keine BTC-Historie")
        sys.exit(1)

    # Aktuellen Preis aus crypto_raw_data.json lesen (wenn vorhanden)
    current_btc = None
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'crypto_raw_data.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                raw = json.load(f)
            current_btc = raw.get('data', {}).get('btc_price')
            if current_btc:
                log(f"Aktueller BTC (JSON): ${current_btc:,.2f}")
        except Exception:
            pass

    # Engine
    result = run_cycle_engine(btc_prices, dates, current_btc)

    # Output
    print(f"\n{'='*60}")
    print(f"ERGEBNIS:")
    print(f"  Status:     {result['status']}")
    print(f"  Allokation: {result['alloc']:.0%}")
    print(f"  Ensemble:   {result['ensemble']:.2f}")
    print(f"  Momentum:   1M={'✅' if result['mom']['1M'] else '❌'} "
          f"3M={'✅' if result['mom']['3M'] else '❌'} "
          f"6M={'✅' if result['mom']['6M'] else '❌'} "
          f"12M={'✅' if result['mom']['12M'] else '❌'}")
    if result['wma_200']:
        print(f"  200WMA:     ${result['wma_200']:,.0f} "
              f"{'← UNTER (Bonus aktiv)' if result['below_wma'] else '← ÜBER'}")
    print(f"  BTC Tage:   {result['btc_history_days']}")
    print(f"{'='*60}")

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"Fertig in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
