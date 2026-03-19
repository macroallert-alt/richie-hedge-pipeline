#!/usr/bin/env python3
"""
crypto_orchestrator.py — Crypto Circle Weekly Orchestrator V1.0
================================================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

Führt die wöchentliche Pipeline aus:
  1. data_collector.py → Rohdaten sammeln (BG, CoinGecko, Binance, V16)
  2. cycle_engine.py   → BTC Momentum Ensemble + 200WMA Bottom Bonus
  3. signal_engine.py  → Trickle-Down Phase aus BTC.D
  4. risk_engine.py    → Phase 4 Warning + NO-ACTION Band + finale Allokation
  5. Sheet Write       → CALC_Cycle, CALC_Risk, CALC_Allocation, IND_Market
  6. JSON Output       → crypto_state.json (für Frontend)
  7. Git Commit        → Automatisch in GitHub Actions

Datenfluss:
  data_collector → crypto_raw_data.json
                     ↓
  cycle_engine  ← BTC Historie (V16 Sheet / CoinMetrics)
                     ↓ (alloc, ensemble, mom, wma)
  signal_engine ← BTC.D (aus crypto_raw_data.json)
                     ↓ (phase, weights)
  risk_engine   ← cycle_result + signal_result + vorherige Allokation
                     ↓ (finale Positionen, Action)
  Sheet Write   → CALC_Cycle + CALC_Risk + CALC_Allocation + IND_Market
  JSON Write    → crypto_state.json

Usage:
  # GitHub Actions (Sonntag 05:00 UTC):
  python -m step_0y_crypto.crypto_orchestrator

  # Colab Test (kein Sheet Write):
  python crypto_orchestrator.py --skip-write

  # Colab Test (kein V16, kein Write):
  python crypto_orchestrator.py --skip-write --skip-v16
"""
import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════
# CONFIG IMPORTS
# ═══════════════════════════════════════════════════════
try:
    from step_0y_crypto.config import (
        CRYPTO_SHEET_ID, V16_SHEET_ID, V16_TABS,
        CRYPTO_TABS, CONFIG_VERSION,
        V8_ENSEMBLE, V8_BOTTOM_BONUS, V8_TRICKLE_DOWN,
        V8_PHASE4_WARNING, V8_NO_ACTION_BAND,
        V16_STATE_NAMES,
    )
    from step_0y_crypto.cycle_engine import calc_ensemble, load_btc_history
    from step_0y_crypto.signal_engine import (
        calc_phase_single, get_tier_weights, PHASE_NAMES,
    )
    from step_0y_crypto.risk_engine import (
        apply_phase4_warning, calc_positions, check_no_action_band,
    )
except (ImportError, ModuleNotFoundError):
    _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/content/step_0y_crypto'
    sys.path.insert(0, _dir)
    sys.path.insert(0, os.path.dirname(_dir))
    from config import (
        CRYPTO_SHEET_ID, V16_SHEET_ID, V16_TABS,
        CRYPTO_TABS, CONFIG_VERSION,
        V8_ENSEMBLE, V8_BOTTOM_BONUS, V8_TRICKLE_DOWN,
        V8_PHASE4_WARNING, V8_NO_ACTION_BAND,
        V16_STATE_NAMES,
    )
    from cycle_engine import calc_ensemble, load_btc_history
    from signal_engine import (
        calc_phase_single, get_tier_weights, PHASE_NAMES,
    )
    from risk_engine import (
        apply_phase4_warning, calc_positions, check_no_action_band,
    )

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

NOW = datetime.now(timezone.utc)
VERSION = "crypto_orchestrator V1.0"

# Basispfad — einmal festlegen, überall nutzen
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


# ═══════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════
def log(msg):
    print(f"  [ORCH] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# GCP AUTH (gleiche Logik wie data_collector)
# ═══════════════════════════════════════════════════════
def get_gspread_client():
    """GCP Auth — sucht SA Key in bekannten Pfaden + Env."""
    if not HAS_GSPREAD:
        return None
    scopes = ['https://www.googleapis.com/auth/spreadsheets',
              'https://www.googleapis.com/auth/drive']
    # Datei-basiert
    for p in ['/content/service_account.json',
              '/content/drive/MyDrive/keys/service_account.json',
              'service_account.json']:
        if os.path.exists(p):
            gc = gspread.authorize(Credentials.from_service_account_file(p, scopes=scopes))
            log(f"SA: {p}")
            return gc
    # Env-basiert (GitHub Actions)
    sa = os.environ.get('GOOGLE_CREDENTIALS') or os.environ.get('GCP_SA_KEY')
    if sa:
        gc = gspread.authorize(Credentials.from_service_account_info(
            json.loads(sa), scopes=scopes))
        log("SA: Env")
        return gc
    return None


# ═══════════════════════════════════════════════════════
# ROHDATEN LADEN
# ═══════════════════════════════════════════════════════
def load_raw_data():
    """Lade crypto_raw_data.json vom data_collector."""
    json_path = os.path.join(DATA_DIR, 'crypto_raw_data.json')
    if not os.path.exists(json_path):
        log(f"WARNUNG: {json_path} nicht gefunden")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    data = raw.get('data', {})
    log(f"Rohdaten: {len(data)} Felder, {raw.get('metadata',{}).get('degradation','?')}")
    return data


def load_previous_allocation(gc):
    """Lade vorherige Allokation aus CALC_Allocation Sheet (letzte Datenzeile)."""
    if gc is None:
        return None
    try:
        sh = gc.open_by_key(CRYPTO_SHEET_ID)
        ws = sh.worksheet(CRYPTO_TABS['calc_alloc'])
        vals = ws.get_all_values()
        if len(vals) < 3:
            log("Keine vorherige Allokation im Sheet")
            return None
        # Letzte Datenzeile (Row 2 = Header, Row 3+ = Daten)
        last_row = vals[-1]
        hdrs = vals[1]  # Row 2 = Spalten-Header

        def col_val(name):
            try:
                idx = hdrs.index(name)
                v = last_row[idx]
                return float(v) if v else None
            except (ValueError, IndexError):
                return None

        prev = {
            'btc': col_val('BTC_Position%'),
            'eth': col_val('ETH_Position%'),
            'sol': col_val('SOL_Position%'),
            'cash': col_val('Cash%'),
            'total_invested': col_val('Final_Crypto_Exposure'),
        }
        # Prüfe ob gültige Werte
        if prev['total_invested'] is not None:
            log(f"Vorherige Allokation: {prev['total_invested']:.0%} "
                f"(BTC={prev.get('btc',0) or 0:.1%} ETH={prev.get('eth',0) or 0:.1%} "
                f"SOL={prev.get('sol',0) or 0:.1%})")
            return prev
        log("Keine gültige vorherige Allokation")
        return None
    except Exception as e:
        log(f"Vorherige Allokation ERR: {e}")
        return None


# ═══════════════════════════════════════════════════════
# SHEET WRITE — 4 Tabs
# ═══════════════════════════════════════════════════════

def clean_val(v):
    """Wert für Sheet vorbereiten (USER_ENTERED mit US Locale)."""
    if v is None or v == '':
        return ''
    elif isinstance(v, bool):
        return 'TRUE' if v else 'FALSE'
    elif isinstance(v, float):
        return f'{v}'
    elif isinstance(v, (int, np.integer)):
        return str(int(v))
    elif isinstance(v, np.floating):
        return f'{float(v)}'
    elif isinstance(v, np.bool_):
        return 'TRUE' if v else 'FALSE'
    else:
        return str(v)


def write_calc_cycle(ws, raw_data, cycle_result):
    """Schreibe CALC_Cycle Tab (18 Spalten)."""
    row = [
        # [1-2] META
        NOW.strftime('%Y-%m-%d'),
        raw_data.get('btc_price', ''),
        # [3-7] MOMENTUM SIGNALE
        '',  # Ret_1M_Smooth — nicht direkt verfügbar als Einzelwert
        cycle_result['mom']['1M'],
        cycle_result['mom']['3M'],
        cycle_result['mom']['6M'],
        cycle_result['mom']['12M'],
        # [8-9] ENSEMBLE
        cycle_result['ensemble'],
        sum(1 for v in cycle_result['mom'].values() if v),
        # [10-12] 200-WOCHEN-MA
        cycle_result['wma_200'],
        cycle_result['below_wma'],
        cycle_result['below_wma'],  # Bottom Bonus = Below WMA
        # [13-14] ALLOKATION
        cycle_result['alloc'],
        'ENSEMBLE+BONUS' if cycle_result['below_wma'] else 'ENSEMBLE',
        # [15-18] DISPLAY
        raw_data.get('rainbow_band', ''),
        raw_data.get('rainbow_score', ''),
        raw_data.get('v16_macro_state', ''),
        raw_data.get('v16_state_name', ''),
    ]
    ws.append_row([clean_val(v) for v in row], value_input_option='USER_ENTERED')
    log(f"  CALC_Cycle: {len(row)} Werte ✅")


def write_calc_risk(ws, raw_data, signal_result, risk_result):
    """Schreibe CALC_Risk Tab (16 Spalten)."""
    p4 = risk_result['components']['phase4_warning']
    exec_plan = risk_result.get('execution', {})

    row = [
        # [1-2] META
        NOW.strftime('%Y-%m-%d'),
        raw_data.get('btc_price', ''),
        # [3-5] PHASE 4 WARNING
        signal_result['phase'],
        p4['active'],
        V8_PHASE4_WARNING['multiplier'] if p4['active'] else 1.0,
        # [6-9] NO-ACTION BAND
        risk_result['allocation']['total_invested'],
        risk_result.get('deltas', {}).get('total', ''),
        risk_result.get('deltas', {}).get('total', ''),
        risk_result['action'],
        # [10-13] EXECUTION
        exec_plan.get('type', ''),
        exec_plan.get('week1', {}).get('total_invested', ''),
        exec_plan.get('week2', {}).get('total_invested', ''),
        exec_plan.get('week3', {}).get('total_invested', ''),
        # [14-16] DISPLAY
        raw_data.get('btc_realvol_60d', ''),
        raw_data.get('fear_greed_7d_avg', ''),
        raw_data.get('funding_btc_3d_avg', ''),
    ]
    ws.append_row([clean_val(v) for v in row], value_input_option='USER_ENTERED')
    log(f"  CALC_Risk: {len(row)} Werte ✅")


def write_calc_allocation(ws, raw_data, cycle_result, signal_result, risk_result, prev_alloc):
    """Schreibe CALC_Allocation Tab (22 Spalten)."""
    a = risk_result['allocation']
    p4 = risk_result['components']['phase4_warning']
    w = signal_result['weights']
    w_str = f"{int(w['BTC']*100)}/{int(w['ETH']*100)}/{int(w['SOL']*100)}"

    prev_exp = prev_alloc.get('total_invested', '') if prev_alloc else ''
    delta = ''
    if prev_alloc and prev_alloc.get('total_invested') is not None:
        delta = round((a['total_invested'] - prev_alloc['total_invested']) * 100, 1)

    row = [
        # [1-2] META
        NOW.strftime('%Y-%m-%d'),
        NOW.strftime('%Y-%m-%d %H:%M UTC'),
        # [3-7] V8 ENSEMBLE
        cycle_result['ensemble'],
        cycle_result['mom']['1M'],
        cycle_result['mom']['3M'],
        cycle_result['mom']['6M'],
        cycle_result['mom']['12M'],
        # [8-12] TRICKLE-DOWN
        signal_result.get('btc_dominance', ''),
        signal_result.get('btc_d_change', ''),
        signal_result['phase'],
        signal_result['phase_name'],
        w_str,
        # [13-14] PHASE 4 WARNING
        p4['active'],
        p4['alloc_before'],
        # [15-19] FINALE ALLOKATION
        a['total_invested'],
        a['btc'],
        a['eth'],
        a['sol'],
        a['cash'],
        # [20-22] NO-ACTION BAND
        prev_exp,
        delta,
        risk_result['action'],
    ]
    ws.append_row([clean_val(v) for v in row], value_input_option='USER_ENTERED')
    log(f"  CALC_Allocation: {len(row)} Werte ✅")


def write_ind_market(ws, raw_data, signal_result):
    """Schreibe IND_Market Tab (28 Spalten)."""
    w = signal_result['weights']

    row = [
        # [1-2] META
        NOW.strftime('%Y-%m-%d'),
        raw_data.get('btc_price', ''),
        # [3-9] TRICKLE-DOWN
        signal_result.get('btc_dominance', ''),
        signal_result.get('btc_d_change', ''),
        signal_result['phase'],
        signal_result['phase_name'],
        w['BTC'], w['ETH'], w['SOL'],
        # [10-13] BTC DOMINANCE DETAIL
        raw_data.get('btc_dom_7d_chg', ''),
        '',  # 90d change — noch nicht berechnet
        'CoinGecko',
        '',  # Proxy CF — nur bei Proxy relevant
        # [14-17] TIER PREISE
        raw_data.get('eth_price', ''),
        raw_data.get('sol_price', ''),
        raw_data.get('eth_24h_chg', ''),
        raw_data.get('sol_24h_chg', ''),
        # [18-21] TIER RATIOS
        raw_data.get('eth_btc_ratio', ''),
        raw_data.get('sol_btc_ratio', ''),
        '',  # ETH/BTC 30d Mom — Engine berechnet
        '',  # SOL/BTC 30d Mom — Engine berechnet
        # [22-25] DISPLAY: LIQUIDITÄT
        raw_data.get('stablecoin_supply_total', ''),
        raw_data.get('etf_balance_bg', ''),
        raw_data.get('howell_liq_dir', ''),
        '',  # M2 YoY — Engine berechnet
        # [26-28] DISPLAY: DERIVATE
        raw_data.get('funding_btc_3d_avg', ''),
        raw_data.get('oi_mcap_ratio', ''),
        raw_data.get('liquidations_24h', ''),
    ]
    ws.append_row([clean_val(v) for v in row], value_input_option='USER_ENTERED')
    log(f"  IND_Market: {len(row)} Werte ✅")


def write_all_sheets(gc, raw_data, cycle_result, signal_result, risk_result, prev_alloc):
    """Schreibe alle 4 Engine-Tabs."""
    log("Sheet Write...")
    try:
        sh = gc.open_by_key(CRYPTO_SHEET_ID)

        ws_cycle = sh.worksheet(CRYPTO_TABS['calc_cycle'])
        write_calc_cycle(ws_cycle, raw_data, cycle_result)
        time.sleep(1.0)

        ws_risk = sh.worksheet(CRYPTO_TABS['calc_risk'])
        write_calc_risk(ws_risk, raw_data, signal_result, risk_result)
        time.sleep(1.0)

        ws_alloc = sh.worksheet(CRYPTO_TABS['calc_alloc'])
        write_calc_allocation(ws_alloc, raw_data, cycle_result, signal_result, risk_result, prev_alloc)
        time.sleep(1.0)

        ws_market = sh.worksheet(CRYPTO_TABS['ind_market'])
        write_ind_market(ws_market, raw_data, signal_result)

        log("Sheet Write KOMPLETT ✅")
    except Exception as e:
        log(f"Sheet Write ERR: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# JSON OUTPUT — crypto_state.json
# ═══════════════════════════════════════════════════════
def write_state_json(raw_data, cycle_result, signal_result, risk_result):
    """Schreibe crypto_state.json (für Frontend + Git)."""
    a = risk_result['allocation']
    state = {
        'version': VERSION,
        'config_version': CONFIG_VERSION,
        'timestamp': NOW.isoformat(),
        'date': NOW.strftime('%Y-%m-%d'),

        'btc_price': raw_data.get('btc_price'),
        'eth_price': raw_data.get('eth_price'),
        'sol_price': raw_data.get('sol_price'),

        'ensemble': {
            'value': cycle_result['ensemble'],
            'mom_1M': cycle_result['mom']['1M'],
            'mom_3M': cycle_result['mom']['3M'],
            'mom_6M': cycle_result['mom']['6M'],
            'mom_12M': cycle_result['mom']['12M'],
        },
        'bottom_bonus': {
            'active': cycle_result['below_wma'],
            'wma_200': cycle_result['wma_200'],
        },
        'trickle_down': {
            'btc_dominance': signal_result.get('btc_dominance'),
            'btc_d_30d_change': signal_result.get('btc_d_change'),
            'phase': signal_result['phase'],
            'phase_name': signal_result['phase_name'],
            'phase4_warning': risk_result['components']['phase4_warning']['active'],
        },
        'allocation': {
            'total': a['total_invested'],
            'btc': a['btc'],
            'eth': a['eth'],
            'sol': a['sol'],
            'cash': a['cash'],
        },
        'weights': signal_result['weights'],
        'action': risk_result['action'],

        'display': {
            'rainbow_band': raw_data.get('rainbow_band'),
            'rainbow_score': raw_data.get('rainbow_score'),
            'mvrv_zscore': raw_data.get('mvrv_zscore'),
            'nupl': raw_data.get('nupl'),
            'fear_greed': raw_data.get('fear_greed_7d_avg'),
            'v16_macro_state': raw_data.get('v16_state_name'),
            'funding_btc': raw_data.get('funding_btc_3d_avg'),
            'halving_phase': raw_data.get('halving_phase'),
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, 'crypto_state.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    log(f"JSON: {p}")
    return state


# ═══════════════════════════════════════════════════════
# MAIN — ORCHESTRATOR
# ═══════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description='Crypto Circle Weekly Orchestrator V1.0')
    pa.add_argument('--skip-write', action='store_true', help='Kein Sheet Write')
    pa.add_argument('--skip-v16', action='store_true', help='Kein V16 Sheet Read')
    pa.add_argument('--skip-collector', action='store_true',
                    help='data_collector überspringen (nutze bestehende crypto_raw_data.json)')
    args = pa.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"CRYPTO CIRCLE — WEEKLY ORCHESTRATOR {VERSION}")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  System: V8+Warn (Ensemble + Trickle-Down + P4 Warning)")
    print(f"  Config: V{CONFIG_VERSION}")
    print(f"  Flags: skip-write={args.skip_write} skip-v16={args.skip_v16} "
          f"skip-collector={args.skip_collector}")
    print("=" * 70)

    # ─── GCP Auth ───
    gc = None
    if HAS_GSPREAD and not args.skip_write:
        gc = get_gspread_client()
        if gc is None:
            log("WARNUNG: Kein GCP Auth — Sheet Write deaktiviert")

    # ─── Schritt 1: Data Collector ───
    if not args.skip_collector:
        print(f"\n{'='*70}")
        print("SCHRITT 1: DATA COLLECTOR")
        print(f"{'='*70}")
        try:
            from step_0y_crypto.data_collector import main as dc_main
        except ImportError:
            from data_collector import main as dc_main

        # data_collector hat eigenes argparse — wir rufen es direkt auf
        # Trick: sys.argv temporär überschreiben
        import copy
        old_argv = copy.copy(sys.argv)
        dc_args = ['data_collector.py']
        if args.skip_write:
            dc_args.append('--skip-write')
        if args.skip_v16:
            dc_args.append('--skip-v16')
        sys.argv = dc_args
        try:
            dc_main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        log("Data Collector fertig")
    else:
        log("Data Collector übersprungen (--skip-collector)")

    # ─── Rohdaten laden ───
    raw_data = load_raw_data()
    if not raw_data.get('btc_price'):
        log("ABBRUCH: Kein BTC-Preis in Rohdaten")
        sys.exit(1)

    # ─── Schritt 2: BTC Historie + Cycle Engine ───
    print(f"\n{'='*70}")
    print("SCHRITT 2: CYCLE ENGINE (Momentum Ensemble)")
    print(f"{'='*70}")

    btc_prices, dates = load_btc_history(gc if not args.skip_v16 else None)
    if len(btc_prices) < 252:
        log(f"ABBRUCH: Nur {len(btc_prices)} Tage BTC-Historie (min 252)")
        sys.exit(1)

    cycle_result = calc_ensemble(btc_prices)
    log(f"Ensemble: {cycle_result['ensemble']:.2f}, "
        f"Alloc: {cycle_result['alloc']:.0%}, "
        f"200WMA: {'UNTER' if cycle_result['below_wma'] else 'ÜBER'}")

    # ─── Schritt 3: Signal Engine (Trickle-Down) ───
    print(f"\n{'='*70}")
    print("SCHRITT 3: SIGNAL ENGINE (Trickle-Down)")
    print(f"{'='*70}")

    btc_d_today = raw_data.get('btc_dominance')
    # BTC.D vor 30 Tagen: aus vorheriger JSON oder Sheet
    # Erste Woche: kein vorheriger Wert → Phase 2 (Default)
    btc_d_30d_ago = None

    # Versuche vorherige Rohdaten
    prev_json_path = os.path.join(DATA_DIR, 'crypto_raw_data_prev.json')
    if os.path.exists(prev_json_path):
        try:
            with open(prev_json_path, 'r') as f:
                prev_raw = json.load(f)
            btc_d_30d_ago = prev_raw.get('data', {}).get('btc_dominance')
            if btc_d_30d_ago:
                log(f"BTC.D 30d ago (prev JSON): {btc_d_30d_ago:.1f}%")
        except Exception:
            pass

    # Fallback: Aus IND_Market Sheet letzte Zeile lesen
    if btc_d_30d_ago is None and gc is not None:
        try:
            sh = gc.open_by_key(CRYPTO_SHEET_ID)
            ws = sh.worksheet(CRYPTO_TABS['ind_market'])
            vals = ws.get_all_values()
            if len(vals) >= 3:
                hdrs = vals[1]
                bd_col = next((i for i, h in enumerate(hdrs) if h == 'BTC_Dominance'), None)
                if bd_col is not None:
                    # Suche 4 Wochen zurück (ca. 30 Tage, wöchentliche Daten)
                    for row in reversed(vals[2:]):
                        try:
                            v = float(row[bd_col])
                            if v > 0:
                                btc_d_30d_ago = v
                                log(f"BTC.D 30d ago (Sheet): {btc_d_30d_ago:.1f}%")
                                break
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            log(f"BTC.D Sheet read ERR: {e}")

    if btc_d_today is not None:
        if btc_d_30d_ago is not None:
            phase, btc_d_change = calc_phase_single(btc_d_today, btc_d_30d_ago)
        else:
            log("Kein BTC.D 30d ago → Phase 2 (Default)")
            phase = V8_TRICKLE_DOWN['default_phase']
            btc_d_change = 0.0
    else:
        log("Kein BTC.D → Phase 2 (Default)")
        phase = V8_TRICKLE_DOWN['default_phase']
        btc_d_change = 0.0

    weights = get_tier_weights(phase)
    signal_result = {
        'status': 'OK',
        'phase': phase,
        'phase_name': PHASE_NAMES[phase],
        'weights': weights,
        'btc_dominance': round(btc_d_today, 2) if btc_d_today else None,
        'btc_d_change': round(btc_d_change, 2),
    }
    log(f"Phase: {phase} ({PHASE_NAMES[phase]}), "
        f"BTC.D: {btc_d_today or '?'}%, Δ: {btc_d_change:+.1f}pp")
    log(f"Gewichte: BTC={weights['BTC']:.0%} ETH={weights['ETH']:.0%} SOL={weights['SOL']:.0%}")

    # ─── Schritt 4: Risk Engine ───
    print(f"\n{'='*70}")
    print("SCHRITT 4: RISK ENGINE (Phase 4 + NO-ACTION)")
    print(f"{'='*70}")

    # Vorherige Allokation laden
    prev_alloc = load_previous_allocation(gc) if gc else None

    # Phase 4 Warning
    raw_alloc = cycle_result['alloc']
    final_alloc, p4_active = apply_phase4_warning(raw_alloc, phase)

    if p4_active:
        log(f"⚠️  PHASE 4 WARNING: {raw_alloc:.0%} × {V8_PHASE4_WARNING['multiplier']} = {final_alloc:.0%}")
    else:
        log(f"Allokation: {final_alloc:.0%} (keine P4 Warning)")

    # Finale Positionen
    positions = calc_positions(final_alloc, weights)

    # NO-ACTION Band
    action, deltas, reason = check_no_action_band(positions, prev_alloc)
    log(f"Action: {action} — {reason}")

    # Risk Result zusammenbauen
    risk_result = {
        'status': 'OK',
        'allocation': positions,
        'action': action,
        'action_reason': reason,
        'deltas': deltas,
        'components': {
            'phase4_warning': {
                'active': p4_active,
                'multiplier': V8_PHASE4_WARNING['multiplier'],
                'alloc_before': raw_alloc,
                'alloc_after': final_alloc,
            },
        },
        'execution': {},
    }

    log(f"Positionen: BTC={positions['btc']:.1%} ETH={positions['eth']:.1%} "
        f"SOL={positions['sol']:.1%} Cash={positions['cash']:.1%}")

    # ─── Schritt 5: Output ───
    print(f"\n{'='*70}")
    print("SCHRITT 5: OUTPUT")
    print(f"{'='*70}")

    # JSON
    state = write_state_json(raw_data, cycle_result, signal_result, risk_result)

    # Sheet Write
    if gc and not args.skip_write:
        write_all_sheets(gc, raw_data, cycle_result, signal_result, risk_result, prev_alloc)
    elif args.skip_write:
        log("Sheet Write übersprungen (--skip-write)")
    else:
        log("Sheet Write übersprungen (kein GCP Auth)")

    # Vorherige Rohdaten archivieren (für nächste Woche BTC.D 30d)
    raw_json_path = os.path.join(DATA_DIR, 'crypto_raw_data.json')
    prev_path = os.path.join(DATA_DIR, 'crypto_raw_data_prev.json')
    if os.path.exists(raw_json_path):
        import shutil
        shutil.copy2(raw_json_path, prev_path)
        log(f"Prev JSON: {prev_path}")

    # ─── Zusammenfassung ───
    elapsed = time.time() - t0
    a = state['allocation']
    print(f"\n{'='*70}")
    print(f"CRYPTO CIRCLE — FERTIG ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Ensemble:  {state['ensemble']['value']:.2f} "
          f"(1M={'✅' if state['ensemble']['mom_1M'] else '❌'} "
          f"3M={'✅' if state['ensemble']['mom_3M'] else '❌'} "
          f"6M={'✅' if state['ensemble']['mom_6M'] else '❌'} "
          f"12M={'✅' if state['ensemble']['mom_12M'] else '❌'})")
    wma_str = f"${state['bottom_bonus']['wma_200']:,.0f}" if state['bottom_bonus']['wma_200'] else 'N/A'
    print(f"  200WMA:    {wma_str} {'← UNTER (Bonus)' if state['bottom_bonus']['active'] else ''}")
    print(f"  Phase:     {state['trickle_down']['phase']} ({state['trickle_down']['phase_name']})")
    if state['trickle_down']['phase4_warning']:
        print(f"  ⚠️  PHASE 4 WARNING AKTIV")
    print(f"  Allok:     {a['total']:.0%}")
    print(f"  Positionen: BTC={a['btc']:.1%} ETH={a['eth']:.1%} "
          f"SOL={a['sol']:.1%} Cash={a['cash']:.1%}")
    print(f"  Action:    {state['action']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
