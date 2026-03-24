#!/usr/bin/env python3
"""
daily_risk_check.py — Crypto Circle Daily Signal Engine V2.0
================================================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

V2.0 — Voller Signal-Pfad (täglich statt nur Ensemble-Vergleich)
  Änderungen V1.0 → V2.0:
    - NEU: signal_engine (Phase-Berechnung aus BTC.D)
    - NEU: risk_engine (Phase 4 Warning + Positionen + NO-ACTION Band)
    - NEU: crypto_state.json wird TÄGLICH überschrieben (Frontend immer aktuell)
    - NEU: CALC_Allocation Sheet-Zeile täglich
    - NEU: BTC.D 30d Lookback aus HIST_Daily_Risk Sheet
    - NEU: Vorherige Allokation aus CALC_Allocation Sheet für NO-ACTION Band
    - NEU: Telegram Alert zeigt jetzt aktuelle + vorherige Positionen
    - BEIBEHALTEN: Alert-Logik (Ensemble-Sprung, WMA-Cross, BTC.D-Delta)
    - BEIBEHALTEN: crypto_daily_check.json (für Briefing, erweitert)
    - BEIBEHALTEN: HIST_Daily_Risk Sheet (erweitert um Daily-Positionen)

Was es tut:
  1. BTC/ETH/SOL-Preise von CoinGecko holen (1 API Call)
  2. BTC.D von CoinGecko Global holen (1 API Call)
  3. BTC-Historie aus V16 Sheet laden
  4. Ensemble berechnen (cycle_engine.calc_ensemble)
  5. BTC.D 30d-Lookback aus HIST_Daily_Risk Sheet
  6. Phase berechnen (signal_engine.calc_phase_single)
  7. Phase 4 Warning + Positionen + NO-ACTION Band (risk_engine)
  8. crypto_state.json überschreiben (Frontend)
  9. CALC_Allocation Sheet-Zeile schreiben
  10. Alert-Check: Daily vs. vorherige crypto_state.json
  11. Telegram Alert bei Trigger
  12. crypto_daily_check.json schreiben (für Briefing)
  13. HIST_Daily_Risk Sheet-Zeile schreiben

Trigger-Kette:
  V16 Daily Runner (06:00 UTC) → ... → Step 7
    → step0y_crypto_daily.yml (workflow_dispatch)
      → daily_risk_check.py
        → crypto_state.json (täglich aktuell)
        → CALC_Allocation (1 Zeile/Tag)
        → Telegram Alert (wenn Trigger)
      → step0u_briefing.yml (workflow_dispatch)

Usage:
  # GitHub Actions (von Step 7 getriggert):
  python -m step_0y_crypto.daily_risk_check

  # Colab Test:
  python daily_risk_check.py --skip-write --skip-telegram

  # Colab Test (kein V16, kein Write):
  python daily_risk_check.py --skip-write --skip-telegram --skip-v16
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
        API_ENDPOINTS,
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
        API_ENDPOINTS,
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
VERSION = "daily_risk_check V2.0"

# Basispfad — einmal festlegen, überall nutzen
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


# ═══════════════════════════════════════════════════════
# ALERT-TRIGGER SCHWELLENWERTE
# ═══════════════════════════════════════════════════════

# Ensemble-Sprung: Jede Änderung des Ensemble-Werts (0.25er Stufen)
# ist signifikant, weil sie die Gesamt-Allokation direkt ändert.
ENSEMBLE_CHANGE_THRESHOLD = 0.01  # Jeder Sprung (0.25→0.50 etc.)

# BTC.D Wöchentliche Veränderung: >3pp deutet auf Phase-Wechsel
BTC_D_WEEKLY_CHANGE_THRESHOLD = 3.0  # pp


# ═══════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════
def log(msg):
    print(f"  [DAILY] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# GCP AUTH (gleiche Logik wie crypto_orchestrator)
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
# PREIS-DATEN VON COINGECKO (1 API Call für BTC+ETH+SOL)
# ═══════════════════════════════════════════════════════
def fetch_prices():
    """Aktuelle BTC/ETH/SOL-Preise von CoinGecko holen."""
    import requests
    url = API_ENDPOINTS.get('coingecko_prices',
          'https://api.coingecko.com/api/v3/simple/price')
    params = {
        'ids': 'bitcoin,ethereum,solana',
        'vs_currencies': 'usd',
        'include_24hr_change': 'true',
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        btc = data.get('bitcoin', {}).get('usd')
        eth = data.get('ethereum', {}).get('usd')
        sol = data.get('solana', {}).get('usd')
        if btc and btc > 0:
            log(f"Preise (CoinGecko): BTC=${btc:,.0f}  ETH=${eth:,.0f}  SOL=${sol:,.2f}")
            return {
                'btc_price': float(btc),
                'eth_price': float(eth) if eth else None,
                'sol_price': float(sol) if sol else None,
            }
    except Exception as e:
        log(f"CoinGecko Preise ERR: {e}")

    # Fallback: nur BTC
    try:
        import requests as req2
        params_btc = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
        resp2 = req2.get(url, params=params_btc, timeout=15)
        resp2.raise_for_status()
        data2 = resp2.json()
        btc2 = data2.get('bitcoin', {}).get('usd')
        if btc2 and btc2 > 0:
            log(f"Fallback BTC: ${btc2:,.0f} (ETH/SOL nicht verfügbar)")
            return {'btc_price': float(btc2), 'eth_price': None, 'sol_price': None}
    except Exception as e2:
        log(f"CoinGecko Fallback ERR: {e2}")

    return None


# ═══════════════════════════════════════════════════════
# BTC.D VON COINGECKO (1 API Call)
# ═══════════════════════════════════════════════════════
def fetch_btc_dominance():
    """Aktuelle BTC Dominance von CoinGecko holen."""
    import requests
    url = API_ENDPOINTS.get('coingecko_global',
          'https://api.coingecko.com/api/v3/global')
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        btc_d = data.get('data', {}).get('market_cap_percentage', {}).get('btc')
        if btc_d and btc_d > 0:
            log(f"BTC.D (CoinGecko): {btc_d:.1f}%")
            return round(float(btc_d), 2)
    except Exception as e:
        log(f"BTC.D ERR: {e}")
    return None


# ═══════════════════════════════════════════════════════
# BTC.D 30d-AGO AUS HIST_DAILY_RISK SHEET
# ═══════════════════════════════════════════════════════
def load_btc_d_30d_ago(gc):
    """Lade BTC.D von vor ~30 Tagen aus HIST_Daily_Risk Sheet.

    Sucht die Zeile die ~30 Tage zurückliegt (±7 Tage Toleranz).
    Fallback: älteste verfügbare BTC.D wenn <30 Tage Daten.
    """
    if gc is None:
        return None
    try:
        sh = gc.open_by_key(CRYPTO_SHEET_ID)
        ws = sh.worksheet(CRYPTO_TABS['hist_daily'])
        vals = ws.get_all_values()
        if len(vals) < 3:
            log("HIST_Daily_Risk: zu wenig Zeilen für 30d Lookback")
            return None

        hdrs = vals[1]  # Row 2 = Header
        # Finde Spalten
        date_col = None
        btc_d_col = None
        for i, h in enumerate(hdrs):
            if h == 'Date':
                date_col = i
            elif h == 'BTC_D_Daily':
                btc_d_col = i

        if date_col is None or btc_d_col is None:
            log("HIST_Daily_Risk: Date oder BTC_D_Daily Spalte nicht gefunden")
            return None

        today = NOW.strftime('%Y-%m-%d')

        # Suche Zeile von vor ~30 Tagen
        best_row = None
        best_diff = 999
        fallback_btc_d = None

        for row in vals[2:]:  # Ab Row 3 = Daten
            try:
                row_date = row[date_col]
                if not row_date:
                    continue
                row_btc_d = row[btc_d_col]
                if not row_btc_d:
                    continue

                # Tage-Differenz berechnen
                from datetime import datetime as dt
                d = dt.strptime(row_date, '%Y-%m-%d')
                diff_days = (NOW.replace(tzinfo=None) - d).days

                btc_d_val = float(row_btc_d)
                if btc_d_val <= 0:
                    continue

                # Track älteste als Fallback
                if fallback_btc_d is None:
                    fallback_btc_d = btc_d_val

                # Suche nächstgelegene zu 30 Tagen (Toleranz 23-37 Tage)
                if 23 <= diff_days <= 37:
                    diff_from_30 = abs(diff_days - 30)
                    if diff_from_30 < best_diff:
                        best_diff = diff_from_30
                        best_row = {'date': row_date, 'btc_d': btc_d_val, 'days_ago': diff_days}

            except (ValueError, IndexError):
                continue

        if best_row:
            log(f"BTC.D 30d ago: {best_row['btc_d']:.1f}% "
                f"(vom {best_row['date']}, {best_row['days_ago']}d zurück)")
            return best_row['btc_d']
        elif fallback_btc_d:
            log(f"BTC.D Fallback (älteste verfügbare): {fallback_btc_d:.1f}%")
            return fallback_btc_d
        else:
            log("Kein BTC.D im HIST_Daily_Risk gefunden")
            return None

    except Exception as e:
        log(f"BTC.D 30d Lookback ERR: {e}")
        return None


# ═══════════════════════════════════════════════════════
# VORHERIGE ALLOKATION AUS CALC_ALLOCATION SHEET
# (gleiche Logik wie crypto_orchestrator)
# ═══════════════════════════════════════════════════════
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
                f"(BTC={prev.get('btc', 0) or 0:.1%} ETH={prev.get('eth', 0) or 0:.1%} "
                f"SOL={prev.get('sol', 0) or 0:.1%})")
            return prev
        log("Keine gültige vorherige Allokation")
        return None
    except Exception as e:
        log(f"Vorherige Allokation ERR: {e}")
        return None


# ═══════════════════════════════════════════════════════
# VORHERIGE crypto_state.json LADEN (für Alert-Vergleich)
# ═══════════════════════════════════════════════════════
def load_previous_state():
    """Lade crypto_state.json (vom letzten Run — kann Daily oder Weekly sein)."""
    json_path = os.path.join(DATA_DIR, 'crypto_state.json')
    if not os.path.exists(json_path):
        log(f"WARNUNG: {json_path} nicht gefunden")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        log(f"Vorheriger State: {state.get('date', '?')}, "
            f"Ensemble={state.get('ensemble', {}).get('value', '?')}")
        return state
    except Exception as e:
        log(f"Vorheriger State ERR: {e}")
        return None


# ═══════════════════════════════════════════════════════
# ALERT-CHECK — Vergleich Daily vs. Vorherigen State
# ═══════════════════════════════════════════════════════
def check_alerts(daily_ensemble, daily_below_wma, daily_wma_200,
                 daily_btc_d, prev_state, btc_price):
    """Prüfe ob signifikante Änderungen seit letztem Run.

    Returns:
        alerts: list of dict {'type': str, 'message': str, 'severity': str}
    """
    alerts = []

    if prev_state is None:
        log("Kein vorheriger State → keine Alert-Prüfung möglich")
        return alerts

    prev_ensemble = prev_state.get('ensemble', {}).get('value')
    prev_below_wma = prev_state.get('bottom_bonus', {}).get('active', False)
    prev_btc_d = prev_state.get('trickle_down', {}).get('btc_dominance')

    # ─── 1. Ensemble-Sprung ───
    if prev_ensemble is not None and abs(daily_ensemble - prev_ensemble) > ENSEMBLE_CHANGE_THRESHOLD:
        direction = "↑" if daily_ensemble > prev_ensemble else "↓"
        severity = 'HIGH' if abs(daily_ensemble - prev_ensemble) >= 0.50 else 'MEDIUM'
        alerts.append({
            'type': 'ENSEMBLE_CHANGE',
            'message': f"Ensemble {direction} {prev_ensemble:.2f} → {daily_ensemble:.2f} "
                       f"(Allokation ändert sich)",
            'severity': severity,
            'previous': prev_ensemble,
            'daily': daily_ensemble,
        })

    # ─── 2. 200WMA Crossing ───
    if daily_below_wma and not prev_below_wma:
        alerts.append({
            'type': 'BELOW_200WMA',
            'message': f"BTC unter 200WMA gefallen! "
                       f"BTC=${btc_price:,.0f}, 200WMA=${daily_wma_200:,.0f} "
                       f"→ Bottom Bonus aktiv",
            'severity': 'HIGH',
        })
    elif not daily_below_wma and prev_below_wma:
        alerts.append({
            'type': 'ABOVE_200WMA',
            'message': f"BTC über 200WMA gestiegen! "
                       f"BTC=${btc_price:,.0f}, 200WMA=${daily_wma_200:,.0f} "
                       f"→ Bottom Bonus deaktiviert",
            'severity': 'MEDIUM',
        })

    # ─── 3. BTC.D großer Sprung (Phase-Wechsel möglich) ───
    if prev_btc_d is not None and daily_btc_d is not None:
        btc_d_delta = daily_btc_d - prev_btc_d
        if abs(btc_d_delta) > BTC_D_WEEKLY_CHANGE_THRESHOLD:
            direction = "↑ BTC_FIRST" if btc_d_delta > 0 else "↓ ALT_ROTATION"
            alerts.append({
                'type': 'BTC_D_SHIFT',
                'message': f"BTC.D {prev_btc_d:.1f}% → {daily_btc_d:.1f}% "
                           f"(Δ {btc_d_delta:+.1f}pp) → Richtung {direction}",
                'severity': 'MEDIUM',
                'prev_btc_d': prev_btc_d,
                'daily_btc_d': daily_btc_d,
            })

    # ─── 4. Phase-Wechsel (NEU in V2.0) ───
    prev_phase = prev_state.get('trickle_down', {}).get('phase')
    # Phase wird unten im daily_result berechnet — Alert prüfen wir nachträglich
    # (wird in main() nach Phase-Berechnung ergänzt)

    return alerts


# ═══════════════════════════════════════════════════════
# TELEGRAM ALERT
# ═══════════════════════════════════════════════════════
def send_telegram_alert(alerts, daily_result):
    """Sende Telegram Alert bei signifikanter Änderung."""
    import requests

    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        log("Telegram: TELEGRAM_BOT_TOKEN oder TELEGRAM_CHAT_ID nicht gesetzt → übersprungen")
        return False

    # Nachricht bauen
    lines = ["🔔 *CRYPTO DAILY SIGNAL*", f"_{NOW.strftime('%Y-%m-%d %H:%M UTC')}_", ""]

    for a in alerts:
        icon = "🔴" if a['severity'] == 'HIGH' else "🟡"
        lines.append(f"{icon} *{a['type']}*")
        lines.append(f"  {a['message']}")
        lines.append("")

    # Aktueller Stand (V2.0: mit Positionen)
    r = daily_result
    lines.append("📊 *Aktueller Stand:*")
    lines.append(f"  BTC: ${r['btc_price']:,.0f}")
    lines.append(f"  Ensemble: {r['ensemble']:.2f} "
                 f"(1M={'✅' if r['mom']['1M'] else '❌'} "
                 f"3M={'✅' if r['mom']['3M'] else '❌'} "
                 f"6M={'✅' if r['mom']['6M'] else '❌'} "
                 f"12M={'✅' if r['mom']['12M'] else '❌'})")
    if r['below_wma']:
        lines.append(f"  ⚠️ BTC UNTER 200WMA (${r['wma_200']:,.0f})")
    lines.append(f"  Phase: {r.get('phase', '?')} ({r.get('phase_name', '?')})")
    if r.get('p4_warning'):
        lines.append(f"  ⚠️ PHASE 4 WARNING AKTIV")
    lines.append(f"  Allokation: {r.get('total_invested', 0):.0%}")
    lines.append(f"  BTC={r.get('pos_btc', 0):.1%} ETH={r.get('pos_eth', 0):.1%} "
                 f"SOL={r.get('pos_sol', 0):.1%} Cash={r.get('pos_cash', 0):.1%}")
    lines.append(f"  Action: {r.get('action', '?')}")

    text = "\n".join(lines)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown',
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            log("Telegram Alert gesendet ✅")
            return True
        else:
            log(f"Telegram ERR: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        log(f"Telegram ERR: {e}")
        return False


# ═══════════════════════════════════════════════════════
# SHEET WRITE — CALC_Allocation (1 Zeile pro Tag)
# ═══════════════════════════════════════════════════════
def write_calc_allocation(gc, daily_result, prev_alloc):
    """Schreibe 1 Zeile in CALC_Allocation Tab (22 Spalten, gleiche Struktur wie Orchestrator)."""
    if gc is None:
        return

    def clean_val(v):
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

    try:
        sh = gc.open_by_key(CRYPTO_SHEET_ID)
        ws = sh.worksheet(CRYPTO_TABS['calc_alloc'])

        r = daily_result
        w = r.get('weights', {})
        w_str = f"{int(w.get('BTC', 0)*100)}/{int(w.get('ETH', 0)*100)}/{int(w.get('SOL', 0)*100)}"

        prev_exp = prev_alloc.get('total_invested', '') if prev_alloc else ''
        delta = ''
        if prev_alloc and prev_alloc.get('total_invested') is not None:
            delta = round((r.get('total_invested', 0) - prev_alloc['total_invested']) * 100, 1)

        row = [
            # [1-2] META
            NOW.strftime('%Y-%m-%d'),
            NOW.strftime('%Y-%m-%d %H:%M UTC'),
            # [3-7] V8 ENSEMBLE
            r['ensemble'],
            r['mom']['1M'],
            r['mom']['3M'],
            r['mom']['6M'],
            r['mom']['12M'],
            # [8-12] TRICKLE-DOWN
            r.get('btc_d_daily', ''),
            r.get('btc_d_change', ''),
            r.get('phase', ''),
            r.get('phase_name', ''),
            w_str,
            # [13-14] PHASE 4 WARNING
            r.get('p4_warning', False),
            r.get('alloc_before_p4', ''),
            # [15-19] FINALE ALLOKATION
            r.get('total_invested', ''),
            r.get('pos_btc', ''),
            r.get('pos_eth', ''),
            r.get('pos_sol', ''),
            r.get('pos_cash', ''),
            # [20-22] NO-ACTION BAND
            prev_exp,
            delta,
            r.get('action', ''),
        ]
        ws.append_row([clean_val(v) for v in row], value_input_option='USER_ENTERED')
        log(f"CALC_Allocation: {len(row)} Werte ✅")

    except Exception as e:
        log(f"CALC_Allocation ERR: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# SHEET WRITE — HIST_Daily_Risk (1 Zeile pro Tag, erweitert V2.0)
# ═══════════════════════════════════════════════════════
def write_hist_daily_risk(gc, daily_result, alerts):
    """Schreibe 1 Zeile in HIST_Daily_Risk Tab.

    Spalten V2.0 (35 Spalten, erweitert um Daily-Positionen):
    Date, BTC_Price, Ensemble_Daily, Ensemble_Prev, Ensemble_Changed,
    Mom_1M, Mom_3M, Mom_6M, Mom_12M,
    WMA_200, Below_200WMA, Below_WMA_Changed,
    BTC_D_Daily, BTC_D_30d_Ago, BTC_D_Change_30d,
    Phase, Phase_Name,
    P4_Warning, Alloc_Before_P4,
    Daily_Alloc_Total, Daily_BTC%, Daily_ETH%, Daily_SOL%, Daily_Cash%,
    Action,
    Prev_Alloc_Total, Prev_BTC%, Prev_ETH%, Prev_SOL%, Prev_Cash%,
    Alert_Count, Alert_Types, Alert_Severities,
    Telegram_Sent, Run_Timestamp, Version
    """
    if gc is None:
        log("Sheet Write: kein GCP Auth → übersprungen")
        return

    def clean_val(v):
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

    try:
        sh = gc.open_by_key(CRYPTO_SHEET_ID)
        ws = sh.worksheet(CRYPTO_TABS['hist_daily'])

        r = daily_result
        alert_types = ",".join(a['type'] for a in alerts) if alerts else ""
        alert_severities = ",".join(a['severity'] for a in alerts) if alerts else ""

        row = [
            # [1-2] META
            NOW.strftime('%Y-%m-%d'),
            r['btc_price'],
            # [3-5] ENSEMBLE
            r['ensemble'],
            r.get('prev_ensemble', ''),
            r.get('ensemble_changed', False),
            # [6-9] MOMENTUM
            r['mom']['1M'],
            r['mom']['3M'],
            r['mom']['6M'],
            r['mom']['12M'],
            # [10-12] 200WMA
            r['wma_200'],
            r['below_wma'],
            r.get('below_wma_changed', False),
            # [13-15] BTC.D
            r.get('btc_d_daily', ''),
            r.get('btc_d_30d_ago', ''),
            r.get('btc_d_change', ''),
            # [16-17] PHASE
            r.get('phase', ''),
            r.get('phase_name', ''),
            # [18-19] PHASE 4 WARNING
            r.get('p4_warning', False),
            r.get('alloc_before_p4', ''),
            # [20-24] DAILY POSITIONEN
            r.get('total_invested', ''),
            r.get('pos_btc', ''),
            r.get('pos_eth', ''),
            r.get('pos_sol', ''),
            r.get('pos_cash', ''),
            # [25] ACTION
            r.get('action', ''),
            # [26-30] PREV POSITIONEN
            r.get('prev_alloc', ''),
            r.get('prev_btc', ''),
            r.get('prev_eth', ''),
            r.get('prev_sol', ''),
            r.get('prev_cash', ''),
            # [31-33] ALERTS
            len(alerts),
            alert_types,
            alert_severities,
            # [34-35] META
            r.get('telegram_sent', False),
            VERSION,
        ]

        ws.append_row([clean_val(v) for v in row], value_input_option='USER_ENTERED')
        log(f"HIST_Daily_Risk: {len(row)} Werte ✅")

    except Exception as e:
        log(f"HIST_Daily_Risk ERR: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# JSON OUTPUT — crypto_state.json (für Frontend)
# Gleiche Struktur wie crypto_orchestrator → Frontend sieht keinen Unterschied
# ═══════════════════════════════════════════════════════
def write_state_json(daily_result, prev_state):
    """Schreibe crypto_state.json (gleiche Struktur wie Weekly Orchestrator)."""
    r = daily_result

    # Display-Daten: aus vorherigem State übernehmen (werden nur weekly aktualisiert)
    prev_display = prev_state.get('display', {}) if prev_state else {}

    state = {
        'version': VERSION,
        'config_version': CONFIG_VERSION,
        'timestamp': NOW.isoformat(),
        'date': NOW.strftime('%Y-%m-%d'),
        'source': 'daily_signal_engine',

        'btc_price': r['btc_price'],
        'eth_price': r.get('eth_price'),
        'sol_price': r.get('sol_price'),

        'ensemble': {
            'value': r['ensemble'],
            'mom_1M': r['mom']['1M'],
            'mom_3M': r['mom']['3M'],
            'mom_6M': r['mom']['6M'],
            'mom_12M': r['mom']['12M'],
        },
        'bottom_bonus': {
            'active': r['below_wma'],
            'wma_200': r['wma_200'],
        },
        'trickle_down': {
            'btc_dominance': r.get('btc_d_daily'),
            'btc_d_30d_change': r.get('btc_d_change'),
            'phase': r.get('phase'),
            'phase_name': r.get('phase_name'),
            'phase4_warning': r.get('p4_warning', False),
        },
        'allocation': {
            'total': r.get('total_invested', 0),
            'btc': r.get('pos_btc', 0),
            'eth': r.get('pos_eth', 0),
            'sol': r.get('pos_sol', 0),
            'cash': r.get('pos_cash', 0),
        },
        'weights': r.get('weights', {}),
        'action': r.get('action', 'HOLD'),

        # Display-Daten: übernehmen aus letztem State (weekly aktualisiert)
        'display': {
            'rainbow_band': prev_display.get('rainbow_band'),
            'rainbow_score': prev_display.get('rainbow_score'),
            'mvrv_zscore': prev_display.get('mvrv_zscore'),
            'nupl': prev_display.get('nupl'),
            'fear_greed': prev_display.get('fear_greed'),
            'v16_macro_state': prev_display.get('v16_macro_state'),
            'funding_btc': prev_display.get('funding_btc'),
            'halving_phase': prev_display.get('halving_phase'),
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, 'crypto_state.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    log(f"crypto_state.json: {p} ✅")
    return state


# ═══════════════════════════════════════════════════════
# JSON OUTPUT — crypto_daily_check.json (für Briefing)
# ═══════════════════════════════════════════════════════
def write_daily_json(daily_result, alerts):
    """Schreibe crypto_daily_check.json (für Step 0u Briefing)."""
    r = daily_result
    output = {
        'version': VERSION,
        'timestamp': NOW.isoformat(),
        'date': NOW.strftime('%Y-%m-%d'),
        'btc_price': r['btc_price'],
        'ensemble': {
            'daily': r['ensemble'],
            'previous': r.get('prev_ensemble'),
            'changed': r.get('ensemble_changed', False),
            'mom_1M': r['mom']['1M'],
            'mom_3M': r['mom']['3M'],
            'mom_6M': r['mom']['6M'],
            'mom_12M': r['mom']['12M'],
        },
        'bottom_bonus': {
            'active': r['below_wma'],
            'wma_200': r['wma_200'],
            'changed': r.get('below_wma_changed', False),
        },
        'btc_dominance': {
            'daily': r.get('btc_d_daily'),
            'thirty_d_ago': r.get('btc_d_30d_ago'),
            'change_30d': r.get('btc_d_change'),
        },
        'signal': {
            'phase': r.get('phase'),
            'phase_name': r.get('phase_name'),
            'phase4_warning': r.get('p4_warning', False),
            'weights': r.get('weights', {}),
        },
        'allocation': {
            'total': r.get('total_invested', 0),
            'btc': r.get('pos_btc', 0),
            'eth': r.get('pos_eth', 0),
            'sol': r.get('pos_sol', 0),
            'cash': r.get('pos_cash', 0),
        },
        'action': r.get('action', 'HOLD'),
        'previous_allocation': {
            'total': r.get('prev_alloc'),
            'btc': r.get('prev_btc'),
            'eth': r.get('prev_eth'),
            'sol': r.get('prev_sol'),
            'cash': r.get('prev_cash'),
        },
        'alerts': alerts,
        'alert_count': len(alerts),
        'telegram_sent': r.get('telegram_sent', False),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, 'crypto_daily_check.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"crypto_daily_check.json: {p} ✅")
    return output


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description='Crypto Circle Daily Signal Engine V2.0')
    pa.add_argument('--skip-write', action='store_true', help='Kein Sheet Write')
    pa.add_argument('--skip-telegram', action='store_true', help='Keine Telegram Alerts')
    pa.add_argument('--skip-v16', action='store_true', help='Kein V16 Sheet Read (CoinMetrics Fallback)')
    args = pa.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"CRYPTO CIRCLE — DAILY SIGNAL ENGINE {VERSION}")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  System: V8+Warn (Ensemble + Trickle-Down + P4 Warning)")
    print(f"  Config: V{CONFIG_VERSION}")
    print(f"  Flags: skip-write={args.skip_write} skip-telegram={args.skip_telegram} "
          f"skip-v16={args.skip_v16}")
    print("=" * 70)

    # ─── GCP Auth ───
    gc = None
    if HAS_GSPREAD and not args.skip_write:
        gc = get_gspread_client()
        if gc is None:
            log("WARNUNG: Kein GCP Auth — Sheet Write deaktiviert")

    # ─── Schritt 1: Preise ───
    print(f"\n{'='*60}")
    print("SCHRITT 1: PREISE (BTC/ETH/SOL)")
    print(f"{'='*60}")

    prices = fetch_prices()
    if prices is None or prices.get('btc_price') is None:
        log("ABBRUCH: Kein BTC-Preis")
        sys.exit(1)

    btc_price = prices['btc_price']
    eth_price = prices.get('eth_price')
    sol_price = prices.get('sol_price')

    # ─── Schritt 2: BTC.D ───
    print(f"\n{'='*60}")
    print("SCHRITT 2: BTC DOMINANCE")
    print(f"{'='*60}")

    btc_d_daily = fetch_btc_dominance()

    # ─── Schritt 3: BTC Historie + Ensemble ───
    print(f"\n{'='*60}")
    print("SCHRITT 3: ENSEMBLE BERECHNUNG")
    print(f"{'='*60}")

    btc_prices, dates = load_btc_history(gc if not args.skip_v16 else None)
    if len(btc_prices) < 252:
        log(f"ABBRUCH: Nur {len(btc_prices)} Tage BTC-Historie (min 252)")
        sys.exit(1)

    # Aktuellen Preis an Historie anhängen (falls Intraday neuer als Sheet)
    if len(dates) > 0:
        last_date = dates[-1]
        today_str = NOW.strftime('%Y-%m-%d')
        if last_date < today_str:
            btc_prices = np.append(btc_prices, btc_price)
            dates.append(today_str)
            log(f"BTC-Preis für {today_str} angehängt: ${btc_price:,.0f}")

    cycle_result = calc_ensemble(btc_prices)
    daily_ensemble = cycle_result['ensemble']
    daily_alloc = cycle_result['alloc']
    daily_below_wma = cycle_result['below_wma']
    daily_wma_200 = cycle_result['wma_200']

    log(f"Ensemble: {daily_ensemble:.2f}, Alloc: {daily_alloc:.0%}")
    log(f"200WMA: {'UNTER' if daily_below_wma else 'ÜBER'} "
        f"(${daily_wma_200:,.0f})" if daily_wma_200 else "200WMA: N/A")

    # ─── Schritt 4: BTC.D 30d Lookback + Phase ───
    print(f"\n{'='*60}")
    print("SCHRITT 4: SIGNAL ENGINE (Trickle-Down Phase)")
    print(f"{'='*60}")

    btc_d_30d_ago = load_btc_d_30d_ago(gc)

    if btc_d_daily is not None:
        if btc_d_30d_ago is not None:
            phase, btc_d_change = calc_phase_single(btc_d_daily, btc_d_30d_ago)
        else:
            log("Kein BTC.D 30d ago → Phase 2 (Default)")
            phase = V8_TRICKLE_DOWN['default_phase']
            btc_d_change = 0.0
    else:
        log("Kein BTC.D → Phase 2 (Default)")
        phase = V8_TRICKLE_DOWN['default_phase']
        btc_d_change = 0.0

    weights = get_tier_weights(phase)
    phase_name = PHASE_NAMES[phase]

    log(f"Phase: {phase} ({phase_name}), "
        f"BTC.D: {btc_d_daily or '?'}%, Δ30d: {btc_d_change:+.1f}pp")
    log(f"Gewichte: BTC={weights['BTC']:.0%} ETH={weights['ETH']:.0%} SOL={weights['SOL']:.0%}")

    # ─── Schritt 5: Risk Engine ───
    print(f"\n{'='*60}")
    print("SCHRITT 5: RISK ENGINE (Phase 4 + Positionen + NO-ACTION)")
    print(f"{'='*60}")

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
    log(f"Positionen: BTC={positions['btc']:.1%} ETH={positions['eth']:.1%} "
        f"SOL={positions['sol']:.1%} Cash={positions['cash']:.1%}")

    # ─── Schritt 6: Vorherigen State laden + Alert-Check ───
    print(f"\n{'='*60}")
    print("SCHRITT 6: ALERT-CHECK")
    print(f"{'='*60}")

    prev_state = load_previous_state()

    # Vorherige Werte extrahieren
    prev_ensemble = prev_state.get('ensemble', {}).get('value') if prev_state else None
    prev_below_wma = prev_state.get('bottom_bonus', {}).get('active', False) if prev_state else False
    prev_phase = prev_state.get('trickle_down', {}).get('phase') if prev_state else None
    prev_alloc_total = prev_state.get('allocation', {}).get('total') if prev_state else None
    prev_btc_pos = prev_state.get('allocation', {}).get('btc') if prev_state else None
    prev_eth_pos = prev_state.get('allocation', {}).get('eth') if prev_state else None
    prev_sol_pos = prev_state.get('allocation', {}).get('sol') if prev_state else None
    prev_cash_pos = prev_state.get('allocation', {}).get('cash') if prev_state else None

    ensemble_changed = (prev_ensemble is not None and
                        abs(daily_ensemble - prev_ensemble) > ENSEMBLE_CHANGE_THRESHOLD)
    below_wma_changed = (daily_below_wma != prev_below_wma)

    alerts = check_alerts(
        daily_ensemble=daily_ensemble,
        daily_below_wma=daily_below_wma,
        daily_wma_200=daily_wma_200 or 0,
        daily_btc_d=btc_d_daily,
        prev_state=prev_state,
        btc_price=btc_price,
    )

    # Phase-Wechsel Alert (braucht berechnete Phase)
    if prev_phase is not None and phase != prev_phase:
        prev_phase_name = PHASE_NAMES.get(prev_phase, f'Phase {prev_phase}')
        alerts.append({
            'type': 'PHASE_CHANGE',
            'message': f"Phase {prev_phase} ({prev_phase_name}) → {phase} ({phase_name})",
            'severity': 'MEDIUM',
            'prev_phase': prev_phase,
            'daily_phase': phase,
        })

    if alerts:
        for a in alerts:
            icon = "🔴" if a['severity'] == 'HIGH' else "🟡"
            log(f"{icon} {a['type']}: {a['message']}")
    else:
        log("Keine Alerts — alles im Rahmen")

    # ─── Daily Result zusammenbauen ───
    daily_result = {
        'btc_price': btc_price,
        'eth_price': eth_price,
        'sol_price': sol_price,
        'ensemble': daily_ensemble,
        'alloc_raw': raw_alloc,
        'mom': cycle_result['mom'],
        'below_wma': daily_below_wma,
        'wma_200': daily_wma_200,
        'btc_d_daily': btc_d_daily,
        'btc_d_30d_ago': btc_d_30d_ago,
        'btc_d_change': round(btc_d_change, 2),
        'phase': phase,
        'phase_name': phase_name,
        'weights': weights,
        'p4_warning': p4_active,
        'alloc_before_p4': raw_alloc,
        'total_invested': positions['total_invested'],
        'pos_btc': positions['btc'],
        'pos_eth': positions['eth'],
        'pos_sol': positions['sol'],
        'pos_cash': positions['cash'],
        'action': action,
        'action_reason': reason,
        'deltas': deltas,
        'prev_ensemble': prev_ensemble,
        'ensemble_changed': ensemble_changed,
        'below_wma_changed': below_wma_changed,
        'prev_alloc': prev_alloc_total,
        'prev_btc': prev_btc_pos,
        'prev_eth': prev_eth_pos,
        'prev_sol': prev_sol_pos,
        'prev_cash': prev_cash_pos,
        'telegram_sent': False,
    }

    # ─── Schritt 7: Telegram Alert ───
    if alerts and not args.skip_telegram:
        print(f"\n{'='*60}")
        print("SCHRITT 7: TELEGRAM ALERT")
        print(f"{'='*60}")

        sent = send_telegram_alert(alerts, daily_result)
        daily_result['telegram_sent'] = sent
    elif alerts:
        log("Telegram übersprungen (--skip-telegram)")
    else:
        log("Kein Telegram nötig (keine Alerts)")

    # ─── Schritt 8: Output ───
    print(f"\n{'='*60}")
    print("SCHRITT 8: OUTPUT")
    print(f"{'='*60}")

    # crypto_state.json (Frontend — TÄGLICH aktualisiert)
    write_state_json(daily_result, prev_state)

    # crypto_daily_check.json (Briefing)
    write_daily_json(daily_result, alerts)

    # Sheet Writes
    if gc and not args.skip_write:
        write_calc_allocation(gc, daily_result, prev_alloc)
        time.sleep(1.0)
        write_hist_daily_risk(gc, daily_result, alerts)
    elif args.skip_write:
        log("Sheet Write übersprungen (--skip-write)")
    else:
        log("Sheet Write übersprungen (kein GCP Auth)")

    # ─── Zusammenfassung ───
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"CRYPTO CIRCLE — DAILY SIGNAL ENGINE FERTIG ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  BTC:       ${btc_price:>10,.0f}  ETH: ${eth_price or 0:>8,.0f}  SOL: ${sol_price or 0:>8,.2f}")
    print(f"  Ensemble:  {daily_ensemble:.2f} "
          f"(1M={'✅' if cycle_result['mom']['1M'] else '❌'} "
          f"3M={'✅' if cycle_result['mom']['3M'] else '❌'} "
          f"6M={'✅' if cycle_result['mom']['6M'] else '❌'} "
          f"12M={'✅' if cycle_result['mom']['12M'] else '❌'})")
    wma_str = f"${daily_wma_200:,.0f}" if daily_wma_200 else 'N/A'
    print(f"  200WMA:    {wma_str} {'← UNTER (Bonus)' if daily_below_wma else ''}")
    print(f"  BTC.D:     {btc_d_daily or '?'}% (30d ago: {btc_d_30d_ago or '?'}%)")
    print(f"  Phase:     {phase} ({phase_name})")
    if p4_active:
        print(f"  ⚠️  PHASE 4 WARNING AKTIV")
    print(f"  Allok:     {positions['total_invested']:.0%}")
    print(f"  Positionen: BTC={positions['btc']:.1%} ETH={positions['eth']:.1%} "
          f"SOL={positions['sol']:.1%} Cash={positions['cash']:.1%}")
    print(f"  Action:    {action} — {reason}")
    print(f"  Alerts:    {len(alerts)}")
    if alerts:
        for a in alerts:
            icon = "🔴" if a['severity'] == 'HIGH' else "🟡"
            print(f"    {icon} {a['type']}")
    print(f"  Telegram:  {'✅ gesendet' if daily_result['telegram_sent'] else '— nicht gesendet'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
