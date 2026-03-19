#!/usr/bin/env python3
"""
daily_risk_check.py — Crypto Circle Daily Risk Check V1.0
============================================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

Leichtgewichtiger täglicher Check, getriggert von Step 7 (Dashboard Writer).
KEIN eigener Cron, KEIN voller data_collector Run.

Was es tut:
  1. BTC-Preis von CoinGecko holen (1 API Call)
  2. BTC-Historie aus V16 Sheet laden
  3. Ensemble berechnen (cycle_engine.calc_ensemble)
  4. crypto_state.json laden (letztes Weekly-Signal)
  5. Vergleich: Hat sich was signifikant geändert?
  6. Bei Trigger → Telegram Alert senden
  7. Ergebnis in crypto_daily_check.json schreiben (für Briefing)
  8. HIST_Daily_Risk Tab im Sheet beschreiben (1 Zeile pro Tag)

Alert-Trigger:
  - Ensemble-Sprung: Weekly Ensemble ≠ Daily Ensemble
  - BTC unter 200WMA (Bottom Bonus würde aktiv)
  - BTC.D Δ > 3pp in einer Woche (Phase-Wechsel möglich)

Trigger-Kette:
  V16 Daily Runner (06:00 UTC) → ... → Step 7
    → step0y_crypto_daily.yml (workflow_dispatch)
      → daily_risk_check.py
        → Telegram Alert (wenn Trigger)
      → step0u_briefing.yml (workflow_dispatch)

Usage:
  # GitHub Actions (von Step 7 getriggert):
  python -m step_0y_crypto.daily_risk_check

  # Colab Test:
  python daily_risk_check.py --skip-write --skip-telegram
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
        API_ENDPOINTS,
    )
    from step_0y_crypto.cycle_engine import calc_ensemble, load_btc_history
    from step_0y_crypto.signal_engine import PHASE_NAMES
except (ImportError, ModuleNotFoundError):
    _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/content/step_0y_crypto'
    sys.path.insert(0, _dir)
    sys.path.insert(0, os.path.dirname(_dir))
    from config import (
        CRYPTO_SHEET_ID, V16_SHEET_ID, V16_TABS,
        CRYPTO_TABS, CONFIG_VERSION,
        V8_ENSEMBLE, V8_BOTTOM_BONUS, V8_TRICKLE_DOWN,
        API_ENDPOINTS,
    )
    from cycle_engine import calc_ensemble, load_btc_history
    from signal_engine import PHASE_NAMES

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

NOW = datetime.now(timezone.utc)
VERSION = "daily_risk_check V1.0"

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
# BTC PREIS VON COINGECKO (1 API Call)
# ═══════════════════════════════════════════════════════
def fetch_btc_price():
    """Aktuellen BTC-Preis von CoinGecko holen."""
    import requests
    url = API_ENDPOINTS.get('coingecko_prices',
          'https://api.coingecko.com/api/v3/simple/price')
    params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        price = data.get('bitcoin', {}).get('usd')
        if price and price > 0:
            log(f"BTC Preis (CoinGecko): ${price:,.2f}")
            return float(price)
    except Exception as e:
        log(f"CoinGecko ERR: {e}")
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
# WEEKLY STATE LADEN (letztes Sonntags-Signal)
# ═══════════════════════════════════════════════════════
def load_weekly_state():
    """Lade crypto_state.json (vom letzten Weekly Orchestrator Run)."""
    json_path = os.path.join(DATA_DIR, 'crypto_state.json')
    if not os.path.exists(json_path):
        log(f"WARNUNG: {json_path} nicht gefunden")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        log(f"Weekly State: {state.get('date', '?')}, "
            f"Ensemble={state.get('ensemble', {}).get('value', '?')}")
        return state
    except Exception as e:
        log(f"Weekly State ERR: {e}")
        return None


# ═══════════════════════════════════════════════════════
# ALERT-CHECK — Vergleich Daily vs. Weekly
# ═══════════════════════════════════════════════════════
def check_alerts(daily_ensemble, daily_below_wma, daily_wma_200,
                 daily_btc_d, weekly_state, btc_price):
    """Prüfe ob signifikante Änderungen seit letztem Weekly.

    Returns:
        alerts: list of dict {'type': str, 'message': str, 'severity': str}
    """
    alerts = []

    if weekly_state is None:
        log("Kein Weekly State → keine Alert-Prüfung möglich")
        return alerts

    weekly_ensemble = weekly_state.get('ensemble', {}).get('value')
    weekly_below_wma = weekly_state.get('bottom_bonus', {}).get('active', False)
    weekly_btc_d = weekly_state.get('trickle_down', {}).get('btc_dominance')

    # ─── 1. Ensemble-Sprung ───
    if weekly_ensemble is not None and abs(daily_ensemble - weekly_ensemble) > ENSEMBLE_CHANGE_THRESHOLD:
        direction = "↑" if daily_ensemble > weekly_ensemble else "↓"
        severity = 'HIGH' if abs(daily_ensemble - weekly_ensemble) >= 0.50 else 'MEDIUM'
        alerts.append({
            'type': 'ENSEMBLE_CHANGE',
            'message': f"Ensemble {direction} {weekly_ensemble:.2f} → {daily_ensemble:.2f} "
                       f"(Allokation {weekly_ensemble:.0%} → {daily_ensemble:.0%})",
            'severity': severity,
            'weekly': weekly_ensemble,
            'daily': daily_ensemble,
        })

    # ─── 2. 200WMA Crossing ───
    if daily_below_wma and not weekly_below_wma:
        alerts.append({
            'type': 'BELOW_200WMA',
            'message': f"BTC unter 200WMA gefallen! "
                       f"BTC=${btc_price:,.0f}, 200WMA=${daily_wma_200:,.0f} "
                       f"→ Bottom Bonus aktiv",
            'severity': 'HIGH',
        })
    elif not daily_below_wma and weekly_below_wma:
        alerts.append({
            'type': 'ABOVE_200WMA',
            'message': f"BTC über 200WMA gestiegen! "
                       f"BTC=${btc_price:,.0f}, 200WMA=${daily_wma_200:,.0f} "
                       f"→ Bottom Bonus deaktiviert",
            'severity': 'MEDIUM',
        })

    # ─── 3. BTC.D großer Sprung (Phase-Wechsel möglich) ───
    if weekly_btc_d is not None and daily_btc_d is not None:
        btc_d_delta = daily_btc_d - weekly_btc_d
        if abs(btc_d_delta) > BTC_D_WEEKLY_CHANGE_THRESHOLD:
            direction = "↑ BTC_FIRST" if btc_d_delta > 0 else "↓ ALT_ROTATION"
            alerts.append({
                'type': 'BTC_D_SHIFT',
                'message': f"BTC.D {weekly_btc_d:.1f}% → {daily_btc_d:.1f}% "
                           f"(Δ {btc_d_delta:+.1f}pp) → Richtung {direction}",
                'severity': 'MEDIUM',
                'weekly_btc_d': weekly_btc_d,
                'daily_btc_d': daily_btc_d,
            })

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
    lines = ["🔔 *CRYPTO DAILY ALERT*", f"_{NOW.strftime('%Y-%m-%d %H:%M UTC')}_", ""]

    for a in alerts:
        icon = "🔴" if a['severity'] == 'HIGH' else "🟡"
        lines.append(f"{icon} *{a['type']}*")
        lines.append(f"  {a['message']}")
        lines.append("")

    # Aktueller Stand
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
    lines.append(f"  Weekly Allok: {r.get('weekly_alloc', '?')}")

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
# SHEET WRITE — HIST_Daily_Risk (1 Zeile pro Tag)
# ═══════════════════════════════════════════════════════
def write_hist_daily_risk(gc, daily_result, alerts):
    """Schreibe 1 Zeile in HIST_Daily_Risk Tab.

    Spalten (29 lt. V147 Header):
    Date, BTC_Price, Ensemble_Daily, Ensemble_Weekly, Ensemble_Changed,
    Mom_1M, Mom_3M, Mom_6M, Mom_12M,
    WMA_200, Below_200WMA, Below_WMA_Changed,
    BTC_D_Daily, BTC_D_Weekly, BTC_D_Delta_pp,
    Weekly_Phase, Weekly_Phase_Name,
    Weekly_Alloc_Total, Weekly_BTC%, Weekly_ETH%, Weekly_SOL%, Weekly_Cash%,
    Alert_Count, Alert_Types, Alert_Severities,
    Telegram_Sent, Run_Timestamp, Source, Version
    """
    if gc is None:
        log("Sheet Write: kein GCP Auth → übersprungen")
        return

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
            r.get('weekly_ensemble', ''),
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
            r.get('btc_d_weekly', ''),
            r.get('btc_d_delta', ''),
            # [16-17] PHASE
            r.get('weekly_phase', ''),
            r.get('weekly_phase_name', ''),
            # [18-22] WEEKLY ALLOKATION
            r.get('weekly_alloc', ''),
            r.get('weekly_btc', ''),
            r.get('weekly_eth', ''),
            r.get('weekly_sol', ''),
            r.get('weekly_cash', ''),
            # [23-25] ALERTS
            len(alerts),
            alert_types,
            alert_severities,
            # [26-29] META
            r.get('telegram_sent', False),
            NOW.strftime('%Y-%m-%d %H:%M UTC'),
            'daily_risk_check',
            VERSION,
        ]

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

        ws.append_row([clean_val(v) for v in row], value_input_option='USER_ENTERED')
        log(f"HIST_Daily_Risk: {len(row)} Werte ✅")

    except Exception as e:
        log(f"HIST_Daily_Risk ERR: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# JSON OUTPUT — crypto_daily_check.json (für Briefing)
# ═══════════════════════════════════════════════════════
def write_daily_json(daily_result, alerts):
    """Schreibe crypto_daily_check.json (für Step 0u Briefing)."""
    output = {
        'version': VERSION,
        'timestamp': NOW.isoformat(),
        'date': NOW.strftime('%Y-%m-%d'),
        'btc_price': daily_result['btc_price'],
        'ensemble': {
            'daily': daily_result['ensemble'],
            'weekly': daily_result.get('weekly_ensemble'),
            'changed': daily_result.get('ensemble_changed', False),
            'mom_1M': daily_result['mom']['1M'],
            'mom_3M': daily_result['mom']['3M'],
            'mom_6M': daily_result['mom']['6M'],
            'mom_12M': daily_result['mom']['12M'],
        },
        'bottom_bonus': {
            'active': daily_result['below_wma'],
            'wma_200': daily_result['wma_200'],
            'changed': daily_result.get('below_wma_changed', False),
        },
        'btc_dominance': {
            'daily': daily_result.get('btc_d_daily'),
            'weekly': daily_result.get('btc_d_weekly'),
            'delta_pp': daily_result.get('btc_d_delta'),
        },
        'weekly_signal': {
            'phase': daily_result.get('weekly_phase'),
            'phase_name': daily_result.get('weekly_phase_name'),
            'allocation': {
                'total': daily_result.get('weekly_alloc'),
                'btc': daily_result.get('weekly_btc'),
                'eth': daily_result.get('weekly_eth'),
                'sol': daily_result.get('weekly_sol'),
                'cash': daily_result.get('weekly_cash'),
            },
        },
        'alerts': alerts,
        'alert_count': len(alerts),
        'telegram_sent': daily_result.get('telegram_sent', False),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, 'crypto_daily_check.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"JSON: {p}")
    return output


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description='Crypto Circle Daily Risk Check V1.0')
    pa.add_argument('--skip-write', action='store_true', help='Kein Sheet Write')
    pa.add_argument('--skip-telegram', action='store_true', help='Keine Telegram Alerts')
    pa.add_argument('--skip-v16', action='store_true', help='Kein V16 Sheet Read (CoinMetrics Fallback)')
    args = pa.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"CRYPTO CIRCLE — DAILY RISK CHECK {VERSION}")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Flags: skip-write={args.skip_write} skip-telegram={args.skip_telegram} "
          f"skip-v16={args.skip_v16}")
    print("=" * 70)

    # ─── GCP Auth ───
    gc = None
    if HAS_GSPREAD and not args.skip_write:
        gc = get_gspread_client()
        if gc is None:
            log("WARNUNG: Kein GCP Auth — Sheet Write deaktiviert")

    # ─── Schritt 1: BTC Preis ───
    print(f"\n{'='*60}")
    print("SCHRITT 1: BTC PREIS")
    print(f"{'='*60}")

    btc_price = fetch_btc_price()
    if btc_price is None:
        log("ABBRUCH: Kein BTC-Preis")
        sys.exit(1)

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
    # Nur wenn der letzte Preis in der Historie != aktueller CoinGecko Preis
    if len(dates) > 0:
        last_date = dates[-1]
        today_str = NOW.strftime('%Y-%m-%d')
        if last_date < today_str:
            # Heute fehlt in der Historie → anhängen
            btc_prices = np.append(btc_prices, btc_price)
            dates.append(today_str)
            log(f"BTC-Preis für {today_str} angehängt: ${btc_price:,.0f}")

    cycle_result = calc_ensemble(btc_prices)
    daily_ensemble = cycle_result['ensemble']
    daily_alloc = cycle_result['alloc']
    daily_below_wma = cycle_result['below_wma']
    daily_wma_200 = cycle_result['wma_200']

    log(f"Daily Ensemble: {daily_ensemble:.2f}, Alloc: {daily_alloc:.0%}")
    log(f"200WMA: {'UNTER' if daily_below_wma else 'ÜBER'} "
        f"(${daily_wma_200:,.0f})" if daily_wma_200 else "200WMA: N/A")

    # ─── Schritt 4: Weekly State laden ───
    print(f"\n{'='*60}")
    print("SCHRITT 4: WEEKLY STATE VERGLEICH")
    print(f"{'='*60}")

    weekly_state = load_weekly_state()

    # Weekly-Werte extrahieren
    weekly_ensemble = None
    weekly_below_wma = False
    weekly_btc_d = None
    weekly_phase = None
    weekly_phase_name = None
    weekly_alloc = None
    weekly_btc = None
    weekly_eth = None
    weekly_sol = None
    weekly_cash = None

    if weekly_state:
        weekly_ensemble = weekly_state.get('ensemble', {}).get('value')
        weekly_below_wma = weekly_state.get('bottom_bonus', {}).get('active', False)
        weekly_btc_d = weekly_state.get('trickle_down', {}).get('btc_dominance')
        weekly_phase = weekly_state.get('trickle_down', {}).get('phase')
        weekly_phase_name = weekly_state.get('trickle_down', {}).get('phase_name')
        wa = weekly_state.get('allocation', {})
        weekly_alloc = wa.get('total')
        weekly_btc = wa.get('btc')
        weekly_eth = wa.get('eth')
        weekly_sol = wa.get('sol')
        weekly_cash = wa.get('cash')

        log(f"Weekly: Ensemble={weekly_ensemble}, Phase={weekly_phase} ({weekly_phase_name}), "
            f"Allok={weekly_alloc}")

    # ─── Schritt 5: Alert-Check ───
    print(f"\n{'='*60}")
    print("SCHRITT 5: ALERT-CHECK")
    print(f"{'='*60}")

    alerts = check_alerts(
        daily_ensemble=daily_ensemble,
        daily_below_wma=daily_below_wma,
        daily_wma_200=daily_wma_200 or 0,
        daily_btc_d=btc_d_daily,
        weekly_state=weekly_state,
        btc_price=btc_price,
    )

    ensemble_changed = (weekly_ensemble is not None and
                        abs(daily_ensemble - weekly_ensemble) > ENSEMBLE_CHANGE_THRESHOLD)
    below_wma_changed = (daily_below_wma != weekly_below_wma)
    btc_d_delta = None
    if btc_d_daily is not None and weekly_btc_d is not None:
        btc_d_delta = round(btc_d_daily - weekly_btc_d, 2)

    if alerts:
        for a in alerts:
            icon = "🔴" if a['severity'] == 'HIGH' else "🟡"
            log(f"{icon} {a['type']}: {a['message']}")
    else:
        log("Keine Alerts — alles im Rahmen")

    # ─── Daily Result zusammenbauen ───
    daily_result = {
        'btc_price': btc_price,
        'ensemble': daily_ensemble,
        'alloc': daily_alloc,
        'mom': cycle_result['mom'],
        'below_wma': daily_below_wma,
        'wma_200': daily_wma_200,
        'btc_d_daily': btc_d_daily,
        'btc_d_weekly': weekly_btc_d,
        'btc_d_delta': btc_d_delta,
        'weekly_ensemble': weekly_ensemble,
        'ensemble_changed': ensemble_changed,
        'below_wma_changed': below_wma_changed,
        'weekly_phase': weekly_phase,
        'weekly_phase_name': weekly_phase_name,
        'weekly_alloc': weekly_alloc,
        'weekly_btc': weekly_btc,
        'weekly_eth': weekly_eth,
        'weekly_sol': weekly_sol,
        'weekly_cash': weekly_cash,
        'telegram_sent': False,
    }

    # ─── Schritt 6: Telegram Alert ───
    if alerts and not args.skip_telegram:
        print(f"\n{'='*60}")
        print("SCHRITT 6: TELEGRAM ALERT")
        print(f"{'='*60}")

        sent = send_telegram_alert(alerts, daily_result)
        daily_result['telegram_sent'] = sent
    elif alerts:
        log("Telegram übersprungen (--skip-telegram)")
    else:
        log("Kein Telegram nötig (keine Alerts)")

    # ─── Schritt 7: Output ───
    print(f"\n{'='*60}")
    print("SCHRITT 7: OUTPUT")
    print(f"{'='*60}")

    # JSON für Briefing
    write_daily_json(daily_result, alerts)

    # Sheet Write
    if gc and not args.skip_write:
        write_hist_daily_risk(gc, daily_result, alerts)
    elif args.skip_write:
        log("Sheet Write übersprungen (--skip-write)")
    else:
        log("Sheet Write übersprungen (kein GCP Auth)")

    # ─── Zusammenfassung ───
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"CRYPTO CIRCLE — DAILY CHECK FERTIG ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  BTC:       ${btc_price:>10,.0f}")
    print(f"  Ensemble:  {daily_ensemble:.2f} "
          f"(Weekly: {weekly_ensemble if weekly_ensemble is not None else '?'}) "
          f"{'← GEÄNDERT' if ensemble_changed else ''}")
    print(f"  200WMA:    {'UNTER' if daily_below_wma else 'ÜBER'} "
          f"{'← GEÄNDERT' if below_wma_changed else ''}")
    if btc_d_daily:
        print(f"  BTC.D:     {btc_d_daily:.1f}% "
              f"(Weekly: {weekly_btc_d if weekly_btc_d else '?'}%) "
              f"Δ {btc_d_delta:+.1f}pp" if btc_d_delta is not None else "")
    print(f"  Alerts:    {len(alerts)}")
    if alerts:
        for a in alerts:
            icon = "🔴" if a['severity'] == 'HIGH' else "🟡"
            print(f"    {icon} {a['type']}")
    print(f"  Telegram:  {'✅ gesendet' if daily_result['telegram_sent'] else '— nicht gesendet'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
