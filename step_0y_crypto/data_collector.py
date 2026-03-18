#!/usr/bin/env python3
"""
data_collector.py — Crypto Circle Data Collector V1.3.1
======================================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

V1.3 — PARADIGMENWECHSEL: BGeometrics via GitHub Raw statt API
  - Keine Rate Limits mehr (war: 8 Calls/Stunde)
  - Format: [[timestamp_ms, value], ...] — letztes Element = aktuell
  - URL: raw.githubusercontent.com/BGeometrics/bgeometrics.github.io/master/files/
  - Dateien werden täglich automatisch aktualisiert (33K+ Commits)

V1.2 Fixes bleiben erhalten:
  Bug A: GELÖST (GitHub Format ist [ts_ms, value] — trivial zu parsen)
  Bug B: Import-Fallback konsistent
  Bug C: GELÖST (kein Rate Limit mehr)
  Bug D: Binance Liquidations non-critical (451 = Geo-Block)
  Bug E: DeFiLlama USDT/USDC robustes Matching + Debug

$0/Monat — Alle Daten gratis:
  - BGeometrics GitHub:  MVRV, NUPL, SOPR (LTH+STH), Puell, Reserve Risk, RHODL,
                         Realized Price (LTH+STH), Supply (LTH+STH), Active Addresses,
                         Funding Rate, Open Interest, ETF, Stablecoins, Hashrate,
                         Pi Cycle, Fear & Greed
  - Binance:             Funding Rates (BTC/ETH/SOL), OI — kein Key
  - CoinGecko:           Preise (BTC/ETH/SOL/USDT/USDC), Global, Top-100 — kein Key
  - DeFiLlama:           Stablecoins Detail, TVL — kein Key
  - Blockchain.com:      Miner Revenue, Active Addresses (Fallback) — kein Key
  - V16 Sheet:           Preise, Macro State, Howell — SA Key vorhanden

Usage:
  python data_collector.py --skip-write      # Colab: nur fetchen
  python data_collector.py --skip-v16        # Ohne V16 Sheet
  python data_collector.py                   # Full run + Sheet write
"""
import os, sys, json, time, argparse, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ═══════════════════════════════════════════════════════
# CONFIG IMPORTS
# ═══════════════════════════════════════════════════════

_IMPORTS = (
    'CRYPTO_SHEET_ID', 'V16_SHEET_ID', 'V16_TABS', 'CRYPTO_TABS',
    'COINGECKO_IDS', 'RAINBOW', 'PI_CYCLE', 'HALVINGS', 'AVG_CYCLE_DAYS',
    'V16_STATE_NAMES', 'V16_MACRO_MODIFIER',
    'V16_INSTABILITY_WINDOW_DAYS', 'V16_INSTABILITY_THRESHOLD',
    'REAL_VOL_WINDOW', 'GRACEFUL_DEGRADATION', 'DEGRADATION_THRESHOLD_MINIMAL',
    'CONFIG_VERSION',
)
try:
    from step_0y_crypto.config import (
        CRYPTO_SHEET_ID, V16_SHEET_ID, V16_TABS, CRYPTO_TABS,
        COINGECKO_IDS, RAINBOW, PI_CYCLE, HALVINGS, AVG_CYCLE_DAYS,
        V16_STATE_NAMES, V16_MACRO_MODIFIER,
        V16_INSTABILITY_WINDOW_DAYS, V16_INSTABILITY_THRESHOLD,
        REAL_VOL_WINDOW, GRACEFUL_DEGRADATION, DEGRADATION_THRESHOLD_MINIMAL,
        CONFIG_VERSION,
    )
except (ImportError, ModuleNotFoundError):
    _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/content/step_0y_crypto'
    sys.path.insert(0, _dir)
    sys.path.insert(0, os.path.dirname(_dir))
    from config import (
        CRYPTO_SHEET_ID, V16_SHEET_ID, V16_TABS, CRYPTO_TABS,
        COINGECKO_IDS, RAINBOW, PI_CYCLE, HALVINGS, AVG_CYCLE_DAYS,
        V16_STATE_NAMES, V16_MACRO_MODIFIER,
        V16_INSTABILITY_WINDOW_DAYS, V16_INSTABILITY_THRESHOLD,
        REAL_VOL_WINDOW, GRACEFUL_DEGRADATION, DEGRADATION_THRESHOLD_MINIMAL,
        CONFIG_VERSION,
    )

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

import requests as req_lib

DATA = {}
FAILURES = []
NOW = datetime.now(timezone.utc)

BINANCE_FAPI = 'https://fapi.binance.com'
COINGECKO = 'https://api.coingecko.com/api/v3'

# BGeometrics GitHub Raw — KEIN Rate Limit, täglich aktualisiert
BG_RAW = 'https://raw.githubusercontent.com/BGeometrics/bgeometrics.github.io/master/files'

def log(msg): print(f"  {msg}")
def fail(k, r=""): FAILURES.append({'source': k, 'reason': r})
def pause(s=1.5): time.sleep(s)

def safe_float(v, d=None):
    try:
        if v is None or v == '' or v == 'N/A': return d
        return float(v)
    except: return d

def api_get(url, headers=None, params=None, timeout=30):
    try:
        r = req_lib.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 200: return r.json()
        log(f"    HTTP {r.status_code}: {url[:80]}")
        return None
    except Exception as e:
        log(f"    ERR: {e} — {url[:80]}")
        return None


# ═══════════════════════════════════════════════════════
# BGeometrics GitHub Raw Parser
# Format: [[timestamp_ms, value], [timestamp_ms, value], ...]
# Letztes Element = aktuellster Wert
# ═══════════════════════════════════════════════════════

def bg_latest(data):
    """Extrahiere den neuesten Wert aus BGeometrics GitHub JSON.
    Format: Liste von [timestamp_ms, value] Paaren.
    Wenn der letzte Wert None ist (noch nicht berechnet), rückwärts suchen.
    """
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    # Rückwärts iterieren — manche On-Chain Metriken haben bis 80+ None-Einträge am Ende
    for i in range(len(data)-1, max(len(data)-100, -1), -1):
        entry = data[i]
        if isinstance(entry, list) and len(entry) >= 2:
            val = safe_float(entry[1])
            if val is not None:
                return val
        else:
            val = safe_float(entry)
            if val is not None:
                return val
    return None

def bg_fetch(filename, key, label=None):
    """Fetche eine BGeometrics GitHub Raw Datei und extrahiere den neuesten Wert."""
    url = f"{BG_RAW}/{filename}"
    try:
        r = req_lib.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            val = bg_latest(data)
            if val is not None:
                DATA[key] = val
                log(f"    {key} = {val}" + (f" ({label})" if label else ""))
                return val
            else:
                fail(key, f'BG parse error: {filename}')
                return None
        else:
            log(f"    HTTP {r.status_code}: {filename}")
            fail(key, f'BG HTTP {r.status_code}: {filename}')
            return None
    except Exception as e:
        log(f"    ERR: {e} — {filename}")
        fail(key, f'BG error: {filename}')
        return None

def bg_fetch_series(filename, last_n=None):
    """Fetche eine BGeometrics GitHub Raw Datei und gib die ganze Serie zurück.
    Returns: Liste von (timestamp_ms, value) Tuples, oder [].
    """
    url = f"{BG_RAW}/{filename}"
    try:
        r = req_lib.get(url, timeout=60)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
        series = []
        for entry in data:
            if isinstance(entry, list) and len(entry) >= 2:
                ts = entry[0]
                val = safe_float(entry[1])
                if val is not None:
                    series.append((ts, val))
        if last_n and len(series) > last_n:
            series = series[-last_n:]
        return series
    except:
        return []


# ═══════════════════════════════════════════════════════
# 1. BGEOMETRICS VIA GITHUB RAW ($0, kein Rate Limit)
# ═══════════════════════════════════════════════════════

def fetch_bgeometrics():
    """BGeometrics On-Chain Daten via GitHub Raw Files.
    Kein Rate Limit — alle Files in einem Rutsch.
    """
    log("BGeometrics (GitHub Raw): On-Chain + Derivate...")

    # ─── Cycle / On-Chain (Klasse 1 + 2) ───
    bg_fetch('mvrv_zscore_data.json', 'mvrv_zscore', 'MVRV Z-Score')
    bg_fetch('nupl_data.json', 'nupl', 'NUPL')
    bg_fetch('sopr_data.json', 'sopr', 'SOPR')
    bg_fetch('lth_sopr.json', 'lth_sopr', 'LTH-SOPR')
    bg_fetch('sth_sopr.json', 'sth_sopr', 'STH-SOPR')
    bg_fetch('puell_multiple_data.json', 'puell_multiple', 'Puell Multiple')
    bg_fetch('reserve_risk.json', 'reserve_risk', 'Reserve Risk')
    bg_fetch('rhodl_1m.json', 'rhodl_ratio', 'RHODL 1M')
    bg_fetch('realized_price.json', 'realized_price', 'Realized Price')
    bg_fetch('sth_realized_price.json', 'sth_realized_price', 'STH Realized Price')
    bg_fetch('lth_realized_price.json', 'lth_realized_price', 'LTH Realized Price')
    pause(0.3)

    # ─── Supply (Klasse 2) ───
    bg_fetch('lth_supply.json', 'supply_lth', 'LTH Supply')
    bg_fetch('sth_supply.json', 'supply_sth', 'STH Supply')
    bg_fetch('profit_loss.json', 'supply_in_profit', 'Supply in Profit/Loss')
    pause(0.3)

    # ─── Network + Miners ───
    bg_fetch('addresses_active.json', 'active_addresses', 'Active Addresses')
    bg_fetch('hashrate.json', 'hashrate', 'Hashrate')
    pause(0.3)

    # ─── Derivate ───
    bg_fetch('funding_rate.json', 'funding_rate_bg', 'Funding Rate')
    bg_fetch('open_interest_futures_btc_price.json', 'open_interest_bg', 'OI Futures')
    pause(0.3)

    # ─── Stablecoins + ETF ───
    # stablecoin_supply_all.json ist ein Dict, kein Array — custom Parser
    try:
        r = req_lib.get(f"{BG_RAW}/stablecoin_supply_all.json", timeout=30)
        if r.status_code == 200:
            sd = r.json()
            if isinstance(sd, dict) and 'stablecoins' in sd:
                # 'stablecoins' enthält die Gesamtliste als [[ts, val], ...]
                val = bg_latest(sd['stablecoins'])
                if val is not None:
                    DATA['stablecoin_supply_bg'] = val
                    log(f"    stablecoin_supply_bg = {val} (Stablecoin Supply)")
                # USDT/USDC auch einzeln verfügbar
                for coin_key, data_key in [('usdt','usdt_bg'),('usdc','usdc_bg')]:
                    if coin_key in sd:
                        cv = bg_latest(sd[coin_key])
                        if cv is not None:
                            DATA[data_key] = cv
            elif isinstance(sd, list):
                val = bg_latest(sd)
                if val is not None:
                    DATA['stablecoin_supply_bg'] = val
                    log(f"    stablecoin_supply_bg = {val} (Stablecoin Supply)")
    except Exception as e:
        fail('stablecoin_supply_bg', f'BG stablecoin: {e}')

    bg_fetch('total_btc_etf_btc.json', 'etf_balance_bg', 'ETF BTC Total')
    pause(0.3)

    # ─── Display-Only ───
    bg_fetch('pi_cycle_price.json', 'pi_cycle_bg', 'Pi Cycle (Display)')

    # fear_greed_data.json ist ein Dict mit 'today', 'yesterday', 'week' — custom Parser
    try:
        r = req_lib.get(f"{BG_RAW}/fear_greed_data.json", timeout=30)
        if r.status_code == 200:
            fg = r.json()
            if isinstance(fg, dict):
                val = safe_float(fg.get('today'))
                if val is not None:
                    DATA['fear_greed_bg'] = val
                    log(f"    fear_greed_bg = {val} (Fear & Greed BG)")
    except Exception as e:
        fail('fear_greed_bg', f'BG fear_greed: {e}')

    # ─── Extras ───
    bg_fetch('lth_mvrv.json', 'lth_mvrv', 'LTH MVRV')
    bg_fetch('sth_mvrv.json', 'sth_mvrv', 'STH MVRV')

    ok_count = sum(1 for k in ['mvrv_zscore','nupl','sopr','lth_sopr','sth_sopr',
                                'puell_multiple','reserve_risk','rhodl_ratio',
                                'realized_price','sth_realized_price','lth_realized_price',
                                'supply_lth','supply_sth','active_addresses','hashrate',
                                'funding_rate_bg'] if k in DATA)
    log(f"  BGeometrics: {ok_count}/16 kritische Metriken OK")


# ═══════════════════════════════════════════════════════
# 2. BINANCE ($0, kein Key)
#    Best-Effort: 451 Geo-Block erwartet (Colab + evtl. GitHub Actions)
#    Fallback: BGeometrics GitHub für BTC Funding + OI
#    ETH/SOL Funding = nice-to-have, kein Failure
# ═══════════════════════════════════════════════════════
def fetch_binance():
    log("Binance: Funding + OI (Best-Effort, BG Fallback)...")

    binance_ok = False
    for sym, k in [('BTCUSDT','funding_btc'), ('ETHUSDT','funding_eth'), ('SOLUSDT','funding_sol')]:
        resp = api_get(f"{BINANCE_FAPI}/fapi/v1/fundingRate", params={'symbol': sym, 'limit': 9})
        if resp and isinstance(resp, list) and resp:
            rates = [safe_float(r.get('fundingRate'), 0) for r in resp]
            DATA[f'{k}_3d_avg'] = round(np.mean(rates) * 100, 4)
            log(f"    {sym} Funding 3d: {DATA[f'{k}_3d_avg']:.4f}%")
            binance_ok = True
        else:
            DATA[f'{k}_3d_avg'] = 0
            # Kein fail() — Binance 451 ist erwartet, BG Fallback greift
            log(f"    {sym}: Binance nicht verfügbar")
        pause(0.5)

    # OI
    resp = api_get(f"{BINANCE_FAPI}/fapi/v1/openInterest", params={'symbol': 'BTCUSDT'})
    if resp:
        DATA['oi_btc'] = safe_float(resp.get('openInterest'))
        log(f"    BTC OI (Binance): {DATA.get('oi_btc')}")
        binance_ok = True
    else:
        log(f"    BTC OI: Binance nicht verfügbar")
    pause(0.5)

    # Fallback: BGeometrics für Funding + OI wenn Binance komplett failed
    if not binance_ok:
        log(f"    Binance komplett geblockt — BGeometrics Fallback aktiv")
        if DATA.get('funding_rate_bg') is not None:
            # BG Funding Rate in % umrechnen (BG liefert als Dezimal, z.B. 2.41e-06)
            DATA['funding_btc_3d_avg'] = round(DATA['funding_rate_bg'] * 100, 4)
            log(f"    BTC Funding (BG Fallback): {DATA['funding_btc_3d_avg']:.4f}%")
        if DATA.get('open_interest_bg') is not None and not DATA.get('oi_btc'):
            DATA['oi_btc'] = DATA['open_interest_bg']
            log(f"    BTC OI (BG Fallback): {DATA['oi_btc']}")

    # Liquidations — NON-CRITICAL
    try:
        r = req_lib.get(f"{BINANCE_FAPI}/fapi/v1/forceOrders",
                        params={'symbol': 'BTCUSDT', 'limit': 100}, timeout=30)
        if r.status_code == 200:
            resp = r.json()
            if isinstance(resp, list):
                total = sum(safe_float(o.get('origQty'),0) * safe_float(o.get('averagePrice'),0) for o in resp)
                DATA['liquidations_24h'] = round(total, 0)
                log(f"    Liquidations: ${total:,.0f}")
            else:
                DATA['liquidations_24h'] = None
        elif r.status_code in (401, 403, 451):
            DATA['liquidations_24h'] = None
            log(f"    Liquidations: HTTP {r.status_code} (non-critical)")
        else:
            DATA['liquidations_24h'] = None
    except Exception as e:
        DATA['liquidations_24h'] = None
    pause(0.5)


# ═══════════════════════════════════════════════════════
# 3. COINGECKO ($0) — Preise + Global
# ═══════════════════════════════════════════════════════
def fetch_coingecko():
    log("CoinGecko: Preise + Global...")
    resp = api_get(f"{COINGECKO}/simple/price", params={
        'ids': 'bitcoin,ethereum,solana,tether,usd-coin', 'vs_currencies': 'usd',
        'include_24hr_change': 'true', 'include_market_cap': 'true'})
    if resp:
        DATA['btc_price'] = safe_float(resp.get('bitcoin',{}).get('usd'))
        DATA['eth_price'] = safe_float(resp.get('ethereum',{}).get('usd'))
        DATA['sol_price'] = safe_float(resp.get('solana',{}).get('usd'))
        DATA['usdt_price'] = safe_float(resp.get('tether',{}).get('usd'))
        DATA['usdc_price'] = safe_float(resp.get('usd-coin',{}).get('usd'))
        DATA['btc_24h_chg'] = safe_float(resp.get('bitcoin',{}).get('usd_24h_change'))
        DATA['eth_24h_chg'] = safe_float(resp.get('ethereum',{}).get('usd_24h_change'))
        DATA['sol_24h_chg'] = safe_float(resp.get('solana',{}).get('usd_24h_change'))
        DATA['btc_mcap'] = safe_float(resp.get('bitcoin',{}).get('usd_market_cap'))
        log(f"    BTC=${DATA.get('btc_price','?'):,.2f} ETH=${DATA.get('eth_price','?'):,.2f} SOL=${DATA.get('sol_price','?'):,.2f}")
    else:
        fail('btc_price', 'CoinGecko prices')
    pause(1.5)

    resp = api_get(f"{COINGECKO}/global")
    if resp and 'data' in resp:
        DATA['btc_dominance'] = safe_float(resp['data'].get('market_cap_percentage',{}).get('btc'))
        DATA['total_crypto_mcap'] = safe_float(resp['data'].get('total_market_cap',{}).get('usd'))
        log(f"    BTC Dom: {DATA.get('btc_dominance','?'):.1f}%")
    else:
        fail('btc_dominance', 'CoinGecko global')


# ═══════════════════════════════════════════════════════
# 3b. COINGECKO TOP-100 (separiert, am Ende, 3s Pause)
#     Liefert nur meme_explosion_count — non-critical
# ═══════════════════════════════════════════════════════
def fetch_coingecko_top100():
    log("CoinGecko: Top-100 (Meme-Check)...")
    pause(3.0)  # Extra Pause nach allen anderen CoinGecko Calls
    resp = api_get(f"{COINGECKO}/coins/markets", params={
        'vs_currency':'usd','order':'market_cap_desc','per_page':100,'page':1,
        'sparkline':'false','price_change_percentage':'7d'})
    if resp and isinstance(resp, list):
        DATA['top_100'] = resp
        DATA['meme_explosion_count'] = sum(1 for c in resp
            if safe_float(c.get('price_change_percentage_7d_in_currency'),0) > 200
            and safe_float(c.get('market_cap'),0) > 1e9)
        log(f"    Top-100 OK, Meme: {DATA['meme_explosion_count']}")
    else:
        # Non-critical — kein fail(), default 0
        DATA['meme_explosion_count'] = 0
        log(f"    Top-100: nicht verfügbar (Rate Limit), Meme: 0 (default)")


# ═══════════════════════════════════════════════════════
# 4. DEFILLAMA ($0)
#    Bug E: Robustes USDT/USDC Matching + Debug
# ═══════════════════════════════════════════════════════
def fetch_defillama():
    log("DeFiLlama: Stablecoins + TVL...")
    resp = api_get('https://stablecoins.llama.fi/stablecoins?includePrices=true')
    if resp and 'peggedAssets' in resp:
        total, usdt, usdc = 0, 0, 0
        debug_top5 = []

        for a in resp['peggedAssets']:
            nm = (a.get('name') or '').strip()
            sym = (a.get('symbol') or '').strip()
            nm_up = nm.upper()
            sym_up = sym.upper()

            circ = a.get('circulating')
            mc = 0
            if isinstance(circ, dict):
                mc = safe_float(circ.get('peggedUSD'), 0)
            elif isinstance(circ, (int, float)):
                mc = safe_float(circ, 0)

            total += mc

            is_usdt = (sym_up == 'USDT' or 'TETHER' in nm_up)
            is_usdc = (sym_up == 'USDC' or 'USD COIN' in nm_up)

            if is_usdt and mc > usdt:
                usdt = mc
            elif is_usdc and mc > usdc:
                usdc = mc

            if len(debug_top5) < 5:
                debug_top5.append({'name': nm, 'symbol': sym, 'mc': mc})
            elif mc > min(d['mc'] for d in debug_top5):
                debug_top5.sort(key=lambda x: x['mc'])
                debug_top5[0] = {'name': nm, 'symbol': sym, 'mc': mc}

        DATA['stablecoin_supply_total'] = total
        DATA['usdt_supply'] = usdt; DATA['usdc_supply'] = usdc
        DATA['usdt_usdc_ratio'] = round(usdt / usdc, 3) if usdc > 0 else 0
        log(f"    Stable: ${total/1e9:.1f}B  USDT: ${usdt/1e9:.1f}B  USDC: ${usdc/1e9:.1f}B")
        log(f"    USDT/USDC Ratio: {DATA['usdt_usdc_ratio']:.3f}")

        debug_top5.sort(key=lambda x: x['mc'], reverse=True)
        for i, d in enumerate(debug_top5):
            log(f"    [DEBUG] Top-{i+1}: {d['name']} ({d['symbol']}) ${d['mc']/1e9:.1f}B")
    else:
        fail('stablecoin_supply', 'DeFiLlama')
    pause(1.5)

    resp = api_get('https://api.llama.fi/v2/historicalChainTvl')
    if resp and isinstance(resp, list) and resp:
        DATA['defi_tvl_total'] = safe_float(resp[-1].get('tvl'))
        log(f"    TVL: ${DATA.get('defi_tvl_total',0)/1e9:.1f}B")
    else:
        fail('defi_tvl', 'DeFiLlama TVL')
    pause(1.0)


# ═══════════════════════════════════════════════════════
# 5. FEAR & GREED ($0) — alternative.me als Primary
#    BGeometrics GitHub hat auch fear_greed_data.json (Backup)
# ═══════════════════════════════════════════════════════
def fetch_fear_greed():
    log("Fear & Greed...")
    resp = api_get('https://api.alternative.me/fng/?limit=30&format=json')
    if resp and 'data' in resp and resp['data']:
        DATA['fear_greed_raw'] = int(resp['data'][0].get('value', 50))
        DATA['fear_greed_7d_avg'] = round(np.mean([int(e.get('value',50)) for e in resp['data'][:7]]), 1)
        log(f"    F&G: {DATA['fear_greed_raw']}, 7d: {DATA['fear_greed_7d_avg']}")
    else:
        # Fallback: BGeometrics GitHub
        if 'fear_greed_bg' in DATA:
            DATA['fear_greed_raw'] = int(DATA['fear_greed_bg'])
            DATA['fear_greed_7d_avg'] = int(DATA['fear_greed_bg'])
            log(f"    F&G: {DATA['fear_greed_raw']} (BG Fallback)")
        else:
            fail('fear_greed', 'alternative.me + BG')


# ═══════════════════════════════════════════════════════
# 6. BLOCKCHAIN.COM ($0) — Puell Fallback + Active Addr Fallback
# ═══════════════════════════════════════════════════════
def fetch_blockchain_com():
    log("Blockchain.com: Miner Rev + Addresses...")
    resp = api_get('https://api.blockchain.info/charts/miners-revenue?timespan=2years&format=json')
    if resp and 'values' in resp and len(resp['values']) >= 365:
        v = resp['values']
        latest = safe_float(v[-1].get('y'), 0)
        avg = np.mean([safe_float(x.get('y'),0) for x in v[-365:]])
        if avg > 0:
            DATA['puell_calc'] = round(latest/avg, 3)
            DATA['miner_revenue'] = round(latest, 0)
            log(f"    Puell(calc): {DATA['puell_calc']:.2f}")
    else:
        fail('puell_multiple', 'Blockchain.com')
    pause(1.0)

    resp = api_get('https://api.blockchain.info/charts/n-unique-addresses?timespan=30days&format=json')
    if resp and 'values' in resp and resp['values']:
        DATA['active_addresses_bc'] = int(safe_float(resp['values'][-1].get('y'), 0))
        log(f"    Active Addr: {DATA['active_addresses_bc']:,}")
    else:
        fail('active_addresses', 'Blockchain.com')
    pause(1.0)


# ═══════════════════════════════════════════════════════
# 7. V16 SHEET
# ═══════════════════════════════════════════════════════
def fetch_v16_sheet(gc):
    log("V16 Sheet...")
    try:
        sh = gc.open_by_key(V16_SHEET_ID)
        ws = sh.worksheet(V16_TABS['macro_state'])
        vals = ws.get_all_values()
        if len(vals) > 2:
            hdrs = vals[0]
            mc = next((i for i,h in enumerate(hdrs) if 'macro_state_num' in h.lower()), None)
            if mc is not None:
                for row in vals[1:6]:
                    if row[0].strip():
                        ms = safe_float(row[mc])
                        if ms is not None:
                            DATA['v16_macro_state'] = int(ms)
                            DATA['v16_state_name'] = V16_STATE_NAMES.get(int(ms), '?')
                            DATA['v16_modifier'] = V16_MACRO_MODIFIER.get(int(ms), 1.0)
                            log(f"    V16: {DATA['v16_state_name']} (×{DATA['v16_modifier']})")
                            break
                if len(vals) > 31:
                    states = set()
                    for row in vals[1:31]:
                        s = safe_float(row[mc])
                        if s: states.add(int(s))
                    DATA['v16_states_30d'] = len(states)
                    DATA['v16_unstable'] = len(states) >= V16_INSTABILITY_THRESHOLD
        pause(1.0)
        try:
            ws = sh.worksheet(V16_TABS['k16_k17'])
            vals = ws.get_all_values()
            if len(vals) > 2:
                hdrs = [h.strip() for h in vals[0]]
                lc = next((i for i,h in enumerate(hdrs) if 'liq_dir' in h.lower()), None)
                vc = next((i for i,h in enumerate(hdrs) if 'vote_sum' in h.lower()), None)
                for row in vals[1:6]:
                    if row[0].strip():
                        if lc: DATA['howell_liq_dir'] = safe_float(row[lc], 0)
                        if vc: DATA['howell_vote_sum'] = safe_float(row[vc].replace(',','.') if isinstance(row[vc],str) else row[vc], 0)
                        break
        except: fail('howell_liquidity', 'V16 Howell')
    except Exception as e:
        fail('v16_macro_state', str(e))

def fetch_btc_history(gc):
    log("V16: BTC Historie...")
    try:
        sh = gc.open_by_key(V16_SHEET_ID)
        ws = sh.worksheet(V16_TABS['prices'])
        vals = ws.get_all_values()
        if len(vals) < 100: return []
        hdrs = vals[0]
        bc = next((i for i,h in enumerate(hdrs) if 'BTC' in h.upper()), None)
        if bc is None: return []
        hist = []
        for row in vals[1:]:
            if not row[0].strip(): continue
            try:
                s = row[bc].replace('.','').replace(',','.') if isinstance(row[bc], str) else str(row[bc])
                v = float(s)
                if v > 0: hist.append((row[0].strip(), v))
            except: continue
        hist.sort(key=lambda x: x[0])
        log(f"    {len(hist)} Tage")
        return hist
    except Exception as e:
        log(f"    ERR: {e}")
        return []


# ═══════════════════════════════════════════════════════
# 8. LOKALE BERECHNUNGEN
# ═══════════════════════════════════════════════════════
def calc_local(btc_hist):
    if not btc_hist or len(btc_hist) < 200:
        log("Lokal: Zu wenig Historie"); return
    prices = np.array([p for _,p in btc_hist])
    dates = [d for d,_ in btc_hist]
    n = len(prices)
    btc_s = pd.Series(prices)
    log(f"Lokal: {n} Tage")

    # Rainbow
    genesis = pd.Timestamp(RAINBOW['genesis_date'])
    days = np.array([(pd.Timestamp(d)-genesis).days for d in dates], dtype=float)
    fitted = RAINBOW['a']*np.log(np.maximum(days,1))+RAINBOW['b']
    res = np.log(max(prices[-1],0.01)) - fitted[-1]
    sc = res/RAINBOW['residual_std'] if RAINBOW['residual_std']>0 else 0
    DATA['rainbow_band'] = int(np.clip(round(sc*1.75+4.5),1,8))
    DATA['rainbow_score'] = round(sc, 3)
    log(f"    Rainbow: Band {DATA['rainbow_band']}, Score {sc:.2f}")

    # Pi (Display)
    if n >= PI_CYCLE['top_ema_period']:
        s111 = btc_s.rolling(111,min_periods=111).mean()
        e350x2 = btc_s.ewm(span=350,adjust=False).mean()*2.0
        DATA['pi_top_111sma'] = round(s111.iloc[-1],2) if pd.notna(s111.iloc[-1]) else None
        DATA['pi_top_2x350ema'] = round(e350x2.iloc[-1],2) if pd.notna(e350x2.iloc[-1]) else None
        if DATA.get('pi_top_111sma') and DATA.get('pi_top_2x350ema') and DATA['pi_top_2x350ema']>0:
            DATA['pi_top_proximity'] = round((DATA['pi_top_111sma']-DATA['pi_top_2x350ema'])/DATA['pi_top_2x350ema'],4)

    # 200-DMA
    if n >= 230:
        dma = btc_s.rolling(200).mean()
        DATA['btc_200dma'] = round(dma.iloc[-1],2)
        DATA['btc_200dma_slope'] = round((dma.iloc[-1]-dma.iloc[-31])/dma.iloc[-31],4) if dma.iloc[-31]>0 else 0
        DATA['below_200dma_falling'] = prices[-1]<dma.iloc[-1] and DATA.get('btc_200dma_slope',0)<0

    # 200-WMA
    if n >= 700:
        wma = btc_s.rolling(min(1400,n),min_periods=700).mean()
        DATA['btc_200wma'] = round(wma.iloc[-1],2)
        DATA['below_200wma'] = prices[-1]<wma.iloc[-1]

    # 2YR MA
    if n >= 730:
        m2 = btc_s.rolling(730).mean()
        DATA['btc_2yr_ma'] = round(m2.iloc[-1],2)
        DATA['btc_2yr_ma_x5'] = round(m2.iloc[-1]*5,2)

    # Halving
    hd = [pd.Timestamp(h) for h in HALVINGS]
    today = pd.Timestamp(NOW.date())
    past = [h for h in hd if h<=today]
    if past:
        ds = (today-past[-1]).days
        DATA['days_since_halving'] = ds
        DATA['halving_phase'] = round(min(ds/AVG_CYCLE_DAYS,1.0),3)

    # Vol
    if n >= REAL_VOL_WINDOW+1:
        r = np.diff(np.log(np.maximum(prices[-REAL_VOL_WINDOW-1:],0.01)))
        DATA['btc_realvol_60d'] = round(np.std(r)*np.sqrt(252),4)

    # Changes
    for d,k in [(7,'7d'),(30,'30d'),(90,'90d')]:
        if n>d: DATA[f'btc_{k}_chg'] = round((prices[-1]/prices[-(d+1)]-1)*100,2)

    # DD
    if n>=252:
        ath = np.max(prices[-252:])
        DATA['btc_ath_252d'] = round(ath,2)
        DATA['btc_dd_from_ath'] = round(prices[-1]/ath-1,4)
    if n>=21:
        DATA['btc_dd_21d'] = round(prices[-1]/np.max(prices[-21:])-1,4)

    # Ratios
    if DATA.get('btc_price') and DATA.get('eth_price'):
        DATA['eth_btc_ratio'] = round(DATA['eth_price']/DATA['btc_price'],6)
    if DATA.get('btc_price') and DATA.get('sol_price'):
        DATA['sol_btc_ratio'] = round(DATA['sol_price']/DATA['btc_price'],6)
    if DATA.get('oi_btc') and DATA.get('btc_mcap') and DATA['btc_mcap']>0:
        DATA['oi_mcap_ratio'] = round(DATA['oi_btc']/DATA['btc_mcap'],4)


# ═══════════════════════════════════════════════════════
# 9. DEGRADATION + OUTPUT
# ═══════════════════════════════════════════════════════
def calc_deg():
    crit = any(GRACEFUL_DEGRADATION.get(f['source'],{}).get('critical',False) for f in FAILURES)
    if crit: return 3,'OFFLINE'
    n = len(FAILURES)
    if n > DEGRADATION_THRESHOLD_MINIMAL: return 2,'MINIMAL'
    if n > 0: return 1,'DEGRADED'
    return 0,'NOMINAL'

def write_json():
    dl,dn = calc_deg()
    out = {'metadata':{'generated_at':NOW.isoformat(),'version':CONFIG_VERSION,
           'degradation':dn,'ok':len(DATA),'failed':len(FAILURES),'failures':FAILURES,
           'cost':'$0','bg_source':'github_raw'},
           'data':{k:v for k,v in DATA.items() if not isinstance(v,list)}}
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d,'crypto_raw_data.json')
    with open(p,'w',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False,default=str)
    log(f"JSON: {p}")

def write_sheet(gc):
    log("Sheet: DATA_Raw...")
    try:
        sh = gc.open_by_key(CRYPTO_SHEET_ID)
        ws = sh.worksheet(CRYPTO_TABS['data_raw'])
        _,dn = calc_deg()

        # Fallback-Logik für Felder mit mehreren Quellen
        puell = DATA.get('puell_multiple') or DATA.get('puell_calc') or ''
        addr = DATA.get('active_addresses') or DATA.get('active_addresses_bc') or ''
        new_addr = DATA.get('new_addresses') or ''
        dom = DATA.get('btc_dominance') or ''
        fund_btc = DATA.get('funding_btc_3d_avg') or DATA.get('funding_rate_bg') or ''
        fund_eth = DATA.get('funding_eth_3d_avg') or ''
        fund_sol = DATA.get('funding_sol_3d_avg') or ''
        stablecoin_bg = DATA.get('stablecoin_supply_bg') or ''
        etf_bal = DATA.get('etf_balance_bg') or ''

        # 92 Spalten — Reihenfolge MUSS exakt dem Sheet-Header entsprechen
        row = [
            # [1-3] META
            NOW.strftime('%Y-%m-%d'), 'weekly_full', dn,
            # [4-6] PREISE
            DATA.get('btc_price',''), DATA.get('eth_price',''), DATA.get('sol_price',''),
            # [7-12] CHANGES
            DATA.get('btc_7d_chg',''), DATA.get('eth_24h_chg',''), DATA.get('sol_24h_chg',''),
            DATA.get('btc_30d_chg',''), '', '',
            # [13-14] RAINBOW
            DATA.get('rainbow_band',''), DATA.get('rainbow_score',''),
            # [15-20] PI CYCLE (Display)
            DATA.get('pi_top_111sma',''), DATA.get('pi_top_2x350ema',''), DATA.get('pi_top_proximity',''),
            '', '', '',  # Pi Bottom — braucht Pre-2010 Daten
            # [21-27] MOVING AVERAGES
            DATA.get('btc_200dma',''), DATA.get('btc_200dma_slope',''), DATA.get('btc_200wma',''),
            DATA.get('btc_2yr_ma',''), DATA.get('btc_2yr_ma_x5',''),
            DATA.get('below_200dma_falling',''), DATA.get('below_200wma',''),
            # [28-38] ON-CHAIN
            DATA.get('mvrv_zscore',''), puell, DATA.get('nupl',''), DATA.get('reserve_risk',''),
            DATA.get('lth_sopr',''), DATA.get('sth_sopr',''),
            DATA.get('supply_lth',''), DATA.get('supply_sth',''),
            DATA.get('sth_realized_price',''), DATA.get('lth_realized_price',''), DATA.get('rhodl_ratio',''),
            # [39-44] DISPLAY-ONLY (Engines)
            '', '', '', '', '', '',
            # [45-47] PEAK COMPOSITE (eigener Score, Engines)
            '', '', '',
            # [48-50] HALVING + ETF REGIME
            DATA.get('days_since_halving',''), DATA.get('halving_phase',''), '',
            # [51-57] SENTIMENT + DERIVATE
            DATA.get('fear_greed_7d_avg',''),
            fund_btc, fund_eth, fund_sol,
            DATA.get('oi_btc',''), DATA.get('oi_mcap_ratio',''), DATA.get('liquidations_24h',''),
            # [58-64] STABLECOINS + DEFI
            DATA.get('stablecoin_supply_total',''), '', '',
            DATA.get('usdt_usdc_ratio',''), '',
            DATA.get('defi_tvl_total',''), '',
            # [65-68] ETF + EXCHANGE
            etf_bal, '', '', '',
            # [69-70] KORRELATION (Engines)
            '', '',
            # [71-73] VOL + M2
            DATA.get('btc_realvol_60d',''), '', '',
            # [74-80] ALTSEASON
            dom, '', '',
            DATA.get('eth_btc_ratio',''), '', DATA.get('sol_btc_ratio',''), '',
            # [81-82] NETWORK
            addr, new_addr,
            # [83-87] V16 BRIDGE
            DATA.get('v16_macro_state',''), DATA.get('v16_state_name',''), DATA.get('v16_modifier',''),
            DATA.get('howell_liq_dir',''), DATA.get('howell_vote_sum',''),
            # [88-92] KILL SWITCHES + EXTRAS
            DATA.get('usdt_price',''), DATA.get('usdc_price',''),
            DATA.get('meme_explosion_count',''),
            DATA.get('v16_states_30d',''), DATA.get('v16_unstable',''),
        ]

        # Alles als Strings — USER_ENTERED mit englischem Locale
        # interpretiert "0.701" korrekt als Dezimalzahl
        clean = []
        for v in row:
            if v is None or v == '':
                clean.append('')
            elif isinstance(v, bool):
                clean.append('TRUE' if v else 'FALSE')
            elif isinstance(v, float):
                # Dezimalzahlen: volle Präzision als String mit Punkt
                clean.append(f'{v}')
            elif isinstance(v, int):
                clean.append(str(v))
            else:
                clean.append(str(v))

        ws.append_row(clean, value_input_option='USER_ENTERED')
        log(f"    ✅ {len(clean)} Werte ({dn})")
    except Exception as e:
        log(f"    ERR: {e}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--skip-write', action='store_true')
    pa.add_argument('--skip-v16', action='store_true')
    args = pa.parse_args()

    print("="*70)
    print("CRYPTO CIRCLE — DATA COLLECTOR V1.3.1 ($0/Monat)")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  BGeometrics: GitHub Raw (kein Rate Limit)")
    print(f"  Quellen: BG-GitHub + Binance + CoinGecko + DeFiLlama + Blockchain.com + V16")
    print("="*70)

    t0 = time.time()
    gc = None
    if HAS_GSPREAD and not args.skip_write:
        for p in ['/content/service_account.json','/content/drive/MyDrive/keys/service_account.json','service_account.json']:
            if os.path.exists(p):
                gc = gspread.authorize(Credentials.from_service_account_file(p, scopes=[
                    'https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']))
                log(f"SA: {p}"); break
        if not gc:
            sa = os.environ.get('GOOGLE_CREDENTIALS') or os.environ.get('GCP_SA_KEY')
            if sa:
                gc = gspread.authorize(Credentials.from_service_account_info(json.loads(sa), scopes=[
                    'https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']))
                log("SA: Env")

    print("\n── SAMMELN ──")
    fetch_bgeometrics()
    fetch_binance()
    fetch_coingecko()
    fetch_defillama()
    fetch_fear_greed()
    fetch_blockchain_com()
    if gc and not args.skip_v16:
        fetch_v16_sheet(gc)
        h = fetch_btc_history(gc)
        if h: calc_local(h)
    # Top-100 am Ende — maximale Pause seit letztem CoinGecko Call
    fetch_coingecko_top100()

    dl,dn = calc_deg()
    print(f"\n── ERGEBNIS ──")
    print(f"  OK: {len(DATA)} | Fehler: {len(FAILURES)} | {dn}")
    for f in FAILURES: print(f"    ❌ {f['source']}: {f['reason']}")

    print(f"\n── WERTE ──")
    for k in ['btc_price','rainbow_band','mvrv_zscore','puell_multiple','nupl',
              'lth_sopr','sth_sopr','reserve_risk','rhodl_ratio',
              'realized_price','sth_realized_price','lth_realized_price',
              'supply_lth','supply_sth','active_addresses','hashrate',
              'fear_greed_7d_avg','funding_btc_3d_avg','funding_rate_bg',
              'oi_btc','oi_mcap_ratio','liquidations_24h',
              'stablecoin_supply_total','usdt_supply','usdc_supply','usdt_usdc_ratio',
              'etf_balance_bg','lth_mvrv','sth_mvrv',
              'v16_state_name','v16_modifier','btc_realvol_60d','halving_phase',
              'btc_dominance','meme_explosion_count',
              'below_200dma_falling','below_200wma']:
        v = DATA.get(k)
        if v is not None: print(f"  {k}: {v}")

    print(f"\n── OUTPUT ──")
    write_json()
    if gc and not args.skip_write: write_sheet(gc)
    elif args.skip_write: log("Sheet übersprungen")

    elapsed = time.time()-t0
    print(f"\n{'='*70}")
    print(f"FERTIG — {elapsed:.0f}s, {len(DATA)} Punkte, {dn}, $0")
    print(f"  BGeometrics: GitHub Raw (kein Rate Limit, ~25 Files)")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
