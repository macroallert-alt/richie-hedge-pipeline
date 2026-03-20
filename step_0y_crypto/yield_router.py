#!/usr/bin/env python3
"""
yield_router.py — Crypto Cash Management Advisor V1.0
======================================================
Baldur Creek Capital | Circle 17 (Crypto Hub) | Yield Router

Systematisches Cash-Management für den Cash-Anteil des V8+Warn Portfolios.
3-Tier System:
  T0 — Liquid Cash (0% APY, sofort verfügbar)
  T1 — Tokenized T-Bills (3.5-5% APY, 1-2 Tage Redemption)
  T2 — DeFi Lending (2-4% risk-adjusted APY, battle-tested Protokolle)
  T3 — Basis Trade (Display only, nicht automatisiert)

Datenquellen:
  - DeFiLlama /pools (Yield Daten, $0, kein Key)
  - DeFiLlama /stablecoinprices (Depeg Check, $0, kein Key)
  - crypto_state.json (Ensemble, Cash-Anteil)

Pipeline-Position:
  Weekly: data_collector → cycle_engine → signal_engine → risk_engine
          → yield_router → Sheet Write + crypto_yield.json + Git Push

Output:
  - crypto_yield.json (für Frontend)
  - CALC_Yield Sheet Tab (1 Zeile pro Woche)

Usage:
  # GitHub Actions (nach risk_engine im Orchestrator):
  python -m step_0y_crypto.yield_router

  # Colab Test (kein Sheet, kein Telegram):
  python yield_router.py --skip-write --skip-telegram

  # Colab Test mit manuellem Kapital + Ensemble:
  python yield_router.py --skip-write --skip-telegram --capital 10000 --ensemble 0.25

Spec: YIELD_ROUTER_SPEC_TEIL1.md + TEIL2.md
"""
import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════
# CONFIG — YIELD ROUTER KONSTANTEN
# Quelle: YIELD_ROUTER_SPEC_TEIL1+2
# Wird in config.py V2.1 ausgelagert
# ═══════════════════════════════════════════════════════

# Ensemble → Regime → Tier-Gewichte (Spec §5.1)
YIELD_ENSEMBLE_REGIMES = {
    'BULL':     {'min': 0.75, 'max': 1.00},
    'MODERATE': {'min': 0.50, 'max': 0.74},
    'CAUTIOUS': {'min': 0.25, 'max': 0.49},
    'BEAR':     {'min': 0.00, 'max': 0.24},
}

YIELD_REGIME_WEIGHTS = {
    'BULL':     {'T0': 0.80, 'T1': 0.20, 'T2': 0.00},
    'MODERATE': {'T0': 0.40, 'T1': 0.40, 'T2': 0.20},
    'CAUTIOUS': {'T0': 0.30, 'T1': 0.50, 'T2': 0.20},
    'BEAR':     {'T0': 0.20, 'T1': 0.50, 'T2': 0.30},
}

# Kapital-Tiers (Spec §5.3)
YIELD_CAPITAL_TIERS = {
    'MICRO':    {'max_cash_eur': 5000},
    'STANDARD': {'max_cash_eur': 50000},
    'LARGE':    {'max_cash_eur': float('inf')},
}

# Depeg Kill Switch (Spec §6.1)
YIELD_DEPEG = {
    'kill_threshold': 0.02,     # >2% = KILL
    'warning_threshold': 0.005, # >0.5% = WARNING
    'warning_score_penalty': 2,
    'warning_max_weight': 0.30,
}

# Stablecoin Safety (Spec §3.3)
YIELD_STABLECOIN_SAFETY = {
    'USDC': {'score': 9, 'issuer': 'Circle'},
    'USDT': {'score': 7, 'issuer': 'Tether'},
    'DAI':  {'score': 7, 'issuer': 'Maker'},
    'USDS': {'score': 7, 'issuer': 'Maker'},
}

# Issuer-Gruppen für Diversifikation (Spec §8.1)
YIELD_ISSUER_GROUPS = {
    'Circle': ['USDC'],
    'Tether': ['USDT'],
    'Maker':  ['DAI', 'USDS', 'sDAI'],
}

# T0 Split (Spec §3.1)
YIELD_T0_SPLIT = {
    'small_threshold': 2000,  # EUR
    'small': {'USDC': 1.00},
    'large': {'USDC': 0.60, 'USDT': 0.40},
}

# T1 Produkte (Spec §3.2)
YIELD_T1_PRODUCTS = {
    'USDY': {'issuer': 'Ondo Finance', 'chain': 'Ethereum', 'redemption_days': 2,
             'defillama_slug': 'ondo-yield-assets', 'static_apy': 4.25, 'min_usd': 500},
    'sDAI': {'issuer': 'MakerDAO DSR', 'chain': 'Ethereum', 'redemption_days': 0,
             'defillama_slug': 'spark-savings', 'static_apy': 3.50, 'min_usd': 0},
    'USDM': {'issuer': 'Mountain Protocol', 'chain': 'Ethereum', 'redemption_days': 2,
             'defillama_slug': 'mountain-protocol', 'static_apy': 4.00, 'min_usd': 100},
}

# T1 Split (Spec §3.2)
YIELD_T1_SPLIT = {
    'small_threshold': 3000,  # EUR
    'small': {'sDAI': 1.00},
    'large': {'USDY': 0.60, 'sDAI': 0.40},
}

# T2 Protokolle (Spec §3.3)
YIELD_T2_PROTOCOLS = {
    'aave-v3':       {'safety': 10, 'expected_loss': 0.0000},
    'aave-v2':       {'safety': 9,  'expected_loss': 0.0000},
    'compound-v3':   {'safety': 9,  'expected_loss': 0.0000},
    'compound-v2':   {'safety': 8,  'expected_loss': 0.0000},
    'spark-savings': {'safety': 8,  'expected_loss': 0.0020},
    'sparklend':     {'safety': 8,  'expected_loss': 0.0020},
    'morpho-v1':     {'safety': 7,  'expected_loss': 0.0050},
}

# Chain Risk Premium (Spec §3.3)
YIELD_CHAIN_RISK = {
    'Ethereum': 0.0000,
    'Arbitrum': 0.0030,
    'Base':     0.0030,
}

# T2 Filter (Spec §9)
YIELD_T2_FILTERS = {
    'min_tvl_usd': 10_000_000,      # $10M
    'min_apy': 0.5,                   # 0.5%
    'max_apy': 15.0,                  # 15% (höher = suspicious)
    'max_issuer_pct': 0.60,           # 60% pro Issuer-Gruppe
    'max_protocol_pct': 0.40,         # 40% pro Protokoll
    'max_chain_pct': 0.70,            # 70% pro Chain (nur bei LARGE)
    'stablecoins': ['USDC', 'USDT', 'DAI', 'USDS'],
    'chains_standard': ['Ethereum'],
    'chains_large': ['Ethereum', 'Arbitrum', 'Base'],
}

# T2 Pool-Anzahl nach Betrag (Spec §9)
YIELD_T2_POOL_COUNT = {
    'small_threshold': 5000,   # EUR
    'medium_threshold': 20000, # EUR
    'small': 2,
    'medium': 3,
    'large': 4,
}

# Rebalancing (Spec §10)
YIELD_REBALANCING = {
    'apy_delta_trigger': 1.5,   # pp
    'min_days_between': 30,
}

# T3 Basis Trade Display (Spec §3.4)
YIELD_T3_DISPLAY = {
    'bull_apy_range': '8-15%',
    'neutral_apy_range': '3-8%',
    'bear_apy_range': '0-3%',
    'note': 'Für fortgeschrittene Nutzer mit dediziertem CEX-Setup',
}

# DeFiLlama Endpoints
DEFILLAMA_POOLS_URL = 'https://yields.llama.fi/pools'
DEFILLAMA_STABLECOIN_PRICES_URL = 'https://stablecoins.llama.fi/stablecoinprices'


# ═══════════════════════════════════════════════════════
# EXTERNAL CONFIG IMPORTS (Sheet IDs etc.)
# ═══════════════════════════════════════════════════════
try:
    from step_0y_crypto.config import CRYPTO_SHEET_ID, CRYPTO_TABS, CONFIG_VERSION
except (ImportError, ModuleNotFoundError):
    _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/content/step_0y_crypto'
    sys.path.insert(0, _dir)
    sys.path.insert(0, os.path.dirname(_dir))
    try:
        from config import CRYPTO_SHEET_ID, CRYPTO_TABS, CONFIG_VERSION
    except ImportError:
        CRYPTO_SHEET_ID = None
        CRYPTO_TABS = {}
        CONFIG_VERSION = '?'

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

NOW = datetime.now(timezone.utc)
VERSION = "yield_router V1.0"

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


# ═══════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════
def log(msg):
    print(f"  [YIELD] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# GCP AUTH (gleiche Logik wie Orchestrator)
# ═══════════════════════════════════════════════════════
def get_gspread_client():
    if not HAS_GSPREAD:
        return None
    scopes = ['https://www.googleapis.com/auth/spreadsheets',
              'https://www.googleapis.com/auth/drive']
    for p in ['/content/service_account.json',
              '/content/drive/MyDrive/keys/service_account.json',
              'service_account.json']:
        if os.path.exists(p):
            return gspread.authorize(Credentials.from_service_account_file(p, scopes=scopes))
    sa = os.environ.get('GOOGLE_CREDENTIALS') or os.environ.get('GCP_SA_KEY')
    if sa:
        return gspread.authorize(Credentials.from_service_account_info(
            json.loads(sa), scopes=scopes))
    return None


# ═══════════════════════════════════════════════════════
# SCHRITT 1: DEPEG KILL SWITCH (Spec §6)
# ═══════════════════════════════════════════════════════

def fetch_stablecoin_prices():
    """Fetch Stablecoin-Preise von DeFiLlama.

    API Response Format: Array von {date, prices} Objekten.
    Erstes Element (date=0) enthält aktuelle Preise.
    Preise sind unter CoinGecko IDs gespeichert (z.B. "usd-coin" für USDC).

    Returns dict: {'USDC': 1.0002, 'USDT': 0.9998, ...}
    """
    import requests

    # CoinGecko ID → unser Symbol
    COINGECKO_ID_MAP = {
        'usd-coin': 'USDC',
        'tether': 'USDT',
        'dai': 'DAI',
        'usds': 'USDS',
    }

    try:
        resp = requests.get(DEFILLAMA_STABLECOIN_PRICES_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        prices = {}

        if isinstance(data, list) and len(data) > 0:
            # Erstes Element (date=0 oder niedrigster date) = aktuelle Preise
            current = data[0]
            price_dict = current.get('prices', {})

            for cg_id, our_symbol in COINGECKO_ID_MAP.items():
                if cg_id in price_dict:
                    prices[our_symbol] = price_dict[cg_id]

            # Fallback: falls USDS nicht als eigener Key, suche nach Varianten
            if 'USDS' not in prices:
                for key, val in price_dict.items():
                    if 'usds' in key.lower() and 'usds' not in key.lower().replace('usds', ''):
                        prices['USDS'] = val
                        break

        elif isinstance(data, dict) and 'prices' in data:
            # Alternativ: direkt ein Dict mit prices
            price_dict = data['prices']
            for cg_id, our_symbol in COINGECKO_ID_MAP.items():
                if cg_id in price_dict:
                    prices[our_symbol] = price_dict[cg_id]

        log(f"Stablecoin Prices: {len(prices)} Coins → "
            + ", ".join(f"{k}=${v:.4f}" for k, v in prices.items()))
        return prices
    except Exception as e:
        log(f"Stablecoin Prices ERR: {e} → Fallback: alle OK")
        return {}


def check_depeg(prices):
    """Depeg Kill Switch — prüfe alle Stablecoins.
    Returns dict: {'USDC': {'status': 'OK'|'WARNING'|'KILL', 'price': float, 'deviation': float}, ...}
    """
    results = {}
    for coin in YIELD_STABLECOIN_SAFETY:
        price = prices.get(coin, 1.0)
        deviation = abs(price - 1.0)

        if deviation >= YIELD_DEPEG['kill_threshold']:
            status = 'KILL'
        elif deviation >= YIELD_DEPEG['warning_threshold']:
            status = 'WARNING'
        else:
            status = 'OK'

        results[coin] = {
            'status': status,
            'price': round(price, 6),
            'deviation': round(deviation, 6),
        }
        if status != 'OK':
            log(f"  ⚠️  {coin}: ${price:.4f} ({deviation*100:.2f}%) → {status}")

    return results


# ═══════════════════════════════════════════════════════
# SCHRITT 2: REGIME BESTIMMEN (Spec §5.1)
# ═══════════════════════════════════════════════════════

def get_regime(ensemble_value):
    """Ensemble → Regime Name."""
    for regime, bounds in YIELD_ENSEMBLE_REGIMES.items():
        if bounds['min'] <= ensemble_value <= bounds['max']:
            return regime
    return 'BEAR'  # Fallback


def get_tier_weights_yield(regime):
    """Regime → T0/T1/T2 Gewichte."""
    return YIELD_REGIME_WEIGHTS.get(regime, YIELD_REGIME_WEIGHTS['BEAR'])


# ═══════════════════════════════════════════════════════
# SCHRITT 3: KAPITAL-TIER (Spec §5.3)
# ═══════════════════════════════════════════════════════

def get_capital_tier(cash_eur):
    """Cash-Betrag → Kapital-Tier."""
    if cash_eur < YIELD_CAPITAL_TIERS['MICRO']['max_cash_eur']:
        return 'MICRO'
    elif cash_eur < YIELD_CAPITAL_TIERS['STANDARD']['max_cash_eur']:
        return 'STANDARD'
    else:
        return 'LARGE'


def apply_capital_overrides(tier_weights, capital_tier):
    """Kapital-abhängige Overrides (Spec §5.3)."""
    w = dict(tier_weights)
    if capital_tier == 'MICRO':
        # T2 = 0%, proportional auf T0 und T1 verteilen
        if w['T2'] > 0:
            t2_share = w['T2']
            t0_t1_total = w['T0'] + w['T1']
            if t0_t1_total > 0:
                w['T0'] = round(w['T0'] + t2_share * (w['T0'] / t0_t1_total), 4)
                w['T1'] = round(w['T1'] + t2_share * (w['T1'] / t0_t1_total), 4)
            else:
                w['T0'] = round(w['T0'] + t2_share, 4)
            w['T2'] = 0.0
    return w


# ═══════════════════════════════════════════════════════
# SCHRITT 4: T1 — T-BILLS BEWERTEN (Spec §3.2)
# ═══════════════════════════════════════════════════════

def score_t1_products(pools_data):
    """Bewerte T1 Produkte mit Live-APY aus DeFiLlama."""
    results = {}
    for product, info in YIELD_T1_PRODUCTS.items():
        slug = info['defillama_slug']
        live_apy = None

        # Suche passenden Pool in DeFiLlama Daten
        for pool in pools_data:
            if pool.get('project', '') == slug:
                sym = (pool.get('symbol') or '').upper()
                # Für sDAI/USDS: spark-savings Pools matchen
                if product == 'sDAI' and ('SDAI' in sym or 'USDS' in sym or 'DAI' in sym):
                    apy = pool.get('apyBase') or pool.get('apy', 0)
                    if apy and apy > 0:
                        live_apy = apy
                        break
                # Für USDY: ondo-yield-assets
                elif product == 'USDY' and ('USDY' in sym or 'USD' in sym):
                    apy = pool.get('apyBase') or pool.get('apy', 0)
                    if apy and apy > 0:
                        live_apy = apy
                        break
                # Für USDM: mountain-protocol
                elif product == 'USDM' and ('USDM' in sym or 'USD' in sym):
                    apy = pool.get('apyBase') or pool.get('apy', 0)
                    if apy and apy > 0:
                        live_apy = apy
                        break

        apy_used = live_apy if live_apy is not None else info['static_apy']
        apy_source = 'LIVE' if live_apy is not None else 'STATIC'

        results[product] = {
            'apy': round(apy_used, 2),
            'apy_source': apy_source,
            'issuer': info['issuer'],
            'chain': info['chain'],
            'redemption_days': info['redemption_days'],
            'min_usd': info['min_usd'],
        }
        log(f"  T1 {product}: {apy_used:.2f}% ({apy_source}) — {info['issuer']}")

    return results


def allocate_t1(t1_amount_eur, t1_products):
    """Allokation innerhalb T1 (Spec §3.2)."""
    if t1_amount_eur <= 0:
        return []

    threshold = YIELD_T1_SPLIT['small_threshold']
    if t1_amount_eur <= threshold:
        split = YIELD_T1_SPLIT['small']
    else:
        split = YIELD_T1_SPLIT['large']

    positions = []
    for product, weight in split.items():
        if product in t1_products:
            info = t1_products[product]
            amount = round(t1_amount_eur * weight, 2)
            positions.append({
                'product': product,
                'weight': weight,
                'amount_eur': amount,
                'apy': info['apy'],
                'apy_source': info['apy_source'],
                'issuer': info['issuer'],
                'chain': info['chain'],
                'tier': 'T1',
            })

    return positions


# ═══════════════════════════════════════════════════════
# SCHRITT 5: T2 — DEFI LENDING BEWERTEN (Spec §3.3, §7, §9)
# ═══════════════════════════════════════════════════════

def fetch_defi_pools():
    """Fetch alle Pools von DeFiLlama /pools."""
    import requests
    try:
        resp = requests.get(DEFILLAMA_POOLS_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pools = data.get('data', data) if isinstance(data, dict) else data
        log(f"DeFiLlama: {len(pools)} Pools fetched")
        return pools
    except Exception as e:
        log(f"DeFiLlama Pools ERR: {e}")
        return []


def filter_t2_pools(pools, depeg_status, capital_tier):
    """Filter-Pipeline für T2 Pools (Spec §9)."""
    allowed_protocols = set(YIELD_T2_PROTOCOLS.keys())
    allowed_stablecoins = set(YIELD_T2_FILTERS['stablecoins'])
    if capital_tier == 'LARGE':
        allowed_chains = set(YIELD_T2_FILTERS['chains_large'])
    else:
        allowed_chains = set(YIELD_T2_FILTERS['chains_standard'])

    # Gesperrte Coins (Depeg KILL)
    killed_coins = {coin for coin, info in depeg_status.items() if info['status'] == 'KILL'}

    qualified = []
    for pool in pools:
        project = pool.get('project', '')
        symbol = (pool.get('symbol') or '').upper()
        chain = pool.get('chain', '')
        tvl = pool.get('tvlUsd', 0) or 0

        # 1. Protokoll-Filter
        if project not in allowed_protocols:
            continue

        # 2. Single-Asset Stablecoin (kein LP)
        # LP Pools haben "/" oder "-" im Symbol mit 2+ Assets
        if '/' in symbol or '-' in symbol:
            continue

        # 3. Stablecoin-Filter: Symbol muss eines der erlaubten sein
        matched_coin = None
        for coin in allowed_stablecoins:
            if coin in symbol:
                matched_coin = coin
                break
        if matched_coin is None:
            continue

        # 4. Depeg KILL Check
        if matched_coin in killed_coins:
            continue

        # 5. Chain-Filter
        if chain not in allowed_chains:
            continue

        # 6. TVL-Filter
        if tvl < YIELD_T2_FILTERS['min_tvl_usd']:
            continue

        # 7. APY-Filter + Plausibilitätscheck (Spec §7.2)
        apy_base = pool.get('apyBase') or 0
        apy_spot = pool.get('apy') or 0
        apy_mean_30d = pool.get('apyMean30d') or 0

        if apy_base and apy_base > 0:
            effective_apy = apy_base
        elif apy_spot > 0 and apy_mean_30d > 0 and apy_spot < apy_mean_30d * 0.5:
            effective_apy = apy_spot
        elif apy_mean_30d > 0:
            effective_apy = apy_mean_30d
        elif apy_spot > 0:
            effective_apy = apy_spot
        else:
            continue

        if effective_apy < YIELD_T2_FILTERS['min_apy'] or effective_apy > YIELD_T2_FILTERS['max_apy']:
            continue

        # 8. Risk-Adjusted APY (Spec §7.1)
        chain_premium = YIELD_CHAIN_RISK.get(chain, 0.01)  # Default 1% für unbekannte
        sc_loss = YIELD_T2_PROTOCOLS[project]['expected_loss']
        risk_adj_apy = effective_apy - (chain_premium * 100) - (sc_loss * 100)

        # 9. Depeg WARNING: Score-Penalty
        coin_depeg = depeg_status.get(matched_coin, {})
        safety_score = YIELD_STABLECOIN_SAFETY.get(matched_coin, {}).get('score', 5)
        if coin_depeg.get('status') == 'WARNING':
            safety_score = max(0, safety_score - YIELD_DEPEG['warning_score_penalty'])

        qualified.append({
            'pool_id': pool.get('pool', ''),
            'project': project,
            'symbol': symbol,
            'coin': matched_coin,
            'chain': chain,
            'tvl_usd': tvl,
            'apy_base': round(apy_base, 2),
            'apy_spot': round(apy_spot, 2),
            'apy_mean_30d': round(apy_mean_30d, 2),
            'effective_apy': round(effective_apy, 2),
            'risk_adj_apy': round(risk_adj_apy, 2),
            'chain_premium': chain_premium,
            'sc_expected_loss': sc_loss,
            'safety_score': safety_score,
            'protocol_safety': YIELD_T2_PROTOCOLS[project]['safety'],
        })

    # Sortieren nach risk_adj_apy (absteigend)
    qualified.sort(key=lambda x: x['risk_adj_apy'], reverse=True)
    log(f"T2 Filter: {len(qualified)} qualifizierte Pools")
    return qualified


def select_t2_pools(qualified_pools, t2_amount_eur, depeg_status):
    """Greedy Selection mit Diversifikation (Spec §8, §9)."""
    if not qualified_pools or t2_amount_eur <= 0:
        return []

    # Pool-Anzahl bestimmen
    if t2_amount_eur < YIELD_T2_POOL_COUNT['small_threshold']:
        max_pools = YIELD_T2_POOL_COUNT['small']
    elif t2_amount_eur < YIELD_T2_POOL_COUNT['medium_threshold']:
        max_pools = YIELD_T2_POOL_COUNT['medium']
    else:
        max_pools = YIELD_T2_POOL_COUNT['large']

    max_issuer_pct = YIELD_T2_FILTERS['max_issuer_pct']
    max_protocol_pct = YIELD_T2_FILTERS['max_protocol_pct']

    selected = []
    issuer_exposure = {}   # Issuer → Anteil
    protocol_exposure = {} # Protocol-Familie → Anteil

    for pool in qualified_pools:
        if len(selected) >= max_pools:
            break

        coin = pool['coin']
        project = pool['project']

        # Issuer-Gruppe bestimmen
        issuer = YIELD_STABLECOIN_SAFETY.get(coin, {}).get('issuer', 'Unknown')

        # Depeg WARNING → niedrigeres Max-Gewicht
        coin_depeg = depeg_status.get(coin, {})
        effective_max_issuer = max_issuer_pct
        if coin_depeg.get('status') == 'WARNING':
            effective_max_issuer = min(max_issuer_pct, YIELD_DEPEG['warning_max_weight'])

        # Issuer-Check
        current_issuer = issuer_exposure.get(issuer, 0.0)
        if current_issuer >= effective_max_issuer:
            continue

        # Protocol-Familie bestimmen (aave-v2/v3 = eine Familie)
        proto_family = project.split('-')[0] if '-' in project else project
        current_proto = protocol_exposure.get(proto_family, 0.0)
        if current_proto >= max_protocol_pct:
            continue

        # Pool auswählen — Gewicht = gleichverteilt
        weight = 1.0 / max_pools
        # Clamp by remaining issuer/protocol budget
        weight = min(weight, effective_max_issuer - current_issuer)
        weight = min(weight, max_protocol_pct - current_proto)

        if weight < 0.05:  # Minimum 5% pro Pool
            continue

        selected.append({
            'pool_id': pool['pool_id'],
            'project': pool['project'],
            'symbol': pool['symbol'],
            'coin': pool['coin'],
            'chain': pool['chain'],
            'tvl_usd': pool['tvl_usd'],
            'effective_apy': pool['effective_apy'],
            'risk_adj_apy': pool['risk_adj_apy'],
            'weight': round(weight, 4),
            'amount_eur': round(t2_amount_eur * weight, 2),
            'tier': 'T2',
        })

        issuer_exposure[issuer] = current_issuer + weight
        protocol_exposure[proto_family] = current_proto + weight

    # Gewichte normalisieren falls < 1.0
    total_weight = sum(p['weight'] for p in selected)
    if selected and total_weight < 0.99:
        for p in selected:
            p['weight'] = round(p['weight'] / total_weight, 4)
            p['amount_eur'] = round(t2_amount_eur * p['weight'], 2)

    return selected


# ═══════════════════════════════════════════════════════
# SCHRITT 6: T0 ALLOKATION (Spec §3.1)
# ═══════════════════════════════════════════════════════

def allocate_t0(t0_amount_eur, depeg_status):
    """Allokation innerhalb T0 (Spec §3.1)."""
    if t0_amount_eur <= 0:
        return []

    # Depeg Fallback-Kaskade
    usdc_killed = depeg_status.get('USDC', {}).get('status') == 'KILL'
    usdt_killed = depeg_status.get('USDT', {}).get('status') == 'KILL'

    if usdc_killed and usdt_killed:
        return [{'coin': 'DAI', 'weight': 1.0, 'amount_eur': t0_amount_eur, 'tier': 'T0', 'apy': 0.0}]
    elif usdc_killed:
        return [{'coin': 'USDT', 'weight': 1.0, 'amount_eur': t0_amount_eur, 'tier': 'T0', 'apy': 0.0}]
    elif usdt_killed:
        return [{'coin': 'USDC', 'weight': 1.0, 'amount_eur': t0_amount_eur, 'tier': 'T0', 'apy': 0.0}]

    # Normal Split
    threshold = YIELD_T0_SPLIT['small_threshold']
    if t0_amount_eur <= threshold:
        split = YIELD_T0_SPLIT['small']
    else:
        split = YIELD_T0_SPLIT['large']

    positions = []
    for coin, weight in split.items():
        positions.append({
            'coin': coin,
            'weight': weight,
            'amount_eur': round(t0_amount_eur * weight, 2),
            'tier': 'T0',
            'apy': 0.0,
        })

    return positions


# ═══════════════════════════════════════════════════════
# GEWICHTETER GESAMT-APY (Spec §7.4)
# ═══════════════════════════════════════════════════════

def calc_weighted_apy(tier_weights, t1_positions, t2_positions):
    """Berechne gewichteten Gesamt-APY."""
    # T0 APY = 0
    t0_contrib = 0.0

    # T1 gewichteter APY
    t1_apy = 0.0
    if t1_positions:
        t1_apy = sum(p['apy'] * p['weight'] for p in t1_positions)
    t1_contrib = tier_weights['T1'] * t1_apy

    # T2 gewichteter APY
    t2_apy = 0.0
    if t2_positions:
        t2_apy = sum(p['risk_adj_apy'] * p['weight'] for p in t2_positions)
    t2_contrib = tier_weights['T2'] * t2_apy

    weighted = t0_contrib + t1_contrib + t2_contrib
    return round(weighted, 2), round(t1_apy, 2), round(t2_apy, 2)


# ═══════════════════════════════════════════════════════
# TELEGRAM DEPEG ALERT (Spec §6.3)
# ═══════════════════════════════════════════════════════

def send_depeg_telegram(depeg_status):
    """Sende Telegram Alert bei Depeg WARNING oder KILL."""
    import requests

    alerts = [v for v in depeg_status.values() if v['status'] != 'OK']
    if not alerts:
        return

    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not bot_token or not chat_id:
        log("Telegram: Credentials nicht gesetzt → übersprungen")
        return

    lines = []
    has_kill = any(depeg_status[c]['status'] == 'KILL' for c in depeg_status)

    if has_kill:
        lines.append("🛑 *STABLECOIN DEPEG — KILL SWITCH*")
    else:
        lines.append("⚠️ *STABLECOIN DEPEG ALERT*")

    lines.append(f"_{NOW.strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")

    for coin, info in depeg_status.items():
        if info['status'] == 'KILL':
            lines.append(f"🛑 {coin}: ${info['price']:.4f} ({info['deviation']*100:.1f}%) — GESPERRT")
        elif info['status'] == 'WARNING':
            lines.append(f"⚠️ {coin}: ${info['price']:.4f} ({info['deviation']*100:.1f}%) — WARNING")

    text = "\n".join(lines)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, json={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}, timeout=15)
        log(f"Telegram Depeg Alert: {'✅' if resp.status_code == 200 else f'ERR {resp.status_code}'}")
    except Exception as e:
        log(f"Telegram ERR: {e}")


# ═══════════════════════════════════════════════════════
# JSON OUTPUT — crypto_yield.json
# ═══════════════════════════════════════════════════════

def write_yield_json(inputs, depeg_status, regime, tier_weights, capital_tier,
                     t0_positions, t1_positions, t1_products, t2_positions,
                     t2_qualified_count, weighted_apy, t1_apy, t2_apy):
    """Schreibe crypto_yield.json (für Frontend + Git)."""

    # T3 Display
    if regime == 'BULL':
        t3_apy = YIELD_T3_DISPLAY['bull_apy_range']
    elif regime in ('MODERATE', 'CAUTIOUS'):
        t3_apy = YIELD_T3_DISPLAY['neutral_apy_range']
    else:
        t3_apy = YIELD_T3_DISPLAY['bear_apy_range']

    cash_eur = inputs['cash_eur']
    annual_yield = round(cash_eur * weighted_apy / 100, 2) if cash_eur > 0 else 0

    output = {
        'version': VERSION,
        'timestamp': NOW.isoformat(),
        'date': NOW.strftime('%Y-%m-%d'),

        'inputs': {
            'ensemble': inputs['ensemble'],
            'cash_pct': inputs['cash_pct'],
            'cash_eur': inputs['cash_eur'],
            'total_capital_eur': inputs['total_capital'],
        },

        'depeg_status': depeg_status,

        'regime': regime,
        'capital_tier': capital_tier,
        'tier_weights': tier_weights,

        'recommendations': {
            'T0': t0_positions,
            'T1': t1_positions,
            'T2': t2_positions,
        },

        't1_products': t1_products,
        't2_qualified_pools': t2_qualified_count,

        'basis_trade_info': {
            'typical_apy': t3_apy,
            'note': YIELD_T3_DISPLAY['note'],
        },

        'apy': {
            'weighted_total': weighted_apy,
            't1_weighted': t1_apy,
            't2_weighted': t2_apy,
            'annual_yield_eur': annual_yield,
        },

        'rebalancing': {
            'last_date': NOW.strftime('%Y-%m-%d'),
            'next_review': None,  # Wird beim nächsten Run berechnet
            'triggers': [],
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, 'crypto_yield.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"JSON: {p}")
    return output


# ═══════════════════════════════════════════════════════
# SHEET WRITE — CALC_Yield Tab (Spec §12)
# ═══════════════════════════════════════════════════════

def write_calc_yield_sheet(gc, inputs, regime, tier_weights, capital_tier,
                           weighted_apy, annual_yield, depeg_status,
                           t2_positions):
    """Schreibe 1 Zeile in CALC_Yield Tab (20 Spalten)."""
    if gc is None or CRYPTO_SHEET_ID is None:
        log("Sheet Write: Kein GCP Auth oder Sheet ID → übersprungen")
        return

    tab_name = CRYPTO_TABS.get('calc_yield', 'CALC_Yield')

    depeg_kills = ','.join(c for c, v in depeg_status.items() if v['status'] == 'KILL') or 'NONE'
    depeg_warnings = ','.join(c for c, v in depeg_status.items() if v['status'] == 'WARNING') or 'NONE'

    t2_best_pool = t2_positions[0]['project'] + ' ' + t2_positions[0]['coin'] if t2_positions else ''
    t2_best_apy = t2_positions[0]['risk_adj_apy'] if t2_positions else 0

    row = [
        NOW.strftime('%Y-%m-%d'),           # Date
        inputs['ensemble'],                  # Ensemble
        regime,                              # Regime
        inputs['cash_pct'],                  # Cash_Pct
        inputs['cash_eur'],                  # Cash_EUR
        capital_tier,                        # Capital_Tier
        weighted_apy,                        # Weighted_APY
        annual_yield,                        # Annual_Yield_EUR
        tier_weights['T0'],                  # T0_Weight
        tier_weights['T1'],                  # T1_Weight
        tier_weights['T2'],                  # T2_Weight
        round(inputs['cash_eur'] * tier_weights['T0'], 2),  # T0_Amount
        round(inputs['cash_eur'] * tier_weights['T1'], 2),  # T1_Amount
        round(inputs['cash_eur'] * tier_weights['T2'], 2),  # T2_Amount
        depeg_kills,                         # Depeg_Kills
        depeg_warnings,                      # Depeg_Warnings
        len(t2_positions),                   # T2_Pools_Count
        t2_best_pool,                        # T2_Best_Pool
        t2_best_apy,                         # T2_Best_APY
        '',                                  # Rebalance_Trigger (leer beim ersten Run)
    ]

    try:
        sh = gc.open_by_key(CRYPTO_SHEET_ID)
        ws = sh.worksheet(tab_name)
        ws.append_row([str(v) if v is not None else '' for v in row], value_input_option='USER_ENTERED')
        log(f"Sheet: CALC_Yield {len(row)} Werte ✅")
    except Exception as e:
        log(f"Sheet Write ERR: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# LOAD CRYPTO STATE (für Ensemble + Cash)
# ═══════════════════════════════════════════════════════

def load_crypto_state():
    """Lade crypto_state.json."""
    p = os.path.join(DATA_DIR, 'crypto_state.json')
    if not os.path.exists(p):
        log(f"WARNUNG: {p} nicht gefunden")
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════
# MAIN — YIELD ROUTER
# ═══════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description='Crypto Cash Management Advisor V1.0')
    pa.add_argument('--skip-write', action='store_true', help='Kein Sheet Write')
    pa.add_argument('--skip-telegram', action='store_true', help='Kein Telegram Alert')
    pa.add_argument('--capital', type=float, default=None, help='Manuelles Gesamtkapital (EUR)')
    pa.add_argument('--ensemble', type=float, default=None, help='Manueller Ensemble-Wert (0.0-1.0)')
    args = pa.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"CRYPTO CASH MANAGEMENT ADVISOR — {VERSION}")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Spec: YIELD_ROUTER_SPEC_TEIL1+2")
    print(f"  Flags: skip-write={args.skip_write} skip-telegram={args.skip_telegram}")
    if args.capital:
        print(f"  Manual Capital: €{args.capital:,.0f}")
    if args.ensemble is not None:
        print(f"  Manual Ensemble: {args.ensemble:.2f}")
    print("=" * 70)

    # ─── Inputs laden ───
    state = load_crypto_state()

    if args.ensemble is not None:
        ensemble = args.ensemble
    elif state:
        ensemble = state.get('ensemble', {}).get('value', 0.25)
    else:
        ensemble = 0.25
        log("Kein crypto_state.json → Ensemble Fallback 0.25")

    if state:
        cash_pct = state.get('allocation', {}).get('cash', 0.75)
    else:
        cash_pct = 0.75
        log("Kein crypto_state.json → Cash Fallback 75%")

    if args.capital is not None:
        total_capital = args.capital
    else:
        total_capital = 10000  # Default
        log(f"Kein --capital → Default €{total_capital:,.0f}")

    cash_eur = round(total_capital * cash_pct, 2)

    inputs = {
        'ensemble': ensemble,
        'cash_pct': round(cash_pct, 4),
        'cash_eur': cash_eur,
        'total_capital': total_capital,
    }

    log(f"Ensemble: {ensemble:.2f}, Cash: {cash_pct:.0%} = €{cash_eur:,.0f} (Kapital: €{total_capital:,.0f})")

    # ─── Schritt 1: Depeg Check ───
    print(f"\n{'='*70}")
    print("SCHRITT 1: DEPEG KILL SWITCH")
    print(f"{'='*70}")

    stablecoin_prices = fetch_stablecoin_prices()
    depeg_status = check_depeg(stablecoin_prices)

    has_depeg = any(v['status'] != 'OK' for v in depeg_status.values())
    if has_depeg and not args.skip_telegram:
        send_depeg_telegram(depeg_status)

    depeg_kills = [c for c, v in depeg_status.items() if v['status'] == 'KILL']
    depeg_warnings = [c for c, v in depeg_status.items() if v['status'] == 'WARNING']
    log(f"Kills: {depeg_kills or 'NONE'}, Warnings: {depeg_warnings or 'NONE'}")

    # ─── Schritt 2: Regime ───
    print(f"\n{'='*70}")
    print("SCHRITT 2: REGIME + TIER-GEWICHTE")
    print(f"{'='*70}")

    regime = get_regime(ensemble)
    tier_weights = get_tier_weights_yield(regime)

    # ─── Schritt 3: Kapital-Tier ───
    capital_tier = get_capital_tier(cash_eur)
    tier_weights = apply_capital_overrides(tier_weights, capital_tier)

    log(f"Regime: {regime}, Capital-Tier: {capital_tier}")
    log(f"Tier-Gewichte: T0={tier_weights['T0']:.0%} T1={tier_weights['T1']:.0%} T2={tier_weights['T2']:.0%}")

    t0_amount = round(cash_eur * tier_weights['T0'], 2)
    t1_amount = round(cash_eur * tier_weights['T1'], 2)
    t2_amount = round(cash_eur * tier_weights['T2'], 2)
    log(f"Beträge: T0=€{t0_amount:,.0f} T1=€{t1_amount:,.0f} T2=€{t2_amount:,.0f}")

    # ─── Schritt 4: DeFiLlama Pools fetchen ───
    print(f"\n{'='*70}")
    print("SCHRITT 3: DEFI POOLS + T1/T2 SCORING")
    print(f"{'='*70}")

    pools = fetch_defi_pools()

    # ─── Schritt 5: T1 Scoring ───
    t1_products = score_t1_products(pools)
    t1_positions = allocate_t1(t1_amount, t1_products)

    # ─── Schritt 6: T2 Scoring ───
    t2_qualified = filter_t2_pools(pools, depeg_status, capital_tier)
    t2_positions = select_t2_pools(t2_qualified, t2_amount, depeg_status)

    for p in t2_positions:
        log(f"  T2 → {p['project']} {p['coin']} {p['chain']}: "
            f"{p['risk_adj_apy']:.2f}% risk-adj, €{p['amount_eur']:,.0f}")

    # ─── Schritt 7: T0 Allokation ───
    t0_positions = allocate_t0(t0_amount, depeg_status)

    # ─── Schritt 8: Gesamt-APY ───
    print(f"\n{'='*70}")
    print("SCHRITT 4: ERGEBNIS")
    print(f"{'='*70}")

    weighted_apy, t1_apy, t2_apy = calc_weighted_apy(tier_weights, t1_positions, t2_positions)
    annual_yield = round(cash_eur * weighted_apy / 100, 2)

    log(f"Gewichteter APY: {weighted_apy:.2f}% (T1: {t1_apy:.2f}%, T2: {t2_apy:.2f}%)")
    log(f"Geschätzte Jahresrendite: €{annual_yield:,.0f} auf €{cash_eur:,.0f} Cash")

    # ─── Schritt 9: Output ───
    print(f"\n{'='*70}")
    print("SCHRITT 5: OUTPUT")
    print(f"{'='*70}")

    yield_data = write_yield_json(
        inputs, depeg_status, regime, tier_weights, capital_tier,
        t0_positions, t1_positions, t1_products, t2_positions,
        len(t2_qualified), weighted_apy, t1_apy, t2_apy,
    )

    # Sheet Write
    gc = None
    if not args.skip_write:
        gc = get_gspread_client()
        if gc:
            write_calc_yield_sheet(gc, inputs, regime, tier_weights, capital_tier,
                                   weighted_apy, annual_yield, depeg_status, t2_positions)
        else:
            log("Kein GCP Auth → Sheet Write übersprungen")
    else:
        log("Sheet Write übersprungen (--skip-write)")

    # ─── Zusammenfassung ───
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"YIELD ROUTER — FERTIG ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Regime:       {regime} (Ensemble {ensemble:.2f})")
    print(f"  Capital-Tier: {capital_tier} (Cash €{cash_eur:,.0f})")
    print(f"  Tier-Gewichte: T0={tier_weights['T0']:.0%} T1={tier_weights['T1']:.0%} T2={tier_weights['T2']:.0%}")
    print(f"  Positionen:   {len(t0_positions)} T0 + {len(t1_positions)} T1 + {len(t2_positions)} T2")
    print(f"  Depeg:        Kills={depeg_kills or 'NONE'} Warnings={depeg_warnings or 'NONE'}")
    print(f"  APY:          {weighted_apy:.2f}% gewichtet → €{annual_yield:,.0f}/Jahr")

    if t0_positions:
        print(f"\n  T0 (Liquid Cash):")
        for p in t0_positions:
            print(f"    {p['coin']}: {p['weight']:.0%} = €{p['amount_eur']:,.0f}")

    if t1_positions:
        print(f"\n  T1 (T-Bills):")
        for p in t1_positions:
            print(f"    {p['product']}: {p['weight']:.0%} = €{p['amount_eur']:,.0f} @ {p['apy']:.2f}%")

    if t2_positions:
        print(f"\n  T2 (DeFi Lending):")
        for p in t2_positions:
            print(f"    {p['project']} {p['coin']} ({p['chain']}): "
                  f"{p['weight']:.0%} = €{p['amount_eur']:,.0f} @ {p['risk_adj_apy']:.2f}%")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
