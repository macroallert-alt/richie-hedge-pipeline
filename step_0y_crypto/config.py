#!/usr/bin/env python3
"""
config.py — Crypto Circle Konfiguration V2.1
==============================================
Baldur Creek Capital | Circle 17 (Crypto Hub)

V2.1 — CALC_Yield Tab für Yield Router
  Änderungen V2.0 → V2.1:
    - NEU: CRYPTO_TABS['calc_yield'] = 'CALC_Yield'

V2.0 — V8+Warn Produktionssystem
  Quellen der Wahrheit:
    - CRYPTO_CIRCLE_V8_WARN_PRODUKTIONSSPEZIFIKATION.md (Allokationslogik)
    - CRYPTO_CIRCLE_SPEC_V2_TEIL1-4.md + ADDENDUM (Datenquellen, Frontend, Intelligence)
    - YIELD_ROUTER_SPEC_TEIL1.md + TEIL2.md (Cash Management)
    - V149_SYSTEMSTATUSANALYSE.md (Pipeline-Status)

  Änderungen V1.0 → V2.0:
    - NEU: V8_ENSEMBLE, V8_BOTTOM_BONUS, V8_TRICKLE_DOWN, V8_PHASE4_WARNING
    - VERWORFEN: Cluster-Confirmation (CYCLE_STATES, BASE_ALLOCATION, DD_THRESHOLDS,
      TRAILING_STOP, TRANSITION_SPEED, PEAK_SIGNALS, BOTTOM_SIGNALS, CLASS3,
      DISTRIBUTION State, ALTSEASON Logik, LIQUIDITY_SCORE, ALLOCATION_CHAIN,
      CRASH_TAXONOMY, DCA, REENTRY, KILL_SWITCHES automatisch,
      CORRELATION/FUNDING/OI als Modifier, VOL_TARGETING, V16_MACRO_MODIFIER)
    - BEHALTEN: Sheet IDs, API Endpoints (data_collector), COINGECKO_IDS,
      HALVINGS, RAINBOW, PI_CYCLE (Display), V16_STATE_NAMES (Display),
      GRACEFUL_DEGRADATION, SCHEDULE, DISPLAY_INDICATORS
    - RÜCKWÄRTSKOMPATIBEL: Alle Symbole die data_collector V1.3.1 importiert
      bleiben vorhanden (V16_MACRO_MODIFIER, V16_INSTABILITY_*, REAL_VOL_WINDOW)
"""

# ═══════════════════════════════════════════════════════
# SHEET IDs
# ═══════════════════════════════════════════════════════

# Crypto Circle eigenes Sheet
CRYPTO_SHEET_ID = "1WTuxTpsL7mqMIaj6w5H1PY89GlytXf4J4qifLNSNZlI"

# V16 Sheet (READ ONLY — Preise, Macro State, Howell)
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"

# V16 Sheet Tabs die wir lesen
V16_TABS = {
    'prices':       'DATA_Prices',       # BTC, ETH, GLD, SPY — europäisches Zahlenformat
    'macro_state':  'CALC_Macro_State',   # macro_state_num
    'k16_k17':      'DATA_K16_K17',       # Liq_Dir_Confirmed, Vote_Sum_Magnitude
}

# Crypto Sheet Tabs die wir schreiben
CRYPTO_TABS = {
    'data_raw':       'DATA_Raw',
    'ind_cycle':      'IND_Cycle',
    'ind_market':     'IND_Market',
    'calc_cycle':     'CALC_Cycle',
    'calc_risk':      'CALC_Risk',
    'calc_alloc':     'CALC_Allocation',
    'calc_yield':     'CALC_Yield',
    'hist_daily':     'HIST_Daily_Risk',
    'track_override': 'TRACK_Overrides',
    'track_holdings': 'TRACK_Holdings',
    'config':         'CONFIG',
}


# ═══════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════

API_ENDPOINTS = {
    # ─── BGeometrics GitHub Raw (gratis, kein Key, kein Rate Limit) ───
    # V146 Entscheidung 3.135: GitHub Raw statt API
    'bg_github_base':       'https://raw.githubusercontent.com/BGeometrics/bgeometrics.github.io/master/files',

    # ─── Binance Public Futures API (gratis, kein Key — 451 Geo-Block erwartet) ───
    'binance_funding':      'https://fapi.binance.com/fapi/v1/fundingRate',
    'binance_oi':           'https://fapi.binance.com/fapi/v1/openInterest',
    'binance_liquidations': 'https://fapi.binance.com/fapi/v1/forceOrders',

    # ─── CoinGecko (gratis, kein Key) ───
    'coingecko_prices':     'https://api.coingecko.com/api/v3/simple/price',
    'coingecko_global':     'https://api.coingecko.com/api/v3/global',
    'coingecko_markets':    'https://api.coingecko.com/api/v3/coins/markets',
    'coingecko_sol_chart':  'https://api.coingecko.com/api/v3/coins/solana/market_chart',

    # ─── CoinCap (Fallback für SOL Preise) ───
    'coincap_sol':          'https://api.coincap.io/v2/assets/solana/history',

    # ─── CoinMetrics (gratis, historische Preise für Warm-Up) ───
    'coinmetrics_btc':      'https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv',
    'coinmetrics_eth':      'https://raw.githubusercontent.com/coinmetrics/data/master/csv/eth.csv',

    # ─── DeFiLlama (gratis, kein Key) ───
    'defillama_stablecoins': 'https://stablecoins.llama.fi/stablecoins?includePrices=true',
    'defillama_tvl':        'https://api.llama.fi/v2/historicalChainTvl',

    # ─── Sentiment (gratis, kein Key) ───
    'fear_greed':           'https://api.alternative.me/fng/?limit=30&format=json',

    # ─── Blockchain.com (gratis, kein Key) ───
    'blockchain_miner_rev': 'https://api.blockchain.info/charts/miners-revenue?timespan=2years&format=json',
    'blockchain_addresses': 'https://api.blockchain.info/charts/n-unique-addresses?timespan=30days&format=json',

    # ─── FRED (Key vorhanden: FRED_API_KEY) ───
    'fred_m2':              'https://api.stlouisfed.org/fred/series/observations',
}


# ═══════════════════════════════════════════════════════
# ASSET UNIVERSE
# ═══════════════════════════════════════════════════════

COINGECKO_IDS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
}


# ═══════════════════════════════════════════════════════
# HALVING DATEN
# ═══════════════════════════════════════════════════════

HALVINGS = [
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-19',
]
NEXT_HALVING_EST = '2028-04-15'
AVG_CYCLE_DAYS = 1460  # 4 Jahre default


# ═══════════════════════════════════════════════════════
# RAINBOW REGRESSION (PoC verifiziert — Addendum 3.1)
# Display only — kein Einfluss auf V8+Warn Allokation
# ═══════════════════════════════════════════════════════

RAINBOW = {
    'genesis_date':     '2009-01-03',
    'a':                5.9392,       # Slope
    'b':                -40.1555,     # Intercept
    'residual_std':     0.5371,       # StdDev
    'fit_min_price':    1.0,          # Nur BTC > $1 für Fit
    'fit_min_days':     365,          # Mindestens 1 Jahr nach Genesis
    # Band-Mapping: score → band (1-8)
    # score = residual / std
    # band = clip(round(score * 1.75 + 4.5), 1, 8)
}


# ═══════════════════════════════════════════════════════
# PI CYCLE — NUR DISPLAY (Addendum Entscheidung: Option D)
# ═══════════════════════════════════════════════════════

PI_CYCLE = {
    'active_signal':    False,        # Pi beeinflusst KEINE Allokation
    'display_only':     True,         # Wird im Cycles Tab angezeigt
    # Top Indicator
    'top_sma_period':   111,
    'top_ema_period':   350,
    'top_ema_mult':     2.0,
    # Bottom Indicator (Richie's Version)
    'bot_ema_period':   471,
    'bot_ema_mult':     150.0,
    'bot_sma_period':   471,
    # 200-Wochen-MA
    'weekly_200_ma':    1400,         # 200 * 7 = 1400 Tage
}


# ═══════════════════════════════════════════════════════
# V16 MACRO STATE — NUR DISPLAY + DATA_COLLECTOR
# V8+Warn Backtest: V16 Modifier kostet -€2-4M → VERWORFEN
# State-Namen bleiben für Display + data_collector Kompatibilität
# ═══════════════════════════════════════════════════════

V16_STATE_NAMES = {
    1: 'STEADY_GROWTH',
    2: 'FRAGILE_EXPANSION',
    3: 'LATE_EXPANSION',
    4: 'FULL_EXPANSION',
    5: 'REFLATION',
    6: 'NEUTRAL',
    7: 'SOFT_LANDING',
    8: 'STRESS_ELEVATED',
    9: 'CONTRACTION',
    10: 'DEEP_CONTRACTION',
    11: 'FINANCIAL_CRISIS',
    12: 'EARLY_RECOVERY',
}

# RÜCKWÄRTSKOMPATIBEL: data_collector V1.3.1 importiert V16_MACRO_MODIFIER
# In V8+Warn hat der Modifier KEINEN Einfluss auf die Allokation
# Wird nur noch im Sheet als Display-Spalte geschrieben
V16_MACRO_MODIFIER = {
    1:  1.00,   # STEADY_GROWTH
    2:  1.00,   # FRAGILE_EXPANSION
    3:  0.85,   # LATE_EXPANSION
    4:  1.00,   # FULL_EXPANSION
    5:  1.00,   # REFLATION
    6:  0.85,   # NEUTRAL
    7:  0.85,   # SOFT_LANDING
    8:  0.70,   # STRESS_ELEVATED
    9:  0.40,   # CONTRACTION
    10: 0.20,   # DEEP_CONTRACTION
    11: 0.10,   # FINANCIAL_CRISIS
    12: 1.20,   # EARLY_RECOVERY
}

# RÜCKWÄRTSKOMPATIBEL: data_collector V1.3.1 importiert diese
# V16 State-Instabilität — nur noch Display
V16_INSTABILITY_WINDOW_DAYS = 30
V16_INSTABILITY_THRESHOLD = 3    # >= 3 verschiedene States = UNSTABLE


# ═══════════════════════════════════════════════════════
# V8+WARN KOMPONENTE 1: BTC MOMENTUM ENSEMBLE
# Bestimmt WANN investiert wird (Gesamt-Allokation 0-100%)
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 3
# ═══════════════════════════════════════════════════════

V8_ENSEMBLE = {
    # 4 Momentum-Lookbacks (akademischer Standard, Handelstage)
    'lookbacks': {
        '1M':  21,    # 1 Monat
        '3M':  63,    # 1 Quartal
        '6M':  126,   # Halbjahr
        '12M': 252,   # 1 Jahr
    },
    # 1M-Signal Glättung: 5d SMA des 21d-Returns
    # Reduziert Noise von ~357 Wechseln/Jahr auf ~171 (-52%)
    'smooth_1m_window': 5,
    # Gleichgewichtet (Multi-Speed getestet: kein Vorteil, mehr Komplexität)
    'equal_weight': True,
    # Mögliche Ensemble-Werte: 0.00, 0.25, 0.50, 0.75, 1.00
}


# ═══════════════════════════════════════════════════════
# V8+WARN KOMPONENTE 1b: 200-WOCHEN-MA BOTTOM BONUS
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 3.4
# ═══════════════════════════════════════════════════════

V8_BOTTOM_BONUS = {
    'wma_days':         1400,   # 200 × 7 = 1400 Handelstage
    'min_periods':      700,    # Halbe Periode für Initialisierung
    'bonus':            0.50,   # +50pp wenn BTC < 200WMA
    # Blinder Bonus (nicht konditioniert auf 1M ON)
    # Konditioniert: €4.7M vs. blind: €10.0M — blind ist besser
    # Fängt V-Recoveries (COVID, FTX) bevor 1M dreht
}


# ═══════════════════════════════════════════════════════
# V8+WARN KOMPONENTE 2: TRICKLE-DOWN ROTATION
# Bestimmt WOHIN investiert wird (BTC/ETH/SOL Verteilung)
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 4
# ═══════════════════════════════════════════════════════

V8_TRICKLE_DOWN = {
    # BTC Dominance 30d-Veränderung in Prozentpunkten → Phase
    'phase_thresholds': {
        'phase1_above':  2.0,    # BTC.D steigt >+2pp → BTC dominant
        'phase3_below': -2.0,    # BTC.D fällt <-2pp → Altseason aktiv
        'phase4_below': -5.0,    # BTC.D crasht <-5pp → Altseason reif
        # Phase 2 = Default: -2.0 <= change <= +2.0
    },
    # Tier-Gewichte pro Phase (summieren sich immer zu 1.00)
    'phase_weights': {
        1: {'BTC': 0.70, 'ETH': 0.25, 'SOL': 0.05},  # BTC dominant
        2: {'BTC': 0.45, 'ETH': 0.35, 'SOL': 0.20},  # Balanced (Default, 77% der Zeit)
        3: {'BTC': 0.25, 'ETH': 0.35, 'SOL': 0.40},  # Altseason aktiv
        4: {'BTC': 0.25, 'ETH': 0.35, 'SOL': 0.40},  # Altseason reif (mit Warning)
    },
    # BTC.D Lookback für 30d-Veränderung
    'dominance_lookback_days': 30,
    # Fallback wenn BTC.D nicht verfügbar
    'default_phase': 2,
}


# ═══════════════════════════════════════════════════════
# V8+WARN KOMPONENTE 3: PHASE 4 ALTSEASON WARNING
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 5
# ═══════════════════════════════════════════════════════

V8_PHASE4_WARNING = {
    'multiplier': 0.60,   # Gesamt-Allokation × 0.60 bei Phase 4
    # Kostet ~1.5pp CAGR (72.53% vs. 74.07%)
    # Spart 6.27pp MaxDD (-54.73% vs. -61.00%)
    # Verbessert Sharpe um 0.08 (1.66 vs. 1.58)
}


# ═══════════════════════════════════════════════════════
# V8+WARN: NO-ACTION BAND
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 12.3
# ═══════════════════════════════════════════════════════

V8_NO_ACTION_BAND = {
    'total_pp': 10,   # ±10pp Gesamtallokation → KEINE Aktion
    'asset_pp': 5,    # ±5pp pro Asset → KEINE Aktion
}


# ═══════════════════════════════════════════════════════
# V8+WARN: ASYMMETRISCHER AUFBAU (Live-Execution)
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 12.2
# NICHT im Backtest — nur für Live-Execution empfohlen
# ═══════════════════════════════════════════════════════

V8_EXECUTION = {
    'reduce_immediate': True,     # Reduzierung: sofort
    'build_weeks': 3,             # Aufbau: über 3 Wochen
    'build_week1_pct': 0.50,      # Woche 1: 50% des Deltas
    'build_week2_pct': 0.30,      # Woche 2: 30%
    'build_week3_pct': 0.20,      # Woche 3: 20%
}


# ═══════════════════════════════════════════════════════
# V8+WARN: SOL PROXY
# Wenn SOL-Preise nicht verfügbar: ETH × 1.3 als Proxy
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 8.2
# ═══════════════════════════════════════════════════════

V8_SOL_PROXY = {
    'multiplier': 1.3,           # SOL ≈ ETH × 1.3 (Volatilitätsproxy)
    'start_date': '2020-09-01',  # SOL echte Daten erst ab hier verfügbar
}


# ═══════════════════════════════════════════════════════
# V8+WARN: BTC.D PROXY (Korrekturfaktoren nach Jahr)
# Wenn CoinGecko /global nicht verfügbar: BTC MCap / (BTC+ETH MCap) × CF
# Quelle: V8+Warn Produktionsspezifikation Abschnitt 8.3
# ═══════════════════════════════════════════════════════

V8_BTC_D_PROXY = {
    'correction_factors': {
        2010: 0.95, 2011: 0.95, 2012: 0.95, 2013: 0.90, 2014: 0.90,
        2015: 0.85, 2016: 0.80, 2017: 0.60, 2018: 0.55, 2019: 0.65,
        2020: 0.65, 2021: 0.55, 2022: 0.60, 2023: 0.65, 2024: 0.65,
        2025: 0.65, 2026: 0.65,
    },
    'default_cf': 0.65,
    'min_dominance': 20.0,   # BTC.D < 20% = unplausibel
    'max_dominance': 95.0,   # BTC.D > 95% = unplausibel
}


# ═══════════════════════════════════════════════════════
# DISPLAY-ONLY INDIKATOREN (kein Einfluss auf V8+Warn Allokation)
# Werden im CryptoHub Frontend angezeigt
# ═══════════════════════════════════════════════════════

DISPLAY_INDICATORS = [
    # On-Chain (BGeometrics GitHub Raw)
    'mvrv_zscore',
    'nupl',
    'sopr', 'sopr_lth', 'sopr_sth',
    'puell_multiple',
    'reserve_risk',
    'rhodl_ratio',
    'realized_price', 'sth_realized_price', 'lth_realized_price',
    'lth_supply', 'sth_supply',
    'supply_in_profit',
    # Cycle Models
    'pi_cycle_top', 'pi_cycle_bottom',
    'rainbow_chart',
    'halving_phase',
    # Sentiment + Derivate
    'fear_greed',
    'funding_rates',
    'open_interest',
    # Network
    'active_addresses',
    'hashrate',
    # ETF
    'etf_flows',
    # V16 Bridge
    'v16_macro_state',
    'howell_liquidity',
    # Stablecoins
    'stablecoin_supply',
    'defi_tvl',
]


# ═══════════════════════════════════════════════════════
# RÜCKWÄRTSKOMPATIBEL: data_collector V1.3.1 importiert REAL_VOL_WINDOW
# In V8+Warn nicht als Allokations-Input genutzt, nur Display
# ═══════════════════════════════════════════════════════

REAL_VOL_WINDOW = 60   # 60-Tage realized vol (Display)


# ═══════════════════════════════════════════════════════
# GRACEFUL DEGRADATION
# Angepasst auf V8+Warn: Nur BTC Preis ist kritisch
# Alles andere = Display only, kein Einfluss auf Allokation
# ═══════════════════════════════════════════════════════

GRACEFUL_DEGRADATION = {
    # ─── KRITISCH für V8+Warn Allokation ───
    'btc_price':            {'fallback': 'v16_sheet',       'stale_ok_hours': 24,   'critical': True},
    'eth_price':            {'fallback': 'v16_sheet',       'stale_ok_hours': 24,   'critical': True},
    'sol_price':            {'fallback': 'eth_proxy',       'stale_ok_hours': 24,   'critical': False},
    'btc_dominance':        {'fallback': 'mcap_proxy',      'stale_ok_hours': 24,   'critical': False},

    # ─── Display only (kein Einfluss auf Allokation) ───
    'mvrv_zscore':          {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'puell_multiple':       {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'nupl':                 {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'lth_sopr':             {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'sth_sopr':             {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'reserve_risk':         {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'rhodl_ratio':          {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'sth_realized_price':   {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'lth_realized_price':   {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'lth_supply':           {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'sth_supply':           {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'fear_greed':           {'fallback': 'last_value',      'stale_ok_hours': 48,   'critical': False},
    'funding_rates':        {'fallback': 'assume_zero',     'stale_ok_hours': 24,   'critical': False},
    'open_interest':        {'fallback': None,              'stale_ok_hours': 24,   'critical': False},
    'liquidations':         {'fallback': None,              'stale_ok_hours': 24,   'critical': False},
    'stablecoin_supply':    {'fallback': 'last_value',      'stale_ok_hours': 48,   'critical': False},
    'defi_tvl':             {'fallback': 'last_value',      'stale_ok_hours': 336,  'critical': False},
    'etf_flows':            {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'v16_macro_state':      {'fallback': 'last_value',      'stale_ok_hours': 24,   'critical': False},
    'howell_liquidity':     {'fallback': 'last_value',      'stale_ok_hours': 24,   'critical': False},
    'active_addresses':     {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'exchange_reserves':    {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'top_100_coingecko':    {'fallback': None,              'stale_ok_hours': 168,  'critical': False},
}

DEGRADATION_LEVELS = {
    0: 'NOMINAL',       # Alle Quellen OK
    1: 'DEGRADED',      # 1-3 ausgefallen — Allokation läuft normal
    2: 'MINIMAL',       # > 3 ausgefallen — Allokation läuft, Display lückenhaft
    3: 'OFFLINE',       # BTC-Preis nicht verfügbar → Run abbrechen
}
DEGRADATION_THRESHOLD_MINIMAL = 4   # > 3 Ausfälle = MINIMAL


# ═══════════════════════════════════════════════════════
# TIMING + SCHEDULING
# ═══════════════════════════════════════════════════════

SCHEDULE = {
    'weekly_full_run':  '05:00 UTC Sonntag',
    'daily_risk_check': '06:30 UTC täglich',
    'monthly_intel':    '05:30 UTC 1. Sonntag/Monat',
}


# ═══════════════════════════════════════════════════════
# VERSION
# ═══════════════════════════════════════════════════════

CONFIG_VERSION = '2.1'
SPEC_VERSION = 'V8+Warn Produktionsspezifikation + Yield Router Spec'
