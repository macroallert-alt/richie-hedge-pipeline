#!/usr/bin/env python3
"""
config.py — Crypto Circle Konfiguration V1.0
==============================================
Baldur Creek Capital | Circle 17 (Crypto Hub)
Quellen der Wahrheit:
  - CRYPTO_CIRCLE_SPEC_V2_TEIL1-4.md
  - CRYPTO_CIRCLE_SPEC_V2_ADDENDUM.md
  - Session-Entscheidungen (Pi Cycle Option D, CoinGlass Erweiterung)

Alle Schwellenwerte die mit [CALIBRATE] markiert sind werden via
Kalibrierungs-Backtest (PoC V3) endgültig festgelegt. Aktuelle Werte
stammen aus der Spec / Addendum als Startwerte.
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
    'hist_daily':     'HIST_Daily_Risk',
    'track_override': 'TRACK_Overrides',
    'track_holdings': 'TRACK_Holdings',
    'config':         'CONFIG',
}


# ═══════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════

API_ENDPOINTS = {
    # ─── BGeometrics (gratis, kein Key) — On-Chain + Derivate + Macro ───
    'bg_base':              'https://bitcoin-data.com/v1',
    'bg_mvrv':              'https://bitcoin-data.com/v1/mvrv',
    'bg_nupl':              'https://bitcoin-data.com/v1/nupl',
    'bg_sopr':              'https://bitcoin-data.com/v1/sopr',
    'bg_sopr_lth':          'https://bitcoin-data.com/v1/sopr-lth',
    'bg_sopr_sth':          'https://bitcoin-data.com/v1/sopr-sth',
    'bg_puell':             'https://bitcoin-data.com/v1/puell-multiple',
    'bg_reserve_risk':      'https://bitcoin-data.com/v1/reserve-risk',
    'bg_rhodl':             'https://bitcoin-data.com/v1/rhodl',
    'bg_realized_price':    'https://bitcoin-data.com/v1/realized-price',
    'bg_sth_rp':            'https://bitcoin-data.com/v1/sth-realized-price',
    'bg_lth_rp':            'https://bitcoin-data.com/v1/lth-realized-price',
    'bg_supply_lth':        'https://bitcoin-data.com/v1/supply-lth',
    'bg_supply_sth':        'https://bitcoin-data.com/v1/supply-sth',
    'bg_supply_profit':     'https://bitcoin-data.com/v1/supply-in-profit',
    'bg_active_addr':       'https://bitcoin-data.com/v1/active-addresses',
    'bg_hashrate':          'https://bitcoin-data.com/v1/hashrate',
    'bg_stablecoin':        'https://bitcoin-data.com/v1/stablecoin-supply',
    'bg_dominance':         'https://bitcoin-data.com/v1/bitcoin-dominance',
    'bg_funding':           'https://bitcoin-data.com/v1/funding-rate',
    'bg_oi':                'https://bitcoin-data.com/v1/open-interest-futures',
    'bg_etf':               'https://bitcoin-data.com/v1/etf',

    # ─── Binance Public Futures API (gratis, kein Key) ───
    'binance_funding':      'https://fapi.binance.com/fapi/v1/fundingRate',
    'binance_oi':           'https://fapi.binance.com/fapi/v1/openInterest',
    'binance_liquidations': 'https://fapi.binance.com/fapi/v1/forceOrders',

    # ─── CoinGecko (gratis, kein Key) ───
    'coingecko_prices':     'https://api.coingecko.com/api/v3/simple/price',
    'coingecko_global':     'https://api.coingecko.com/api/v3/global',
    'coingecko_markets':    'https://api.coingecko.com/api/v3/coins/markets',

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

    # ─── Fallbacks (Scraping) ───
    'sosovalue_etf':        'https://sosovalue.com/assets/etf/us-btc-spot',
    'farside_etf':          'https://farside.co.uk/btc/',
    'blockchaincenter_alt': 'https://www.blockchaincenter.net/altcoin-season-index/',
}


# ═══════════════════════════════════════════════════════
# ASSET UNIVERSE
# ═══════════════════════════════════════════════════════

TIER_0 = ['BTC']
TIER_1 = ['ETH', 'SOL']
TIER_2_MAX_POSITIONS = 5
TIER_2_MAX_PER_POSITION = 0.05   # 5% pro Einzelposition
TIER_2_MIN_MCAP = 500_000_000    # $500M Market Cap minimum
TIER_2_MIN_LIQUIDITY_MULT = 20   # 24h Volume >= 20x geplante Position

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
# V16 MACRO STATE MODIFIER (Spec TEIL1, Abschnitt 1.3)
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

# V16 State-Instabilität (Addendum 4.4)
V16_INSTABILITY_WINDOW_DAYS = 30
V16_INSTABILITY_THRESHOLD = 3    # >= 3 verschiedene States = UNSTABLE


# ═══════════════════════════════════════════════════════
# CYCLE STATES + BASIS-ALLOKATION (Addendum Abschnitt 9)
# ═══════════════════════════════════════════════════════

CYCLE_STATES = [
    'KAPITULATION',
    'LATE_BEAR',
    'AKKUMULATION',
    'TRANSITION',
    'EARLY_BULL',
    'MID_BULL',
    'DISTRIBUTION',        # NEU — Addendum Abschnitt 5
    'MID_BULL_CAUTION',
    'EUPHORIE',
    'BLOW_OFF_TOP',
]

BASE_ALLOCATION = {
    'KAPITULATION':     0.95,   # DCA
    'LATE_BEAR':        0.30,
    'AKKUMULATION':     0.85,
    'TRANSITION':       0.70,
    'EARLY_BULL':       0.95,
    'MID_BULL':         1.00,
    'DISTRIBUTION':     0.50,   # NEU — Addendum
    'MID_BULL_CAUTION': 0.80,
    'EUPHORIE':         0.40,
    'BLOW_OFF_TOP':     0.10,
}


# ═══════════════════════════════════════════════════════
# DD-SCHWELLEN (Spec TEIL3, Abschnitt 13.2 + Addendum)
# ═══════════════════════════════════════════════════════

DD_LOOKBACK_DAYS = 21

DD_THRESHOLDS = {
    'KAPITULATION':     {'warn': -0.45, 'crit': -0.55, 'w_mult': 0.70, 'c_mult': 0.30},
    'LATE_BEAR':        {'warn': -0.40, 'crit': -0.50, 'w_mult': 0.70, 'c_mult': 0.30},
    'AKKUMULATION':     {'warn': -0.35, 'crit': -0.45, 'w_mult': 0.70, 'c_mult': 0.30},
    'TRANSITION':       {'warn': -0.30, 'crit': -0.40, 'w_mult': 0.60, 'c_mult': 0.20},
    'EARLY_BULL':       {'warn': -0.30, 'crit': -0.40, 'w_mult': 0.60, 'c_mult': 0.20},
    'MID_BULL':         {'warn': -0.25, 'crit': -0.35, 'w_mult': 0.50, 'c_mult': 0.20},
    'DISTRIBUTION':     {'warn': -0.20, 'crit': -0.30, 'w_mult': 0.50, 'c_mult': 0.20},
    'MID_BULL_CAUTION': {'warn': -0.20, 'crit': -0.30, 'w_mult': 0.50, 'c_mult': 0.15},
    'EUPHORIE':         {'warn': -0.15, 'crit': -0.25, 'w_mult': 0.40, 'c_mult': 0.10},
    'BLOW_OFF_TOP':     {'warn': -0.10, 'crit': -0.15, 'w_mult': 0.30, 'c_mult': 0.05},
}


# ═══════════════════════════════════════════════════════
# TRAILING STOP (Spec TEIL3, Abschnitt 13.3)
# ═══════════════════════════════════════════════════════

TRAILING_STOP = {
    'EARLY_BULL':       -0.25,
    'MID_BULL':         -0.25,
    'DISTRIBUTION':     -0.20,
    'MID_BULL_CAUTION': -0.20,
    'EUPHORIE':         -0.15,
    'BLOW_OFF_TOP':     -0.10,
}
TRAILING_ATH_LOOKBACK = 252  # 1-Jahres-Hoch


# ═══════════════════════════════════════════════════════
# VOLATILITY TARGETING (Spec TEIL3, Abschnitt 13.6)
# ═══════════════════════════════════════════════════════

TARGET_VOL = 0.25           # 25% annualisiert
REAL_VOL_WINDOW = 60        # 60-Tage realized vol


# ═══════════════════════════════════════════════════════
# KORRELATION (Spec TEIL3, 13.8 + Addendum 6.2)
# ═══════════════════════════════════════════════════════

CORRELATION = {
    'window_days':          30,
    'btc_spy_high':         0.60,   # Addendum: gesenkt von 0.70
    'btc_spy_low':          0.30,
    'btc_spy_high_mult':    0.70,
    'btc_gld_high':         0.50,
    'btc_gld_neg':          -0.30,
    'btc_gld_fts_mult':     0.80,   # Flight to Safety
}


# ═══════════════════════════════════════════════════════
# LEVERAGE MONITOR (Spec TEIL3, Abschnitt 13.7)
# ═══════════════════════════════════════════════════════

FUNDING = {
    'window_periods':       9,      # 3 Tage × 3 pro Tag (8h Intervall)
    'moderate':             0.05,   # 0.05% = moderate
    'high':                 0.10,   # 0.10% sustained = HIGH
    'extreme':              0.30,   # 0.30% = EXTREME
    'high_mult':            0.85,
    'extreme_mult':         0.60,
}

OI_MCAP = {
    'elevated':             0.03,   # 3% = elevated
    'extreme':              0.05,   # 5% = extreme
    'elevated_mult':        0.90,
}

LIQUIDATION_COOLDOWN = {
    'cascade_threshold':    500_000_000,  # $500M in 24h
    'cooldown_hours':       48,
}


# ═══════════════════════════════════════════════════════
# KILL SWITCHES (Spec TEIL3, Abschnitt 13.5)
# ═══════════════════════════════════════════════════════

KILL_SWITCHES = {
    'stablecoin_depeg_threshold':   0.02,   # 2% vom Peg
    'stablecoin_depeg_hours':       24,     # Sustained 24h
    'manual_env_var':               'CRYPTO_KILL_SWITCH',
}


# ═══════════════════════════════════════════════════════
# NO-ACTION BAND (Spec TEIL3, Abschnitt 13.13)
# ═══════════════════════════════════════════════════════

NO_ACTION_BAND_TOTAL_PP = 10    # ±10pp Gesamtallokation
NO_ACTION_BAND_ASSET_PP = 5     # ±5pp pro Asset


# ═══════════════════════════════════════════════════════
# TRANSITION SPEED (Spec TEIL2, Abschnitt 9.8)
# ═══════════════════════════════════════════════════════

TRANSITION_SPEED = {
    # Richtung Gefahr: sofort oder Mindest-Tage Bestätigung
    'to_danger': {
        'MID_BULL->EUPHORIE':       0,      # Sofort
        'EUPHORIE->BLOW_OFF_TOP':   0,      # Sofort
        'any->KAPITULATION':        0,      # Sofort
        'any->LATE_BEAR':           0,      # Sofort
        'default':                  0,      # Sofort bei Gefahr
    },
    # Richtung Sicherheit: Bestätigung nötig (Tage)
    'to_safety': {
        'EUPHORIE->MID_BULL':       21,     # 3 Wochen
        'MID_BULL->EARLY_BULL':     14,     # 2 Wochen
        'EARLY_BULL->AKKUMULATION': 14,     # 2 Wochen
        'default':                  10,     # ~1.5 Wochen
    },
}


# ═══════════════════════════════════════════════════════
# PEAK / BOTTOM SIGNAL GEWICHTE (Spec TEIL2, 10.2 + 10.4)
# Erweitert um CoinGlass Indikatoren
# [CALIBRATE] = wird via Backtest kalibriert
# ═══════════════════════════════════════════════════════

PEAK_SIGNALS = {
    # ─── Aus Original-Spec (Gratis) ───
    'pi_cycle_top':         {'weight': 0, 'active': False, 'note': 'Display only (Addendum)'},
    'mvrv_gt_7':            {'weight': 2, 'threshold': 7.0},
    'puell_gt_4':           {'weight': 2, 'threshold': 4.0},
    'fear_greed_gt_90':     {'weight': 1, 'threshold': 90},
    'meme_explosion':       {'weight': 2, 'threshold': 3},       # >= 3 Meme-Coins in Top-100
    'funding_extreme':      {'weight': 1, 'threshold': 0.10},    # 3d-avg > 0.10%
    'rainbow_ge_7':         {'weight': 2, 'threshold': 7},       # Addendum: Primary Top Signal

    # ─── NEU: CoinGlass Erweiterung (Gratis via CoinGlass API) ───
    'nupl_euphoria':        {'weight': 2, 'threshold': 0.75},    # [CALIBRATE]
    'lth_sopr_dist':        {'weight': 2, 'threshold': 1.05},    # [CALIBRATE] sustained
    'reserve_risk_high':    {'weight': 1, 'threshold': 0.008},   # [CALIBRATE]
    'rhodl_high':           {'weight': 1, 'threshold': 50000},   # [CALIBRATE]
    '2yr_ma_above_x5':     {'weight': 1, 'threshold': 1.0},     # Preis > 2YR MA × 5
    'btc_m2_divergence':    {'weight': 1, 'threshold': 0.20},    # [CALIBRATE] BTC > 20% über M2 Trend

    # ─── CoinGlass Composite (Meta-Indikator) ───
    'cg_peak_composite':    {'weight': 2, 'threshold': 22},      # [CALIBRATE] >= 22/30 Hits
}

PEAK_WEIGHT_TOTAL = sum(s['weight'] for s in PEAK_SIGNALS.values() if s.get('active', True))

BOTTOM_SIGNALS = {
    # ─── Aus Original-Spec ───
    'pi_cycle_bottom':      {'weight': 0, 'active': False, 'note': 'Display only (Addendum)'},
    'below_200w_ma':        {'weight': 2, 'threshold': True},
    'mvrv_lt_0':            {'weight': 3, 'threshold': 0.0},
    'puell_lt_05':          {'weight': 2, 'threshold': 0.5},
    'fear_greed_lt_10':     {'weight': 1, 'threshold': 10},
    'funding_negative':     {'weight': 1, 'threshold': 0.0},     # 3d-avg < 0
    'rainbow_le_2':         {'weight': 2, 'threshold': 2},

    # ─── NEU: CoinGlass Erweiterung ───
    'nupl_capitulation':    {'weight': 2, 'threshold': 0.0},     # NUPL < 0
    'sth_rp_above_price':   {'weight': 2, 'threshold': True},    # [CALIBRATE] BTC < STH Realized Price
}

BOTTOM_WEIGHT_TOTAL = sum(s['weight'] for s in BOTTOM_SIGNALS.values() if s.get('active', True))

# Puell > 8 Override (Spec TEIL2, 10.5)
PUELL_EXTREME_EUPHORIA = 8.0     # Puell > 8 → Peak Prob MINIMUM 60%
PUELL_EXTREME_MIN_PEAK = 0.60


# ═══════════════════════════════════════════════════════
# COINGLASS BULL MARKET PEAK COMPOSITE
# ═══════════════════════════════════════════════════════

CG_PEAK_COMPOSITE = {
    'warn_threshold':       15,     # [CALIBRATE] >= 15/30 = Warnung
    'sell_threshold':       22,     # [CALIBRATE] >= 22/30 = Sell-Signal
    'display_always':       True,   # Immer im CIO Tab anzeigen
}


# ═══════════════════════════════════════════════════════
# DISPLAY-ONLY INDIKATOREN (kein Einfluss auf Allokation)
# ═══════════════════════════════════════════════════════

DISPLAY_INDICATORS = [
    'pi_cycle_top',             # Pi Cycle Top (Addendum: nur Display)
    'pi_cycle_bottom',          # Pi Cycle Bottom
    'ahr999',                   # Ahr999 Index
    'bubble_index',             # Bitcoin Bubble Index
    'bmo',                      # Bitcoin Macro Oscillator
    'golden_ratio_multiplier',  # Golden Ratio Multiplier
    'profitable_days',          # Bitcoin Profitable Days %
    'active_addresses',         # Network Activity
    'new_addresses',            # Network Growth
    'stock_to_flow',            # S2F — bewusst verworfen als Signal, nur Display
]


# ═══════════════════════════════════════════════════════
# KLASSE 3 GEWICHTE (Addendum Abschnitt 2.7)
# Pi Cycle RAUS, Rainbow + Funding + V16 Instab + V16 FullExp
# ═══════════════════════════════════════════════════════

CLASS3_DANGER = {
    'rainbow_ge_7':             {'weight': 2},
    'funding_sustained_high':   {'weight': 1, 'threshold': 0.10},  # 3d-avg > 0.10%
    'v16_full_expansion':       {'weight': 1},                     # V16 State == FULL_EXPANSION
    'v16_state_instability':    {'weight': 1},                     # >= 3 States in 30d
}
CLASS3_DANGER_THRESHOLD = 3   # [CALIBRATE] Summe >= 3 = DANGER

CLASS3_SAFE = {
    'rainbow_le_2':             {'weight': 2},
    'below_200w_ma':            {'weight': 2},
    'below_200dma_falling':     {'weight': 1},
    'funding_negative':         {'weight': 1, 'threshold': 0.0},   # 3d-avg < 0 sustained
}
CLASS3_SAFE_THRESHOLD = 3     # [CALIBRATE] Summe >= 3 = SAFE


# ═══════════════════════════════════════════════════════
# DISTRIBUTION STATE (Addendum Abschnitt 5.2)
# ═══════════════════════════════════════════════════════

DISTRIBUTION = {
    'rainbow_min':      4,       # Rainbow Band 4-6
    'rainbow_max':      6,
    'v16_state':        3,       # LATE_EXPANSION
    'btc_90d_return':   0.0,     # BTC 90d Return < 0
    'base_allocation':  0.50,
}


# ═══════════════════════════════════════════════════════
# ALTSEASON (Spec TEIL2, Abschnitt 11)
# ═══════════════════════════════════════════════════════

ALTSEASON = {
    # ETH/BTC Momentum (Primary Trigger)
    'eth_btc_start':        0.0,    # > 0% = Altseason beginnt
    'eth_btc_confirmed':    0.10,   # > 10% = bestätigt
    'eth_btc_over':         -0.10,  # < -10% = vorbei

    # Altseason Index (Bestätigung)
    'index_confirmed':      50,     # > 50 + ETH/BTC pos = doppelt bestätigt
    'index_ripe':           75,     # > 75 = REIF → Tier 2 reduzieren

    # BTC Dominance Velocity
    'velocity_fast':        -3.0,   # BTC.D 7d change < -3pp = FAST
    'velocity_normal_low':  -3.0,
    'velocity_normal_high': -1.0,
    'velocity_slow_low':    -1.0,
    'velocity_slow_high':   1.0,
    'velocity_reverse':     1.0,    # BTC.D 7d change > +1pp = REVERSE

    # Modell-Wahl (Spec TEIL2, 11.4)
    'model_a_threshold':    -3.0,   # BTC.D 30d change < -3pp → Klassisch
    'model_b_sol_threshold': 0.20,  # SOL/BTC 30d > 20% → Parallel

    # Meme-Explosion (Spec TEIL2, 11.5)
    'meme_7d_change':       2.0,    # > 200% 7d
    'meme_min_mcap':        1_000_000_000,
    'meme_count_threshold': 3,

    # Tier 1 Verteilung
    'tier1_strong_share':   0.60,   # Stärkerer Coin: 60%
    'tier1_weak_share':     0.40,   # Schwächerer: 40%

    # Überhitzungs-Check
    'overheat_30d_return':  1.0,    # > 100% in 30d
    'overheat_funding':     0.05,   # AND Funding > 0.05%
    'overheat_max_alloc':   0.10,   # → capped bei 10%
}

# Default Tier-Allokation (kein Altseason aktiv)
DEFAULT_TIER_ALLOC = {
    'btc':      0.80,
    'tier1':    0.15,
    'tier2':    0.05,
}


# ═══════════════════════════════════════════════════════
# LIQUIDITY SCORE (Spec TEIL3, Abschnitt 12)
# ═══════════════════════════════════════════════════════

HOWELL_CRYPTO_MULT = {
    1:  1.80,   # Expanding
    0:  1.00,   # Neutral
    -1: 0.25,   # Contracting
}

CRYPTO_NATIVE_WEIGHTS = {
    'stablecoin_minting':       0.25,
    'usdt_usdc_ratio':          0.10,
    'stablecoin_defi_share':    0.15,
    'etf_flows':                0.30,
    'exchange_reserves':        0.20,
}

DUAL_LIQUIDITY_GLOBAL_WEIGHT = 0.50
DUAL_LIQUIDITY_NATIVE_WEIGHT = 0.50

LIQUIDITY_ADJUSTMENT = {
    (80, 101): 1.20,   # 80-100: Tsunami
    (60, 80):  1.00,   # 60-80: Gut
    (40, 60):  0.90,   # 40-60: Abwartend
    (20, 40):  0.70,   # 20-40: Abfluss
    (0, 20):   0.50,   # 0-20: Dürre
}

# ETF Regime Switch (Spec TEIL2, 9.3)
ETF_REGIME = {
    'etf_dominant_threshold':   0.50,   # ETF > 50% Spot Vol = ETF-Dominant
    'crypto_native_threshold':  0.20,   # ETF < 20% = Crypto-Native
}


# ═══════════════════════════════════════════════════════
# CRASH TAXONOMIE (Spec TEIL3, Abschnitt 13.4)
# ═══════════════════════════════════════════════════════

CRASH_TAXONOMY = {
    'type1_liquidity': {
        'btc_dd_7d':                -0.20,    # > 20% in < 7d
        'stablecoin_stable':        True,     # Stablecoin Supply stabil
        'etf_neutral':              True,     # ETF Flows neutral/positiv
        'cooldown_hours':           48,
    },
    'type2_structural': {
        'btc_dd_14d':               -0.20,    # > 20% über 14d
        'stablecoin_redemption':    -500_000_000,
        'etf_outflow_14d':          -1_000_000_000,
        'max_alloc':                0.30,
    },
    'type3_systemic': {
        'triggers': ['exchange_withdrawal_paused', 'stablecoin_depeg', 'v16_financial_crisis'],
        'alloc':                    0.0,      # Sofort auf 0-10%
        'auto_reentry':             False,    # Nur manuell
    },
}


# ═══════════════════════════════════════════════════════
# DCA LOGIK (Spec TEIL3, Abschnitt 13.12)
# ═══════════════════════════════════════════════════════

DCA = {
    'accumulate_bear':      {'pct_per_week': 0.20, 'weeks': 5},     # 20% Delta/Woche
    'accumulate_bull':      {'pct_per_week': 0.33, 'weeks': 3},     # 33% Delta/Woche
    'accumulate_aggressive': {'pct_immediate': 0.50, 'rest_weeks': 2},
}


# ═══════════════════════════════════════════════════════
# RE-ENTRY LOGIK (Spec TEIL3, Abschnitt 13.11)
# ═══════════════════════════════════════════════════════

REENTRY = {
    'after_peak_reduction': {
        'peak_prob_below':      0.30,
        'min_weeks':            2,
    },
    'after_dd_triggered': {
        'cooldown_days':        7,
        'reentry_fraction':     0.80,   # 80% des vorherigen Niveaus
    },
    'after_trailing_stop': {
        'new_21d_high':         True,
        'peak_prob_below':      0.40,
    },
    'after_kill_switch':        'MANUAL_ONLY',
    'after_crash_type1':        {'cooldown_hours': 48},
    'after_crash_type2':        {'weeks': 4},
    'after_crash_type3':        'MANUAL_ONLY',
}


# ═══════════════════════════════════════════════════════
# GRACEFUL DEGRADATION (Spec TEIL4, Abschnitt 16)
# Erweitert um CoinGlass Indikatoren
# ═══════════════════════════════════════════════════════

GRACEFUL_DEGRADATION = {
    # ─── Kritisch ───
    'btc_price':            {'fallback': 'v16_sheet',       'stale_ok_hours': 24,   'critical': True},

    # ─── Preise ───
    'eth_price':            {'fallback': None,              'stale_ok_hours': 24,   'critical': False},
    'sol_price':            {'fallback': None,              'stale_ok_hours': 24,   'critical': False},

    # ─── On-Chain (CoinGlass) ───
    'mvrv_zscore':          {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'puell_multiple':       {'fallback': 'blockchain_calc', 'stale_ok_hours': 168,  'critical': False},
    'nupl':                 {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'lth_sopr':             {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'sth_sopr':             {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'reserve_risk':         {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'rhodl_ratio':          {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'sth_realized_price':   {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'lth_realized_price':   {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'lth_supply':           {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'sth_supply':           {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},

    # ─── CoinGlass Composite ───
    'cg_peak_composite':    {'fallback': 'last_value',      'stale_ok_hours': 48,   'critical': False},

    # ─── Cycle Models (CoinGlass) ───
    'ahr999':               {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'bubble_index':         {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'bmo':                  {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'golden_ratio':         {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    '2yr_ma_mult':          {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},

    # ─── Sentiment ───
    'fear_greed':           {'fallback': 'last_value',      'stale_ok_hours': 48,   'critical': False},

    # ─── Derivate ───
    'funding_rates':        {'fallback': 'assume_zero',     'stale_ok_hours': 24,   'critical': False},
    'open_interest':        {'fallback': None,              'stale_ok_hours': 24,   'critical': False},
    'liquidations':         {'fallback': None,              'stale_ok_hours': 24,   'critical': False},

    # ─── Stablecoins + DeFi ───
    'stablecoin_supply':    {'fallback': 'last_value',      'stale_ok_hours': 48,   'critical': False},
    'defi_tvl':             {'fallback': 'last_value',      'stale_ok_hours': 336,  'critical': False},

    # ─── ETF ───
    'etf_flows':            {'fallback': 'farside',         'stale_ok_hours': 168,  'critical': False},

    # ─── Altseason ───
    'altseason_index':      {'fallback': 'last_value',      'stale_ok_hours': 336,  'critical': False},
    'btc_dominance':        {'fallback': 'coinmarketcap',   'stale_ok_hours': 24,   'critical': False},

    # ─── V16 Bridge ───
    'v16_macro_state':      {'fallback': 'last_value',      'stale_ok_hours': 24,   'critical': False},
    'howell_liquidity':     {'fallback': 'last_value',      'stale_ok_hours': 24,   'critical': False},

    # ─── Kill Switch ───
    'usdt_price':           {'fallback': None,              'stale_ok_hours': 1,    'critical': False},
    'usdc_price':           {'fallback': None,              'stale_ok_hours': 1,    'critical': False},

    # ─── Network ───
    'active_addresses':     {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
    'new_addresses':        {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},

    # ─── BTC vs M2 ───
    'btc_vs_m2':            {'fallback': 'last_value',      'stale_ok_hours': 336,  'critical': False},

    # ─── Meme ───
    'top_100_coingecko':    {'fallback': None,              'stale_ok_hours': 168,  'critical': False},

    # ─── Exchange ───
    'exchange_reserves':    {'fallback': 'last_value',      'stale_ok_hours': 168,  'critical': False},
}

DEGRADATION_LEVELS = {
    0: 'NOMINAL',       # Alle Quellen OK
    1: 'DEGRADED',      # 1-3 ausgefallen
    2: 'MINIMAL',       # > 3 ausgefallen → keine neuen Empfehlungen
    3: 'OFFLINE',       # BTC-Preis nicht verfügbar → Run abbrechen
}
DEGRADATION_THRESHOLD_MINIMAL = 4   # > 3 Ausfälle = MINIMAL


# ═══════════════════════════════════════════════════════
# ALLOKATIONS-KETTE (Spec TEIL3, 13.10)
# Reihenfolge der Modifier-Anwendung
# ═══════════════════════════════════════════════════════

ALLOCATION_CHAIN = [
    'base_allocation',          # 1. Aus Cycle State
    'peak_bottom_adjustment',   # 2. Peak/Bottom Probability
    'liquidity_adjustment',     # 3. Dual Liquidity Score
    'dd_adjustment',            # 4. DD/Trailing Stop
    'leverage_adjustment',      # 5. Funding + OI
    'correlation_adjustment',   # 6. BTC/SPY + BTC/GLD
    'v16_macro_modifier',       # 7. V16 State
    'vol_cap',                  # 8. Vol-Normalized Maximum
    'kill_switch_check',        # 9. Überschreibt alles
    'no_action_band',           # 10. ±10pp → HOLD
]


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

CONFIG_VERSION = '1.0'
SPEC_VERSION = 'V2 + ADDENDUM'
INDICATOR_VERSION = 'V2.1 (CoinGlass erweitert)'
