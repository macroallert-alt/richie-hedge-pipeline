"""
Cycles Circle — Configuration & Cycle Definitions
Baldur Creek Capital | Step 0v (V3.5 — 27 Assets, Legacy Lead-Relationships)

Data Sources:
- V16 Sheet DATA_Prices:     Asset prices (27 tickers — full V16 universe)
- V16 Sheet DATA_Liquidity:  Net Liq, Global Liq Proxy, MAs, Trend
- V16 Sheet CYCLES_Howell:   Howell phases, momentum, cycle position
- V16 Sheet CALC_Macro_State: Current V16 regime + history
- DW Sheet RAW_MARKET:       HY_OAS_SPREAD, VIX_LEVEL (current values)
- FRED API:                  HY_OAS history, INDPRO, FEDFUNDS, CPI, DGS2,
                             DGS10, DXY (DTWEXBGS), Corp Profits (CP),
                             CASS Freight, ACOGNO, ICSA
- Cycles Sheet:              Output — DASHBOARD, PHASES, LEADS, META, HISTORY

Verified working series (March 2026):
- ISM (NAPM/NAPMNOI): DISCONTINUED → replaced with INDPRO + ACOGNO
- DXY: not in V16 Prices → FRED DTWEXBGS (scale ~110-130)
- HY_OAS FRED: values in % → multiply x100 for bps
- BDI: no API source → replaced with CASS Freight (FRED: FRGSHPUSM649NCIS)
- Earnings: FMP returns 0 for ETFs → FRED Corporate Profits (CP)

V3.5 Changes:
- PRICE_TICKERS expanded from 15 → 27 (full V16 DATA_Prices universe)
- lead_relationships → legacy_lead_relationships (V1.0 empirically disproven)
"""

import os

# ---------------------------------------------------------------------------
# Sheet IDs
# ---------------------------------------------------------------------------
DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"
CYCLES_SHEET_ID = "1Eo0ZlkYNX1gD5_bMP6cHNb366yg5T3TvP1FDrthtZxc"

# ---------------------------------------------------------------------------
# V16 Sheet Tabs
# ---------------------------------------------------------------------------
V16_DATA_PRICES_TAB = "DATA_Prices"
V16_DATA_LIQUIDITY_TAB = "DATA_Liquidity"
V16_CYCLES_HOWELL_TAB = "CYCLES_Howell"
V16_CALC_MACRO_STATE_TAB = "CALC_Macro_State"

# ---------------------------------------------------------------------------
# DW Sheet Tabs (Row 1 = Title, Row 2 = Header, Row 3+ = Data)
# ---------------------------------------------------------------------------
DW_RAW_MARKET_TAB = "RAW_MARKET"

DW_MARKET_INDICATORS = {
    "HY_OAS": "HY_OAS_SPREAD",
    "VIX": "VIX_LEVEL",
}

# ---------------------------------------------------------------------------
# Cycles Sheet Tabs (Output)
# ---------------------------------------------------------------------------
CYCLES_DASHBOARD_TAB = "DASHBOARD"
CYCLES_PHASES_TAB = "PHASES"
CYCLES_LEADS_TAB = "LEADS"
CYCLES_META_TAB = "META"
CYCLES_HISTORY_TAB = "HISTORY"

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# LLM (Phase 4 Synthesis — not used in Phase 1)
CLAUDE_MODEL = "claude-sonnet-4-6"
SYNTHESIS_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_MODULE_DIR, "data")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
CYCLE_DATA_FILE = os.path.join(DATA_DIR, "cycle_data.json")

# ---------------------------------------------------------------------------
# Tier Hierarchy (Spec §2.1)
# ---------------------------------------------------------------------------
TIER_1 = ["LIQUIDITY", "CREDIT", "COMMODITY", "CHINA_CREDIT"]
TIER_2 = ["DOLLAR", "BUSINESS", "FED_RATES", "EARNINGS"]
TIER_3 = ["TRADE", "POLITICAL"]
OVERLAYS = ["BENNER", "REIT"]

# ---------------------------------------------------------------------------
# FRED Series — ALL VERIFIED WORKING (March 2026)
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # Credit
    "HY_OAS_FRED": "BAMLH0A0HYM2",      # HY OAS daily (values in %, x100 for bps!)
    "IG_OAS":      "BAMLC0A0CM",          # IG OAS daily
    # Fed / Rates
    "FEDFUNDS":    "FEDFUNDS",            # Effective FFR (monthly)
    "CPI":         "CPIAUCSL",            # CPI All Urban (monthly)
    "DGS2":        "DGS2",               # 2Y Treasury (daily)
    "DGS10":       "DGS10",              # 10Y Treasury (daily)
    # Business (INDPRO replaces ISM NAPM — discontinued)
    "INDPRO":      "INDPRO",             # Industrial Production Index (monthly)
    "ACOGNO":      "ACOGNO",             # Manufacturers New Orders (monthly)
    "ICSA":        "ICSA",               # Initial Jobless Claims (weekly)
    # Dollar (not in V16 DATA_Prices)
    "DXY":         "DTWEXBGS",            # Trade Weighted Dollar Broad (daily, ~110-130)
    # Earnings (FRED CP replaces FMP — no ETF income stmts)
    "CORP_PROFITS": "CP",                # Corporate Profits After Tax (quarterly)
    # Trade (CASS replaces BDI — no BDI API source)
    "CASS":        "FRGSHPUSM649NCIS",   # Cass Freight Shipments Index (monthly)
    # REIT overlay
    "CASE_SHILLER":    "CSUSHPINSA",     # Case-Shiller Home Price (monthly)
    "BUILDING_PERMITS": "PERMIT",         # Building Permits (monthly)
}

FRED_OBSERVATION_COUNT = 300
FRED_BACKFILL_START = "2004-01-01"
FRED_BACKFILL_LIMIT = 10000

# ---------------------------------------------------------------------------
# V16 DATA_Prices columns — FULL 27-ASSET UNIVERSE (V3.5)
# Order matches V16 Sheet column headers (GLD first, ETH last)
# 12 new vs V3.4: GDXJ, SIL, PLATINUM, IWM, VGK, XLY, XLF, XLE, XLV, VNQ, TIP, LQD
# ---------------------------------------------------------------------------
PRICE_TICKERS = [
    "GLD", "SLV", "GDX", "GDXJ", "SIL",
    "SPY", "XLY", "XLI", "XLF", "XLE",
    "IWM", "XLV", "XLP", "XLU", "VNQ",
    "XLK", "EEM", "VGK", "TLT", "TIP",
    "LQD", "HYG", "DBC", "PLATINUM", "COPPER",
    "BTC", "ETH",
]

PRICE_ROW_LIMIT = 520
V16_BACKFILL_ROW_LIMIT = 6000

# ---------------------------------------------------------------------------
# CYCLE DEFINITIONS (10 Cycles)
# ---------------------------------------------------------------------------

CYCLE_DEFINITIONS = {

    "LIQUIDITY": {
        "name": "Global Liquidity",
        "tier": 1,
        "typical_duration_months": 65,
        "statistical_quality": "HIGH",
        "n_complete_cycles": 8,
        "primary_indicator": {"key": "NET_LIQ", "source": "V16_LIQUIDITY", "column": "Fed_Net_Liq"},
        "primary_asset_overlay": "DBC",
        "phases": {
            "TROUGH":          {"v16_mapping": ["DEEP_CONTRACTION", "FINANCIAL_CRISIS"]},
            "EARLY_RECOVERY":  {"v16_mapping": ["EARLY_RECOVERY"]},
            "EXPANSION":       {"v16_mapping": ["STEADY_GROWTH", "FULL_EXPANSION"]},
            "LATE_EXPANSION":  {"v16_mapping": ["FRAGILE_EXPANSION", "LATE_EXPANSION"]},
            "PEAK":            {"v16_mapping": ["LATE_EXPANSION", "STRESS_ELEVATED"]},
            "CONTRACTION":     {"v16_mapping": ["CONTRACTION", "STRESS_ELEVATED"]},
        },
        "danger_zones": [
            {"type": "VELOCITY", "threshold": -0.02, "severity": "DANGER",
             "description": "Net Liq Velocity unter -2%/Mo bei Niveau unter 12M-MA"},
            {"type": "ABSOLUTE", "threshold": 5.0e12, "severity": "EXTREME",
             "description": "Net Liquidity unter $5.0T"},
        ],
        "legacy_lead_relationships": [
            {"asset": "SPY", "lead_months": 7.5, "direction": "positive", "r_squared": 0.44},
            {"asset": "DBC", "lead_months": 7.5, "direction": "positive", "r_squared": 0.46},
            {"asset": "GLD", "lead_months": 9.0, "direction": "positive", "r_squared": 0.42},
            {"asset": "HYG", "lead_months": 4.5, "direction": "positive", "r_squared": 0.38},
        ],
    },

    "CREDIT": {
        "name": "Credit Cycle",
        "tier": 1,
        "typical_duration_months": 108,
        "statistical_quality": "HIGH",
        "n_complete_cycles": 5,
        "primary_indicator": {"key": "HY_OAS", "source": "FRED", "series_id": "BAMLH0A0HYM2",
                              "note": "values in %, multiply x100 for bps"},
        "primary_asset_overlay": "HYG",
        "phases": {
            "DISTRESS":        {"v16_mapping": ["FINANCIAL_CRISIS", "DEEP_CONTRACTION"]},
            "REPAIR":          {"v16_mapping": ["EARLY_RECOVERY"]},
            "RECOVERY":        {"v16_mapping": ["STEADY_GROWTH"]},
            "EXPANSION":       {"v16_mapping": ["FULL_EXPANSION"]},
            "LATE_EXPANSION":  {"v16_mapping": ["FRAGILE_EXPANSION", "LATE_EXPANSION"]},
            "DETERIORATION":   {"v16_mapping": ["STRESS_ELEVATED", "CONTRACTION"]},
        },
        "danger_zones": [
            {"type": "ABSOLUTE", "threshold": 500, "severity": "ELEVATED",
             "description": "HY Spreads >500bps"},
            {"type": "ABSOLUTE", "threshold": 700, "severity": "DANGER",
             "description": "HY Spreads >700bps — Credit Crisis"},
        ],
        "legacy_lead_relationships": [
            {"asset": "HYG", "lead_months": 0.5, "direction": "negative", "r_squared": 0.85},
            {"asset": "SPY", "lead_months": 4.5, "direction": "negative", "r_squared": 0.48},
        ],
    },

    "COMMODITY": {
        "name": "Commodity Supercycle",
        "tier": 1,
        "typical_duration_months": 360,
        "statistical_quality": "LOW",
        "n_complete_cycles": 2,
        "primary_indicator": {"key": "CRB_REAL", "source": "COMPUTED",
                              "note": "DBC / CPI x 100"},
        "primary_asset_overlay": "DBC",
        "phases": {
            "BEAR":            {"v16_mapping": []},
            "TROUGH":          {"v16_mapping": []},
            "EARLY_BULL":      {"v16_mapping": ["EARLY_RECOVERY"]},
            "MID_BULL":        {"v16_mapping": ["STEADY_GROWTH", "FULL_EXPANSION"]},
            "EUPHORIA":        {"v16_mapping": []},
            "OVERINVESTMENT":  {"v16_mapping": []},
        },
        "danger_zones": [
            {"type": "VELOCITY", "threshold": 0.15, "severity": "EUPHORIA",
             "description": "CRB_REAL >15% in 3 Monaten"},
        ],
        "legacy_lead_relationships": [
            {"asset": "DBC", "lead_months": 1.5, "direction": "positive", "r_squared": 0.90},
            {"asset": "GLD", "lead_months": 3.0, "direction": "positive", "r_squared": 0.50},
        ],
    },

    "CHINA_CREDIT": {
        "name": "China Credit Impulse",
        "tier": 1,
        "typical_duration_months": 48,
        "statistical_quality": "MEDIUM",
        "n_complete_cycles": 5,
        "primary_indicator": {"key": "COPPER_GOLD_RATIO", "source": "COMPUTED",
                              "note": "COPPER / GLD price ratio (proxy for TSF)"},
        "primary_asset_overlay": "DBC",
        "phases": {
            "CONTRACTION":     {"v16_mapping": []},
            "TROUGH":          {"v16_mapping": []},
            "EARLY_STIMULUS":  {"v16_mapping": ["REFLATION"]},
            "EXPANSION":       {"v16_mapping": []},
            "PEAK":            {"v16_mapping": []},
            "WITHDRAWAL":      {"v16_mapping": []},
        },
        "danger_zones": [
            {"type": "ABSOLUTE", "threshold": 0.008, "severity": "DEMAND_COLLAPSE",
             "description": "Cu/Au < 0.008 — Extreme Risikoaversion"},
        ],
        "legacy_lead_relationships": [
            {"asset": "DBC", "lead_months": 10.5, "direction": "positive", "r_squared": 0.55},
        ],
    },

    "DOLLAR": {
        "name": "US Dollar Cycle",
        "tier": 2,
        "typical_duration_months": 192,
        "statistical_quality": "MEDIUM",
        "n_complete_cycles": 3,
        "primary_indicator": {"key": "DXY", "source": "FRED", "series_id": "DTWEXBGS",
                              "note": "Trade Weighted Broad, scale ~110-130"},
        "primary_asset_overlay": "GLD",
        "phases": {
            "TROUGH":          {"v16_mapping": []},
            "STRENGTHENING":   {"v16_mapping": []},
            "PLATEAU":         {"v16_mapping": ["LATE_EXPANSION"]},
            "PEAK":            {"v16_mapping": ["LATE_EXPANSION", "STRESS_ELEVATED"]},
            "WEAKENING":       {"v16_mapping": ["REFLATION"]},
        },
        "danger_zones": [
            {"type": "ABSOLUTE", "threshold": 130, "severity": "DANGER",
             "description": "DTWEXBGS > 130 — Dollar Squeeze"},
        ],
        "legacy_lead_relationships": [
            {"asset": "DBC", "lead_months": 4.5, "direction": "negative", "r_squared": 0.47},
            {"asset": "GLD", "lead_months": 4.5, "direction": "negative", "r_squared": 0.52},
            {"asset": "EEM", "lead_months": 4.5, "direction": "negative", "r_squared": 0.45},
        ],
    },

    "BUSINESS": {
        "name": "Business Cycle",
        "tier": 2,
        "typical_duration_months": 108,
        "statistical_quality": "HIGH",
        "n_complete_cycles": 6,
        "primary_indicator": {"key": "INDPRO_YOY", "source": "FRED", "series_id": "INDPRO",
                              "note": "Industrial Production YoY Growth (ISM NAPM discontinued)"},
        "secondary_indicator": {"key": "NEW_ORDERS_YOY", "source": "FRED", "series_id": "ACOGNO"},
        "primary_asset_overlay": "SPY",
        "phases": {
            "RECESSION":       {"v16_mapping": ["CONTRACTION", "DEEP_CONTRACTION"]},
            "TROUGH":          {"v16_mapping": ["DEEP_CONTRACTION", "EARLY_RECOVERY"]},
            "EARLY_RECOVERY":  {"v16_mapping": ["EARLY_RECOVERY"]},
            "EXPANSION":       {"v16_mapping": ["STEADY_GROWTH", "FULL_EXPANSION"]},
            "LATE_EXPANSION":  {"v16_mapping": ["FRAGILE_EXPANSION", "LATE_EXPANSION"]},
            "PEAK":            {"v16_mapping": ["LATE_EXPANSION"]},
        },
        "danger_zones": [
            {"type": "YOY_GROWTH", "threshold": -2.0, "severity": "RECESSION_RISK",
             "description": "INDPRO YoY < -2%"},
            {"type": "NEW_ORDERS", "threshold": -5.0, "severity": "SEVERE",
             "description": "New Orders YoY < -5%"},
        ],
        "legacy_lead_relationships": [
            {"asset": "SPY", "lead_months": 4.0, "direction": "positive", "r_squared": 0.48},
            {"asset": "DBC", "lead_months": 4.5, "direction": "positive", "r_squared": 0.40},
            {"asset": "HYG", "lead_months": 3.0, "direction": "positive", "r_squared": 0.42},
        ],
    },

    "FED_RATES": {
        "name": "Fed / Interest Rate Cycle",
        "tier": 2,
        "typical_duration_months": 66,
        "statistical_quality": "HIGH",
        "n_complete_cycles": 7,
        "primary_indicator": {"key": "REAL_FFR", "source": "COMPUTED",
                              "note": "FEDFUNDS - CPI YoY. NEUTRAL=0-2%, RESTRICTIVE>2%"},
        "primary_asset_overlay": "GLD",
        "phases": {
            "EASING":          {"v16_mapping": ["EARLY_RECOVERY", "REFLATION"]},
            "NEUTRAL":         {"v16_mapping": ["STEADY_GROWTH"]},
            "TIGHTENING":      {"v16_mapping": ["FULL_EXPANSION", "LATE_EXPANSION"]},
            "RESTRICTIVE":     {"v16_mapping": ["LATE_EXPANSION", "FRAGILE_EXPANSION"]},
            "PRE_PIVOT":       {"v16_mapping": ["LATE_EXPANSION", "STRESS_ELEVATED"]},
            "PIVOT":           {"v16_mapping": []},
        },
        "danger_zones": [
            {"type": "ABSOLUTE", "threshold": 3.0, "severity": "HIGHLY_RESTRICTIVE",
             "description": "Real FFR > 3%"},
            {"type": "SPREAD", "threshold": -1.0, "severity": "RECESSION_SIGNAL",
             "description": "2Y - FFR < -100bps"},
        ],
        "legacy_lead_relationships": [
            {"asset": "GLD", "lead_months": 4.5, "direction": "positive", "r_squared": 0.58},
            {"asset": "HYG", "lead_months": 2.0, "direction": "positive", "r_squared": 0.45},
            {"asset": "TLT", "lead_months": 2.0, "direction": "positive", "r_squared": 0.55},
        ],
    },

    "EARNINGS": {
        "name": "Earnings / Profit Cycle",
        "tier": 2,
        "typical_duration_months": 48,
        "statistical_quality": "HIGH",
        "n_complete_cycles": 7,
        "primary_indicator": {"key": "CORP_PROFITS_YOY", "source": "FRED", "series_id": "CP",
                              "note": "Corporate Profits After Tax YoY Growth (quarterly)"},
        "primary_asset_overlay": "SPY",
        "phases": {
            "CONTRACTION":     {"v16_mapping": ["CONTRACTION", "DEEP_CONTRACTION"]},
            "TROUGH":          {"v16_mapping": ["EARLY_RECOVERY"]},
            "RECOVERY":        {"v16_mapping": ["EARLY_RECOVERY", "STEADY_GROWTH"]},
            "EXPANSION":       {"v16_mapping": ["FULL_EXPANSION"]},
            "LATE_EXPANSION":  {"v16_mapping": ["FRAGILE_EXPANSION", "LATE_EXPANSION"]},
            "PEAK":            {"v16_mapping": ["LATE_EXPANSION"]},
        },
        "danger_zones": [
            {"type": "DURATION", "threshold": 2, "severity": "EARNINGS_RECESSION",
             "description": "Corp Profits YoY < 0% fuer 2+ Quartale"},
        ],
        "legacy_lead_relationships": [
            {"asset": "SPY", "lead_months": 3.0, "direction": "positive", "r_squared": 0.51},
            {"asset": "HYG", "lead_months": 4.5, "direction": "positive", "r_squared": 0.40},
        ],
    },

    "TRADE": {
        "name": "Global Trade / Shipping",
        "tier": 3,
        "typical_duration_months": 42,
        "statistical_quality": "HIGH",
        "n_complete_cycles": 8,
        "primary_indicator": {"key": "CASS_YOY", "source": "FRED", "series_id": "FRGSHPUSM649NCIS",
                              "note": "Cass Freight Shipments YoY Growth (proxy for BDI)"},
        "primary_asset_overlay": "DBC",
        "phases": {
            "COLLAPSE":        {"v16_mapping": ["CONTRACTION", "DEEP_CONTRACTION"]},
            "CONTRACTION":     {"v16_mapping": ["STRESS_ELEVATED", "CONTRACTION"]},
            "TROUGH":          {"v16_mapping": ["EARLY_RECOVERY"]},
            "RECOVERY":        {"v16_mapping": ["EARLY_RECOVERY", "REFLATION"]},
            "EXPANSION":       {"v16_mapping": ["STEADY_GROWTH", "FULL_EXPANSION"]},
            "LATE":            {"v16_mapping": ["LATE_EXPANSION"]},
        },
        "danger_zones": [
            {"type": "YOY_GROWTH", "threshold": -10.0, "severity": "TRADE_COLLAPSE",
             "description": "CASS YoY < -10% — severe trade contraction"},
        ],
        "legacy_lead_relationships": [
            {"asset": "DBC", "lead_months": 2.0, "direction": "positive", "r_squared": 0.44},
            {"asset": "EEM", "lead_months": 3.0, "direction": "positive", "r_squared": 0.38},
        ],
    },

    "POLITICAL": {
        "name": "Political / Presidential Cycle",
        "tier": 3,
        "typical_duration_months": 48,
        "statistical_quality": "HIGH",
        "n_complete_cycles": 24,
        "primary_indicator": {"key": "PRESIDENTIAL_YEAR", "source": "CALENDAR",
                              "note": "2025=Y1, 2026=Y2(Midterm), 2027=Y3, 2028=Y4"},
        "primary_asset_overlay": "SPY",
        "phases": {
            "POST_INAUGURATION": {"v16_mapping": []},
            "MIDTERM":           {"v16_mapping": []},
            "PRE_ELECTION":      {"v16_mapping": []},
            "ELECTION":          {"v16_mapping": []},
        },
        "danger_zones": [
            {"type": "SEASONAL", "threshold": None, "severity": "WEAK_PERIOD",
             "description": "Midterm H1 historisch schwach: +1.2% avg"},
        ],
        "legacy_lead_relationships": [
            {"asset": "SPY", "lead_months": 0, "direction": "calendar", "r_squared": 0.30},
        ],
    },
}

# ---------------------------------------------------------------------------
# OVERLAY DEFINITIONS
# ---------------------------------------------------------------------------
OVERLAY_DEFINITIONS = {
    "BENNER": {
        "name": "Benner Cycle", "type": "static_years", "default_visible": False,
        "data": {"good_times": [2026, 2035], "hard_times_begin": [2027], "panic_years": [2029]},
    },
    "REIT": {
        "name": "REIT / Real Estate Cycle", "type": "indicator", "default_visible": False,
        "primary_indicator": {"key": "CASE_SHILLER", "source": "FRED", "series_id": "CSUSHPINSA"},
    },
}

# ---------------------------------------------------------------------------
# DISPLAY NAMES
# ---------------------------------------------------------------------------
CYCLE_NAMES = {
    "LIQUIDITY": "Global Liquidity", "CREDIT": "Credit Cycle",
    "COMMODITY": "Commodity Supercycle", "CHINA_CREDIT": "China Credit Impulse",
    "DOLLAR": "US Dollar Cycle", "BUSINESS": "Business Cycle",
    "FED_RATES": "Fed / Interest Rate", "EARNINGS": "Earnings / Profit",
    "TRADE": "Global Trade / Shipping", "POLITICAL": "Political / Presidential",
}

# ---------------------------------------------------------------------------
# PHASE COLORS (matches frontend constants.js)
# ---------------------------------------------------------------------------
CYCLE_PHASE_COLORS = {
    "EXPANSION": "signalGreen", "EARLY_RECOVERY": "signalGreen",
    "RECOVERY": "signalGreen", "MID_BULL": "signalGreen",
    "EARLY_BULL": "signalGreen", "EARLY_STIMULUS": "signalGreen",
    "EASING": "signalGreen", "NEUTRAL": "signalGreen",
    "PRE_ELECTION": "signalGreen",
    "LATE_EXPANSION": "signalYellow", "PEAK": "signalYellow",
    "PLATEAU": "signalYellow", "LATE": "signalYellow",
    "OVERINVESTMENT": "signalYellow", "TIGHTENING": "signalYellow",
    "RESTRICTIVE": "signalYellow", "MIDTERM": "signalYellow",
    "POST_INAUGURATION": "signalYellow", "ELECTION": "signalYellow",
    "PRE_PIVOT": "signalYellow", "WITHDRAWAL": "signalYellow",
    "PIVOT": "signalYellow", "REPAIR": "signalYellow",
    "CONTRACTION": "signalOrange", "DETERIORATION": "signalOrange",
    "STRENGTHENING": "signalOrange", "WEAKENING": "signalOrange",
    "BEAR": "signalOrange",
    "TROUGH": "signalRed", "DISTRESS": "signalRed",
    "RECESSION": "signalRed", "COLLAPSE": "signalRed",
    "EUPHORIA": "signalRed",
}

# ---------------------------------------------------------------------------
# V16 REGIME → LEAD MULTIPLIER (Spec Teil 2, §4.2.3)
# ---------------------------------------------------------------------------
REGIME_LEAD_MULTIPLIERS = {
    "STEADY_GROWTH": 1.1, "FULL_EXPANSION": 1.0, "EARLY_RECOVERY": 1.0,
    "FRAGILE_EXPANSION": 0.9, "LATE_EXPANSION": 0.85, "NEUTRAL": 1.0,
    "SOFT_LANDING": 0.9, "REFLATION": 1.0,
    "STRESS_ELEVATED": 0.6, "CONTRACTION": 0.5,
    "DEEP_CONTRACTION": 0.3, "FINANCIAL_CRISIS": 0.2,
}
