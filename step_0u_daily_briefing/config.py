"""
Daily Briefing System — Configuration & Constants
Baldur Creek Capital | Step 0u
Based on: DAILY_BRIEFING_SYSTEM_SPEC_TEIL1.md §3.1-3.2

All indicator definitions, Composite Score weights, regime-conditional
profiles, warning triggers, and system constants.
"""

import os

# ---------------------------------------------------------------------------
# Sheet IDs & Drive
# ---------------------------------------------------------------------------
DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"
DRIVE_CURRENT_FOLDER = "1JelM_zZgPeX8TluTfaNqQmsTm3tXkG_8"
DRIVE_ROOT_FOLDER = "1Tng3i4Cly7isKOxIkGqiTmGiZNEtPj3D"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-sonnet-4-6"
NEWSLETTER_MAX_TOKENS = 8192
ANCHOR_MAX_TOKENS = 3000
ANCHOR_TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# Timing (Spec §2.2)
# ---------------------------------------------------------------------------
WEEKDAY_RUN_HOUR_CET = 7   # 07:00 CET Mo-Fr
WEEKEND_RUN_HOUR_CET = 8   # 08:00 CET Sa-So

# ---------------------------------------------------------------------------
# Brave Search (Breaking News Scanner, Spec §3.1.3)
# ---------------------------------------------------------------------------
BRAVE_API_KEY = "BSAhIFYaR9j2nxpVnWFRzhHw_-Uzaih"
BRAVE_NEWS_MAX_RESULTS = 10
BRAVE_NEWS_LOOKBACK_HOURS = 12

BREAKING_NEWS_KEYWORDS = {
    "GEOPOLITIK": [
        "Iran attack", "Hormuz", "OPEC emergency", "war escalation",
        "sanctions", "Taiwan conflict", "NATO",
    ],
    "ZENTRALBANKEN": [
        "Fed emergency", "ECB rate decision", "PBOC cut",
        "BOJ intervention", "rate surprise",
    ],
    "CREDIT_SYSTEMISCH": [
        "bank failure", "credit default", "margin call",
        "liquidity crisis", "bailout", "bank run",
    ],
    "COMMODITIES": [
        "oil supply disruption", "OPEC cut", "gold record",
        "LNG shutdown", "commodity shock",
    ],
    "REGULIERUNG": [
        "crypto ban", "SEC enforcement", "capital controls",
    ],
}

# ---------------------------------------------------------------------------
# FRED Series IDs (Spec §3.1.2)
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # Kern-Indikatoren
    "HY_OAS":       "BAMLH0A0HYM2",     # ICE BofA US HY OAS
    "2Y10Y":        "T10Y2Y",            # 10Y-2Y Treasury Spread
    "3M10Y":        "T10Y3M",            # 10Y-3M Treasury Spread
    "REAL_YIELD":   "DFII10",            # 10Y TIPS Yield
    # Net Liquidity components
    "WALCL":        "WALCL",             # Fed Balance Sheet
    "RRP":          "RRPONTSYD",         # Reverse Repo
    "TGA":          "WTREGEN",           # Treasury General Account
    # Watchlist
    "TED_SPREAD":   "TEDRATE",           # TED Spread
}

# ---------------------------------------------------------------------------
# EODHD Tickers (Spec §3.1.2)
# ---------------------------------------------------------------------------
EODHD_TICKERS = {
    "VIX":          "VIX.INDX",
    "VIX3M":        "VIX3M.INDX",
    "MOVE":         "MOVE.INDX",
    "DXY":          "DX-Y.NYB",
    # Breadth — NYSE A/D via EODHD or fallback from DW Sheet
    "SPY":          "SPY.US",
    "HYG":          "HYG.US",
    "DBC":          "DBC.US",
    "GLD":          "GLD.US",
    "XLU":          "XLU.US",
    "XLP":          "XLP.US",
    "TLT":          "TLT.US",
}

# ---------------------------------------------------------------------------
# Composite Score Zones (Spec §3.2.1)
# ---------------------------------------------------------------------------
COMPOSITE_ZONES = {
    "CALM":     (70, 100),
    "ELEVATED": (50, 69),
    "STRESS":   (30, 49),
    "PANIC":    (0, 29),
}

def get_composite_zone(score):
    """Return zone label for a composite score 0-100."""
    if score >= 70:
        return "CALM"
    if score >= 50:
        return "ELEVATED"
    if score >= 30:
        return "STRESS"
    return "PANIC"

# ---------------------------------------------------------------------------
# Regime-Conditional Indicator Weights (Spec §3.2.2)
#
# Each regime maps indicator_key -> weight (must sum to 1.0).
# Indicators not listed for a regime get weight 0.
# ---------------------------------------------------------------------------
REGIME_WEIGHTS = {
    "LATE_EXPANSION": {
        "HY_OAS":           0.20,
        "BREADTH":          0.15,
        "CU_AU":            0.15,
        "VIX_TERM":         0.12,
        "NET_LIQ":          0.10,
        "2Y10Y":            0.08,
        "MOVE":             0.10,
        "HYG_FLOWS":        0.10,
    },
    "EARLY_RECOVERY": {
        "NET_LIQ":          0.20,
        "2Y10Y":            0.18,
        "REAL_YIELD":       0.15,
        "BREADTH":          0.12,
        "3M10Y":            0.10,
        "HY_OAS":           0.10,
        "VIX_TERM":         0.08,
        "EM_SPREADS":       0.07,
    },
    "CONTRACTION": {
        "HY_OAS":           0.20,
        "PUT_CALL":         0.15,
        "TED_SPREAD":       0.15,
        "NET_LIQ":          0.15,
        "VIX_TERM":         0.12,
        "BREADTH":          0.10,
        "2Y10Y":            0.08,
        "MOVE":             0.05,
    },
    "FULL_EXPANSION": {
        "BREADTH":          0.18,
        "VIX_TERM":         0.15,
        "NET_LIQ":          0.15,
        "HY_OAS":           0.12,
        "2Y10Y":            0.10,
        "CU_AU":            0.10,
        "MOVE":             0.10,
        "REAL_YIELD":       0.10,
    },
    "STRESS_ELEVATED": {
        "HY_OAS":           0.22,
        "VIX_TERM":         0.18,
        "NET_LIQ":          0.15,
        "MOVE":             0.12,
        "TED_SPREAD":       0.10,
        "BREADTH":          0.10,
        "2Y10Y":            0.08,
        "PUT_CALL":         0.05,
    },
    "DEEP_CONTRACTION": {
        "HY_OAS":           0.22,
        "VIX_TERM":         0.18,
        "NET_LIQ":          0.15,
        "TED_SPREAD":       0.12,
        "PUT_CALL":         0.10,
        "MOVE":             0.10,
        "BREADTH":          0.08,
        "2Y10Y":            0.05,
    },
    "FINANCIAL_CRISIS": {
        "HY_OAS":           0.25,
        "VIX_TERM":         0.20,
        "TED_SPREAD":       0.15,
        "NET_LIQ":          0.15,
        "MOVE":             0.10,
        "BREADTH":          0.08,
        "PUT_CALL":         0.07,
    },
}

# STRUCTURAL timeframe override — emphasizes slow-moving indicators
# regardless of current regime (used by composite.py for STRUCTURAL score)
REGIME_WEIGHTS["STRUCTURAL_OVERRIDE"] = {
    "2Y10Y":        0.20,
    "3M10Y":        0.15,
    "REAL_YIELD":   0.15,
    "NET_LIQ":      0.20,
    "HY_OAS":       0.12,
    "BREADTH":      0.08,
    "VIX_TERM":     0.05,
    "MOVE":         0.05,
}

# Default weights for regimes not explicitly defined above
DEFAULT_WEIGHTS = {
    "HY_OAS":           0.18,
    "NET_LIQ":          0.15,
    "VIX_TERM":         0.13,
    "BREADTH":          0.12,
    "2Y10Y":            0.10,
    "3M10Y":            0.08,
    "REAL_YIELD":       0.08,
    "MOVE":             0.08,
    "PUT_CALL":         0.08,
}

def get_regime_weights(regime):
    """Return indicator weight dict for given V16 regime."""
    return REGIME_WEIGHTS.get(regime, DEFAULT_WEIGHTS)

# ---------------------------------------------------------------------------
# Warning Triggers (Spec §3.2.5)
# ---------------------------------------------------------------------------
WARNING_TRIGGERS = {
    "VIX_INVERSION":     {"penalty": -15, "description": "VIX Term Structure Inversion"},
    "HY_SPIKE":          {"penalty": -10, "description": "HY OAS Spike (z-score > 2.0)"},
    "BREADTH_COLLAPSE":  {"penalty": -10, "description": "Breadth Collapse (< 0.5 for 3+ days)"},
    "NET_LIQ_DRAIN":     {"penalty": -8,  "description": "Net Liquidity Drain > $50B / 7d"},
    "CROSS_SOURCE_TEMP": {"penalty": -5,  "description": "Cross-Source Temperature Spike (3+ sources)"},
    "RO_EMERGENCY":      {"penalty": -20, "description": "Risk Officer EMERGENCY active"},
    "REGIME_CONFLICT":   {"penalty": -5,  "description": "V16 vs Market Analyst Regime Conflict"},
}

# ---------------------------------------------------------------------------
# Velocity & Acceleration (Spec §3.2.3)
# ---------------------------------------------------------------------------
VELOCITY_RAPID_DETERIORATION = -5   # vel < this → "RAPID DETERIORATION"
VELOCITY_RAPID_IMPROVEMENT = 5      # vel > this → "RAPID IMPROVEMENT"
ACCELERATION_STRESS_THRESHOLD = -3  # acc < this AND score < 50 → alert

# ---------------------------------------------------------------------------
# Data Integrity Thresholds (Spec §3.2.4)
# ---------------------------------------------------------------------------
DATA_INTEGRITY_GREEN = 90    # > 90% → green
DATA_INTEGRITY_YELLOW = 70   # 70-90% → yellow
# Below 70% → Composite greyed out

# ---------------------------------------------------------------------------
# Indicator Normalization
#
# Each indicator is normalized to a 0-100 "health" scale where:
#   100 = maximally bullish/calm
#   0   = maximally bearish/stressed
#
# Normalization uses linear interpolation between "good" and "bad" bounds.
# Values outside bounds are clamped to 0 or 100.
# ---------------------------------------------------------------------------
INDICATOR_NORMALIZATION = {
    "HY_OAS": {
        "good": 250,     # OAS <= 250bps → 100
        "bad": 700,      # OAS >= 700bps → 0
        "invert": True,  # lower is better
        "unit": "bps",
    },
    "NET_LIQ": {
        "good": 6.5e12,  # $6.5T+ → 100
        "bad": 5.0e12,   # $5.0T- → 0
        "invert": False,  # higher is better
        "unit": "USD",
    },
    "VIX_TERM": {
        # VIX / VIX3M ratio. < 0.85 = calm, > 1.1 = extreme stress
        "good": 0.80,
        "bad": 1.15,
        "invert": True,   # lower ratio is better
        "unit": "ratio",
    },
    "BREADTH": {
        # NYSE A/D ratio. > 1.5 = healthy, < 0.4 = collapse
        "good": 1.5,
        "bad": 0.4,
        "invert": False,   # higher is better
        "unit": "ratio",
    },
    "2Y10Y": {
        # Yield curve spread in %. +1.5 = steep, -0.8 = deeply inverted
        "good": 1.5,
        "bad": -0.8,
        "invert": False,   # higher (steeper) is better
        "unit": "pct",
    },
    "3M10Y": {
        "good": 2.0,
        "bad": -1.0,
        "invert": False,
        "unit": "pct",
    },
    "REAL_YIELD": {
        # 10Y TIPS Yield. Lower real yield = easier conditions
        "good": -0.5,
        "bad": 3.0,
        "invert": True,    # lower is better (easier financial conditions)
        "unit": "pct",
    },
    "CU_AU": {
        # Copper/Gold ratio. Higher = cyclical strength
        "good": 0.25,
        "bad": 0.12,
        "invert": False,
        "unit": "ratio",
    },
    "MOVE": {
        # MOVE Index (bond vol). Lower = calmer
        "good": 80,
        "bad": 180,
        "invert": True,
        "unit": "index",
    },
    "PUT_CALL": {
        # Put/Call ratio. Very high = fear, moderate = normal
        # Inverted: high P/C = stress
        "good": 0.7,
        "bad": 1.3,
        "invert": True,
        "unit": "ratio",
    },
    "HYG_FLOWS": {
        # Weekly fund flows in USD. Positive = inflows = good
        "good": 500_000_000,
        "bad": -1_000_000_000,
        "invert": False,
        "unit": "USD",
    },
    "EM_SPREADS": {
        # EM bond spreads (bps). Lower = healthier
        "good": 250,
        "bad": 600,
        "invert": True,
        "unit": "bps",
    },
    "TED_SPREAD": {
        "good": 20,
        "bad": 150,
        "invert": True,
        "unit": "bps",
    },
}

def normalize_indicator(key, value):
    """Normalize raw indicator value to 0-100 health scale."""
    cfg = INDICATOR_NORMALIZATION.get(key)
    if cfg is None or value is None:
        return None

    good = cfg["good"]
    bad = cfg["bad"]
    invert = cfg["invert"]

    if invert:
        # Lower raw value = higher health score
        if value <= good:
            return 100.0
        if value >= bad:
            return 0.0
        return round(100.0 * (bad - value) / (bad - good), 1)
    else:
        # Higher raw value = higher health score
        if value >= good:
            return 100.0
        if value <= bad:
            return 0.0
        return round(100.0 * (value - bad) / (good - bad), 1)

# ---------------------------------------------------------------------------
# Watchlist Indicators — shown only on anomaly (Spec §3.1.2)
# ---------------------------------------------------------------------------
WATCHLIST_TRIGGERS = {
    "LIBOR_OIS":     {"threshold": 50, "unit": "bps", "meaning": "Interbank Funding Stress"},
    "TED_SPREAD":    {"threshold": 100, "unit": "bps", "meaning": "Systemic Credit Stress"},
    "DXY_VELOCITY":  {"threshold": 2.0, "unit": "pct_5d", "meaning": "Dollar Squeeze / EM Stress"},
    "GOLD_SILVER":   {"threshold": 90, "unit": "ratio", "meaning": "Extreme Risk Aversion"},
    "HYG_OUTFLOWS":  {"threshold": -1_000_000_000, "unit": "USD_week", "meaning": "Credit Exit"},
}

# ---------------------------------------------------------------------------
# Anchor Adaptive Length (Spec §4.3)
# ---------------------------------------------------------------------------
ANCHOR_LENGTH = {
    "RUHIG":    {"min_minutes": 2, "max_minutes": 3, "condition": "composite > 70, no breaking news, no event today"},
    "NORMAL":   {"min_minutes": 3, "max_minutes": 4, "condition": "composite 50-70, or event today, or 1+ warning"},
    "CRITICAL": {"min_minutes": 4, "max_minutes": 5, "condition": "composite < 50, or EMERGENCY, or regime shift, or breaking news HIGH"},
}

def get_anchor_type(composite_tactical, has_breaking_news_high, has_emergency, has_regime_shift):
    """Determine anchor type based on conditions."""
    if (composite_tactical < 50
            or has_emergency
            or has_regime_shift
            or has_breaking_news_high):
        return "CRITICAL"
    if composite_tactical <= 70:
        return "NORMAL"
    return "RUHIG"

# ---------------------------------------------------------------------------
# Newsletter Format by Weekday (Spec §9.1)
# ---------------------------------------------------------------------------
NEWSLETTER_FORMATS = {
    0: "DAILY",          # Monday
    1: "DAILY",          # Tuesday
    2: "DAILY",          # Wednesday
    3: "DAILY",          # Thursday
    4: "DAILY_CONTRARIAN",  # Friday — + Contrarian Check
    5: "WOCHENRUECKBLICK",  # Saturday
    6: "WOCHENVORSCHAU",    # Sunday
}

# ---------------------------------------------------------------------------
# Telegram (Spec §8)
# ---------------------------------------------------------------------------
# Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (from env / GitHub Secrets)
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# ---------------------------------------------------------------------------
# Risk Heatmap Categories (Spec §3.3, Block 5)
# ---------------------------------------------------------------------------
RISK_FACTORS = ["Credit Spreads", "Oil/Iran", "Fed Policy"]

RISK_HEATMAP_MAPPING = {
    # asset -> {risk_factor: severity}
    # Severity: DIREKT, INDIREKT, MINIMAL, SAFE_HAVEN
    # This is the default; overridden dynamically based on IC + regime
    "HYG": {"Credit Spreads": "DIREKT", "Oil/Iran": "INDIREKT", "Fed Policy": "DIREKT"},
    "DBC": {"Credit Spreads": "INDIREKT", "Oil/Iran": "DIREKT", "Fed Policy": "INDIREKT"},
    "GLD": {"Credit Spreads": "MINIMAL", "Oil/Iran": "SAFE_HAVEN", "Fed Policy": "MINIMAL"},
    "XLU": {"Credit Spreads": "INDIREKT", "Oil/Iran": "MINIMAL", "Fed Policy": "INDIREKT"},
    "XLP": {"Credit Spreads": "MINIMAL", "Oil/Iran": "MINIMAL", "Fed Policy": "MINIMAL"},
}

# ---------------------------------------------------------------------------
# File Paths
#
# CRITICAL: Use absolute paths based on this file's location so that
# saves work correctly regardless of the process working directory
# (fixes GitHub Actions runner path mismatch).
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(_MODULE_DIR, "data", "history")
COMPOSITE_HISTORY_FILE = os.path.join(HISTORY_DIR, "composite_history.json")
PREDICTION_LOG_FILE = os.path.join(HISTORY_DIR, "prediction_log.json")
INDICATOR_HISTORY_FILE = os.path.join(HISTORY_DIR, "indicator_history.json")

# Drive archive path
DRIVE_NEWSLETTER_FOLDER = "HISTORY/newsletter"

# ---------------------------------------------------------------------------
# Idempotency (Spec §2.3)
# ---------------------------------------------------------------------------
IDEMPOTENCY_FLAG_FILE = os.path.join(HISTORY_DIR, "last_run_date.txt")
