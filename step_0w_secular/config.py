"""
Säkulare Trends Circle — Configuration & Definitions
Baldur Creek Capital | Step 0w (V1.1 — korrigierte Datenquellen)

5 Regime-Blöcke in kausaler Kette:
  Demographic Cliff → Deglobalisierung → Fiscal Dominance
  → Financial Repression → Great Divergence

Data Sources:
  - 18 FRED Series (verifiziert, alle funktionierend)
  - 4 EOD Series (GLD, SLV, SPY, DBC — monatliche Schlusskurse)
  - 2 Statische Serien (China + Germany Working Age Pop — jährlich, hardcoded)

Gekillt/geändert seit V1.0:
  - GOLDAMGBD228NLBM → GEKILLT (IBA-Daten Jan 2022). Ersetzt durch GLD.US (EOD)
  - SLVPRUSD → GEKILLT (IBA-Daten Jan 2022). Ersetzt durch SLV.US (EOD)
  - SP500 (FRED) → nur 10 Jahre. Ersetzt durch SPY.US (EOD)
  - LFWA64TTCNM647S → 400 Error. Ersetzt durch statische Jahreswerte
  - LFWA64TTDEM647S → 400 Error. Ersetzt durch statische Jahreswerte

Update-Frequenz: Monatlich (1. Sonntag im Monat, 03:00 UTC)
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_MODULE_DIR, "data")

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
EOD_BASE_URL = "https://eodhd.com/api"

# LLM (Etappe 3)
CLAUDE_MODEL = "claude-sonnet-4-6"
LLM_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# FRED SERIES — 18 verifiziert funktionierend
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # --- Block 1: Demographic Cliff ---
    "CIVPART":           "CIVPART",              # US Labor Force Participation Rate (monthly, 1948)
    "WAP_US":            "LFWA64TTUSM647S",      # Working Age Pop US 15-64 (monthly, ~1977)

    # --- Block 2: Deglobalisierung & Reshoring ---
    "IMPGS":             "IMPGS",                 # US Imports Goods & Services (quarterly, 1947)
    "MANEMP":            "MANEMP",                # Manufacturing Employment (monthly, 1939)
    "PAYEMS":            "PAYEMS",                # Total Nonfarm Payrolls (monthly, 1939)
    "BOPGSTB":           "BOPGSTB",               # US Trade Balance (quarterly, 1992)

    # --- Block 3: Fiscal Dominance ---
    "NET_INTEREST":      "A091RC1Q027SBEA",       # Federal Net Interest Payments (quarterly, 1947)
    "DEFENSE":           "FDEFX",                 # Federal Defense Spending (quarterly, 1947)
    "DEBT_GDP":          "GFDEGDQ188S",           # Federal Debt / GDP (quarterly, 1966)

    # --- Block 4: Financial Repression ---
    "GS10":              "GS10",                  # 10Y Treasury Yield (monthly, 1953)
    "CPIAUCSL":          "CPIAUCSL",              # CPI All Urban (monthly, 1947)

    # --- Block 5: Great Divergence ---
    "OIL":               "DCOILWTICO",            # WTI Oil Price (daily, 1986)
    "CORP_PROFITS":      "CP",                    # Corporate Profits After Tax (quarterly, 1947)

    # --- Shared ---
    "GDP":               "GDP",                   # Nominal GDP (quarterly, 1947)
    "M2":                "M2SL",                  # M2 Money Supply (monthly, 1959)

    # --- Fragilitäts-Indikatoren ---
    "PRODUCTIVITY":      "OPHNFB",                # Nonfarm Business Labor Productivity (quarterly)
    "IMPORTS_CHINA":     "IMPCH",                 # US Imports from China (monthly, 1985)

    # --- Bewertungs-Kaskade ---
    "COPPER":            "PCOPPUSDM",             # Copper Price USD (monthly, 2003)
}

# ---------------------------------------------------------------------------
# EOD SERIES — 4 Serien (Gold, Silver, SPY, DBC)
# FRED Gold/Silver gekillt (IBA Jan 2022), FRED SP500 nur 10J
# ---------------------------------------------------------------------------
EOD_TICKERS = {
    "SPY":    "SPY.US",     # S&P 500 ETF (ab 1993)
    "DBC":    "DBC.US",     # Invesco DB Commodity ETF (ab 2006)
    "GOLD":   "GLD.US",     # Gold ETF (ab 2004, Proxy für Goldpreis)
    "SILVER": "SLV.US",     # Silver ETF (ab 2006, Proxy für Silberpreis)
}

# ---------------------------------------------------------------------------
# STATISCHE DATEN — China + Germany Working Age Pop
# FRED IDs LFWA64TTCNM647S / LFWA64TTDEM647S geben 400 Error.
# Daten bewegen sich extrem langsam (jährlich). Quelle: OECD/World Bank.
# Manuell 1x pro Jahr updaten.
# ---------------------------------------------------------------------------
STATIC_WAP_DATA = {
    "WAP_CN": {
        "label": "China",
        "color": "#E74C3C",
        "unit": "millions",
        # OECD Working Age Population 15-64, China (Millionen)
        "data": {
            1980: 583.3, 1985: 654.4, 1990: 740.0, 1995: 800.3,
            2000: 845.6, 2001: 852.3, 2002: 859.0, 2003: 866.0,
            2004: 872.8, 2005: 879.7, 2006: 888.5, 2007: 897.0,
            2008: 905.0, 2009: 912.5, 2010: 919.4, 2011: 925.0,
            2012: 928.0, 2013: 929.0, 2014: 928.5, 2015: 926.0,
            2016: 922.0, 2017: 917.0, 2018: 911.0, 2019: 904.0,
            2020: 896.0, 2021: 886.0, 2022: 875.0, 2023: 863.0,
            2024: 851.0, 2025: 839.0,
        },
    },
    "WAP_DE": {
        "label": "Deutschland",
        "color": "#F5A623",
        "unit": "millions",
        # OECD Working Age Population 15-64, Germany (Millionen)
        "data": {
            1980: 52.3, 1985: 54.0, 1990: 54.4, 1995: 55.5,
            2000: 55.1, 2001: 55.1, 2002: 55.1, 2003: 55.0,
            2004: 54.9, 2005: 54.8, 2006: 54.6, 2007: 54.3,
            2008: 54.1, 2009: 53.9, 2010: 53.9, 2011: 53.8,
            2012: 53.7, 2013: 53.5, 2014: 53.3, 2015: 53.2,
            2016: 53.4, 2017: 53.5, 2018: 53.4, 2019: 53.2,
            2020: 52.9, 2021: 52.6, 2022: 52.4, 2023: 52.1,
            2024: 51.8, 2025: 51.5,
        },
    },
}

# ---------------------------------------------------------------------------
# REGIME BLOCK DEFINITIONS — Kausale Kette
# ---------------------------------------------------------------------------

REGIME_WEIGHTS = {
    "demographic_cliff":     0.30,
    "deglobalization":       0.20,
    "fiscal_dominance":      0.15,
    "financial_repression":  0.10,
    "great_divergence":      0.25,
}

REGIME_BLOCKS = {

    # ===================================================================
    # Block 1: "The Demographic Cliff"
    # ===================================================================
    "demographic_cliff": {
        "name": "The Demographic Cliff",
        "name_de": "Der Demografische Abgrund",
        "robustness": "UNVERRÜCKBAR",
        "robustness_bar": 100,
        "horizon": "20-50 Jahre",
        "weight": 0.30,

        "charts": [
            {
                "id": "civpart",
                "name": "US Labor Force Participation Rate",
                "series": ["CIVPART"],
                "type": "single_line",
                "unit": "%",
                "directional_score_method": "low_is_active",
                "chart_weight": 0.6,
                "annotations": [
                    {"date": "1965-01", "label": "Frauen treten in Arbeitsmarkt ein"},
                    {"date": "2000-04", "label": "Peak 67.3%"},
                    {"date": "2020-04", "label": "COVID-Schock"},
                ],
            },
            {
                "id": "working_age_pop",
                "name": "Working Age Population Growth",
                "series": ["WAP_US", "WAP_CN", "WAP_DE"],
                "type": "multi_line",
                "unit": "% YoY",
                "transform": "yoy_growth",
                "lines": [
                    {"key": "WAP_US", "label": "USA",         "color": "#4A90D9"},
                    {"key": "WAP_CN", "label": "China",       "color": "#E74C3C"},
                    {"key": "WAP_DE", "label": "Deutschland", "color": "#F5A623"},
                ],
                "reference_line": {"value": 0, "label": "Bevölkerungsschrumpfung"},
                "directional_score_method": "wap_growth",
                "chart_weight": 0.4,
                "annotations": [
                    {"date": "1972-01", "label": "Peak US Baby Boom Effekt"},
                    {"date": "2015-01", "label": "China Working Age Peak"},
                    {"date": "2022-01", "label": "China absoluter Rückgang"},
                ],
            },
        ],

        "asset_implications": {
            "gold": +0.7, "silver_copper": +0.6, "oil_commodities": +0.3,
            "spy_real": -0.5, "bonds": -0.7,
        },
    },

    # ===================================================================
    # Block 2: "Deglobalisierung & Reshoring"
    # ===================================================================
    "deglobalization": {
        "name": "Deglobalization & Reshoring",
        "name_de": "Deglobalisierung & Reshoring",
        "robustness": "ROBUST",
        "robustness_bar": 75,
        "horizon": "5-20 Jahre",
        "weight": 0.20,

        "charts": [
            {
                "id": "imports_gdp",
                "name": "US Imports / GDP",
                "series": ["IMPGS", "GDP"],
                "type": "ratio",
                "numerator": "IMPGS",
                "denominator": "GDP",
                "multiply": 100,
                "unit": "%",
                "directional_score_method": "low_is_active",
                "chart_weight": 0.4,
                "annotations": [
                    {"date": "1947-01", "label": "Nachkriegs-Isolation (~4%)"},
                    {"date": "1994-01", "label": "NAFTA"},
                    {"date": "2001-12", "label": "China WTO-Beitritt"},
                    {"date": "2012-01", "label": "Globalisierungs-Peak (~18%)"},
                ],
            },
            {
                "id": "mfg_employment",
                "name": "Manufacturing Empl. / Total Empl.",
                "series": ["MANEMP", "PAYEMS"],
                "type": "ratio",
                "numerator": "MANEMP",
                "denominator": "PAYEMS",
                "multiply": 100,
                "unit": "%",
                "directional_score_method": "mfg_employment",
                "chart_weight": 0.3,
                "annotations": [
                    {"date": "1944-01", "label": "WW2 Peak (~38%)"},
                    {"date": "1979-01", "label": "Beginn Deindustrialisierung"},
                    {"date": "2010-01", "label": "Tief (~8.5%)"},
                ],
            },
            {
                "id": "trade_balance_gdp",
                "name": "US Trade Balance / GDP",
                "series": ["BOPGSTB", "GDP"],
                "type": "ratio",
                "numerator": "BOPGSTB",
                "denominator": "GDP",
                "multiply": 100,
                "unit": "%",
                "directional_score_method": "trade_deficit",
                "reference_line": {"value": 0, "label": "Ausgeglichen"},
                "chart_weight": 0.3,
                "annotations": [
                    {"date": "1975-01", "label": "Erstes strukturelles Defizit"},
                    {"date": "2006-01", "label": "Peak Defizit (~-6.0%)"},
                ],
            },
        ],

        "asset_implications": {
            "gold": +0.6, "silver_copper": +0.8, "oil_commodities": +0.7,
            "spy_real": -0.4, "bonds": -0.6,
        },
    },

    # ===================================================================
    # Block 3: "Fiscal Dominance"
    # ===================================================================
    "fiscal_dominance": {
        "name": "Fiscal Dominance",
        "name_de": "Fiskalische Dominanz",
        "robustness": "MITTELFRIST",
        "robustness_bar": 55,
        "horizon": "5-15 Jahre",
        "weight": 0.15,

        "charts": [
            {
                "id": "interest_vs_defense",
                "name": "Federal Debt Interest vs. Defense Spending",
                "series": ["NET_INTEREST", "DEFENSE"],
                "type": "dual_line",
                "lines": [
                    {"key": "NET_INTEREST", "label": "Zinslast",     "color": "#E74C3C"},
                    {"key": "DEFENSE",      "label": "Verteidigung", "color": "#2C3E50"},
                ],
                "unit": "Mrd. USD (ann.)",
                "directional_score_method": "interest_defense_ratio",
                "chart_weight": 0.4,
                "annotations": [
                    {"date": "1946-01", "label": "WW2 Demobilisierung"},
                    {"date": "1985-01", "label": "Reagan Defense Buildup"},
                    {"date": "2025-01", "label": "Zinslast überholt Verteidigung"},
                ],
            },
            {
                "id": "debt_gdp",
                "name": "Federal Debt / GDP",
                "series": ["DEBT_GDP"],
                "type": "single_line",
                "unit": "%",
                "directional_score_method": "high_is_active",
                "reference_line": {"value": 100, "label": "Schulden = Wirtschaftsleistung"},
                "chart_weight": 0.35,
                "annotations": [
                    {"date": "1981-01", "label": "Reagan Defizit-Ära (~30%)"},
                    {"date": "2008-09", "label": "GFC → Schuldensprung"},
                    {"date": "2020-03", "label": "COVID → Explosion"},
                ],
            },
            {
                "id": "spy_m2",
                "name": "SPY / M2 (Aktien kaufkraftbereinigt)",
                "series": ["SPY", "M2"],
                "type": "ratio",
                "numerator": "SPY",
                "denominator": "M2",
                "denominator_scale": 1e3,
                "unit": "Ratio",
                "directional_score_method": "high_is_active",
                "chart_weight": 0.25,
                "annotations": [
                    {"date": "2000-03", "label": "Dot-Com Peak (real)"},
                    {"date": "2007-10", "label": "Pre-GFC Peak (real)"},
                    {"date": "2021-12", "label": "Post-COVID Peak"},
                ],
            },
        ],

        "asset_implications": {
            "gold": +0.9, "silver_copper": +0.6, "oil_commodities": +0.5,
            "spy_real": -0.2, "bonds": -0.9,
        },
    },

    # ===================================================================
    # Block 4: "Financial Repression"
    # ===================================================================
    "financial_repression": {
        "name": "Financial Repression",
        "name_de": "Finanzielle Repression",
        "robustness": "FRAGIL",
        "robustness_bar": 30,
        "horizon": "3-10 Jahre",
        "weight": 0.10,

        "charts": [
            {
                "id": "real_rate",
                "name": "US Real Interest Rate (10Y - CPI YoY)",
                "series": ["GS10", "CPIAUCSL"],
                "type": "computed_real_rate",
                "unit": "%",
                "directional_score_method": "real_rate",
                "reference_line": {"value": 0, "label": "Sparer gewinnen / verlieren"},
                "color_zones": True,
                "chart_weight": 1.0,
                "annotations": [
                    {"date": "1953-01", "label": "Negative Realzinsen (WW2 Schuldenabbau)"},
                    {"date": "1980-06", "label": "Volcker Peak (+9%)"},
                    {"date": "2008-01", "label": "Financial Repression 2.0"},
                ],
            },
            {
                "id": "gold_vs_real_rates",
                "name": "Gold vs. Real Rates (Dual-Axis)",
                "series": ["GOLD", "GS10", "CPIAUCSL"],
                "type": "dual_axis_gold_realrate",
                "unit_left": "USD/oz",
                "unit_right": "% (invertiert)",
                "directional_score_method": "none",
                "chart_weight": 0.0,
                "annotations": [
                    {"date": "2004-11", "label": "GLD ETF Launch"},
                    {"date": "2011-09", "label": "Gold-Bull-Peak"},
                    {"date": "2019-06", "label": "Gold-Bull 2.0"},
                ],
            },
        ],

        "asset_implications": {
            "gold": +1.0, "silver_copper": +0.6, "oil_commodities": +0.5,
            "spy_real": -0.1, "bonds": -1.0,
        },
    },

    # ===================================================================
    # Block 5: "The Great Divergence"
    # ===================================================================
    "great_divergence": {
        "name": "The Great Divergence",
        "name_de": "Die Große Divergenz",
        "robustness": "MITTELFRIST-LANG",
        "robustness_bar": 65,
        "horizon": "10-30 Jahre",
        "weight": 0.25,

        "charts": [
            {
                "id": "gold_spy_ratio",
                "name": "Gold / SPY Ratio",
                "series": ["GOLD", "SPY"],
                "type": "ratio",
                "numerator": "GOLD",
                "denominator": "SPY",
                "unit": "Ratio",
                "directional_score_method": "low_is_active",
                "reference_line_type": "mean",
                "chart_weight": 0.4,
                "annotations": [
                    {"date": "2004-11", "label": "GLD ETF Launch"},
                    {"date": "2011-09", "label": "Gold-Bull-Peak"},
                    {"date": "2020-03", "label": "Neuer Superzyklus?"},
                ],
            },
            {
                "id": "oil_m2",
                "name": "Oil / M2 (Öl kaufkraftbereinigt)",
                "series": ["OIL", "M2"],
                "type": "ratio",
                "numerator": "OIL",
                "denominator": "M2",
                "unit": "Ratio",
                "directional_score_method": "low_is_active",
                "chart_weight": 0.3,
                "annotations": [
                    {"date": "1990-08", "label": "Gulf War Spike"},
                    {"date": "2008-07", "label": "Commodity-Superzyklus Peak"},
                    {"date": "2015-01", "label": "Long-Term Base Building"},
                ],
            },
            {
                "id": "corp_profits_gdp",
                "name": "Corporate Profits / GDP",
                "series": ["CORP_PROFITS", "GDP"],
                "type": "ratio",
                "numerator": "CORP_PROFITS",
                "denominator": "GDP",
                "multiply": 100,
                "unit": "%",
                "directional_score_method": "high_is_active",
                "reference_line_type": "mean",
                "chart_weight": 0.3,
                "annotations": [
                    {"date": "1950-01", "label": "Stabile Gewinnmargen (~5-6%)"},
                    {"date": "2012-01", "label": "Historische Spitze (~11-12%)"},
                ],
            },
        ],

        "asset_implications": {
            "gold": +0.9, "silver_copper": +0.8, "oil_commodities": +0.8,
            "spy_real": -0.6, "bonds": -0.5,
        },
    },
}

# ---------------------------------------------------------------------------
# FRAGILITY INDICATORS
# ---------------------------------------------------------------------------

FRAGILITY_INDICATORS = {
    "demographic_cliff": {
        "name": "Labor Productivity Growth",
        "series": "PRODUCTIVITY",
        "transform": "yoy_growth",
        "threshold": 2.5,
        "threshold_direction": "above",
        "sustained_quarters": 4,
        "description_de": "Produktivitäts-Boom durch Technologie (>2.5% p.a.)",
        "frontend_text": "Was dieses Regime brechen würde: Produktivitäts-Boom durch Technologie (>2.5% p.a.)",
    },
    "deglobalization": {
        "name": "US Imports from China",
        "series": "IMPORTS_CHINA",
        "transform": "yoy_growth",
        "threshold": 10.0,
        "threshold_direction": "above",
        "sustained_quarters": 4,
        "description_de": "Geopolitische Entspannung + China-Handel normalisiert",
        "frontend_text": "Was dieses Regime brechen würde: Geopolitische Entspannung + China-Handel normalisiert",
    },
    "fiscal_dominance": {
        "name": "GDP Growth vs. 10Y Yield",
        "series": ["GDP", "GS10"],
        "transform": "gdp_minus_gs10",
        "threshold": 0.0,
        "threshold_direction": "above",
        "sustained_quarters": 4,
        "description_de": "Produktivitäts-Boom hebt GDP-Wachstum über Zinskosten",
        "frontend_text": "Was dieses Regime brechen würde: Produktivitäts-Boom hebt GDP-Wachstum über Zinskosten",
    },
    "financial_repression": {
        "name": "Real Rates (GS10 - CPI)",
        "series": ["GS10", "CPIAUCSL"],
        "transform": "real_rate",
        "threshold": 2.0,
        "threshold_direction": "above",
        "sustained_months": 6,
        "description_de": "Volcker 2.0 — bewusste Rezession um Inflation zu brechen",
        "frontend_text": "Was dieses Regime brechen würde: Volcker 2.0 — bewusste Rezession um Inflation zu brechen",
    },
    "great_divergence": {
        "name": "Gold/SPY 12M Momentum",
        "series": ["GOLD", "SPY"],
        "transform": "ratio_momentum_12m",
        "threshold": -0.15,
        "threshold_direction": "below",
        "sustained_months": 12,
        "description_de": "Technologie-Disruption macht Commodities obsolet",
        "frontend_text": "Was dieses Regime brechen würde: Technologie-Disruption macht Commodities obsolet",
    },
}

# ---------------------------------------------------------------------------
# ASSET CLASSES
# ---------------------------------------------------------------------------

ASSET_CLASSES = ["gold", "silver_copper", "oil_commodities", "spy_real", "bonds"]

ASSET_CLASS_LABELS = {
    "gold": "Gold", "silver_copper": "Silber / Kupfer",
    "oil_commodities": "Öl / Rohstoffe", "spy_real": "SPY (real)", "bonds": "Bonds",
}

# ---------------------------------------------------------------------------
# ACTIVATION THRESHOLD
# ---------------------------------------------------------------------------

ACTIVE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# REGIME ORDER — kausale Kette
# ---------------------------------------------------------------------------

REGIME_ORDER = [
    "demographic_cliff", "deglobalization", "fiscal_dominance",
    "financial_repression", "great_divergence",
]

# ---------------------------------------------------------------------------
# ROBUSTNESS MAP
# ---------------------------------------------------------------------------

ROBUSTNESS_MAP = {
    "demographic_cliff": "UNVERRÜCKBAR", "deglobalization": "ROBUST",
    "fiscal_dominance": "MITTELFRIST", "financial_repression": "FRAGIL",
    "great_divergence": "MITTELFRIST-LANG",
}

ROBUST_CATEGORIES = ["UNVERRÜCKBAR", "ROBUST", "MITTELFRIST-LANG"]

# ---------------------------------------------------------------------------
# BEWERTUNGS-KASKADE — 6 Ratios
# ---------------------------------------------------------------------------

VALUATION_RATIOS = {
    "SPY_M2": {
        "name": "SPY / M2",
        "numerator": "SPY", "denominator": "M2",
        "denominator_scale": 1e3,
        "direction": "high_is_expensive",
        "level": 1, "available_from": 1993,
    },
    "GOLD_SPY": {
        "name": "Gold / SPY",
        "numerator": "GOLD", "denominator": "SPY",
        "direction": "low_is_cheap",
        "level": 1, "available_from": 2004,
    },
    "DBC_SPY": {
        "name": "DBC / SPY",
        "numerator": "DBC", "denominator": "SPY",
        "direction": "low_is_cheap",
        "level": 1, "available_from": 2006,
    },
    "GOLD_SILVER": {
        "name": "Gold / Silver",
        "numerator": "GOLD", "denominator": "SILVER",
        "direction": "high_is_cheap",
        "level": 2, "available_from": 2006,
    },
    "OIL_GOLD": {
        "name": "Oil / Gold",
        "numerator": "OIL", "denominator": "GOLD",
        "direction": "low_is_cheap",
        "level": 2, "available_from": 2004,
    },
    "COPPER_GOLD": {
        "name": "Copper / Gold",
        "numerator": "COPPER", "denominator": "GOLD",
        "direction": "low_is_cheap",
        "level": 2, "available_from": 2004,
    },
}

# ---------------------------------------------------------------------------
# OUTPUT CONFIGURATION
# ---------------------------------------------------------------------------

OUTPUT_FILE = os.path.join(DATA_DIR, "secular_trends.json")
