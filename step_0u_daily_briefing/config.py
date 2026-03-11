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
    "HY_OAS":       "BAMLH0A0HYM2",
    "2Y10Y":        "T10Y2Y",
    "3M10Y":        "T10Y3M",
    "REAL_YIELD":   "DFII10",
    "WALCL":        "WALCL",
    "RRP":          "RRPONTSYD",
    "TGA":          "WTREGEN",
    "TED_SPREAD":   "TEDRATE",
}

# ---------------------------------------------------------------------------
# EODHD Tickers (Spec §3.1.2)
# ---------------------------------------------------------------------------
EODHD_TICKERS = {
    "VIX":          "VIX.INDX",
    "VIX3M":        "VIX3M.INDX",
    "MOVE":         "MOVE.INDX",
    "DXY":          "DX-Y.NYB",
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
    if score >= 70: return "CALM"
    if score >= 50: return "ELEVATED"
    if score >= 30: return "STRESS"
    return "PANIC"

# ---------------------------------------------------------------------------
# Regime-Conditional Indicator Weights (Spec §3.2.2)
# ---------------------------------------------------------------------------
REGIME_WEIGHTS = {
    "LATE_EXPANSION": {
        "HY_OAS": 0.20, "BREADTH": 0.15, "CU_AU": 0.15, "VIX_TERM": 0.12,
        "NET_LIQ": 0.10, "2Y10Y": 0.08, "MOVE": 0.10, "HYG_FLOWS": 0.10,
    },
    "EARLY_RECOVERY": {
        "NET_LIQ": 0.20, "2Y10Y": 0.18, "REAL_YIELD": 0.15, "BREADTH": 0.12,
        "3M10Y": 0.10, "HY_OAS": 0.10, "VIX_TERM": 0.08, "EM_SPREADS": 0.07,
    },
    "CONTRACTION": {
        "HY_OAS": 0.20, "PUT_CALL": 0.15, "TED_SPREAD": 0.15, "NET_LIQ": 0.15,
        "VIX_TERM": 0.12, "BREADTH": 0.10, "2Y10Y": 0.08, "MOVE": 0.05,
    },
    "FULL_EXPANSION": {
        "BREADTH": 0.18, "VIX_TERM": 0.15, "NET_LIQ": 0.15, "HY_OAS": 0.12,
        "2Y10Y": 0.10, "CU_AU": 0.10, "MOVE": 0.10, "REAL_YIELD": 0.10,
    },
    "STRESS_ELEVATED": {
        "HY_OAS": 0.22, "VIX_TERM": 0.18, "NET_LIQ": 0.15, "MOVE": 0.12,
        "TED_SPREAD": 0.10, "BREADTH": 0.10, "2Y10Y": 0.08, "PUT_CALL": 0.05,
    },
    "DEEP_CONTRACTION": {
        "HY_OAS": 0.22, "VIX_TERM": 0.18, "NET_LIQ": 0.15, "TED_SPREAD": 0.12,
        "PUT_CALL": 0.10, "MOVE": 0.10, "BREADTH": 0.08, "2Y10Y": 0.05,
    },
    "FINANCIAL_CRISIS": {
        "HY_OAS": 0.25, "VIX_TERM": 0.20, "TED_SPREAD": 0.15, "NET_LIQ": 0.15,
        "MOVE": 0.10, "BREADTH": 0.08, "PUT_CALL": 0.07,
    },
}

REGIME_WEIGHTS["STRUCTURAL_OVERRIDE"] = {
    "2Y10Y": 0.20, "3M10Y": 0.15, "REAL_YIELD": 0.15, "NET_LIQ": 0.20,
    "HY_OAS": 0.12, "BREADTH": 0.08, "VIX_TERM": 0.05, "MOVE": 0.05,
}

DEFAULT_WEIGHTS = {
    "HY_OAS": 0.18, "NET_LIQ": 0.15, "VIX_TERM": 0.13, "BREADTH": 0.12,
    "2Y10Y": 0.10, "3M10Y": 0.08, "REAL_YIELD": 0.08, "MOVE": 0.08, "PUT_CALL": 0.08,
}

def get_regime_weights(regime):
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

VELOCITY_RAPID_DETERIORATION = -5
VELOCITY_RAPID_IMPROVEMENT = 5
ACCELERATION_STRESS_THRESHOLD = -3

DATA_INTEGRITY_GREEN = 90
DATA_INTEGRITY_YELLOW = 70

# ---------------------------------------------------------------------------
# Indicator Normalization (0-100 health scale)
# ---------------------------------------------------------------------------
INDICATOR_NORMALIZATION = {
    "HY_OAS":     {"good": 250, "bad": 700, "invert": True, "unit": "bps"},
    "NET_LIQ":    {"good": 6.5e12, "bad": 5.0e12, "invert": False, "unit": "USD"},
    "VIX_TERM":   {"good": 0.80, "bad": 1.15, "invert": True, "unit": "ratio"},
    "BREADTH":    {"good": 1.5, "bad": 0.4, "invert": False, "unit": "ratio"},
    "2Y10Y":      {"good": 1.5, "bad": -0.8, "invert": False, "unit": "pct"},
    "3M10Y":      {"good": 2.0, "bad": -1.0, "invert": False, "unit": "pct"},
    "REAL_YIELD": {"good": -0.5, "bad": 3.0, "invert": True, "unit": "pct"},
    "CU_AU":      {"good": 0.25, "bad": 0.12, "invert": False, "unit": "ratio"},
    "MOVE":       {"good": 80, "bad": 180, "invert": True, "unit": "index"},
    "PUT_CALL":   {"good": 0.7, "bad": 1.3, "invert": True, "unit": "ratio"},
    "HYG_FLOWS":  {"good": 500_000_000, "bad": -1_000_000_000, "invert": False, "unit": "USD"},
    "EM_SPREADS": {"good": 250, "bad": 600, "invert": True, "unit": "bps"},
    "TED_SPREAD": {"good": 20, "bad": 150, "invert": True, "unit": "bps"},
}

def normalize_indicator(key, value):
    cfg = INDICATOR_NORMALIZATION.get(key)
    if cfg is None or value is None:
        return None
    good, bad, invert = cfg["good"], cfg["bad"], cfg["invert"]
    if invert:
        if value <= good: return 100.0
        if value >= bad: return 0.0
        return round(100.0 * (bad - value) / (bad - good), 1)
    else:
        if value >= good: return 100.0
        if value <= bad: return 0.0
        return round(100.0 * (value - bad) / (good - bad), 1)

# ---------------------------------------------------------------------------
# Indicator Descriptions — Investor-Ready (Static Part)
#
# Each indicator gets: full name, what it measures (plain language),
# threshold zones, and why it matters for the portfolio.
# The LLM adds a dynamic "current_assessment" at runtime.
# ---------------------------------------------------------------------------
INDICATOR_DESCRIPTIONS = {
    "HY_OAS": {
        "name": "High Yield Credit Spreads (OAS)",
        "what": "Der Risikoaufschlag, den Unternehmen mit schwacher Bonitaet gegenueber Staatsanleihen zahlen muessen. Misst den Grad an Credit-Stress im Markt — je hoeher der Spread, desto mehr Angst vor Zahlungsausfaellen.",
        "unit_label": "bps",
        "thresholds": "Unter 300 bps = entspannt (Maerkte sorglos). 300-450 bps = normal. 450-600 bps = erhoehte Vorsicht. Ueber 600 bps = akuter Stress (2008: 2000+ bps).",
        "why_it_matters": "HYG ist die direkteste Verbindung — jede Spread-Ausweitung um 50 bps kostet ca. 2-3% im HYG-Preis. Auch DBC und Rohstoffe leiden indirekt, weil hohe Spreads Rezessionsangst signalisieren.",
    },
    "VIX_TERM": {
        "name": "VIX Term Structure (VIX/VIX3M Ratio)",
        "what": "Verhaeltnis zwischen kurzfristiger (30d) und mittelfristiger (90d) erwarteter Aktienvolatilitaet. Normalerweise ist kurzfristige Vol niedriger (Ratio unter 1.0). Wenn das Ratio ueber 1.0 steigt (Inversion), erwarten Haendler JETZT mehr Turbulenzen als in 3 Monaten — ein Panik-Signal.",
        "unit_label": "Ratio",
        "thresholds": "Unter 0.85 = sehr ruhig. 0.85-0.95 = normal. 0.95-1.0 = erhoehte Wachsamkeit. Ueber 1.0 = INVERTIERT — automatisch -15 Punkte Composite Penalty.",
        "why_it_matters": "Inversion trifft alle Risk-On-Positionen (HYG, DBC). GLD profitiert typischerweise als Safe Haven. Einer der zuverlaessigsten Stress-Indikatoren ueberhaupt.",
    },
    "NET_LIQ": {
        "name": "Netto-Liquiditaet (Fed Reserve Drain Proxy)",
        "what": "Verfuegbare Liquiditaet im Finanzsystem: Fed-Bilanz minus Treasury General Account minus Reverse Repo. Steigende Liquiditaet hebt alle Risiko-Assets. Fallende Liquiditaet drueckt alles.",
        "unit_label": "$",
        "thresholds": "Ueber $6.5T = reichlich. $5.5-6.5T = normal. Unter $5.5T = restriktiv. Unter $5.0T = Stress. Die Richtung (steigend/fallend) ist wichtiger als das Niveau.",
        "why_it_matters": "Liquiditaet ist der wichtigste Makro-Treiber fuer ALLE Assets im Portfolio. Steigende Liq hebt DBC, GLD und HYG gleichzeitig. Fallende Liq drueckt alles — kein Versteck ausser Cash und kurze Treasuries.",
    },
    "BREADTH": {
        "name": "Marktbreite (Advance/Decline Proxy)",
        "what": "Misst wie breit eine Rallye oder ein Sell-Off getragen wird. Hohe Breadth = viele Aktien steigen (gesunde Rallye). Niedrige Breadth = nur wenige Mega-Caps tragen den Index (fragil).",
        "unit_label": "Ratio",
        "thresholds": "Ueber 1.5 = sehr gesund. 1.0-1.5 = normal. 0.5-1.0 = schwaecher. Unter 0.5 fuer 3+ Tage = Breadth Collapse (-10 Punkte Composite Penalty).",
        "why_it_matters": "Fallende Breadth in LATE_EXPANSION ist ein klassisches Zyklusende-Signal. Wenn nur 5-10 Mega-Caps den Markt tragen, steigt das Risiko eines breiten Sell-Offs der alle Positionen trifft.",
    },
    "2Y10Y": {
        "name": "Yield Curve 10Y-2Y (Zinsstrukturkurve)",
        "what": "Differenz zwischen 10-Jahres- und 2-Jahres-US-Staatsanleihenrendite. Negativ = invertiert — historisch der zuverlaessigste Rezessions-Indikator. Jede US-Rezession seit 1970 wurde vorhergesagt (Vorlauf 6-18 Monate).",
        "unit_label": "%",
        "thresholds": "Ueber +1.0% = steil (Erholung). 0 bis +1.0% = normal. -0.5 bis 0% = flach/leicht invertiert (Warnsignal). Unter -0.5% = tief invertiert (Rezession wahrscheinlich). ACHTUNG: Re-Steepening (zurueck zu positiv) ist oft die gefaehrlichste Phase — signalisiert unmittelbar bevorstehende Rezession.",
        "why_it_matters": "Direkte Implikation fuer V16-Regime: Reinversion oder schnelles Steepening kann Regime-Wechsel ausloesen. HYG besonders verwundbar bei Shift zu CONTRACTION.",
    },
    "3M10Y": {
        "name": "Yield Curve 10Y-3M",
        "what": "Wie 2Y10Y, aber mit 3-Monats-Zinssatz. In der Forschung der praezisere Rezessions-Indikator, weil der 3M-Satz die aktuelle Fed-Politik direkter abbildet.",
        "unit_label": "%",
        "thresholds": "Ueber +1.5% = steil. 0 bis +1.5% = normal. Unter 0% = invertiert (starkes Signal). Unter -1.0% = tief invertiert (historisch selten, immer gefolgt von Rezession).",
        "why_it_matters": "Bestaetigt oder widerspricht 2Y10Y. Wenn beide invertiert: Rezessions-Wahrscheinlichkeit steigt drastisch. 15% Gewicht im STRUCTURAL Score.",
    },
    "REAL_YIELD": {
        "name": "Real Yield (10Y TIPS)",
        "what": "Rendite inflationsgeschuetzter US-Staatsanleihen — die 'echte' Rendite nach Inflation. Niedrige/negative Real Yields = lockere Finanzbedingungen. Hohe Real Yields = restriktiv.",
        "unit_label": "%",
        "thresholds": "Unter 0% = sehr lockere Bedingungen (Gold profitiert stark). 0-1.5% = neutral. 1.5-2.5% = restriktiv. Ueber 2.5% = stark restriktiv (drueckt alle Bewertungen).",
        "why_it_matters": "Gold (GLD) ist besonders sensitiv: negative Real Yields sind historisch der staerkste Gold-Treiber. Steigende Real Yields sind Gift fuer GLD. Auch DBC leidet bei hohen Real Yields (staerkerer Dollar).",
    },
    "CU_AU": {
        "name": "Copper/Gold Ratio",
        "what": "Kupferpreis geteilt durch Goldpreis. Kupfer = Industrie (steigt bei Wachstum), Gold = Sicherheit (steigt bei Angst). Das Ratio misst 'Wachstum vs. Sicherheit' im Markt.",
        "unit_label": "Ratio",
        "thresholds": "Ueber 0.25 = starke Wirtschaft. 0.18-0.25 = normal. 0.12-0.18 = Schwaeche. Unter 0.12 = extreme Risikoaversion. Trend wichtiger als Niveau.",
        "why_it_matters": "Direkt relevant fuer DBC: fallendes Ratio = nachlassende Industrienachfrage = DBC unter Druck. GLD profitiert gleichzeitig. In LATE_EXPANSION hat CU/AU 15% Gewicht — dritthoechstes im Composite.",
    },
    "MOVE": {
        "name": "MOVE Index (Bond-Volatilitaet)",
        "what": "Der 'VIX fuer Anleihen' — erwartete Volatilitaet am US-Treasury-Markt. Hohe Werte = Unsicherheit ueber Zinsentwicklung und Fed-Politik. Seit 2022 besonders bedeutend, weil die Fed der dominante Markt-Treiber ist.",
        "unit_label": "Index",
        "thresholds": "Unter 80 = ruhig (selten seit 2022). 80-120 = erhoehte Basisvol (neue Normalitaet). 120-150 = Stress. Ueber 150 = akute Anleihen-Krise. 180+ = Panik (Maerz 2023, Sept 2022).",
        "why_it_matters": "Hohe Bond-Vol strahlt auf alle Assets: HYG leidet direkt (Spreads), DBC indirekt (Dollar-Vol), GLD kann profitieren (Safe Haven) oder leiden (steigende Real Yields). MOVE ueber 150 = systemisches Warnsignal.",
    },
    "PUT_CALL": {
        "name": "Put/Call Ratio",
        "what": "Verhaeltnis von Put-Optionen (Absicherung/Wetten auf Kursfall) zu Call-Optionen (Wetten auf Anstieg). Eine der direktesten Messungen von Angst vs. Gier am Markt.",
        "unit_label": "Ratio",
        "thresholds": "Unter 0.7 = extreme Sorglosigkeit (kontraer: Warnsignal). 0.7-0.9 = normal. 0.9-1.1 = erhoehte Vorsicht. Ueber 1.1 = Angst. Ueber 1.3 = Panik (kontraer: oft Boden).",
        "why_it_matters": "Kontraerer Indikator: Sehr niedriges P/C in LATE_EXPANSION = Markt nicht abgesichert, Schock trifft haerter. Sehr hohes P/C in CONTRACTION = Panik moeglicherweise eingepreist (kontraeres Kaufsignal).",
    },
    "HYG_FLOWS": {
        "name": "HYG Fund Flows (Wochenzufluesse)",
        "what": "Netto-Kapitalzufluesse in High-Yield-Bond-Fonds. Positive Flows = Anleger kaufen Kreditrisiko. Negative = Anleger verkaufen. Direktes Mass fuer Nachfrage nach genau der Asset-Klasse unserer groessten Position.",
        "unit_label": "$",
        "thresholds": "Ueber +$500M/Wo = starke Zufluesse. 0 bis +$500M = neutral. -$500M bis 0 = leichte Abfluesse. -$500M bis -$1B = erhoehte Abfluesse. Unter -$1B = Kapitalflucht (akutes Warnsignal).",
        "why_it_matters": "Direkt relevant fuer HYG — anhaltende Abfluesse druecken den Kurs auch ohne fundamentale Spread-Veraenderung, weil ETF-Market-Maker Anleihen liquidieren muessen.",
    },
    "EM_SPREADS": {
        "name": "EM Bond Spreads (Emerging Markets)",
        "what": "Risikoaufschlag fuer Schwellenlaender-Anleihen. Misst globale Risikobereitschaft — wenn Investoren aus EM fliehen, steigen Spreads. Fruehindikator fuer breiteren Risk-Off, weil EM typischerweise VOR Industrielaender verkauft wird.",
        "unit_label": "bps",
        "thresholds": "Unter 250 bps = Risk-On. 250-400 bps = normal. 400-600 bps = erhoehter Stress. Ueber 600 bps = EM-Krise (Ansteckungsgefahr).",
        "why_it_matters": "Steigende EM-Spreads = globaler Risikoabbau — trifft typischerweise 2-4 Wochen spaeter HY und Commodities. Besonders gefaehrlich wenn gleichzeitig Dollar steigt.",
    },
    "TED_SPREAD": {
        "name": "TED Spread (Interbank-Stress)",
        "what": "Differenz zwischen 3M-LIBOR und 3M-Treasury-Rendite. Misst Vertrauen der Banken untereinander — hoeher = weniger gegenseitiges Vertrauen. Der klassische Indikator fuer systemischen Bankenstress.",
        "unit_label": "bps",
        "thresholds": "Unter 25 bps = normal. 25-50 bps = leicht erhoehte Vorsicht. 50-100 bps = ernstes Warnsignal. Ueber 100 bps = systemischer Stress. 2008: ueber 400 bps.",
        "why_it_matters": "Wenn Banken sich nicht mehr vertrauen, trocknet Kreditvergabe aus — trifft die gesamte Wirtschaft. TED ueber 100 bps = existenzielles Risiko fuer alle Credit-Positionen (HYG) und Commodities (DBC).",
    },
}

# ---------------------------------------------------------------------------
# Risk Heatmap Descriptions — Investor-Ready (Static Part)
#
# Each position × risk factor gets: severity label + detailed mechanism.
# The LLM adds dynamic "current_relevance" at runtime.
# ---------------------------------------------------------------------------
RISK_HEATMAP_DESCRIPTIONS = {
    "HYG": {
        "Credit Spreads": {
            "severity": "DIREKT",
            "mechanism": "HYG besteht aus High-Yield-Unternehmensanleihen. Wenn Spreads steigen, faellt der Kurs direkt — jede 50 bps Spread-Widening kostet ca. 2-3% im HYG-Preis. Das ist der primaere Risikofaktor fuer die groesste Position.",
        },
        "Oil/Iran": {
            "severity": "INDIREKT",
            "mechanism": "Oel-Eskalation trifft HYG ueber steigende Energiekosten fuer HY-Emittenten, besonders Energie- und Transport-Sektor. Nicht sofort sichtbar, aber nach 2-4 Wochen in hoeheren Default-Erwartungen und Spread-Widening reflektiert.",
        },
        "Fed Policy": {
            "severity": "DIREKT",
            "mechanism": "Hawkishe Fed drueckt HYG doppelt: steigende Zinsen erhoehen Refinanzierungskosten (Default-Risiko steigt) UND machen risikolose Treasuries attraktiver (Kapitalabfluss). Jede unerwartete Zinserhoehung kostet HYG sofort 1-2%.",
        },
    },
    "DBC": {
        "Credit Spreads": {
            "severity": "INDIREKT",
            "mechanism": "Steigende Spreads signalisieren Rezessionsangst — drueckt Nachfrage-Erwartungen fuer Rohstoffe. DBC faellt mit 1-2 Wochen Verzoegerung bei Spreads ueber 450 bps. Zusammenhang ueber Sentiment, nicht mechanisch.",
        },
        "Oil/Iran": {
            "severity": "DIREKT",
            "mechanism": "Oel ist ~30% des DBC-Baskets. Jede Oel-Eskalation trifft DBC sofort. Hormuz-Blockade = Oel +30-50% = DBC +10-15% in Tagen. Deeskalation/Waffenstillstand = DBC -5-10%. Der staerkste kurzfristige Treiber.",
        },
        "Fed Policy": {
            "severity": "INDIREKT",
            "mechanism": "Wirkt ueber Dollar-Kanal: Hawkishe Fed = staerkerer Dollar = Rohstoffe teurer fuer Nicht-Dollar-Kaeufer = weniger Nachfrage. Effekt moderat und langsam (Wochen), aber persistent.",
        },
    },
    "GLD": {
        "Credit Spreads": {
            "severity": "MINIMAL",
            "mechanism": "Gold reagiert kaum auf Spreads. Bei extremem Stress (ueber 600 bps) kann Gold sogar profitieren als sicherer Hafen.",
        },
        "Oil/Iran": {
            "severity": "SAFE_HAVEN",
            "mechanism": "Geopolitische Eskalation ist der staerkste kurzfristige Gold-Treiber. Iran-Konflikte = Gold typisch +3-8% in der Eskalationsphase. Gold ist die natuerliche Absicherung gegen geopolitisches Risiko — wenn DBC durch Iran steigt, steigt GLD noch staerker.",
        },
        "Fed Policy": {
            "severity": "MINIMAL",
            "mechanism": "Kurzfristig wenig Reaktion auf einzelne Fed-Entscheidungen. Mittelfristig entscheidend: steigende Real Yields = negativ, fallende = positiv. Fed-Pause oder Pivot = stark positiv fuer GLD.",
        },
    },
    "XLU": {
        "Credit Spreads": {
            "severity": "INDIREKT",
            "mechanism": "Utilities sind kapitalintensiv mit hohen Schulden. Steigende Spreads erhoehen Refi-Kosten, aber als regulierte Monopole mit stabilen Cashflows ist der Effekt gedaempft.",
        },
        "Oil/Iran": {
            "severity": "MINIMAL",
            "mechanism": "US-Utilities basieren auf Gas und Erneuerbare. Oel-Schock trifft kaum direkt. Genereller Risk-Off kann XLU als 'defensive' Aktie sogar stuetzen.",
        },
        "Fed Policy": {
            "severity": "INDIREKT",
            "mechanism": "Utilities = 'Bond-Proxies': steigende Zinsen machen echte Bonds attraktiver, XLU verliert. Zinssenkungen = XLU profitiert ueberproportional. Moderat, aber konsistent.",
        },
    },
    "XLP": {
        "Credit Spreads": {
            "severity": "MINIMAL",
            "mechanism": "Consumer Staples (P&G, Coca-Cola) = stabile Cashflows, Investment-Grade-Bonitaet. Wenig sensitiv — Konsumenten kaufen auch in Rezessionen Zahnpasta.",
        },
        "Oil/Iran": {
            "severity": "MINIMAL",
            "mechanism": "Direkte Auswirkung gering. Steigende Energiekosten belasten Margen minimal — Staples geben Preissteigerungen an Konsumenten weiter.",
        },
        "Fed Policy": {
            "severity": "MINIMAL",
            "mechanism": "Relativ zinsunsensitiv. Leichter Profit bei Zinssenkungen als Bond-Proxy, aber Effekt schwaecher als XLU. Defensivste Position im Portfolio.",
        },
    },
}

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
    "RUHIG":    {"min_minutes": 2, "max_minutes": 3},
    "NORMAL":   {"min_minutes": 3, "max_minutes": 4},
    "CRITICAL": {"min_minutes": 4, "max_minutes": 5},
}

def get_anchor_type(composite_tactical, has_breaking_news_high, has_emergency, has_regime_shift):
    if composite_tactical < 50 or has_emergency or has_regime_shift or has_breaking_news_high:
        return "CRITICAL"
    if composite_tactical <= 70:
        return "NORMAL"
    return "RUHIG"

# ---------------------------------------------------------------------------
# Newsletter Format by Weekday (Spec §9.1)
# ---------------------------------------------------------------------------
NEWSLETTER_FORMATS = {
    0: "DAILY", 1: "DAILY", 2: "DAILY", 3: "DAILY",
    4: "DAILY_CONTRARIAN", 5: "WOCHENRUECKBLICK", 6: "WOCHENVORSCHAU",
}

TELEGRAM_MAX_MESSAGE_LENGTH = 4096

RISK_FACTORS = ["Credit Spreads", "Oil/Iran", "Fed Policy"]

RISK_HEATMAP_MAPPING = {
    "HYG": {"Credit Spreads": "DIREKT", "Oil/Iran": "INDIREKT", "Fed Policy": "DIREKT"},
    "DBC": {"Credit Spreads": "INDIREKT", "Oil/Iran": "DIREKT", "Fed Policy": "INDIREKT"},
    "GLD": {"Credit Spreads": "MINIMAL", "Oil/Iran": "SAFE_HAVEN", "Fed Policy": "MINIMAL"},
    "XLU": {"Credit Spreads": "INDIREKT", "Oil/Iran": "MINIMAL", "Fed Policy": "INDIREKT"},
    "XLP": {"Credit Spreads": "MINIMAL", "Oil/Iran": "MINIMAL", "Fed Policy": "MINIMAL"},
}

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(_MODULE_DIR, "data", "history")
COMPOSITE_HISTORY_FILE = os.path.join(HISTORY_DIR, "composite_history.json")
PREDICTION_LOG_FILE = os.path.join(HISTORY_DIR, "prediction_log.json")
INDICATOR_HISTORY_FILE = os.path.join(HISTORY_DIR, "indicator_history.json")
DRIVE_NEWSLETTER_FOLDER = "HISTORY/newsletter"

IDEMPOTENCY_FLAG_FILE = os.path.join(HISTORY_DIR, "last_run_date.txt")
