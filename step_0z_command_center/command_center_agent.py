#!/usr/bin/env python3
"""
command_center_agent.py — System Command Center V1.2
=====================================================
Baldur Creek Capital | Circle 18 | System Command Center

Der Knotenpunkt der alle 9 Systeme gleichzeitig liest und gegeneinander abgleicht.
Beantwortet: "Was muss ich morgen anders machen als heute?"
An 95% der Tage: Nichts.

ETAPPE A: Daten-Layer (10 deterministische Berechnungen, kein LLM)
  1.  FMP Econ Calendar
  2.  Portfolio Daily P&L + YTD
  3.  Surprise-Berechnung
  4.  Cross-Asset Divergenz (5 Ratio-Paare + VIX Proxy + Cu/Au)
  5.  Liquiditätsindikator (FRED: Fed BS - TGA - RRP)
  6.  Alignment Matrix (6 Systeme)
  7.  Converging Timelines
  8.  Volatilitäts-Kompression (informational only, kein Trigger)
  9.  Surprise-Decay-Timer
  10. Regret-Matrix
  + Markt-Reaktion (Absorbed/Rejected)

V1.1 Änderungen (V152 Backtest V3.1 kalibriert):
  - FMP Endpoint /stable/economic-calendar + FMP_API_KEY
  - Sheet absteigend: neueste=Row 2, reverse() + leere Zeilen Skip
  - YTD: erster Handelstag MIT Preisen (nicht 1.Jan)
  - Divergenz-Paare NEU: DBC/SPY, VGK/SPY, DBC/TLT, TIP/SPY, XLF/SPY
  - Z-Schwellen kalibriert + richtungsspezifisch (Backtest V3.1)
  - Multi-Signal: 2+=WARNING, 3+=CRITICAL
  - Liquiditäts-Kombi: Liq Z<-1.5 + Paar extrem = WARNING
  - Vol-Kompression: kein Trigger (Backtest widerlegt)
  - VIX Z>+2.0 Bestätigung, Korr>-0.2 WATCH
  - Cu/Au Z<-1.5 Bearish (war -2.0)

Spec: MACRO_EVENTS_SPEC_TEIL1-6.md (Name: Command Center)
Quelle der Wahrheit: V152_SYSTEMSTATUSANALYSE.md, Backtest V3.1

Usage:
  # GitHub Actions (Daily):
  python -m step_0z_command_center.command_center_agent --mode daily

  # Colab Test (kein Write, kein Telegram):
  python -m step_0z_command_center.command_center_agent --mode daily --skip-write --skip-telegram

  # Force Intelligence Layer (Etappe B):
  python -m step_0z_command_center.command_center_agent --mode daily --force-intelligence
"""
import os
import sys
import json
import math
import time
import argparse
import traceback
from datetime import datetime, timezone, timedelta

# ═══════════════════════════════════════════════════════════════
# CONFIG — COMMAND CENTER KONSTANTEN (inline, nicht in config.py)
# Spec: MACRO_EVENTS_SPEC_TEIL6 §44
# Quelle der Wahrheit für Schwellenwerte: Backtest V3.1
# ═══════════════════════════════════════════════════════════════

# ── Version ──
CC_VERSION = "1.2"

# ── Paths ──
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CC_DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CC_DAILY_OUTPUT = os.path.join(CC_DATA_DIR, "command_center.json")
CC_WEEKLY_OUTPUT = os.path.join(CC_DATA_DIR, "command_center_weekly.json")
CC_DAILY_HISTORY_DIR = os.path.join(CC_DATA_DIR, "daily_history")
CC_NARRATIVE_HISTORY = os.path.join(CC_DATA_DIR, "narrative_history.json")
CC_SLOW_BURN_HISTORY = os.path.join(CC_DATA_DIR, "slow_burn_history.json")
CC_LOOP_HISTORY = os.path.join(CC_DATA_DIR, "feedback_loop_history.json")

# ── FMP API (V1.1 Fix #1 + #2) ──
FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_CALENDAR_ENDPOINT = "/economic-calendar"

# ── FRED API ──
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = {
    "fed_bs": "WALCL",
    "tga": "WTREGEN",
    "rrp": "RRPONTSYD",
    "vix": "VIXCLS",
}
FRED_LOOKBACK_DAYS = 750  # ~2 Jahre, bestätigt via Colab Test

# ── Sheets ──
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"
V16_PRICES_TAB = "DATA_Prices"
V16_K16_K17_TAB = "DATA_K16_K17"

# ── latest.json (V16 Gewichte) ──
LATEST_JSON_PATH = os.path.join(REPO_ROOT, "data", "dashboard", "latest.json")

# ── Event Impact Tiers (Spec TEIL2 §6.3) ──
EVENT_TIER_A = [
    "FOMC", "Fed Funds Rate", "ECB Interest Rate", "ECB Rate",
    "Non-Farm Payrolls", "NFP", "Nonfarm Payrolls",
    "CPI (YoY)", "CPI (MoM)", "Consumer Price Index",
]
EVENT_TIER_B = [
    "PPI", "Producer Price Index", "FOMC Minutes",
    "BOJ Interest Rate", "BOJ Rate", "BOE Interest Rate",
    "GDP", "Gross Domestic Product",
    "PCE", "Personal Consumption",
    "Retail Sales", "ISM Manufacturing",
]
EVENT_TIER_C = [
    "Initial Jobless Claims", "Jobless Claims",
    "Consumer Confidence", "Michigan Consumer",
    "Housing Starts", "Building Permits",
    "Industrial Production", "Durable Goods",
    "ISM Services", "ISM Non-Manufacturing",
    "PMI", "Purchasing Managers",
    "Trade Balance",
]
EVENT_TIER_SCORES = {"A": 10, "B": 7, "C": 4, "D": 2, "E": 1}

# ── Geo-Faktoren (Spec TEIL2 §6.3) ──
GEO_FACTORS = {
    "US": 1.0, "United States": 1.0,
    "EU": 0.7, "Euro Area": 0.7, "Eurozone": 0.7, "Germany": 0.7, "France": 0.7,
    "JP": 0.5, "Japan": 0.5,
    "CN": 0.5, "China": 0.5,
    "GB": 0.3, "United Kingdom": 0.3,
    "CA": 0.2, "Canada": 0.2,
    "AU": 0.2, "Australia": 0.2,
}

# ── Event-Typ-Klassifikation (Spec TEIL2 §8.1) ──
INFLATION_EVENTS = [
    "CPI", "PPI", "PCE", "Core CPI", "Core PPI", "Core PCE",
    "Import Prices", "Export Prices", "Wage Growth", "Average Hourly Earnings",
]
GROWTH_EVENTS = [
    "NFP", "Non-Farm", "Nonfarm", "GDP", "ISM", "PMI", "Retail Sales",
    "Industrial Production", "Durable Goods", "Housing Starts",
    "Consumer Confidence", "Michigan Consumer", "Employment", "Unemployment",
]
RATE_EVENTS = [
    "FOMC", "Fed Funds", "ECB", "BOJ", "BOE", "Interest Rate", "Rate Decision",
]

# ── Surprise-Schwellen (Spec TEIL2 §8.2) ──
SURPRISE_TRIGGER_PCT_TIER_AB = 0.15
SURPRISE_TRIGGER_PCT_TIER_A_ONLY = 0.05

# ═══════════════════════════════════════════════════════════════
# DIVERGENZ-PAARE — V1.1 KALIBRIERT (Backtest V3.1 Top 5)
# V152 Regeln 70-71: Richtungsspezifische Trigger
#
# trigger_direction: "positive" = Z>threshold triggert
#                    "negative" = Z<-threshold triggert
# trigger_z:        kalibrierte Schwelle (immer positiv angegeben)
# forward_window:   Forward-Window für Timing-Kontext
# backtest_stats:   Referenz aus V3.1 für Regret-Matrix
# ═══════════════════════════════════════════════════════════════

DIVERGENCE_PAIRS = [
    {
        "a": "DBC", "b": "SPY",
        "name": "Commod vs Equity",
        "category": "Macro",
        "trigger_direction": "positive",   # Z > +2.0 = Commodities outperformen = Inflations-Stress
        "trigger_z": 2.0,
        "forward_window": 63,
        "backtest_stats": {
            "train_dd5_pct": 75.0, "validate_dd5_pct": 40.0,
            "excess_pct": 45.2, "ci90": "50-100%",
            "lead_days": 80, "dd_median_day": 41,
            "recovery_median_days": 34,
        },
    },
    {
        "a": "VGK", "b": "SPY",
        "name": "Europe vs US",
        "category": "Equity",
        "trigger_direction": "negative",   # Z < -3.0 = Europa crasht vs US = Risk-Off global
        "trigger_z": 3.0,
        "forward_window": 63,
        "backtest_stats": {
            "train_dd5_pct": 83.3, "validate_dd5_pct": 60.0,
            "excess_pct": 53.6, "ci90": "50-100%",
            "lead_days": 76, "dd_median_day": 51,
            "recovery_median_days": 55,
        },
    },
    {
        "a": "DBC", "b": "TLT",
        "name": "Commod vs Bonds",
        "category": "Macro",
        "trigger_direction": "negative",   # Z < -2.0 = Commodities underperformen vs Bonds = Deflationsschock
        "trigger_z": 2.0,
        "forward_window": 21,              # Schnellstes Signal: DD Median Tag 19
        "backtest_stats": {
            "train_dd5_pct": 54.5, "validate_dd5_pct": 25.0,
            "excess_pct": 38.8, "ci90": "36-82%",
            "lead_days": 65, "dd_median_day": 19,
            "recovery_median_days": 26,
        },
    },
    {
        "a": "TIP", "b": "SPY",
        "name": "RealRates vs Equity",
        "category": "Bonds",
        "trigger_direction": "positive",   # Z > +2.0 = TIPS outperformen = Inflationserwartungen eskalieren
        "trigger_z": 2.0,
        "forward_window": 63,
        "backtest_stats": {
            "train_dd5_pct": 66.7, "validate_dd5_pct": 75.0,
            "excess_pct": 36.9, "ci90": "44-89%",
            "lead_days": None, "dd_median_day": 32,
            "recovery_median_days": 27,
        },
    },
    {
        "a": "XLF", "b": "SPY",
        "name": "Financials Stress",
        "category": "Equity",
        "trigger_direction": "negative",   # Z < -2.5 = Financials brechen ein = Credit Stress
        "trigger_z": 2.5,
        "forward_window": 42,
        "backtest_stats": {
            "train_dd5_pct": 61.5, "validate_dd5_pct": 50.0,
            "excess_pct": 36.4, "ci90": "38-85%",
            "lead_days": 66, "dd_median_day": 20,
            "recovery_median_days": 30,
        },
    },
]

DIVERGENCE_LOOKBACK_DAYS = 252

# ── Cu/Au Ratio (V1.1 Fix #11: Z<-1.5 Bearish, Z>+1.5 Bullish) ──
CU_AU_BEARISH_Z = -1.5
CU_AU_BULLISH_Z = 1.5

# ── VIX Proxy (V1.1 Fix #10) ──
VIX_Z_CONFIRMATION = 2.0        # Z>+2.0 = Bestätigungssignal (kein eigenständiger Trigger)
VIX_CORR_WATCH_THRESHOLD = -0.2 # Korrelation > -0.2 = WATCH (V1.0 war -0.3)

# ── Liquiditäts-Kombi (V1.1 Fix #8, V152 Regel 73) ──
LIQ_Z_WARNING = -1.5   # Liq Z < -1.5 + Paar extrem = WARNING
LIQ_Z_CRITICAL = -2.5  # Liq Z < -2.5 + Paar extrem = CRITICAL

# ── Multi-Signal (V1.1 Fix #7, V152 Regel 72) ──
MULTI_SIGNAL_WARNING = 2   # 2+ Paare auf bestätigtem Extrem = WARNING
MULTI_SIGNAL_CRITICAL = 3  # 3+ Paare = CRITICAL (46.2% DD bei 63d)

# ── Alignment (Spec TEIL1 §4.2) ──
ALIGNMENT_DROP_TRIGGER = 0.20
V16_BULLISH_STATES = [
    "STEADY_GROWTH", "FRAGILE_EXPANSION", "REFLATION", "EARLY_RECOVERY",
]
V16_NEUTRAL_STATES = [
    "NEUTRAL", "LATE_EXPANSION", "FULL_EXPANSION", "SOFT_LANDING",
]
V16_BEARISH_STATES = [
    "STRESS_ELEVATED", "CONTRACTION", "DEEP_CONTRACTION", "FINANCIAL_CRISIS",
]

# ── Vol Compression (V1.1 Fix #9: informational only, KEIN Trigger) ──
# VOL_COMPRESSION_TRIGGER entfernt — Backtest widerlegt (V152 Regel 69)

# ── Timeline Konvergenz (Spec TEIL2 §12) ──
TIMELINE_CONVERGENCE_TRIGGER = 3
TIMELINE_WINDOW_DAILY = 14

# ── Decay Halbwertszeiten (Spec TEIL2 §14.2) ──
DECAY_HALFLIFE = {
    "FOMC": 21, "ECB": 21, "BOJ": 21, "BOE": 21,
    "FOMC Minutes": 14,
    "CPI": 7, "PPI": 7, "PCE": 7,
    "NFP": 7, "Employment": 7,
    "GDP": 30,
    "ISM": 14, "PMI": 14,
    "Retail Sales": 7,
    "Jobless Claims": 3,
    "Emergency": 30, "Circuit Breaker": 60,
    "DEFAULT": 7,
}
DECAY_VISIBILITY_THRESHOLD = 0.10

# ── Regret-Matrix Base Rates (V1.1: aktualisiert für neue Paare aus V3.1) ──
BASE_RATES = {
    "DBC_SPY_EXTREME": {
        "probability": 0.75, "expected_loss": -0.06, "timeframe_days": 63,
        "description": "Commod outperform → Inflations-Stress",
    },
    "VGK_SPY_EXTREME": {
        "probability": 0.83, "expected_loss": -0.08, "timeframe_days": 63,
        "description": "Europa crasht → Global Risk-Off",
    },
    "DBC_TLT_EXTREME": {
        "probability": 0.55, "expected_loss": -0.05, "timeframe_days": 21,
        "description": "Deflationsschock — schnellstes Signal",
    },
    "TIP_SPY_EXTREME": {
        "probability": 0.67, "expected_loss": -0.06, "timeframe_days": 63,
        "description": "Inflationserwartungen eskalieren",
    },
    "XLF_SPY_EXTREME": {
        "probability": 0.62, "expected_loss": -0.07, "timeframe_days": 42,
        "description": "Financials Stress → Credit Contagion",
    },
    "CU_AU_EXTREME": {
        "probability": 0.55, "expected_loss": -0.12, "timeframe_days": 120,
        "description": "Wachstum vs. Angst",
    },
    "MULTI_SIGNAL_3PLUS": {
        "probability": 0.46, "expected_loss": -0.068, "timeframe_days": 63,
        "description": "3+ Paare extrem (V3.1: 46.2% DD, AvgDD -6.77%)",
    },
    "LIQ_KOMBI_DBC_TLT": {
        "probability": 0.56, "expected_loss": -0.06, "timeframe_days": 63,
        "description": "Liq Contraction + DBC/TLT extrem (stärkstes Kombi)",
    },
    "TIMELINE_CONVERGENCE": {
        "probability": 0.40, "expected_loss": -0.03, "timeframe_days": 14,
        "description": "3+ Zeitlinien konvergieren",
    },
}
EXPECTED_ANNUAL_CARRY = {
    "HYG": 0.055, "LQD": 0.04, "TLT": 0.035, "SPY": 0.02,
    "GLD": 0.0, "DBC": 0.0, "IWM": 0.015, "XLF": 0.02,
    "XLP": 0.02, "XLU": 0.03, "DEFAULT": 0.02,
}

# ── System-Input-Pfade (Spec TEIL1 §3.1) ──
SYSTEM_INPUTS = {
    "cycles_transition": "step_0v_cycles/data/transition_engine.json",
    "theses": "step_0x_theses/data/theses.json",
    "secular_trends": "step_0w_secular/data/secular_trends.json",
    "disruptions": "data/disruptions/disruptions_history.json",
    "ic_beliefs": "step_0i_ic_pipeline/data/history/beliefs.json",
    "crypto_state": "step_0y_crypto/data/crypto_state.json",
    "ratio_context": "step_0x_theses/data/ratio_context.json",
}

# ── Länder-Filter (Spec TEIL2 §6.3) ──
CALENDAR_COUNTRIES = ["US", "United States", "EU", "Euro Area", "Eurozone",
                       "Germany", "France", "GB", "United Kingdom",
                       "JP", "Japan", "CN", "China", "CA", "Canada",
                       "AU", "Australia"]

# ── LLM Config (Etappe B: Intelligence Layer, Spec TEIL3 §19) ──
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 16000
LLM_TEMPERATURE = 0.2


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def log(msg):
    print(f"  [CC] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_json_safe(path):
    """Lade JSON mit Graceful Degradation."""
    try:
        full = os.path.join(REPO_ROOT, path) if not os.path.isabs(path) else path
        if not os.path.exists(full):
            log(f"JSON nicht gefunden: {path}")
            return None
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"JSON Fehler {path}: {e}")
        return None


def safe_float(val, default=None):
    """Konvertiere zu float, europäisches Format berücksichtigen."""
    if val is None or val == "" or val == ".":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        try:
            return float(str(val).replace(".", "").replace(",", "."))
        except (ValueError, TypeError):
            return default


def get_event_tier(event_name):
    """Bestimme Impact-Tier für ein Event (Spec TEIL2 §6.3)."""
    name_upper = (event_name or "").upper()
    for kw in EVENT_TIER_A:
        if kw.upper() in name_upper:
            return "A"
    for kw in EVENT_TIER_B:
        if kw.upper() in name_upper:
            return "B"
    for kw in EVENT_TIER_C:
        if kw.upper() in name_upper:
            return "C"
    return "D"


def get_geo_factor(country):
    """Geo-Exposure-Faktor (Spec TEIL2 §6.3)."""
    return GEO_FACTORS.get(country, 0.1)


def classify_event_type(event_name):
    """Klassifiziere Event als inflation/growth/rate (Spec TEIL2 §8.1)."""
    name_upper = (event_name or "").upper()
    for kw in INFLATION_EVENTS:
        if kw.upper() in name_upper:
            return "inflation"
    for kw in GROWTH_EVENTS:
        if kw.upper() in name_upper:
            return "growth"
    for kw in RATE_EVENTS:
        if kw.upper() in name_upper:
            return "rate"
    return "other"


def get_decay_halflife(event_name):
    """Halbwertszeit für ein Event (Spec TEIL2 §14.2)."""
    name_upper = (event_name or "").upper()
    for kw, hl in DECAY_HALFLIFE.items():
        if kw == "DEFAULT":
            continue
        if kw.upper() in name_upper:
            return hl
    return DECAY_HALFLIFE["DEFAULT"]


def find_first_data_row(rows, col_idx=0):
    """Finde erste Zeile mit nicht-leerem Wert in gegebener Spalte.
    V1.1 Fix #3: Sheet ist absteigend, Row 2 kann leer sein (Runner noch nicht gelaufen).
    """
    for i, row in enumerate(rows):
        if row and len(row) > col_idx:
            val = row[col_idx].strip() if isinstance(row[col_idx], str) else str(row[col_idx]).strip()
            if val and val != "":
                return i
    return 0


NOW = datetime.now(timezone.utc)
TODAY_STR = NOW.strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════
# GCP AUTH
# ═══════════════════════════════════════════════════════════════

def get_gspread_client():
    """GCP Auth — gleiche Logik wie yield_router.py."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        log("gspread/google-auth nicht installiert")
        return None

    creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
    if not creds_json:
        log("GOOGLE_CREDENTIALS nicht gesetzt")
        return None

    try:
        import json as _json
        creds_dict = _json.loads(creds_json)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        log(f"GCP Auth Fehler: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 1: FMP ECON CALENDAR (Spec TEIL2 §6)
# V1.1 Fix #1: /stable/economic-calendar
# V1.1 Fix #2: FMP_API_KEY (nicht EODHD)
# ═══════════════════════════════════════════════════════════════

def fetch_fmp_calendar():
    """Fetch Economic Calendar von FMP (21-Tage-Fenster)."""
    import requests

    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        log("FMP_API_KEY nicht gesetzt — Calendar leer")
        return {"yesterday": [], "today": [], "this_week": [], "next_week": [],
                "cluster_flags": [], "total_events": 0, "high_impact_today": 0}

    date_from = (NOW - timedelta(days=7)).strftime("%Y-%m-%d")
    date_to = (NOW + timedelta(days=14)).strftime("%Y-%m-%d")

    url = f"{FMP_BASE_URL}{FMP_CALENDAR_ENDPOINT}"
    params = {"from": date_from, "to": date_to, "apikey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        raw_events = resp.json()
        log(f"FMP Calendar: {len(raw_events)} raw Events")
    except Exception as e:
        log(f"FMP Calendar Fehler: {e}")
        return {"yesterday": [], "today": [], "this_week": [], "next_week": [],
                "cluster_flags": [], "total_events": 0, "high_impact_today": 0}

    yesterday_str = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
    today_end = NOW.date()
    week_end = today_end + timedelta(days=7)
    next_week_end = today_end + timedelta(days=14)

    yesterday, today_events, this_week, next_week = [], [], [], []
    total_kept = 0

    for ev in raw_events:
        country = ev.get("country", "")
        impact = ev.get("impact", "")
        event_name = ev.get("event", "")
        ev_date_str = (ev.get("date") or "")[:10]

        # Länder-Filter
        if not any(c.lower() in country.lower() for c in CALENDAR_COUNTRIES):
            continue

        # Impact-Filter: Low-Impact nur für US/EU behalten
        if impact == "Low" and not any(
            c.lower() in country.lower() for c in ["US", "United States", "EU", "Euro Area"]
        ):
            continue

        tier = get_event_tier(event_name)
        geo = get_geo_factor(country)
        impact_score = round(EVENT_TIER_SCORES.get(tier, 2) * geo, 1)

        processed = {
            "event": event_name,
            "country": country,
            "date": ev_date_str,
            "previous": safe_float(ev.get("previous")),
            "consensus": safe_float(ev.get("estimate")),
            "actual": safe_float(ev.get("actual")),
            "impact_score": impact_score,
            "tier": tier,
            "decay_halflife_days": get_decay_halflife(event_name),
        }

        total_kept += 1
        if ev_date_str == yesterday_str:
            yesterday.append(processed)
        elif ev_date_str == TODAY_STR:
            today_events.append(processed)
        else:
            try:
                ev_date = datetime.strptime(ev_date_str, "%Y-%m-%d").date()
                if today_end < ev_date <= week_end:
                    this_week.append(processed)
                elif week_end < ev_date <= next_week_end:
                    next_week.append(processed)
            except ValueError:
                pass

    # Cluster-Erkennung (Spec TEIL2 §6.4)
    cluster_flags = []
    week_impact = sum(e["impact_score"] for e in today_events + this_week)
    n_tier_a = sum(1 for e in today_events + this_week if e["tier"] == "A")
    n_tier_b = sum(1 for e in today_events + this_week if e["tier"] == "B")
    if week_impact > 30 or n_tier_a >= 2 or (n_tier_a >= 1 and n_tier_b >= 3):
        cluster_flags.append("HEAVY_WEEK")
    today_high = [e for e in today_events if e["impact_score"] >= 4]
    if len(today_high) >= 3:
        cluster_flags.append("HEAVY_DAY")

    log(f"Calendar: {len(yesterday)} gestern, {len(today_events)} heute, "
        f"{len(this_week)} Woche, {len(next_week)} nächste Woche")

    return {
        "yesterday": sorted(yesterday, key=lambda e: e["impact_score"], reverse=True),
        "today": sorted(today_events, key=lambda e: e["impact_score"], reverse=True),
        "this_week": sorted(this_week, key=lambda e: e["impact_score"], reverse=True),
        "next_week": sorted(next_week, key=lambda e: e["impact_score"], reverse=True),
        "cluster_flags": cluster_flags,
        "total_events": total_kept,
        "high_impact_today": len(today_high),
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 2: PORTFOLIO DAILY P&L + YTD (Spec TEIL2 §7)
# V1.1 Fix #3: Sheet absteigend + leere Zeilen Skip
# V1.1 Fix #4: YTD erster Handelstag MIT Preisen
# ═══════════════════════════════════════════════════════════════

def fetch_sheet_prices(gc):
    """Lese DATA_Prices Tab — Sheet ist ABSTEIGEND (neueste=Row 2).
    V1.1: reverse + leere Zeilen Skip + YTD Fix.
    """
    if not gc:
        return None
    try:
        sh = gc.open_by_key(V16_SHEET_ID)
        ws = sh.worksheet(V16_PRICES_TAB)
        all_vals = ws.get_all_values()
        if len(all_vals) < 3:
            log("DATA_Prices: zu wenige Zeilen")
            return None
        header = all_vals[0]
        data_rows = all_vals[1:]  # Ohne Header

        # Sheet ist ABSTEIGEND: Row 2 = neueste, letzte Zeile = älteste
        # Row 2 kann leer sein (Runner noch nicht gelaufen)
        first_data_idx = find_first_data_row(data_rows, col_idx=0)
        today_row = data_rows[first_data_idx]
        # Gestern = nächste nicht-leere Zeile nach heute
        prev_row = None
        for i in range(first_data_idx + 1, len(data_rows)):
            if data_rows[i] and data_rows[i][0].strip():
                prev_row = data_rows[i]
                break
        if prev_row is None:
            log("DATA_Prices: keine Vortags-Zeile gefunden")
            return None

        # YTD Start: V1.1 Fix #4 — erster Handelstag MIT Preisen im aktuellen Jahr
        # Sheet ist absteigend → wir suchen von oben nach unten die LETZTE Zeile
        # im aktuellen Jahr (= der früheste Tag des Jahres)
        # Dann prüfen wir ob dieser Tag tatsächlich Preise hat
        current_year = str(NOW.year)
        ytd_row = None
        # Suche von unten (älteste) nach oben die erste Zeile im aktuellen Jahr
        for i in range(len(data_rows) - 1, -1, -1):
            row = data_rows[i]
            if row and row[0].strip().startswith(current_year):
                # Prüfe ob mindestens ein Preis vorhanden
                has_price = False
                for val in row[1:6]:  # Prüfe erste 5 Preis-Spalten
                    if safe_float(val) is not None and safe_float(val) > 0:
                        has_price = True
                        break
                if has_price:
                    ytd_row = row
                    break

        log(f"DATA_Prices: {len(all_vals)} Zeilen, Header: {header[:5]}...")
        log(f"  Today: {today_row[0] if today_row else '?'}, "
            f"Prev: {prev_row[0] if prev_row else '?'}, "
            f"YTD Start: {ytd_row[0] if ytd_row else '?'}")
        return {"header": header, "today": today_row, "yesterday": prev_row,
                "ytd_start": ytd_row}
    except Exception as e:
        log(f"DATA_Prices Fehler: {e}")
        return None


def load_v16_weights():
    """Lade V16 Allokationsgewichte aus latest.json."""
    data = load_json_safe(LATEST_JSON_PATH)
    if not data:
        log("latest.json nicht gefunden — P&L ohne Gewichte")
        return {}
    weights = data.get("v16", {}).get("current_weights", {})
    # Nur Ticker mit Gewicht > 0
    active = {k: v for k, v in weights.items() if v and v > 0.001}
    if active:
        log(f"V16 Gewichte: {len(active)} Positionen — "
            + ", ".join(f"{k} {v:.1%}" for k, v in
                        sorted(active.items(), key=lambda x: -x[1])[:5]))
    else:
        log("V16 Gewichte: keine aktiven Positionen gefunden")
    return active


def compute_portfolio_pnl(sheet_data, weights):
    """Berechne Portfolio Daily P&L + YTD (Spec TEIL2 §7.2)."""
    if not sheet_data or not weights:
        return {
            "daily_return_pct": 0, "ytd_return_pct": 0, "daily_return_eur": 0,
            "top_3_contributors": [], "top_3_detractors": [],
            "prices_date": TODAY_STR, "ytd_start_date": None,
            "n_assets_tracked": 0, "error": "Daten nicht verfügbar",
        }

    header = sheet_data["header"]
    today_row = sheet_data["today"]
    prev_row = sheet_data["yesterday"]
    ytd_row = sheet_data.get("ytd_start")

    # Header → Index Mapping
    col_map = {}
    for i, h in enumerate(header):
        h_clean = h.strip().upper()
        col_map[h_clean] = i

    # Daily Returns
    contributions = []
    portfolio_daily = 0.0
    n_tracked = 0

    for ticker, weight in weights.items():
        t_upper = ticker.upper()
        idx = col_map.get(t_upper)
        if idx is None:
            continue
        p_today = safe_float(today_row[idx]) if idx < len(today_row) else None
        p_prev = safe_float(prev_row[idx]) if idx < len(prev_row) else None
        if p_today and p_prev and p_prev > 0:
            daily_ret = (p_today - p_prev) / p_prev
            contribution = weight * daily_ret
            portfolio_daily += contribution
            contributions.append({
                "ticker": ticker, "weight": round(weight, 4),
                "return": round(daily_ret * 100, 2),
                "contribution": round(contribution * 100, 4),
            })
            n_tracked += 1

    # YTD Returns
    portfolio_ytd = 0.0
    ytd_start_date = None
    if ytd_row:
        ytd_start_date = ytd_row[0] if ytd_row else None
        for ticker, weight in weights.items():
            t_upper = ticker.upper()
            idx = col_map.get(t_upper)
            if idx is None:
                continue
            p_today = safe_float(today_row[idx]) if idx < len(today_row) else None
            p_ytd = safe_float(ytd_row[idx]) if idx < len(ytd_row) else None
            if p_today and p_ytd and p_ytd > 0:
                ytd_ret = (p_today - p_ytd) / p_ytd
                portfolio_ytd += weight * ytd_ret

    # Sortiere Contributors
    sorted_contrib = sorted(contributions, key=lambda c: c["contribution"], reverse=True)
    top_3 = sorted_contrib[:3]
    bottom_3 = sorted(contributions, key=lambda c: c["contribution"])[:3]

    prices_date = today_row[0] if today_row else TODAY_STR
    total_capital = 10000  # Default
    daily_eur = round(total_capital * portfolio_daily, 2)

    return {
        "daily_return_pct": round(portfolio_daily * 100, 2),
        "ytd_return_pct": round(portfolio_ytd * 100, 2),
        "daily_return_eur": daily_eur,
        "top_3_contributors": top_3,
        "top_3_detractors": bottom_3,
        "prices_date": prices_date,
        "ytd_start_date": ytd_start_date,
        "n_assets_tracked": n_tracked,
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 3: SURPRISE-FAKTOR (Spec TEIL2 §8)
# ═══════════════════════════════════════════════════════════════

def compute_surprises(calendar):
    """Berechne Surprise-Faktor für gestrige Events."""
    yesterday_surprises = []
    any_trigger = False

    for ev in calendar.get("yesterday", []):
        actual = ev.get("actual")
        consensus = ev.get("consensus")
        if actual is None or consensus is None:
            continue

        surprise_raw = actual - consensus
        surprise_pct = abs(surprise_raw) / abs(consensus) if consensus != 0 else 0

        # Richtung
        ev_type = classify_event_type(ev["event"])
        if abs(surprise_pct) < 0.02:
            direction = "INLINE"
        elif ev_type == "inflation":
            direction = "HOT" if surprise_raw > 0 else "COLD"
        elif ev_type == "growth":
            direction = "STRONG" if surprise_raw > 0 else "WEAK"
        elif ev_type == "rate":
            direction = "HAWKISH" if surprise_raw > 0 else "DOVISH"
        else:
            direction = "ABOVE" if surprise_raw > 0 else "BELOW"

        # Trigger-Check
        tier = ev.get("tier", "D")
        triggers = False
        if tier in ("A", "B") and surprise_pct > SURPRISE_TRIGGER_PCT_TIER_AB:
            triggers = True
        if tier == "A" and surprise_pct > SURPRISE_TRIGGER_PCT_TIER_A_ONLY:
            triggers = True

        if triggers:
            any_trigger = True

        yesterday_surprises.append({
            "event": ev["event"],
            "country": ev.get("country", ""),
            "consensus": consensus,
            "actual": actual,
            "surprise_raw": round(surprise_raw, 4),
            "surprise_pct": round(surprise_pct, 4),
            "direction": direction,
            "impact_score": ev.get("impact_score", 0),
            "triggers_intelligence": triggers,
        })

    return {
        "yesterday_surprises": yesterday_surprises,
        "n_surprises": len(yesterday_surprises),
        "max_surprise_pct": round(max((s["surprise_pct"] for s in yesterday_surprises), default=0), 4),
        "any_trigger": any_trigger,
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 4: CROSS-ASSET DIVERGENZ
# V1.1 Fix #5: Neue Paare (DBC/SPY, VGK/SPY, DBC/TLT, TIP/SPY, XLF/SPY)
# V1.1 Fix #6: Richtungsspezifische Z-Schwellen (Backtest V3.1)
# ═══════════════════════════════════════════════════════════════

def fetch_price_history(gc, tickers, n_days=260):
    """Lese Preishistorie aus DATA_Prices Sheet.
    V1.1 Fix #3: Sheet ist ABSTEIGEND — reverse für chronologische Reihenfolge.
    """
    if not gc:
        return {}
    try:
        sh = gc.open_by_key(V16_SHEET_ID)
        ws = sh.worksheet(V16_PRICES_TAB)
        all_vals = ws.get_all_values()
        header = all_vals[0]

        col_map = {}
        for i, h in enumerate(header):
            col_map[h.strip().upper()] = i

        # Sheet ist ABSTEIGEND: erste Daten-Zeile = neueste
        data_rows = all_vals[1:]
        # Reverse für chronologische Reihenfolge (älteste zuerst)
        data_rows_chrono = list(reversed(data_rows))
        # Letzte n_days Zeilen (neueste am Ende)
        recent = data_rows_chrono[-n_days:] if len(data_rows_chrono) > n_days else data_rows_chrono

        result = {}
        for ticker in tickers:
            idx = col_map.get(ticker.upper())
            if idx is None:
                continue
            prices = []
            for row in recent:
                val = safe_float(row[idx]) if idx < len(row) else None
                if val is not None and val > 0:
                    prices.append(val)
            if prices:
                result[ticker.upper()] = prices

        log(f"Price History: {len(result)} Ticker, ~{len(recent)} Tage")
        return result
    except Exception as e:
        log(f"Price History Fehler: {e}")
        return {}


def fetch_cu_au_ratio(gc, n_days=260):
    """Lese Cu/Au Ratio aus DATA_K16_K17 Tab.
    V1.1 Fix #3: Auch absteigend sortiert → reverse.
    V152 Entscheidung #10: Cu/Au direkt aus DATA_K16_K17 Spalte [1].
    """
    if not gc:
        return []
    try:
        sh = gc.open_by_key(V16_SHEET_ID)
        ws = sh.worksheet(V16_K16_K17_TAB)
        all_vals = ws.get_all_values()
        header = all_vals[0]

        # Cu/Au ist Spalte [1] (V152 Entscheidung #10)
        col_idx = 1
        # Fallback: suche explizit
        for i, h in enumerate(header):
            h_low = h.strip().lower().replace(" ", "_")
            if "cu" in h_low and "au" in h_low:
                col_idx = i
                break
            if "copper" in h_low and "gold" in h_low:
                col_idx = i
                break

        data_rows = all_vals[1:]
        # Absteigend → reverse für chronologische Reihenfolge
        data_rows_chrono = list(reversed(data_rows))
        recent = data_rows_chrono[-n_days:] if len(data_rows_chrono) > n_days else data_rows_chrono
        values = []
        for row in recent:
            val = safe_float(row[col_idx]) if col_idx < len(row) else None
            if val is not None and val > 0:
                values.append(val)

        log(f"Cu/Au Ratio: {len(values)} Datenpunkte aus K16_K17")
        return values
    except Exception as e:
        log(f"Cu/Au Ratio Fehler: {e}")
        return []


def compute_z_score(values, lookback=252):
    """Z-Score des letzten Wertes vs. Lookback-Fenster."""
    if len(values) < 20:
        return None, None, None
    window = values[-lookback:] if len(values) >= lookback else values
    current = window[-1]
    mean = sum(window) / len(window)
    variance = sum((x - mean) ** 2 for x in window) / len(window)
    std = math.sqrt(variance) if variance > 0 else 0.001
    z = (current - mean) / std
    return round(z, 2), round(mean, 4), round(std, 4)


def compute_log_ratio_series(prices_a, prices_b):
    """Berechne Log-Ratio Zeitreihe (Spec TEIL2 §9.2)."""
    n = min(len(prices_a), len(prices_b))
    if n < 20:
        return []
    ratios = []
    for i in range(n):
        a = prices_a[-(n - i)]
        b = prices_b[-(n - i)]
        if a > 0 and b > 0:
            ratios.append(math.log(a / b))
    return ratios


def compute_divergences(price_history, cu_au_ratios, vix_data, spy_prices):
    """Berechne alle Divergenz-Signale.
    V1.1: Richtungsspezifische Trigger (nur Backtest-bestätigte Seite).
    Gegenrichtung = monitor_only, zählt NICHT für Multi-Signal/Liq-Kombi/Regret.
    """
    pairs_result = []
    n_triggered = 0  # Zähler für Multi-Signal (nur bestätigte Richtung)

    # ── 5 Ratio-Paare (V1.1 kalibriert) ──
    for pair in DIVERGENCE_PAIRS:
        a_prices = price_history.get(pair["a"].upper(), [])
        b_prices = price_history.get(pair["b"].upper(), [])
        log_ratios = compute_log_ratio_series(a_prices, b_prices)

        if len(log_ratios) < 20:
            pairs_result.append({
                "pair": f"{pair['a']}/{pair['b']}", "name": pair["name"],
                "category": pair.get("category", ""),
                "z_score": None, "signal": "UNAVAILABLE",
                "trigger_direction": pair["trigger_direction"],
                "trigger_z": pair["trigger_z"],
                "forward_window": pair["forward_window"],
                "monitor_only": False,
                "interpretation": "Daten nicht verfügbar", "days_at_extreme": 0,
            })
            continue

        z, mean_val, std_val = compute_z_score(log_ratios, DIVERGENCE_LOOKBACK_DAYS)

        # Momentum (21d)
        z_21d_ago = None
        if len(log_ratios) > 21:
            z_21d, _, _ = compute_z_score(log_ratios[:-21], DIVERGENCE_LOOKBACK_DAYS)
            z_21d_ago = z_21d
        z_momentum = round(z - z_21d_ago, 2) if z is not None and z_21d_ago is not None else None

        # ── Richtungsspezifischer Trigger-Check ──
        triggered = False
        monitor_only = False
        if z is not None:
            threshold = pair["trigger_z"]
            if pair["trigger_direction"] == "positive" and z >= threshold:
                triggered = True
            elif pair["trigger_direction"] == "negative" and z <= -threshold:
                triggered = True
            # Gegenrichtung: extrem aber nicht in bestätigter Richtung
            elif abs(z) >= threshold:
                monitor_only = True

        if triggered:
            signal = "EXTREME"
            n_triggered += 1
        elif monitor_only:
            signal = "EXTREME_UNCONFIRMED"
        elif z is not None and abs(z) >= 1.5:
            signal = "ELEVATED"
        elif z is not None and abs(z) >= 1.0:
            signal = "MODERATE"
        else:
            signal = "NORMAL"

        interp = None
        if triggered:
            dir_label = ">" if pair["trigger_direction"] == "positive" else "<"
            interp = (f"{pair['name']} — Z={z:+.2f} ({dir_label}{pair['trigger_z']:+.1f} bestätigt). "
                      f"Backtest: {pair['backtest_stats']['train_dd5_pct']:.0f}% DD-Rate, "
                      f"Lead {pair['backtest_stats'].get('lead_days', '?')}d.")
        elif monitor_only:
            interp = (f"{pair['name']} — Z={z:+.2f} (Gegenrichtung, nicht bestätigt). "
                      f"Monitoring only.")
        elif signal == "ELEVATED":
            interp = f"{pair['name']} — Z={z:+.2f}. Erhöht, noch kein Trigger."

        pairs_result.append({
            "pair": f"{pair['a']}/{pair['b']}", "name": pair["name"],
            "category": pair.get("category", ""),
            "z_score": z, "z_momentum_21d": z_momentum,
            "signal": signal,
            "trigger_direction": pair["trigger_direction"],
            "trigger_z": pair["trigger_z"],
            "forward_window": pair["forward_window"],
            "monitor_only": monitor_only,
            "interpretation": interp, "days_at_extreme": 0,
        })

    # ── Cu/Au Ratio (V1.1 Fix #11: Z<-1.5 Bearish, Z>+1.5 Bullish) ──
    cu_au_signal = "UNAVAILABLE"
    cu_au_z = None
    cu_au_interp = None
    if cu_au_ratios and len(cu_au_ratios) >= 20:
        cu_au_z, _, _ = compute_z_score(cu_au_ratios, DIVERGENCE_LOOKBACK_DAYS)
        if cu_au_z is not None:
            if cu_au_z <= CU_AU_BEARISH_Z:
                cu_au_signal = "BEARISH"
                cu_au_interp = f"Cu/Au Z={cu_au_z:+.2f} — Bearish (Wachstum schwächt sich ab)"
            elif cu_au_z >= CU_AU_BULLISH_Z:
                cu_au_signal = "BULLISH"
                cu_au_interp = f"Cu/Au Z={cu_au_z:+.2f} — Bullish (Wachstumsoptimismus)"
            else:
                cu_au_signal = "NEUTRAL"
    pairs_result.append({
        "pair": "COPPER/GLD", "name": "Wachstum vs. Angst",
        "z_score": cu_au_z, "signal": cu_au_signal,
        "interpretation": cu_au_interp, "days_at_extreme": 0,
        "monitor_only": False,
    })

    # ── VIX Proxy (V1.1 Fix #10) ──
    vix_z = None
    vix_z_signal = "UNAVAILABLE"
    vix_corr_signal = "UNAVAILABLE"
    vix_corr_val = None
    vix_interp = None

    if vix_data and len(vix_data) >= 20:
        vix_z, _, _ = compute_z_score(vix_data, min(252, len(vix_data)))
        if vix_z is not None and vix_z >= VIX_Z_CONFIRMATION:
            vix_z_signal = "CONFIRMATION"
            vix_interp = f"VIX Z={vix_z:+.2f} — Bestätigungssignal (>+{VIX_Z_CONFIRMATION})"
        elif vix_z is not None:
            vix_z_signal = "NORMAL"

    if vix_data and spy_prices and len(vix_data) >= 21 and len(spy_prices) >= 22:
        # SPY 5d returns
        spy_recent = spy_prices[-22:]
        spy_5d_returns = []
        for i in range(5, len(spy_recent)):
            ret = (spy_recent[i] - spy_recent[i - 5]) / spy_recent[i - 5]
            spy_5d_returns.append(ret)

        vix_recent = vix_data[-len(spy_5d_returns):]
        if len(vix_recent) == len(spy_5d_returns) and len(spy_5d_returns) >= 10:
            # Pearson Korrelation
            n = len(spy_5d_returns)
            mean_x = sum(spy_5d_returns) / n
            mean_y = sum(vix_recent) / n
            cov = sum((spy_5d_returns[i] - mean_x) * (vix_recent[i] - mean_y) for i in range(n))
            var_x = sum((x - mean_x) ** 2 for x in spy_5d_returns)
            var_y = sum((y - mean_y) ** 2 for y in vix_recent)
            denom = math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 1
            corr = cov / denom
            vix_corr_val = round(corr, 2)
            # V1.1 Fix #10: Schwelle auf -0.2 (war -0.3)
            if corr > VIX_CORR_WATCH_THRESHOLD:
                vix_corr_signal = "WATCH"
                vix_interp = (vix_interp or "") + f" Korrelation VIX/SPY={corr:+.2f} — WATCH (>{VIX_CORR_WATCH_THRESHOLD})"
            else:
                vix_corr_signal = "NORMAL"

    pairs_result.append({
        "pair": "VIX_PROXY", "name": "VIX Bestätigung + Complacency",
        "vix_z": vix_z, "vix_z_signal": vix_z_signal,
        "correlation_21d": vix_corr_val, "corr_signal": vix_corr_signal,
        "signal": vix_z_signal if vix_z_signal == "CONFIRMATION" else vix_corr_signal,
        "interpretation": vix_interp, "days_at_extreme": 0,
        "monitor_only": True,  # VIX ist IMMER nur Bestätigung, kein eigenständiger Trigger
    })

    # ── Multi-Signal Aggregation (V1.1 Fix #7) ──
    # Nur bestätigte EXTREME zählen (nicht EXTREME_UNCONFIRMED)
    n_extreme_confirmed = n_triggered
    n_elevated = sum(1 for p in pairs_result if p.get("signal") == "ELEVATED")
    n_unconfirmed = sum(1 for p in pairs_result if p.get("signal") == "EXTREME_UNCONFIRMED")
    vix_confirms = vix_z_signal == "CONFIRMATION"
    corr_watch = vix_corr_signal == "WATCH"

    if n_extreme_confirmed >= MULTI_SIGNAL_CRITICAL:
        alert_level = "CRITICAL"
    elif n_extreme_confirmed >= MULTI_SIGNAL_WARNING:
        alert_level = "HIGH"
    elif n_extreme_confirmed >= 1:
        alert_level = "MODERATE"
    elif n_elevated >= 2 or corr_watch:
        alert_level = "ELEVATED"
    else:
        alert_level = "LOW"

    triggers = n_extreme_confirmed >= 1

    return {
        "pairs": pairs_result,
        "alert_level": alert_level,
        "n_extreme_confirmed": n_extreme_confirmed,
        "n_extreme_unconfirmed": n_unconfirmed,
        "n_elevated": n_elevated,
        "vix_confirms": vix_confirms,
        "vix_corr_watch": corr_watch,
        "multi_signal_level": ("CRITICAL" if n_extreme_confirmed >= MULTI_SIGNAL_CRITICAL
                                else "WARNING" if n_extreme_confirmed >= MULTI_SIGNAL_WARNING
                                else "NORMAL"),
        "triggers_intelligence": triggers,
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 5: LIQUIDITÄTSINDIKATOR (Spec TEIL2 §10)
# + V1.1 Fix #8: Liquiditäts-Z-Score für Kombi-Signal
# ═══════════════════════════════════════════════════════════════

def fetch_fred_data():
    """Fetch alle FRED Series (VIX, WALCL, WTREGEN, RRPONTSYD)."""
    import requests

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log("FRED_API_KEY nicht gesetzt")
        return {}

    end_date = NOW.strftime("%Y-%m-%d")
    start_date = (NOW - timedelta(days=FRED_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    fred_data = {}
    for key, series_id in FRED_SERIES.items():
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "asc",
        }
        try:
            resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
            parsed = []
            for obs in observations:
                if obs["value"] != ".":
                    parsed.append({
                        "date": obs["date"],
                        "value": float(obs["value"]),
                    })
            fred_data[key] = parsed
            log(f"FRED {series_id}: {len(parsed)} Datenpunkte")
        except Exception as e:
            log(f"FRED {series_id} Fehler: {e}")
            fred_data[key] = []

    return fred_data


def compute_liquidity(fred_data):
    """Berechne Net Liquidity = Fed BS - TGA - RRP (Spec TEIL2 §10).
    V1.1: Inkl. Z-Score für Liquiditäts-Kombi-Signal.
    """
    fed_bs = fred_data.get("fed_bs", [])
    tga = fred_data.get("tga", [])
    rrp = fred_data.get("rrp", [])

    if not fed_bs or not tga:
        return {"error": "FRED Daten nicht verfügbar", "direction": "UNKNOWN",
                "liq_z_score": None}

    # Aktuelle Werte
    fed_current = fed_bs[-1]["value"] if fed_bs else 0   # Mio USD
    tga_current = tga[-1]["value"] if tga else 0          # Mio USD
    # RRPONTSYD ist in Milliarden!
    rrp_current = rrp[-1]["value"] * 1000 if rrp else 0   # Mrd → Mio USD

    net_liq = fed_current - tga_current - rrp_current
    net_liq_T = round(net_liq / 1_000_000, 2)

    # Veränderungen
    changes = {}
    for label, weeks_back in [("1w", 1), ("4w", 4), ("13w", 13)]:
        idx = -(weeks_back + 1)
        if len(fed_bs) > abs(idx) and len(tga) > abs(idx):
            past_fed = fed_bs[idx]["value"]
            past_tga = tga[idx]["value"]
            past_rrp_idx = min(abs(idx) * 5, len(rrp) - 1) if rrp else 0
            past_rrp = rrp[-(past_rrp_idx + 1)]["value"] * 1000 if rrp and past_rrp_idx < len(rrp) else 0
            past_net = past_fed - past_tga - past_rrp
            change = net_liq - past_net
            changes[label] = round(change / 1000, 1)  # In Milliarden

    # Richtung
    c1w = changes.get("1w", 0)
    c4w = changes.get("4w", 0)
    if c4w > 0 and c1w > 0:
        direction = "EXPANDING"
    elif c4w < 0 and c1w < 0:
        direction = "CONTRACTING"
    elif c4w > 0 and c1w < 0:
        direction = "DECELERATING"
    elif c4w < 0 and c1w > 0:
        direction = "BOTTOMING"
    else:
        direction = "FLAT"

    # V1.1: Liquiditäts-Z-Score (4w-Veränderung, 252d Lookback)
    # Berechne 4w-Change Serie für Z-Score
    liq_z = None
    if len(fed_bs) >= 30 and len(tga) >= 30:
        # Baue Net-Liq Serie (wöchentlich, nur fed_bs + tga, rrp ist täglich)
        net_liq_series = []
        min_len = min(len(fed_bs), len(tga))
        for i in range(min_len):
            nl = fed_bs[i]["value"] - tga[i]["value"]
            if rrp and i < len(rrp):
                nl -= rrp[i]["value"] * 1000
            net_liq_series.append(nl)

        # 4w (20-Punkt) Veränderung
        if len(net_liq_series) >= 25:
            change_series = []
            for i in range(4, len(net_liq_series)):
                if net_liq_series[i - 4] != 0:
                    ch = (net_liq_series[i] - net_liq_series[i - 4]) / abs(net_liq_series[i - 4])
                    change_series.append(ch)
            if len(change_series) >= 20:
                liq_z, _, _ = compute_z_score(change_series, min(252, len(change_series)))
                log(f"Liq Z-Score (4w-Change): {liq_z}")

    # Treiber-Attribution
    main_driver = "unknown"
    if len(fed_bs) > 4 and len(tga) > 4:
        delta_fed = fed_bs[-1]["value"] - fed_bs[-5]["value"]
        delta_tga = -(tga[-1]["value"] - tga[-5]["value"])
        drivers = {"Fed_BS": delta_fed, "TGA": delta_tga}
        main_driver = max(drivers, key=lambda k: abs(drivers[k]))

    data_date = fed_bs[-1]["date"] if fed_bs else TODAY_STR
    lag = (NOW.date() - datetime.strptime(data_date, "%Y-%m-%d").date()).days

    return {
        "net_liquidity_usd_T": net_liq_T,
        "fed_bs_usd_T": round(fed_current / 1_000_000, 2),
        "tga_usd_T": round(tga_current / 1_000_000, 2),
        "rrp_usd_T": round(rrp_current / 1_000_000, 3),
        "change_1w_usd_B": changes.get("1w", 0),
        "change_4w_usd_B": changes.get("4w", 0),
        "change_13w_usd_B": changes.get("13w", 0),
        "direction": direction,
        "main_driver": main_driver,
        "data_date": data_date,
        "data_lag_days": lag,
        "liq_z_score": liq_z,
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 5b: LIQUIDITÄTS-KOMBI (V1.1 Fix #8)
# V152 Regel 73: Liq Z + Paar extrem = verstärkt
# ═══════════════════════════════════════════════════════════════

def compute_liq_kombi(liquidity, divergences):
    """Berechne Liquiditäts-Kombi-Signal (V152 Regel 73).
    Liq Z<-1.5 + mindestens ein Paar auf bestätigtem EXTREME = WARNING
    Liq Z<-2.5 + mindestens ein Paar auf bestätigtem EXTREME = CRITICAL
    Stärkstes Kombi: Liq + DBC/TLT = 55.6% DD-Rate (Backtest V3.1)
    """
    liq_z = liquidity.get("liq_z_score")
    if liq_z is None:
        return {"signal": "UNAVAILABLE", "liq_z": None, "kombi_pairs": [],
                "triggers_intelligence": False}

    # Finde bestätigte EXTREME Paare
    extreme_pairs = []
    for p in divergences.get("pairs", []):
        if p.get("signal") == "EXTREME" and not p.get("monitor_only", False):
            extreme_pairs.append(p.get("pair", ""))

    if not extreme_pairs or liq_z >= LIQ_Z_WARNING:
        return {
            "signal": "NORMAL",
            "liq_z": liq_z,
            "kombi_pairs": [],
            "triggers_intelligence": False,
        }

    if liq_z <= LIQ_Z_CRITICAL:
        signal = "CRITICAL"
    else:
        signal = "WARNING"

    # Interpretation mit stärkstem Paar
    strongest = "DBC/TLT" if "DBC/TLT" in extreme_pairs else extreme_pairs[0]
    interp = (f"Liq Z={liq_z:+.2f} + {', '.join(extreme_pairs)} EXTREME. "
              f"Signal: {signal}. "
              f"Backtest V3.1: Liq+DBC/TLT = 55.6% DD-Rate.")

    return {
        "signal": signal,
        "liq_z": liq_z,
        "kombi_pairs": extreme_pairs,
        "strongest_kombi": strongest,
        "interpretation": interp,
        "triggers_intelligence": True,
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 6: ALIGNMENT MATRIX (Spec TEIL1 §4)
# ═══════════════════════════════════════════════════════════════

def compute_alignment(liquidity, divergences):
    """Berechne Alignment Matrix aus 6 Systemen (Spec TEIL1 §4.2)."""
    directions = {}

    # V16 State — aus latest.json
    latest = load_json_safe(LATEST_JSON_PATH)
    v16_regime = "UNKNOWN"
    if latest:
        v16_regime = latest.get("header", {}).get("v16_regime", "UNKNOWN")
    if v16_regime in V16_BULLISH_STATES:
        directions["V16"] = {"direction": "BULLISH", "detail": v16_regime}
    elif v16_regime in V16_BEARISH_STATES:
        directions["V16"] = {"direction": "BEARISH", "detail": v16_regime}
    else:
        directions["V16"] = {"direction": "NEUTRAL", "detail": v16_regime}

    # Cycles — transition_engine.json
    transition = load_json_safe(SYSTEM_INPUTS["cycles_transition"])
    if transition:
        score = transition.get("confirmation_counter", {}).get("confirmation_score", 0)
        d = "BULLISH" if score > 0.3 else ("BEARISH" if score < -0.3 else "NEUTRAL")
        directions["Cycles"] = {"direction": d, "detail": f"Score {score:+.2f}"}
    else:
        directions["Cycles"] = {"direction": "UNAVAILABLE", "detail": "Daten fehlen"}

    # Thesen
    theses = load_json_safe(SYSTEM_INPUTS["theses"])
    if theses:
        tier1 = [t for t in theses.get("theses", []) if t.get("tier") == 1]
        n_bull = sum(1 for t in tier1 if "BULLISH" in (t.get("direction", "") or "").upper())
        n_bear = sum(1 for t in tier1 if "BEARISH" in (t.get("direction", "") or "").upper())
        d = "BULLISH" if n_bull > n_bear else ("BEARISH" if n_bear > n_bull else "NEUTRAL")
        directions["Thesen"] = {"direction": d, "detail": f"{n_bull}B/{n_bear}Be/{len(tier1) - n_bull - n_bear}N"}
    else:
        directions["Thesen"] = {"direction": "UNAVAILABLE", "detail": "Daten fehlen"}

    # Secular Trends
    secular = load_json_safe(SYSTEM_INPUTS["secular_trends"])
    if secular:
        wa = secular.get("conviction_summary", {}).get("weighted_activation", 0.5)
        d = "BULLISH" if wa > 0.6 else ("BEARISH" if wa < 0.3 else "NEUTRAL")
        directions["Secular"] = {"direction": d, "detail": f"Activation {wa:.2f}"}
    else:
        directions["Secular"] = {"direction": "UNAVAILABLE", "detail": "Daten fehlen"}

    # Crypto
    crypto = load_json_safe(SYSTEM_INPUTS["crypto_state"])
    if crypto:
        ens = crypto.get("ensemble", {}).get("value", 0.5)
        d = "BULLISH" if ens > 0.60 else ("BEARISH" if ens < 0.25 else "NEUTRAL")
        directions["Crypto"] = {"direction": d, "detail": f"Ensemble {ens:.2f}"}
    else:
        directions["Crypto"] = {"direction": "UNAVAILABLE", "detail": "Daten fehlen"}

    # MacroEvents (eigene Indikatoren)
    liq_dir = liquidity.get("direction", "UNKNOWN")
    div_alert = divergences.get("alert_level", "LOW")
    if liq_dir == "EXPANDING" and div_alert == "LOW":
        me_dir = "BULLISH"
    elif liq_dir in ("CONTRACTING",) or div_alert in ("HIGH", "CRITICAL"):
        me_dir = "BEARISH"
    else:
        me_dir = "NEUTRAL"
    directions["MacroEvents"] = {"direction": me_dir, "detail": f"Liq {liq_dir}, Div {div_alert}"}

    # Alignment Score berechnen
    available = {k: v for k, v in directions.items() if v["direction"] != "UNAVAILABLE"}
    n_systems = len(available)
    if n_systems == 0:
        return {"systems": directions, "score": 0, "n_systems": 0,
                "interpretation": "Keine Daten verfügbar"}

    # Dominant Direction
    counts = {"BULLISH": 0, "NEUTRAL": 0, "BEARISH": 0}
    for v in available.values():
        d = v["direction"]
        if d in counts:
            counts[d] += 1
    dominant = max(counts, key=counts.get)
    agreement = counts[dominant]
    score = round(agreement / n_systems, 2)

    if score >= 0.80:
        interp = "HIGH ALIGNMENT"
    elif score >= 0.60:
        interp = "MODERATE ALIGNMENT"
    elif score >= 0.40:
        interp = "LOW ALIGNMENT — KONFLIKT"
    else:
        interp = "EXTREME DIVERGENCE"

    # Konflikt-Beschreibung
    conflict = None
    if score < 0.80:
        dissenters = [k for k, v in available.items() if v["direction"] != dominant]
        if dissenters:
            conflict = f"{', '.join(dissenters)} divergent vs. {dominant} Mehrheit"

    # Alignment-Drop (vs. gestern) — lese Vortag aus History
    score_yesterday = None
    score_change = None
    triggers = False
    yesterday_file = os.path.join(CC_DAILY_HISTORY_DIR,
                                   f"command_center_{(NOW - timedelta(days=1)).strftime('%Y-%m-%d')}.json")
    prev_data = load_json_safe(yesterday_file)
    if prev_data:
        score_yesterday = prev_data.get("alignment", {}).get("score")
        if score_yesterday is not None:
            score_change = round(score - score_yesterday, 2)
            if abs(score_change) > ALIGNMENT_DROP_TRIGGER:
                triggers = True

    return {
        "systems": directions,
        "score": score,
        "score_yesterday": score_yesterday,
        "score_change": score_change,
        "dominant_direction": dominant,
        "agreement_count": agreement,
        "n_systems": n_systems,
        "interpretation": interp,
        "conflict_description": conflict,
        "triggers_intelligence": triggers,
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 7: CONVERGING TIMELINES (Spec TEIL2 §12)
# ═══════════════════════════════════════════════════════════════

def compute_timelines(calendar):
    """Prüfe ob ≥3 Zeitlinien in 14 Tagen konvergieren."""
    active = []
    today = NOW.date()
    window_end = today + timedelta(days=TIMELINE_WINDOW_DAILY)

    # Timeline 1: Monetary Policy (aus Calendar)
    for ev in calendar.get("today", []) + calendar.get("this_week", []) + calendar.get("next_week", []):
        ev_name = (ev.get("event") or "").upper()
        if any(kw in ev_name for kw in ["FOMC", "ECB", "BOJ", "BOE", "RATE DECISION"]):
            active.append({"timeline": "Monetary Policy", "event": ev["event"],
                           "date": ev["date"], "impact": ev.get("impact_score", 7)})
            break  # Nur einmal pro Timeline

    # Timeline 4: Market Structure (statisch)
    for d in range(TIMELINE_WINDOW_DAILY):
        check = today + timedelta(days=d)
        # OPEX: 3. Freitag
        if check.weekday() == 4:
            first = check.replace(day=1)
            first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
            third_fri = first_fri + timedelta(days=14)
            if check == third_fri:
                active.append({"timeline": "Market Structure", "event": "OPEX",
                               "date": check.isoformat(), "impact": 6})
        # Quarter-End
        if check.month in (3, 6, 9, 12) and check.day >= 25:
            active.append({"timeline": "Market Structure", "event": "Quarter-End",
                           "date": check.isoformat(), "impact": 5})
            break

    # Timeline 6: Seasonal (statisch)
    for d in range(TIMELINE_WINDOW_DAILY):
        check = today + timedelta(days=d)
        if check.month == 5 and 1 <= check.day <= 7:
            active.append({"timeline": "Seasonal", "event": "Sell in May Window",
                           "date": check.isoformat(), "impact": 4})
            break
        if check.month in (3, 6, 9, 12):
            month_end = check.replace(day=28) + timedelta(days=4)
            month_end = month_end.replace(day=1) - timedelta(days=1)
            if (month_end - check).days <= 5:
                active.append({"timeline": "Seasonal", "event": "Window Dressing",
                               "date": check.isoformat(), "impact": 3})
                break

    # Timeline 2, 3, 5: Fiscal, Geopolitical, Credit — aus Weekly (Etappe C)
    # Für V1.1 Daily: nur statische + Calendar

    n_timelines = len(set(a["timeline"] for a in active))
    if n_timelines >= TIMELINE_CONVERGENCE_TRIGGER:
        level = "CONVERGENCE_WARNING"
    elif n_timelines == 2:
        level = "CONVERGENCE_WATCH"
    else:
        level = "NORMAL"

    return {
        "window_days": TIMELINE_WINDOW_DAILY,
        "active_timelines": active,
        "n_active": n_timelines,
        "convergence_level": level,
        "triggers_intelligence": n_timelines >= TIMELINE_CONVERGENCE_TRIGGER,
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 8: VOLATILITÄTS-KOMPRESSION (Spec TEIL2 §13)
# V1.1 Fix #9: Kein Trigger (Backtest widerlegt). Informational only.
# ═══════════════════════════════════════════════════════════════

def compute_vol_compression(spy_prices):
    """Berechne Vol-Kompression aus SPY Preisen.
    V1.1: triggers_intelligence ist IMMER False (Backtest widerlegt, V152 Regel 69).
    """
    if not spy_prices or len(spy_prices) < 30:
        return {"signal": "UNAVAILABLE", "compression_score": 0,
                "triggers_intelligence": False, "error": "SPY Daten nicht ausreichend"}

    # Daily Returns
    returns = [(spy_prices[i] - spy_prices[i - 1]) / spy_prices[i - 1]
               for i in range(1, len(spy_prices))]

    # Realisierte Vol (21 Tage, annualisiert)
    recent_21 = returns[-21:] if len(returns) >= 21 else returns
    mean_r = sum(recent_21) / len(recent_21)
    var_r = sum((r - mean_r) ** 2 for r in recent_21) / len(recent_21)
    realized_vol = math.sqrt(var_r) * math.sqrt(252) * 100

    # Perzentil (vs. letzte 252 Tage)
    if len(returns) >= 252:
        all_vols = []
        for i in range(21, len(returns)):
            window = returns[i - 21:i]
            m = sum(window) / len(window)
            v = sum((r - m) ** 2 for r in window) / len(window)
            all_vols.append(math.sqrt(v) * math.sqrt(252) * 100)
        if all_vols:
            rank = sum(1 for v in all_vols if v < realized_vol)
            percentile = round(rank / len(all_vols) * 100)
        else:
            percentile = 50
    else:
        percentile = 50

    # Tage seit >1% Move
    days_since = 0
    for r in reversed(returns):
        if abs(r) > 0.01:
            break
        days_since += 1

    # Bollinger Band Width (20d)
    recent_20_prices = spy_prices[-20:]
    if len(recent_20_prices) == 20:
        sma = sum(recent_20_prices) / 20
        std = math.sqrt(sum((p - sma) ** 2 for p in recent_20_prices) / 20)
        bb_width = (2 * std * 2) / sma * 100 if sma > 0 else 0
    else:
        bb_width = 0
    bb_pctl = 50  # Vereinfacht — vollständiges Perzentil braucht mehr History

    # Compression Score (Spec TEIL2 §13.2) — informational only
    compression_score = round(
        (100 - percentile) * 0.4 +
        min(days_since, 30) * 2 +
        (100 - bb_pctl) * 0.2
    )
    compression_score = max(0, min(100, compression_score))

    if compression_score >= 80:
        signal = "EXTREME_COMPRESSION"
    elif compression_score >= 60:
        signal = "HIGH_COMPRESSION"
    elif compression_score >= 40:
        signal = "MODERATE"
    else:
        signal = "NORMAL"

    interp = None
    if signal in ("EXTREME_COMPRESSION", "HIGH_COMPRESSION"):
        interp = (f"SPY Vol auf {percentile}. Perzentil. "
                  f"{days_since}d ohne >1% Move. Informational only (Backtest widerlegt).")

    return {
        "realized_vol_21d": round(realized_vol, 1),
        "vol_percentile": percentile,
        "days_since_1pct_move": days_since,
        "bb_width": round(bb_width, 1),
        "compression_score": compression_score,
        "signal": signal,
        "interpretation": interp,
        "triggers_intelligence": False,  # V1.1: IMMER False (Backtest widerlegt)
    }


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 9: SURPRISE-DECAY-TIMER (Spec TEIL2 §14)
# ═══════════════════════════════════════════════════════════════

def compute_decay_timer(calendar):
    """Berechne Decay für alle Events der letzten 30 Tage."""
    # Lese vorherige Daily History
    active_events = []
    for i in range(30):
        date_str = (NOW - timedelta(days=i + 1)).strftime("%Y-%m-%d")
        hist_file = os.path.join(CC_DAILY_HISTORY_DIR, f"command_center_{date_str}.json")
        hist = load_json_safe(hist_file)
        if not hist:
            continue
        for ev in hist.get("calendar", {}).get("yesterday", []):
            if ev.get("actual") is not None:
                days_since = i + 1
                halflife = ev.get("decay_halflife_days", DECAY_HALFLIFE["DEFAULT"])
                decay = 0.5 ** (days_since / halflife)
                if decay >= DECAY_VISIBILITY_THRESHOLD:
                    active_events.append({
                        "event": ev["event"],
                        "date": ev.get("date", date_str),
                        "days_since": days_since,
                        "halflife": halflife,
                        "decay_factor": round(decay, 2),
                        "adjusted_impact": round(ev.get("impact_score", 2) * decay, 1),
                        "still_prominent": decay >= 0.3,
                    })

    # Auch heutige gestrige Events hinzufügen
    for ev in calendar.get("yesterday", []):
        if ev.get("actual") is not None:
            active_events.append({
                "event": ev["event"],
                "date": ev.get("date", TODAY_STR),
                "days_since": 0,
                "halflife": ev.get("decay_halflife_days", DECAY_HALFLIFE["DEFAULT"]),
                "decay_factor": 1.0,
                "adjusted_impact": ev.get("impact_score", 2),
                "still_prominent": True,
            })

    return sorted(active_events, key=lambda e: e["adjusted_impact"], reverse=True)[:20]


# ═══════════════════════════════════════════════════════════════
# BERECHNUNG 10: REGRET-MATRIX (Spec TEIL2 §15)
# V1.1: Aktualisiert für neue Paare + Multi-Signal + Liq-Kombi
# ═══════════════════════════════════════════════════════════════

def compute_regret_matrix(divergences, vol_compression, timelines, liq_kombi, weights):
    """Berechne Regret-Matrix für offene Threats.
    V1.1: Neue Paare, Multi-Signal Threat, Liq-Kombi Threat.
    """
    total_capital = 10000  # Default
    active_threats = []

    # ── Einzelne Divergenz-Paare auf bestätigtem EXTREME ──
    pair_to_base_rate = {
        "DBC/SPY": "DBC_SPY_EXTREME",
        "VGK/SPY": "VGK_SPY_EXTREME",
        "DBC/TLT": "DBC_TLT_EXTREME",
        "TIP/SPY": "TIP_SPY_EXTREME",
        "XLF/SPY": "XLF_SPY_EXTREME",
        "COPPER/GLD": "CU_AU_EXTREME",
    }
    pair_to_exposed = {
        "DBC/SPY": ["DBC", "SPY"],
        "VGK/SPY": ["SPY"],
        "DBC/TLT": ["DBC", "TLT"],
        "TIP/SPY": ["SPY", "TIP"],
        "XLF/SPY": ["XLF", "SPY"],
        "COPPER/GLD": ["SPY", "DBC"],
    }

    for pair in divergences.get("pairs", []):
        if pair.get("signal") != "EXTREME" or pair.get("monitor_only", False):
            continue

        pair_name = pair.get("pair", "")
        base_rate_key = pair_to_base_rate.get(pair_name)
        exposed = pair_to_exposed.get(pair_name, ["SPY"])

        if not base_rate_key or base_rate_key not in BASE_RATES:
            continue

        br = BASE_RATES[base_rate_key]
        exposure_pct = sum(weights.get(t, 0) for t in exposed)
        exposure_eur = round(total_capital * exposure_pct, 2)

        cost_ignore = round(exposure_eur * abs(br["expected_loss"]) * br["probability"], 0)
        carry = EXPECTED_ANNUAL_CARRY.get(exposed[0] if exposed else "", 0.02)
        cost_act = round(exposure_eur * carry * (1 - br["probability"]) *
                         br["timeframe_days"] / 365, 0)

        regret_ratio = round(cost_ignore / max(cost_act, 1), 1)

        if regret_ratio > 3.0:
            rec = "HANDELN EMPFOHLEN"
        elif regret_ratio > 1.5:
            rec = "HEDGING ERWÄGEN"
        else:
            rec = "BEOBACHTEN"

        active_threats.append({
            "threat": f"{pair.get('name', pair_name)} ({pair_name} Z={pair.get('z_score', '?')})",
            "source": "DIVERGENCE",
            "exposed_assets": exposed,
            "exposure_pct": round(exposure_pct, 3),
            "exposure_eur": exposure_eur,
            "base_rate": br["probability"],
            "expected_loss_pct": br["expected_loss"] * 100,
            "cost_if_ignore": cost_ignore,
            "cost_if_act": cost_act,
            "regret_ratio": regret_ratio,
            "recommendation": rec,
            "forward_window_days": br["timeframe_days"],
        })

    # ── Multi-Signal Threat (V1.1 Fix #7) ──
    n_extreme = divergences.get("n_extreme_confirmed", 0)
    if n_extreme >= MULTI_SIGNAL_CRITICAL:
        br = BASE_RATES["MULTI_SIGNAL_3PLUS"]
        exposure_pct = sum(weights.values())  # Ganzes Portfolio
        exposure_eur = round(total_capital * exposure_pct, 2)
        cost_ignore = round(exposure_eur * abs(br["expected_loss"]) * br["probability"], 0)
        active_threats.append({
            "threat": f"Multi-Signal ({n_extreme} Paare EXTREME)",
            "source": "MULTI_SIGNAL",
            "exposed_assets": list(weights.keys()),
            "exposure_pct": round(exposure_pct, 3),
            "base_rate": br["probability"],
            "expected_loss_pct": br["expected_loss"] * 100,
            "cost_if_ignore": cost_ignore,
            "cost_if_act": 0,
            "regret_ratio": 99.9,
            "recommendation": "HANDELN EMPFOHLEN — 3+ Signale historisch 46.2% DD",
            "forward_window_days": br["timeframe_days"],
        })

    # ── Liq-Kombi Threat (V1.1 Fix #8) ──
    if liq_kombi.get("triggers_intelligence"):
        br_key = "LIQ_KOMBI_DBC_TLT"
        br = BASE_RATES.get(br_key, {})
        if br:
            exposure_pct = sum(weights.values())
            exposure_eur = round(total_capital * exposure_pct, 2)
            cost_ignore = round(exposure_eur * abs(br["expected_loss"]) * br["probability"], 0)
            active_threats.append({
                "threat": f"Liq-Kombi ({liq_kombi.get('signal', '?')}): "
                          f"Liq Z={liq_kombi.get('liq_z', '?')} + {', '.join(liq_kombi.get('kombi_pairs', []))}",
                "source": "LIQ_KOMBI",
                "exposed_assets": list(weights.keys()),
                "exposure_pct": round(exposure_pct, 3),
                "base_rate": br["probability"],
                "expected_loss_pct": br["expected_loss"] * 100,
                "cost_if_ignore": cost_ignore,
                "cost_if_act": 0,
                "regret_ratio": 99.9,
                "recommendation": f"HANDELN EMPFOHLEN — Liq+Divergenz historisch {br['probability']*100:.0f}% DD",
                "forward_window_days": br["timeframe_days"],
            })

    # ── Timeline-Konvergenz Threat ──
    if timelines.get("triggers_intelligence"):
        br = BASE_RATES.get("TIMELINE_CONVERGENCE", {})
        if br:
            active_threats.append({
                "threat": f"Timeline-Konvergenz ({timelines.get('n_active', 0)} in {TIMELINE_WINDOW_DAILY}d)",
                "source": "TIMELINE",
                "exposed_assets": [],
                "base_rate": br["probability"],
                "expected_loss_pct": br["expected_loss"] * 100,
                "regret_ratio": 0,
                "recommendation": "ERHÖHTE WACHSAMKEIT",
                "forward_window_days": br["timeframe_days"],
            })

    # Vol-Kompression: V1.1 — KEIN Threat mehr (Backtest widerlegt)

    highest_rr = max((t["regret_ratio"] for t in active_threats), default=0)

    return {
        "active_threats": active_threats,
        "n_active_threats": len(active_threats),
        "highest_regret_ratio": highest_rr,
    }


# ═══════════════════════════════════════════════════════════════
# MARKT-REAKTION (Spec TEIL2 §16)
# ═══════════════════════════════════════════════════════════════

def compute_market_reaction(calendar, price_history):
    """Berechne Absorbed/Rejected für High-Impact Events."""
    reactions = []
    spy = price_history.get("SPY", [])
    if len(spy) < 2:
        return {"reactions": [], "n_absorbed": 0, "n_rejected": 0, "n_as_expected": 0}

    spy_daily_ret = (spy[-1] - spy[-2]) / spy[-2] if spy[-2] > 0 else 0

    for ev in calendar.get("yesterday", []):
        if ev.get("impact_score", 0) < 7:
            continue
        if ev.get("actual") is None or ev.get("consensus") is None:
            continue

        surprise_raw = ev["actual"] - ev["consensus"]
        ev_type = classify_event_type(ev["event"])

        # Erwartete Reaktion
        if ev_type == "inflation" and surprise_raw > 0:
            expected_neg = True
        elif ev_type == "inflation" and surprise_raw < 0:
            expected_neg = False
        elif ev_type == "growth" and surprise_raw > 0:
            expected_neg = False
        elif ev_type == "growth" and surprise_raw < 0:
            expected_neg = True
        elif ev_type == "rate" and surprise_raw > 0:
            expected_neg = True
        else:
            continue

        if expected_neg and spy_daily_ret >= 0:
            reaction = "ABSORBED"
        elif not expected_neg and spy_daily_ret <= 0:
            reaction = "REJECTED"
        else:
            reaction = "AS_EXPECTED"

        interp = None
        if reaction == "ABSORBED":
            interp = "Markt absorbiert schlechte Nachrichten — bullish Signal"
        elif reaction == "REJECTED":
            interp = "Markt verkauft gute Nachrichten — bearish Signal"

        reactions.append({
            "event": ev["event"],
            "surprise_direction": ev_type,
            "actual_spy_return": round(spy_daily_ret, 4),
            "reaction": reaction,
            "interpretation": interp,
        })

    return {
        "reactions": reactions,
        "n_absorbed": sum(1 for r in reactions if r["reaction"] == "ABSORBED"),
        "n_rejected": sum(1 for r in reactions if r["reaction"] == "REJECTED"),
        "n_as_expected": sum(1 for r in reactions if r["reaction"] == "AS_EXPECTED"),
    }


# ═══════════════════════════════════════════════════════════════
# ANOMALIE-CHECK (Spec TEIL3 §21)
# ═══════════════════════════════════════════════════════════════

def check_anomalies(calendar):
    """Prüfe auf Emergency/Unscheduled Events."""
    anomalies = []
    emergency_kw = [
        "emergency", "unscheduled", "special", "extraordinary",
        "circuit breaker", "trading halt", "flash crash",
        "intervention", "emergency meeting",
    ]
    for ev in calendar.get("yesterday", []) + calendar.get("today", []):
        ev_name = (ev.get("event") or "").lower()
        if any(kw in ev_name for kw in emergency_kw):
            anomalies.append({
                "type": "EMERGENCY_EVENT",
                "event": ev["event"],
                "date": ev.get("date"),
                "urgency": "CRITICAL",
            })
    return anomalies


# ═══════════════════════════════════════════════════════════════
# TRIGGER CHECK (Spec TEIL3 §17)
# V1.1: Vol-Kompression entfernt, Liq-Kombi hinzugefügt
# ═══════════════════════════════════════════════════════════════

def check_triggers(surprises, divergences, alignment, timelines, liq_kombi, anomalies):
    """Prüfe ob Intelligence-Layer aktiviert werden soll.
    V1.1: Vol-Kompression ist KEIN Trigger mehr. Liq-Kombi NEU.
    """
    reasons = []
    if surprises.get("any_trigger"):
        reasons.append("SURPRISE")
    if divergences.get("triggers_intelligence"):
        reasons.append("DIVERGENCE")
    if divergences.get("multi_signal_level") == "CRITICAL":
        reasons.append("MULTI_SIGNAL_CRITICAL")
    elif divergences.get("multi_signal_level") == "WARNING":
        reasons.append("MULTI_SIGNAL_WARNING")
    if liq_kombi.get("triggers_intelligence"):
        reasons.append("LIQ_KOMBI")
    if alignment.get("triggers_intelligence"):
        reasons.append("ALIGNMENT_DROP")
    if timelines.get("triggers_intelligence"):
        reasons.append("CONVERGENCE")
    # Vol-Kompression: V1.1 — KEIN Trigger (Backtest widerlegt)
    if anomalies:
        reasons.append("ANOMALY")
    return reasons


# ═══════════════════════════════════════════════════════════════
# ETAPPE B: INTELLIGENCE LAYER (Spec TEIL3 §17-22)
# Läuft NUR wenn Trigger aktiv. Liest alle System-JSONs,
# baut System-Kontext, ruft Claude API mit Web Search.
# ═══════════════════════════════════════════════════════════════

def load_json_safe(filepath):
    """Lade JSON mit Graceful Degradation."""
    full = os.path.join(REPO_ROOT, filepath) if not os.path.isabs(filepath) else filepath
    if not os.path.exists(full):
        log(f"  JSON nicht gefunden: {filepath}")
        return None
    try:
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"  JSON Fehler {filepath}: {e}")
        return None


def load_system_jsons():
    """Lade alle System-JSONs für den Intelligence Layer.
    Graceful Degradation: fehlende Systeme werden als None markiert."""
    systems = {}
    for key, path in SYSTEM_INPUTS.items():
        systems[key] = load_json_safe(path)
        status = "OK" if systems[key] else "FEHLT"
        log(f"  System {key}: {status}")
    # V16 State aus latest.json
    latest = load_json_safe(LATEST_JSON_PATH)
    systems["v16_latest"] = latest
    log(f"  System v16_latest: {'OK' if latest else 'FEHLT'}")
    return systems


def build_system_context(systems):
    """Baut kompakten System-Kontext-Text für den LLM-Prompt (Spec TEIL3 §18.2-18.3).
    Max ~2000 Zeichen pro System, Gesamt max ~6000 Zeichen."""
    lines = ["=== BALDUR CREEK CAPITAL — SYSTEM-KONTEXT ===\n"]

    # ── V16 State ──
    latest = systems.get("v16_latest") or {}
    v16_header = latest.get("header", {})
    v16_state = v16_header.get("macro_state_name", v16_header.get("macro_state", "?"))
    v16_sa = v16_header.get("sa_score", "?")
    v16_weights = latest.get("v16", {}).get("current_weights", {})
    top5 = sorted(v16_weights.items(), key=lambda x: x[1], reverse=True)[:5] if isinstance(v16_weights, dict) else []
    weights_str = ", ".join(f"{k}={v:.0%}" for k, v in top5) if top5 else "?"
    lines.append(f"V16 STATE: {v16_state} (SA Score: {v16_sa})")
    lines.append(f"V16 POSITIONING: {weights_str}")

    # ── Cycles ──
    trans = systems.get("cycles_transition")
    if trans:
        oa = trans.get("overall_assessment", {})
        cc = trans.get("confirmation_counter", {})
        lines.append(f"\nCYCLES: {oa.get('verdict', '?')[:120]}")
        lines.append(f"  Cascade: {oa.get('cascade_severity', '?')}, Confirmation Score: {cc.get('confirmation_score', '?')}")
        bull = cc.get("bullish_cycles", [])
        bear = cc.get("bearish_cycles", [])
        if bull: lines.append(f"  Bullish: {', '.join(bull[:6])}")
        if bear: lines.append(f"  Bearish: {', '.join(bear[:6])}")
        ext = oa.get("extended_cycles", [])
        if ext: lines.append(f"  Extended: {', '.join(ext[:6])}")
    else:
        lines.append("\nCYCLES: Daten nicht verfügbar")

    # ── Thesen ──
    theses = systems.get("theses")
    if theses:
        all_t = theses.get("theses", [])
        tier1 = [t for t in all_t if t.get("tier") == 1]
        eh = theses.get("epistemic_health", {}).get("overall", "?")
        lines.append(f"\nTHESEN: {len(tier1)} Tier-1 von {len(all_t)} total (Health: {eh})")
        for t in tier1[:5]:
            pending = [c.get("event", "?") for c in t.get("catalysts", []) if c.get("status") == "PENDING"]
            cat_str = f" | Pending: {', '.join(pending[:2])}" if pending else ""
            lines.append(f"  {t.get('id','?')}: {t.get('title_short', t.get('title', '?'))[:50]} "
                         f"[{t.get('direction', '?')}] Conv={t.get('conviction', '?')}{cat_str}")
    else:
        lines.append("\nTHESEN: Daten nicht verfügbar")

    # ── Secular Trends ──
    secular = systems.get("secular_trends")
    if secular:
        cs = secular.get("conviction_summary", {})
        lines.append(f"\nSÄKULAR: Activation={cs.get('weighted_activation', '?')}, "
                     f"Direction={cs.get('convergence_direction', '?')}")
    else:
        lines.append("\nSÄKULAR: Daten nicht verfügbar")

    # ── Disruptions ──
    disrupt = systems.get("disruptions")
    if disrupt:
        # disruptions_history.json kann eine Liste oder ein Dict mit "categories" sein
        if isinstance(disrupt, list):
            cats = disrupt
        else:
            cats = disrupt.get("categories", [])
        active = [c for c in cats if isinstance(c, dict) and c.get("phase") in ("ACCELERATING", "MATURING")]
        lines.append(f"\nDISRUPTIONS: {len(cats)} total, {len(active)} aktiv (ACCELERATING/MATURING)")
    else:
        lines.append("\nDISRUPTIONS: Daten nicht verfügbar")

    # ── IC Beliefs ──
    beliefs = systems.get("ic_beliefs")
    if beliefs:
        sources = beliefs.get("sources", {})
        n = len(sources)
        lines.append(f"\nIC BELIEFS: {n} Quellen aktiv")
    else:
        lines.append("\nIC BELIEFS: Daten nicht verfügbar")

    # ── Crypto ──
    crypto = systems.get("crypto_state")
    if crypto:
        ens = crypto.get("ensemble", {}).get("value", "?")
        phase = crypto.get("trickle_down", {}).get("phase_name", "?")
        btc = crypto.get("btc_price", "?")
        action = crypto.get("action", "?")
        lines.append(f"\nCRYPTO: Ensemble={ens}, Phase={phase}, BTC=${btc}, Action={action}")
    else:
        lines.append("\nCRYPTO: Daten nicht verfügbar")

    # ── Ratio Context ──
    ratios = systems.get("ratio_context")
    if ratios:
        all_r = ratios.get("ratios", [])
        extreme = [r for r in all_r if abs(r.get("analysis", {}).get("z_full", 0)) >= 1.5]
        if extreme:
            lines.append(f"\nEXTREME RATIOS ({len(extreme)} Paare |Z|≥1.5):")
            for r in extreme[:5]:
                z = r["analysis"]["z_full"]
                hl = r["analysis"].get("halflife")
                hl_str = f", HL={hl:.0f}d" if hl else ""
                lines.append(f"  {r['pair']} ({r.get('description', '?')}): Z={z:+.2f} → "
                             f"{r['analysis'].get('signal', '?')}{hl_str}")
        else:
            lines.append("\nRATIOS: Keine Extremwerte (|Z|≥1.5)")
    else:
        lines.append("\nRATIOS: Daten nicht verfügbar")

    return "\n".join(lines)


def build_intelligence_prompt(trigger_reasons, calendar, surprises, divergences,
                                liquidity, liq_kombi, alignment, timelines,
                                vol_compression, market_reactions, regret_matrix,
                                system_context_text):
    """Baut den vollständigen User Message für den Intelligence LLM Call (Spec TEIL3 §19.2)."""

    # Trigger-Zusammenfassung
    triggers_text = "\n".join(f"  • {t}" for t in trigger_reasons) if trigger_reasons else "  Keine"

    # Extreme Divergenzen kompakt
    extreme_pairs = [p for p in divergences.get("pairs", [])
                     if p.get("signal") in ("EXTREME", "EXTREME_UNCONFIRMED", "ELEVATED")]
    div_text = "\n".join(
        f"  {p['pair']} Z={p.get('z_score', '?'):+.2f} → {p['signal']}"
        + (f" ({p.get('interpretation', '')})" if p.get("interpretation") else "")
        for p in extreme_pairs
    ) if extreme_pairs else "  Keine auffälligen Paare"

    # Threats kompakt
    threats = regret_matrix.get("active_threats", [])
    threats_text = "\n".join(
        f"  {t['threat']}: RR={t['regret_ratio']:.1f}x, {t['recommendation']}"
        for t in threats
    ) if threats else "  Keine"

    msg = f"""Datum: {TODAY_STR}

=== TRIGGER (warum dieser Call aktiviert wurde) ===
{triggers_text}

=== CROSS-ASSET DIVERGENZEN (Alert: {divergences.get('alert_level', '?')}) ===
{div_text}

=== PORTFOLIO P&L ===
  Daily: {calendar.get('_pnl_daily', '?'):+.2f}%  YTD: {calendar.get('_pnl_ytd', '?'):+.2f}%

=== LIQUIDITÄT ===
  Net: ${liquidity.get('net_liquidity_usd_T', '?')}T, Direction: {liquidity.get('direction', '?')}, Z(4W): {liquidity.get('liq_z_score', '?')}
  1W: {liquidity.get('change_1w_usd_B', '?')}B, 4W: {liquidity.get('change_4w_usd_B', '?')}B

=== LIQ-KOMBI ===
  Signal: {liq_kombi.get('signal', '?')} — {liq_kombi.get('interpretation', '')}

=== ALIGNMENT ===
  Score: {alignment.get('score', '?'):.2f} — {alignment.get('interpretation', '?')}
  Systeme: {', '.join(f"{k}={v['direction']}" for k, v in alignment.get('systems', {}).items())}

=== TIMELINES ===
  {timelines.get('n_active', 0)} aktiv, Level: {timelines.get('convergence_level', '?')}

=== VOL-KOMPRESSION (nur Kontext, kein Trigger) ===
  Score: {vol_compression.get('compression_score', '?')}, Signal: {vol_compression.get('signal', '?')}

=== MARKT-REAKTIONEN GESTERN ===
  Absorbed: {market_reactions.get('n_absorbed', 0)}, Rejected: {market_reactions.get('n_rejected', 0)}

=== AKTIVE THREATS (Regret-Matrix) ===
{threats_text}

=== INTERNE SYSTEM-DATEN ===
{system_context_text}

=== EVENTS HEUTE ===
{json.dumps([e.get('event', '?') + ' ' + e.get('country', '') for e in calendar.get('today', [])[:5]], ensure_ascii=False)}

AUFGABE:
1. Analysiere die Trigger durch alle verfügbaren System-Linsen (V16, Cycles, Thesen, Secular, Crypto, Ratios).
2. Identifiziere Zeitlücken: Was siehst du JETZT das V16 erst in 4-8 Wochen sehen wird?
3. Finde Second-Order Effects die der Markt noch nicht einpreist.
4. Verankere jede Aussage in historischen Mustern wo möglich.
5. Erstelle Threats (Portfolio-Bedrohungen) und Signals (Katalysatoren, Zeitlücken).
6. EIN Satz Summary: Was muss ich heute wissen?"""

    return msg


INTELLIGENCE_SYSTEM_PROMPT = """Du bist der Event Intelligence Analyst von Baldur Creek Capital,
einem systematischen Macro Hedgefund mit 18 Investment Circles.

Dein Job ist NICHT Events aufzulisten. Dein Job ist:
1. Events und Signale durch die Linse unserer internen Systeme zu interpretieren
2. Die Zeitlücke zu identifizieren: was sieht der Agent JETZT das V16 erst in 4-8 Wochen sehen wird?
3. Second-Order Effects zu finden die der Markt noch nicht einpreist
4. Base Rates zu verankern: "In X von Y historischen Fällen folgte Z"

REGELN:
- Jede Faktenbehauptung braucht Evidenz oder historische Basis.
- Unterscheide FACT / INFERENCE / SPECULATION klar.
- Keine generischen Aussagen wie "das könnte zu Volatilität führen."
  Stattdessen: "In 7 von 11 Fällen mit dieser Signatur folgte SPY -3% in 30 Tagen."
- Keine "KI" oder "AI" in deinen Texten.
- Halte dich kurz und präzise. Kein Fülltext.

Antworte NUR mit validem JSON (kein Markdown, keine Backticks). Schema:
{
  "trigger_analysis": {
    "primary_trigger": "...",
    "trigger_type": "SURPRISE|DIVERGENCE|ALIGNMENT|CONVERGENCE|LIQ_KOMBI|ANOMALY",
    "severity": "CRITICAL|HIGH|MODERATE",
    "interpretation": "1-2 Sätze was der Trigger bedeutet"
  },
  "system_lens_analysis": {
    "v16_regime": "Wie verändert das den V16 State? Zeitlücke?",
    "cycles": "Beschleunigt/bremst das eine Transition?",
    "thesen": "Triggert das einen Katalysator? Welche These?",
    "secular": "Verstärkt/schwächt das einen Trend?",
    "crypto": "Impact auf Ensemble?",
    "relative_value": "Verschiebt das ein extremes Ratio?"
  },
  "time_gap_warning": {
    "exists": true,
    "description": "V16 wird in ~X Wochen auf Y wechseln weil...",
    "confidence": "HIGH|MEDIUM|LOW",
    "historical_basis": "In N von M Fällen..."
  },
  "second_order_effects": [
    {
      "effect": "...",
      "mechanism": "A → B → C",
      "affected_assets": ["SPY", "TLT"],
      "timeframe": "2-4 Wochen"
    }
  ],
  "threats": [
    {
      "title": "...",
      "severity": "CRITICAL|HIGH|MODERATE",
      "description": "...",
      "exposed_assets": ["SPY"],
      "time_horizon": "30-60 Tage",
      "action_suggestion": "..."
    }
  ],
  "signals": [
    {
      "title": "...",
      "type": "TIME_GAP|CATALYST_TRIGGERED|REGIME_SHIFT|DIVERGENCE_CONFIRMED|MARKET_ABSORBED",
      "description": "...",
      "affected_assets": ["..."]
    }
  ],
  "portfolio_action_required": false,
  "summary_one_liner": "Ein Satz. Was muss ich heute wissen?"
}"""


def run_intelligence_layer(trigger_reasons, calendar, pnl, surprises, divergences,
                            liquidity, liq_kombi, alignment, timelines,
                            vol_compression, market_reactions, regret_matrix):
    """Führt den Intelligence Layer aus: System-JSONs laden, Kontext bauen, LLM Call.
    Returns: intelligence dict oder None bei Fehler."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log("ANTHROPIC_API_KEY nicht gesetzt — Intelligence Layer übersprungen")
        return None

    try:
        import anthropic
    except ImportError:
        log("anthropic Paket nicht installiert — Intelligence Layer übersprungen")
        return None

    # 1. System-JSONs laden
    log("Intelligence Layer: Lade System-JSONs...")
    systems = load_system_jsons()
    n_available = sum(1 for v in systems.values() if v is not None)
    log(f"  {n_available}/{len(systems)} Systeme verfügbar")

    # 2. System-Kontext bauen
    log("Intelligence Layer: Baue System-Kontext...")
    system_context_text = build_system_context(systems)
    log(f"  Kontext: {len(system_context_text)} Zeichen")

    # 3. User Message bauen
    # P&L in calendar einfügen für den Prompt
    calendar_with_pnl = dict(calendar)
    calendar_with_pnl["_pnl_daily"] = pnl.get("daily_return_pct", 0)
    calendar_with_pnl["_pnl_ytd"] = pnl.get("ytd_return_pct", 0)

    user_msg = build_intelligence_prompt(
        trigger_reasons, calendar_with_pnl, surprises, divergences,
        liquidity, liq_kombi, alignment, timelines,
        vol_compression, market_reactions, regret_matrix,
        system_context_text,
    )
    log(f"  User Message: {len(user_msg)} Zeichen")

    # 4. Anthropic API Call
    log("Intelligence Layer: Claude API Call (mit Web Search)...")
    t0 = time.time()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=INTELLIGENCE_SYSTEM_PROMPT,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": user_msg}],
        )

        elapsed = time.time() - t0
        log(f"  API Call: {elapsed:.1f}s, Stop: {response.stop_reason}")

        # 5. Response parsen
        # Sammle allen Text aus der Response (kann mehrere text blocks haben wegen Web Search)
        text_parts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                text_parts.append(block.text)

        full_text = "\n".join(text_parts)

        # JSON extrahieren (LLM könnte Markdown-Backticks drumherum haben)
        json_text = full_text.strip()
        if json_text.startswith("```"):
            # Entferne Markdown-Backticks
            lines = json_text.split("\n")
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            json_text = "\n".join(lines[start:end])

        intelligence = json.loads(json_text)
        log(f"  Intelligence JSON geparst: {len(intelligence)} top-level Keys")

        # Token-Usage loggen
        usage = response.usage
        if usage:
            log(f"  Tokens: {usage.input_tokens} input, {usage.output_tokens} output")

        return intelligence

    except json.JSONDecodeError as e:
        log(f"  JSON Parse Fehler: {e}")
        log(f"  Raw Response (erste 500 Zeichen): {full_text[:500]}")
        # Fallback: Roh-Text als summary speichern
        return {
            "trigger_analysis": {"primary_trigger": ", ".join(trigger_reasons), "severity": "MODERATE"},
            "summary_one_liner": full_text[:200] if full_text else "Intelligence Layer konnte Response nicht parsen.",
            "_raw_response": full_text[:2000],
            "_parse_error": str(e),
        }
    except Exception as e:
        log(f"  Intelligence Layer Fehler: {e}")
        traceback.print_exc()
        return {
            "trigger_analysis": {"primary_trigger": ", ".join(trigger_reasons), "severity": "MODERATE"},
            "summary_one_liner": f"Intelligence Layer Fehler: {str(e)[:100]}",
            "_error": str(e),
        }

def assemble_daily_output(calendar, pnl, surprises, divergences, liquidity,
                           liq_kombi, alignment, timelines, vol_compression,
                           decay_events, regret_matrix, market_reactions,
                           anomalies, trigger_reasons, intelligence=None):
    """Assembliere das vollständige Daily Output JSON (Spec TEIL6 §42)."""

    # CIO Paragraph (kompakter Überblick)
    cio_parts = []
    cio_parts.append(f"System Alignment {alignment.get('score', 0):.2f} ({alignment.get('interpretation', '?')}).")
    liq_dir = liquidity.get("direction", "?")
    liq_1w = liquidity.get("change_1w_usd_B", 0)
    if liq_dir != "FLAT":
        cio_parts.append(f"Liquidität {liq_dir} ({liq_1w:+.0f}B/W).")
    n_threats = regret_matrix.get("n_active_threats", 0)
    if n_threats > 0:
        top_threat = regret_matrix["active_threats"][0]
        cio_parts.append(f"{n_threats} offene Threats. Top: {top_threat.get('threat', '?')}.")
    else:
        cio_parts.append("Keine aktiven Threats.")
    if intelligence and intelligence.get("summary_one_liner"):
        cio_parts.append(intelligence["summary_one_liner"])
    cio_paragraph = " ".join(cio_parts)

    # Telegram Message (kompakt)
    tg_lines = []
    tg_lines.append(f"📊 Portfolio {pnl['daily_return_pct']:+.2f}% | "
                     f"YTD {pnl['ytd_return_pct']:+.2f}% | "
                     f"Alignment {alignment.get('score', 0):.2f}")
    if intelligence:
        threats = intelligence.get("threats", [])
        signals = intelligence.get("signals", [])
        if threats:
            tg_lines.append(f"🔴 {len(threats)} Threats")
            for t in threats[:2]:
                tg_lines.append(f"  • {t.get('title', '?')}")
        if signals:
            tg_lines.append(f"⚡ {len(signals)} Signals")
            for s in signals[:2]:
                tg_lines.append(f"  • {s.get('title', '?')}")
    today_events = sorted(calendar.get("today", []),
                           key=lambda e: e.get("impact_score", 0), reverse=True)[:3]
    if today_events:
        ev_strs = [f"{e.get('time', '?')} {e.get('event', '?')}" for e in today_events]
        tg_lines.append(f"📅 Heute: {' | '.join(ev_strs)}")
    if liq_dir and liq_dir not in ("FLAT", "?"):
        arrow = "↑" if liq_dir == "EXPANDING" else "↓" if liq_dir == "CONTRACTING" else "→"
        tg_lines.append(f"💧 Liquidität: {liq_dir} {arrow}")
    if intelligence and intelligence.get("summary_one_liner"):
        tg_lines.append(f"\n💡 {intelligence['summary_one_liner']}")
    telegram_message = "\n".join(tg_lines)

    return {
        "version": f"command_center V{CC_VERSION}",
        "timestamp": NOW.isoformat(),
        "date": TODAY_STR,
        "run_type": "daily",
        "intelligence_triggered": len(trigger_reasons) > 0,
        "trigger_reasons": trigger_reasons,
        "portfolio_pnl": pnl,
        "calendar": calendar,
        "surprises": surprises,
        "market_reactions": market_reactions,
        "divergences": divergences,
        "liquidity": liquidity,
        "liq_kombi": liq_kombi,
        "alignment": alignment,
        "timelines": timelines,
        "vol_compression": vol_compression,
        "active_events_decay": decay_events,
        "regret_matrix": regret_matrix,
        "anomalies": anomalies,
        # Intelligence-Layer Output (Etappe B)
        "intelligence": intelligence,
        # CIO + Telegram (Etappe D — jetzt inline generiert)
        "cio_paragraph": cio_paragraph,
        "telegram_message": telegram_message,
    }


# ═══════════════════════════════════════════════════════════════
# WRITE OUTPUTS
# ═══════════════════════════════════════════════════════════════

def write_outputs(output, skip_write=False):
    """Schreibe JSON Outputs."""
    os.makedirs(CC_DATA_DIR, exist_ok=True)
    os.makedirs(CC_DAILY_HISTORY_DIR, exist_ok=True)

    # Haupt-Output
    with open(CC_DAILY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(CC_DAILY_OUTPUT)
    log(f"Output: {CC_DAILY_OUTPUT} ({size:,} bytes)")

    # Daily History Archiv
    hist_file = os.path.join(CC_DAILY_HISTORY_DIR, f"command_center_{TODAY_STR}.json")
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    log(f"History: {hist_file}")

    # Alte History aufräumen (>30 Tage)
    cutoff = (NOW - timedelta(days=35)).strftime("%Y-%m-%d")
    for fn in os.listdir(CC_DAILY_HISTORY_DIR):
        if fn.startswith("command_center_") and fn.endswith(".json"):
            date_part = fn.replace("command_center_", "").replace(".json", "")
            if date_part < cutoff:
                os.remove(os.path.join(CC_DAILY_HISTORY_DIR, fn))
                log(f"Alte History gelöscht: {fn}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="System Command Center V1.1")
    pa.add_argument("--mode", choices=["daily", "weekly"], required=True)
    pa.add_argument("--skip-write", action="store_true")
    pa.add_argument("--skip-telegram", action="store_true")
    pa.add_argument("--force-intelligence", action="store_true")
    args = pa.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"SYSTEM COMMAND CENTER — V{CC_VERSION}")
    print(f"  Mode: {args.mode}")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Flags: skip-write={args.skip_write} skip-telegram={args.skip_telegram}"
          f" force-intel={args.force_intelligence}")
    print("=" * 70)

    if args.mode == "weekly":
        log("Weekly Run — Etappe C (noch nicht implementiert)")
        print("=" * 70)
        return

    # ─── DAILY RUN ───────────────────────────────────────
    import requests  # Hier damit Colab Test ohne requests nicht crasht

    # GCP Auth
    gc = get_gspread_client()
    if gc:
        log("GCP Auth OK")
    else:
        log("GCP Auth FEHLT — Sheet-Berechnungen degradiert")

    # V16 Gewichte laden
    weights = load_v16_weights()

    # ─── Berechnung 1: FMP Calendar ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 1: FMP ECON CALENDAR")
    print(f"{'─'*50}")
    calendar = fetch_fmp_calendar()

    # ─── Berechnung 2: Portfolio P&L ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 2: PORTFOLIO P&L + YTD")
    print(f"{'─'*50}")
    sheet_data = fetch_sheet_prices(gc)
    pnl = compute_portfolio_pnl(sheet_data, weights)
    log(f"P&L: {pnl['daily_return_pct']:+.2f}% daily, {pnl['ytd_return_pct']:+.2f}% YTD")

    # ─── Berechnung 3: Surprises ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 3: SURPRISE-FAKTOR")
    print(f"{'─'*50}")
    surprises = compute_surprises(calendar)
    log(f"Surprises: {surprises['n_surprises']} Events, Trigger: {surprises['any_trigger']}")

    # ─── FRED Daten holen (für Berechnung 4, 5, 8) ───
    print(f"\n{'─'*50}")
    print("FRED DATEN LADEN")
    print(f"{'─'*50}")
    fred_data = fetch_fred_data()

    # ─── Price History holen (für Berechnung 4, 8, Markt-Reaktion) ───
    print(f"\n{'─'*50}")
    print("SHEET PRICE HISTORY LADEN")
    print(f"{'─'*50}")
    # V1.1: Neue Ticker für neue Paare (DBC, VGK, XLF dazu)
    needed_tickers = ["SPY", "DBC", "VGK", "TLT", "TIP", "XLF", "GLD", "HYG", "LQD", "IWM"]
    price_history = fetch_price_history(gc, needed_tickers, n_days=260)

    # Cu/Au Ratio
    cu_au_ratios = fetch_cu_au_ratio(gc, n_days=260)

    # VIX Daten (aus FRED)
    vix_data = [d["value"] for d in fred_data.get("vix", [])]

    # SPY Preise (aus Sheet)
    spy_prices = price_history.get("SPY", [])

    # ─── Berechnung 4: Cross-Asset Divergenz ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 4: CROSS-ASSET DIVERGENZ")
    print(f"{'─'*50}")
    divergences = compute_divergences(price_history, cu_au_ratios, vix_data, spy_prices)
    log(f"Divergenzen: Alert={divergences['alert_level']}, "
        f"Extreme(bestätigt)={divergences['n_extreme_confirmed']}, "
        f"Extreme(unbestätigt)={divergences['n_extreme_unconfirmed']}, "
        f"Elevated={divergences['n_elevated']}")
    if divergences.get("vix_confirms"):
        log("  VIX bestätigt (Z>+2.0)")
    if divergences.get("vix_corr_watch"):
        log("  VIX/SPY Korrelation WATCH (>-0.2)")
    log(f"  Multi-Signal: {divergences['multi_signal_level']}")

    # ─── Berechnung 5: Liquidität ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 5: LIQUIDITÄTSINDIKATOR")
    print(f"{'─'*50}")
    liquidity = compute_liquidity(fred_data)
    log(f"Liquidität: Net ${liquidity.get('net_liquidity_usd_T', '?')}T, "
        f"Direction: {liquidity.get('direction', '?')}, "
        f"Liq-Z: {liquidity.get('liq_z_score', '?')}")

    # ─── Berechnung 5b: Liq-Kombi ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 5b: LIQUIDITÄTS-KOMBI")
    print(f"{'─'*50}")
    liq_kombi = compute_liq_kombi(liquidity, divergences)
    log(f"Liq-Kombi: {liq_kombi['signal']}")
    if liq_kombi.get("triggers_intelligence"):
        log(f"  ⚠️ Liq-Kombi TRIGGER: {liq_kombi.get('interpretation', '')}")

    # ─── Berechnung 6: Alignment Matrix ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 6: ALIGNMENT MATRIX")
    print(f"{'─'*50}")
    alignment = compute_alignment(liquidity, divergences)
    log(f"Alignment: {alignment['score']:.2f} — {alignment['interpretation']}")
    for name, sys_data in alignment.get("systems", {}).items():
        log(f"  {name:12s} → {sys_data['direction']:10s} ({sys_data['detail']})")

    # ─── Berechnung 7: Converging Timelines ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 7: CONVERGING TIMELINES")
    print(f"{'─'*50}")
    timelines = compute_timelines(calendar)
    log(f"Timelines: {timelines['n_active']} aktiv, Level: {timelines['convergence_level']}")

    # ─── Berechnung 8: Vol-Kompression (informational only) ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 8: VOLATILITÄTS-KOMPRESSION (informational)")
    print(f"{'─'*50}")
    vol_compression = compute_vol_compression(spy_prices)
    log(f"Vol: {vol_compression.get('realized_vol_21d', '?')}% "
        f"({vol_compression.get('vol_percentile', '?')}. Pctl), "
        f"Score: {vol_compression.get('compression_score', '?')}, "
        f"Signal: {vol_compression.get('signal', '?')} (kein Trigger)")

    # ─── Berechnung 9: Surprise-Decay ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 9: SURPRISE-DECAY-TIMER")
    print(f"{'─'*50}")
    decay_events = compute_decay_timer(calendar)
    log(f"Decay: {len(decay_events)} aktive Events im Rolling 30d Window")

    # ─── Berechnung 10: Regret-Matrix ───
    print(f"\n{'─'*50}")
    print("BERECHNUNG 10: REGRET-MATRIX")
    print(f"{'─'*50}")
    regret_matrix = compute_regret_matrix(divergences, vol_compression, timelines, liq_kombi, weights)
    log(f"Regret: {regret_matrix['n_active_threats']} aktive Threats, "
        f"höchstes RR: {regret_matrix['highest_regret_ratio']}")

    # ─── Markt-Reaktion ───
    print(f"\n{'─'*50}")
    print("MARKT-REAKTION (ABSORBED/REJECTED)")
    print(f"{'─'*50}")
    market_reactions = compute_market_reaction(calendar, price_history)
    log(f"Reaktionen: {market_reactions['n_absorbed']} absorbed, "
        f"{market_reactions['n_rejected']} rejected")

    # ─── Anomalie-Check ───
    anomalies = check_anomalies(calendar)
    if anomalies:
        log(f"⚠️ ANOMALIEN: {len(anomalies)}")

    # ─── Trigger-Check ───
    print(f"\n{'─'*50}")
    print("TRIGGER-CHECK")
    print(f"{'─'*50}")
    trigger_reasons = check_triggers(surprises, divergences, alignment,
                                      timelines, liq_kombi, anomalies)
    if trigger_reasons:
        log(f"🔴 INTELLIGENCE TRIGGERED: {', '.join(trigger_reasons)}")
    else:
        log("✅ Kein Trigger — ruhiger Tag. Nur Daten-Layer Output.")

    # ─── Intelligence Layer (Etappe B) ───
    intelligence = None
    if trigger_reasons or args.force_intelligence:
        print(f"\n{'─'*50}")
        print("INTELLIGENCE LAYER (Etappe B)")
        print(f"{'─'*50}")
        if args.force_intelligence and not trigger_reasons:
            log("Force-Intelligence aktiv (kein natürlicher Trigger)")
        intelligence = run_intelligence_layer(
            trigger_reasons, calendar, pnl, surprises, divergences,
            liquidity, liq_kombi, alignment, timelines,
            vol_compression, market_reactions, regret_matrix,
        )
        if intelligence:
            log(f"Intelligence Layer: OK — {intelligence.get('summary_one_liner', '?')[:80]}")
        else:
            log("Intelligence Layer: Kein Output (Fehler oder kein API Key)")
    else:
        log("Intelligence Layer: Nicht aktiviert (kein Trigger)")

    # ─── Output assemblieren ───
    print(f"\n{'─'*50}")
    print("OUTPUT ASSEMBLIERUNG")
    print(f"{'─'*50}")
    output = assemble_daily_output(
        calendar, pnl, surprises, divergences, liquidity, liq_kombi,
        alignment, timelines, vol_compression, decay_events,
        regret_matrix, market_reactions, anomalies, trigger_reasons,
        intelligence=intelligence,
    )

    # ─── Write ───
    if not args.skip_write:
        write_outputs(output)
    else:
        log("Write übersprungen (--skip-write)")
        # Trotzdem JSON in stdout zeigen für Colab
        size = len(json.dumps(output, default=str))
        log(f"Output Größe: {size:,} bytes (nicht geschrieben)")

    # ─── Zusammenfassung ───
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMMAND CENTER — DAILY RUN FERTIG ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  P&L:           {pnl['daily_return_pct']:+.2f}% daily, {pnl['ytd_return_pct']:+.2f}% YTD")
    print(f"  Alignment:     {alignment['score']:.2f} — {alignment['interpretation']}")
    print(f"  Divergenzen:   {divergences['alert_level']} "
          f"({divergences['n_extreme_confirmed']} bestätigt, "
          f"{divergences['n_extreme_unconfirmed']} unbestätigt)")
    print(f"  Multi-Signal:  {divergences['multi_signal_level']}")
    print(f"  Liq-Kombi:     {liq_kombi['signal']}")
    print(f"  Liquidität:    {liquidity.get('direction', '?')} "
          f"(Net ${liquidity.get('net_liquidity_usd_T', '?')}T, Z={liquidity.get('liq_z_score', '?')})")
    print(f"  Vol:           {vol_compression.get('signal', '?')} "
          f"(Score {vol_compression.get('compression_score', '?')}) — kein Trigger")
    print(f"  Threats:       {regret_matrix['n_active_threats']}")
    intel_status = "Nicht aktiviert"
    if intelligence:
        summary = intelligence.get("summary_one_liner", "OK")[:80]
        intel_status = f"OK — {summary}"
    elif trigger_reasons:
        intel_status = "TRIGGERED aber kein Output (API Key?)"
    print(f"  Intelligence:  {intel_status}")
    print(f"  Timelines:     {timelines['convergence_level']} ({timelines['n_active']} aktiv)")
    print(f"  Surprises:     {surprises['n_surprises']} Events")
    print(f"  Anomalien:     {len(anomalies)}")
    if divergences.get("vix_confirms"):
        print(f"  VIX:           BESTÄTIGUNG (Z>+{VIX_Z_CONFIRMATION})")
    if divergences.get("vix_corr_watch"):
        print(f"  VIX/SPY Korr:  WATCH (>{VIX_CORR_WATCH_THRESHOLD})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
