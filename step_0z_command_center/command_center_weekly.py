#!/usr/bin/env python3
"""
command_center_weekly.py — System Command Center Weekly Run V2.0
================================================================
Baldur Creek Capital | Circle 18 | System Command Center — Etappe C FULL

V2.0 Änderungen (Etappe C Teil 2):
  - 3 LLM Calls implementiert: Narrative, Unpriced Risk, Positioning
  - System-Kontext-Builder (build_weekly_system_context)
  - Narrative History Tracker
  - Slow-Burn Tracker
  - Feedback-Loop Tracker
  - PENDING_LLM Placeholder ersetzt durch echte Ergebnisse
  - Telegram erweitert um Narrative, Slow-Burns, Loops
  - metadata.etappe = "C_FULL", llm_calls = 3

V1.0 (Etappe C Teil 1):
  - 6 deterministische Berechnungen (Liquidität, Rolling Events, Clusters,
    Bias-Check, Timeline-Forecast, Signal-Review)

Aufgerufen von command_center_agent.py main() wenn --mode weekly.

Spec: MACRO_EVENTS_SPEC_TEIL3.md (§18-23), TEIL4.md (§24-30)

Usage:
  # Wird von command_center_agent.py aufgerufen:
  from command_center_weekly import run_weekly
  run_weekly(args, gc)
"""
import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone, timedelta

# ═══════════════════════════════════════════════════════════════
# PATHS (relativ zu Repo Root — gleiche Logik wie Agent)
# ═══════════════════════════════════════════════════════════════
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CC_DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CC_WEEKLY_OUTPUT = os.path.join(CC_DATA_DIR, "command_center_weekly.json")
CC_DAILY_HISTORY_DIR = os.path.join(CC_DATA_DIR, "daily_history")
CC_NARRATIVE_HISTORY = os.path.join(CC_DATA_DIR, "narrative_history.json")
CC_SLOW_BURN_HISTORY = os.path.join(CC_DATA_DIR, "slow_burn_history.json")
CC_LOOP_HISTORY = os.path.join(CC_DATA_DIR, "feedback_loop_history.json")

NOW = datetime.now(timezone.utc)
TODAY_STR = NOW.strftime("%Y-%m-%d")
VERSION = "command_center_weekly V2.0"

# ── FRED API ──
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_LIQUIDITY_SERIES = {
    "fed_bs": "WALCL",       # Fed Total Assets (Wöchentlich, Mio USD)
    "tga": "WTREGEN",         # Treasury General Account (Wöchentlich, Mio USD)
    "rrp": "RRPONTSYD",       # Reverse Repo (Täglich, Mrd USD)
}

# ── System-Input-Pfade (gleiche wie im Agent) ──
SYSTEM_INPUTS = {
    "cycles_transition": "step_0v_cycles/data/transition_engine.json",
    "theses": "step_0x_theses/data/theses.json",
    "secular_trends": "step_0w_secular/data/secular_trends.json",
    "disruptions": "data/disruptions/disruptions_history.json",
    "ic_beliefs": "step_0i_ic_pipeline/data/history/beliefs.json",
    "crypto_state": "step_0y_crypto/data/crypto_state.json",
    "ratio_context": "step_0x_theses/data/ratio_context.json",
}
LATEST_JSON_PATH = os.path.join(REPO_ROOT, "data", "dashboard", "latest.json")

# ── Alignment Richtungen (gleiche wie im Agent) ──
V16_BULLISH_STATES = [
    "STEADY_GROWTH", "FRAGILE_EXPANSION", "REFLATION", "EARLY_RECOVERY",
]
V16_BEARISH_STATES = [
    "STRESS_ELEVATED", "CONTRACTION", "DEEP_CONTRACTION", "FINANCIAL_CRISIS",
]

# ── LLM Config (gleiche wie im Agent, Spec TEIL3 §19) ──
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 16000
LLM_TEMPERATURE = 0.2


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════
def log(msg):
    print(f"  [CC-W] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def load_json_safe(path):
    """Lade JSON mit Graceful Degradation."""
    full = os.path.join(REPO_ROOT, path) if not os.path.isabs(path) else path
    if not os.path.exists(full):
        return None
    try:
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_json_from_llm_response(response):
    """Extrahiert JSON aus LLM Response (kann mehrere text blocks haben wegen Web Search).
    Gleiche Logik wie im Agent run_intelligence_layer."""
    text_parts = []
    for block in response.content:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)

    full_text = "\n".join(text_parts)

    # JSON extrahieren (LLM könnte Markdown-Backticks drumherum haben)
    json_text = full_text.strip()
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        json_text = "\n".join(lines[start:end])

    return json.loads(json_text), full_text


# ═══════════════════════════════════════════════════════════════
# 1. LIQUIDITÄTS-TREND (52 Wochen FRED Serie)
# Spec TEIL4 §25.1 + §25.2
# ═══════════════════════════════════════════════════════════════

def fetch_liquidity_timeseries():
    """Holt 52+ Wochen Liquiditätsdaten von FRED für Trend-Analyse."""
    import requests

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log("FRED_API_KEY nicht gesetzt")
        return {}

    end_date = NOW.strftime("%Y-%m-%d")
    start_date = (NOW - timedelta(days=420)).strftime("%Y-%m-%d")  # ~60W für Puffer

    data = {}
    for key, series_id in FRED_LIQUIDITY_SERIES.items():
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
            data[key] = parsed
            log(f"FRED {series_id}: {len(parsed)} Datenpunkte")
        except Exception as e:
            log(f"FRED {series_id} Fehler: {e}")
            data[key] = []

    return data


def compute_liquidity_trend(fred_data):
    """Berechnet Net Liquidity Trend über 52 Wochen (Spec TEIL4 §25.2)."""
    fed_bs = {d["date"]: d["value"] for d in fred_data.get("fed_bs", [])}
    tga = {d["date"]: d["value"] for d in fred_data.get("tga", [])}
    rrp = {d["date"]: d["value"] * 1000 for d in fred_data.get("rrp", [])}  # Mrd → Mio

    # Gemeinsame Daten (wöchentlich, fed_bs + tga haben gleiche Frequenz)
    common_dates = sorted(set(fed_bs.keys()) & set(tga.keys()))

    if len(common_dates) < 4:
        log("Zu wenige Datenpunkte für Liquiditäts-Trend")
        return {"error": "Zu wenige Datenpunkte", "trend": "UNKNOWN"}

    # Net Liquidity Serie bauen
    net_liq_series = []
    for date in common_dates:
        # RRP: nächster verfügbarer Wert <= date
        rrp_val = 0
        for rrp_date in sorted(rrp.keys(), reverse=True):
            if rrp_date <= date:
                rrp_val = rrp[rrp_date]
                break
        net = fed_bs[date] - tga[date] - rrp_val
        net_liq_series.append({"date": date, "net_liquidity_mio": net})

    # Aktuelle Werte
    current = net_liq_series[-1]

    # Veränderungen über verschiedene Zeitfenster
    changes = {}
    for label, weeks_back in [("1w", 1), ("4w", 4), ("13w", 13), ("26w", 26), ("52w", 52)]:
        idx = -(weeks_back + 1)
        if len(net_liq_series) > weeks_back:
            past = net_liq_series[idx]
            change_mio = current["net_liquidity_mio"] - past["net_liquidity_mio"]
            change_pct = 0
            if abs(past["net_liquidity_mio"]) > 0:
                change_pct = change_mio / abs(past["net_liquidity_mio"]) * 100
            changes[label] = {
                "change_mio": round(change_mio),
                "change_B": round(change_mio / 1000, 1),
                "change_pct": round(change_pct, 2),
            }

    # Trend-Richtung (basierend auf 4W und 13W)
    c4w = changes.get("4w", {}).get("change_mio", 0)
    c13w = changes.get("13w", {}).get("change_mio", 0)

    if c4w > 0 and c13w > 0:
        trend = "EXPANDING"
    elif c4w < 0 and c13w < 0:
        trend = "CONTRACTING"
    elif c13w > 0 and c4w < 0:
        trend = "DECELERATING"
    elif c13w < 0 and c4w > 0:
        trend = "BOTTOMING"
    else:
        trend = "FLAT"

    # Treiber-Attribution (was hat sich am meisten verändert in 4W?)
    main_driver = "unknown"
    if len(common_dates) > 4:
        past_date = common_dates[-5] if len(common_dates) >= 5 else common_dates[0]
        delta_fed = fed_bs.get(common_dates[-1], 0) - fed_bs.get(past_date, 0)
        delta_tga = -(tga.get(common_dates[-1], 0) - tga.get(past_date, 0))
        drivers = {"Fed_BS": delta_fed, "TGA": delta_tga}
        main_driver = max(drivers, key=lambda k: abs(drivers[k]))

    # Komponenten einzeln
    fed_current = fed_bs.get(common_dates[-1], 0)
    tga_current = tga.get(common_dates[-1], 0)
    rrp_latest = 0
    for rrp_date in sorted(rrp.keys(), reverse=True):
        rrp_latest = rrp[rrp_date]
        break

    result = {
        "current_net_liquidity_T": round(current["net_liquidity_mio"] / 1_000_000, 2),
        "current_date": current["date"],
        "fed_bs_T": round(fed_current / 1_000_000, 2),
        "tga_T": round(tga_current / 1_000_000, 2),
        "rrp_T": round(rrp_latest / 1_000_000, 3),
        "changes": changes,
        "trend": trend,
        "main_driver_4w": main_driver,
        "series_length_weeks": len(net_liq_series),
        # Letzte 52 Wochen für Frontend-Chart (kompakt: nur date + T)
        "series_52w": [
            {"date": s["date"], "net_T": round(s["net_liquidity_mio"] / 1_000_000, 2)}
            for s in net_liq_series[-52:]
        ],
    }

    log(f"Liquiditäts-Trend: Net ${result['current_net_liquidity_T']}T, "
        f"Trend: {trend}, Treiber: {main_driver}")
    for label in ["1w", "4w", "13w", "26w", "52w"]:
        if label in changes:
            log(f"  {label}: {changes[label]['change_B']:+.1f}B ({changes[label]['change_pct']:+.1f}%)")

    return result


# ═══════════════════════════════════════════════════════════════
# 2. ROLLING 30d EVENT-AGGREGATION
# Spec TEIL4 §25.3
# ═══════════════════════════════════════════════════════════════

def aggregate_rolling_events():
    """Aggregiert Events der letzten 30 Tage aus Daily History für Cluster- und Trend-Analyse."""
    cutoff = (NOW - timedelta(days=30)).strftime("%Y-%m-%d")

    all_events = []
    all_surprises = []
    all_threats = []
    all_signals = []
    files_read = 0

    if not os.path.exists(CC_DAILY_HISTORY_DIR):
        log(f"Daily History Dir nicht vorhanden: {CC_DAILY_HISTORY_DIR}")
        return {
            "window_days": 30, "total_events": 0, "total_surprises": 0,
            "total_threats": 0, "total_signals": 0, "building_trends": [],
            "surprise_distribution": {"HOT": 0, "COLD": 0, "INLINE": 0},
            "files_read": 0,
        }

    for filename in sorted(os.listdir(CC_DAILY_HISTORY_DIR)):
        if not filename.startswith("command_center_") or not filename.endswith(".json"):
            continue

        # Datum aus Dateiname extrahieren
        date_part = filename.replace("command_center_", "").replace(".json", "")
        if date_part < cutoff:
            continue

        filepath = os.path.join(CC_DAILY_HISTORY_DIR, filename)
        data = load_json_safe(filepath)
        if not data:
            continue
        files_read += 1

        event_date = data.get("date", date_part)

        # Events sammeln (gestern + heute)
        for ev in data.get("calendar", {}).get("yesterday", []):
            ev["source_date"] = event_date
            all_events.append(ev)

        # Surprises sammeln
        for s in data.get("surprises", {}).get("yesterday_surprises", []):
            s["source_date"] = event_date
            all_surprises.append(s)

        # Intelligence Threats und Signals
        intel = data.get("intelligence") or {}
        for t in intel.get("threats", []):
            t["source_date"] = event_date
            all_threats.append(t)
        for s in intel.get("signals", []):
            s["source_date"] = event_date
            all_signals.append(s)

    # Cluster-Analyse: wiederkehrende Surprise-Richtungen pro Event-Typ
    surprise_directions = {}
    for s in all_surprises:
        event_type = s.get("event", "unknown")
        direction = s.get("direction", "INLINE")
        if event_type not in surprise_directions:
            surprise_directions[event_type] = []
        surprise_directions[event_type].append(direction)

    # Trend-Erkennung: gleiche Richtung 2+ Mal
    building_trends = []
    for event_type, directions in surprise_directions.items():
        if len(directions) >= 2:
            non_inline = [d for d in directions if d not in ("INLINE", "")]
            if len(non_inline) >= 2 and len(set(non_inline)) == 1:
                building_trends.append({
                    "event_type": event_type,
                    "direction": non_inline[0],
                    "streak": len(non_inline),
                    "total_readings": len(directions),
                    "interpretation": f"{event_type} {non_inline[0]} {len(non_inline)}x in 30d — Trend bildet sich",
                })

    # Surprise-Verteilung
    hot_keywords = ("HOT", "HAWKISH", "STRONG", "ABOVE")
    cold_keywords = ("COLD", "DOVISH", "WEAK", "BELOW")
    dist = {
        "HOT": sum(1 for s in all_surprises if s.get("direction", "").upper() in hot_keywords),
        "COLD": sum(1 for s in all_surprises if s.get("direction", "").upper() in cold_keywords),
        "INLINE": sum(1 for s in all_surprises if s.get("direction", "").upper() in ("INLINE", "")),
    }

    result = {
        "window_days": 30,
        "total_events": len(all_events),
        "total_surprises": len(all_surprises),
        "total_threats": len(all_threats),
        "total_signals": len(all_signals),
        "building_trends": building_trends,
        "surprise_distribution": dist,
        "files_read": files_read,
    }

    log(f"Rolling 30d: {files_read} Tage gelesen, "
        f"{len(all_events)} Events, {len(all_surprises)} Surprises, "
        f"{len(all_threats)} Threats, {len(all_signals)} Signals")
    if building_trends:
        for bt in building_trends:
            log(f"  Trend: {bt['event_type']} {bt['direction']} {bt['streak']}x")

    return result


# ═══════════════════════════════════════════════════════════════
# 3. CLUSTER-PROMOTION
# Spec TEIL4 §25.4
# ═══════════════════════════════════════════════════════════════

def promote_clusters(building_trends):
    """Promotet tägliche Cluster zu wöchentlichen Trends.
    Streak >= 3 = CONFIRMED_TREND, Streak == 2 = EMERGING_TREND."""

    promoted = []

    for trend in building_trends:
        if trend["streak"] >= 3:
            promoted.append({
                "trend": trend["event_type"],
                "direction": trend["direction"],
                "streak": trend["streak"],
                "status": "CONFIRMED_TREND",
                "interpretation": f"{trend['event_type']} zeigt konsistentes {trend['direction']} Pattern — "
                                  f"kein Einzelevent mehr, sondern Regime-Signal",
            })
        elif trend["streak"] == 2:
            promoted.append({
                "trend": trend["event_type"],
                "direction": trend["direction"],
                "streak": trend["streak"],
                "status": "EMERGING_TREND",
                "interpretation": f"{trend['event_type']} {trend['direction']} 2x in 30d — beobachten",
            })

    log(f"Cluster-Promotion: {len(promoted)} Trends "
        f"({sum(1 for p in promoted if p['status'] == 'CONFIRMED_TREND')} confirmed, "
        f"{sum(1 for p in promoted if p['status'] == 'EMERGING_TREND')} emerging)")

    return promoted


# ═══════════════════════════════════════════════════════════════
# 4. SYSTEM-BIAS-CHECK (deterministisch)
# Spec TEIL4 §27
# ═══════════════════════════════════════════════════════════════

def compute_system_bias_check(alignment, liquidity_trend, vol_signal):
    """Identifiziert blinde Flecken basierend auf System-Charakteristiken.
    Kein LLM — rein deterministisch."""

    biases = []

    systems = alignment.get("systems", {})

    # BIAS 1: V16 Momentum-Lag
    v16_dir = systems.get("V16", {}).get("direction", "UNKNOWN")
    me_dir = systems.get("MacroEvents", {}).get("direction", "UNKNOWN")
    if v16_dir == "BULLISH" and me_dir == "BEARISH":
        biases.append({
            "bias": "V16_MOMENTUM_LAG",
            "description": "V16 ist bullish aber tägliche Indikatoren zeigen Stress. "
                          "V16 reagiert auf monatliche Daten — könnte 4-8 Wochen hinter der Realität sein.",
            "severity": "HIGH",
            "affected_system": "V16",
        })

    # BIAS 2: High Alignment Blind Spot
    alignment_score = alignment.get("score", 0)
    if alignment_score > 0.85:
        biases.append({
            "bias": "HIGH_ALIGNMENT_BLIND_SPOT",
            "description": "Alle Systeme stimmen überein (Alignment > 0.85). "
                          "Kann korrekt sein — oder alle übersehen dasselbe Risiko.",
            "severity": "MODERATE",
            "affected_system": "ALL",
        })

    # BIAS 3: US-Centric (permanent)
    biases.append({
        "bias": "US_CENTRIC",
        "description": "Systeme sind US-lastig. Japan Carry Trade, China Property, "
                      "Europe Sovereign Stress nur im Weekly Unpriced Risk Scan.",
        "severity": "LOW",
        "affected_system": "ALL",
    })

    # BIAS 4: Vol-Suppression Complacency
    if vol_signal in ("EXTREME_COMPRESSION", "HIGH_COMPRESSION"):
        biases.append({
            "bias": "VOL_SUPPRESSION_COMPLACENCY",
            "description": "Volatilität auf Extrem-Tief. Systeme sehen 'Ruhe' — "
                          "aber die Ruhe selbst ist das Risiko (gespannte Feder).",
            "severity": "HIGH",
            "affected_system": "V16, Cycles",
        })

    # BIAS 5: Periodizitäts-Annahme
    cycles_dir = systems.get("Cycles", {}).get("direction", "UNKNOWN")
    if cycles_dir != "NEUTRAL" and cycles_dir != "UNAVAILABLE":
        biases.append({
            "bias": "PERIODICITY_ASSUMPTION",
            "description": "Cycles Circle nimmt an dass Zyklen sich wiederholen. "
                          "Strukturbrüche können historische Muster ungültig machen.",
            "severity": "LOW",
            "affected_system": "Cycles",
        })

    # BIAS 6: Liquiditäts-Divergenz (V16 bullish aber Liquidität kontrahiert)
    liq_trend = liquidity_trend.get("trend", "UNKNOWN")
    if v16_dir == "BULLISH" and liq_trend == "CONTRACTING":
        biases.append({
            "bias": "LIQUIDITY_DIVERGENCE",
            "description": "V16 bullish aber Net Liquidity kontrahiert. "
                          "Historisch führt Liquiditätsentzug um 4-12 Wochen.",
            "severity": "HIGH",
            "affected_system": "V16",
        })

    result = {
        "biases": biases,
        "n_high": sum(1 for b in biases if b["severity"] == "HIGH"),
        "n_moderate": sum(1 for b in biases if b["severity"] == "MODERATE"),
        "n_low": sum(1 for b in biases if b["severity"] == "LOW"),
    }

    log(f"Bias-Check: {result['n_high']} HIGH, {result['n_moderate']} MODERATE, {result['n_low']} LOW")
    for b in biases:
        if b["severity"] in ("HIGH", "MODERATE"):
            log(f"  {b['severity']}: {b['bias']}")

    return result


# ═══════════════════════════════════════════════════════════════
# 5. TIMELINE-KONVERGENZ-FORECAST (30 Tage voraus)
# Spec TEIL4 §28
# ═══════════════════════════════════════════════════════════════

def forecast_timeline_convergence_30d():
    """Schaut 30 Tage voraus. Identifiziert Wochen mit hoher Event-Dichte.
    Nutzt statische Kalender-Events (FOMC, OPEX, Quarter-End)."""

    today = NOW.date()
    weeks = []

    # Bekannte Termine (statisch — die wichtigsten)
    # FOMC 2026 Termine (geschätzt basierend auf üblichem Muster)
    fomc_dates = [
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ]
    ecb_dates = [
        "2026-01-22", "2026-03-12", "2026-04-30", "2026-06-11",
        "2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17",
    ]

    for w in range(5):  # 5 Wochen voraus
        week_start = today + timedelta(days=w * 7)
        week_end = week_start + timedelta(days=6)

        week_events = []
        week_impact = 0

        # FOMC
        for fd in fomc_dates:
            try:
                fd_date = datetime.strptime(fd, "%Y-%m-%d").date()
                if week_start <= fd_date <= week_end:
                    week_events.append({
                        "event": "FOMC Rate Decision",
                        "date": fd,
                        "impact_score": 10,
                        "timeline": "Monetary Policy",
                    })
                    week_impact += 10
            except ValueError:
                pass

        # ECB
        for ed in ecb_dates:
            try:
                ed_date = datetime.strptime(ed, "%Y-%m-%d").date()
                if week_start <= ed_date <= week_end:
                    week_events.append({
                        "event": "ECB Rate Decision",
                        "date": ed,
                        "impact_score": 8,
                        "timeline": "Monetary Policy EU",
                    })
                    week_impact += 8
            except ValueError:
                pass

        # OPEX: 3. Freitag des Monats
        for d in range(7):
            check_date = week_start + timedelta(days=d)
            if check_date.weekday() == 4:  # Freitag
                first_day = check_date.replace(day=1)
                first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
                third_friday = first_friday + timedelta(days=14)
                if check_date == third_friday:
                    week_events.append({
                        "event": "Options Expiry (OPEX)",
                        "date": check_date.isoformat(),
                        "impact_score": 6,
                        "timeline": "Market Structure",
                    })
                    week_impact += 6

            # Quarter-End
            if check_date.month in (3, 6, 9, 12) and check_date.day >= 25:
                week_events.append({
                    "event": "Quarter-End Window",
                    "date": check_date.isoformat(),
                    "impact_score": 5,
                    "timeline": "Market Structure",
                })
                week_impact += 5

        # NFP: erster Freitag des Monats
        for d in range(7):
            check_date = week_start + timedelta(days=d)
            if check_date.weekday() == 4:  # Freitag
                first_day = check_date.replace(day=1)
                first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
                if check_date == first_friday:
                    week_events.append({
                        "event": "NFP (Non-Farm Payrolls)",
                        "date": check_date.isoformat(),
                        "impact_score": 8,
                        "timeline": "Employment",
                    })
                    week_impact += 8

        # CPI: typischerweise 10.-14. des Monats (vereinfacht: 12.)
        for d in range(7):
            check_date = week_start + timedelta(days=d)
            if check_date.day in (10, 11, 12, 13) and check_date.weekday() < 5:
                week_events.append({
                    "event": "CPI Release Window",
                    "date": check_date.isoformat(),
                    "impact_score": 9,
                    "timeline": "Inflation",
                })
                week_impact += 9
                break  # Nur einmal pro Woche

        n_timelines = len(set(ev.get("timeline", "") for ev in week_events))

        weeks.append({
            "week_number": w + 1,
            "start": week_start.isoformat(),
            "end": week_end.isoformat(),
            "n_events": len(week_events),
            "total_impact": round(week_impact, 1),
            "n_timelines": n_timelines,
            "is_heavy": week_impact > 20 or n_timelines >= 3,
            "top_events": sorted(week_events, key=lambda e: e.get("impact_score", 0), reverse=True)[:5],
        })

    # Convergence-Wochen identifizieren
    convergence_weeks = [w for w in weeks if w["is_heavy"]]

    result = {
        "forecast_weeks": weeks,
        "convergence_weeks": len(convergence_weeks),
        "worst_week": max(weeks, key=lambda w: w["total_impact"]) if weeks else None,
        "next_30d_total_impact": sum(w["total_impact"] for w in weeks),
    }

    log(f"Timeline-Forecast 30d: {len(convergence_weeks)} Heavy Weeks, "
        f"Total Impact: {result['next_30d_total_impact']}")
    if result["worst_week"]:
        ww = result["worst_week"]
        log(f"  Worst Week: KW{ww['week_number']} ({ww['start']}) — "
            f"Impact {ww['total_impact']}, {ww['n_events']} Events")

    return result


# ═══════════════════════════════════════════════════════════════
# 6. SIGNAL-PERFORMANCE-REVIEW
# Spec TEIL4 §25.5 (vereinfacht für V1)
# ═══════════════════════════════════════════════════════════════

def compute_signal_review():
    """Prüft vergangene Threats/Signals: Waren sie korrekt?

    V1 Ansatz (vereinfacht): Zählt Threats aus den letzten 30 Tagen
    und prüft ob der Markt (SPY) danach gefallen ist.
    Vollständige Base-Rate-Kalibrierung erst in V2."""

    if not os.path.exists(CC_DAILY_HISTORY_DIR):
        return {"total_reviewed": 0, "note": "Keine Daily History verfügbar"}

    threats_with_outcome = []
    cutoff_30d = (NOW - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_7d = (NOW - timedelta(days=7)).strftime("%Y-%m-%d")

    for filename in sorted(os.listdir(CC_DAILY_HISTORY_DIR)):
        if not filename.startswith("command_center_") or not filename.endswith(".json"):
            continue
        date_part = filename.replace("command_center_", "").replace(".json", "")

        # Nur Threats älter als 7 Tage (damit wir Outcome sehen können)
        if date_part < cutoff_30d or date_part > cutoff_7d:
            continue

        filepath = os.path.join(CC_DAILY_HISTORY_DIR, filename)
        data = load_json_safe(filepath)
        if not data:
            continue

        intel = data.get("intelligence") or {}
        threats = intel.get("threats", [])
        for t in threats:
            threats_with_outcome.append({
                "date": date_part,
                "title": t.get("title", "?"),
                "confidence": t.get("confidence", "?"),
                # Outcome wird erst in V2 automatisch geprüft
                # (braucht Preis-Daten nach dem Threat-Datum)
                "outcome": "PENDING_REVIEW",
            })

    result = {
        "total_reviewed": len(threats_with_outcome),
        "threats_last_30d": threats_with_outcome[:10],  # Max 10 für Übersicht
        "note": "V1: Outcomes werden manuell bewertet. Automatische Base-Rate-Kalibrierung in V2.",
    }

    log(f"Signal-Review: {len(threats_with_outcome)} Threats in letzten 30d (7d+ alt)")

    return result


# ═══════════════════════════════════════════════════════════════
# 7. ALIGNMENT LADEN (aus letztem Daily Output)
# ═══════════════════════════════════════════════════════════════

def load_latest_daily_output():
    """Lade den neuesten Daily Output für Alignment, Vol-Compression etc."""
    daily_output_path = os.path.join(CC_DATA_DIR, "command_center.json")
    data = load_json_safe(daily_output_path)
    if data:
        log(f"Daily Output geladen: {data.get('date', '?')}")
    else:
        log("Kein Daily Output verfügbar — Bias-Check degradiert")
    return data


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# ETAPPE C TEIL 2: INTELLIGENCE LAYER (3 LLM Calls)
# Spec TEIL3 §18-20, TEIL4 §26
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════


# ── System-JSONs laden (gleiche Struktur wie Agent) ──

def load_system_jsons():
    """Lade alle System-JSONs für den System-Kontext.
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


def build_weekly_system_context(systems):
    """Baut kompakten System-Kontext-Text für die Weekly LLM-Prompts.
    Gleiche Struktur wie build_system_context() im Agent (Spec TEIL3 §18.2-18.3)."""

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
        if bull:
            lines.append(f"  Bullish: {', '.join(bull[:6])}")
        if bear:
            lines.append(f"  Bearish: {', '.join(bear[:6])}")
        ext = oa.get("extended_cycles", [])
        if ext:
            lines.append(f"  Extended: {', '.join(ext[:6])}")
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
            lines.append(f"\nEXTREME RATIOS ({len(extreme)} Paare |Z|>=1.5):")
            for r in extreme[:5]:
                z = r["analysis"]["z_full"]
                hl = r["analysis"].get("halflife")
                hl_str = f", HL={hl:.0f}d" if hl else ""
                lines.append(f"  {r['pair']} ({r.get('description', '?')}): Z={z:+.2f} -> "
                             f"{r['analysis'].get('signal', '?')}{hl_str}")
        else:
            lines.append("\nRATIOS: Keine Extremwerte (|Z|>=1.5)")
    else:
        lines.append("\nRATIOS: Daten nicht verfügbar")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# LLM PROMPTS (Spec TEIL3 §19.3, §19.4, §19.5)
# ═══════════════════════════════════════════════════════════════

WEEKLY_NARRATIVE_PROMPT = """Du bist der Narrative Analyst von Baldur Creek Capital,
einem systematischen Macro Hedgefund mit 18 Investment Circles.

Dein Job: Identifiziere das DOMINANTE MACRO-NARRATIV und ob es sich verschoben hat.

Narrative sind die maechtigste Kraft an den Maerkten. "Inflation is transitory" ->
"Higher for longer" -> "Soft landing" -> "No landing" -> "Stagflation".
Jeder Shift bewegt die Maerkte mehr als jeder einzelne Datenpunkt.

AUFGABE:
1. Was ist das aktuell dominante Narrativ? (1 Satz)
2. Hat es sich in den letzten 7 Tagen verschoben? Wenn ja: wovon -> wohin?
3. Welche Datenpunkte STUETZEN das Narrativ? (mindestens 3, mit Quellen)
4. Welche Datenpunkte WIDERSPRECHEN dem Narrativ? (mindestens 3, mit Quellen)
   Das ist der Anti-Narrativ-Check — suche nach QUALITATIVEN Gegenbeweisen.
5. Wie positioniert ist der Markt relativ zum Narrativ?
   (Ueberfuellter Trade? Contrarian Opportunity?)

REGELN:
- Suche aktiv im Web nach Evidenz FUER und GEGEN das Narrativ.
- Mindestens 10 verschiedene Suchbegriffe.
- Narrative Shifts passieren nicht ueber Nacht. Suche nach RISSEN im aktuellen Narrativ.
- Jede Faktenbehauptung braucht Evidenz.
- Unterscheide FACT / INFERENCE / SPECULATION klar.
- Keine generischen Aussagen. Konkrete Daten, konkrete Assets, konkrete Zeitrahmen.
- Formuliere Web-Search-Ergebnisse in eigenen Worten. Keine Zitate.
- KEINE <cite> Tags, KEINE XML Tags, KEINE Markdown. Nur reiner Text in den JSON-Werten.

Antworte NUR mit validem JSON (kein Markdown, keine Backticks). Schema:
{
  "current_narrative": {
    "title": "Kurzer Titel des Narrativs (z.B. 'Soft Landing Consensus')",
    "description": "2-3 Saetze: Was glaubt der Markt gerade und warum",
    "since_when": "Seit wann dominiert dieses Narrativ (z.B. 'Seit Mitte Februar 2026')",
    "strength": "DOMINANT|EMERGING|FADING"
  },
  "narrative_shift": {
    "shifted": false,
    "from": "Vorheriges Narrativ oder null",
    "to": "Neues Narrativ oder null",
    "trigger_event": "Was hat den Shift ausgeloest oder null",
    "shift_confidence": "HIGH|MEDIUM|LOW"
  },
  "supporting_evidence": [
    {"fact": "Konkreter Datenpunkt", "source": "Publikation + Datum", "date": "2026-03-XX"}
  ],
  "contradicting_evidence": [
    {"fact": "Konkreter Gegenbeweis", "source": "Publikation + Datum", "date": "2026-03-XX",
     "why_important": "Warum das fuer das Narrativ relevant ist"}
  ],
  "market_positioning": {
    "crowded_trade": "Welcher Trade ist ueberfuellt relativ zum Narrativ",
    "contrarian_opportunity": "Welche Contrarian-Position ergibt sich",
    "positioning_risk": "HIGH|MEDIUM|LOW"
  },
  "narrative_risk": "2-3 Saetze: Was passiert wenn das Narrativ kippt? Welche Assets bewegen sich am meisten?"
}"""


WEEKLY_UNPRICED_RISK_PROMPT = """Du bist der Blind-Spot-Detektor von Baldur Creek Capital,
einem systematischen Macro Hedgefund mit 18 Investment Circles.

Dein Job: Finde was NICHT im Kalender steht und trotzdem alles aendern koennte.

Der Markt ist effizient fuer bekannte Risiken. Er ist KATASTROPHAL bei:
- Slow-Burn Risks: Dinge die sich ueber Monate aufbauen
- Korrelationsbrueche: Annahmen die ploetzlich nicht mehr gelten
- Non-linear Tipping Points: Alles ist OK bis ploetzlich nichts mehr OK ist
- Stille als Signal: Was HAETTE passieren sollen aber NICHT passiert ist

PFLICHT-SCANS (immer durchfuehren):
1. Private Credit Stress: BDC NAV Discounts, Redemption Stops, Default Rates
2. Japan: BOJ Balance Sheet + Yen Level + JGB Yields + Carry Trade Risk
3. China: Property Sales, LGFV Stress, Capital Outflows, FX Reserves
4. Europe: TARGET2, Bank CDS, Sovereign Spreads (IT, GR, ES)
5. US Commercial Real Estate: Office Vacancy, Loan Maturities, Regional Banks
6. Geopolitisch: Hormuz, Taiwan, Sanctions, Trade Wars
7. Grosse Policy-Releases: IMF/Weltbank Reports, Strategiepapiere

REFLEXIVITAETS-CHECK:
Gibt es aktuell einen FEEDBACK LOOP der sich aufschaukelt?
- Dollar steigt -> EM Stress -> Kapitalflucht -> Dollar steigt mehr
- Zinsen steigen -> Immobilien fallen -> Banken Verluste -> Kreditvergabe sinkt
- VIX faellt -> weniger Hedging -> groessere Exposure -> naechster Schock wird schlimmer

STILLE-CHECK:
Was haette passieren sollen aber nicht? Beispiele:
- BOJ interveniert nicht bei bestimmtem Yen-Level (warum nicht?)
- Fed erwaehnt Financial Stability nicht in Minutes (warum nicht?)
- China verkuendet keinen neuen Stimulus trotz schwacher Daten (warum nicht?)

REGELN:
- Jede Faktenbehauptung braucht Evidenz.
- Unterscheide FACT / INFERENCE / SPECULATION klar.
- Keine generischen Aussagen. Konkrete Daten, konkrete Assets, konkrete Zeitrahmen.
- Formuliere Web-Search-Ergebnisse in eigenen Worten. Keine Zitate.
- KEINE <cite> Tags, KEINE XML Tags, KEINE Markdown. Nur reiner Text in den JSON-Werten.

Antworte NUR mit validem JSON (kein Markdown, keine Backticks). Schema:
{
  "slow_burns": [
    {
      "topic": "Kurzer Titel",
      "current_status": "2-3 Saetze: Aktueller Stand mit konkreten Zahlen",
      "trend_direction": "WORSENING|STABLE|IMPROVING",
      "tipping_point": "Was wuerde den Kipppunkt markieren",
      "time_to_tipping": "Geschaetzte Zeit bis Kipppunkt",
      "affected_assets": ["SPY", "HYG"],
      "sources": [{"publication": "...", "date": "..."}],
      "severity": "CRITICAL|HIGH|MODERATE|LOW"
    }
  ],
  "feedback_loops": [
    {
      "loop_description": "A -> B -> C -> A (vollstaendige Kausalkette in 2-3 Saetzen)",
      "current_stage": "Wo im Loop sind wir gerade",
      "is_accelerating": false,
      "break_condition": "Was wuerde den Loop stoppen",
      "severity": "CRITICAL|HIGH|MODERATE"
    }
  ],
  "silence_signals": [
    {
      "expected_event": "Was haette passieren sollen",
      "why_notable": "Warum ist das Ausbleiben bemerkenswert",
      "possible_explanations": ["Erklaerung 1", "Erklaerung 2"],
      "implication": "Was bedeutet das fuer das Portfolio"
    }
  ],
  "major_releases_missed": [
    {
      "release": "Name des Reports/Releases",
      "by_whom": "Institution",
      "date": "2026-03-XX",
      "summary": "Kernaussage in 1-2 Saetzen",
      "portfolio_relevance": "Warum relevant fuer uns"
    }
  ],
  "correlation_breaks": [
    {
      "pair": "Asset A / Asset B",
      "normal_correlation": "Beschreibung der normalen Beziehung",
      "current_behavior": "Was passiert gerade anders",
      "last_time_this_happened": "Historische Parallele",
      "what_followed": "Was folgte damals"
    }
  ]
}"""


WEEKLY_POSITIONING_PROMPT = """Du bist der Positioning & Macro Machine Analyst von
Baldur Creek Capital, einem systematischen Macro Hedgefund mit 18 Investment Circles.

TEIL 1: POSITIONING
Suche nach aktuellen Daten zu:
1. CFTC Commitment of Traders: Wo sind Spekulanten netto long/short?
   Suche: "CFTC commitment of traders latest", "COT report this week"
2. Fund Flows: Wohin fliesst Geld? (ETF Flows, Bond Funds, Equity Funds)
   Suche: "ETF fund flows weekly", "equity fund flows"
3. Sentiment: AAII Bull/Bear Ratio, Put/Call Ratio Trend
   Suche: "AAII sentiment survey latest", "put call ratio CBOE"
4. Leverage: Margin Debt Trend, Hedge Fund Gross Exposure
   Suche: "FINRA margin debt latest", "hedge fund leverage"

Fuer jede Position: Ist das ein CROWDED TRADE (Contrarian-Signal)?

TEIL 2: MASCHINEN-ZUSTAND
Die Wirtschaft ist eine Maschine mit Zahnraedern. Bewerte den Zustand jedes Zahnrads:

1. Kreditzyklus: Expansion oder Kontraktion?
   (Bank Lending Standards, Credit Growth, Spreads)
2. Fiskal-Impuls: Government Spending beschleunigt oder bremst?
   (Budget Deficit Trend, Fiscal Impulse)
3. Globale Dollar-Liquiditaet: Dollar-Knappheit oder -Ueberfluss?
   (DXY, EM Spreads, Swap Line Nutzung)

REGELN:
- Jede Faktenbehauptung braucht Evidenz.
- Unterscheide FACT / INFERENCE / SPECULATION klar.
- Keine generischen Aussagen. Konkrete Daten, konkrete Assets, konkrete Zeitrahmen.
- Formuliere Web-Search-Ergebnisse in eigenen Worten. Keine Zitate.
- KEINE <cite> Tags, KEINE XML Tags, KEINE Markdown. Nur reiner Text in den JSON-Werten.

Antworte NUR mit validem JSON (kein Markdown, keine Backticks). Schema:
{
  "positioning": {
    "cftc_highlights": [
      {"asset": "S&P 500", "net_position": "Beschreibung", "extreme": true,
       "contrarian_signal": "Was bedeutet das"}
    ],
    "fund_flows": [
      {"category": "US Equity ETFs", "flow": "Beschreibung", "streak_weeks": 5, "notable": true}
    ],
    "sentiment": {
      "aaii_bull_pct": 45,
      "aaii_bear_pct": 25,
      "aaii_signal": "BULLISH_EXTREME|NORMAL|BEARISH_EXTREME",
      "put_call_ratio": 0.85,
      "put_call_signal": "COMPLACENT|NORMAL|FEARFUL"
    },
    "leverage": {
      "margin_debt_trend": "RISING|STABLE|FALLING",
      "margin_debt_level": "Beschreibung mit Zahlen",
      "risk_assessment": "1-2 Saetze"
    },
    "crowded_trades": ["Trade 1", "Trade 2"],
    "sources": [{"publication": "...", "date": "..."}]
  },
  "machine_state": {
    "credit_cycle": {
      "phase": "EXPANSION|LATE_EXPANSION|CONTRACTION|EARLY_RECOVERY",
      "trend": "Beschreibung",
      "evidence": "Konkrete Datenpunkte"
    },
    "fiscal_impulse": {
      "direction": "ACCELERATING|DECELERATING|NEUTRAL",
      "evidence": "Konkrete Datenpunkte"
    },
    "dollar_liquidity": {
      "status": "ABUNDANT|TIGHT|CRISIS",
      "dxy_trend": "Beschreibung",
      "evidence": "Konkrete Datenpunkte"
    },
    "overall_machine": "HEALTHY|SLOWING|STRESSED|BREAKING",
    "main_concern": "1-2 Saetze: Was ist die groesste Sorge"
  }
}"""


# ═══════════════════════════════════════════════════════════════
# LLM CALL FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _make_llm_call(system_prompt, user_message, call_name):
    """Generische LLM Call Funktion mit Web Search.
    Returns: (parsed_json, token_usage) oder (fallback_dict, None) bei Fehler."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log(f"  {call_name}: ANTHROPIC_API_KEY nicht gesetzt — uebersprungen")
        return {"_error": "ANTHROPIC_API_KEY nicht gesetzt", "_skipped": True}, None

    try:
        import anthropic
    except ImportError:
        log(f"  {call_name}: anthropic Paket nicht installiert — uebersprungen")
        return {"_error": "anthropic nicht installiert", "_skipped": True}, None

    log(f"  {call_name}: Claude API Call (mit Web Search)...")
    t0 = time.time()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": user_message}],
        )

        elapsed = time.time() - t0
        log(f"  {call_name}: {elapsed:.1f}s, Stop: {response.stop_reason}")

        # Token-Usage loggen
        usage = response.usage
        if usage:
            log(f"  {call_name}: Tokens: {usage.input_tokens} input, {usage.output_tokens} output")

        # JSON extrahieren
        parsed, raw_text = extract_json_from_llm_response(response)
        log(f"  {call_name}: JSON geparst OK ({len(parsed)} top-level Keys)")

        return parsed, usage

    except json.JSONDecodeError as e:
        elapsed = time.time() - t0
        log(f"  {call_name}: JSON Parse Fehler nach {elapsed:.1f}s: {e}")
        # Versuche raw text zu retten
        try:
            text_parts = []
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    text_parts.append(block.text)
            raw = "\n".join(text_parts)
            log(f"  {call_name}: Raw (erste 300 Zeichen): {raw[:300]}")
        except Exception:
            raw = ""
        return {
            "_error": f"JSON Parse: {str(e)[:100]}",
            "_raw_response": raw[:2000] if raw else "",
        }, None

    except Exception as e:
        elapsed = time.time() - t0
        log(f"  {call_name}: Fehler nach {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return {"_error": str(e)[:200]}, None


def run_narrative_call(system_context_text, previous_narrative, rolling_events, liquidity_trend):
    """LLM Call 2: Weekly Narrative-Analyse (Spec TEIL3 §19.3)."""

    # Previous Narrative für den Prompt
    prev_title = "Noch kein vorheriges Narrativ (erster Run)"
    prev_desc = ""
    if previous_narrative and isinstance(previous_narrative, list) and len(previous_narrative) > 0:
        last = previous_narrative[-1]
        prev_title = last.get("narrative_title", "?")
        prev_desc = f" (Stärke: {last.get('strength', '?')}, Datum: {last.get('date', '?')})"

    # Surprise-Kontext aus Rolling Events
    dist = rolling_events.get("surprise_distribution", {})
    hot = dist.get("HOT", 0)
    cold = dist.get("COLD", 0)

    user_msg = f"""Datum: {TODAY_STR}

=== NARRATIV DER LETZTEN WOCHE ===
{prev_title}{prev_desc}

=== LIQUIDITAETS-TREND (FRED 52W) ===
Trend: {liquidity_trend.get('trend', '?')}
Net Liquidity: ${liquidity_trend.get('current_net_liquidity_T', '?')}T
4W Veraenderung: {liquidity_trend.get('changes', {}).get('4w', {}).get('change_B', '?')}B
13W Veraenderung: {liquidity_trend.get('changes', {}).get('13w', {}).get('change_B', '?')}B
Treiber: {liquidity_trend.get('main_driver_4w', '?')}

=== 30d SURPRISE-VERTEILUNG ===
HOT/Hawkish: {hot}, COLD/Dovish: {cold}

=== INTERNE SYSTEM-DATEN ===
{system_context_text}

AUFGABE: Identifiziere das dominante Macro-Narrativ und ob es sich verschoben hat.
Suche aktiv im Web nach Evidenz FUER und GEGEN das Narrativ."""

    return _make_llm_call(WEEKLY_NARRATIVE_PROMPT, user_msg, "NARRATIVE")


def run_unpriced_risk_call(system_context_text, slow_burn_history, liquidity_trend):
    """LLM Call 3: Weekly Unpriced Risk Scan (Spec TEIL3 §19.4)."""

    # Vorherige Slow-Burns als Kontext
    prev_burns = []
    if slow_burn_history and isinstance(slow_burn_history, dict):
        for topic, data in slow_burn_history.items():
            obs = data.get("observations", [])
            if obs:
                latest = obs[-1]
                prev_burns.append(f"  {topic}: {latest.get('status', '?')} "
                                  f"(Trend: {latest.get('trend', '?')}, "
                                  f"Severity: {latest.get('severity', '?')})")

    prev_burns_text = "\n".join(prev_burns) if prev_burns else "  Keine vorherigen Slow-Burns (erster Run)"

    user_msg = f"""Datum: {TODAY_STR}

=== VORHERIGE SLOW-BURNS (letzter Weekly Run) ===
{prev_burns_text}

=== LIQUIDITAETS-TREND ===
Trend: {liquidity_trend.get('trend', '?')}
Net Liquidity: ${liquidity_trend.get('current_net_liquidity_T', '?')}T

=== INTERNE SYSTEM-DATEN ===
{system_context_text}

AUFGABE: Fuehre alle 7 Pflicht-Scans durch (Private Credit, Japan, China, Europe, US CRE,
Geopolitik, Policy-Releases). Pruefe auf Feedback Loops und Stille-Signale.
Suche aktiv im Web nach aktuellen Daten zu jedem Scan-Bereich."""

    return _make_llm_call(WEEKLY_UNPRICED_RISK_PROMPT, user_msg, "UNPRICED_RISK")


def run_positioning_call(system_context_text, liquidity_trend):
    """LLM Call 4: Weekly Positioning + Machine (Spec TEIL3 §19.5)."""

    user_msg = f"""Datum: {TODAY_STR}

=== LIQUIDITAETS-TREND (FRED — deterministisch berechnet) ===
Trend: {liquidity_trend.get('trend', '?')}
Net Liquidity: ${liquidity_trend.get('current_net_liquidity_T', '?')}T
Fed BS: ${liquidity_trend.get('fed_bs_T', '?')}T
TGA: ${liquidity_trend.get('tga_T', '?')}T
RRP: ${liquidity_trend.get('rrp_T', '?')}T
4W: {liquidity_trend.get('changes', {}).get('4w', {}).get('change_B', '?')}B
13W: {liquidity_trend.get('changes', {}).get('13w', {}).get('change_B', '?')}B
Treiber: {liquidity_trend.get('main_driver_4w', '?')}

=== INTERNE SYSTEM-DATEN ===
{system_context_text}

AUFGABE:
1. Suche nach aktuellen CFTC COT Daten, ETF Fund Flows, AAII Sentiment, Put/Call Ratio, Margin Debt.
2. Bewerte den Maschinen-Zustand: Kreditzyklus, Fiskal-Impuls, Dollar-Liquiditaet.
3. Identifiziere Crowded Trades."""

    return _make_llm_call(WEEKLY_POSITIONING_PROMPT, user_msg, "POSITIONING")


# ═══════════════════════════════════════════════════════════════
# HISTORY TRACKERS (Spec TEIL4 §26.2-26.4)
# ═══════════════════════════════════════════════════════════════

def load_history_file(filepath):
    """Lade History JSON (kann Liste oder Dict sein)."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_history_file(filepath, data):
    """Speichere History JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def update_narrative_history(narrative_result, narrative_history):
    """Speichert Narrativ-Verlauf fuer Shift-Detektion ueber Wochen (Spec TEIL4 §26.2)."""

    history = narrative_history if isinstance(narrative_history, list) else []

    entry = {
        "date": TODAY_STR,
        "narrative_title": narrative_result.get("current_narrative", {}).get("title", "?"),
        "strength": narrative_result.get("current_narrative", {}).get("strength", "?"),
        "shifted": narrative_result.get("narrative_shift", {}).get("shifted", False),
        "shift_from": narrative_result.get("narrative_shift", {}).get("from"),
        "shift_to": narrative_result.get("narrative_shift", {}).get("to"),
    }

    history.append(entry)

    # Nur letzte 52 Wochen behalten
    history = history[-52:]

    return history


def update_slow_burn_tracker(unpriced_result, slow_burn_history):
    """Trackt Slow-Burn Risiken ueber Wochen (Spec TEIL4 §26.3)."""

    history = slow_burn_history if isinstance(slow_burn_history, dict) else {}

    for burn in unpriced_result.get("slow_burns", []):
        topic = burn.get("topic", "unknown")

        if topic not in history:
            history[topic] = {
                "first_seen": TODAY_STR,
                "observations": [],
            }

        history[topic]["observations"].append({
            "date": TODAY_STR,
            "status": burn.get("current_status", "?"),
            "trend": burn.get("trend_direction", "?"),
            "severity": burn.get("severity", "?"),
        })

        # Nur letzte 26 Wochen pro Topic
        history[topic]["observations"] = history[topic]["observations"][-26:]

    # Trend berechnen pro Slow-Burn
    severity_rank = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
    for topic, data in history.items():
        obs = data.get("observations", [])
        if len(obs) >= 3:
            ranks = [severity_rank.get(o.get("severity", ""), 0) for o in obs[-3:]]
            if ranks[-1] > ranks[0]:
                data["overall_trend"] = "ESCALATING"
            elif ranks[-1] < ranks[0]:
                data["overall_trend"] = "DEESCALATING"
            else:
                data["overall_trend"] = "STABLE"
            data["weeks_tracked"] = len(obs)

    return history


def update_feedback_loop_tracker(unpriced_result, loop_history):
    """Trackt identifizierte Feedback Loops ueber Wochen (Spec TEIL4 §26.4)."""

    history = loop_history if isinstance(loop_history, dict) else {}

    for loop in unpriced_result.get("feedback_loops", []):
        loop_id = loop.get("loop_description", "unknown")[:80]  # Truncate fuer Key

        if loop_id not in history:
            history[loop_id] = {
                "first_seen": TODAY_STR,
                "observations": [],
            }

        history[loop_id]["observations"].append({
            "date": TODAY_STR,
            "stage": loop.get("current_stage", "?"),
            "accelerating": loop.get("is_accelerating", False),
            "severity": loop.get("severity", "?"),
        })

        history[loop_id]["observations"] = history[loop_id]["observations"][-26:]

        # Ist der Loop aktiv und beschleunigt sich?
        recent = history[loop_id]["observations"][-3:]
        if len(recent) >= 2 and all(o.get("accelerating") for o in recent):
            history[loop_id]["alert"] = "ACCELERATING_LOOP"
        elif len(recent) >= 2 and not any(o.get("accelerating") for o in recent[-2:]):
            history[loop_id]["alert"] = "LOOP_STALLING"
        else:
            history[loop_id]["alert"] = "ACTIVE"

    return history


# ═══════════════════════════════════════════════════════════════
# ASSEMBLIERUNG + OUTPUT
# ═══════════════════════════════════════════════════════════════

def assemble_weekly_output(liquidity_trend, rolling_events, promoted_clusters,
                            bias_check, timeline_forecast, signal_review,
                            narrative_result, unpriced_result, positioning_result,
                            narrative_history, slow_burn_history, loop_history,
                            llm_calls_made):
    """Assembliert das vollstaendige Weekly Output JSON (Spec TEIL4 §29.1).
    V2.0: LLM-Ergebnisse statt PENDING_LLM Placeholder."""

    # Prüfe ob LLM-Ergebnisse Fehler haben
    narrative_ok = not narrative_result.get("_error") and not narrative_result.get("_skipped")
    unpriced_ok = not unpriced_result.get("_error") and not unpriced_result.get("_skipped")
    positioning_ok = not positioning_result.get("_error") and not positioning_result.get("_skipped")

    etappe = "C_FULL" if (narrative_ok and unpriced_ok and positioning_ok) else "C_PARTIAL"

    output = {
        "metadata": {
            "generated_at": NOW.isoformat(),
            "version": VERSION,
            "llm_calls": llm_calls_made,
            "web_search_calls": llm_calls_made,  # Jeder LLM Call hat Web Search
            "etappe": etappe,
            "narrative_ok": narrative_ok,
            "unpriced_ok": unpriced_ok,
            "positioning_ok": positioning_ok,
        },

        # Maschinen-Zustand (Kern des Weekly)
        "machine_state": {
            "liquidity": liquidity_trend,
            "credit_cycle": positioning_result.get("machine_state", {}).get("credit_cycle",
                            {"status": "LLM_FAILED"}) if positioning_ok else {"status": "LLM_FAILED"},
            "fiscal_impulse": positioning_result.get("machine_state", {}).get("fiscal_impulse",
                              {"status": "LLM_FAILED"}) if positioning_ok else {"status": "LLM_FAILED"},
            "dollar_liquidity": positioning_result.get("machine_state", {}).get("dollar_liquidity",
                                {"status": "LLM_FAILED"}) if positioning_ok else {"status": "LLM_FAILED"},
            "overall": positioning_result.get("machine_state", {}).get("overall_machine",
                       liquidity_trend.get("trend", "UNKNOWN")) if positioning_ok
                       else liquidity_trend.get("trend", "UNKNOWN"),
        },

        # Narrativ (LLM Call 2)
        "narrative": narrative_result if narrative_ok else {
            "status": "LLM_FAILED", "_error": narrative_result.get("_error", "?")
        },
        "narrative_history": narrative_history,

        # Positioning (LLM Call 4)
        "positioning": positioning_result.get("positioning", {}) if positioning_ok else {
            "status": "LLM_FAILED", "_error": positioning_result.get("_error", "?")
        },

        # Unpriced Risks (LLM Call 3)
        "slow_burns": unpriced_result.get("slow_burns", []) if unpriced_ok else [],
        "slow_burn_tracker": slow_burn_history,
        "feedback_loops": unpriced_result.get("feedback_loops", []) if unpriced_ok else [],
        "feedback_loop_tracker": loop_history,
        "silence_signals": unpriced_result.get("silence_signals", []) if unpriced_ok else [],
        "major_releases_missed": unpriced_result.get("major_releases_missed", []) if unpriced_ok else [],
        "correlation_breaks": unpriced_result.get("correlation_breaks", []) if unpriced_ok else [],

        # Trends (deterministisch)
        "rolling_30d_summary": rolling_events,
        "promoted_clusters": promoted_clusters,

        # Forecasts (deterministisch)
        "timeline_forecast_30d": timeline_forecast,

        # System Health (deterministisch)
        "bias_check": bias_check,
        "signal_track_record": signal_review,

        # Telegram (kompakt)
        "telegram_message": _format_weekly_telegram(
            liquidity_trend, rolling_events, promoted_clusters,
            bias_check, timeline_forecast, signal_review,
            narrative_result if narrative_ok else None,
            unpriced_result if unpriced_ok else None,
            positioning_result if positioning_ok else None,
        ),
    }

    return output


def _format_weekly_telegram(liquidity_trend, rolling_events, promoted_clusters,
                             bias_check, timeline_forecast, signal_review,
                             narrative_result=None, unpriced_result=None,
                             positioning_result=None):
    """Formatiert Weekly Summary fuer Telegram (Spec TEIL4 §30).
    V2.0: Erweitert um Narrative, Slow-Burns, Loops."""

    lines = [f"📊 WEEKLY MACRO INTELLIGENCE ({TODAY_STR})\n"]

    # Maschine / Liquiditaet
    trend = liquidity_trend.get("trend", "?")
    net_t = liquidity_trend.get("current_net_liquidity_T", "?")
    c4w = liquidity_trend.get("changes", {}).get("4w", {}).get("change_B", 0)
    lines.append(f"💧 Liquiditaet: {trend} (Net ${net_t}T, 4W: {c4w:+.0f}B)")

    # Machine State (aus Positioning LLM)
    if positioning_result:
        machine = positioning_result.get("machine_state", {})
        overall = machine.get("overall_machine", "?")
        if overall != "?":
            lines.append(f"🔧 Maschine: {overall}")

    # Narrativ (aus Narrative LLM)
    if narrative_result:
        narr = narrative_result.get("current_narrative", {})
        title = narr.get("title", "?")
        strength = narr.get("strength", "?")
        lines.append(f"\n📖 Narrativ: {title} ({strength})")
        shift = narrative_result.get("narrative_shift", {})
        if shift.get("shifted"):
            lines.append(f"⚠️ SHIFT: {shift.get('from', '?')} → {shift.get('to', '?')}")

    # Crowded Trades (aus Positioning LLM)
    if positioning_result:
        crowded = positioning_result.get("positioning", {}).get("crowded_trades", [])
        if crowded:
            lines.append(f"\n🎯 Crowded Trades: {', '.join(str(c) for c in crowded[:3])}")

    # Slow-Burns (aus Unpriced Risk LLM)
    if unpriced_result:
        slow_burns = unpriced_result.get("slow_burns", [])
        critical = [sb for sb in slow_burns if sb.get("severity") == "CRITICAL"]
        high = [sb for sb in slow_burns if sb.get("severity") == "HIGH"]
        if critical or high:
            lines.append(f"\n🔥 Slow-Burns: {len(critical)} CRITICAL, {len(high)} HIGH")
            for sb in (critical + high)[:3]:
                lines.append(f"  • {sb.get('topic', '?')}: {sb.get('trend_direction', '?')}")

        # Feedback Loops
        loops = unpriced_result.get("feedback_loops", [])
        accel = [l for l in loops if l.get("is_accelerating")]
        if accel:
            lines.append(f"\n🔄 Feedback Loops: {len(accel)} beschleunigend")
            for l in accel[:2]:
                desc = l.get("loop_description", "?")
                lines.append(f"  • {desc[:60]}")

    # Rolling Trends
    confirmed = [p for p in promoted_clusters if p["status"] == "CONFIRMED_TREND"]
    emerging = [p for p in promoted_clusters if p["status"] == "EMERGING_TREND"]
    if confirmed:
        lines.append(f"\n📈 Trends: {len(confirmed)} bestaetigt")
        for c in confirmed[:3]:
            lines.append(f"  • {c['trend']} {c['direction']} ({c['streak']}x)")
    if emerging:
        lines.append(f"🔍 Emerging: {len(emerging)}")
        for e in emerging[:2]:
            lines.append(f"  • {e['trend']} {e['direction']}")

    # 30d Surprises
    dist = rolling_events.get("surprise_distribution", {})
    hot = dist.get("HOT", 0)
    cold = dist.get("COLD", 0)
    if hot + cold > 0:
        bias = "HOT-Bias" if hot > cold * 1.5 else ("COLD-Bias" if cold > hot * 1.5 else "Ausgeglichen")
        lines.append(f"\n🌡️ 30d Surprises: {hot} HOT / {cold} COLD → {bias}")

    # Forecast
    worst = timeline_forecast.get("worst_week")
    if worst and worst.get("is_heavy"):
        lines.append(f"\n📅 Heavy Week: KW{worst['week_number']} ab {worst['start']} "
                     f"({worst['n_events']} Events, Impact {worst['total_impact']})")

    # Bias
    n_high = bias_check.get("n_high", 0)
    if n_high > 0:
        lines.append(f"\n⚠️ {n_high} HIGH-Severity Blind Spots aktiv")

    # Signal Review
    n_reviewed = signal_review.get("total_reviewed", 0)
    if n_reviewed > 0:
        lines.append(f"\n📋 {n_reviewed} Threats im Review (30d)")

    return "\n".join(lines)


def write_weekly_outputs(output, narrative_history, slow_burn_history, loop_history):
    """Schreibt Weekly JSON Output + History-Dateien."""
    os.makedirs(CC_DATA_DIR, exist_ok=True)

    # Haupt-Output
    with open(CC_WEEKLY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(CC_WEEKLY_OUTPUT)
    log(f"Weekly Output: {CC_WEEKLY_OUTPUT} ({size:,} bytes)")

    # Narrative History
    save_history_file(CC_NARRATIVE_HISTORY, narrative_history)
    log(f"Narrative History: {len(narrative_history)} Eintraege")

    # Slow-Burn History
    save_history_file(CC_SLOW_BURN_HISTORY, slow_burn_history)
    log(f"Slow-Burn History: {len(slow_burn_history)} Topics")

    # Feedback-Loop History
    save_history_file(CC_LOOP_HISTORY, loop_history)
    log(f"Loop History: {len(loop_history)} Loops")


# ═══════════════════════════════════════════════════════════════
# RUN WEEKLY — Aufgerufen von command_center_agent.py main()
# ═══════════════════════════════════════════════════════════════

def run_weekly(skip_write=False):
    """Hauptfunktion fuer Weekly Run.
    Aufgerufen von command_center_agent.py wenn --mode weekly.

    V2.0: 6 deterministische Berechnungen + 3 LLM Calls + History Tracker."""

    t0 = time.time()
    print("=" * 70)
    print(f"COMMAND CENTER — WEEKLY RUN {VERSION}")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Etappe C FULL: Daten-Layer + 3 LLM Calls")
    print(f"  Flags: skip-write={skip_write}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # TEIL 1: DETERMINISTISCHER DATEN-LAYER (wie V1.0)
    # ═══════════════════════════════════════════════════════════

    # ─── 1. Liquiditäts-Trend ───
    print(f"\n{'─'*50}")
    print("WEEKLY 1: LIQUIDITAETS-TREND (FRED 52W)")
    print(f"{'─'*50}")
    fred_data = fetch_liquidity_timeseries()
    liquidity_trend = compute_liquidity_trend(fred_data)

    # ─── 2. Rolling 30d Aggregation ───
    print(f"\n{'─'*50}")
    print("WEEKLY 2: ROLLING 30d EVENT-AGGREGATION")
    print(f"{'─'*50}")
    rolling_events = aggregate_rolling_events()

    # ─── 3. Cluster-Promotion ───
    print(f"\n{'─'*50}")
    print("WEEKLY 3: CLUSTER-PROMOTION")
    print(f"{'─'*50}")
    building_trends = rolling_events.get("building_trends", [])
    promoted_clusters = promote_clusters(building_trends)

    # ─── 4. Bias-Check ───
    print(f"\n{'─'*50}")
    print("WEEKLY 4: SYSTEM-BIAS-CHECK")
    print(f"{'─'*50}")
    daily_output = load_latest_daily_output()
    alignment = daily_output.get("alignment", {}) if daily_output else {}
    vol_signal = ""
    if daily_output:
        vol_signal = daily_output.get("vol_compression", {}).get("signal", "")
    bias_check = compute_system_bias_check(alignment, liquidity_trend, vol_signal)

    # ─── 5. Timeline-Forecast ───
    print(f"\n{'─'*50}")
    print("WEEKLY 5: TIMELINE-KONVERGENZ-FORECAST (30d)")
    print(f"{'─'*50}")
    timeline_forecast = forecast_timeline_convergence_30d()

    # ─── 6. Signal-Performance-Review ───
    print(f"\n{'─'*50}")
    print("WEEKLY 6: SIGNAL-PERFORMANCE-REVIEW")
    print(f"{'─'*50}")
    signal_review = compute_signal_review()

    # ═══════════════════════════════════════════════════════════
    # TEIL 2: INTELLIGENCE LAYER (3 LLM Calls)
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'─'*50}")
    print("WEEKLY 7: SYSTEM-KONTEXT BAUEN")
    print(f"{'─'*50}")
    log("Lade System-JSONs...")
    systems = load_system_jsons()
    n_available = sum(1 for v in systems.values() if v is not None)
    log(f"  {n_available}/{len(systems)} Systeme verfuegbar")
    system_context_text = build_weekly_system_context(systems)
    log(f"  Kontext: {len(system_context_text)} Zeichen")

    # History-Dateien laden
    narrative_history = load_history_file(CC_NARRATIVE_HISTORY) or []
    slow_burn_history = load_history_file(CC_SLOW_BURN_HISTORY) or {}
    loop_history = load_history_file(CC_LOOP_HISTORY) or {}

    llm_calls_made = 0

    # ─── LLM Call 2: Narrative ───
    print(f"\n{'─'*50}")
    print("WEEKLY 8: LLM CALL — NARRATIVE-ANALYSE")
    print(f"{'─'*50}")
    narrative_result, narrative_usage = run_narrative_call(
        system_context_text, narrative_history, rolling_events, liquidity_trend)
    if not narrative_result.get("_error"):
        llm_calls_made += 1
        narrative_history = update_narrative_history(narrative_result, narrative_history)
        log(f"  Narrative History aktualisiert ({len(narrative_history)} Eintraege)")
    else:
        log(f"  Narrative fehlgeschlagen: {narrative_result.get('_error', '?')}")

    # ─── LLM Call 3: Unpriced Risk ───
    print(f"\n{'─'*50}")
    print("WEEKLY 9: LLM CALL — UNPRICED RISK SCAN")
    print(f"{'─'*50}")
    unpriced_result, unpriced_usage = run_unpriced_risk_call(
        system_context_text, slow_burn_history, liquidity_trend)
    if not unpriced_result.get("_error"):
        llm_calls_made += 1
        slow_burn_history = update_slow_burn_tracker(unpriced_result, slow_burn_history)
        loop_history = update_feedback_loop_tracker(unpriced_result, loop_history)
        log(f"  Slow-Burn Tracker: {len(slow_burn_history)} Topics")
        log(f"  Loop Tracker: {len(loop_history)} Loops")
    else:
        log(f"  Unpriced Risk fehlgeschlagen: {unpriced_result.get('_error', '?')}")

    # ─── LLM Call 4: Positioning + Machine ───
    print(f"\n{'─'*50}")
    print("WEEKLY 10: LLM CALL — POSITIONING + MACHINE")
    print(f"{'─'*50}")
    positioning_result, positioning_usage = run_positioning_call(
        system_context_text, liquidity_trend)
    if not positioning_result.get("_error"):
        llm_calls_made += 1
    else:
        log(f"  Positioning fehlgeschlagen: {positioning_result.get('_error', '?')}")

    # ═══════════════════════════════════════════════════════════
    # ASSEMBLIERUNG + OUTPUT
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'─'*50}")
    print("WEEKLY: ASSEMBLIERUNG")
    print(f"{'─'*50}")
    output = assemble_weekly_output(
        liquidity_trend, rolling_events, promoted_clusters,
        bias_check, timeline_forecast, signal_review,
        narrative_result, unpriced_result, positioning_result,
        narrative_history, slow_burn_history, loop_history,
        llm_calls_made,
    )

    # ─── Write ───
    if not skip_write:
        write_weekly_outputs(output, narrative_history, slow_burn_history, loop_history)
    else:
        log("Write uebersprungen (--skip-write)")
        size = len(json.dumps(output, default=str))
        log(f"Output Groesse: {size:,} bytes (nicht geschrieben)")

    # ─── Zusammenfassung ───
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMMAND CENTER — WEEKLY RUN FERTIG ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Liquiditaet:    {liquidity_trend.get('trend', '?')} "
          f"(Net ${liquidity_trend.get('current_net_liquidity_T', '?')}T)")
    print(f"  30d Events:    {rolling_events.get('total_events', 0)} Events, "
          f"{rolling_events.get('total_surprises', 0)} Surprises")
    print(f"  Trends:        {sum(1 for p in promoted_clusters if p['status'] == 'CONFIRMED_TREND')} bestaetigt, "
          f"{sum(1 for p in promoted_clusters if p['status'] == 'EMERGING_TREND')} emerging")
    print(f"  Bias:          {bias_check.get('n_high', 0)} HIGH, "
          f"{bias_check.get('n_moderate', 0)} MODERATE")
    convergence = timeline_forecast.get("convergence_weeks", 0)
    print(f"  30d Forecast:  {convergence} Heavy Weeks")
    print(f"  Signal Review: {signal_review.get('total_reviewed', 0)} Threats reviewed")
    print(f"  LLM Calls:     {llm_calls_made}/3 erfolgreich")

    # LLM Details
    if not narrative_result.get("_error"):
        narr_title = narrative_result.get("current_narrative", {}).get("title", "?")
        narr_strength = narrative_result.get("current_narrative", {}).get("strength", "?")
        print(f"  Narrativ:      {narr_title} ({narr_strength})")
        if narrative_result.get("narrative_shift", {}).get("shifted"):
            print(f"  ⚠️ SHIFT:     {narrative_result['narrative_shift'].get('from', '?')} "
                  f"→ {narrative_result['narrative_shift'].get('to', '?')}")
    else:
        print(f"  Narrativ:      FEHLER — {narrative_result.get('_error', '?')[:80]}")

    if not unpriced_result.get("_error"):
        n_burns = len(unpriced_result.get("slow_burns", []))
        n_loops = len(unpriced_result.get("feedback_loops", []))
        n_silence = len(unpriced_result.get("silence_signals", []))
        print(f"  Unpriced:      {n_burns} Slow-Burns, {n_loops} Loops, {n_silence} Silence")
    else:
        print(f"  Unpriced:      FEHLER — {unpriced_result.get('_error', '?')[:80]}")

    if not positioning_result.get("_error"):
        overall = positioning_result.get("machine_state", {}).get("overall_machine", "?")
        n_crowded = len(positioning_result.get("positioning", {}).get("crowded_trades", []))
        print(f"  Maschine:      {overall}, {n_crowded} Crowded Trades")
    else:
        print(f"  Positioning:   FEHLER — {positioning_result.get('_error', '?')[:80]}")

    print(f"{'='*70}")

    return output
