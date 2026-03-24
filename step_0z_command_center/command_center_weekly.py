#!/usr/bin/env python3
"""
command_center_weekly.py — System Command Center Weekly Run V1.0
================================================================
Baldur Creek Capital | Circle 18 | System Command Center — Etappe C

Weekly Daten-Layer (deterministisch, kein LLM in Teil 1):
  1. Liquiditäts-Trend (52W FRED Serie + Trend-Richtung + Treiber)
  2. Rolling 30d Event-Aggregation (Daily History → Surprise-Trends)
  3. Cluster-Promotion (tägliche Cluster → wöchentliche Trends)
  4. System-Bias-Check (deterministisch, Blind-Spot-Analyse)
  5. Timeline-Konvergenz-Forecast (30 Tage voraus, 5 Wochen)
  6. Signal-Performance-Review (vergangene Threats/Signals prüfen)
  7. Assemblierung + JSON Output

Teil 2 (spätere Session):
  - 3 LLM Calls (Narrative, Unpriced Risk, Positioning)
  - Narrative History + Slow-Burn Tracker + Loop Tracker

Aufgerufen von command_center_agent.py main() wenn --mode weekly.

Spec: MACRO_EVENTS_SPEC_TEIL4.md (§24-30)

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
VERSION = "command_center_weekly V1.0"

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
# ASSEMBLIERUNG + OUTPUT
# ═══════════════════════════════════════════════════════════════

def assemble_weekly_output(liquidity_trend, rolling_events, promoted_clusters,
                            bias_check, timeline_forecast, signal_review):
    """Assembliert das vollständige Weekly Output JSON (Spec TEIL4 §29.1).

    Teil 2 (LLM Calls) wird hier als Placeholder eingefügt.
    Wenn die LLM-Ergebnisse vorliegen, werden sie hier eingesetzt."""

    output = {
        "metadata": {
            "generated_at": NOW.isoformat(),
            "version": VERSION,
            "llm_calls": 0,   # Teil 2: wird 3
            "web_search_calls": 0,  # Teil 2: wird 3
            "etappe": "C_DATA_LAYER",  # Wird "C_FULL" nach Teil 2
        },

        # Maschinen-Zustand (Kern des Weekly)
        "machine_state": {
            "liquidity": liquidity_trend,
            # Teil 2: credit_cycle, fiscal_impulse, dollar_liquidity (aus LLM)
            "credit_cycle": {"status": "PENDING_LLM", "note": "Teil 2 — LLM Positioning Call"},
            "fiscal_impulse": {"status": "PENDING_LLM"},
            "dollar_liquidity": {"status": "PENDING_LLM"},
            "overall": liquidity_trend.get("trend", "UNKNOWN"),
        },

        # Narrativ (Teil 2: LLM Call 2)
        "narrative": {
            "status": "PENDING_LLM",
            "note": "Teil 2 — LLM Narrative-Analyse Call",
        },

        # Positioning (Teil 2: LLM Call 4)
        "positioning": {
            "status": "PENDING_LLM",
            "note": "Teil 2 — LLM Positioning Call",
        },

        # Unpriced Risks (Teil 2: LLM Call 3)
        "slow_burns": [],   # Teil 2: aus LLM
        "feedback_loops": [],  # Teil 2: aus LLM
        "silence_signals": [],  # Teil 2: aus LLM

        # Trends (deterministisch — jetzt verfügbar)
        "rolling_30d_summary": rolling_events,
        "promoted_clusters": promoted_clusters,

        # Forecasts (deterministisch — jetzt verfügbar)
        "timeline_forecast_30d": timeline_forecast,

        # System Health (deterministisch — jetzt verfügbar)
        "bias_check": bias_check,
        "signal_track_record": signal_review,

        # Telegram (kompakt)
        "telegram_message": _format_weekly_telegram(
            liquidity_trend, rolling_events, promoted_clusters,
            bias_check, timeline_forecast, signal_review,
        ),
    }

    return output


def _format_weekly_telegram(liquidity_trend, rolling_events, promoted_clusters,
                             bias_check, timeline_forecast, signal_review):
    """Formatiert Weekly Summary für Telegram (Spec TEIL4 §30)."""

    lines = [f"📊 WEEKLY MACRO INTELLIGENCE ({TODAY_STR})\n"]

    # Maschine / Liquidität
    trend = liquidity_trend.get("trend", "?")
    net_t = liquidity_trend.get("current_net_liquidity_T", "?")
    c4w = liquidity_trend.get("changes", {}).get("4w", {}).get("change_B", 0)
    lines.append(f"💧 Liquidität: {trend} (Net ${net_t}T, 4W: {c4w:+.0f}B)")

    # Rolling Trends
    confirmed = [p for p in promoted_clusters if p["status"] == "CONFIRMED_TREND"]
    emerging = [p for p in promoted_clusters if p["status"] == "EMERGING_TREND"]
    if confirmed:
        lines.append(f"\n📈 Trends: {len(confirmed)} bestätigt")
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


def write_weekly_outputs(output):
    """Schreibt Weekly JSON Output."""
    os.makedirs(CC_DATA_DIR, exist_ok=True)

    with open(CC_WEEKLY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    size = os.path.getsize(CC_WEEKLY_OUTPUT)
    log(f"Weekly Output: {CC_WEEKLY_OUTPUT} ({size:,} bytes)")


# ═══════════════════════════════════════════════════════════════
# RUN WEEKLY — Aufgerufen von command_center_agent.py main()
# ═══════════════════════════════════════════════════════════════

def run_weekly(skip_write=False):
    """Hauptfunktion für Weekly Run.
    Aufgerufen von command_center_agent.py wenn --mode weekly."""

    t0 = time.time()
    print("=" * 70)
    print(f"COMMAND CENTER — WEEKLY RUN {VERSION}")
    print(f"  {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Etappe C: Daten-Layer (deterministisch)")
    print(f"  Flags: skip-write={skip_write}")
    print("=" * 70)

    # ─── 1. Liquiditäts-Trend ───
    print(f"\n{'─'*50}")
    print("WEEKLY 1: LIQUIDITÄTS-TREND (FRED 52W)")
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
    # Alignment + Vol aus letztem Daily Output laden
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

    # ─── Assemblierung ───
    print(f"\n{'─'*50}")
    print("WEEKLY: ASSEMBLIERUNG")
    print(f"{'─'*50}")
    output = assemble_weekly_output(
        liquidity_trend, rolling_events, promoted_clusters,
        bias_check, timeline_forecast, signal_review,
    )

    # ─── Write ───
    if not skip_write:
        write_weekly_outputs(output)
    else:
        log("Write übersprungen (--skip-write)")
        size = len(json.dumps(output, default=str))
        log(f"Output Größe: {size:,} bytes (nicht geschrieben)")

    # ─── Zusammenfassung ───
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMMAND CENTER — WEEKLY RUN FERTIG ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Liquidität:    {liquidity_trend.get('trend', '?')} "
          f"(Net ${liquidity_trend.get('current_net_liquidity_T', '?')}T)")
    print(f"  30d Events:    {rolling_events.get('total_events', 0)} Events, "
          f"{rolling_events.get('total_surprises', 0)} Surprises")
    print(f"  Trends:        {sum(1 for p in promoted_clusters if p['status'] == 'CONFIRMED_TREND')} bestätigt, "
          f"{sum(1 for p in promoted_clusters if p['status'] == 'EMERGING_TREND')} emerging")
    print(f"  Bias:          {bias_check.get('n_high', 0)} HIGH, "
          f"{bias_check.get('n_moderate', 0)} MODERATE")
    convergence = timeline_forecast.get("convergence_weeks", 0)
    print(f"  30d Forecast:  {convergence} Heavy Weeks")
    print(f"  Signal Review: {signal_review.get('total_reviewed', 0)} Threats reviewed")
    print(f"  LLM Calls:     0 (Teil 2 — Narrative/Unpriced/Positioning)")
    print(f"{'='*70}")

    return output
