#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/main.py
Disruptions Agent — Entry Point + Orchestrierung
Frequenz: Woechentlich (Sonntag)
Spec: DISRUPTIONS_AGENT_SPEC TEIL 1-3 + DISRUPTIONS_FRONTEND_ERWEITERUNG_SPEC

Pipeline:
  Stufe 1: Automatisches Screening (alle Kategorien, ~5 min)
  Stufe 2: LLM Deep Dive (Top 5 nach Screening-Score, ~25 min)
  → Score-Berechnung → Exposure Check → Vulnerability Analysis
  → Contrarian Scanner → Causal Chains → Dependencies
  → Intelligence Briefing (Phase A)
  → Sheet Write → Dashboard Update → Git Commit
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone

# --- Module imports (alle im selben Verzeichnis) ---
from screening import run_screening
from deep_dive import run_deep_dive
from scoring import calculate_scores, determine_phases, determine_watchlist_status
from exposure_check import run_exposure_check
from vulnerability_analysis import run_vulnerability_analysis
from contrarian import run_contrarian_scan
from causal_chains import run_causal_chains
from dependencies import update_dependencies, detect_convergence_zones
from intelligence_briefing import generate_intelligence_briefing
from sheet_writer import write_all_tabs
from dashboard_update import update_dashboard_json


# ===== CONFIG =====

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'disruptions_config.json')
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dashboard', 'latest.json')
HISTORY_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'disruptions')
HISTORY_PATH = os.path.join(HISTORY_DIR, 'disruptions_history.json')

TOP_N_DEEP_DIVE = 5


def load_config():
    """Lade disruptions_config.json."""
    config_path = os.path.abspath(CONFIG_PATH)
    print(f"[CONFIG] Lade {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    active_cats = [c for c in config['categories'] if c['status'] == 'ACTIVE']
    print(f"[CONFIG] {len(active_cats)} aktive Kategorien, {len(config['etf_universe'])} ETFs, {len(config['dependency_seed'])} Dependency-Kanten")
    return config


def load_previous_trends():
    """Lade vorherige Trend-Daten aus History fuer Velocity/Acceleration."""
    if not os.path.exists(HISTORY_PATH):
        print("[HISTORY] Keine vorherige History gefunden — erster Lauf")
        return []
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        history = json.load(f)
    print(f"[HISTORY] {len(history)} historische Snapshots geladen")
    return history


def save_history_snapshot(trends, run_date):
    """Speichere woechentlichen Snapshot in disruptions_history.json."""
    os.makedirs(HISTORY_DIR, exist_ok=True)

    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            history = json.load(f)

    snapshot = {
        "date": run_date,
        "trends": [
            {
                "id": t["id"],
                "maturity": t.get("maturity", 0),
                "momentum": t.get("momentum", 0),
                "acceleration": t.get("acceleration", 0),
                "inflection_score": t.get("inflection_score", 0),
                "phase": t.get("phase", "EMERGING"),
                "watchlist_status": t.get("watchlist_status", "WATCH")
            }
            for t in trends
        ]
    }
    history.append(snapshot)
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[HISTORY] Snapshot fuer {run_date} gespeichert ({len(history)} total)")


def load_v16_weights():
    """Lade aktuelle V16-Gewichte aus latest.json."""
    if not os.path.exists(DASHBOARD_PATH):
        print("[V16] latest.json nicht gefunden — verwende leere Gewichte")
        return {}
    with open(DASHBOARD_PATH, 'r', encoding='utf-8') as f:
        dashboard = json.load(f)
    # V16 Gewichte aus verschiedenen moeglichen Pfaden
    v16 = dashboard.get('v16', {})
    weights = v16.get('weights', {})
    if not weights:
        # Fallback: aus signal_generator
        sig = dashboard.get('signal_generator', {})
        v16_trades = sig.get('v16_trades', {})
        weights = v16_trades.get('weights', {})
    if weights:
        non_zero = {k: v for k, v in weights.items() if v != 0}
        print(f"[V16] Gewichte geladen: {len(non_zero)} aktive Positionen")
    else:
        print("[V16] Keine V16-Gewichte gefunden")
    return weights


def load_v16_regime():
    """Lade aktuelles V16 Regime aus latest.json."""
    if not os.path.exists(DASHBOARD_PATH):
        return 'NEUTRAL'
    with open(DASHBOARD_PATH, 'r', encoding='utf-8') as f:
        dashboard = json.load(f)
    # Regime aus verschiedenen moeglichen Pfaden (Prioritaet: v16.regime > header.v16_regime)
    regime = dashboard.get('v16', {}).get('regime', '')
    if not regime:
        regime = dashboard.get('header', {}).get('v16_regime', '')
    if not regime:
        regime = 'NEUTRAL'
    print(f"[V16] Regime: {regime}")
    return regime


def load_regime_history():
    """Lade Regime-History aus disruptions_history.json fuer 4W Sparkline."""
    if not os.path.exists(DASHBOARD_PATH):
        return []
    # Versuche aus vorherigen disruptions Bloecken
    try:
        with open(DASHBOARD_PATH, 'r', encoding='utf-8') as f:
            dashboard = json.load(f)
        existing_history = (
            dashboard.get('disruptions', {})
            .get('regime_context', {})
            .get('regime_history_4w', [])
        )
        if existing_history:
            return existing_history
    except Exception:
        pass
    return []


def main():
    """Hauptorchestierung des Disruptions Agent."""
    start_time = time.time()
    run_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"  DISRUPTIONS AGENT — Woechentlicher Scan")
    print(f"  Datum: {run_date}")
    print(f"{'='*60}\n")

    # --- Vorbereitung ---
    config = load_config()
    active_categories = [c for c in config['categories'] if c['status'] == 'ACTIVE']
    etf_universe = config['etf_universe']
    exposure_map = config['v16_exposure_map']
    dependency_seed = config['dependency_seed']
    thresholds = config['thresholds']
    previous_history = load_previous_trends()
    v16_weights = load_v16_weights()
    v16_regime = load_v16_regime()

    errors = []
    meta = {
        "run_date": run_date,
        "categories_scanned": len(active_categories),
        "deep_dive_count": 0,
        "sources_total": 0,
        "avg_source_quality": 0,
        "llm_model": "claude-sonnet-4-20250514",
        "run_duration_s": 0,
        "errors": []
    }

    # ===== STUFE 1: AUTOMATISCHES SCREENING =====
    print(f"\n--- STUFE 1: Screening ({len(active_categories)} Kategorien) ---")
    try:
        screening_results = run_screening(
            categories=active_categories,
            etf_universe=etf_universe,
            screening_weights=config.get('screening_weights', {})
        )
        print(f"[SCREENING] {len(screening_results)} Kategorien gescannt")
        for sr in sorted(screening_results, key=lambda x: x['screening_score'], reverse=True):
            print(f"  {sr['category_id']} {sr['category_name']}: Score {sr['screening_score']:.0f}")
    except Exception as e:
        print(f"[ERROR] Screening fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Screening: {str(e)}")
        screening_results = []

    # Top N fuer Deep Dive selektieren
    screening_results.sort(key=lambda x: x['screening_score'], reverse=True)
    top_categories = screening_results[:TOP_N_DEEP_DIVE]
    remaining_categories = screening_results[TOP_N_DEEP_DIVE:]
    top_ids = [tc['category_id'] for tc in top_categories]
    meta['deep_dive_count'] = len(top_categories)

    print(f"\n[SELECTION] Top {len(top_categories)} fuer Deep Dive: {', '.join(top_ids)}")

    # ===== STUFE 2: LLM DEEP DIVE =====
    print(f"\n--- STUFE 2: Deep Dive ({len(top_categories)} Kategorien) ---")
    deep_dive_results = {}
    for tc in top_categories:
        cat_id = tc['category_id']
        cat_obj = next((c for c in active_categories if c['id'] == cat_id), None)
        if not cat_obj:
            continue
        try:
            print(f"\n[DEEP DIVE] {cat_id} — {cat_obj['name']}...")
            dd_result = run_deep_dive(
                category=cat_obj,
                screening_data=tc,
                etf_universe=etf_universe,
                source_quality_weights=config.get('source_quality_weights', {})
            )
            deep_dive_results[cat_id] = dd_result
            meta['sources_total'] += dd_result.get('sources_scanned', 0)
            print(f"  [OK] {cat_id}: Maturity={dd_result.get('maturity', '?')}, Momentum={dd_result.get('momentum', '?')}")
        except Exception as e:
            print(f"  [ERROR] {cat_id} Deep Dive fehlgeschlagen: {e}")
            traceback.print_exc()
            errors.append(f"Deep Dive {cat_id}: {str(e)}")

    # ===== SCORE-BERECHNUNG =====
    print(f"\n--- SCORE-BERECHNUNG ---")
    try:
        trends = calculate_scores(
            deep_dive_results=deep_dive_results,
            screening_results=screening_results,
            previous_history=previous_history,
            config=config
        )
        trends = determine_phases(trends)
        trends = determine_watchlist_status(trends, thresholds)
        print(f"[SCORES] {len(trends)} Trends bewertet")
    except Exception as e:
        print(f"[ERROR] Score-Berechnung fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Scoring: {str(e)}")
        trends = []

    # ===== EXPOSURE CHECK =====
    print(f"\n--- EXPOSURE CHECK ---")
    exposure_result = {}
    try:
        exposure_result = run_exposure_check(
            trends=trends,
            v16_weights=v16_weights,
            exposure_map=exposure_map,
            thresholds=thresholds
        )
        blind_spots = exposure_result.get('blind_spots', [])
        threats_list = exposure_result.get('threats', [])
        print(f"[EXPOSURE] {len(blind_spots)} Blind Spots, {len(threats_list)} Threats")
    except Exception as e:
        print(f"[ERROR] Exposure Check fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Exposure: {str(e)}")

    # ===== VULNERABILITY ANALYSIS (ersetzt Short-Analyse) =====
    print(f"\n--- VULNERABILITY ANALYSIS ---")
    vulnerability_watchlist = []
    try:
        vulnerability_watchlist = run_vulnerability_analysis(
            trends=trends,
            v16_weights=v16_weights,
            exposure_map=exposure_map,
            etf_universe=etf_universe
        )
        print(f"[VULN] {len(vulnerability_watchlist)} vulnerable Assets")
    except Exception as e:
        print(f"[ERROR] Vulnerability Analysis fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Vulnerability: {str(e)}")

    # ===== CONTRARIAN SCAN =====
    print(f"\n--- CONTRARIAN SCAN ---")
    contrarian_alerts = []
    try:
        contrarian_alerts = run_contrarian_scan(
            trends=trends,
            etf_universe=etf_universe,
            thresholds=thresholds
        )
        print(f"[CONTRARIAN] {len(contrarian_alerts)} Contrarian Alerts")
    except Exception as e:
        print(f"[ERROR] Contrarian Scan fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Contrarian: {str(e)}")

    # ===== CAUSAL CHAINS =====
    print(f"\n--- CAUSAL CHAINS ---")
    causal_chains = []
    try:
        causal_chains = run_causal_chains(
            trends=[t for t in trends if t.get('watchlist_status') in ('ACTIVE', 'WATCH')],
            etf_universe=etf_universe
        )
        print(f"[CAUSAL] {len(causal_chains)} Causal Chains generiert")
    except Exception as e:
        print(f"[ERROR] Causal Chains fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Causal Chains: {str(e)}")

    # ===== DEPENDENCY-NETZWERK =====
    print(f"\n--- DEPENDENCY-NETZWERK ---")
    dependencies = dependency_seed
    convergence_zones = []
    try:
        dependencies = update_dependencies(
            trends=trends,
            seed_dependencies=dependency_seed
        )
        convergence_zones = detect_convergence_zones(
            trends=trends,
            dependencies=dependencies,
            thresholds=thresholds
        )
        print(f"[DEPS] {len(dependencies)} Kanten, {len(convergence_zones)} Convergence Zones")
    except Exception as e:
        print(f"[ERROR] Dependencies fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Dependencies: {str(e)}")

    # ===== INTELLIGENCE BRIEFING (Phase A — NEU) =====
    print(f"\n--- INTELLIGENCE BRIEFING ---")
    intelligence_result = None
    try:
        intelligence_result = generate_intelligence_briefing(
            trends=trends,
            exposure_result=exposure_result,
            contrarian_alerts=contrarian_alerts,
            causal_chains=causal_chains,
            convergence_zones=convergence_zones,
            vulnerability_watchlist=vulnerability_watchlist,
            v16_weights=v16_weights,
            v16_regime=v16_regime,
            config=config,
            run_date=run_date,
        )

        # Regime History 4W aktualisieren
        if intelligence_result and intelligence_result.get('regime_context'):
            regime_history = load_regime_history()
            # Neuen Eintrag hinzufuegen
            regime_history.append({
                'date': run_date,
                'regime': v16_regime,
            })
            # Nur letzte 4 behalten
            regime_history = regime_history[-4:]
            intelligence_result['regime_context']['regime_history_4w'] = regime_history

        print(f"[INTEL] Intelligence Briefing generiert")
    except Exception as e:
        print(f"[ERROR] Intelligence Briefing fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Intelligence: {str(e)}")

    # ===== READINESS SCORE =====
    print(f"\n--- READINESS SCORE ---")
    readiness_score = 0
    readiness_label = "BACKWARD-LOOKING"
    try:
        readiness_score = calculate_readiness_score(trends, exposure_result, thresholds)
        if readiness_score >= 80:
            readiness_label = "FUTURE-READY"
        elif readiness_score >= 50:
            readiness_label = "GAPS_PRESENT"
        else:
            readiness_label = "BACKWARD-LOOKING"
        print(f"[READINESS] Score: {readiness_score} — {readiness_label}")
    except Exception as e:
        print(f"[ERROR] Readiness Score fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Readiness: {str(e)}")

    # ===== SHEET WRITE =====
    print(f"\n--- SHEET WRITE ---")
    try:
        write_all_tabs(
            trends=trends,
            screening_results=screening_results,
            exposure_result=exposure_result,
            short_results=[],  # Legacy — leer, Vulnerability Watchlist uebernimmt
            contrarian_alerts=contrarian_alerts,
            causal_chains=causal_chains,
            dependencies=dependencies,
            convergence_zones=convergence_zones,
            etf_universe=etf_universe,
            config=config,
            run_date=run_date
        )
        print("[SHEET] Alle 7 Tabs geschrieben")
    except Exception as e:
        print(f"[ERROR] Sheet Write fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Sheet Write: {str(e)}")

    # ===== DASHBOARD UPDATE =====
    print(f"\n--- DASHBOARD UPDATE ---")
    try:
        duration_s = int(time.time() - start_time)
        meta['run_duration_s'] = duration_s
        meta['errors'] = errors
        if meta['sources_total'] > 0 and meta['deep_dive_count'] > 0:
            meta['avg_source_quality'] = round(meta['sources_total'] / meta['deep_dive_count'] / 5, 1)

        update_dashboard_json(
            dashboard_path=DASHBOARD_PATH,
            trends=trends,
            exposure_result=exposure_result,
            short_results=[],  # Legacy — leer
            contrarian_alerts=contrarian_alerts,
            causal_chains=causal_chains,
            dependencies=dependencies,
            convergence_zones=convergence_zones,
            readiness_score=readiness_score,
            readiness_label=readiness_label,
            meta=meta,
            intelligence_result=intelligence_result,  # Phase A NEU
        )
        print("[DASHBOARD] latest.json aktualisiert")
    except Exception as e:
        print(f"[ERROR] Dashboard Update fehlgeschlagen: {e}")
        traceback.print_exc()
        errors.append(f"Dashboard: {str(e)}")

    # ===== HISTORY SNAPSHOT =====
    try:
        save_history_snapshot(trends, run_date)
    except Exception as e:
        print(f"[ERROR] History Snapshot fehlgeschlagen: {e}")
        errors.append(f"History: {str(e)}")

    # ===== SUMMARY =====
    duration_s = int(time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"  DISRUPTIONS AGENT — ABGESCHLOSSEN")
    print(f"  Dauer: {duration_s}s ({duration_s // 60}m {duration_s % 60}s)")
    print(f"  Kategorien: {len(active_categories)} gescannt, {len(top_categories)} Deep Dive")
    print(f"  Trends: {len(trends)} bewertet")
    print(f"  Blind Spots: {len(exposure_result.get('blind_spots', []))}")
    print(f"  Threats: {len(exposure_result.get('threats', []))}")
    print(f"  Vulnerable Assets: {len(vulnerability_watchlist)}")
    print(f"  Contrarian: {len(contrarian_alerts)}")
    print(f"  Causal Chains: {len(causal_chains)}")
    print(f"  Convergence Zones: {len(convergence_zones)}")
    print(f"  Intelligence Briefing: {'OK' if intelligence_result else 'FEHLT'}")
    print(f"  V16 Regime: {v16_regime}")
    print(f"  Readiness: {readiness_score} ({readiness_label})")
    print(f"  Errors: {len(errors)}")
    if errors:
        for err in errors:
            print(f"    - {err}")
    print(f"{'='*60}\n")

    # Exit mit Error-Code wenn kritische Fehler
    if len(errors) > 3:
        print("[FATAL] Zu viele Fehler — Exit 1")
        sys.exit(1)

    return 0


def calculate_readiness_score(trends, exposure_result, thresholds):
    """
    Disruption Readiness Score (Portfolio-Gesamt).
    Spec §4.6:
      readiness = coverage × 0.35 + vulnerability × 0.35 + urgency × 0.30
    """
    active_trends = [t for t in trends if t.get('watchlist_status') == 'ACTIVE']
    if not active_trends:
        return 50  # Neutral wenn keine ACTIVE Trends

    # Coverage: Wie viele ACTIVE Trends haben >1% Exposure?
    blind_spots = exposure_result.get('blind_spots', [])
    blind_spot_ids = {bs.get('category', bs.get('category_id', '')) for bs in blind_spots}
    covered = [t for t in active_trends if t['id'] not in blind_spot_ids]
    coverage_score = (len(covered) / len(active_trends)) * 100 if active_trends else 50

    # Vulnerability (invertiert): Wie viele Threat-Assets im Portfolio?
    threats = exposure_result.get('threats', [])
    critical_threats = [t for t in threats if t.get('threat_level') in ('HIGH', 'CRITICAL')]
    vulnerability_raw = len(critical_threats) * 25  # 0, 25, 50, 75, 100
    vulnerability_score = max(0, 100 - vulnerability_raw)  # invertiert: weniger Threats = hoeher

    # Urgency: Wie nah sind kritische Trends am Inflection Point?
    inflection_scores = [t.get('inflection_score', 0) for t in active_trends]
    avg_inflection = sum(inflection_scores) / len(inflection_scores) if inflection_scores else 0
    urgency_score = min(100, avg_inflection)

    readiness = (
        coverage_score * 0.35 +
        vulnerability_score * 0.35 +
        urgency_score * 0.30
    )

    return round(readiness)


if __name__ == '__main__':
    sys.exit(main() or 0)
