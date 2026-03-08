#!/usr/bin/env python3
"""
G7 WORLD ORDER MONITOR — Step 0s
Main Orchestrator: 10-Phase Engine

Frequenz: Woechentlich (Sonntag ~08:00 UTC) + Quartalsweise + Interim + Ad-Hoc
Modell: Claude Sonnet (fuer LLM-Phasen, Temperature 0.3)

Phasen:
  1  DATA COLLECTION          -> Raw Data Store
  2  DATA VALIDATION          -> Validated Data + Freshness Tags
  3  SCORING ENGINE            -> 84 Signal-Triplets + Power Scores
  4  OVERLAY COMPUTATION       -> SCSI, DDI, FDP, SIT, EWI, Feedback (STUB Etappe 1)
  5  STATUS DETERMINATION      -> G7_STATUS (deterministisch)
  6  SCENARIO ENGINE           -> G7_THESIS (STUB Etappe 1)
  7  WEB SEARCH                -> External Intelligence (STUB Etappe 1)
  8  NARRATIVE GENERATION      -> G7_NARRATIVE (STUB Etappe 1)
  9  OUTPUT VALIDATION         -> Validated Outputs
  10 TAB WRITING               -> Google Sheets
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- Add parent dir to path for shared imports ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# --- Local module imports ---
from step_0s_g7_monitor.data_collection import phase1_data_collection
from step_0s_g7_monitor.data_enrichment import phase1b_data_enrichment
from step_0s_g7_monitor.scoring_engine import phase3_scoring_engine
from step_0s_g7_monitor.overlays import phase4_overlay_computation
from step_0s_g7_monitor.scenario_engine import phase6_scenario_engine
from step_0s_g7_monitor.narrative_engine import phase8_narrative_generation
from step_0s_g7_monitor.sheet_writer import G7SheetWriter
from step_0s_g7_monitor.display_writer import G7DisplayWriter
from step_0s_g7_monitor.dashboard_update import update_dashboard_json as update_g7_dashboard

# ============================================================
# CONSTANTS
# ============================================================

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
THRESHOLDS_PATH = os.path.join(CONFIG_DIR, "G7_THRESHOLDS.json")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

G7_SHEET_ID = "1TVl-GNYxK7Sppn8Tv8lSlMVgFfCwr8WslWSwABpOybk"

DASHBOARD_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "dashboard", "latest.json"
)


def load_thresholds():
    """Load G7_THRESHOLDS.json — all configurable thresholds."""
    with open(THRESHOLDS_PATH, "r") as f:
        data = json.load(f)
    return data


def now_utc():
    return datetime.now(timezone.utc)


def today_iso():
    return now_utc().strftime("%Y-%m-%d")


# ============================================================
# PHASE 2 — DATA VALIDATION & FRESHNESS TAGGING
# ============================================================

FRESHNESS_THRESHOLDS = {
    "fred":         {"fresh": 7,   "recent": 30,  "stale": 120},
    "imf_weo":      {"fresh": 90,  "recent": 180, "stale": 365},
    "imf_cofer":    {"fresh": 30,  "recent": 90,  "stale": 180},
    "worldbank":    {"fresh": 90,  "recent": 365, "stale": 730},
    "yfinance":     {"fresh": 3,   "recent": 7,   "stale": 30},
    "un_pop":       {"fresh": 365, "recent": 730, "stale": 1460},
    "bis":          {"fresh": 30,  "recent": 120, "stale": 240},
    "swift":        {"fresh": 30,  "recent": 60,  "stale": 120},
    "gpr":          {"fresh": 14,  "recent": 45,  "stale": 90},
    "acled":        {"fresh": 7,   "recent": 14,  "stale": 30},
    "polymarket":   {"fresh": 1,   "recent": 3,   "stale": 7},
    "worldmonitor": {"fresh": 1,   "recent": 3,   "stale": 7},
}

STRUCTURAL_SOURCES = {"un_pop", "worldbank"}

DIMENSION_SOURCE_MAP = {
    "D1_economic":      ["fred", "imf_weo", "worldbank", "yfinance"],
    "D2_demographics":  ["un_pop", "worldbank"],
    "D3_technology":    ["worldbank"],
    "D4_energy":        ["fred", "yfinance", "worldbank"],
    "D5_military":      ["worldbank", "acled", "gpr"],
    "D6_fiscal":        ["fred", "imf_weo", "bis"],
    "D7_currency":      ["imf_cofer", "swift", "yfinance", "bis"],
    "D8_capital_mkt":   ["yfinance", "fred", "bis"],
    "D9_flows":         ["bis", "worldbank"],
    "D10_social":       ["acled", "worldmonitor"],
    "D11_geopolitical": ["gpr", "acled"],
    "D12_feedback":     [],
}


def compute_source_freshness(source, data):
    """Determine freshness tag for a source based on data age."""
    if data is None:
        return "UNAVAILABLE"

    latest_date = None
    if isinstance(data, dict):
        for key in ["last_date", "latest_date", "date", "timestamp"]:
            if key in data and data[key]:
                try:
                    if isinstance(data[key], str):
                        latest_date = datetime.fromisoformat(
                            data[key].replace("Z", "+00:00")
                        )
                    break
                except (ValueError, TypeError):
                    pass

    if latest_date is None:
        return "FRESH"

    now = now_utc()
    if latest_date.tzinfo is None:
        latest_date = latest_date.replace(tzinfo=timezone.utc)
    age_days = (now - latest_date).days

    thresholds = FRESHNESS_THRESHOLDS.get(
        source, {"fresh": 7, "recent": 30, "stale": 180}
    )

    if age_days <= thresholds["fresh"]:
        return "FRESH"
    elif age_days <= thresholds["recent"]:
        return "RECENT"
    elif age_days <= thresholds["stale"]:
        return "STALE"
    else:
        if source in STRUCTURAL_SOURCES:
            return "STRUCTURAL"
        return "STALE"


def aggregate_freshness(freshness_values):
    """Aggregate multiple freshness tags into one (worst-of)."""
    PRIORITY = {
        "UNAVAILABLE": 0, "STALE": 1, "RECENT": 2,
        "STRUCTURAL": 3, "FRESH": 4,
    }
    if not freshness_values:
        return "UNAVAILABLE"
    return min(freshness_values, key=lambda f: PRIORITY.get(f, 0))


def phase2_data_validation(collection_result):
    """Phase 2: Validate data and assign freshness tags."""
    print("[Phase 2] Data Validation & Freshness Tagging...")
    raw_data = collection_result.get("raw_data", {})
    errors = list(collection_result.get("errors", []))

    validated = {}
    freshness_by_source = {}

    for source, data in raw_data.items():
        if data is None:
            freshness_by_source[source] = "UNAVAILABLE"
            continue

        if isinstance(data, dict) and len(data) == 0:
            freshness_by_source[source] = "UNAVAILABLE"
            errors.append({
                "source": source,
                "error": "Empty data returned",
                "severity": "LOW",
            })
            continue

        validated[source] = data
        freshness_by_source[source] = compute_source_freshness(source, data)

    # Freshness per dimension
    freshness_by_dimension = {}
    for dim, sources in DIMENSION_SOURCE_MAP.items():
        dim_freshness = [
            freshness_by_source.get(s, "UNAVAILABLE") for s in sources
        ]
        if dim_freshness:
            freshness_by_dimension[dim] = aggregate_freshness(dim_freshness)
        else:
            freshness_by_dimension[dim] = "FRESH"

    available_count = sum(
        1 for v in freshness_by_source.values() if v != "UNAVAILABLE"
    )
    total_count = len(freshness_by_source)

    print(f"  Sources: {available_count}/{total_count} available")
    for src, fresh in freshness_by_source.items():
        if fresh == "UNAVAILABLE":
            print(f"  WARNING: {src} UNAVAILABLE")

    return {
        "validated_data": validated,
        "freshness_by_source": freshness_by_source,
        "freshness_by_dimension": freshness_by_dimension,
        "validation_errors": errors,
        "sources_available": available_count,
        "sources_total": total_count,
    }


# ============================================================
# PHASE 5 — STATUS DETERMINATION (deterministisch)
# ============================================================

def phase5_status_determination(power_scores, gap_data, overlays,
                                previous_status, thresholds_config):
    """
    Phase 5: Deterministic status determination.
    Uses G7_THRESHOLDS.json for all thresholds.
    No LLM — pure rule-based.

    Status hierarchy: STRUCTURAL_BREAK > ELEVATED_RISK > SHIFTING > STABLE
    """
    print("[Phase 5] Status Determination (deterministic)...")

    t = thresholds_config.get("status_determination", {})

    gt = t.get("gap", {})
    GAP_CRITICAL = gt.get("GAP_CRITICAL", 5)
    GAP_ELEVATED = gt.get("GAP_ELEVATED", 10)
    GAP_SHIFTING = gt.get("GAP_SHIFTING", 15)

    gp = t.get("gpr", {})
    GPR_CRITICAL = gp.get("GPR_CRITICAL", 350)
    GPR_ELEVATED = gp.get("GPR_ELEVATED", 250)
    GPR_SHIFTING = gp.get("GPR_SHIFTING", 150)

    sc = t.get("scsi", {})
    SCSI_CRITICAL = sc.get("SCSI_CRITICAL", 80)
    SCSI_ELEVATED = sc.get("SCSI_ELEVATED", 60)
    SCSI_SHIFTING = sc.get("SCSI_SHIFTING", 40)

    dd = t.get("ddi", {})
    DDI_SHIFTING = dd.get("DDI_SHIFTING", 65)

    fp = t.get("fdp", {})
    FDP_ELEVATED = fp.get("FDP_ELEVATED", 0.85)

    fbl = t.get("feedback_loops", {})
    FB_ELEV_COUNT = fbl.get("FEEDBACK_ELEVATED_COUNT", 2)
    FB_ELEV_SEV = fbl.get("FEEDBACK_ELEVATED_SEVERITY", 3)

    ew = t.get("ewi", {})
    EWI_SHIFTING = ew.get("EWI_SHIFTING_COUNT", 3)
    EWI_ELEVATED = ew.get("EWI_ELEVATED_COUNT", 5)

    sst = t.get("scenario_shift", {})
    SCEN_SHIFT_CRIT = sst.get("SCENARIO_SHIFT_CRITICAL", 0.20)

    # --- Current values ---
    gap = gap_data.get("gap", 50)
    gpr = overlays.get("gpr_index_current", 100)
    scsi_comp = overlays.get("scsi", {}).get("composite", 0)
    ddi_comp = overlays.get("ddi", {}).get("composite", 0)
    fdp_usa = overlays.get("fdp", {}).get("USA", {}).get(
        "composite_proximity", 0
    )

    feedback_loops = overlays.get("feedback_loops", [])
    hi_sev_loops = [
        l for l in feedback_loops
        if l.get("status") == "ACTIVE"
        and l.get("severity", 0) > FB_ELEV_SEV
    ]

    ewi_active = overlays.get("ewi", {}).get("active_signals", 0)
    max_scen_shift = overlays.get("max_scenario_shift", 0)

    # === STRUCTURAL_BREAK CHECK ===
    sb_reasons = []
    if gap < GAP_CRITICAL:
        sb_reasons.append(f"USA-China Gap {gap:.1f} < {GAP_CRITICAL} (CRITICAL)")
    if gpr > GPR_CRITICAL:
        sb_reasons.append(f"GPR Index {gpr:.0f} > {GPR_CRITICAL} (CRITICAL)")
    if scsi_comp > SCSI_CRITICAL:
        sb_reasons.append(f"SCSI {scsi_comp:.1f} > {SCSI_CRITICAL} (CRITICAL)")
    if max_scen_shift > SCEN_SHIFT_CRIT:
        sb_reasons.append(f"Scenario shift {max_scen_shift:.0%} > {SCEN_SHIFT_CRIT:.0%} (CRITICAL)")
    if sb_reasons:
        print(f"  Status: STRUCTURAL_BREAK ({len(sb_reasons)} triggers)")
        return _build_status("STRUCTURAL_BREAK", sb_reasons, gap_data, power_scores, overlays, previous_status)

    # === ELEVATED_RISK CHECK ===
    er_reasons = []
    if gap < GAP_ELEVATED:
        er_reasons.append(f"USA-China Gap {gap:.1f} < {GAP_ELEVATED} (ELEVATED)")
    if gpr > GPR_ELEVATED:
        er_reasons.append(f"GPR Index {gpr:.0f} > {GPR_ELEVATED} (ELEVATED)")
    if scsi_comp > SCSI_ELEVATED:
        er_reasons.append(f"SCSI {scsi_comp:.1f} > {SCSI_ELEVATED} (ELEVATED)")
    if fdp_usa > FDP_ELEVATED:
        er_reasons.append(f"FDP USA {fdp_usa:.2f} > {FDP_ELEVATED} (ELEVATED)")
    if len(hi_sev_loops) >= FB_ELEV_COUNT:
        names = [l["name"] for l in hi_sev_loops[:3]]
        er_reasons.append(f"{len(hi_sev_loops)} active loops sev>{FB_ELEV_SEV}: {', '.join(names)}")
    if ewi_active >= EWI_ELEVATED:
        er_reasons.append(f"EWI {ewi_active} active >= {EWI_ELEVATED} (ELEVATED)")
    if er_reasons:
        print(f"  Status: ELEVATED_RISK ({len(er_reasons)} triggers)")
        return _build_status("ELEVATED_RISK", er_reasons, gap_data, power_scores, overlays, previous_status)

    # === SHIFTING CHECK ===
    sh_reasons = []
    if gap < GAP_SHIFTING:
        sh_reasons.append(f"USA-China Gap {gap:.1f} < {GAP_SHIFTING} (SHIFTING)")
    if gpr > GPR_SHIFTING:
        sh_reasons.append(f"GPR Index {gpr:.0f} > {GPR_SHIFTING} (SHIFTING)")
    if scsi_comp > SCSI_SHIFTING:
        sh_reasons.append(f"SCSI {scsi_comp:.1f} > {SCSI_SHIFTING} (SHIFTING)")
    if ddi_comp > DDI_SHIFTING:
        sh_reasons.append(f"DDI {ddi_comp:.1f} > {DDI_SHIFTING} (SHIFTING)")
    if ewi_active >= EWI_SHIFTING:
        sh_reasons.append(f"EWI {ewi_active} active >= {EWI_SHIFTING} (SHIFTING)")
    if sh_reasons:
        print(f"  Status: SHIFTING ({len(sh_reasons)} triggers)")
        return _build_status("SHIFTING", sh_reasons, gap_data, power_scores, overlays, previous_status)

    # === STABLE (default) ===
    print("  Status: STABLE")
    return _build_status("STABLE", [], gap_data, power_scores, overlays, previous_status)


def _build_status(status, active_shifts, gap_data, power_scores, overlays, previous_status):
    """Build the G7_STATUS output contract (Spec Teil 1 Abschnitt 3)."""
    ps_summary = {}
    ps_momenta = {}
    for region in REGIONS:
        ps = power_scores.get(region, {})
        ps_summary[region] = round(ps.get("score", 0), 1)
        ps_momenta[region] = round(ps.get("momentum", 0), 2)

    portfolio_relevance = None
    if status != "STABLE" and active_shifts:
        portfolio_relevance = f"G7 Status {status}: {active_shifts[0]}"

    scsi = overlays.get("scsi", {})
    ddi = overlays.get("ddi", {})
    fdp = overlays.get("fdp", {})
    ewi = overlays.get("ewi", {})

    feedback_loops = overlays.get("feedback_loops", [])
    dominant_loops = [
        {"name": l.get("name", ""), "region": l.get("region", ""),
         "severity": l.get("severity", 0), "trend": l.get("trend", "STABLE")}
        for l in feedback_loops
        if l.get("status") in ("ACTIVE", "LATENT") and l.get("severity", 0) > 1
    ][:5]

    attention_flag = _compute_attention_flag(status, ewi)
    prev_st = previous_status.get("g7_status", "STABLE") if previous_status else "STABLE"

    return {
        "date": today_iso(),
        "g7_status": status,
        "active_shifts": active_shifts,
        "portfolio_relevance": portfolio_relevance,
        "available": True,
        "last_update": now_utc().isoformat(),
        "power_scores_summary": ps_summary,
        "power_score_momenta": ps_momenta,
        "usa_china_gap": gap_data.get("gap", 0),
        "usa_china_gap_trend": gap_data.get("trend", "STABLE"),
        "gpr_index_current": overlays.get("gpr_index_current", 0),
        "gpr_index_trend": overlays.get("gpr_index_trend", "STABLE"),
        "gpr_index_zscore": round(overlays.get("gpr_index_zscore", 0), 2),
        "supply_chain_stress_index": {
            "composite": scsi.get("composite", 0),
            "trend": scsi.get("trend", "STABLE"),
            "active_chokepoint_alerts": scsi.get("active_chokepoint_alerts", 0),
            "chokepoints": scsi.get("chokepoints", {}),
        },
        "dedollarization_index": {
            "composite": ddi.get("composite", 0),
            "trend": ddi.get("trend", "STABLE"),
            "acceleration": ddi.get("acceleration", 0),
        },
        "fiscal_dominance_proximity": {
            r: fdp.get(r, {}).get("composite_proximity", 0) for r in REGIONS
        },
        "early_warning_index": {
            "active_signals": ewi.get("active_signals", 0),
            "total_signals": ewi.get("total_signals", 10),
            "severity": ewi.get("severity", "NONE"),
        },
        "dominant_feedback_loops": dominant_loops,
        "attention_flag": attention_flag,
        "status_changed": status != prev_st,
        "previous_status": prev_st,
    }


def _compute_attention_flag(status, ewi):
    """Compute G7 Attention Flag (Spec Teil 5 Abschnitt 7)."""
    ewi_active = ewi.get("active_signals", 0)
    if status == "STRUCTURAL_BREAK":
        return "URGENT"
    if status == "ELEVATED_RISK" or ewi_active >= 4:
        return "ATTENTION"
    if status == "SHIFTING" or ewi_active >= 2:
        return "NOTE"
    return "NONE"


# ============================================================
# PHASE 9 — OUTPUT VALIDATION
# ============================================================

def phase9_output_validation(g7_status, scenario_result, narrative_result):
    """Phase 9: Validate all outputs before writing."""
    print("[Phase 9] Output Validation...")
    errors = []
    warnings = []

    for f in ["date", "g7_status", "available", "last_update"]:
        if f not in g7_status:
            errors.append(f"G7_STATUS missing: {f}")

    valid_statuses = ["STABLE", "SHIFTING", "ELEVATED_RISK", "STRUCTURAL_BREAK"]
    if g7_status.get("g7_status") not in valid_statuses:
        errors.append(f"G7_STATUS invalid: {g7_status.get('g7_status')}")

    if scenario_result and scenario_result.get("thesis_updated"):
        thesis = scenario_result.get("thesis", {})
        if not thesis.get("dominant_thesis"):
            warnings.append("G7_THESIS missing dominant_thesis")

    is_valid = len(errors) == 0
    if errors:
        print(f"  ERRORS: {errors}")
    if warnings:
        print(f"  WARNINGS: {warnings}")
    if is_valid:
        print("  All outputs valid")
    return {"valid": is_valid, "errors": errors, "warnings": warnings}


# ============================================================
# MAIN ENGINE
# ============================================================

def run_g7_monitor(run_type="WEEKLY", dry_run=False):
    """Main entry point. run_type: WEEKLY | QUARTERLY | INTERIM | AD_HOC"""
    start_time = time.time()
    run_id = f"g7_{run_type.lower()}_{now_utc().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 70)
    print(f"G7 WORLD ORDER MONITOR — {run_type} RUN")
    print(f"Run ID: {run_id} | Dry-run: {dry_run}")
    print(f"Started: {now_utc().isoformat()}")
    print("=" * 70)

    # Load thresholds
    try:
        thresholds = load_thresholds()
        print(f"[Config] G7_THRESHOLDS.json v{thresholds.get('_meta', {}).get('version', '?')}")
    except Exception as e:
        print(f"[Config] ERROR: {e} — using defaults")
        thresholds = {}

    # Sheet writer
    writer = None
    if not dry_run:
        writer = G7SheetWriter(G7_SHEET_ID)
        if not writer.connect():
            print("[CRITICAL] Cannot connect to Google Sheets")
            writer = None

    # Load previous state
    previous_status = None
    previous_scores = None
    previous_thesis = None
    thesis_history = []
    if writer:
        print("[State] Loading previous state...")
        previous_status = writer.read_previous_g7_status()
        previous_scores = writer.read_previous_g7_scores()
        previous_thesis = writer.read_previous_g7_thesis()
        thesis_history = writer.read_g7_thesis_history()

    run_log = {
        "run_id": run_id, "run_type": run_type,
        "started": now_utc().isoformat(), "phases": {}, "errors": [],
    }

    # ---- PHASE 1 ----
    ps = time.time()
    try:
        collection = phase1_data_collection()
        run_log["phases"]["P1"] = {"status": "OK", "sources_ok": collection.get("sources_available", 0), "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 1] CRITICAL: {e}"); traceback.print_exc()
        collection = {"raw_data": {}, "errors": [str(e)]}
        run_log["phases"]["P1"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P1: {e}")

    # ---- PHASE 1b: DATA ENRICHMENT (Brave Search + LLM) ----
    ps = time.time()
    enrichment_data = None
    try:
        enrichment_data = phase1b_data_enrichment(run_type=run_type)
        enrich_source = enrichment_data.get("enrichment_source", "UNKNOWN") if enrichment_data else "NONE"
        run_log["phases"]["P1b"] = {"status": "OK", "source": enrich_source, "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 1b] ERROR: {e}"); traceback.print_exc()
        enrichment_data = None
        run_log["phases"]["P1b"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P1b: {e}")

    # ---- PHASE 2 ----
    ps = time.time()
    try:
        validation = phase2_data_validation(collection)
        # Inject enrichment extracted_data into validated_data
        if enrichment_data and isinstance(enrichment_data, dict):
            validation["validated_data"]["enrichment"] = enrichment_data.get("extracted_data", {})
        run_log["phases"]["P2"] = {"status": "OK", "available": validation.get("sources_available", 0), "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 2] ERROR: {e}"); traceback.print_exc()
        validation = {"validated_data": collection.get("raw_data", {}), "freshness_by_source": {}, "freshness_by_dimension": {}, "validation_errors": [str(e)]}
        run_log["phases"]["P2"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P2: {e}")

    # ---- PHASE 3 ----
    ps = time.time()
    try:
        scoring = phase3_scoring_engine(
            validated_data=validation["validated_data"],
            freshness=validation.get("freshness_by_dimension", {}),
            previous_scores=previous_scores or {},
            enrichment_data=enrichment_data,
        )
        power_scores = scoring.get("power_scores", {})
        gap_data = scoring.get("gap_data", {"gap": 50, "trend": "STABLE", "gap_momentum": 0})
        run_log["phases"]["P3"] = {"status": "OK", "quant_dims": scoring.get("quant_dimensions_complete", 0), "enriched_dims": scoring.get("llm_dimensions_enriched", 0), "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 3] ERROR: {e}"); traceback.print_exc()
        scoring = {}
        power_scores = {r: {"score": 50, "momentum": 0, "acceleration": 0} for r in REGIONS}
        gap_data = {"gap": 50, "trend": "STABLE", "gap_momentum": 0}
        run_log["phases"]["P3"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P3: {e}")

    # ---- PHASE 4 ----
    ps = time.time()
    try:
        overlays = phase4_overlay_computation(scores=scoring, validated_data=validation.get("validated_data", {}), previous_overlays={}, run_type=run_type)
        run_log["phases"]["P4"] = {"status": "OK", "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 4] ERROR: {e}"); traceback.print_exc()
        overlays = _default_overlays()
        run_log["phases"]["P4"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P4: {e}")

    # ---- PHASE 5 ----
    ps = time.time()
    try:
        g7_status = phase5_status_determination(power_scores, gap_data, overlays, previous_status, thresholds)
        run_log["phases"]["P5"] = {"status": "OK", "g7_status": g7_status["g7_status"], "attention_flag": g7_status.get("attention_flag", "NONE"), "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 5] CRITICAL: {e}"); traceback.print_exc()
        g7_status = {"date": today_iso(), "g7_status": "STABLE", "active_shifts": [], "portfolio_relevance": None, "available": False, "last_update": now_utc().isoformat(), "error": str(e)}
        run_log["phases"]["P5"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P5: {e}")

    # ---- PHASE 6 ----
    ps = time.time()
    try:
        scenario_result = phase6_scenario_engine(scores=scoring, overlays=overlays, gap_data=gap_data, validated_data=validation.get("validated_data", {}), previous_thesis=previous_thesis, scenario_history=thesis_history, run_type=run_type)
        run_log["phases"]["P6"] = {"status": "OK", "thesis_updated": scenario_result.get("thesis_updated", False), "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 6] ERROR: {e}"); traceback.print_exc()
        scenario_result = {"thesis_updated": False, "current_thesis": previous_thesis}
        run_log["phases"]["P6"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P6: {e}")

    # ---- PHASE 7 (STUB) ----
    run_log["phases"]["P7"] = {"status": "STUB", "duration_s": 0}

    # ---- PHASE 8 ----
    ps = time.time()
    try:
        narrative_result = phase8_narrative_generation(power_scores=power_scores, gap_data=gap_data, overlays=overlays, g7_status=g7_status, scenario_result=scenario_result, web_search_results=None, previous_narrative=None)
        llm_model = narrative_result.get("llm_model", "unknown") if narrative_result else "error"
        run_log["phases"]["P8"] = {"status": "OK", "llm_model": llm_model, "duration_s": round(time.time() - ps, 1)}
    except Exception as e:
        print(f"[Phase 8] ERROR: {e}"); traceback.print_exc()
        narrative_result = _default_narrative(g7_status)
        run_log["phases"]["P8"] = {"status": "ERROR", "error": str(e)}
        run_log["errors"].append(f"P8: {e}")

    # ---- PHASE 9 ----
    val = phase9_output_validation(g7_status, scenario_result, narrative_result)
    run_log["phases"]["P9"] = {"status": "OK" if val["valid"] else "WARNINGS", "errors": val["errors"], "warnings": val["warnings"]}

    # ---- PHASE 10 ----
    ps = time.time()

    # Finalize run_log BEFORE writing so Sheet gets complete data
    total_time = time.time() - start_time
    run_log["completed"] = now_utc().isoformat()
    run_log["total_duration_s"] = round(total_time, 1)
    run_log["final_status"] = g7_status["g7_status"]
    run_log["errors_count"] = len(run_log["errors"])

    if writer and not dry_run:
        try:
            print("[Phase 10] Writing to Google Sheets...")
            writer.write_g7_status(g7_status)
            print("  G7_STATUS written")
            writer.write_g7_power_score_history(power_scores, gap_data)
            print("  G7_POWER_SCORE_HISTORY appended")
            if scenario_result.get("thesis_updated"):
                writer.write_g7_thesis(scenario_result.get("thesis", {}))
                print("  G7_THESIS written")
            if narrative_result:
                writer.write_g7_narrative(narrative_result)
                print("  G7_NARRATIVE written")
            writer.write_g7_data_cache(validation.get("validated_data", {}))
            print("  G7_DATA_CACHE written")
            writer.write_g7_run_log(run_log)
            print("  G7_RUN_LOG appended")

            # --- Display Writer: 11 Layout Tabs ---
            try:
                display = G7DisplayWriter(G7_SHEET_ID)
                if display.connect():
                    display.write_all(
                        scoring_result=scoring,
                        overlays=overlays,
                        g7_status=g7_status,
                        scenario_result=scenario_result,
                        validated_data=validation.get("validated_data", {}),
                        freshness_by_source=validation.get("freshness_by_source", {}),
                        collection_errors=collection.get("errors", []),
                    )
                    print("  Display Writer: 11 layout tabs updated")
                else:
                    print("  Display Writer: connection failed — skipping")
            except Exception as e:
                print(f"  Display Writer ERROR: {e}"); traceback.print_exc()

            run_log["phases"]["P10"] = {"status": "OK", "duration_s": round(time.time() - ps, 1)}
        except Exception as e:
            print(f"[Phase 10] ERROR: {e}"); traceback.print_exc()
            run_log["phases"]["P10"] = {"status": "ERROR", "error": str(e)}
            run_log["errors"].append(f"P10: {e}")
    else:
        run_log["phases"]["P10"] = {"status": "SKIPPED" if dry_run else "NO_CONNECTION"}
        if dry_run:
            print("[Phase 10] Dry-run — skipping writes")
            print("\nG7_STATUS output:")
            print(json.dumps(g7_status, indent=2, default=str))

    # ---- PHASE 10b: Dashboard JSON (g7 block for Vercel) ----
    try:
        print("[Phase 10b] Updating dashboard.json g7 block...")
        sit_result = overlays.get("sit", {})
        update_g7_dashboard(
            dashboard_json_path=DASHBOARD_JSON_PATH,
            power_scores=power_scores,
            gap_data=gap_data,
            overlays=overlays,
            g7_status=g7_status,
            scenario_result=scenario_result,
            narrative=narrative_result or _default_narrative(g7_status),
            scoring_result=scoring,
            sit_result=sit_result,
            run_metadata={
                "run_id": run_id,
                "run_type": run_type,
                "duration_s": round(time.time() - start_time, 1),
                "errors_count": len(run_log["errors"]),
            },
            sheet_writer=writer,
        )
        print("  Dashboard g7 block written")
    except Exception as e:
        print(f"[Phase 10b] Dashboard update failed (non-fatal): {e}")
        run_log["errors"].append(f"P10b: {e}")

    # ---- PRINT SUMMARY ----
    print("\n" + "=" * 70)
    print(f"G7 MONITOR COMPLETE — {g7_status['g7_status']}")
    print(f"Attention: {g7_status.get('attention_flag', 'NONE')} | {run_log['total_duration_s']}s | Errors: {len(run_log['errors'])}")
    print("=" * 70)

    return {"g7_status": g7_status, "scenario_result": scenario_result, "narrative_result": narrative_result, "run_log": run_log}


def _default_overlays():
    """Neutral overlay defaults when Phase 4 fails."""
    return {
        "scsi": {"composite": 0, "trend": "STABLE", "active_chokepoint_alerts": 0, "chokepoints": {}},
        "ddi": {"composite": 0, "trend": "STABLE", "acceleration": 0},
        "fdp": {r: {"composite_proximity": 0} for r in REGIONS},
        "ewi": {"active_signals": 0, "total_signals": 10, "severity": "NONE"},
        "feedback_loops": [],
        "gpr_index_current": 100, "gpr_index_trend": "STABLE", "gpr_index_zscore": 0,
        "max_scenario_shift": 0,
    }


def _default_narrative(g7_status):
    """Minimal narrative when Phase 8 fails."""
    return {
        "headline": f"G7 Status: {g7_status.get('g7_status', 'UNKNOWN')}",
        "weekly_shift_narrative": "Narrative generation pending (Etappe 3).",
        "top_signals": [],
        "scenario_implications": "Scenario engine pending (Etappe 3).",
        "portfolio_context": "Portfolio-first framing pending (Etappe 3).",
        "counter_narrative": {}, "unasked_question": "", "cascade_watch": "None",
        "regime_congruence": {"congruent": True, "tension": None},
        "regime_congruence_note": "", "historical_analog": {},
        "liquidity_distribution_map": {}, "correlation_regime": {},
        "attention_flag": g7_status.get("attention_flag", "NONE"),
        "word_count": 0, "llm_model": "stub", "generation_time_seconds": 0,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="G7 World Order Monitor — Step 0s")
    parser.add_argument("--run-type", choices=["WEEKLY", "QUARTERLY", "INTERIM", "AD_HOC"], default="WEEKLY")
    parser.add_argument("--dry-run", action="store_true", help="No Sheet writes")
    args = parser.parse_args()
    return run_g7_monitor(run_type=args.run_type, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
