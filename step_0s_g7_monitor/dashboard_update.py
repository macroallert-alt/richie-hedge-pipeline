"""
step_0s_g7_monitor/dashboard_update.py
Writes the g7 block into data/dashboard/latest.json

Called at the end of Phase 10 in main.py.
Pattern: same as step4_cio/dashboard_update.py

The g7 block provides ALL data the Vercel G7Detail.jsx needs:
  - Power Scores with 12-week history (for radar chart + sparklines)
  - Scenario probabilities + tilts matrix + action map
  - Overlays with explanations
  - SIT summary with portfolio vetos
  - Narrative texts (headline, weekly shift, implications, counter-narrative)
  - Dashboard explanations per section
  - 12-dimension deep dive scores
"""

import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger("g7_dashboard")

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]
SCENARIOS = ["managed_decline", "conflict_escalation", "us_renewal", "multipolar_chaos"]

SCENARIO_LABELS = {
    "managed_decline": "Managed Decline",
    "conflict_escalation": "Conflict Escalation",
    "us_renewal": "US Renewal",
    "multipolar_chaos": "Multipolar Chaos",
}

SCENARIO_COLORS = {
    "managed_decline": "#F59E0B",       # amber
    "conflict_escalation": "#EF4444",   # red
    "us_renewal": "#22C55E",            # green
    "multipolar_chaos": "#6B7280",      # gray
}

CYCLE_PHASES = {
    "USA": "Late Decline?", "CHINA": "Peak / Early Decline?",
    "EU": "Managed Decline", "INDIA": "Early Rise",
    "JP_KR_TW": "Tech Power, Geo Risk", "GULF": "Rising Wealth",
    "REST_EM": "Frontier Growth",
}

DIMENSION_NAMES = {
    "D1_economic": "D1 Economic Output",
    "D2_demographics": "D2 Demographics",
    "D3_technology": "D3 Technology",
    "D4_energy": "D4 Energy Security",
    "D5_military": "D5 Military Power",
    "D6_fiscal": "D6 Fiscal Health",
    "D7_currency": "D7 Currency Strength",
    "D8_capital_mkt": "D8 Capital Markets",
    "D9_flows": "D9 Capital Flows",
    "D10_social": "D10 Social Cohesion",
    "D11_geopolitical": "D11 Geopolitical Influence",
    "D12_feedback": "D12 Feedback Loops",
}

# Top 10 assets for tilts matrix display
TILTS_DISPLAY_ASSETS = [
    "GLD", "SPY", "QQQ", "FXI", "KWEB",
    "TLT", "EEM", "INDA", "BTC", "DBC",
]


def update_dashboard_json(
    dashboard_json_path: str,
    power_scores: dict,
    gap_data: dict,
    overlays: dict,
    g7_status: dict,
    scenario_result: dict,
    narrative: dict,
    scoring_result: dict,
    sit_result: dict,
    run_metadata: dict,
    sheet_writer=None,
):
    """
    Read latest.json, update the g7 block, write back.

    Args:
        dashboard_json_path: absolute path to data/dashboard/latest.json
        power_scores: {region: {score, momentum, acceleration}} from scoring_engine
        gap_data: {gap, trend} from scoring_engine
        overlays: full overlay dict from phase4
        g7_status: status dict from phase5
        scenario_result: full result from phase6
        narrative: full result from phase8
        scoring_result: full result from phase3 (contains dimension scores)
        sit_result: SIT result from overlays (may be inside overlays dict)
        run_metadata: {run_id, run_type, duration_s, errors_count}
        sheet_writer: G7SheetWriter instance for reading history (optional)
    """
    try:
        # Read existing latest.json
        dashboard = {}
        if os.path.exists(dashboard_json_path):
            with open(dashboard_json_path, "r") as f:
                dashboard = json.load(f)

        # Build g7 block
        g7 = _build_g7_block(
            power_scores, gap_data, overlays, g7_status,
            scenario_result, narrative, scoring_result,
            sit_result, run_metadata, sheet_writer, dashboard,
        )

        # Write back
        dashboard["g7"] = g7
        os.makedirs(os.path.dirname(dashboard_json_path), exist_ok=True)
        with open(dashboard_json_path, "w") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Dashboard: g7 block written to {dashboard_json_path}")

    except Exception as e:
        logger.error(f"Dashboard g7 update failed (non-fatal): {e}")


def _build_g7_block(
    power_scores, gap_data, overlays, g7_status,
    scenario_result, narrative, scoring_result,
    sit_result, run_metadata, sheet_writer, existing_dashboard,
):
    """Assemble the complete g7 block for the Vercel frontend."""

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── 1. Power Scores ──────────────────────────────────────
    ps_block = {}
    for region in REGIONS:
        ps = power_scores.get(region, {})
        ps_block[region] = {
            "score": _r1(ps.get("score", 0)),
            "momentum": _r2(ps.get("momentum", 0)),
            "acceleration": _r2(ps.get("acceleration", 0)),
            "phase": CYCLE_PHASES.get(region, "—"),
        }

    # ── 2. Power Scores History (12 weeks for sparklines + radar ghost) ──
    ps_history = _read_power_score_history(sheet_writer, existing_dashboard)

    # ── 3. Gap ────────────────────────────────────────────────
    gap_block = {
        "value": _r1(gap_data.get("gap", 0)),
        "trend": gap_data.get("trend", "STABLE"),
    }

    # ── 4. Scenarios ──────────────────────────────────────────
    thesis = _extract_thesis(scenario_result)
    probs = thesis.get("scenario_probabilities", {})
    scenarios_block = {
        "probabilities": {s: _r3(probs.get(s, 0.25)) for s in SCENARIOS},
        "dominant": thesis.get("dominant_thesis", "managed_decline"),
        "confidence": thesis.get("confidence", "LOW"),
        "interim_flag": thesis.get("interim_flag", False),
        "probability_source": thesis.get("probability_source", ""),
    }

    # ── 5. Tilts Matrix (top 10 assets × 4 scenarios + composite) ──
    from step_0s_g7_monitor.scenario_engine import ASSET_EXPOSURE_VECTORS
    tilts_matrix = []
    for asset in TILTS_DISPLAY_ASSETS:
        exp = ASSET_EXPOSURE_VECTORS.get(asset, {})
        weighted = sum(
            probs.get(s, 0.25) * exp.get(s, 0) for s in SCENARIOS
        )
        entry = {
            "asset": asset,
            "managed_decline": exp.get("managed_decline", 0),
            "conflict_escalation": exp.get("conflict_escalation", 0),
            "us_renewal": exp.get("us_renewal", 0),
            "multipolar_chaos": exp.get("multipolar_chaos", 0),
            "composite": _r3(weighted),
        }
        tilts_matrix.append(entry)
    # Sort by absolute composite (strongest signal first)
    tilts_matrix.sort(key=lambda x: abs(x["composite"]), reverse=True)

    # ── 6. Action Map (per scenario: overweight/underweight/veto) ──
    action_map = {}
    for s in SCENARIOS:
        ow = []
        uw = []
        vetos = []
        for asset, exp in ASSET_EXPOSURE_VECTORS.items():
            val = exp.get(s, 0)
            if val >= 0.5:
                ow.append(asset)
            elif val <= -0.5:
                uw.append(asset)
            if val <= -0.8:
                vetos.append(asset)
        action_map[s] = {
            "label": SCENARIO_LABELS.get(s, s),
            "overweight": sorted(ow),
            "underweight": sorted(uw),
            "vetos": sorted(vetos),
        }

    # ── 7. PermOpt ────────────────────────────────────────────
    permopt_block = {}
    permopt = thesis.get("perm_opt_allocation", {})
    if permopt:
        permopt_block = {
            "total_pct": permopt.get("total_pct", 0),
            "assets": permopt.get("assets", []),
            "ddi_level": permopt.get("ddi_level", 0),
        }

    # ── 8. Overlays ───────────────────────────────────────────
    scsi = overlays.get("scsi", {})
    ddi = overlays.get("ddi", {})
    fdp = overlays.get("fdp", {})
    ewi = overlays.get("ewi", {})

    overlays_block = {
        "scsi": {
            "value": _r1(scsi.get("composite", 0)),
            "trend": scsi.get("trend", "STABLE"),
        },
        "ddi": {
            "value": _r1(ddi.get("composite", 0)),
            "trend": ddi.get("trend", "STABLE"),
        },
        "fdp_usa": {
            "value": _r2(fdp.get("USA", 0) if isinstance(fdp, dict) else 0),
        },
        "ewi": {
            "active_signals": ewi.get("active_signals", 0),
            "severity": ewi.get("severity", "NONE"),
        },
    }

    # Overlay history (append current values, keep max 12 entries)
    overlay_history = _update_overlay_history(
        existing_dashboard, overlays_block, now_iso
    )

    # ── 9. SIT ────────────────────────────────────────────────
    sit_block = _build_sit_block(sit_result, overlays)

    # ── 10. Challenge ─────────────────────────────────────────
    counter_narr = narrative.get("counter_narrative", {})
    challenge_block = {
        "counter_narrative": counter_narr if isinstance(counter_narr, dict) else {},
        "unasked_question": narrative.get("unasked_question", ""),
        "stress_test_result": _extract_stress_test(scenario_result),
    }

    # ── 11. Dimensions (12 × 7 for deep dive) ────────────────
    dim_scores = scoring_result.get("scores", {})
    dimensions_block = {}
    for dim_key, dim_label in DIMENSION_NAMES.items():
        dim_data = dim_scores.get(dim_key, {})
        dimensions_block[dim_key] = {
            "label": dim_label,
            "scores": {r: _r1(dim_data.get(r, 0)) for r in REGIONS},
        }

    # ── 12. Narrative texts ───────────────────────────────────
    narrative_block = {
        "headline": narrative.get("headline", ""),
        "weekly_shift": narrative.get("weekly_shift_narrative", ""),
        "scenario_implications": narrative.get("scenario_implications", ""),
        "portfolio_context": narrative.get("portfolio_context", ""),
    }

    # ── 13. Explanations ──────────────────────────────────────
    dash_expl = narrative.get("dashboard_explanations", {})
    explanations_block = {
        "status": dash_expl.get("status_explanation", ""),
        "power_gap": dash_expl.get("power_gap_explanation", ""),
        "scsi": dash_expl.get("scsi_explanation", ""),
        "ddi": dash_expl.get("ddi_explanation", ""),
        "fdp": dash_expl.get("fdp_explanation", ""),
        "ewi": dash_expl.get("ewi_explanation", ""),
    }

    # ── 14. Meta ──────────────────────────────────────────────
    meta_block = {
        "last_run": now_iso,
        "run_type": run_metadata.get("run_type", "WEEKLY"),
        "run_id": run_metadata.get("run_id", ""),
        "duration_s": run_metadata.get("duration_s", 0),
        "errors_count": run_metadata.get("errors_count", 0),
    }

    # ── ASSEMBLE ──────────────────────────────────────────────
    return {
        "status": g7_status.get("g7_status", "STABLE"),
        "attention_flag": g7_status.get("attention_flag", "NONE"),
        "available": True,

        "power_scores": ps_block,
        "power_scores_history": ps_history,
        "gap": gap_block,

        "scenarios": scenarios_block,
        "tilts_matrix": tilts_matrix,
        "action_map": action_map,
        "permopt": permopt_block,

        "overlays": overlays_block,
        "overlay_history": overlay_history,

        "sit": sit_block,

        "challenge": challenge_block,

        "dimensions": dimensions_block,

        "narrative": narrative_block,
        "explanations": explanations_block,

        "meta": meta_block,
    }


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _r1(val):
    """Round to 1 decimal."""
    try:
        return round(float(val), 1)
    except (TypeError, ValueError):
        return 0.0


def _r2(val):
    """Round to 2 decimals."""
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return 0.0


def _r3(val):
    """Round to 3 decimals."""
    try:
        return round(float(val), 3)
    except (TypeError, ValueError):
        return 0.0


def _extract_thesis(scenario_result):
    """Extract thesis dict from scenario_result (handles both key patterns)."""
    if not scenario_result or not isinstance(scenario_result, dict):
        return {}
    for key in ("current_thesis", "thesis"):
        t = scenario_result.get(key)
        if isinstance(t, dict) and t.get("dominant_thesis"):
            return t
    return {}


def _extract_stress_test(scenario_result):
    """Extract stress test result summary."""
    if not scenario_result or not isinstance(scenario_result, dict):
        return ""
    st = scenario_result.get("stress_test")
    if isinstance(st, dict):
        dangerous = st.get("most_dangerous_alternative", "")
        attack = st.get("attack_summary", "")
        if dangerous:
            return f"Most dangerous alternative: {dangerous}. {attack}"
    return ""


def _read_power_score_history(sheet_writer, existing_dashboard):
    """
    Read last 12 weeks of power score history.
    Primary: from G7_POWER_SCORE_HISTORY tab via sheet_writer.
    Fallback: from existing dashboard g7 block (append current, cap at 12).
    """
    # Try Sheet read
    if sheet_writer:
        try:
            rows = sheet_writer._read_range("G7_POWER_SCORE_HISTORY!A1:Z500")
            if rows and len(rows) >= 2:
                header = rows[0]
                history = []
                # Take last 12 data rows
                for row in rows[-12:] if len(rows) > 13 else rows[1:]:
                    entry = {"date": row[0] if row else ""}
                    scores = {}
                    for region in REGIONS:
                        col_name = f"{region}_score"
                        if col_name in header:
                            idx = header.index(col_name)
                            try:
                                scores[region] = round(float(row[idx]), 1) if idx < len(row) else 0
                            except (ValueError, TypeError):
                                scores[region] = 0
                    entry["scores"] = scores

                    # Gap
                    if "usa_china_gap" in header:
                        idx = header.index("usa_china_gap")
                        try:
                            entry["gap"] = round(float(row[idx]), 1) if idx < len(row) else 0
                        except (ValueError, TypeError):
                            entry["gap"] = 0

                    history.append(entry)
                return history
        except Exception as e:
            logger.warning(f"Power score history read failed: {e}")

    # Fallback: use existing history from dashboard and return as-is
    existing_g7 = existing_dashboard.get("g7", {})
    return existing_g7.get("power_scores_history", [])


def _update_overlay_history(existing_dashboard, overlays_current, date_str):
    """
    Append current overlay values to history, keep max 12 entries.
    Stored inside the g7 block itself (no separate Sheet tab).
    """
    existing_g7 = existing_dashboard.get("g7", {})
    history = list(existing_g7.get("overlay_history", []))

    # Only append if latest entry is a different date
    entry_date = date_str[:10]  # "2026-03-08"
    if history and history[-1].get("date", "")[:10] == entry_date:
        # Update today's entry instead of appending
        history[-1] = {
            "date": entry_date,
            "scsi": overlays_current["scsi"]["value"],
            "ddi": overlays_current["ddi"]["value"],
            "fdp_usa": overlays_current["fdp_usa"]["value"],
            "ewi_signals": overlays_current["ewi"]["active_signals"],
        }
    else:
        history.append({
            "date": entry_date,
            "scsi": overlays_current["scsi"]["value"],
            "ddi": overlays_current["ddi"]["value"],
            "fdp_usa": overlays_current["fdp_usa"]["value"],
            "ewi_signals": overlays_current["ewi"]["active_signals"],
        })

    # Cap at 12
    return history[-12:]


def _build_sit_block(sit_result, overlays):
    """
    Build compact SIT block for frontend.
    Source: overlays dict contains sit data from phase4.
    """
    sit = sit_result if sit_result else overlays.get("sit", {})
    if not sit or not isinstance(sit, dict):
        return {
            "global_trend": "UNKNOWN",
            "dominant_driver": "",
            "portfolio_vetos": [],
            "severity_by_region": {},
            "highlights": {},
        }

    # Extract from SIT result structure
    global_trend = sit.get("global_escalation_trend", sit.get("escalation_trend", "UNKNOWN"))

    # Severity by region
    severity_by_region = {}
    highlights = {}
    regions_data = sit.get("regions", sit.get("region_data", {}))
    if isinstance(regions_data, dict):
        for region in REGIONS:
            rd = regions_data.get(region, {})
            if isinstance(rd, dict):
                severity_by_region[region] = rd.get("severity_score", 0)
                hl = rd.get("highlight", "")
                if hl:
                    highlights[region] = hl

    # Dominant driver: region with highest severity
    dominant_driver = ""
    if severity_by_region:
        top_region = max(severity_by_region, key=severity_by_region.get)
        top_sev = severity_by_region[top_region]
        if top_sev > 2:
            dominant_driver = highlights.get(top_region, f"{top_region} sanctions elevated")

    # Portfolio vetos: assets affected by high-severity sanctions
    portfolio_vetos = []
    china_sev = severity_by_region.get("CHINA", 0)
    if china_sev >= 4:
        portfolio_vetos.extend(["KWEB", "FXI"])
    russia_sev = severity_by_region.get("REST_EM", 0)
    if russia_sev >= 6:
        portfolio_vetos.append("RSX")

    return {
        "global_trend": global_trend,
        "dominant_driver": dominant_driver,
        "portfolio_vetos": portfolio_vetos,
        "severity_by_region": severity_by_region,
        "highlights": highlights,
    }
