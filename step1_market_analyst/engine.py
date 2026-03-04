"""
Step 1 — Market Analyst Engine
Main orchestrator. Runs daily at 06:00 UTC via GitHub Actions.

10 Phases:
  0: Setup (load configs)
  1: Read inputs (RAW_MARKET, RAW_MACRO, IC, V16 state, history)
  2: Normalize (field -> sub-score)
  3: Layer scores (sub-scores -> weighted score -> regime)
  4: Dynamics (velocity, acceleration, direction, surprise, regime history, transition)
  5: Conviction (4D calculation)
  6: Cross-layer checks (divergence detection)
  7: Cascade detection (temporal lag checks)
  8: Templates (key driver + tension generation)
  9: System synthesis + fragility monitor
  10: Output (DW tabs + JSON files)

Runtime target: <30 seconds. LLM calls: 0. External API calls: 0.
"""

import os
import sys
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

# --- Modules ---
from modules.normalization import normalize_field
from modules.layer_calculator import (
    calculate_layer_score, assign_regime, calculate_data_clarity, get_layer_id,
)
from modules.signal_phase import detect_signal_phase
from modules.signal_quality import (
    determine_signal_quality, apply_staleness_penalty,
)
from modules.dynamics import (
    calculate_velocity, calculate_acceleration, calculate_direction,
    get_score_n_days_ago, get_historical_daily_deltas, extract_layer_history,
)
from modules.surprise import calculate_surprise
from modules.transitions import (
    calculate_regime_history, calculate_transition_proximity,
)
from modules.conviction import calculate_conviction
from modules.cross_checks import run_cross_checks
from modules.cascades import check_cascades
from modules.catalysts import calculate_catalyst_exposure
from modules.ic_integration import (
    calculate_ic_status, determine_ic_weight, detect_thesis_shifts, get_data_direction,
)
from modules.templates import select_key_driver, generate_tension
from modules.system_synthesis import synthesize_system_regime
from modules.fragility_monitor import (
    calculate_fragility_state, check_crisis_condition, get_consequence_recommendations,
)

# --- CONFIG ---

DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"
DRIVE_ROOT_ID = "1V4BHq3IRd0-ApxmjWE0vOiLZhOVA8tFE"

LAYER_NAMES = [
    "Global Liquidity Cycle (L1)",
    "Macro Regime (L2)",
    "Earnings & Fundamentals (L3)",
    "Cross-Border Flows & FX (L4)",
    "Risk Appetite & Sentiment (L5)",
    "Relative Value & Asset Rotation (L6)",
    "Central Bank Policy Divergence (L7)",
    "Tail Risk & Black Swan (L8)",
]

CONFIG_DIR = Path(__file__).parent / "config"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Step1")


# ============================================================
# PHASE 0: SETUP
# ============================================================

def load_configs() -> dict:
    configs = {}
    config_files = [
        "field_properties", "normalization", "layer_field_mapping",
        "layer_regimes", "ic_integration", "cross_checks", "cascades",
        "catalysts", "templates", "signal_quality_rules", "fragility_monitor",
    ]
    for name in config_files:
        path = CONFIG_DIR / f"{name}.json"
        with open(path) as f:
            configs[name] = json.load(f)
        log.info(f"  Loaded {name}.json")
    return configs


def connect_google():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    if not os.path.exists(creds_path):
        raw = os.environ.get("GOOGLE_CREDENTIALS", "")
        if not raw:
            raw = os.environ.get("GCP_SA_KEY", "")
        if raw:
            with open(creds_path, "w") as f:
                f.write(raw)
        else:
            raise FileNotFoundError("No GCP credentials found")
    creds = Credentials.from_service_account_file(creds_path, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    gc = gspread.authorize(creds)
    dw_sheet = gc.open_by_key(DW_SHEET_ID)
    drive_service = build("drive", "v3", credentials=creds)
    return dw_sheet, drive_service


# ============================================================
# PHASE 1: READ INPUTS
# ============================================================

def read_raw_data_structured(dw_sheet) -> dict:
    """
    Reads RAW_AGENT2 tab (fed by step_0b_agent_feeder).
    DW stores fields as rows with standardized columns:
    Field | Value | Pctl_1Y | Direction | Delta_5D | Delta_5D_Norm | Confidence | Anomaly
    """
    raw_data = {}

    for tab_name in ["RAW_AGENT2"]:
        try:
            ws = dw_sheet.worksheet(tab_name)
            data = ws.get_all_values()
            if len(data) < 3:
                log.warning(f"  {tab_name}: Empty or no data rows")
                continue

            # RAW_AGENT2 has title in row 1, headers in row 2, data from row 3
            # Detect by checking if row 0 looks like a title (no 'field' header)
            row0_lower = [h.strip().lower() for h in data[0]]
            if "field" in row0_lower or "field_name" in row0_lower:
                header_idx = 0
                data_start = 1
            else:
                header_idx = 1
                data_start = 2

            headers = [h.strip().lower() for h in data[header_idx]]
            field_col = _find_col(headers, ["field", "field_name", "name"])
            value_col = _find_col(headers, ["value", "latest"])
            pctl_col = _find_col(headers, ["pctl_1y", "pctl", "percentile"])
            dir_col = _find_col(headers, ["direction", "dir"])
            d5_col = _find_col(headers, ["delta_5d", "delta5d"])
            d5n_col = _find_col(headers, ["delta_5d_norm", "delta5d_norm"])
            conf_col = _find_col(headers, ["confidence", "conf"])
            anom_col = _find_col(headers, ["anomaly", "anomaly_flag"])

            for row in data[data_start:]:
                if field_col is None or field_col >= len(row):
                    continue
                field_name = row[field_col].strip()
                if not field_name:
                    continue

                raw_data[field_name] = {
                    "value": _safe_float(row, value_col),
                    "pctl_1y": _safe_float(row, pctl_col),
                    "direction": _safe_str(row, dir_col, "FLAT"),
                    "delta_5d": _safe_float(row, d5_col),
                    "delta_5d_norm": _safe_float(row, d5n_col, 0),
                    "confidence": _safe_float(row, conf_col, 1.0),
                    "anomaly_flag": _safe_str(row, anom_col, "OK"),
                }

            log.info(f"  {tab_name}: {len([r for r in data[1:] if r and r[0].strip()])} fields read")
        except gspread.exceptions.WorksheetNotFound:
            log.warning(f"  {tab_name}: Tab not found")
        except Exception as e:
            log.error(f"  {tab_name}: Error — {e}")

    return raw_data


def read_v16_state(dw_sheet) -> dict:
    try:
        ws = dw_sheet.worksheet("CONFIG")
        data = ws.get_all_values()
        config_dict = {}
        for row in data[1:]:
            if len(row) >= 2:
                config_dict[row[0].strip()] = row[1].strip()
        return {
            "v16_state": config_dict.get("v16_state", "Risk-On"),
            "v16_confluence": _safe_float_val(config_dict.get("v16_confluence", "0")),
        }
    except Exception as e:
        log.warning(f"  V16 state read failed: {e} — defaulting Risk-On")
        return {"v16_state": "Risk-On", "v16_confluence": 0}


def read_ic_data(drive_service) -> dict:
    try:
        folder_id = _find_folder(drive_service, "CURRENT", DRIVE_ROOT_ID)
        if not folder_id:
            log.warning("  CURRENT/ folder not found — IC DEGRADED")
            return {}

        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and name='step0b_ic_intelligence.json' and trashed=false",
            fields="files(id,name)",
        ).execute()
        files = results.get("files", [])
        if not files:
            log.info("  No IC data found — graceful degradation")
            return {}

        content = drive_service.files().get_media(fileId=files[0]["id"]).execute()
        return json.loads(content.decode("utf-8"))
    except Exception as e:
        log.warning(f"  IC read failed: {e} — running without IC")
        return {}


def read_history(dw_sheet) -> list:
    try:
        ws = dw_sheet.worksheet("BELIEFS")
        data = ws.get_all_values()
        if len(data) < 2:
            return []

        history = []
        for row in data[1:31]:
            if not row or not row[0]:
                continue
            day_record = {"date": row[0], "layers": {}}
            col = 1
            for layer_name in LAYER_NAMES:
                if col + 2 < len(row):
                    day_record["layers"][layer_name] = {
                        "score": _safe_int(row[col]),
                        "regime": row[col + 1] if col + 1 < len(row) else "UNKNOWN",
                        "direction": row[col + 2] if col + 2 < len(row) else "STABLE",
                    }
                col += 3
            history.append(day_record)

        log.info(f"  BELIEFS: {len(history)} days of history")
        return history
    except gspread.exceptions.WorksheetNotFound:
        log.info("  BELIEFS tab not found — first run, no history")
        return []
    except Exception as e:
        log.warning(f"  BELIEFS read failed: {e}")
        return []


# ============================================================
# PHASES 2-9: CORE PROCESSING
# ============================================================

def run_analysis(raw_data, v16_state, ic_data, history, configs, today):
    v16_s = v16_state.get("v16_state", "Risk-On")
    field_props = configs["field_properties"]
    norm_config = configs["normalization"]
    layer_mapping = configs["layer_field_mapping"]
    layer_regimes = configs["layer_regimes"]
    ic_config = configs["ic_integration"]
    sq_config = configs["signal_quality_rules"]
    cross_check_config = configs["cross_checks"]
    cascade_config = configs["cascades"]
    catalyst_config = configs["catalysts"]
    template_config = configs["templates"]
    fragility_config = configs["fragility_monitor"]

    # PHASE 2: Normalize
    log.info("Phase 2: Normalizing fields...")
    all_sub_scores = {}
    for field_name, norm_params in norm_config.items():
        if field_name.startswith("_"):
            continue
        field_data = raw_data.get(field_name, {})
        if not isinstance(field_data, dict):
            field_data = {"value": field_data}
        confidence = field_data.get("confidence", 1.0) or 1.0
        sub_score = normalize_field(field_data, norm_params)
        sub_score = apply_staleness_penalty(sub_score, confidence)
        all_sub_scores[field_name] = sub_score
    log.info(f"  {len(all_sub_scores)} fields normalized")

    # PHASE 3: Layer Scores
    log.info("Phase 3: Calculating layer scores...")
    layer_results = {}
    for layer_name in LAYER_NAMES:
        layer_fields_config = layer_mapping.get(layer_name, {})
        data_field_names = [f for f in layer_fields_config if not f.startswith("_")]

        layer_sub_scores = {}
        for field in data_field_names:
            if field.startswith("ic_"):
                continue
            if field in all_sub_scores:
                layer_sub_scores[field] = all_sub_scores[field]

        data_clarity = calculate_data_clarity(layer_sub_scores)

        # IC integration
        data_dir = get_data_direction(layer_sub_scores)
        ic_status = calculate_ic_status(ic_data, layer_name, ic_config, data_dir)
        ic_weight = determine_ic_weight(data_clarity, ic_status, ic_config)
        ic_status["ic_weight_used"] = ic_weight
        high_novelty = ic_data.get("high_novelty_claims", [])
        shifts = detect_thesis_shifts(high_novelty, layer_name, ic_config["topic_layer_mapping"])
        ic_status["ic_thesis_shift"] = shifts

        for field in data_field_names:
            if field.startswith("ic_") and ic_status.get("ic_score") is not None:
                layer_sub_scores[field] = ic_status["ic_score"]
                layer_fields_config[field] = {"risk_on": ic_weight, "risk_off": ic_weight}

        score = calculate_layer_score(layer_sub_scores, layer_fields_config, v16_s)
        signal_phase = detect_signal_phase(layer_sub_scores, {}, field_props)
        data_fields = [f for f in data_field_names if not f.startswith("ic_")]
        sq = determine_signal_quality(layer_name, data_fields, raw_data, sq_config, data_clarity)

        layer_results[layer_name] = {
            "layer_id": get_layer_id(layer_name),
            "layer_name": layer_name,
            "score": score,
            "sub_scores": layer_sub_scores,
            "data_clarity": data_clarity,
            "signal_phase": signal_phase,
            "signal_quality": sq,
            "ic_status": ic_status,
        }

    for layer_name, lr in layer_results.items():
        lr["regime"] = assign_regime(lr["score"], layer_name, layer_regimes)
    log.info("  8 layers scored")

    # PHASE 4: Dynamics
    log.info("Phase 4: Calculating dynamics...")
    for layer_name, lr in layer_results.items():
        layer_hist = extract_layer_history(history, layer_name)
        s = lr["score"]
        s_1d = get_score_n_days_ago(layer_hist, 1)
        s_5d = get_score_n_days_ago(layer_hist, 5)
        s_10d = get_score_n_days_ago(layer_hist, 10)
        s_21d = get_score_n_days_ago(layer_hist, 21)

        velocity = calculate_velocity(s, s_1d, s_5d)
        d5c = (s - s_5d) if s_5d is not None else None
        d5p = (s_5d - s_10d) if (s_5d is not None and s_10d is not None) else None
        acceleration = calculate_acceleration(d5c, d5p)
        direction = calculate_direction(s, s_5d, s_21d)
        daily_deltas = get_historical_daily_deltas(layer_hist)
        surprise = calculate_surprise(s, s_1d, daily_deltas)
        regime_hist = calculate_regime_history(history, layer_name)
        layer_regime_config = layer_regimes.get(layer_name, {})
        transition = calculate_transition_proximity(s, lr["regime"], layer_regime_config, velocity, acceleration)

        lr.update({
            "velocity": velocity, "acceleration": acceleration, "direction": direction,
            "surprise": surprise, "regime_history": regime_hist, "transition_proximity": transition,
        })
        lr["regime"] = assign_regime(lr["score"], layer_name, layer_regimes, direction)

    # PHASE 5: Conviction
    log.info("Phase 5: Calculating conviction...")
    for layer_name, lr in layer_results.items():
        catalyst_exp = calculate_catalyst_exposure(layer_name, catalyst_config, ic_data.get("catalysts", []), today)
        lr["catalyst_exposure"] = catalyst_exp
        lr["conviction"] = calculate_conviction({
            "raw_data_clarity": lr["data_clarity"],
            "ic_confirmation": lr["ic_status"]["ic_confirmation"],
            "ic_dissent": lr["ic_status"]["ic_dissent"],
            "signal_phase": lr["signal_phase"],
            "catalyst_exposure": catalyst_exp,
            "regime_history": lr["regime_history"],
            "surprise": lr["surprise"],
        })

    # PHASE 6: Cross-Layer Checks
    log.info("Phase 6: Cross-layer checks...")
    active_cross_checks = run_cross_checks(layer_results, cross_check_config)
    log.info(f"  {len(active_cross_checks)} active flags")

    # PHASE 7: Cascade Detection
    log.info("Phase 7: Cascade detection...")
    active_cascades = check_cascades(layer_results, history, cascade_config, today)
    log.info(f"  {len(active_cascades)} cascades ({len([c for c in active_cascades if c['status']=='EXPECTED'])} EXPECTED)")

    # PHASE 8: Templates
    log.info("Phase 8: Generating templates...")
    for layer_name, lr in layer_results.items():
        lfc = layer_mapping.get(layer_name, {})
        lr["key_driver"] = select_key_driver(layer_name, lr["sub_scores"], raw_data, template_config, lfc)
        lr["tension"] = generate_tension(layer_name, lr["sub_scores"], raw_data, template_config)

    # PHASE 9: System Synthesis + Fragility
    log.info("Phase 9: System synthesis...")
    system_regime = synthesize_system_regime(layer_results)

    fragility_state = calculate_fragility_state(
        hhi=_get_raw_value(raw_data, "sp500_hhi"),
        breadth_pct=_get_raw_value(raw_data, "pct_above_200dma"),
        spy_rsp_6m_delta=_get_raw_value(raw_data, "spy_rsp_6m_delta"),
        ai_gap_data=raw_data.get("ai_capex_revenue_gap"),
        fragility_config=fragility_config,
    )
    fragility_recs = get_consequence_recommendations(fragility_state["state"], fragility_config)
    log.info(f"  System: {system_regime['regime']} | Fragility: {fragility_state['state']}")

    surprise_alerts = [
        {"layer": ln, "z_score": lr["surprise"]["z_score"], "category": lr["surprise"]["category"]}
        for ln, lr in layer_results.items()
        if lr["surprise"]["category"] in ("EXTREME", "HIGH")
    ]

    return {
        "date": today.isoformat(),
        "run_timestamp": datetime.utcnow().isoformat(),
        "v16_state": v16_state,
        "system_regime": system_regime,
        "fragility_state": fragility_state,
        "fragility_recommendations": fragility_recs,
        "active_cross_checks": active_cross_checks,
        "active_cascades": active_cascades,
        "surprise_alerts": surprise_alerts,
        "layers": {ln: _build_layer_output(lr) for ln, lr in layer_results.items()},
        "layer_summary": {
            ln: {
                "score": lr["score"], "regime": lr["regime"], "direction": lr["direction"],
                "velocity": lr["velocity"], "conviction": lr["conviction"]["composite"],
                "limiting_factor": lr["conviction"]["limiting_factor"]["factor"],
            }
            for ln, lr in layer_results.items()
        },
    }


def _build_layer_output(lr):
    return {
        "layer_id": lr["layer_id"], "layer_name": lr["layer_name"],
        "score": lr["score"], "regime": lr["regime"],
        "direction": lr["direction"], "velocity": lr["velocity"], "acceleration": lr["acceleration"],
        "conviction": lr["conviction"], "surprise": lr["surprise"],
        "transition_proximity": lr["transition_proximity"], "regime_history": lr["regime_history"],
        "signal_phase": lr["signal_phase"], "signal_quality": lr["signal_quality"],
        "key_driver": lr["key_driver"], "tension": lr["tension"],
        "catalyst_exposure": lr["catalyst_exposure"],
        "ic_status": {
            "confirmation": lr["ic_status"]["ic_confirmation"],
            "dissent": lr["ic_status"]["ic_dissent"],
            "thesis_shift": lr["ic_status"]["ic_thesis_shift"],
            "weight_used": lr["ic_status"]["ic_weight_used"],
        },
        "sub_scores": lr["sub_scores"], "data_clarity": lr["data_clarity"],
    }


# ============================================================
# PHASE 10: OUTPUT
# ============================================================

def write_scores_tab(dw_sheet, output):
    try:
        ws = dw_sheet.worksheet("SCORES")
    except gspread.exceptions.WorksheetNotFound:
        log.warning("  SCORES tab not found — skipping")
        return
    today = output["date"]
    rows = []
    for layer_name in LAYER_NAMES:
        lr = output["layers"].get(layer_name, {})
        rows.append([
            today, lr.get("layer_id", ""), layer_name, lr.get("score", 0),
            lr.get("regime", "UNKNOWN"), lr.get("direction", "STABLE"),
            lr.get("velocity", "STEADY"), lr.get("conviction", {}).get("composite", "LOW"),
            lr.get("conviction", {}).get("limiting_factor", {}).get("factor", ""),
            lr.get("signal_phase", "MIXED"), lr.get("signal_quality", {}).get("status", "CONFIRMED"),
            lr.get("key_driver", "")[:300], (lr.get("tension") or "")[:300],
        ])
    _write_rows_to_tab(ws, rows, today, 13)
    log.info(f"  SCORES: {len(rows)} rows written")


def write_divergence_tab(dw_sheet, output):
    try:
        ws = dw_sheet.worksheet("DIVERGENCE")
    except gspread.exceptions.WorksheetNotFound:
        log.warning("  DIVERGENCE tab not found — skipping")
        return
    today = output["date"]
    rows = []
    for check in output.get("active_cross_checks", []):
        rows.append([today, "CROSS_CHECK", check["check_id"], check["name"],
                      ", ".join(check.get("layers_involved", [])), check["tension"][:300]])
    for cascade in output.get("active_cascades", []):
        rows.append([today, "CASCADE", cascade["cascade_id"], cascade["name"],
                      f"{cascade['source_layer']} -> {cascade['target_layer']}",
                      f"{cascade['status']} ({cascade['lag_window']})"[:300]])
    if rows:
        _write_rows_to_tab(ws, rows, today, 6)
    log.info(f"  DIVERGENCE: {len(rows)} rows written")


def write_agent_summary(dw_sheet, output):
    try:
        ws = dw_sheet.worksheet("AGENT_SUMMARY")
    except gspread.exceptions.WorksheetNotFound:
        log.warning("  AGENT_SUMMARY tab not found — skipping")
        return
    today = output["date"]
    sys_reg = output["system_regime"]["regime"]
    frag = output["fragility_state"]["state"]
    n_cross = len(output.get("active_cross_checks", []))
    n_cascade = len([c for c in output.get("active_cascades", []) if c["status"] == "EXPECTED"])
    n_surprise = len(output.get("surprise_alerts", []))
    layer_parts = []
    for ln in LAYER_NAMES:
        ls = output["layer_summary"].get(ln, {})
        lid = get_layer_id(ln)
        layer_parts.append(f"{lid}:{ls.get('score', 0)}({ls.get('regime', '?')[:4]})")
    layer_str = " | ".join(layer_parts)

    # Find the "Agent 1" / "Senior Macro" row and overwrite it
    all_data = ws.get_all_values()
    target_row = None
    for i, row in enumerate(all_data):
        if len(row) >= 3 and ("Agent 1" in str(row[1]) or "Senior Macro" in str(row[2])):
            target_row = i + 1  # 1-indexed
            break
    if target_row is None:
        # Fallback: write to row 2 (after title)
        target_row = 2

    row_data = [today, "Step1_MarketAnalyst", sys_reg, frag, n_cross, n_cascade, n_surprise,
                layer_str[:300], output["run_timestamp"]]
    # Pad to 9 cols
    while len(row_data) < 9:
        row_data.append("")
    row_data = [str(v) if v is not None else "" for v in row_data]

    ws.update(values=[row_data], range_name=f"A{target_row}:I{target_row}",
              value_input_option="RAW")
    log.info(f"  AGENT_SUMMARY: {sys_reg} | Fragility={frag} (row {target_row})")


def write_beliefs_tab(dw_sheet, output):
    try:
        ws = dw_sheet.worksheet("BELIEFS")
    except gspread.exceptions.WorksheetNotFound:
        log.warning("  BELIEFS tab not found — skipping")
        return
    today = output["date"]
    row = [today]
    for layer_name in LAYER_NAMES:
        lr = output["layers"].get(layer_name, {})
        row.extend([lr.get("score", 0), lr.get("regime", "UNKNOWN"), lr.get("direction", "STABLE")])
    row = [str(v) if v is not None else "" for v in row]
    n_cols = len(row)
    end_col = _col_letter(n_cols)
    # Write to row 2 (after title row 1)
    ws.update(values=[row], range_name=f"A2:{end_col}2",
              value_input_option="RAW")
    log.info("  BELIEFS: 1 row written (row 2)")


def write_json_to_drive(drive_service, output):
    try:
        json_bytes = json.dumps(output, indent=2, default=str).encode("utf-8")
        filename = "step1_market_analyst.json"
        current_id = _find_folder(drive_service, "CURRENT", DRIVE_ROOT_ID)
        if current_id:
            _upload_or_replace(drive_service, current_id, filename, json_bytes)
            log.info(f"  Drive CURRENT/{filename} written")
        archive_id = _find_folder(drive_service, "ARCHIVE", DRIVE_ROOT_ID)
        if archive_id:
            date_folder_id = _find_or_create_folder(drive_service, output["date"], archive_id)
            _upload_or_replace(drive_service, date_folder_id, filename, json_bytes)
            log.info(f"  Drive ARCHIVE/{output['date']}/{filename} written")
    except Exception as e:
        log.warning(f"  Drive write failed (non-fatal): {e}")
        log.warning("  Sheet outputs were written successfully — Drive archive skipped")


# ============================================================
# GOOGLE DRIVE HELPERS
# ============================================================

def _find_folder(drive_service, name, parent_id):
    try:
        results = drive_service.files().list(
            q=f"'{parent_id}' in parents and name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id)",
        ).execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None


def _find_or_create_folder(drive_service, name, parent_id):
    folder_id = _find_folder(drive_service, name, parent_id)
    if folder_id:
        return folder_id
    metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = drive_service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def _upload_or_replace(drive_service, folder_id, filename, content_bytes):
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and name='{filename}' and trashed=false",
        fields="files(id)",
    ).execute()
    existing = results.get("files", [])
    media = MediaInMemoryUpload(content_bytes, mimetype="application/json")
    if existing:
        drive_service.files().update(fileId=existing[0]["id"], media_body=media).execute()
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        drive_service.files().create(body=metadata, media_body=media).execute()


# ============================================================
# SHEET WRITE HELPERS
# ============================================================

def _write_rows_to_tab(ws, rows, today_str, expected_cols):
    """Write rows to tab using update (not insert) to preserve layout.
    
    Strategy: Find the data section (rows between title and next section header),
    write into those rows using update(). Never insert or delete rows.
    """
    padded = []
    for row in rows:
        r = [str(v) if v is not None else "" for v in row]
        while len(r) < expected_cols:
            r.append("")
        padded.append(r)

    if not padded:
        return

    # Find first empty or data row after row 1 (title row)
    # The data section starts at row 2 in all tabs
    all_data = ws.get_all_values()
    
    # Find where to write: look for first row that is either empty,
    # has a date, or has "—" in col A (placeholder)
    data_start = 2  # Row 2 (1-indexed) = after title
    
    # Check if row 2 is a header row (contains column names like DATE, LAYER, etc.)
    if len(all_data) >= 2:
        row2_upper = [str(c).strip().upper() for c in all_data[1]]
        header_keywords = {"DATE", "LAYER", "BELIEF_ID", "COLUMN", "COMBINATION", "SOURCE"}
        if any(kw in row2_upper for kw in header_keywords):
            data_start = 3  # Skip header, write from row 3

    # First: clear any existing rows for today (overwrite with blanks)
    for i, existing_row in enumerate(all_data):
        row_idx = i + 1  # 1-indexed
        if row_idx < data_start:
            continue
        if len(existing_row) > 0 and existing_row[0] == today_str:
            blank = [""] * expected_cols
            ws.update(values=[blank], range_name=f"A{row_idx}:{chr(64+expected_cols)}{row_idx}",
                      value_input_option="RAW")

    # Write new data starting at data_start
    end_row = data_start + len(padded) - 1
    # Use column letter calculation for ranges > 26 cols
    end_col = _col_letter(expected_cols)
    ws.update(values=padded, range_name=f"A{data_start}:{end_col}{end_row}",
              value_input_option="RAW")


def _col_letter(n):
    """Convert column number (1-indexed) to Excel letter (A, B, ..., Z, AA, AB...)."""
    result = ""
    while n > 0:
        n -= 1
        result = chr(65 + n % 26) + result
        n //= 26
    return result


# ============================================================
# DATA HELPERS
# ============================================================

def _find_col(headers, candidates):
    for i, h in enumerate(headers):
        if h in candidates:
            return i
    return None


def _safe_float(row, col_idx, default=None):
    if col_idx is None or col_idx >= len(row):
        return default
    try:
        val = row[col_idx].strip()
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _safe_float_val(val, default=0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_str(row, col_idx, default=""):
    if col_idx is None or col_idx >= len(row):
        return default
    val = row[col_idx].strip()
    return val if val else default


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _get_raw_value(raw_data, field_name):
    fd = raw_data.get(field_name)
    if isinstance(fd, dict):
        return fd.get("value")
    return fd


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=" * 60)
    log.info("Step 1 — Market Analyst — Starting")
    log.info("=" * 60)

    today = date.today()

    log.info("Phase 0: Loading configs...")
    configs = load_configs()

    log.info("Phase 0: Connecting to Google...")
    dw_sheet, drive_service = connect_google()

    log.info("Phase 1: Reading inputs...")
    raw_data = read_raw_data_structured(dw_sheet)
    v16_state = read_v16_state(dw_sheet)
    ic_data = read_ic_data(drive_service)
    history = read_history(dw_sheet)
    log.info(f"  {len(raw_data)} fields, V16={v16_state['v16_state']}, "
             f"IC={'available' if ic_data else 'DEGRADED'}, History={len(history)}d")

    output = run_analysis(raw_data, v16_state, ic_data, history, configs, today)

    log.info("Phase 10: Writing outputs...")
    write_scores_tab(dw_sheet, output)
    write_divergence_tab(dw_sheet, output)
    write_agent_summary(dw_sheet, output)
    write_beliefs_tab(dw_sheet, output)
    write_json_to_drive(drive_service, output)

    sys_reg = output["system_regime"]["regime"]
    frag = output["fragility_state"]["state"]
    log.info("=" * 60)
    log.info(f"Step 1 COMPLETE — {sys_reg} | Fragility={frag}")
    for ln in LAYER_NAMES:
        ls = output["layer_summary"][ln]
        lid = get_layer_id(ln)
        log.info(f"  {lid}: {ls['score']:+d} {ls['regime']} [{ls['direction']}] "
                 f"Conv={ls['conviction']} LF={ls['limiting_factor']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
