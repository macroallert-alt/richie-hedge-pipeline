"""
step3_risk_officer/main.py
Entry Point — liest Daten aus V16 Production + Market Analyst (Drive JSON / DW Sheet),
fuehrt Risk Officer aus, schreibt RISK_ALERTS + RISK_HISTORY ins DW Sheet,
schreibt JSON nach Drive CURRENT/ + ARCHIVE/.

Aufruf:
  python -m step3_risk_officer                    # Normaler Run
  python -m step3_risk_officer --dry-run          # Kein Sheet-Write
  python -m step3_risk_officer --date 2026-03-04  # Bestimmtes Datum
"""

import os
import sys
import json
import argparse
from datetime import date, datetime

from .engine import run_risk_officer
from .utils.helpers import log_info, log_warning, log_error, parse_date

# ─── Google Sheet IDs ─────────────────────────────────────────────

DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"
V16_SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"

# ─── V16 Tabs ─────────────────────────────────────────────────────

TAB_SIGNAL_HISTORY = "SIGNAL_HISTORY"
TAB_CALC_MACRO_STATE = "CALC_Macro_State"

# ─── DW Tabs ──────────────────────────────────────────────────────

TAB_SCORES = "SCORES"
TAB_RISK_ALERTS = "RISK_ALERTS"
TAB_RISK_HISTORY = "RISK_HISTORY"

# ─── V16 Asset Order (SIGNAL_HISTORY Spalten F-AD) ────────────────

V16_ASSETS = [
    "GLD", "SLV", "GDX", "GDXJ", "SIL",
    "SPY", "XLY", "XLI", "XLF", "XLE",
    "IWM", "XLV", "XLP", "XLU", "VNQ",
    "XLK", "EEM", "VGK", "TLT", "TIP",
    "LQD", "HYG", "DBC", "BTC", "ETH"
]

# ─── DD-Protect State Mapping ─────────────────────────────────────

DD_PROTECT_STATES = {10, 11, 12}
RISK_ON_STATES = {1, 2, 3, 4, 5, 6}
RISK_OFF_STATES = {7, 8, 9}


# ═══════════════════════════════════════════════════════════════════
# GOOGLE SERVICES
# ═══════════════════════════════════════════════════════════════════

def _get_credentials():
    """Erstellt Credentials aus Environment Variable. Gibt (creds, creds_path) zurueck."""
    from google.oauth2.service_account import Credentials

    creds_json = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
    if not creds_json:
        log_error("No GCP_SA_KEY or GOOGLE_CREDENTIALS found in environment")
        return None

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(creds_json)
        creds_path = f.name

    creds = Credentials.from_service_account_file(
        creds_path,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
    )
    os.unlink(creds_path)
    return creds


def get_sheets_service(creds):
    """Erstellt Google Sheets API Service."""
    try:
        from googleapiclient.discovery import build
        service = build("sheets", "v4", credentials=creds)
        return service.spreadsheets()
    except Exception as e:
        log_error(f"Failed to create Sheets service: {e}")
        return None


def get_drive_service(creds):
    """Erstellt Google Drive API Service."""
    try:
        from googleapiclient.discovery import build
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        log_error(f"Failed to create Drive service: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# SHEET HELPERS
# ═══════════════════════════════════════════════════════════════════

def read_sheet_range(sheets, spreadsheet_id, range_str):
    """Liest Range aus Google Sheet. Returns list of lists."""
    try:
        result = sheets.values().get(
            spreadsheetId=spreadsheet_id, range=range_str
        ).execute()
        return result.get("values", [])
    except Exception as e:
        log_error(f"Failed to read {range_str}: {e}")
        return []


def write_sheet_range(sheets, spreadsheet_id, range_str, values):
    """Schreibt Values in Google Sheet."""
    try:
        sheets.values().update(
            spreadsheetId=spreadsheet_id,
            range=range_str,
            valueInputOption="RAW",
            body={"values": values}
        ).execute()
        log_info(f"Written to {range_str}: {len(values)} rows")
        return True
    except Exception as e:
        log_error(f"Failed to write {range_str}: {e}")
        return False


def clear_and_write(sheets, spreadsheet_id, range_str, values):
    """Cleared Range und schreibt neu."""
    try:
        sheets.values().clear(
            spreadsheetId=spreadsheet_id, range=range_str, body={}
        ).execute()
    except Exception:
        pass
    return write_sheet_range(sheets, spreadsheet_id, range_str, values)


# ═══════════════════════════════════════════════════════════════════
# DRIVE HELPERS
# ═══════════════════════════════════════════════════════════════════

def _find_folder(drive_service, name, parent_id):
    """Sucht Ordner nach Name unter parent_id."""
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
    """Sucht oder erstellt Ordner."""
    folder_id = _find_folder(drive_service, name, parent_id)
    if folder_id:
        return folder_id
    metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = drive_service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def _upload_or_replace(drive_service, folder_id, filename, content_bytes):
    """Upload oder Update einer Datei in einem Ordner."""
    from googleapiclient.http import MediaInMemoryUpload

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


def _read_json_from_drive(drive_service, folder_id, filename):
    """Liest eine JSON-Datei aus einem Drive-Ordner. Returns dict oder None."""
    try:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and name='{filename}' and trashed=false",
            fields="files(id,name)",
        ).execute()
        files = results.get("files", [])
        if not files:
            return None
        content = drive_service.files().get_media(fileId=files[0]["id"]).execute()
        return json.loads(content.decode("utf-8"))
    except Exception as e:
        log_warning(f"Drive read failed for {filename}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# INPUT READER
# ═══════════════════════════════════════════════════════════════════

def read_v16_production(sheets):
    """
    Liest V16 State + Gewichte aus V16 Production Sheet.

    SIGNAL_HISTORY: Letzte Zeile = aktuelle Gewichte
      Col A=Date, B=Macro_State, C=Growth, D=Liq_Dir, E=Stress,
      Col F-AD = FM_GLD ... FM_ETH (25 Assets)

    CALC_Macro_State: Letzte Zeile = aktueller Macro State
      Col I=Macro_State_Num, J=Macro_State_Name
    """
    v16 = {}

    # ─── SIGNAL_HISTORY: Gewichte aus letzter Zeile ───────────
    sig_rows = read_sheet_range(sheets, V16_SHEET_ID, f"{TAB_SIGNAL_HISTORY}!A:AD")
    if len(sig_rows) < 2:
        log_warning("SIGNAL_HISTORY empty or header-only")
        return None

    # Letzte non-empty Zeile finden
    last_row = None
    for row in reversed(sig_rows):
        if row and len(row) > 5 and row[0] and row[0] != "Date":
            last_row = row
            break

    if not last_row:
        log_warning("No data rows in SIGNAL_HISTORY")
        return None

    log_info(f"  V16 SIGNAL_HISTORY last row date: {last_row[0]}")

    # Macro State aus SIGNAL_HISTORY
    macro_state_num = None
    try:
        macro_state_num = int(float(last_row[1]))
    except (ValueError, TypeError, IndexError):
        pass

    # Growth, Liq_Dir, Stress
    try:
        v16["growth"] = int(float(last_row[2])) if len(last_row) > 2 and last_row[2] else 0
        v16["liq_dir"] = int(float(last_row[3])) if len(last_row) > 3 and last_row[3] else 0
        v16["stress"] = int(float(last_row[4])) if len(last_row) > 4 and last_row[4] else 0
    except (ValueError, TypeError):
        v16["growth"] = 0
        v16["liq_dir"] = 0
        v16["stress"] = 0

    # Gewichte (Spalte F-AD = Index 5-29)
    weights = {}
    for i, asset in enumerate(V16_ASSETS):
        col_idx = 5 + i
        if col_idx < len(last_row) and last_row[col_idx]:
            try:
                w = float(last_row[col_idx])
                if w > 0:
                    weights[asset] = w
            except (ValueError, TypeError):
                continue

    v16["weights"] = weights

    if not weights:
        log_warning("V16 weights all zero or empty")
        return None

    log_info(f"  V16 weights: {len(weights)} active positions, "
             f"top 3: {sorted(weights.items(), key=lambda x: -x[1])[:3]}")

    # ─── CALC_Macro_State: Aktueller State ────────────────────
    state_rows = read_sheet_range(
        sheets, V16_SHEET_ID, f"{TAB_CALC_MACRO_STATE}!A:J"
    )

    last_state_row = None
    for row in reversed(state_rows):
        if row and len(row) > 8 and row[0] and row[0] != "Date" and row[0] != "Datum":
            last_state_row = row
            break

    if last_state_row:
        try:
            state_num = int(float(last_state_row[8]))  # Col I = Macro_State_Num
        except (ValueError, TypeError, IndexError):
            state_num = macro_state_num or 3

        state_name = last_state_row[9] if len(last_state_row) > 9 else "UNKNOWN"

        log_info(f"  V16 State: {state_num} ({state_name})")

        # V16 State ableiten
        if state_num in DD_PROTECT_STATES:
            v16["v16_state"] = "DD-Protect"
            v16["dd_protect_active"] = True
        elif state_num in RISK_OFF_STATES:
            v16["v16_state"] = "Risk-Off"
            v16["dd_protect_active"] = False
        else:
            v16["v16_state"] = "Risk-On"
            v16["dd_protect_active"] = False

        v16["v16_regime"] = state_name
        v16["macro_state_num"] = state_num
    else:
        v16["v16_state"] = "Risk-On"
        v16["v16_regime"] = "UNKNOWN"
        v16["dd_protect_active"] = False
        v16["macro_state_num"] = macro_state_num

    v16["dd_protect_trigger_level"] = -0.12
    v16["current_drawdown_from_peak"] = 0.0  # TODO: berechnen wenn Performance-Daten verfuegbar

    return v16


def read_market_analyst_from_drive(drive_service):
    """
    PRIMAER: Liest Market Analyst JSON von Drive CURRENT/step1_market_analyst.json.

    Returns dict mit system_regime, fragility_state, layer_scores — oder None.
    """
    drive_root_id = os.environ.get("DRIVE_ROOT_ID", "")
    if not drive_root_id:
        log_warning("DRIVE_ROOT_ID not set — cannot read from Drive")
        return None

    current_id = _find_folder(drive_service, "CURRENT", drive_root_id)
    if not current_id:
        log_warning("CURRENT/ folder not found on Drive")
        return None

    ma_data = _read_json_from_drive(drive_service, current_id, "step1_market_analyst.json")
    if not ma_data:
        log_warning("step1_market_analyst.json not found or empty on Drive")
        return None

    # Validierung: Pflichtfelder pruefen
    if "system_regime" not in ma_data or "layers" not in ma_data:
        log_warning("Market Analyst JSON missing required fields (system_regime, layers)")
        return None

    # Layer Scores extrahieren (layer_id -> score)
    layer_scores = {}
    for layer_name, layer_data in ma_data.get("layers", {}).items():
        layer_id = layer_data.get("layer_id", "")
        if layer_id:
            layer_scores[layer_id] = layer_data.get("score", 0)

    log_info(f"  Market Analyst from Drive: {len(layer_scores)} layers, "
             f"regime={ma_data['system_regime'].get('regime', 'UNKNOWN')}, "
             f"fragility={ma_data.get('fragility_state', {}).get('state', 'UNKNOWN')}")

    return {
        "source": "DRIVE_JSON",
        "system_regime": ma_data["system_regime"],
        "fragility_state": ma_data.get("fragility_state", {"state": "UNKNOWN"}),
        "layer_scores": layer_scores,
        "full_json": ma_data,
    }


def read_layer_analysis_from_sheet(sheets):
    """
    FALLBACK: Liest Market Analyst Output aus DW SCORES Tab.

    SCORES Tab Format:
      Col A = Date
      Col B = Layer-ID (L1, L2, ..., L8)
      Col C = Layer-Name
      Col D = SCORE_RAW (Integer -10 bis +10)
      Col E = Regime
      Col F = Direction
    """
    rows = read_sheet_range(sheets, DW_SHEET_ID, f"{TAB_SCORES}!A:F")

    layer_scores = {}
    layer_regimes = {}
    layer_directions = {}

    for row in rows:
        if not row or len(row) < 4:
            continue

        # Col B = Layer-ID (Index 1)
        layer_id = str(row[1]).strip()
        if not layer_id.startswith("L"):
            continue

        # Col D = SCORE_RAW (Index 3)
        try:
            score = int(float(row[3]))
        except (ValueError, TypeError):
            continue

        layer_scores[layer_id] = score

        # Col E = Regime (Index 4), Col F = Direction (Index 5) — optional
        if len(row) > 4:
            layer_regimes[layer_id] = str(row[4]).strip()
        if len(row) > 5:
            layer_directions[layer_id] = str(row[5]).strip()

    if not layer_scores:
        log_warning("SCORES tab has no valid layer scores")
        return None

    log_info(f"  Layer Analysis from Sheet (FALLBACK): {len(layer_scores)} layers loaded")
    log_info(f"  Scores: {layer_scores}")

    return {
        "source": "SHEET_FALLBACK",
        "system_regime": {"regime": "UNKNOWN", "lean": "UNKNOWN"},
        "fragility_state": {"state": "UNKNOWN", "fallback": True},
        "layer_scores": layer_scores,
    }


def read_layer_analysis(sheets, drive_service):
    """
    Orchestrierer: Versucht Drive JSON zuerst, faellt auf Sheet zurueck.
    """
    # Primaer: Drive JSON
    if drive_service:
        result = read_market_analyst_from_drive(drive_service)
        if result:
            return result
        log_warning("Drive read failed — falling back to SCORES tab")

    # Fallback: Sheet
    if sheets:
        result = read_layer_analysis_from_sheet(sheets)
        if result:
            return result

    # Beides fehlgeschlagen
    log_error("Layer Analysis completely unavailable — using safe defaults")
    return {
        "source": "DEFAULTS",
        "system_regime": {"regime": "UNKNOWN", "lean": "UNKNOWN"},
        "fragility_state": {"state": "UNKNOWN"},
        "layer_scores": {},
    }


def read_risk_history(sheets):
    """Liest gestrigen Risk Officer Output fuer Trend-Erkennung."""
    rows = read_sheet_range(sheets, DW_SHEET_ID, f"{TAB_RISK_HISTORY}!A:L")
    if len(rows) < 2:
        return None

    header = rows[0]
    last = rows[-1]
    history = dict(zip(header, last))

    try:
        return {
            "portfolio_status": history.get("portfolio_status", "GREEN"),
            "v16_state": history.get("v16_state", "Risk-On"),
            "router_state": history.get("router_state", "US_DOMESTIC"),
            "fragility_state": history.get("fragility_state", "HEALTHY"),
            "alerts": json.loads(history.get("alerts_json", "[]"))
        }
    except (json.JSONDecodeError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════
# OUTPUT WRITER — SHEETS
# ═══════════════════════════════════════════════════════════════════

def write_risk_alerts(sheets, output):
    """Schreibt aktuellen Risk Officer Output in RISK_ALERTS Tab."""
    header = [
        "date", "run_timestamp", "execution_path", "portfolio_status",
        "portfolio_status_reason", "alerts_json", "ongoing_json",
        "spy_beta", "effective_positions", "g7_status",
        "risk_summary", "metadata_json"
    ]

    row = [
        output["date"],
        output["run_timestamp"],
        output["execution_path"],
        output["portfolio_status"],
        output["portfolio_status_reason"],
        json.dumps(output["alerts"], default=str),
        json.dumps(output["ongoing_conditions"], default=str),
        str(output["sensitivity"].get("spy_beta") or ""),
        str(output["sensitivity"].get("effective_positions") or ""),
        output["g7_context"].get("status", ""),
        output["risk_summary"],
        json.dumps(output["metadata"], default=str)
    ]

    return clear_and_write(
        sheets, DW_SHEET_ID, f"{TAB_RISK_ALERTS}!A1",
        [header, row]
    )


def append_risk_history(sheets, output):
    """Haengt kompakte Zeile an RISK_HISTORY Tab an."""
    existing = read_sheet_range(sheets, DW_SHEET_ID, f"{TAB_RISK_HISTORY}!A1:A1")
    if not existing:
        header = [
            "date", "portfolio_status", "alerts_json", "spy_beta",
            "effective_positions", "v16_state", "router_state",
            "fragility_state", "alerts_count", "ongoing_count",
            "execution_path", "execution_time_ms"
        ]
        write_sheet_range(
            sheets, DW_SHEET_ID, f"{TAB_RISK_HISTORY}!A1", [header]
        )

    compact_alerts = [
        {
            "check_id": a.get("check_id"),
            "severity": a.get("severity"),
            "trend": a.get("trend"),
            "days_active": a.get("days_active")
        }
        for a in output["alerts"]
        if a.get("severity") != "RESOLVED"
    ]

    row = [
        output["date"],
        output["portfolio_status"],
        json.dumps(compact_alerts, default=str),
        str(output["sensitivity"].get("spy_beta") or ""),
        str(output["sensitivity"].get("effective_positions") or ""),
        output["metadata"].get("v16_state", ""),
        output["metadata"].get("router_state", ""),
        output["metadata"].get("fragility_state", ""),
        str(output["metadata"].get("alerts_count", 0)),
        str(output["metadata"].get("ongoing_conditions_count", 0)),
        output["execution_path"],
        str(output["metadata"].get("execution_time_ms", 0))
    ]

    return sheets.values().append(
        spreadsheetId=DW_SHEET_ID,
        range=f"{TAB_RISK_HISTORY}!A:L",
        valueInputOption="RAW",
        body={"values": [row]}
    ).execute()


# ═══════════════════════════════════════════════════════════════════
# OUTPUT WRITER — DRIVE
# ═══════════════════════════════════════════════════════════════════

def write_json_to_drive(drive_service, output):
    """Schreibt Risk Officer JSON nach CURRENT/ und ARCHIVE/YYYY-MM-DD/."""
    drive_root_id = os.environ.get("DRIVE_ROOT_ID", "")
    if not drive_root_id:
        log_warning("DRIVE_ROOT_ID not set — skipping Drive write")
        return

    try:
        json_bytes = json.dumps(output, indent=2, default=str).encode("utf-8")
        filename = "step3_risk_officer.json"

        current_id = _find_folder(drive_service, "CURRENT", drive_root_id)
        if current_id:
            _upload_or_replace(drive_service, current_id, filename, json_bytes)
            log_info(f"  Drive CURRENT/{filename} written")

        archive_id = _find_folder(drive_service, "ARCHIVE", drive_root_id)
        if archive_id:
            date_folder_id = _find_or_create_folder(drive_service, output["date"], archive_id)
            _upload_or_replace(drive_service, date_folder_id, filename, json_bytes)
            log_info(f"  Drive ARCHIVE/{output['date']}/{filename} written")
    except Exception as e:
        log_warning(f"  Drive write failed (non-fatal): {e}")
        log_warning("  Sheet outputs were written successfully — Drive write skipped")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Risk Officer V1")
    parser.add_argument("--dry-run", action="store_true", help="Kein Sheet-Write")
    parser.add_argument("--date", type=str, help="Run-Datum (YYYY-MM-DD)")
    args = parser.parse_args()

    run_date = parse_date(args.date) if args.date else date.today()
    log_info(f"Risk Officer V1 starting — date: {run_date}, dry-run: {args.dry_run}")

    # ─── Google Services verbinden ────────────────────────────
    sheets = None
    drive_service = None

    if not args.dry_run:
        creds = _get_credentials()
        if not creds:
            log_error("Cannot create Google credentials — aborting")
            sys.exit(1)

        sheets = get_sheets_service(creds)
        if not sheets:
            log_error("Cannot connect to Google Sheets — aborting")
            sys.exit(1)

        drive_service = get_drive_service(creds)
        if not drive_service:
            log_warning("Cannot connect to Google Drive — will use Sheet fallback only")

    # ─── Inputs lesen ─────────────────────────────────────────
    inputs = {}

    if sheets:
        v16 = read_v16_production(sheets)
        if v16:
            inputs["v16_production"] = v16
        else:
            log_error("V16 Production unavailable — running with limited checks")

        layer = read_layer_analysis(sheets, drive_service)
        if layer:
            inputs["layer_analysis"] = layer
            log_info(f"  Layer Analysis source: {layer.get('source', 'UNKNOWN')}")
        else:
            log_warning("Layer Analysis unavailable — using defaults")
            inputs["layer_analysis"] = {
                "source": "DEFAULTS",
                "system_regime": {"regime": "UNKNOWN", "lean": "UNKNOWN"},
                "fragility_state": {"state": "UNKNOWN"}
            }

        risk_history = read_risk_history(sheets)
    else:
        log_info("Dry-run mode — using dummy inputs")
        inputs = {
            "v16_production": {
                "v16_state": "Risk-On",
                "v16_regime": "LATE_EXPANSION",
                "dd_protect_active": False,
                "dd_protect_trigger_level": -0.12,
                "current_drawdown_from_peak": -0.02,
                "weights": {
                    "HYG": 0.274, "DBC": 0.204, "GLD": 0.186,
                    "XLU": 0.182, "XLP": 0.154
                }
            },
            "layer_analysis": {
                "source": "DRY_RUN",
                "system_regime": {"regime": "SELECTIVE", "lean": "POSITIVE"},
                "fragility_state": {"state": "HEALTHY"}
            }
        }
        risk_history = None

    # ─── Engine ausfuehren ────────────────────────────────────
    output = run_risk_officer(inputs, risk_history=risk_history, run_date=run_date)

    # ─── Output ───────────────────────────────────────────────
    log_info(f"\n{'='*60}")
    log_info(f"RESULT: {output['portfolio_status']}")
    log_info(f"{'='*60}")
    log_info(f"\n{output['risk_summary']}")

    # ─── Sheet schreiben ──────────────────────────────────────
    if sheets and not args.dry_run:
        log_info("Writing to Google Sheets...")
        write_risk_alerts(sheets, output)
        append_risk_history(sheets, output)
        log_info("Sheet write complete.")

    # ─── Drive schreiben ──────────────────────────────────────
    if drive_service and not args.dry_run:
        log_info("Writing to Google Drive...")
        write_json_to_drive(drive_service, output)

    if args.dry_run:
        log_info("Dry-run — skipping all writes")
        print(json.dumps(output, indent=2, default=str))

    return output


if __name__ == "__main__":
    main()