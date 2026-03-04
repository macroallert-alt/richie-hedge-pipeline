"""
step_3_risk_officer/main.py
Entry Point — liest Daten aus Google Sheet DW, fuehrt Risk Officer aus,
schreibt RISK_ALERTS + RISK_HISTORY Tabs.

Aufruf:
  python -m step_3_risk_officer                    # Normaler Run
  python -m step_3_risk_officer --dry-run          # Kein Sheet-Write
  python -m step_3_risk_officer --date 2026-03-04  # Bestimmtes Datum
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

# ─── Tabs ─────────────────────────────────────────────────────────

TAB_V16_WEIGHTS = "PORTFOLIO_IST"
TAB_V16_STATE = "V16_STATE"
TAB_LAYER_ANALYSIS = "LAYER_ANALYSIS"
TAB_RISK_ALERTS = "RISK_ALERTS"
TAB_RISK_HISTORY = "RISK_HISTORY"


def get_sheets_service():
    """Erstellt Google Sheets API Service."""
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

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
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        os.unlink(creds_path)

        service = build("sheets", "v4", credentials=creds)
        return service.spreadsheets()

    except Exception as e:
        log_error(f"Failed to create Sheets service: {e}")
        return None


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
        pass  # Tab existiert evtl noch nicht
    return write_sheet_range(sheets, spreadsheet_id, range_str, values)


# ─── INPUT READER ─────────────────────────────────────────────────

def read_v16_production(sheets):
    """Liest V16 State + Gewichte aus V16 Production Sheet."""
    v16 = {}

    # V16 State Tab
    state_rows = read_sheet_range(sheets, V16_SHEET_ID, f"{TAB_V16_STATE}!A:B")
    state_dict = {}
    for row in state_rows:
        if len(row) >= 2:
            state_dict[row[0].strip()] = row[1].strip()

    if state_dict:
        v16["v16_state"] = state_dict.get("v16_state", "Risk-On")
        v16["v16_regime"] = state_dict.get("v16_regime", "UNKNOWN")
        v16["dd_protect_active"] = state_dict.get("dd_protect_active", "FALSE").upper() == "TRUE"
        v16["dd_protect_trigger_level"] = -0.12

        dd_str = state_dict.get("current_drawdown_from_peak", "0")
        try:
            v16["current_drawdown_from_peak"] = float(dd_str)
        except ValueError:
            v16["current_drawdown_from_peak"] = 0.0

    # Weights aus PORTFOLIO_IST
    weight_rows = read_sheet_range(sheets, V16_SHEET_ID, f"{TAB_V16_WEIGHTS}!A:B")
    weights = {}
    for row in weight_rows:
        if len(row) >= 2:
            ticker = row[0].strip()
            if ticker and ticker != "Asset" and ticker != "Ticker":
                try:
                    w = float(row[1])
                    if w > 0:
                        weights[ticker] = w
                except (ValueError, TypeError):
                    continue
    v16["weights"] = weights

    if not weights:
        log_warning("V16 weights empty — PORTFOLIO_IST may be missing or malformed")
        return None

    return v16


def read_layer_analysis(sheets):
    """Liest Market Analyst Output aus DW."""
    rows = read_sheet_range(sheets, DW_SHEET_ID, f"{TAB_LAYER_ANALYSIS}!A:B")
    la = {}
    for row in rows:
        if len(row) >= 2:
            la[row[0].strip()] = row[1].strip()

    if not la:
        log_warning("LAYER_ANALYSIS empty or missing")
        return None

    return {
        "system_regime": {
            "regime": la.get("system_regime", "UNKNOWN"),
            "lean": la.get("regime_lean", "UNKNOWN")
        },
        "fragility_state": {
            "state": la.get("fragility_state", "HEALTHY")
        }
    }


def read_risk_history(sheets):
    """Liest gestrigen Risk Officer Output fuer Trend-Erkennung."""
    rows = read_sheet_range(sheets, DW_SHEET_ID, f"{TAB_RISK_HISTORY}!A:L")
    if len(rows) < 2:
        return None  # Kein Header oder keine Daten

    header = rows[0]
    # Letzte Zeile = gestern
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


# ─── OUTPUT WRITER ────────────────────────────────────────────────

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
    # Pruefe ob Header existiert
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

    # Kompakte Alert-JSON (nur check_id + severity + days_active)
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


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Risk Officer V1")
    parser.add_argument("--dry-run", action="store_true", help="Kein Sheet-Write")
    parser.add_argument("--date", type=str, help="Run-Datum (YYYY-MM-DD)")
    args = parser.parse_args()

    run_date = parse_date(args.date) if args.date else date.today()
    log_info(f"Risk Officer V1 starting — date: {run_date}, dry-run: {args.dry_run}")

    # ─── Google Sheets verbinden ──────────────────────────────
    if not args.dry_run:
        sheets = get_sheets_service()
        if not sheets:
            log_error("Cannot connect to Google Sheets — aborting")
            sys.exit(1)
    else:
        sheets = None

    # ─── Inputs lesen ─────────────────────────────────────────
    inputs = {}

    if sheets:
        v16 = read_v16_production(sheets)
        if v16:
            inputs["v16_production"] = v16
        else:
            log_error("V16 Production unavailable — running with limited checks")

        layer = read_layer_analysis(sheets)
        if layer:
            inputs["layer_analysis"] = layer
        else:
            log_error("Layer Analysis unavailable")

        risk_history = read_risk_history(sheets)
    else:
        # Dry-run: Dummy-Daten
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
    elif args.dry_run:
        log_info("Dry-run — skipping Sheet write")
        # JSON Output fuer Debugging
        print(json.dumps(output, indent=2, default=str))

    return output


if __name__ == "__main__":
    main()