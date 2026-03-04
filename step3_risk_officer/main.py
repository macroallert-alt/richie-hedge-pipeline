"""
step3_risk_officer/main.py
Entry Point — liest Daten aus V16 Production + DW Sheet,
fuehrt Risk Officer aus, schreibt RISK_ALERTS + RISK_HISTORY ins DW.

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
# CALC_Macro_State → Macro_State_Name enthält den State
# States 10-12 sind DD-Protect States

DD_PROTECT_STATES = {10, 11, 12}
RISK_ON_STATES = {1, 2, 3, 4, 5, 6}
RISK_OFF_STATES = {7, 8, 9}


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
        pass
    return write_sheet_range(sheets, spreadsheet_id, range_str, values)


# ─── INPUT READER ─────────────────────────────────────────────────

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


def read_layer_analysis(sheets):
    """
    Liest Market Analyst Output aus DW SCORES Tab.

    SCORES Tab hat:
      Row 1: Header-Block
      Row 2: LAYER, SCORE_RAW, SCORE_7D, ...
      Row 3+: L2 Sentiment, L3 Intelligence, ...
    """
    rows = read_sheet_range(sheets, DW_SHEET_ID, f"{TAB_SCORES}!A:F")

    # Layer Scores sammeln
    layer_scores = {}
    for row in rows:
        if not row or len(row) < 2:
            continue
        layer_name = str(row[0]).strip()
        if not layer_name.startswith("L"):
            continue
        try:
            score = float(row[1])
            layer_scores[layer_name] = score
        except (ValueError, TypeError):
            continue

    if not layer_scores:
        log_warning("SCORES tab has no valid layer scores")
        return None

    log_info(f"  Layer Analysis: {len(layer_scores)} layers loaded")

    # System Regime ableiten aus Layer Scores
    # Einfache Heuristik: Durchschnitt der verfuegbaren Scores
    avg_score = sum(layer_scores.values()) / len(layer_scores) if layer_scores else 5.0

    if avg_score >= 6.5:
        regime = "BROAD_RISK_ON"
        lean = "POSITIVE"
    elif avg_score >= 4.5:
        regime = "SELECTIVE"
        lean = "NEUTRAL"
    elif avg_score >= 3.0:
        regime = "CONFLICTED"
        lean = "NEGATIVE"
    else:
        regime = "BROAD_RISK_OFF"
        lean = "NEGATIVE"

    # Fragility ableiten
    # L5 Fragility Score: hoch = fragil
    l5 = layer_scores.get("L5 Fragility", 5.0)
    if l5 >= 8.0:
        fragility = "CRISIS"
    elif l5 >= 6.5:
        fragility = "EXTREME"
    elif l5 >= 5.0:
        fragility = "ELEVATED"
    else:
        fragility = "HEALTHY"

    return {
        "system_regime": {
            "regime": regime,
            "lean": lean
        },
        "fragility_state": {
            "state": fragility
        },
        "layer_scores": layer_scores
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
            log_warning("Layer Analysis unavailable — using defaults")
            inputs["layer_analysis"] = {
                "system_regime": {"regime": "UNKNOWN", "lean": "UNKNOWN"},
                "fragility_state": {"state": "HEALTHY"}
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
        print(json.dumps(output, indent=2, default=str))

    return output


if __name__ == "__main__":
    main()