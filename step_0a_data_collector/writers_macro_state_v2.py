"""
writers_macro_state_v2.py — Sheet Writer fuer CALC_Macro_State_V2 + DATA_K16_K17 V2
=====================================================================================
Wird von writers.py importiert. Erweitert V16SheetWriter um 2 neue Methoden.

USAGE in writers.py:
  from writers_macro_state_v2 import write_macro_state_v2, write_k16_v2
  # Dann als Methoden an V16SheetWriter binden oder direkt aufrufen.

Oder: In V16SheetWriter.write_all() nach den 3 bestehenden Tabs aufrufen.
"""

import logging
from datetime import date
from typing import Dict, Any

logger = logging.getLogger("data_collector.writers")


def write_macro_state_v2(sheet_writer, macro_row: dict, trade_date: date) -> bool:
    """
    Schreibt eine Zeile in CALC_Macro_State_V2 Tab.
    Punkt als Dezimaltrenner (NICHT deutsches Format).

    Args:
        sheet_writer: V16SheetWriter Instanz (fuer _connect und _workbook)
        macro_row: dict aus MacroStateEngine.compute()[0]
        trade_date: Handelstag
    """
    try:
        sheet_writer._connect()
        tab_name = "CALC_Macro_State_V2"

        # Tab erstellen falls nicht vorhanden
        try:
            ws = sheet_writer._workbook.worksheet(tab_name)
        except Exception:
            # Tab existiert nicht — erstellen mit Header
            ws = sheet_writer._workbook.add_worksheet(title=tab_name, rows=7000, cols=25)
            headers = [
                'Date', 'Growth_Signal', 'Growth_Detail', 'Liq_Direction', 'Liq_Detail',
                'Stress_Score', 'Stress_Detail', 'HY_Threshold', 'Macro_State_Num',
                'Macro_State_Name', 'State_Confidence', 'Prev_State', 'Transition_Dir',
                'Transition_Modifier', 'Uncertainty_Flag', 'Velocity_Score', 'Howell_Phase',
                'VIX', 'HY_Spread', 'NFCI', 'MOVE',
            ]
            ws.insert_row(headers, index=1, value_input_option='RAW')
            logger.info(f"Created tab {tab_name} with headers")

        # Check ob Datum schon existiert
        date_str = trade_date.strftime('%Y-%m-%d')
        existing = ws.col_values(1)
        if date_str in existing:
            logger.info(f"{tab_name}: {date_str} already exists, skipping")
            return True

        # Build row (Punkt als Dezimal, KEIN deutsches Format)
        row = [
            macro_row.get('Date', date_str),
            _fmt(macro_row.get('Growth_Signal', 0)),
            str(macro_row.get('Growth_Detail', '')),
            _fmt(macro_row.get('Liq_Direction', 0)),
            str(macro_row.get('Liq_Detail', '')),
            _fmt(macro_row.get('Stress_Score', 0)),
            str(macro_row.get('Stress_Detail', '')),
            _fmt_dec(macro_row.get('HY_Threshold', 7.0)),
            _fmt(macro_row.get('Macro_State_Num', 6)),
            str(macro_row.get('Macro_State_Name', 'NEUTRAL')),
            _fmt(macro_row.get('State_Confidence', 50)),
            _fmt(macro_row.get('Prev_State', 6)),
            str(macro_row.get('Transition_Dir', 'STABLE')),
            _fmt_dec(macro_row.get('Transition_Modifier', 1.00)),
            str(macro_row.get('Uncertainty_Flag', 'FALSE')),
            _fmt(macro_row.get('Velocity_Score', 0)),
            _fmt(macro_row.get('Howell_Phase', 2)),
            _fmt_dec(macro_row.get('VIX', '')),
            _fmt_dec(macro_row.get('HY_Spread', '')),
            _fmt_dec(macro_row.get('NFCI', '')),
            _fmt_dec(macro_row.get('MOVE', '')),
        ]

        # Insert at row 2 (nach Header, neuestes oben)
        ws.insert_row(row, index=2, value_input_option='RAW')
        state = macro_row.get('Macro_State_Num', '?')
        name = macro_row.get('Macro_State_Name', '?')
        logger.info(f"{tab_name}: wrote {date_str} — State {state} ({name})")
        return True

    except Exception as e:
        logger.error(f"CALC_Macro_State_V2 write failed: {e}")
        return False


def write_k16_v2(sheet_writer, k16_row: dict, trade_date: date) -> bool:
    """
    Schreibt erweiterte K16_K17 Daten.
    Aktualisiert die bestehende DATA_K16_K17 Zeile mit allen neuen Spalten.
    Punkt als Dezimaltrenner.

    Args:
        sheet_writer: V16SheetWriter Instanz
        k16_row: dict aus MacroStateEngine.compute()[1]
        trade_date: Handelstag
    """
    try:
        sheet_writer._connect()
        ws = sheet_writer._workbook.worksheet("DATA_K16_K17")

        date_str = trade_date.strftime('%Y-%m-%d')

        # Spalten-Reihenfolge gemaess Spec 10.2
        column_order = [
            'Date', 'Cu_Au_Ratio', 'Cu_Au_Mom6M', 'K16_Vote',
            'HY_OAS_Mom6M', 'K17_Vote', 'GLP_Acceleration', 'K4_Vote',
            'Howell_Phase', 'Howell_Mom6M', 'Howell_Vote',
            'YC_Spread', 'YC_Mom3M', 'K5_Vote',
            'Vote_Sum', 'Liq_Dir_Raw', 'Liq_Dir_Final', 'Liq_Dir_Confirmed',
            'Vote_Sum_Magnitude',
        ]

        # Prüfe ob Zeile schon existiert
        existing = ws.col_values(1)
        if date_str in existing:
            # Update bestehende Zeile
            row_idx = existing.index(date_str) + 1  # gspread ist 1-indexed
            row = [_fmt_dec(k16_row.get(col, '')) for col in column_order]
            row[0] = date_str  # Date bleibt String
            # Update range A:S fuer diese Zeile
            cell_range = f"A{row_idx}:S{row_idx}"
            ws.update(cell_range, [row], value_input_option='RAW')
            logger.info(f"DATA_K16_K17: updated {date_str} with V2 columns")
        else:
            # Neue Zeile einfügen
            row = [date_str]
            for col in column_order[1:]:
                row.append(_fmt_dec(k16_row.get(col, '')))
            ws.insert_row(row, index=3, value_input_option='RAW')
            logger.info(f"DATA_K16_K17: wrote {date_str} with V2 columns")

        return True

    except Exception as e:
        logger.error(f"DATA_K16_K17 V2 write failed: {e}")
        return False


def _fmt(val) -> str:
    """Formatiert int/float ohne Dezimalstellen."""
    if val is None or val == '':
        return ''
    try:
        v = int(float(val))
        return str(v)
    except (ValueError, TypeError):
        return str(val)


def _fmt_dec(val) -> str:
    """Formatiert float mit Punkt-Dezimaltrenner."""
    if val is None or val == '':
        return ''
    try:
        v = float(val)
        if v == int(v) and abs(v) < 1000:
            return str(int(v))
        return str(round(v, 6))
    except (ValueError, TypeError):
        return str(val)
