#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/sheet_writer.py
Google Sheet Schreiblogik fuer das Disruptions Sheet (7 Tabs).
Spec: DISRUPTIONS_AGENT_SPEC TEIL 3 §15

Auth: googleapiclient.discovery (NICHT gspread), via GCP_SA_KEY/GOOGLE_CREDENTIALS.
Sheet ID: DISRUPTIONS_SHEET_ID (GitHub Secret).

Tabs:
  1. TRENDS — Eine Zeile pro Trend (Spalten A:W)
  2. ETF_MAP — Thematische ETFs mit woechentlichen Daten (A:O)
  3. SIGNALS — Screening-Rohdaten (A:I)
  4. EXPOSURE — V16 Portfolio Exposure Check (A:G)
  5. HISTORY — Woechentliche Score-Snapshots (A:H)
  6. JOURNAL — Idea Journal (A:K)
  7. CONFIG — Konfiguration key/value (A:D)
"""

import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
DISRUPTIONS_SHEET_ID = os.environ.get('DISRUPTIONS_SHEET_ID', '')


def _get_sheets_service():
    """Erstelle Google Sheets Service via Service Account."""
    sa_key = os.environ.get('GCP_SA_KEY') or os.environ.get('GOOGLE_CREDENTIALS', '')
    if not sa_key:
        raise RuntimeError("Kein GCP_SA_KEY oder GOOGLE_CREDENTIALS gefunden")

    sa_info = json.loads(sa_key)
    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
    return service.spreadsheets()


def _clear_and_write(sheets, sheet_id, tab_range, rows):
    """Loesche bestehende Daten und schreibe neue."""
    # Clear
    sheets.values().clear(
        spreadsheetId=sheet_id,
        range=tab_range,
        body={}
    ).execute()

    # Write
    if rows:
        sheets.values().update(
            spreadsheetId=sheet_id,
            range=tab_range,
            valueInputOption='RAW',
            body={'values': rows}
        ).execute()


def _append_rows(sheets, sheet_id, tab_range, rows):
    """Haenge Zeilen an (append-only Tabs wie HISTORY, JOURNAL)."""
    if rows:
        sheets.values().append(
            spreadsheetId=sheet_id,
            range=tab_range,
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body={'values': rows}
        ).execute()


def write_all_tabs(trends, screening_results, exposure_result, short_results,
                   contrarian_alerts, causal_chains, dependencies, convergence_zones,
                   etf_universe, config, run_date):
    """
    Schreibe alle 7 Tabs des Disruptions Sheets.
    """
    if not DISRUPTIONS_SHEET_ID:
        print("    [SHEET] Kein DISRUPTIONS_SHEET_ID — Sheet Write uebersprungen")
        return

    sheets = _get_sheets_service()
    sid = DISRUPTIONS_SHEET_ID

    # Tab 1: TRENDS
    _write_trends_tab(sheets, sid, trends, run_date)

    # Tab 2: ETF_MAP
    _write_etf_map_tab(sheets, sid, etf_universe, run_date)

    # Tab 3: SIGNALS
    _write_signals_tab(sheets, sid, screening_results, run_date)

    # Tab 4: EXPOSURE
    _write_exposure_tab(sheets, sid, trends, exposure_result, run_date)

    # Tab 5: HISTORY (append)
    _write_history_tab(sheets, sid, trends, run_date)

    # Tab 6: JOURNAL (append — neue Ideas)
    _write_journal_tab(sheets, sid, trends, short_results, contrarian_alerts, run_date)

    # Tab 7: CONFIG
    _write_config_tab(sheets, sid, config, run_date)


# ===== TAB 1: TRENDS =====

def _write_trends_tab(sheets, sheet_id, trends, run_date):
    """
    Spec §15.2 Tab 1: TRENDS — Spalten A:W.
    Eine Zeile pro Trend. Wird jede Woche komplett ueberschrieben.
    """
    header = [
        'date', 'category_id', 'category_name', 'watchlist_status', 'phase',
        'maturity', 'momentum', 'acceleration', 'relevance', 'hype',
        'inflection_score', 'velocity', 'crowding', 'model_risk', 'dead_zone',
        'convergence_member', 'headline', 'bull_case', 'bear_case',
        'top_etf', 'top_short', 'historical_analogy', 'last_state_change'
    ]

    rows = [header]
    for t in trends:
        rows.append([
            run_date,
            t.get('id', ''),
            t.get('name', ''),
            t.get('watchlist_status', ''),
            t.get('phase', ''),
            t.get('maturity', 0),
            t.get('momentum', 0),
            t.get('acceleration', 0),
            t.get('relevance', 0),
            t.get('hype', 0),
            t.get('inflection_score', 0),
            t.get('velocity_label', 'LOW'),
            t.get('crowding', 0),
            t.get('model_risk', 'NONE'),
            str(t.get('dead_zone', False)).upper(),
            t.get('convergence_member', ''),
            t.get('headline', ''),
            t.get('bull_case', ''),
            t.get('bear_case', ''),
            t.get('top_etf', ''),
            t.get('top_short', ''),
            t.get('historical_analogy', ''),
            t.get('last_state_change', ''),
        ])

    _clear_and_write(sheets, sheet_id, 'TRENDS!A:W', rows)
    print(f"    [SHEET] TRENDS: {len(rows) - 1} Zeilen geschrieben")


# ===== TAB 2: ETF_MAP =====

def _write_etf_map_tab(sheets, sheet_id, etf_universe, run_date):
    """
    Spec §15.2 Tab 2: ETF_MAP — Spalten A:O.
    Statische ETF-Daten + woechentlich aktualisierte Marktdaten.
    In V1: Nur statische Daten. Marktdaten werden spaeter via EODHD gefuellt.
    """
    header = [
        'ticker', 'name', 'category_ids', 'price', 'price_1w_pct',
        'price_1m_pct', 'price_3m_pct', 'aum', 'aum_1m_change_pct',
        'expense_ratio', 'pe_ratio', 'short_interest', 'volume_avg_20d',
        'top5_holdings', 'last_updated'
    ]

    rows = [header]
    for etf in etf_universe:
        rows.append([
            etf.get('ticker', ''),
            etf.get('name', ''),
            ','.join(etf.get('category_ids', [])),
            '',  # price — wird spaeter via EODHD gefuellt
            '',  # price_1w_pct
            '',  # price_1m_pct
            '',  # price_3m_pct
            '',  # aum
            '',  # aum_1m_change_pct
            '',  # expense_ratio
            '',  # pe_ratio
            '',  # short_interest
            '',  # volume_avg_20d
            '',  # top5_holdings
            run_date,
        ])

    _clear_and_write(sheets, sheet_id, 'ETF_MAP!A:O', rows)
    print(f"    [SHEET] ETF_MAP: {len(rows) - 1} ETFs geschrieben")


# ===== TAB 3: SIGNALS =====

def _write_signals_tab(sheets, sheet_id, screening_results, run_date):
    """
    Spec §15.2 Tab 3: SIGNALS — Spalten A:I.
    Screening-Rohdaten pro Kategorie.
    """
    header = [
        'date', 'category_id', 'brave_hit_count', 'brave_sentiment',
        'google_trends_value', 'google_trends_1m_change', 'etf_flow_1w',
        'screening_score', 'deep_dive_selected'
    ]

    rows = [header]
    for sr in screening_results:
        rows.append([
            run_date,
            sr.get('category_id', ''),
            sr.get('brave_hit_count', 0),
            sr.get('brave_sentiment', 50),
            sr.get('google_trends_value', 50),
            sr.get('google_trends_1m_change', 0),
            sr.get('etf_flow_1w', 50),
            sr.get('screening_score', 0),
            str(sr.get('deep_dive_selected', False)).upper(),
        ])

    _clear_and_write(sheets, sheet_id, 'SIGNALS!A:I', rows)
    print(f"    [SHEET] SIGNALS: {len(rows) - 1} Zeilen geschrieben")


# ===== TAB 4: EXPOSURE =====

def _write_exposure_tab(sheets, sheet_id, trends, exposure_result, run_date):
    """
    Spec §15.2 Tab 4: EXPOSURE — Spalten A:G.
    """
    header = [
        'date', 'category_id', 'portfolio_exposure_pct',
        'positive_exposure', 'negative_exposure', 'blind_spot', 'threat_level'
    ]

    portfolio_exposure = exposure_result.get('portfolio_exposure', {})
    blind_spots = {bs['category'] for bs in exposure_result.get('blind_spots', [])}
    threats_map = {}
    for th in exposure_result.get('threats', []):
        threats_map[th['category']] = th.get('threat_level', 'NONE')

    rows = [header]
    for t in trends:
        cat_id = t['id']
        exp = portfolio_exposure.get(cat_id, {})
        sources = exp.get('sources', [])

        positive_parts = [
            f"{s['asset']}:{s['contribution']:.3f}"
            for s in sources if s.get('contribution', 0) > 0
        ]
        negative_parts = [
            f"{s['asset']}:{s['contribution']:.3f}"
            for s in sources if s.get('contribution', 0) < 0
        ]

        rows.append([
            run_date,
            cat_id,
            round(exp.get('exposure_pct', 0), 4),
            ', '.join(positive_parts) if positive_parts else '',
            ', '.join(negative_parts) if negative_parts else '',
            str(cat_id in blind_spots).upper(),
            threats_map.get(cat_id, 'NONE'),
        ])

    _clear_and_write(sheets, sheet_id, 'EXPOSURE!A:G', rows)
    print(f"    [SHEET] EXPOSURE: {len(rows) - 1} Zeilen geschrieben")


# ===== TAB 5: HISTORY (APPEND) =====

def _write_history_tab(sheets, sheet_id, trends, run_date):
    """
    Spec §15.2 Tab 5: HISTORY — Spalten A:H.
    Append-only: Jede Woche neue Zeilen.
    """
    rows = []
    for t in trends:
        rows.append([
            run_date,
            t.get('id', ''),
            t.get('maturity', 0),
            t.get('momentum', 0),
            t.get('acceleration', 0),
            t.get('inflection_score', 0),
            t.get('phase', ''),
            t.get('watchlist_status', ''),
        ])

    if rows:
        # Pruefe ob Header existiert — wenn nicht, zuerst Header schreiben
        try:
            existing = sheets.values().get(
                spreadsheetId=sheet_id,
                range='HISTORY!A1:H1'
            ).execute()
            has_header = bool(existing.get('values', []))
        except Exception:
            has_header = False

        if not has_header:
            header = ['date', 'category_id', 'maturity', 'momentum',
                      'acceleration', 'inflection_score', 'phase', 'watchlist_status']
            rows = [header] + rows
            _clear_and_write(sheets, sheet_id, 'HISTORY!A:H', rows)
        else:
            _append_rows(sheets, sheet_id, 'HISTORY!A:H', rows)

        print(f"    [SHEET] HISTORY: {len(rows)} Zeilen appended")


# ===== TAB 6: JOURNAL (APPEND) =====

def _write_journal_tab(sheets, sheet_id, trends, short_results, contrarian_alerts, run_date):
    """
    Spec §15.2 Tab 6: JOURNAL — Spalten A:K.
    Append-only: Neue Empfehlungen als Journal-Eintraege.
    """
    rows = []

    # Long-Empfehlungen aus ACTIVE Trends
    for t in trends:
        if t.get('watchlist_status') != 'ACTIVE':
            continue
        top_etf = t.get('top_etf', '')
        if not top_etf:
            continue

        rows.append([
            run_date,                           # A: date_generated
            t['id'],                            # B: category_id
            'LONG',                             # C: recommendation_type
            top_etf,                            # D: instrument
            t.get('headline', ''),              # E: thesis
            '',                                 # F: entry_price (wird spaeter gefuellt)
            '',                                 # G: current_price
            '',                                 # H: hypothetical_pnl_pct
            0,                                  # I: weeks_since
            'OPEN',                             # J: outcome_status
            'PENDING',                          # K: operator_action
        ])

    # Short-Empfehlungen
    for s in short_results:
        if s.get('confidence') in ('HIGH', 'MEDIUM'):
            rows.append([
                run_date,
                s.get('threat_source_id', ''),
                'SHORT',
                s.get('ticker', ''),
                s.get('combined_thesis', ''),
                '', '', '', 0, 'OPEN', 'PENDING',
            ])

    # Contrarian-Empfehlungen
    for c in contrarian_alerts:
        if c.get('alert_level') in ('STRONG', 'MODERATE'):
            rows.append([
                run_date,
                c.get('tailwind_source', {}).get('category', ''),
                'CONTRARIAN',
                c.get('etf', ''),
                c.get('thesis_short', ''),
                '', '', '', 0, 'OPEN', 'PENDING',
            ])

    # Pair Trades aus Short-Ergebnissen
    for s in short_results:
        pair = s.get('pair_trade', {})
        if pair and pair.get('long') and pair.get('short'):
            rows.append([
                run_date,
                s.get('threat_source_id', ''),
                'PAIR',
                f"{pair['long']}/{pair['short']}",
                pair.get('thesis', ''),
                '', '', '', 0, 'OPEN', 'PENDING',
            ])

    if rows:
        # Pruefe ob Header existiert
        try:
            existing = sheets.values().get(
                spreadsheetId=sheet_id,
                range='JOURNAL!A1:K1'
            ).execute()
            has_header = bool(existing.get('values', []))
        except Exception:
            has_header = False

        if not has_header:
            header = ['date_generated', 'category_id', 'recommendation_type',
                      'instrument', 'thesis', 'entry_price', 'current_price',
                      'hypothetical_pnl_pct', 'weeks_since', 'outcome_status',
                      'operator_action']
            rows = [header] + rows
            _clear_and_write(sheets, sheet_id, 'JOURNAL!A:K', rows)
        else:
            _append_rows(sheets, sheet_id, 'JOURNAL!A:K', rows)

        print(f"    [SHEET] JOURNAL: {len(rows)} Eintraege")
    else:
        print(f"    [SHEET] JOURNAL: Keine neuen Eintraege")


# ===== TAB 7: CONFIG =====

def _write_config_tab(sheets, sheet_id, config, run_date):
    """
    Spec §15.2 Tab 7: CONFIG — Spalten A:D.
    Key/Value Konfiguration. Wird komplett ueberschrieben.
    """
    header = ['config_key', 'config_value', 'description', 'last_modified']

    rows = [header]

    # Kategorien
    for cat in config.get('categories', []):
        rows.append([
            f"category_{cat['id']}",
            json.dumps({
                'id': cat['id'],
                'name': cat['name'],
                'status': cat['status'],
                'keywords_brave': cat.get('keywords_brave', []),
                'keywords_trends': cat.get('keywords_trends', []),
            }, ensure_ascii=False),
            f"Kategorie {cat['id']}: {cat['name']}",
            run_date,
        ])

    # Thresholds
    for key, value in config.get('thresholds', {}).items():
        rows.append([
            f"threshold_{key}",
            str(value),
            f"Schwellenwert: {key}",
            run_date,
        ])

    # Screening Weights
    for key, value in config.get('screening_weights', {}).items():
        rows.append([
            f"screening_weight_{key}",
            str(value),
            f"Screening-Gewichtung: {key}",
            run_date,
        ])

    # Score Weights (flattened)
    for score_type, weights in config.get('score_weights', {}).items():
        for key, value in weights.items():
            rows.append([
                f"score_weight_{score_type}_{key}",
                str(value),
                f"Score-Gewichtung: {score_type} / {key}",
                run_date,
            ])

    _clear_and_write(sheets, sheet_id, 'CONFIG!A:D', rows)
    print(f"    [SHEET] CONFIG: {len(rows) - 1} Eintraege geschrieben")
