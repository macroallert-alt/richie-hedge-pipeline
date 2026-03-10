#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/contrarian.py
Contrarian Signal Machine — "Hated + Tailwind" Scanner.
Spec: DISRUPTIONS_AGENT_SPEC TEIL 2 §12

Aktiv nach Sektoren suchen die der Markt aufgegeben hat,
die aber einen versteckten Disruptions-Tailwind haben.

Schritt 1: "Hated" identifizieren (Outflows, Sentiment, Short Interest, P/E)
Schritt 2: "Tailwind" pruefen (ACTIVE/WATCH Trend mit positiver Verbindung)
Schritt 3: Alert generieren mit Thesis + Historical Analogy

Alert-Level: WEAK / MODERATE / STRONG
"""

import os
import time
import requests
import json

EODHD_API_KEY = os.environ.get('EODHD_API_KEY', '')
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
BRAVE_API_KEY = os.environ.get('BRAVE_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
LLM_MODEL = 'claude-sonnet-4-6'
FMP_DELAY_S = 0.3
BRAVE_DELAY_S = 1.0


# ===== CONTRARIAN TAILWIND MAP =====
# Welche ETFs koennten von welchen Disruptions-Trends profitieren,
# obwohl sie aktuell gehasst sind?

CONTRARIAN_CANDIDATES = [
    {
        'sector': 'Nuclear Energy',
        'etf': 'URA',
        'tailwind_categories': ['D1', 'D3'],
        'tailwind_logic': 'AI-Rechenzentren brauchen massive Baseload Power. Kernkraft ist einzige skalierbare Loesung.',
    },
    {
        'sector': 'Uranium Miners',
        'etf': 'URNM',
        'tailwind_categories': ['D1', 'D3'],
        'tailwind_logic': 'Uran-Nachfrage steigt durch SMR + AI-Energiebedarf bevor Fusion reif ist.',
    },
    {
        'sector': 'Clean Energy',
        'etf': 'ICLN',
        'tailwind_categories': ['D3', 'D9'],
        'tailwind_logic': 'Regulatorischer Push + steigende fossile Kosten treiben Transition-Investitionen.',
    },
    {
        'sector': 'Solar',
        'etf': 'TAN',
        'tailwind_categories': ['D3'],
        'tailwind_logic': 'Solar-Kosten fallen weiter, Netzparitaet in immer mehr Maerkten erreicht.',
    },
    {
        'sector': 'Biotech',
        'etf': 'XBI',
        'tailwind_categories': ['D4'],
        'tailwind_logic': 'GLP-1 + Gene Editing Durchbrueche, aber Sektor durch Zinsen abgestraft.',
    },
    {
        'sector': 'Genomics',
        'etf': 'ARKG',
        'tailwind_categories': ['D4'],
        'tailwind_logic': 'Genomik-Revolution: Preisverfall bei Sequenzierung, erste klinische Erfolge.',
    },
    {
        'sector': 'Rare Earth & Critical Minerals',
        'etf': 'REMX',
        'tailwind_categories': ['D3', 'D8', 'D9'],
        'tailwind_logic': 'Reshoring + Energy Transition + Defense brauchen Critical Minerals.',
    },
    {
        'sector': 'Copper Miners',
        'etf': 'COPX',
        'tailwind_categories': ['D3', 'D8'],
        'tailwind_logic': 'Elektrifizierung + Grid-Ausbau = massiver Kupferbedarf, Angebotsdefizit.',
    },
    {
        'sector': 'Infrastructure',
        'etf': 'PAVE',
        'tailwind_categories': ['D1', 'D3', 'D8'],
        'tailwind_logic': 'AI-Rechenzentren + Reshoring + Grid-Ausbau = Infrastruktur-Boom.',
    },
    {
        'sector': 'Fintech',
        'etf': 'ARKF',
        'tailwind_categories': ['D7'],
        'tailwind_logic': 'Tokenization + CBDC + DeFi 2.0 nach dem Krypto-Winter.',
    },
    {
        'sector': 'Space Economy',
        'etf': 'UFO',
        'tailwind_categories': ['D5'],
        'tailwind_logic': 'Satelliten-Konstellationen + Space Tourism + Defense-Budgets.',
    },
    {
        'sector': 'Cybersecurity',
        'etf': 'HACK',
        'tailwind_categories': ['D1', 'D10'],
        'tailwind_logic': 'AI-getriebene Angriffe + Post-Quantum Crypto = strukturell steigende Nachfrage.',
    },
]


def run_contrarian_scan(trends, etf_universe, thresholds):
    """
    Fuehre Contrarian Scanner durch.

    Args:
        trends: Alle bewerteten Trends
        etf_universe: Thematische ETFs
        thresholds: Schwellenwerte aus Config

    Returns:
        Liste von Contrarian-Alert-Dicts
    """
    active_watch_ids = {
        t['id'] for t in trends
        if t.get('watchlist_status') in ('ACTIVE', 'WATCH')
    }

    if not active_watch_ids:
        print("    [CONTRARIAN] Keine ACTIVE/WATCH Trends")
        return []

    hated_threshold_outflow = thresholds.get('hated_etf_outflow_3m_pct', -10)
    hated_threshold_short = thresholds.get('hated_short_interest_pct', 15)
    hated_threshold_pe = thresholds.get('hated_pe_vs_5y_avg', 0.50)

    alerts = []

    for candidate in CONTRARIAN_CANDIDATES:
        etf_ticker = candidate['etf']
        tailwind_cats = candidate['tailwind_categories']

        # Pruefe ob mindestens ein Tailwind-Trend ACTIVE/WATCH ist
        active_tailwinds = [cat for cat in tailwind_cats if cat in active_watch_ids]
        if not active_tailwinds:
            continue

        # --- Schritt 1: "Hated" pruefen ---
        hated_signals = _check_hated_signals(
            etf_ticker, hated_threshold_outflow, hated_threshold_short, hated_threshold_pe
        )

        hated_count = sum(1 for v in hated_signals.values() if v.get('triggered', False))

        if hated_count == 0:
            continue  # Nicht gehasst genug

        # --- Schritt 2: Tailwind-Staerke ---
        tailwind_trends = [
            t for t in trends if t['id'] in active_tailwinds
        ]
        max_momentum = max((t.get('momentum', 0) for t in tailwind_trends), default=0)
        tailwind_source_id = active_tailwinds[0]
        tailwind_source = next((t for t in trends if t['id'] == tailwind_source_id), {})

        # Connection strength: Basierend auf Momentum des Tailwind-Trends
        connection_strength = min(1.0, max_momentum / 100)

        # --- Schritt 3: Alert-Level ---
        if hated_count >= 3 and connection_strength > 0.6:
            alert_level = 'STRONG'
        elif hated_count >= 2 and connection_strength > 0.4:
            alert_level = 'MODERATE'
        else:
            alert_level = 'WEAK'

        alert = {
            'sector': candidate['sector'],
            'etf': etf_ticker,
            'hated_signals': hated_signals,
            'hated_count': hated_count,
            'tailwind_source': {
                'category': tailwind_source_id,
                'category_name': tailwind_source.get('name', ''),
                'connection': candidate['tailwind_logic'],
                'connection_strength': round(connection_strength, 2),
                'momentum': max_momentum,
            },
            'alert_level': alert_level,
            'thesis_short': f"Markt hasst {candidate['sector']}, aber {tailwind_source.get('name', '?')} treibt Renaissance.",
        }

        alerts.append(alert)
        print(f"    [CONTRARIAN] {alert_level}: {candidate['sector']} ({etf_ticker}) — {hated_count} Hated Signals, Tailwind {tailwind_source_id}")

    # Sortiere: STRONG > MODERATE > WEAK
    level_order = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1}
    alerts.sort(key=lambda x: level_order.get(x['alert_level'], 0), reverse=True)

    return alerts


# ===== HATED SIGNALS =====

def _check_hated_signals(etf_ticker, outflow_threshold, short_threshold, pe_threshold):
    """
    Spec §12.2: Pruefe ob ein ETF "hated" ist.
    - ETF-Outflows > 10% AUM in 3 Monaten
    - Negative Media Sentiment
    - Short Interest > 15%
    - P/E < 50% des 5-Jahres-Durchschnitts
    """
    signals = {
        'etf_outflow_3m': {'triggered': False, 'value': None},
        'media_sentiment': {'triggered': False, 'value': None},
        'short_interest': {'triggered': False, 'value': None},
        'pe_vs_5y_avg': {'triggered': False, 'value': None},
    }

    # --- ETF Performance als Outflow-Proxy (FMP) ---
    if FMP_API_KEY:
        try:
            change_3m = _get_price_change_pct_fmp(etf_ticker, days=90)
            if change_3m is not None:
                signals['etf_outflow_3m']['value'] = change_3m
                if change_3m < outflow_threshold:
                    signals['etf_outflow_3m']['triggered'] = True
        except Exception as e:
            print(f"      [WARN] FMP {etf_ticker} 3M: {e}")
    elif EODHD_API_KEY:
        try:
            change_3m = _get_price_change_pct_eodhd(etf_ticker, days=90)
            if change_3m is not None:
                signals['etf_outflow_3m']['value'] = change_3m
                if change_3m < outflow_threshold:
                    signals['etf_outflow_3m']['triggered'] = True
        except Exception as e:
            print(f"      [WARN] EODHD {etf_ticker} 3M: {e}")

    # --- Media Sentiment via Brave ---
    if BRAVE_API_KEY:
        try:
            sentiment = _check_media_sentiment(etf_ticker)
            signals['media_sentiment']['value'] = sentiment
            if sentiment < 40:  # Unter 40 = negativ
                signals['media_sentiment']['triggered'] = True
        except Exception as e:
            print(f"      [WARN] Brave Sentiment {etf_ticker}: {e}")

    # --- Short Interest (FMP primary, EODHD fallback) ---
    if FMP_API_KEY:
        try:
            short_pct = _get_short_interest_fmp(etf_ticker)
            if short_pct is not None:
                signals['short_interest']['value'] = short_pct
                if short_pct > short_threshold:
                    signals['short_interest']['triggered'] = True
        except Exception:
            pass  # Short Interest oft nicht verfuegbar fuer ETFs
    elif EODHD_API_KEY:
        try:
            short_pct = _get_short_interest_eodhd(etf_ticker)
            if short_pct is not None:
                signals['short_interest']['value'] = short_pct
                if short_pct > short_threshold:
                    signals['short_interest']['triggered'] = True
        except Exception:
            pass

    # --- P/E vs 5Y Average (vereinfacht) ---
    # Fuer ETFs schwer zu berechnen — ueberspringe in V1, nutze Performance als Proxy
    # PE-basierte Bewertung kommt in Phase 2 mit detaillierten Holdings-Daten

    return signals


def _get_price_change_pct_fmp(ticker, days=90):
    """Hole Kursaenderung ueber N Tage via FMP (stable endpoint)."""
    from datetime import datetime, timedelta

    from_date = (datetime.now() - timedelta(days=days + 7)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    url = 'https://financialmodelingprep.com/stable/historical-price-eod/full'
    params = {
        'symbol': ticker,
        'apikey': FMP_API_KEY,
        'from': from_date,
        'to': to_date,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # stable endpoint returns flat array (newest first)
    if not data or not isinstance(data, list) or len(data) < 2:
        return None

    latest = float(data[0].get('close', 0))
    oldest = float(data[-1].get('close', 0))

    if oldest == 0:
        return None

    time.sleep(FMP_DELAY_S)
    return round(((latest - oldest) / oldest) * 100, 2)


def _check_media_sentiment(etf_ticker):
    """Brave Search Sentiment fuer einen ETF."""
    url = 'https://api.search.brave.com/res/v1/web/search'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': BRAVE_API_KEY
    }
    params = {
        'q': f'{etf_ticker} ETF outlook analysis',
        'count': 5,
        'freshness': 'pm'
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get('web', {}).get('results', [])

    pos_words = ['bullish', 'opportunity', 'undervalued', 'recovery', 'growth', 'buy', 'upgrade']
    neg_words = ['bearish', 'overvalued', 'decline', 'sell', 'downgrade', 'crash', 'risk', 'outflow']

    pos_count = 0
    neg_count = 0
    for r in results:
        text = f"{r.get('title', '')} {r.get('description', '')}".lower()
        pos_count += sum(1 for w in pos_words if w in text)
        neg_count += sum(1 for w in neg_words if w in text)

    total = pos_count + neg_count
    if total == 0:
        return 50

    sentiment = ((pos_count - neg_count) / total + 1) / 2 * 100
    time.sleep(BRAVE_DELAY_S)
    return round(max(0, min(100, sentiment)), 1)


def _get_price_change_pct_eodhd(ticker, days=90):
    """Hole Kursaenderung ueber N Tage via EODHD (Fallback)."""
    from datetime import datetime, timedelta

    url = f'https://eodhd.com/api/eod/{ticker}.US'
    from_date = (datetime.now() - timedelta(days=days + 7)).strftime('%Y-%m-%d')
    params = {
        'api_token': EODHD_API_KEY,
        'fmt': 'json',
        'period': 'd',
        'order': 'd',
        'from': from_date,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data or len(data) < 2:
        return None

    latest = float(data[0].get('adjusted_close', data[0].get('close', 0)))
    oldest = float(data[-1].get('adjusted_close', data[-1].get('close', 0)))

    if oldest == 0:
        return None

    return round(((latest - oldest) / oldest) * 100, 2)


def _get_short_interest_fmp(ticker):
    """Hole Short Interest via FMP (stable endpoint)."""
    url = 'https://financialmodelingprep.com/stable/key-metrics'
    params = {'symbol': ticker, 'apikey': FMP_API_KEY, 'limit': 1}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            short_pct = data[0].get('shortPercentFloat', None)
            if short_pct is not None:
                time.sleep(FMP_DELAY_S)
                return float(short_pct) * 100  # FMP returns as decimal
    except Exception:
        pass

    time.sleep(FMP_DELAY_S)
    return None


def _get_short_interest_eodhd(ticker):
    """Hole Short Interest via EODHD Fundamentals (Fallback)."""
    url = f'https://eodhd.com/api/fundamentals/{ticker}.US'
    params = {
        'api_token': EODHD_API_KEY,
        'fmt': 'json',
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data or isinstance(data, list):
        return None

    highlights = data.get('Highlights', {})
    short_pct = highlights.get('ShortPercentFloat', None)
    if short_pct is not None:
        try:
            return float(short_pct)
        except (ValueError, TypeError):
            return None
    return None
