#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/screening.py
Stufe 1 — Automatisches Screening (alle Kategorien, ~5 Minuten)
Keine LLM-Nutzung — rein datengetrieben.

Datenquellen:
  - Brave Search: 2-3 Queries pro Kategorie (Hit Count + Sentiment)
  - Google Trends: 1-2 Keywords pro Kategorie (pytrends)
  - ETF Flows: AUM-Veraenderung als Flow-Proxy (EODHD)

Output: Screening-Score pro Kategorie (0-100)
"""

import os
import time
import requests
import traceback

# ===== API KEYS =====
BRAVE_API_KEY = os.environ.get('BRAVE_API_KEY', '')
EODHD_API_KEY = os.environ.get('EODHD_API_KEY', '')

# Rate limiting
BRAVE_DELAY_S = 1.0
EODHD_DELAY_S = 0.5
TRENDS_DELAY_S = 2.0


def run_screening(categories, etf_universe, screening_weights):
    """
    Fuehre Stufe 1 Screening fuer alle aktiven Kategorien durch.

    Args:
        categories: Liste aktiver Kategorie-Dicts aus Config
        etf_universe: Liste aller thematischen ETFs
        screening_weights: Gewichtung der Screening-Komponenten

    Returns:
        Liste von Screening-Result-Dicts, sortiert nach Score
    """
    weights = {
        'brave_hit_count': screening_weights.get('brave_hit_count', 0.30),
        'brave_sentiment': screening_weights.get('brave_sentiment', 0.20),
        'google_trends_value': screening_weights.get('google_trends_value', 0.25),
        'etf_flow_1w': screening_weights.get('etf_flow_1w', 0.25),
    }

    results = []

    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        print(f"  [SCREEN] {cat_id} {cat_name}...")

        # --- Brave Search ---
        brave_hits = 0
        brave_sentiment = 50  # neutral default
        try:
            brave_hits, brave_sentiment = _brave_search_category(cat)
        except Exception as e:
            print(f"    [WARN] Brave fehlgeschlagen fuer {cat_id}: {e}")

        # --- Google Trends ---
        trends_value = 50  # neutral default
        trends_1m_change = 0
        try:
            trends_value, trends_1m_change = _google_trends_category(cat)
        except Exception as e:
            print(f"    [WARN] Google Trends fehlgeschlagen fuer {cat_id}: {e}")

        # --- ETF Flows ---
        etf_flow_score = 50  # neutral default
        try:
            etf_flow_score = _etf_flow_category(cat_id, etf_universe)
        except Exception as e:
            print(f"    [WARN] ETF Flows fehlgeschlagen fuer {cat_id}: {e}")

        # --- Screening Score berechnen ---
        # Normalisiere alle Komponenten auf 0-100
        brave_hit_norm = min(100, brave_hits * 5)  # 20 hits = 100
        brave_sent_norm = brave_sentiment  # bereits 0-100
        trends_norm = trends_value  # bereits 0-100
        etf_flow_norm = etf_flow_score  # bereits 0-100

        screening_score = (
            brave_hit_norm * weights['brave_hit_count'] +
            brave_sent_norm * weights['brave_sentiment'] +
            trends_norm * weights['google_trends_value'] +
            etf_flow_norm * weights['etf_flow_1w']
        )

        result = {
            'category_id': cat_id,
            'category_name': cat_name,
            'brave_hit_count': brave_hits,
            'brave_sentiment': brave_sentiment,
            'google_trends_value': trends_value,
            'google_trends_1m_change': trends_1m_change,
            'etf_flow_1w': etf_flow_score,
            'screening_score': round(screening_score, 1),
            'deep_dive_selected': False  # wird in main.py gesetzt
        }
        results.append(result)
        print(f"    Score: {screening_score:.0f} (Brave={brave_hits}/{brave_sentiment:.0f}, Trends={trends_value:.0f}, ETF={etf_flow_score:.0f})")

    # Sortiere nach Score
    results.sort(key=lambda x: x['screening_score'], reverse=True)
    return results


# ===== BRAVE SEARCH =====

def _brave_search_category(category):
    """
    Brave Search fuer eine Kategorie.
    2-3 Queries, zaehle relevante Ergebnisse + analysiere Sentiment.

    Returns: (hit_count, sentiment_score 0-100)
    """
    if not BRAVE_API_KEY:
        print("    [SKIP] Kein BRAVE_API_KEY")
        return 0, 50

    keywords = category.get('keywords_brave', [])
    if not keywords:
        return 0, 50

    total_hits = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for query in keywords[:3]:  # max 3 Queries
        try:
            results = _brave_search(query, count=10)
            hits = len(results)
            total_hits += hits

            # Einfache Sentiment-Analyse via Keyword-Matching in Titeln/Snippets
            for r in results:
                text = f"{r.get('title', '')} {r.get('description', '')}".lower()
                pos_words = ['breakthrough', 'growth', 'surge', 'boom', 'record', 'milestone',
                             'investment', 'innovation', 'launch', 'expand', 'accelerat']
                neg_words = ['crash', 'fail', 'decline', 'cut', 'layoff', 'scandal',
                             'fraud', 'bankrupt', 'delay', 'cancel', 'stall']
                pos = sum(1 for w in pos_words if w in text)
                neg = sum(1 for w in neg_words if w in text)
                if pos > neg:
                    positive_count += 1
                elif neg > pos:
                    negative_count += 1
                else:
                    neutral_count += 1

            time.sleep(BRAVE_DELAY_S)
        except Exception as e:
            print(f"    [WARN] Brave Query '{query}' fehlgeschlagen: {e}")

    # Sentiment Score: 0 = sehr negativ, 50 = neutral, 100 = sehr positiv
    total_articles = positive_count + negative_count + neutral_count
    if total_articles > 0:
        sentiment = ((positive_count - negative_count) / total_articles + 1) / 2 * 100
        sentiment = max(0, min(100, sentiment))
    else:
        sentiment = 50

    return total_hits, round(sentiment, 1)


def _brave_search(query, count=10):
    """Einzelne Brave Search API Abfrage."""
    url = 'https://api.search.brave.com/res/v1/web/search'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': BRAVE_API_KEY
    }
    params = {
        'q': query,
        'count': count,
        'freshness': 'pw'  # past week
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = data.get('web', {}).get('results', [])
    return results


# ===== GOOGLE TRENDS =====

def _google_trends_category(category):
    """
    Google Trends Abfrage fuer eine Kategorie.
    Nutzt pytrends (kein API Key noetig, rate-limited).

    Returns: (current_value 0-100, 1m_change)
    """
    keywords = category.get('keywords_trends', [])
    if not keywords:
        return 50, 0

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("    [SKIP] pytrends nicht installiert")
        return 50, 0

    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        # Nimm erstes Keyword
        kw = keywords[0]
        pytrends.build_payload([kw], timeframe='today 3-m', geo='')
        data = pytrends.interest_over_time()

        if data.empty or kw not in data.columns:
            return 50, 0

        values = data[kw].values
        current = int(values[-1]) if len(values) > 0 else 50

        # 1-Monat Change: Vergleiche letzte Woche mit vor 4 Wochen
        if len(values) >= 5:
            recent = float(values[-1])
            month_ago = float(values[-5]) if values[-5] > 0 else 1
            change = ((recent - month_ago) / month_ago) * 100
        else:
            change = 0

        time.sleep(TRENDS_DELAY_S)
        return current, round(change, 1)

    except Exception as e:
        print(f"    [WARN] Google Trends '{keywords[0]}' fehlgeschlagen: {e}")
        return 50, 0


# ===== ETF FLOWS (EODHD) =====

def _etf_flow_category(category_id, etf_universe):
    """
    Berechne aggregierte ETF-Performance/Flows fuer eine Kategorie.
    Nutzt EODHD API fuer Preis-Daten als Flow-Proxy.

    Returns: Flow-Score 0-100 (50 = neutral, >50 = Inflows, <50 = Outflows)
    """
    if not EODHD_API_KEY:
        print("    [SKIP] Kein EODHD_API_KEY")
        return 50

    # Finde ETFs fuer diese Kategorie
    cat_etfs = [
        etf for etf in etf_universe
        if category_id in etf.get('category_ids', [])
    ]

    if not cat_etfs:
        return 50

    price_changes = []
    for etf in cat_etfs[:5]:  # max 5 ETFs pro Kategorie
        ticker = etf['ticker']
        try:
            change_1w = _get_etf_price_change(ticker, period='1w')
            if change_1w is not None:
                price_changes.append(change_1w)
            time.sleep(EODHD_DELAY_S)
        except Exception as e:
            print(f"    [WARN] EODHD {ticker} fehlgeschlagen: {e}")

    if not price_changes:
        return 50

    # Durchschnittliche 1-Wochen Kursaenderung
    avg_change = sum(price_changes) / len(price_changes)

    # Normalisiere auf 0-100: -5% = 0, 0% = 50, +5% = 100
    flow_score = max(0, min(100, 50 + avg_change * 10))

    return round(flow_score, 1)


def _get_etf_price_change(ticker, period='1w'):
    """
    Hole Kursaenderung fuer einen ETF via EODHD.

    Returns: Prozentuale Aenderung (z.B. 2.5 fuer +2.5%)
    """
    url = f'https://eodhd.com/api/eod/{ticker}.US'
    params = {
        'api_token': EODHD_API_KEY,
        'fmt': 'json',
        'period': 'd',
        'order': 'd',
        'from': '',  # letzten 10 Tage reichen
    }

    # Berechne from-Datum (10 Tage zurueck fuer 1w)
    from datetime import datetime, timedelta
    from_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    params['from'] = from_date

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data or len(data) < 2:
        return None

    # Neuester und aeltester Kurs in der Range
    latest = float(data[0].get('adjusted_close', data[0].get('close', 0)))
    oldest = float(data[-1].get('adjusted_close', data[-1].get('close', 0)))

    if oldest == 0:
        return None

    change_pct = ((latest - oldest) / oldest) * 100
    return round(change_pct, 2)
