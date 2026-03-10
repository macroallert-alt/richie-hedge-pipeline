#!/usr/bin/env python3
"""
step_0t_disruptions_monitor/screening.py
Stufe 1 — Automatisches Screening (alle Kategorien, ~5 Minuten)
Keine LLM-Nutzung — rein datengetrieben.

Datenquellen:
  - Brave Search: 2-3 Queries pro Kategorie (Hit Count + Sentiment)
  - Brave Trending Proxy: Ersetzt Google Trends (Hit-Frequenz + Freshness als Trending-Signal)
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

        # --- Trending Score (via Brave als Proxy fuer Google Trends) ---
        trends_value = 50  # neutral default
        trends_1m_change = 0
        try:
            trends_value, trends_1m_change = _brave_trending_proxy(cat)
        except Exception as e:
            print(f"    [WARN] Brave Trending fehlgeschlagen fuer {cat_id}: {e}")

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


# ===== TRENDING SCORE (Brave Search als Google Trends Proxy) =====

def _brave_trending_proxy(category):
    """
    Brave Search als Proxy fuer Google Trends.
    Misst: Wie viele aktuelle Ergebnisse gibt es? Wie frisch sind sie?

    Nutzt keywords_trends aus der Config (dieselben Keywords wie frueher fuer pytrends).
    Query: "[keyword] trend 2026" mit freshness=pw (past week).

    Scoring:
      - Hit Count: Mehr Ergebnisse = mehr Interesse (0-100)
      - Freshness Bonus: Ergebnisse der letzten Woche werden hoeher gewertet
      - 1m_change wird approximiert durch Vergleich pw vs pm Ergebnisse

    Returns: (trending_score 0-100, approx_1m_change)
    """
    if not BRAVE_API_KEY:
        return 50, 0

    keywords = category.get('keywords_trends', [])
    if not keywords:
        keywords = category.get('keywords_brave', [])
    if not keywords:
        return 50, 0

    kw = keywords[0]

    # Query 1: Past week — misst aktuelles Interesse
    pw_hits = 0
    try:
        results_pw = _brave_search(f'{kw} trend', count=20)
        pw_hits = len(results_pw)
        time.sleep(BRAVE_DELAY_S)
    except Exception as e:
        print(f"    [WARN] Brave Trending '{kw}' (pw) fehlgeschlagen: {e}")

    # Query 2: Past month — fuer 1m Vergleich
    pm_hits = 0
    try:
        pm_results = _brave_search_with_freshness(f'{kw} trend', count=20, freshness='pm')
        pm_hits = len(pm_results)
        time.sleep(BRAVE_DELAY_S)
    except Exception as e:
        print(f"    [WARN] Brave Trending '{kw}' (pm) fehlgeschlagen: {e}")

    # Trending Score: 0 hits = 25 (low), 10 hits = 50 (avg), 20 hits = 100 (high)
    trending_score = min(100, max(0, 25 + pw_hits * 3.75))

    # 1m Change Approximation: pw/pm ratio
    if pm_hits > 3:
        # Erwartung: pw sollte ~25% von pm sein (1 Woche von 4)
        expected_pw = pm_hits * 0.25
        if expected_pw > 0:
            ratio = pw_hits / expected_pw
            approx_change = (ratio - 1.0) * 100  # +100% = doppelt so viel wie erwartet
            approx_change = max(-50, min(200, approx_change))
        else:
            approx_change = 0
    else:
        approx_change = 0

    return round(trending_score, 1), round(approx_change, 1)


def _brave_search_with_freshness(query, count=10, freshness='pw'):
    """Brave Search mit konfigurierbarer Freshness."""
    url = 'https://api.search.brave.com/res/v1/web/search'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': BRAVE_API_KEY
    }
    params = {
        'q': query,
        'count': count,
        'freshness': freshness,
    }
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get('web', {}).get('results', [])


# ===== ETF FLOWS (FMP primary, EODHD fallback) =====

FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_DELAY_S = 0.3


def _etf_flow_category(category_id, etf_universe):
    """
    Berechne aggregierte ETF-Performance/Flows fuer eine Kategorie.
    Nutzt FMP stable API (primary) oder EODHD (fallback) fuer Preis-Daten als Flow-Proxy.

    Returns: Flow-Score 0-100 (50 = neutral, >50 = Inflows, <50 = Outflows)
    """
    if not FMP_API_KEY and not EODHD_API_KEY:
        print("    [SKIP] Kein FMP_API_KEY oder EODHD_API_KEY")
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
        change_1w = None

        # Try FMP first
        if FMP_API_KEY and change_1w is None:
            try:
                change_1w = _get_etf_price_change_fmp(ticker)
                time.sleep(FMP_DELAY_S)
            except Exception as e:
                print(f"    [WARN] FMP {ticker} fehlgeschlagen: {e}")

        # Fallback to EODHD
        if EODHD_API_KEY and change_1w is None:
            try:
                change_1w = _get_etf_price_change_eodhd(ticker)
                time.sleep(EODHD_DELAY_S)
            except Exception as e:
                print(f"    [WARN] EODHD {ticker} fehlgeschlagen: {e}")

        if change_1w is not None:
            price_changes.append(change_1w)

    if not price_changes:
        return 50

    # Durchschnittliche 1-Wochen Kursaenderung
    avg_change = sum(price_changes) / len(price_changes)

    # Normalisiere auf 0-100: -5% = 0, 0% = 50, +5% = 100
    flow_score = max(0, min(100, 50 + avg_change * 10))

    return round(flow_score, 1)


def _get_etf_price_change_fmp(ticker):
    """Hole 1-Wochen Kursaenderung via FMP stable endpoint."""
    from datetime import datetime, timedelta

    from_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
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

    if not data or not isinstance(data, list) or len(data) < 2:
        return None

    latest = float(data[0].get('close', 0))
    oldest = float(data[-1].get('close', 0))

    if oldest == 0:
        return None

    return round(((latest - oldest) / oldest) * 100, 2)


def _get_etf_price_change_eodhd(ticker):
    """Hole 1-Wochen Kursaenderung via EODHD (fallback)."""
    from datetime import datetime, timedelta

    url = f'https://eodhd.com/api/eod/{ticker}.US'
    from_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
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
