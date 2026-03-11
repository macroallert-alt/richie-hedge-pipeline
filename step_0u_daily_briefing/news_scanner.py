"""
Daily Briefing System — Breaking News Scanner
Baldur Creek Capital | Step 0u
Based on: DAILY_BRIEFING_SYSTEM_SPEC_TEIL1.md §3.1.3

Scans Brave Search for high-impact news in the last 12 hours.
Each hit gets a portfolio transmission assessment.
This is a Fast-Track IC — headlines + impact, not deep analysis.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("news_scanner")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not available — news scanner will be disabled")

from .config import (
    BRAVE_API_KEY,
    BRAVE_NEWS_MAX_RESULTS,
    BRAVE_NEWS_LOOKBACK_HOURS,
    BREAKING_NEWS_KEYWORDS,
    CLAUDE_MODEL,
)


# ---------------------------------------------------------------------------
# Brave News Search API
# ---------------------------------------------------------------------------

BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"


def search_brave_news(query, max_results=5):
    """
    Search Brave News API for a query.

    Returns list of {title, url, source, published, description}.
    """
    if not HAS_REQUESTS:
        logger.warning("requests not installed — skipping Brave search")
        return []

    try:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY,
        }
        params = {
            "q": query,
            "count": max_results,
            "freshness": "pd",  # past day
        }

        resp = requests.get(BRAVE_NEWS_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("meta_url", {}).get("hostname", item.get("url", "")[:50]),
                "published": item.get("age", ""),
                "description": (item.get("description") or "")[:300],
            })

        return results

    except Exception as e:
        logger.error(f"Brave News search failed for '{query}': {e}")
        return []


# ---------------------------------------------------------------------------
# Scan all keyword categories
# ---------------------------------------------------------------------------

def scan_breaking_news():
    """
    Scan all keyword categories for breaking news.

    Returns list of {category, keyword, title, url, source, published, description}.
    """
    all_hits = []
    seen_urls = set()

    for category, keywords in BREAKING_NEWS_KEYWORDS.items():
        for keyword in keywords:
            results = search_brave_news(keyword, max_results=3)
            for r in results:
                # Deduplicate by URL
                if r["url"] in seen_urls:
                    continue
                seen_urls.add(r["url"])

                all_hits.append({
                    "category": category,
                    "keyword": keyword,
                    "title": r["title"],
                    "url": r["url"],
                    "source": r["source"],
                    "published": r["published"],
                    "description": r["description"],
                })

    logger.info(f"Breaking news scan: {len(all_hits)} unique hits across {len(BREAKING_NEWS_KEYWORDS)} categories")
    return all_hits


# ---------------------------------------------------------------------------
# Impact Assessment (deterministic first pass, LLM optional)
# ---------------------------------------------------------------------------

# Keywords that indicate HIGH impact (case-insensitive match in title)
HIGH_IMPACT_SIGNALS = [
    "emergency", "crisis", "crash", "collapse", "war", "attack",
    "invasion", "default", "bailout", "panic", "shutdown",
    "record high", "record low", "surge", "plunge", "halt",
    "intervention", "martial law", "nuclear", "sanctions",
]

MEDIUM_IMPACT_SIGNALS = [
    "concern", "warning", "risk", "tension", "escalat",
    "decline", "drop", "rise", "rally", "cut", "hike",
    "surprise", "unexpected", "volatil", "spread",
]


def assess_impact_deterministic(hit):
    """
    Quick deterministic impact assessment based on title keywords.
    Returns "HIGH", "MEDIUM", or "LOW".
    """
    title_lower = hit["title"].lower()
    desc_lower = hit.get("description", "").lower()
    combined = title_lower + " " + desc_lower

    for signal in HIGH_IMPACT_SIGNALS:
        if signal in combined:
            return "HIGH"

    for signal in MEDIUM_IMPACT_SIGNALS:
        if signal in combined:
            return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------------
# Portfolio Transmission (deterministic mapping)
# ---------------------------------------------------------------------------

# Category -> default asset impact mapping
CATEGORY_PORTFOLIO_MAP = {
    "GEOPOLITIK": {
        "DBC": {"direction": "BULLISH", "mechanism": "Geopolitical risk → commodity bid"},
        "HYG": {"direction": "BEARISH", "mechanism": "Risk-off → credit spread widening"},
        "GLD": {"direction": "BULLISH", "mechanism": "Safe haven flows"},
    },
    "ZENTRALBANKEN": {
        "HYG": {"direction": "BEARISH", "mechanism": "Rate surprise → spread repricing"},
        "GLD": {"direction": "BULLISH", "mechanism": "Monetary uncertainty → gold bid"},
        "XLU": {"direction": "BEARISH", "mechanism": "Rate-sensitive utilities impacted"},
    },
    "CREDIT_SYSTEMISCH": {
        "HYG": {"direction": "BEARISH", "mechanism": "Direct credit stress exposure"},
        "GLD": {"direction": "BULLISH", "mechanism": "Flight to safety"},
        "DBC": {"direction": "BEARISH", "mechanism": "Demand destruction risk"},
    },
    "COMMODITIES": {
        "DBC": {"direction": "BULLISH", "mechanism": "Direct commodity exposure"},
        "GLD": {"direction": "BULLISH", "mechanism": "Commodity supercycle / inflation hedge"},
        "HYG": {"direction": "NEUTRAL", "mechanism": "Indirect via inflation expectations"},
    },
    "REGULIERUNG": {
        "HYG": {"direction": "NEUTRAL", "mechanism": "Minimal direct impact"},
        "GLD": {"direction": "NEUTRAL", "mechanism": "Minimal direct impact"},
        "DBC": {"direction": "NEUTRAL", "mechanism": "Minimal direct impact"},
    },
}


def compute_portfolio_transmission(hit, v16_weights):
    """
    Map a news hit to portfolio transmission.

    Args:
        hit: news hit dict with "category"
        v16_weights: dict of {ticker: weight_decimal}

    Returns:
        dict of {asset: {direction, mechanism, exposure_pct}}
    """
    category = hit.get("category", "")
    mapping = CATEGORY_PORTFOLIO_MAP.get(category, {})

    transmission = {}
    for asset, impact in mapping.items():
        weight = v16_weights.get(asset, 0)
        if weight > 0 or impact["direction"] != "NEUTRAL":
            transmission[asset] = {
                "direction": impact["direction"],
                "mechanism": impact["mechanism"],
                "exposure_pct": round(weight * 100, 1),
            }

    return transmission


# ---------------------------------------------------------------------------
# LLM-Enhanced Assessment (optional, for HIGH impact news)
# ---------------------------------------------------------------------------

def assess_with_llm(hits_high, v16_weights, api_key=None):
    """
    Use Claude to assess HIGH impact news for detailed portfolio transmission.
    Only called for HIGH impact hits to save API costs.

    Args:
        hits_high: list of HIGH-impact news hit dicts
        v16_weights: V16 weight dict
        api_key: Anthropic API key (from env if None)

    Returns:
        list of enhanced hit dicts with LLM assessment
    """
    if not hits_high:
        return []

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("No ANTHROPIC_API_KEY — skipping LLM news assessment")
        return hits_high  # Return unenhanced

    if not HAS_REQUESTS:
        return hits_high

    # Build compact portfolio context
    top_positions = sorted(v16_weights.items(), key=lambda x: -x[1])[:5]
    portfolio_str = ", ".join(f"{t}: {w*100:.1f}%" for t, w in top_positions if w > 0)

    # Build news summary for LLM
    news_block = ""
    for i, h in enumerate(hits_high[:5]):  # Max 5 to control tokens
        news_block += f"\n{i+1}. [{h['category']}] {h['title']}\n   Quelle: {h['source']}\n   {h['description'][:200]}\n"

    prompt = f"""Du bist der Risk Analyst von Baldur Creek Capital.

Aktuelles Portfolio: {portfolio_str}

Breaking News der letzten 12 Stunden:
{news_block}

Für jede News-Meldung: Bewerte den Portfolio-Impact.
Antworte NUR als JSON-Array, keine Erklärung:
[
  {{
    "news_index": 1,
    "impact": "HIGH" oder "MEDIUM",
    "portfolio_impact": {{
      "TICKER": {{"direction": "BULLISH/BEARISH/NEUTRAL", "mechanism": "1 Satz", "severity": 1-10}}
    }},
    "one_line_summary": "1 Satz was das für das Portfolio bedeutet"
  }}
]"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Parse JSON from response
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        assessments = json.loads(text)

        # Merge LLM assessments back into hits
        for assessment in assessments:
            idx = assessment.get("news_index", 0) - 1
            if 0 <= idx < len(hits_high):
                hits_high[idx]["llm_impact"] = assessment.get("impact", "HIGH")
                hits_high[idx]["llm_portfolio_impact"] = assessment.get("portfolio_impact", {})
                hits_high[idx]["llm_summary"] = assessment.get("one_line_summary", "")

        logger.info(f"LLM news assessment: {len(assessments)} hits assessed")

    except Exception as e:
        logger.error(f"LLM news assessment failed: {e}")

    return hits_high


# ---------------------------------------------------------------------------
# MAIN: Full news scan pipeline
# ---------------------------------------------------------------------------

def run_news_scanner(v16_weights, api_key=None):
    """
    Run the full breaking news scanner pipeline.

    Args:
        v16_weights: dict of {ticker: weight_decimal} from V16
        api_key: Anthropic API key for LLM assessment (optional)

    Returns:
        dict with:
          - hits: list of all news hits with impact + transmission
          - high_impact_count: number of HIGH impact hits
          - summary: 1-line summary of breaking news situation
    """
    # 1. Scan Brave News
    raw_hits = scan_breaking_news()

    if not raw_hits:
        return {
            "hits": [],
            "high_impact_count": 0,
            "summary": "Keine High-Impact News in den letzten 12 Stunden.",
            "scan_time": datetime.now(timezone.utc).isoformat(),
        }

    # 2. Assess impact (deterministic)
    for hit in raw_hits:
        hit["impact"] = assess_impact_deterministic(hit)
        hit["portfolio_transmission"] = compute_portfolio_transmission(hit, v16_weights)

    # 3. Separate HIGH impact for LLM assessment
    high_hits = [h for h in raw_hits if h["impact"] == "HIGH"]
    medium_hits = [h for h in raw_hits if h["impact"] == "MEDIUM"]

    # 4. LLM assessment for HIGH impact only
    if high_hits:
        high_hits = assess_with_llm(high_hits, v16_weights, api_key)

    # 5. Combine and sort (HIGH first, then MEDIUM, drop LOW)
    all_hits = high_hits + medium_hits
    all_hits.sort(key=lambda h: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(h["impact"], 3))

    # 6. Generate summary
    if high_hits:
        top = high_hits[0]
        summary = top.get("llm_summary") or f"BREAKING: {top['title'][:80]}"
    elif medium_hits:
        summary = f"{len(medium_hits)} Medium-Impact Meldungen. Keine HIGH-Impact News."
    else:
        summary = "Keine relevanten Breaking News in den letzten 12 Stunden."

    return {
        "hits": all_hits[:10],  # Max 10 hits in output
        "high_impact_count": len(high_hits),
        "medium_impact_count": len(medium_hits),
        "summary": summary,
        "scan_time": datetime.now(timezone.utc).isoformat(),
    }
