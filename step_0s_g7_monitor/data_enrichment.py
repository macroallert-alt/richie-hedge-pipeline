"""
step_0s_g7_monitor/data_enrichment.py
Phase 1b: Data Enrichment via Brave Search + LLM

Fills gaps that APIs cannot cover:
  - SIPRI Military Spending (absolute + %GDP) -> STRUCTURAL D5
  - WIPO Patent Filings -> STRUCTURAL D3
  - IEA Energy Dependency / Renewables -> STRUCTURAL D4
  - WGC Central Bank Gold Holdings + Purchases -> FINANCIAL D7
  - Edelman Trust Barometer -> LEADING D10
  - Stock Market Cap / GDP -> FINANCIAL D8
  - FDI Inflows -> LEADING D9
  - SIA Semiconductor Revenue -> STRUCTURAL D3
  - LLM Dimension Scoring (D3, D5, D10, D11, D12) -> scoring_engine

Frequenz: Quartalsweise (check cache age, skip if <90 days old).
Pattern: Brave Search API -> collect snippets -> ONE Claude Sonnet call -> structured JSON.

KERN-ENTSCHEIDUNG: Keine manuelle Datenpflege. LLM extrahiert publizierte Zahlen
aus Brave Search Ergebnissen. Fuer die 5 LLM-Dimensions bewertet Claude qualitativ
auf 0-100 Skala basierend auf recherchierten Fakten.
"""

import os
import json
import time
import traceback
from datetime import datetime, timezone

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ENRICHMENT_CACHE_FILE = os.path.join(CACHE_DIR, "enrichment.json")
ENRICHMENT_CACHE_MAX_AGE_DAYS = 90  # Quartalsweise

# Region labels for natural-language queries
REGION_LABELS = {
    "USA": "United States",
    "CHINA": "China",
    "EU": "European Union (Germany, France, Italy, Spain, Netherlands)",
    "INDIA": "India",
    "JP_KR_TW": "Japan, South Korea, Taiwan",
    "GULF": "Saudi Arabia, UAE, Gulf states",
    "REST_EM": "Brazil, Mexico, South Africa, Indonesia, Turkey",
}


# ============================================================
# CACHE
# ============================================================

def load_enrichment_cache():
    """Load enrichment cache if fresh enough."""
    if not os.path.exists(ENRICHMENT_CACHE_FILE):
        return None
    try:
        with open(ENRICHMENT_CACHE_FILE, "r") as f:
            data = json.load(f)
        cached_date = data.get("enrichment_date", "")
        if cached_date:
            cached_dt = datetime.fromisoformat(cached_date.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - cached_dt).days
            if age_days < ENRICHMENT_CACHE_MAX_AGE_DAYS:
                print(f"  [Enrichment] Cache valid (age: {age_days}d < {ENRICHMENT_CACHE_MAX_AGE_DAYS}d)")
                return data
            else:
                print(f"  [Enrichment] Cache stale (age: {age_days}d >= {ENRICHMENT_CACHE_MAX_AGE_DAYS}d)")
                return None
        return None
    except Exception as e:
        print(f"  [Enrichment] Cache load error: {e}")
        return None


def save_enrichment_cache(data):
    """Save enrichment data to cache."""
    try:
        data["enrichment_date"] = datetime.now(timezone.utc).isoformat()
        with open(ENRICHMENT_CACHE_FILE, "w") as f:
            json.dump(data, f, default=str, indent=2)
        print("  [Enrichment] Cache saved")
    except Exception as e:
        print(f"  [Enrichment] Cache save error: {e}")


# ============================================================
# BRAVE SEARCH API
# ============================================================

def brave_search(query, api_key, count=5, timeout=15):
    """
    Search Brave Web Search API. Returns list of {title, url, snippet}.
    """
    import requests

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": count}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        return results
    except Exception as e:
        print(f"  [Brave] Query '{query}' failed: {e}")
        return []


def run_brave_searches(api_key):
    """
    Run all Brave Search queries for data enrichment.
    Returns dict of search results keyed by topic.
    """
    print("  [Enrichment] Running Brave Search queries...")

    queries = {
        # STRUCTURAL D3: Technology
        "wipo_patents": "WIPO patent filings 2024 2025 by country top filers",
        "semiconductor_revenue": "SIA semiconductor revenue 2024 2025 market share by region",
        "ai_research": "AI research papers 2024 2025 country comparison USA China",
        # STRUCTURAL D4: Energy (supplement existing)
        "energy_dependency": "IEA energy import dependency 2024 2025 by country net importer exporter",
        "renewables": "renewable electricity share 2024 2025 by country solar wind",
        # STRUCTURAL D5: Military
        "military_spending_pct": "SIPRI military expenditure 2024 2025 percent GDP by country",
        "military_spending_abs": "SIPRI military spending 2024 2025 billion dollars by country top spenders",
        "nuclear_warheads": "nuclear warheads count 2024 2025 by country",
        # FINANCIAL D7: Gold
        "cb_gold_holdings": "world gold council central bank gold reserves 2024 2025 tonnes by country",
        "cb_gold_purchases": "central bank gold purchases 2024 2025 net buyers tonnes",
        # FINANCIAL D8: Capital Markets
        "market_cap_gdp": "stock market capitalization to GDP ratio 2024 2025 by country",
        # LEADING D9: FDI
        "fdi_inflows": "UNCTAD FDI foreign direct investment inflows 2024 2025 by country billion",
        # LEADING D10: Social Cohesion
        "trust_barometer": "Edelman trust barometer 2025 2026 results by country government trust",
        "gini_coefficient": "Gini coefficient inequality 2024 2025 by country latest",
        # LEADING D11: Geopolitical
        "sanctions_overview": "active sanctions programs 2025 USA EU China Russia count",
        "alliance_strength": "NATO alliance strength 2025 military spending defense commitment",
        # General context for LLM scoring
        "geopolitical_landscape": "global geopolitical risk 2025 2026 major tensions conflicts",
    }

    all_results = {}
    for topic, query in queries.items():
        results = brave_search(query, api_key, count=5)
        all_results[topic] = results
        if results:
            print(f"    {topic}: {len(results)} results")
        else:
            print(f"    {topic}: NO RESULTS")
        time.sleep(0.3)  # Rate limit courtesy

    return all_results


# ============================================================
# LLM EXTRACTION + DIMENSION SCORING
# ============================================================

def build_extraction_prompt(search_results):
    """
    Build ONE comprehensive prompt for Claude to:
    1. Extract published data from search snippets
    2. Score LLM dimensions (D3, D5, D10, D11, D12) per region
    """

    # Compile all search snippets into context
    context_blocks = []
    for topic, results in search_results.items():
        if results:
            snippets = "\n".join(
                f"  - [{r['title']}] {r['snippet']}" for r in results[:5]
            )
            context_blocks.append(f"### {topic}\n{snippets}")

    search_context = "\n\n".join(context_blocks)

    prompt = f"""You are the G7 World Order Monitor's Data Enrichment Engine.

TASK: Extract published data from search results AND score 5 dimensions for 7 regions.

REGIONS: USA, CHINA, EU, INDIA, JP_KR_TW (Japan/Korea/Taiwan), GULF (Saudi/UAE), REST_EM (Brazil/Mexico/SA/Indonesia/Turkey)

SEARCH RESULTS:
{search_context}

RESPOND WITH THIS EXACT JSON (no markdown fences, no preamble):
{{
  "extracted_data": {{
    "sipri_military": {{
      "description": "Military spending from SIPRI or equivalent",
      "data": {{
        "USA": {{"pct_gdp": <float>, "absolute_bn_usd": <float>}},
        "CHINA": {{"pct_gdp": <float>, "absolute_bn_usd": <float>}},
        "EU": {{"pct_gdp": <float>, "absolute_bn_usd": <float>}},
        "INDIA": {{"pct_gdp": <float>, "absolute_bn_usd": <float>}},
        "JP_KR_TW": {{"pct_gdp": <float>, "absolute_bn_usd": <float>}},
        "GULF": {{"pct_gdp": <float>, "absolute_bn_usd": <float>}},
        "REST_EM": {{"pct_gdp": <float>, "absolute_bn_usd": <float>}}
      }},
      "year": "<year of data>",
      "source_note": "<brief source attribution>"
    }},
    "nuclear_warheads": {{
      "data": {{
        "USA": <int>, "CHINA": <int>, "EU": <int>, "INDIA": <int>,
        "JP_KR_TW": <int>, "GULF": <int>, "REST_EM": <int>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "wipo_patents": {{
      "description": "Patent filings from WIPO",
      "data": {{
        "USA": <int>, "CHINA": <int>, "EU": <int>, "INDIA": <int>,
        "JP_KR_TW": <int>, "GULF": <int>, "REST_EM": <int>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "semiconductor_revenue_share": {{
      "description": "Semiconductor revenue share % by region from SIA",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "energy_import_dependency": {{
      "description": "Net energy import dependency % (negative = net exporter)",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "renewable_electricity_share": {{
      "description": "Renewable share of electricity generation %",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "cb_gold_holdings_tonnes": {{
      "description": "Central bank gold holdings in tonnes",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "cb_gold_purchases_tonnes_yr": {{
      "description": "Net central bank gold purchases tonnes/year (recent year)",
      "global_total": <float>,
      "top_buyers": ["<country1>", "<country2>", "<country3>"],
      "year": "<year>",
      "source_note": "<source>"
    }},
    "market_cap_gdp_pct": {{
      "description": "Stock market capitalization as % of GDP",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "fdi_inflows_bn": {{
      "description": "FDI inflows in billion USD",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "trust_in_government_pct": {{
      "description": "Trust in government % from Edelman or equivalent",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "gini_coefficient": {{
      "description": "Gini coefficient (0-1 scale)",
      "data": {{
        "USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>,
        "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }},
    "sanctions_active_count": {{
      "description": "Approximate count of active sanctions packages imposed BY this region",
      "data": {{
        "USA": <int>, "CHINA": <int>, "EU": <int>, "INDIA": <int>,
        "JP_KR_TW": <int>, "GULF": <int>, "REST_EM": <int>
      }},
      "year": "<year>",
      "source_note": "<source>"
    }}
  }},
  "dimension_scores": {{
    "D3_technology": {{
      "USA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "CHINA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "EU": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "INDIA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "JP_KR_TW": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "GULF": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "REST_EM": {{"score": <0-100>, "rationale": "<1-2 sentences>"}}
    }},
    "D5_military": {{
      "USA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "CHINA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "EU": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "INDIA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "JP_KR_TW": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "GULF": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "REST_EM": {{"score": <0-100>, "rationale": "<1-2 sentences>"}}
    }},
    "D10_social": {{
      "USA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "CHINA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "EU": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "INDIA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "JP_KR_TW": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "GULF": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "REST_EM": {{"score": <0-100>, "rationale": "<1-2 sentences>"}}
    }},
    "D11_geopolitical": {{
      "USA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "CHINA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "EU": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "INDIA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "JP_KR_TW": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "GULF": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "REST_EM": {{"score": <0-100>, "rationale": "<1-2 sentences>"}}
    }},
    "D12_feedback": {{
      "USA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "CHINA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "EU": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "INDIA": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "JP_KR_TW": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "GULF": {{"score": <0-100>, "rationale": "<1-2 sentences>"}},
      "REST_EM": {{"score": <0-100>, "rationale": "<1-2 sentences>"}}
    }}
  }}
}}

SCORING INSTRUCTIONS FOR DIMENSION SCORES:
- Score 0-100 RELATIVE scale: best region ~90-95, worst ~10-15
- D3 Technology: AI leadership, semiconductor dominance, patent volume, R&D, VC, STEM talent
- D5 Military: Defense spending absolute + %GDP, nuclear capability, power projection, foreign bases
- D10 Social: Government trust, social cohesion, inequality (Gini), protest intensity, polarization
- D11 Geopolitical: Alliance strength, sanctions power, trade leverage, conflict proximity
- D12 Feedback: Systemic risk from debt-demographics spirals, currency-fiscal loops, tech-security dilemma
  (INVERSE: higher score = FEWER dangerous feedback loops = MORE stable)

DATA EXTRACTION RULES:
- Extract ONLY data you find in the search snippets or can confidently infer from them
- If a value is not findable, use your best estimate based on well-known published figures
- For regional aggregates (EU, JP_KR_TW, GULF, REST_EM), use reasonable weighted averages
- Always note the data year — prefer 2024 or 2025 data, fall back to 2023 if needed
- Values must be NUMBERS, not strings
"""
    return prompt


def call_llm_extraction(prompt, timeout=120):
    """Call Claude Sonnet for data extraction + dimension scoring."""
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  [Enrichment] No ANTHROPIC_API_KEY — cannot run LLM extraction")
        return None

    print("  [Enrichment] Calling Claude Sonnet for extraction + scoring...")
    start = time.time()

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 6000,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("content", [])
        text = ""
        for block in content:
            if block.get("type") == "text":
                text += block.get("text", "")

        elapsed = time.time() - start
        print(f"  [Enrichment] LLM response: {len(text)} chars, {elapsed:.1f}s")

        # Parse JSON — strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        print(f"  [Enrichment] JSON parse error: {e}")
        # Try to salvage partial JSON
        try:
            # Find the outermost braces
            start_idx = text.index("{")
            end_idx = text.rindex("}") + 1
            result = json.loads(text[start_idx:end_idx])
            print("  [Enrichment] Salvaged partial JSON")
            return result
        except Exception:
            print("  [Enrichment] Could not salvage JSON")
            return None
    except Exception as e:
        print(f"  [Enrichment] LLM call failed: {e}")
        traceback.print_exc()
        return None


# ============================================================
# FALLBACK: DETERMINISTIC ESTIMATES
# ============================================================

def build_fallback_enrichment():
    """
    Deterministic fallback when Brave Search or LLM is unavailable.
    Uses well-known published figures as estimates.
    Better than 50.0 stubs but less accurate than LLM extraction.
    """
    print("  [Enrichment] Building deterministic fallback estimates...")

    return {
        "extracted_data": {
            "sipri_military": {
                "description": "SIPRI estimates (fallback)",
                "data": {
                    "USA": {"pct_gdp": 3.4, "absolute_bn_usd": 916},
                    "CHINA": {"pct_gdp": 1.7, "absolute_bn_usd": 296},
                    "EU": {"pct_gdp": 1.9, "absolute_bn_usd": 350},
                    "INDIA": {"pct_gdp": 2.4, "absolute_bn_usd": 84},
                    "JP_KR_TW": {"pct_gdp": 1.8, "absolute_bn_usd": 120},
                    "GULF": {"pct_gdp": 6.0, "absolute_bn_usd": 100},
                    "REST_EM": {"pct_gdp": 1.5, "absolute_bn_usd": 80},
                },
                "year": "2024",
                "source_note": "Deterministic fallback from SIPRI known figures",
            },
            "nuclear_warheads": {
                "data": {"USA": 5550, "CHINA": 500, "EU": 290, "INDIA": 172,
                         "JP_KR_TW": 0, "GULF": 0, "REST_EM": 0},
                "year": "2024", "source_note": "FAS estimates (fallback)",
            },
            "wipo_patents": {
                "description": "WIPO estimates (fallback)",
                "data": {"USA": 505000, "CHINA": 1600000, "EU": 350000,
                         "INDIA": 65000, "JP_KR_TW": 620000, "GULF": 5000, "REST_EM": 50000},
                "year": "2023", "source_note": "WIPO known figures (fallback)",
            },
            "semiconductor_revenue_share": {
                "description": "SIA estimates (fallback)",
                "data": {"USA": 48.0, "CHINA": 9.0, "EU": 8.0, "INDIA": 0.5,
                         "JP_KR_TW": 28.0, "GULF": 0.0, "REST_EM": 1.5},
                "year": "2024", "source_note": "SIA known figures (fallback)",
            },
            "cb_gold_holdings_tonnes": {
                "description": "WGC estimates (fallback)",
                "data": {"USA": 8133, "CHINA": 2264, "EU": 10770, "INDIA": 854,
                         "JP_KR_TW": 846, "GULF": 430, "REST_EM": 1200},
                "year": "2024", "source_note": "WGC known figures (fallback)",
            },
            "cb_gold_purchases_tonnes_yr": {
                "global_total": 1037, "top_buyers": ["China", "Poland", "Turkey"],
                "year": "2024", "source_note": "WGC (fallback)",
            },
            "market_cap_gdp_pct": {
                "description": "Estimates (fallback)",
                "data": {"USA": 195.0, "CHINA": 65.0, "EU": 60.0, "INDIA": 115.0,
                         "JP_KR_TW": 130.0, "GULF": 80.0, "REST_EM": 35.0},
                "year": "2024", "source_note": "World Bank / WFE estimates (fallback)",
            },
            "fdi_inflows_bn": {
                "description": "UNCTAD estimates (fallback)",
                "data": {"USA": 285.0, "CHINA": 33.0, "EU": 180.0, "INDIA": 42.0,
                         "JP_KR_TW": 30.0, "GULF": 25.0, "REST_EM": 80.0},
                "year": "2023", "source_note": "UNCTAD (fallback)",
            },
            "trust_in_government_pct": {
                "description": "Edelman estimates (fallback)",
                "data": {"USA": 42.0, "CHINA": 83.0, "EU": 45.0, "INDIA": 79.0,
                         "JP_KR_TW": 40.0, "GULF": 75.0, "REST_EM": 35.0},
                "year": "2025", "source_note": "Edelman Trust Barometer (fallback)",
            },
            "gini_coefficient": {
                "description": "World Bank estimates (fallback)",
                "data": {"USA": 0.39, "CHINA": 0.38, "EU": 0.31, "INDIA": 0.35,
                         "JP_KR_TW": 0.33, "GULF": 0.45, "REST_EM": 0.48},
                "year": "2023", "source_note": "World Bank (fallback)",
            },
            "sanctions_active_count": {
                "description": "Estimates (fallback)",
                "data": {"USA": 35, "CHINA": 5, "EU": 20, "INDIA": 0,
                         "JP_KR_TW": 10, "GULF": 0, "REST_EM": 2},
                "year": "2025", "source_note": "OFAC/EU estimates (fallback)",
            },
            "energy_import_dependency": {
                "description": "IEA estimates (fallback)",
                "data": {"USA": -5.0, "CHINA": 55.0, "EU": 58.0, "INDIA": 75.0,
                         "JP_KR_TW": 88.0, "GULF": -200.0, "REST_EM": 10.0},
                "year": "2023", "source_note": "IEA (fallback)",
            },
            "renewable_electricity_share": {
                "description": "IEA/IRENA estimates (fallback)",
                "data": {"USA": 22.0, "CHINA": 32.0, "EU": 44.0, "INDIA": 20.0,
                         "JP_KR_TW": 22.0, "GULF": 3.0, "REST_EM": 25.0},
                "year": "2023", "source_note": "IEA/IRENA (fallback)",
            },
        },
        "dimension_scores": {
            "D3_technology": {
                "USA": {"score": 92, "rationale": "AI leadership, semiconductor design dominance, top VC ecosystem"},
                "CHINA": {"score": 78, "rationale": "Massive patent volume, EV/battery leadership, AI catching up"},
                "EU": {"score": 55, "rationale": "Strong automotive R&D, ASML, but weak in AI and VC"},
                "INDIA": {"score": 35, "rationale": "IT services strong, but low R&D spend, limited hardware"},
                "JP_KR_TW": {"score": 82, "rationale": "TSMC/Samsung semiconductor fab, strong patents, weak in AI"},
                "GULF": {"score": 15, "rationale": "Minimal R&D, investing in AI but from low base"},
                "REST_EM": {"score": 12, "rationale": "Limited tech ecosystem, brain drain to US/EU"},
            },
            "D5_military": {
                "USA": {"score": 95, "rationale": "Largest budget, global projection, 5500 nukes, 800+ bases"},
                "CHINA": {"score": 70, "rationale": "2nd largest budget, nuclear modernization, regional projection growing"},
                "EU": {"score": 55, "rationale": "Combined budget significant, French/UK nukes, but fragmented command"},
                "INDIA": {"score": 48, "rationale": "Nuclear power, large army, but limited projection capability"},
                "JP_KR_TW": {"score": 40, "rationale": "Advanced tech military, US alliance dependent, no nukes"},
                "GULF": {"score": 35, "rationale": "High spending per capita, advanced equipment, limited projection"},
                "REST_EM": {"score": 18, "rationale": "Minimal projection, some nuclear (Pakistan via proxy), fragmented"},
            },
            "D10_social": {
                "USA": {"score": 40, "rationale": "Deep polarization, moderate inequality, declining institutional trust"},
                "CHINA": {"score": 55, "rationale": "High gov trust but suppressed dissent, rising inequality concerns"},
                "EU": {"score": 60, "rationale": "Strong social safety nets, moderate inequality, rising populism"},
                "INDIA": {"score": 45, "rationale": "High gov trust but caste divisions, rising religious tensions"},
                "JP_KR_TW": {"score": 65, "rationale": "Low inequality, high cohesion, but demographic despair in Korea/Japan"},
                "GULF": {"score": 50, "rationale": "Stability through wealth distribution, expatriate underclass tension"},
                "REST_EM": {"score": 30, "rationale": "High inequality, political instability, protest-prone"},
            },
            "D11_geopolitical": {
                "USA": {"score": 85, "rationale": "Strongest alliance network, dominant sanctions power, global bases"},
                "CHINA": {"score": 60, "rationale": "BRI/SCO influence growing, but fewer formal alliances than US"},
                "EU": {"score": 55, "rationale": "Sanctions power via market access, NATO membership, soft power"},
                "INDIA": {"score": 50, "rationale": "Non-aligned leverage, growing Quad role, limited hard power projection"},
                "JP_KR_TW": {"score": 45, "rationale": "US alliance dependent, Taiwan exposure, tech leverage"},
                "GULF": {"score": 40, "rationale": "Energy leverage, OPEC influence, hedging between US and China"},
                "REST_EM": {"score": 20, "rationale": "Limited geopolitical leverage individually, subject to great power dynamics"},
            },
            "D12_feedback": {
                "USA": {"score": 35, "rationale": "Debt-demographics loop activating, fiscal dominance proximity rising"},
                "CHINA": {"score": 30, "rationale": "Property-demographics doom loop, capital flight risk, Thucydides trap"},
                "EU": {"score": 45, "rationale": "Energy dependency loop fading, but demographics + fiscal pressure in south"},
                "INDIA": {"score": 65, "rationale": "Fewer systemic loops, demographic dividend still positive"},
                "JP_KR_TW": {"score": 25, "rationale": "Most severe debt-demographics loop globally, BOJ trapped"},
                "GULF": {"score": 60, "rationale": "Oil transition loop long-term, but current cash buffers provide stability"},
                "REST_EM": {"score": 35, "rationale": "Currency-fiscal loops active in Turkey/Argentina, capital flight risk"},
            },
        },
        "enrichment_source": "DETERMINISTIC_FALLBACK",
        "enrichment_date": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# MAIN ENRICHMENT FUNCTION
# ============================================================

def phase1b_data_enrichment(run_type="WEEKLY", force=False):
    """
    Phase 1b: Data Enrichment via Brave Search + LLM.

    Runs if:
      - run_type == "QUARTERLY" OR
      - force == True OR
      - No valid cache exists (age > 90 days)

    Returns enrichment dict with extracted_data + dimension_scores.
    """
    print("[Phase 1b] Data Enrichment...")
    start = time.time()

    # Check if we should run
    if not force and run_type not in ("QUARTERLY", "AD_HOC"):
        cached = load_enrichment_cache()
        if cached:
            print(f"  [Enrichment] Using cache ({run_type} run, cache valid)")
            return cached

    # Check cache even for quarterly (unless forced)
    if not force:
        cached = load_enrichment_cache()
        if cached:
            print(f"  [Enrichment] Cache still valid — skipping refresh")
            return cached

    # Need fresh data — check Brave API key
    brave_key = os.environ.get("BRAVE_API_KEY", "")
    if not brave_key:
        print("  [Enrichment] No BRAVE_API_KEY — using fallback")
        fallback = build_fallback_enrichment()
        save_enrichment_cache(fallback)
        return fallback

    # Run Brave Search
    try:
        search_results = run_brave_searches(brave_key)
        total_results = sum(len(r) for r in search_results.values())
        print(f"  [Enrichment] Brave Search: {total_results} total results from {len(search_results)} queries")
    except Exception as e:
        print(f"  [Enrichment] Brave Search failed: {e}")
        traceback.print_exc()
        fallback = build_fallback_enrichment()
        save_enrichment_cache(fallback)
        return fallback

    # Build prompt and call LLM
    prompt = build_extraction_prompt(search_results)

    try:
        llm_result = call_llm_extraction(prompt)
    except Exception as e:
        print(f"  [Enrichment] LLM extraction failed: {e}")
        llm_result = None

    if llm_result and "dimension_scores" in llm_result:
        # Validate dimension scores
        valid = True
        for dim in ["D3_technology", "D5_military", "D10_social", "D11_geopolitical", "D12_feedback"]:
            dim_data = llm_result.get("dimension_scores", {}).get(dim, {})
            if not dim_data or len(dim_data) < 7:
                print(f"  [Enrichment] Missing/incomplete dimension: {dim}")
                valid = False
                break
            for region in REGIONS:
                entry = dim_data.get(region, {})
                score = entry.get("score") if isinstance(entry, dict) else None
                if score is None or not isinstance(score, (int, float)):
                    print(f"  [Enrichment] Invalid score for {dim}/{region}")
                    valid = False
                    break

        if valid:
            llm_result["enrichment_source"] = "BRAVE_SEARCH_LLM"
            llm_result["enrichment_date"] = datetime.now(timezone.utc).isoformat()
            llm_result["brave_queries_count"] = len(search_results)
            llm_result["brave_results_total"] = total_results
            save_enrichment_cache(llm_result)

            elapsed = time.time() - start
            print(f"  [Enrichment] Complete: LLM extraction successful, {elapsed:.1f}s")
            return llm_result
        else:
            print("  [Enrichment] LLM result invalid — using fallback")
    else:
        print("  [Enrichment] No valid LLM result — using fallback")

    # Fallback
    fallback = build_fallback_enrichment()
    save_enrichment_cache(fallback)
    elapsed = time.time() - start
    print(f"  [Enrichment] Complete: fallback used, {elapsed:.1f}s")
    return fallback
