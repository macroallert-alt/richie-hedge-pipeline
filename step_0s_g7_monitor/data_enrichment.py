"""
step_0s_g7_monitor/data_enrichment.py
Phase 1b: Data Enrichment via Brave Search + LLM

VOLLSTAENDIG — deckt ALLE leeren Zellen in den Layout-Tabs ab.
42 Brave Queries -> Claude Sonnet -> ~45 Datenzeilen + 5 Dimension Scores.

Frequenz: Quartalsweise (cache 90 days).
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
ENRICHMENT_CACHE_MAX_AGE_DAYS = 90


# ============================================================
# CACHE
# ============================================================

def load_enrichment_cache():
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
            print(f"  [Enrichment] Cache stale (age: {age_days}d)")
        return None
    except Exception as e:
        print(f"  [Enrichment] Cache load error: {e}")
        return None


def save_enrichment_cache(data):
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
    import requests
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip",
               "X-Subscription-Token": api_key}
    params = {"q": query, "count": count}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [{"title": item.get("title", ""), "url": item.get("url", ""),
                 "snippet": item.get("description", "")}
                for item in data.get("web", {}).get("results", [])]
    except Exception as e:
        print(f"  [Brave] Query '{query}' failed: {e}")
        return []


def run_brave_searches(api_key):
    print("  [Enrichment] Running Brave Search queries...")
    queries = {
        "gdp_per_capita_ppp": "IMF GDP per capita PPP 2024 2025 by country dollars",
        "labor_productivity": "OECD labor productivity growth 2024 2025 by country",
        "manufacturing_gdp": "manufacturing share of GDP 2024 2025 by country",
        "working_age_pop": "working age population growth rate 2024 2025 by country UN",
        "median_age": "median age by country 2024 2025 UN population",
        "net_migration": "net migration rate 2024 2025 per 1000 by country",
        "youth_unemployment": "youth unemployment rate 2024 2025 by country ILO",
        "rd_spend_gdp": "R&D spending percent GDP 2024 2025 by country OECD",
        "wipo_patents": "WIPO patent filings 2024 2025 by country top filers",
        "ai_papers": "AI research papers 2024 2025 by country count USA China",
        "semiconductor_revenue": "SIA semiconductor revenue 2024 2025 market share by region",
        "vc_deep_tech": "venture capital deep tech investment 2024 2025 by country",
        "top_tech_companies": "Forbes top 100 technology companies headquarters country 2024",
        "stem_graduates": "STEM graduates per year 2024 by country thousands",
        "energy_dependency": "IEA energy import dependency 2024 2025 by country",
        "renewables": "renewable electricity share 2024 2025 by country",
        "critical_minerals": "critical mineral processing share by country 2024 China",
        "strategic_petroleum": "strategic petroleum reserve days import by country 2024",
        "lng_export": "LNG export capacity 2024 2025 by country billion cubic meters",
        "military_spending_abs": "SIPRI military spending 2024 2025 billion dollars top spenders",
        "nuclear_warheads": "nuclear warheads count 2024 2025 by country FAS SIPRI",
        "aircraft_carriers": "aircraft carriers by country 2024 2025 active navy",
        "military_bases": "foreign military bases by country 2024 2025 overseas",
        "cb_balance_gdp": "central bank balance sheet percent GDP 2024 Fed ECB BOJ PBOC",
        "cb_gold_holdings": "world gold council central bank gold reserves 2024 2025 tonnes",
        "cb_gold_purchases": "central bank gold purchases 2024 2025 net buyers tonnes",
        "swift_payment": "SWIFT payment share USD EUR RMB 2024 2025 percentage",
        "currency_vs_usd": "currency performance vs US dollar 5 year change 2020 2025",
        "market_cap_gdp": "stock market capitalization to GDP ratio 2024 2025 by country",
        "bond_market_gdp": "bond market size GDP ratio 2024 2025 by country BIS",
        "property_rights": "Heritage Foundation property rights index 2024 2025 by country",
        "capital_controls": "IMF capital controls index 2024 severity by country",
        "rule_of_law": "World Justice Project rule of law index 2024 2025 score",
        "fdi_inflows": "UNCTAD FDI inflows 2024 2025 billion by country",
        "fdi_gdp": "FDI inflows percent GDP 2024 2025 by country UNCTAD",
        "treasury_holdings": "foreign holdings US Treasury securities 2024 2025 by country TIC",
        "trust_barometer": "Edelman trust barometer 2025 2026 government trust by country",
        "gini_coefficient": "Gini coefficient inequality 2024 2025 by country",
        "polarization": "political polarization index 2024 2025 by country V-Dem",
        "social_mobility": "WEF social mobility index 2024 2025 ranking by country",
        "sanctions_overview": "active sanctions programs 2025 USA EU count OFAC",
        "trade_gdp_ratio": "trade to GDP ratio 2024 2025 by country World Bank",
        "reshoring_index": "Kearney reshoring index 2024 2025 nearshoring",
        "geopolitical_landscape": "global geopolitical risk tensions conflicts 2025 2026",
    }
    all_results = {}
    for topic, query in queries.items():
        results = brave_search(query, api_key, count=5)
        all_results[topic] = results
        print(f"    {topic}: {len(results)} results" if results else f"    {topic}: NO RESULTS")
        time.sleep(0.25)
    return all_results


# ============================================================
# LLM EXTRACTION PROMPT — COMPLETE SCHEMA
# ============================================================

def build_extraction_prompt(search_results):
    context_blocks = []
    for topic, results in search_results.items():
        if results:
            snippets = "\n".join(f"  - [{r['title']}] {r['snippet']}" for r in results[:5])
            context_blocks.append(f"### {topic}\n{snippets}")
    search_context = "\n\n".join(context_blocks)

    # Build the full JSON schema as a string to keep the prompt clean
    r7 = '"USA": <float>, "CHINA": <float>, "EU": <float>, "INDIA": <float>, "JP_KR_TW": <float>, "GULF": <float>, "REST_EM": <float>'
    r7i = '"USA": <int>, "CHINA": <int>, "EU": <int>, "INDIA": <int>, "JP_KR_TW": <int>, "GULF": <int>, "REST_EM": <int>'
    mil = '"USA": {"pct_gdp": <float>, "absolute_bn_usd": <float>}, "CHINA": {"pct_gdp": <float>, "absolute_bn_usd": <float>}, "EU": {"pct_gdp": <float>, "absolute_bn_usd": <float>}, "INDIA": {"pct_gdp": <float>, "absolute_bn_usd": <float>}, "JP_KR_TW": {"pct_gdp": <float>, "absolute_bn_usd": <float>}, "GULF": {"pct_gdp": <float>, "absolute_bn_usd": <float>}, "REST_EM": {"pct_gdp": <float>, "absolute_bn_usd": <float>}'

    def _dim_block():
        return ', '.join(f'"{r}": {{"score": <0-100>, "rationale": "<1-2 sentences>"}}' for r in REGIONS)

    prompt = f"""You are the G7 World Order Monitor's Data Enrichment Engine.

TASK: Extract ALL published data from search results for 7 regions AND score 5 dimensions.
REGIONS: USA, CHINA, EU (DE/FR/IT/ES/NL avg), INDIA, JP_KR_TW (JP/KR/TW avg), GULF (SA/AE avg), REST_EM (BR/MX/ZA/ID/TR avg)

SEARCH RESULTS:
{search_context}

RESPOND WITH EXACT JSON. No markdown fences. No preamble. No comments. Every data block needs all 7 regions. Use search snippets + well-known published figures.

{{
  "extracted_data": {{
    "gdp_per_capita_ppp": {{"data": {{{r7}}}, "unit": "USD", "year": "<yr>", "source_note": "<src>"}},
    "labor_productivity_growth": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "manufacturing_gdp_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "working_age_pop_growth": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "median_age": {{"data": {{{r7}}}, "unit": "years", "year": "<yr>", "source_note": "<src>"}},
    "net_migration_rate": {{"data": {{{r7}}}, "unit": "per 1000", "year": "<yr>", "source_note": "<src>"}},
    "youth_unemployment": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "rd_spend_gdp_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "wipo_patents": {{"data": {{{r7i}}}, "unit": "filings", "year": "<yr>", "source_note": "<src>"}},
    "ai_papers_published": {{"data": {{{r7i}}}, "unit": "papers", "year": "<yr>", "source_note": "<src>"}},
    "semiconductor_revenue_share": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "vc_deep_tech_bn": {{"data": {{{r7}}}, "unit": "USD bn", "year": "<yr>", "source_note": "<src>"}},
    "top_100_tech_hq_count": {{"data": {{{r7i}}}, "unit": "companies", "year": "<yr>", "source_note": "<src>"}},
    "stem_graduates_thousands": {{"data": {{{r7}}}, "unit": "thousands", "year": "<yr>", "source_note": "<src>"}},
    "energy_import_dependency": {{"data": {{{r7}}}, "unit": "% neg=exporter", "year": "<yr>", "source_note": "<src>"}},
    "renewable_electricity_share": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "critical_mineral_processing_share": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "strategic_petroleum_reserves_days": {{"data": {{{r7}}}, "unit": "days", "year": "<yr>", "source_note": "<src>"}},
    "lng_export_capacity_bcm": {{"data": {{{r7}}}, "unit": "bcm/yr", "year": "<yr>", "source_note": "<src>"}},
    "sipri_military": {{"data": {{{mil}}}, "year": "<yr>", "source_note": "<src>"}},
    "nuclear_warheads": {{"data": {{{r7i}}}, "year": "<yr>", "source_note": "<src>"}},
    "aircraft_carriers": {{"data": {{{r7i}}}, "year": "<yr>", "source_note": "<src>"}},
    "foreign_military_bases": {{"data": {{{r7i}}}, "year": "<yr>", "source_note": "<src>"}},
    "cb_balance_sheet_gdp_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "cb_gold_holdings_tonnes": {{"data": {{{r7}}}, "year": "<yr>", "source_note": "<src>"}},
    "cb_gold_purchases_tonnes_yr": {{"global_total": <float>, "top_buyers": ["<c1>","<c2>","<c3>"], "year": "<yr>", "source_note": "<src>"}},
    "swift_payment_share_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "currency_vs_usd_5y_pct": {{"data": {{{r7}}}, "unit": "% chg", "year": "2020-2025", "source_note": "<src>"}},
    "market_cap_gdp_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "bond_market_gdp_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "property_rights_score": {{"data": {{{r7}}}, "unit": "0-100", "year": "<yr>", "source_note": "<src>"}},
    "capital_controls_severity": {{"data": {{{r7}}}, "unit": "0-10", "year": "<yr>", "source_note": "<src>"}},
    "rule_of_law_score": {{"data": {{{r7}}}, "unit": "0-1", "year": "<yr>", "source_note": "<src>"}},
    "fdi_inflows_bn": {{"data": {{{r7}}}, "unit": "USD bn", "year": "<yr>", "source_note": "<src>"}},
    "fdi_gdp_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "treasury_holdings_bn": {{"data": {{{r7}}}, "unit": "USD bn", "year": "<yr>", "source_note": "<src>"}},
    "trust_in_government_pct": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "gini_coefficient": {{"data": {{{r7}}}, "unit": "0-1", "year": "<yr>", "source_note": "<src>"}},
    "political_polarization_score": {{"data": {{{r7}}}, "unit": "0-10", "year": "<yr>", "source_note": "<src>"}},
    "social_mobility_rank": {{"data": {{{r7i}}}, "unit": "rank", "year": "<yr>", "source_note": "<src>"}},
    "sanctions_active_count": {{"data": {{{r7i}}}, "year": "<yr>", "source_note": "<src>"}},
    "alliance_strength_score": {{"data": {{{r7}}}, "unit": "0-10", "year": "<yr>", "source_note": "LLM assessment"}},
    "trade_gdp_ratio": {{"data": {{{r7}}}, "unit": "%", "year": "<yr>", "source_note": "<src>"}},
    "conflict_proximity_score": {{"data": {{{r7}}}, "unit": "0-10", "year": "<yr>", "source_note": "LLM assessment"}}
  }},
  "dimension_scores": {{
    "D3_technology": {{{_dim_block()}}},
    "D5_military": {{{_dim_block()}}},
    "D10_social": {{{_dim_block()}}},
    "D11_geopolitical": {{{_dim_block()}}},
    "D12_feedback": {{{_dim_block()}}}
  }}
}}

SCORING: 0-100 relative. Best ~90-95, worst ~10-15.
D3=AI/semis/patents/R&D/VC/STEM. D5=spending/nukes/carriers/bases/projection.
D10=trust/cohesion/Gini/protests/polarization. D11=alliances/sanctions/trade/conflict.
D12=systemic loops INVERSE (high=stable, low=dangerous loops).
ALL values NUMBERS. No null. No strings for numeric fields."""
    return prompt


# ============================================================
# LLM CALL
# ============================================================

def call_llm_extraction(prompt, timeout=180):
    import requests
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  [Enrichment] No ANTHROPIC_API_KEY")
        return None
    print("  [Enrichment] Calling Claude Sonnet for extraction + scoring...")
    start = time.time()
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 8000,
                  "temperature": 0.2,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        elapsed = time.time() - start
        print(f"  [Enrichment] LLM response: {len(text)} chars, {elapsed:.1f}s")
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [Enrichment] JSON parse error: {e}")
        try:
            si = text.index("{"); ei = text.rindex("}") + 1
            result = json.loads(text[si:ei])
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
# FALLBACK
# ============================================================

def build_fallback_enrichment():
    print("  [Enrichment] Building deterministic fallback estimates...")
    def _d(usa, chn, eu, ind, jpk, gul, rem):
        return {"USA": usa, "CHINA": chn, "EU": eu, "INDIA": ind,
                "JP_KR_TW": jpk, "GULF": gul, "REST_EM": rem}
    return {
        "extracted_data": {
            "gdp_per_capita_ppp":          {"data": _d(85000, 23000, 52000, 9500, 48000, 55000, 14000), "unit": "USD", "year": "2024", "source_note": "IMF (fallback)"},
            "labor_productivity_growth":   {"data": _d(1.5, 3.8, 0.8, 4.5, 0.6, 1.2, 1.8), "unit": "%", "year": "2024", "source_note": "OECD (fallback)"},
            "manufacturing_gdp_pct":       {"data": _d(11.0, 28.0, 15.0, 14.0, 20.0, 12.0, 16.0), "unit": "%", "year": "2023", "source_note": "World Bank (fallback)"},
            "working_age_pop_growth":      {"data": _d(0.3, -0.4, -0.3, 1.0, -0.5, 2.5, 0.8), "unit": "%", "year": "2024", "source_note": "UN (fallback)"},
            "median_age":                  {"data": _d(38.5, 39.0, 44.0, 28.7, 48.5, 32.0, 30.5), "unit": "years", "year": "2024", "source_note": "UN (fallback)"},
            "net_migration_rate":          {"data": _d(3.0, -0.2, 2.0, -0.4, 0.5, 12.0, -0.5), "unit": "per 1000", "year": "2024", "source_note": "UN (fallback)"},
            "youth_unemployment":          {"data": _d(8.5, 16.0, 14.5, 23.0, 8.0, 25.0, 22.0), "unit": "%", "year": "2024", "source_note": "ILO (fallback)"},
            "rd_spend_gdp_pct":            {"data": _d(3.5, 2.6, 2.2, 0.7, 3.4, 0.6, 0.8), "unit": "%", "year": "2023", "source_note": "OECD (fallback)"},
            "wipo_patents":                {"data": _d(505000, 1600000, 350000, 65000, 620000, 5000, 50000), "unit": "filings", "year": "2023", "source_note": "WIPO (fallback)"},
            "ai_papers_published":         {"data": _d(45000, 55000, 28000, 18000, 15000, 500, 5000), "unit": "papers", "year": "2024", "source_note": "Estimate (fallback)"},
            "semiconductor_revenue_share": {"data": _d(48.0, 9.0, 8.0, 0.5, 28.0, 0.0, 1.5), "unit": "%", "year": "2024", "source_note": "SIA (fallback)"},
            "vc_deep_tech_bn":             {"data": _d(120.0, 35.0, 25.0, 12.0, 8.0, 3.0, 5.0), "unit": "USD bn", "year": "2024", "source_note": "Crunchbase (fallback)"},
            "top_100_tech_hq_count":       {"data": _d(55, 18, 12, 5, 8, 0, 2), "unit": "companies", "year": "2024", "source_note": "Forbes (fallback)"},
            "stem_graduates_thousands":    {"data": _d(800, 4700, 1200, 2600, 850, 120, 1500), "unit": "thousands", "year": "2023", "source_note": "UNESCO (fallback)"},
            "energy_import_dependency":    {"data": _d(-5.0, 55.0, 58.0, 75.0, 88.0, -200.0, 10.0), "unit": "%", "year": "2023", "source_note": "IEA (fallback)"},
            "renewable_electricity_share": {"data": _d(22.0, 32.0, 44.0, 20.0, 22.0, 3.0, 25.0), "unit": "%", "year": "2023", "source_note": "IEA (fallback)"},
            "critical_mineral_processing_share": {"data": _d(5.0, 65.0, 3.0, 5.0, 8.0, 0.0, 8.0), "unit": "%", "year": "2023", "source_note": "IEA (fallback)"},
            "strategic_petroleum_reserves_days": {"data": _d(40.0, 80.0, 90.0, 12.0, 150.0, 0.0, 20.0), "unit": "days", "year": "2024", "source_note": "IEA (fallback)"},
            "lng_export_capacity_bcm":     {"data": _d(120.0, 0.0, 0.0, 0.0, 0.0, 75.0, 15.0), "unit": "bcm", "year": "2024", "source_note": "IGU (fallback)"},
            "sipri_military":              {"data": {"USA": {"pct_gdp": 3.4, "absolute_bn_usd": 916}, "CHINA": {"pct_gdp": 1.7, "absolute_bn_usd": 296}, "EU": {"pct_gdp": 1.9, "absolute_bn_usd": 350}, "INDIA": {"pct_gdp": 2.4, "absolute_bn_usd": 84}, "JP_KR_TW": {"pct_gdp": 1.8, "absolute_bn_usd": 120}, "GULF": {"pct_gdp": 6.0, "absolute_bn_usd": 100}, "REST_EM": {"pct_gdp": 1.5, "absolute_bn_usd": 80}}, "year": "2024", "source_note": "SIPRI (fallback)"},
            "nuclear_warheads":            {"data": _d(5550, 500, 290, 172, 0, 0, 0), "year": "2024", "source_note": "FAS (fallback)"},
            "aircraft_carriers":           {"data": _d(11, 3, 4, 2, 4, 0, 1), "year": "2024", "source_note": "IISS (fallback)"},
            "foreign_military_bases":      {"data": _d(750, 5, 20, 3, 2, 1, 0), "year": "2024", "source_note": "Various (fallback)"},
            "cb_balance_sheet_gdp_pct":    {"data": _d(25.0, 35.0, 45.0, 18.0, 130.0, 15.0, 20.0), "unit": "%", "year": "2024", "source_note": "BIS (fallback)"},
            "cb_gold_holdings_tonnes":     {"data": _d(8133, 2264, 10770, 854, 846, 430, 1200), "year": "2024", "source_note": "WGC (fallback)"},
            "cb_gold_purchases_tonnes_yr": {"global_total": 1037, "top_buyers": ["China", "Poland", "Turkey"], "year": "2024", "source_note": "WGC (fallback)"},
            "swift_payment_share_pct":     {"data": _d(47.0, 4.7, 32.0, 0.2, 3.0, 0.5, 2.0), "unit": "%", "year": "2024", "source_note": "SWIFT (fallback)"},
            "currency_vs_usd_5y_pct":      {"data": _d(0.0, -8.0, -5.0, -12.0, -25.0, 0.0, -20.0), "unit": "%", "year": "2020-2025", "source_note": "Estimate (fallback)"},
            "market_cap_gdp_pct":          {"data": _d(195.0, 65.0, 60.0, 115.0, 130.0, 80.0, 35.0), "unit": "%", "year": "2024", "source_note": "WFE (fallback)"},
            "bond_market_gdp_pct":         {"data": _d(200.0, 115.0, 150.0, 30.0, 250.0, 25.0, 45.0), "unit": "%", "year": "2024", "source_note": "BIS (fallback)"},
            "property_rights_score":       {"data": _d(78.0, 45.0, 75.0, 50.0, 72.0, 55.0, 42.0), "unit": "0-100", "year": "2024", "source_note": "Heritage (fallback)"},
            "capital_controls_severity":   {"data": _d(1.0, 8.0, 1.5, 5.0, 2.0, 3.0, 4.5), "unit": "0-10", "year": "2024", "source_note": "IMF (fallback)"},
            "rule_of_law_score":           {"data": _d(0.71, 0.47, 0.72, 0.49, 0.78, 0.52, 0.42), "unit": "0-1", "year": "2024", "source_note": "WJP (fallback)"},
            "fdi_inflows_bn":              {"data": _d(285.0, 33.0, 180.0, 42.0, 30.0, 25.0, 80.0), "unit": "USD bn", "year": "2023", "source_note": "UNCTAD (fallback)"},
            "fdi_gdp_pct":                 {"data": _d(1.1, 0.2, 1.2, 1.3, 0.6, 1.5, 1.8), "unit": "%", "year": "2023", "source_note": "UNCTAD (fallback)"},
            "treasury_holdings_bn":        {"data": _d(0.0, 775.0, 1200.0, 230.0, 1350.0, 250.0, 400.0), "unit": "USD bn", "year": "2024", "source_note": "TIC (fallback)"},
            "trust_in_government_pct":     {"data": _d(42.0, 83.0, 45.0, 79.0, 40.0, 75.0, 35.0), "unit": "%", "year": "2025", "source_note": "Edelman (fallback)"},
            "gini_coefficient":            {"data": _d(0.39, 0.38, 0.31, 0.35, 0.33, 0.45, 0.48), "unit": "0-1", "year": "2023", "source_note": "World Bank (fallback)"},
            "political_polarization_score": {"data": _d(8.5, 3.0, 5.5, 6.0, 4.0, 2.5, 6.5), "unit": "0-10", "year": "2024", "source_note": "V-Dem (fallback)"},
            "social_mobility_rank":        {"data": _d(27, 45, 12, 76, 15, 50, 60), "unit": "rank", "year": "2024", "source_note": "WEF (fallback)"},
            "sanctions_active_count":      {"data": _d(35, 5, 20, 0, 10, 0, 2), "year": "2025", "source_note": "OFAC/EU (fallback)"},
            "alliance_strength_score":     {"data": _d(9.5, 5.0, 7.5, 5.5, 7.0, 4.5, 2.5), "unit": "0-10", "year": "2025", "source_note": "LLM (fallback)"},
            "trade_gdp_ratio":             {"data": _d(25.0, 37.0, 85.0, 42.0, 65.0, 75.0, 50.0), "unit": "%", "year": "2023", "source_note": "World Bank (fallback)"},
            "conflict_proximity_score":    {"data": _d(3.0, 6.0, 4.0, 5.0, 7.0, 8.0, 4.5), "unit": "0-10", "year": "2025", "source_note": "LLM (fallback)"},
        },
        "dimension_scores": {
            "D3_technology":    {"USA": {"score": 92, "rationale": "AI+semi design dominance"}, "CHINA": {"score": 78, "rationale": "Patent volume, EV/battery"}, "EU": {"score": 55, "rationale": "ASML, auto R&D, weak AI"}, "INDIA": {"score": 35, "rationale": "IT services, low R&D"}, "JP_KR_TW": {"score": 82, "rationale": "TSMC/Samsung fab"}, "GULF": {"score": 15, "rationale": "Minimal R&D"}, "REST_EM": {"score": 12, "rationale": "Limited tech"}},
            "D5_military":      {"USA": {"score": 95, "rationale": "Largest budget, global projection"}, "CHINA": {"score": 70, "rationale": "2nd budget, modernizing"}, "EU": {"score": 55, "rationale": "FR/UK nukes, fragmented"}, "INDIA": {"score": 48, "rationale": "Nuclear, large army"}, "JP_KR_TW": {"score": 40, "rationale": "Advanced, US-dependent"}, "GULF": {"score": 35, "rationale": "High spend/capita"}, "REST_EM": {"score": 18, "rationale": "Minimal projection"}},
            "D10_social":       {"USA": {"score": 40, "rationale": "Deep polarization"}, "CHINA": {"score": 55, "rationale": "High trust, suppressed"}, "EU": {"score": 60, "rationale": "Safety nets, populism"}, "INDIA": {"score": 45, "rationale": "Trust but tensions"}, "JP_KR_TW": {"score": 65, "rationale": "Low inequality, aging"}, "GULF": {"score": 50, "rationale": "Wealth stability"}, "REST_EM": {"score": 30, "rationale": "High inequality"}},
            "D11_geopolitical": {"USA": {"score": 85, "rationale": "Alliances+sanctions"}, "CHINA": {"score": 60, "rationale": "BRI/SCO growing"}, "EU": {"score": 55, "rationale": "Market+NATO"}, "INDIA": {"score": 50, "rationale": "Non-aligned, Quad"}, "JP_KR_TW": {"score": 45, "rationale": "US-dependent, Taiwan"}, "GULF": {"score": 40, "rationale": "Energy leverage"}, "REST_EM": {"score": 20, "rationale": "Limited leverage"}},
            "D12_feedback":     {"USA": {"score": 35, "rationale": "Debt-demo activating"}, "CHINA": {"score": 30, "rationale": "Property-demo doom loop"}, "EU": {"score": 45, "rationale": "Energy fading, south fiscal"}, "INDIA": {"score": 65, "rationale": "Fewer loops"}, "JP_KR_TW": {"score": 25, "rationale": "Worst debt-demo loop"}, "GULF": {"score": 60, "rationale": "Oil transition, cash buffer"}, "REST_EM": {"score": 35, "rationale": "Currency-fiscal loops"}},
        },
        "enrichment_source": "DETERMINISTIC_FALLBACK",
        "enrichment_date": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# MAIN
# ============================================================

def phase1b_data_enrichment(run_type="WEEKLY", force=False):
    print("[Phase 1b] Data Enrichment...")
    start = time.time()
    if not force and run_type not in ("QUARTERLY", "AD_HOC"):
        cached = load_enrichment_cache()
        if cached:
            return cached
    if not force:
        cached = load_enrichment_cache()
        if cached:
            return cached
    brave_key = os.environ.get("BRAVE_API_KEY", "")
    if not brave_key:
        print("  [Enrichment] No BRAVE_API_KEY — using fallback")
        fb = build_fallback_enrichment()
        save_enrichment_cache(fb)
        return fb
    try:
        search_results = run_brave_searches(brave_key)
        total = sum(len(r) for r in search_results.values())
        print(f"  [Enrichment] Brave: {total} results from {len(search_results)} queries")
    except Exception as e:
        print(f"  [Enrichment] Brave failed: {e}")
        fb = build_fallback_enrichment()
        save_enrichment_cache(fb)
        return fb
    prompt = build_extraction_prompt(search_results)
    try:
        llm_result = call_llm_extraction(prompt)
    except Exception as e:
        print(f"  [Enrichment] LLM failed: {e}")
        llm_result = None
    if llm_result and "dimension_scores" in llm_result:
        valid = True
        for dim in ["D3_technology", "D5_military", "D10_social", "D11_geopolitical", "D12_feedback"]:
            dd = llm_result.get("dimension_scores", {}).get(dim, {})
            if not dd or len(dd) < 7:
                valid = False; break
            for r in REGIONS:
                e = dd.get(r, {})
                s = e.get("score") if isinstance(e, dict) else None
                if s is None or not isinstance(s, (int, float)):
                    valid = False; break
        if valid:
            llm_result["enrichment_source"] = "BRAVE_SEARCH_LLM"
            llm_result["enrichment_date"] = datetime.now(timezone.utc).isoformat()
            llm_result["brave_queries_count"] = len(search_results)
            llm_result["brave_results_total"] = total
            save_enrichment_cache(llm_result)
            print(f"  [Enrichment] Complete: LLM OK, {time.time()-start:.1f}s")
            return llm_result
    print("  [Enrichment] LLM invalid — fallback")
    fb = build_fallback_enrichment()
    save_enrichment_cache(fb)
    print(f"  [Enrichment] Complete: fallback, {time.time()-start:.1f}s")
    return fb
