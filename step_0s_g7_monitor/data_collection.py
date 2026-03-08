"""
step_0s_g7_monitor/data_collection.py
Phase 1: Data Collection

Sammelt alle Rohdaten aus Stufe 1 + Stufe 2 Quellen.
Timeout pro Quelle: 30 Sekunden.
Graceful Degradation: Bei Ausfall -> JSON Cache verwenden.

Stufe 1 (Quantitative):
  - FRED (16 Serien)
  - IMF WEO (GDP, Growth, Inflation, Debt/GDP fuer 7 Regionen)
  - IMF COFER (Reserve Currency Composition)
  - World Bank (11 Indikatoren fuer 7 Regionen)
  - yfinance (12+ Tickers: DXY, Gold, Indices, BDI, VVIX etc.)
  - UN Population (lokaler Cache)
  - BIS (Credit/GDP, REER — optional, oft offline)
  - SWIFT (RMB Tracker — optional, scraping-basiert)

Stufe 2 (Risk Indices):
  - GPR Index (Caldara-Iacoviello CSV)
  - ACLED (Armed Conflict Events — optional, Key pending)
  - Polymarket (Prediction Markets — optional)
  - WorldMonitor (Chokepoints, CII — optional)
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# REGION MAPPING
# ============================================================

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

# ISO country codes per region (for World Bank, IMF queries)
REGION_COUNTRIES = {
    "USA":      ["US"],
    "CHINA":    ["CN"],
    "EU":       ["DE", "FR", "IT", "ES", "NL"],
    "INDIA":    ["IN"],
    "JP_KR_TW": ["JP", "KR"],
    "GULF":     ["SA", "AE"],
    "REST_EM":  ["BR", "MX", "ZA", "ID", "TR"],
}

# yfinance tickers
YFINANCE_TICKERS = [
    "DX-Y.NYB",     # DXY Dollar Index
    "GC=F",          # Gold Futures
    "^GSPC",         # S&P 500
    "^HSI",          # Hang Seng (China proxy)
    "^STOXX50E",     # Euro Stoxx 50
    "^NSEI",         # Nifty 50 (India)
    "^N225",         # Nikkei 225 (Japan)
    "CL=F",          # Crude Oil
    "BTC-USD",       # Bitcoin
    "^VIX",          # VIX
    "^VVIX",         # VVIX
    "^VIX3M",        # VIX 3-Month (term structure)
]

# FRED series
FRED_SERIES = [
    "GDP",                  # US Real GDP
    "GFDEBTN",              # Federal Debt
    "FYFSGDA188S",          # Deficit/GDP
    "A091RC1Q027SBEA",      # Interest Payments
    "FGRECPT",              # Federal Revenue
    "DGS10",                # 10Y Treasury
    "DGS2",                 # 2Y Treasury
    "DTWEXBGS",             # USD Index (broad)
    "DCOILWTICO",           # WTI Crude
    "DCOILBRENTEU",         # Brent Crude
    "PCU483111483111",      # Deep Sea Freight PPI
    "STLFSI4",              # Financial Stress Index
    "T10Y2Y",               # 10Y-2Y Spread
    "BAMLH0A0HYM2",         # HY Spread
    "BAMLC0A0CM",           # IG Corporate OAS (Credit Stress)
    "ANFCI",                # Adjusted NFCI (Chicago Fed)
    "GOLDAMGBD228NLBM",     # Gold Price (London Fix)
    "BAMLMOVE",             # MOVE Index (bond volatility)
]

# World Bank indicators
WB_INDICATORS = [
    "NY.GDP.MKTP.KD.ZG",   # GDP Growth
    "SP.POP.TOTL",          # Population
    "SP.POP.DPND",          # Dependency Ratio
    "SP.DYN.TFRT.IN",       # Fertility Rate
    "SE.XPD.TOTL.GD.ZS",   # Education Spending/GDP
    "GB.XPD.RSDV.GD.ZS",   # R&D Spending/GDP
    "MS.MIL.XPND.GD.ZS",   # Military Spending/GDP
    "GC.DOD.TOTL.GD.ZS",   # Central Govt Debt/GDP
    "FI.RES.TOTL.CD",       # Total Reserves
    "BN.CAB.XOKA.GD.ZS",   # Current Account/GDP
]


# ============================================================
# CACHE HELPERS
# ============================================================

def _cache_path(source):
    return os.path.join(CACHE_DIR, f"{source}.json")


def load_cached(source):
    """Load cached data for a source. Returns dict or None."""
    path = _cache_path(source)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            age_hours = (
                time.time() - os.path.getmtime(path)
            ) / 3600
            print(f"  [Cache] Loaded {source} (age: {age_hours:.0f}h)")
            return data
        except Exception:
            return None
    return None


def save_cache(source, data):
    """Save data to cache."""
    try:
        path = _cache_path(source)
        with open(path, "w") as f:
            json.dump(data, f, default=str)
    except Exception as e:
        print(f"  [Cache] Save failed for {source}: {e}")


# ============================================================
# FRED API
# ============================================================

def fetch_fred(api_key, timeout=30):
    """Fetch all FRED series. Returns dict keyed by series ID."""
    import requests

    print("[Phase 1] Fetching FRED data...")
    result = {}
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    for series_id in FRED_SERIES:
        try:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 10,
            }
            resp = requests.get(base_url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            observations = data.get("observations", [])
            if observations:
                latest = observations[0]
                try:
                    value = float(latest["value"])
                except (ValueError, TypeError):
                    value = None

                result[series_id] = {
                    "value": value,
                    "date": latest.get("date", ""),
                    "series_id": series_id,
                }
        except Exception as e:
            print(f"  FRED {series_id}: {e}")
            result[series_id] = None

    result["last_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"  FRED: {sum(1 for v in result.values() if v is not None)}/{len(FRED_SERIES)} series")
    return result


# ============================================================
# IMF WEO API
# ============================================================

def fetch_imf_weo(timeout=30):
    """
    Fetch IMF World Economic Outlook data.
    Uses the IMF DataMapper API.
    """
    import requests

    print("[Phase 1] Fetching IMF WEO data...")
    result = {}
    base_url = "https://www.imf.org/external/datamapper/api/v1"

    # WEO indicators we need
    indicators = {
        "NGDPD":       "GDP Nominal USD",
        "NGDP_RPCH":   "GDP Growth %",
        "PCPIPCH":     "Inflation %",
        "GGXWDG_NGDP": "Govt Debt/GDP %",
    }

    # IMF uses different country codes
    imf_countries = {
        "USA": "USA", "CHINA": "CHN", "EU": "DEU",
        "INDIA": "IND", "JP_KR_TW": "JPN", "GULF": "SAU",
        "REST_EM": "BRA",
    }

    for ind_code, ind_name in indicators.items():
        try:
            url = f"{base_url}/{ind_code}"
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            values = data.get("values", {}).get(ind_code, {})

            for region, imf_code in imf_countries.items():
                country_data = values.get(imf_code, {})
                if country_data:
                    # Get latest year
                    years = sorted(country_data.keys(), reverse=True)
                    if years:
                        latest_year = years[0]
                        try:
                            val = float(country_data[latest_year])
                        except (ValueError, TypeError):
                            val = None

                        key = f"{ind_code}_{region}"
                        result[key] = {
                            "value": val,
                            "year": latest_year,
                            "indicator": ind_code,
                            "region": region,
                        }
        except Exception as e:
            print(f"  IMF WEO {ind_code}: {e}")

    result["last_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"  IMF WEO: {len(result) - 1} data points")
    return result


# ============================================================
# IMF COFER
# ============================================================

def fetch_imf_cofer(timeout=30):
    """
    Fetch IMF COFER (Currency Composition of Official Foreign Exchange Reserves).
    KERN-Datenquelle fuer DDI.
    """
    import requests

    print("[Phase 1] Fetching IMF COFER data...")
    try:
        # COFER via IMF Data API
        url = "https://data.imf.org/api/v1/data/COFER/Q..?"
        resp = requests.get(url, timeout=timeout)

        if resp.status_code == 200:
            data = resp.json()
            result = {
                "raw": data,
                "last_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
            return result
        else:
            # Fallback: use known recent values
            print(f"  IMF COFER: HTTP {resp.status_code} — using estimates")
            return _cofer_estimates()
    except Exception as e:
        print(f"  IMF COFER: {e} — using estimates")
        return _cofer_estimates()


def _cofer_estimates():
    """Fallback COFER estimates based on latest available data."""
    return {
        "USD_share": 58.4,
        "EUR_share": 19.8,
        "CNY_share": 2.3,
        "JPY_share": 5.4,
        "GBP_share": 4.7,
        "other_share": 9.4,
        "note": "Estimated from Q3 2025 data — COFER API unavailable",
        "last_date": "2025-09-30",
    }


# ============================================================
# WORLD BANK API
# ============================================================

def fetch_worldbank(timeout=30):
    """Fetch World Bank indicators for all regions."""
    import requests

    print("[Phase 1] Fetching World Bank data...")
    result = {}

    # Flatten all country codes
    all_countries = []
    for codes in REGION_COUNTRIES.values():
        all_countries.extend(codes)
    country_str = ";".join(all_countries)

    for indicator in WB_INDICATORS:
        try:
            url = (
                f"https://api.worldbank.org/v2/country/{country_str}/"
                f"indicator/{indicator}?format=json&per_page=100"
                f"&date=2020:2025&mrv=1"
            )
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and len(data) > 1:
                for entry in data[1]:
                    if entry and entry.get("value") is not None:
                        country_code = entry.get("country", {}).get("id", "")
                        # Map country to region
                        region = _country_to_region(country_code)
                        if region:
                            key = f"{indicator}_{region}"
                            result[key] = {
                                "value": entry["value"],
                                "date": entry.get("date", ""),
                                "country": country_code,
                                "region": region,
                                "indicator": indicator,
                            }
        except Exception as e:
            print(f"  World Bank {indicator}: {e}")

    result["last_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"  World Bank: {len(result) - 1} data points")
    return result


def _country_to_region(country_code):
    """Map ISO country code to our 7 regions."""
    for region, codes in REGION_COUNTRIES.items():
        if country_code in codes:
            return region
    return None


# ============================================================
# YFINANCE
# ============================================================

def fetch_yfinance(timeout=30):
    """Fetch market data via yfinance."""
    print("[Phase 1] Fetching yfinance data...")

    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — skipping")
        return None

    result = {}

    for ticker in YFINANCE_TICKERS:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="3mo")

            if hist is not None and not hist.empty:
                latest = hist.iloc[-1]
                result[ticker] = {
                    "close": round(float(latest["Close"]), 4),
                    "date": str(hist.index[-1].date()),
                    "high_3m": round(float(hist["Close"].max()), 4),
                    "low_3m": round(float(hist["Close"].min()), 4),
                    "pct_change_1m": round(
                        float(
                            (latest["Close"] - hist["Close"].iloc[-22])
                            / hist["Close"].iloc[-22] * 100
                        )
                        if len(hist) >= 22 else 0,
                        2,
                    ),
                }
            else:
                result[ticker] = None
        except Exception as e:
            print(f"  yfinance {ticker}: {e}")
            result[ticker] = None

    ok_count = sum(1 for v in result.values() if v is not None)
    print(f"  yfinance: {ok_count}/{len(YFINANCE_TICKERS)} tickers")
    result["last_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return result


# ============================================================
# GPR INDEX (Caldara-Iacoviello)
# ============================================================

def fetch_gpr(timeout=30):
    """
    Fetch Geopolitical Risk Index.
    CSV download from matteoiacoviello.com.
    Uses CSV format (not XLS) to avoid xlrd dependency.
    """
    import requests

    print("[Phase 1] Fetching GPR Index...")
    try:
        # CSV URL — no xlrd needed
        url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.csv"
        resp = requests.get(url, timeout=timeout)

        if resp.status_code == 200:
            try:
                import pandas as pd
                import io

                # Try CSV parse
                text = resp.text
                df = pd.read_csv(io.StringIO(text))

                if df.empty:
                    print("  GPR: CSV empty")
                    return None

                latest = df.iloc[-1]
                gpr_val = None

                # Look for GPR column (various naming conventions)
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if "gpr" in col_lower and "daily" not in col_lower:
                        try:
                            gpr_val = float(latest[col])
                        except (ValueError, TypeError):
                            pass
                        break

                # Fallback: second column
                if gpr_val is None and len(df.columns) > 1:
                    try:
                        gpr_val = float(latest.iloc[1])
                    except (ValueError, TypeError):
                        pass

                # Determine date
                date_str = ""
                for col in df.columns:
                    if "date" in col.lower():
                        date_str = str(latest[col])[:10]
                        break
                if not date_str and len(latest) > 0:
                    date_str = str(latest.iloc[0])[:10]

                print(f"  GPR: {gpr_val} ({date_str})")
                return {
                    "gpr_global": gpr_val,
                    "last_date": date_str,
                    "source": "caldara_iacoviello_csv",
                }
            except Exception as e:
                print(f"  GPR: CSV parse error: {e}")
                # Try XLS fallback (if xlrd available)
                return _fetch_gpr_xls_fallback(timeout)
        else:
            print(f"  GPR: HTTP {resp.status_code} — trying XLS fallback")
            return _fetch_gpr_xls_fallback(timeout)
    except Exception as e:
        print(f"  GPR: {e}")
        return None


def _fetch_gpr_xls_fallback(timeout=30):
    """Fallback: try the XLS URL (needs xlrd or openpyxl)."""
    import requests

    try:
        url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            print(f"  GPR XLS fallback: HTTP {resp.status_code}")
            return None

        import pandas as pd
        import io
        df = pd.read_excel(io.BytesIO(resp.content))
        if df.empty:
            return None

        latest = df.iloc[-1]
        gpr_val = None
        for col in df.columns:
            if "gpr" in str(col).lower() and "daily" not in str(col).lower():
                try:
                    gpr_val = float(latest[col])
                except (ValueError, TypeError):
                    pass
                break
        if gpr_val is None and len(df.columns) > 1:
            try:
                gpr_val = float(latest.iloc[1])
            except (ValueError, TypeError):
                pass

        print(f"  GPR (XLS fallback): {gpr_val}")
        return {
            "gpr_global": gpr_val,
            "last_date": str(latest.iloc[0])[:10] if len(latest) > 0 else "",
            "source": "caldara_iacoviello_xls",
        }
    except Exception as e:
        print(f"  GPR XLS fallback: {e}")
        return None


# ============================================================
# ACLED (optional — key may be pending)
# ============================================================

def fetch_acled(timeout=30):
    """
    Fetch ACLED conflict/protest events.
    Optional — graceful degradation if key not available.
    """
    api_key = os.environ.get("ACLED_API_KEY")
    email = os.environ.get("ACLED_EMAIL", "")

    if not api_key:
        print("[Phase 1] ACLED: No API key — skipping (optional)")
        return None

    import requests

    print("[Phase 1] Fetching ACLED data...")
    try:
        # Last 30 days of events, aggregated by region
        date_from = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).strftime("%Y-%m-%d")

        url = "https://api.acleddata.com/acled/read"
        params = {
            "key": api_key,
            "email": email,
            "event_date": f"{date_from}|",
            "event_date_where": "BETWEEN",
            "limit": 0,  # Just get count
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        return {
            "total_events_30d": data.get("count", 0),
            "last_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "source": "acled_api",
        }
    except Exception as e:
        print(f"  ACLED: {e}")
        return None


# ============================================================
# POLYMARKET (optional)
# ============================================================

def fetch_polymarket(timeout=15):
    """
    Fetch relevant Polymarket prediction markets.
    Optional — public API, no key needed.
    """
    import requests

    print("[Phase 1] Fetching Polymarket data...")
    try:
        # Polymarket CLOB API — search for geopolitical markets
        url = "https://clob.polymarket.com/markets"
        params = {"limit": 50, "active": "true"}
        resp = requests.get(url, params=params, timeout=timeout)

        if resp.status_code == 200:
            raw = resp.json()

            # Response type check: API may return list, dict, or string
            if isinstance(raw, dict):
                markets = raw.get("data", raw.get("markets", []))
                if not isinstance(markets, list):
                    markets = []
            elif isinstance(raw, list):
                markets = raw
            else:
                print(f"  Polymarket: unexpected response type: {type(raw).__name__}")
                return None

            # Filter for geopolitically relevant markets
            relevant_keywords = [
                "china", "taiwan", "war", "conflict", "tariff",
                "dollar", "brics", "nato", "nuclear", "sanctions",
                "recession", "gdp",
            ]

            relevant = []
            for market in markets:
                if not isinstance(market, dict):
                    continue
                title = (market.get("question", "") or market.get("title", "") or "").lower()
                if any(kw in title for kw in relevant_keywords):
                    # Extract probability safely
                    prob = market.get("last_trade_price")
                    if prob is not None:
                        try:
                            prob = float(prob)
                        except (ValueError, TypeError):
                            prob = None

                    volume = market.get("volume")
                    if volume is not None:
                        try:
                            volume = float(volume)
                        except (ValueError, TypeError):
                            volume = None

                    relevant.append({
                        "id": market.get("condition_id", market.get("id", "")),
                        "title": market.get("question", market.get("title", "")),
                        "probability": prob,
                        "volume": volume,
                    })

            return {
                "markets": relevant[:20],
                "total_scanned": len(markets),
                "relevant_found": len(relevant),
                "last_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
        else:
            print(f"  Polymarket: HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"  Polymarket: {e}")
        return None


# ============================================================
# WORLDMONITOR (optional)
# ============================================================

def fetch_worldmonitor(timeout=15):
    """
    Fetch WorldMonitor.app data (chokepoints, CII).
    Optional — public API.
    """
    import requests

    print("[Phase 1] Fetching WorldMonitor data...")
    try:
        # Try the public API endpoints
        chokepoints_url = "https://worldmonitor.app/api/chokepoints"
        resp = requests.get(chokepoints_url, timeout=timeout)

        if resp.status_code == 200:
            return {
                "chokepoints": resp.json(),
                "last_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
        else:
            print(f"  WorldMonitor: HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"  WorldMonitor: {e}")
        return None


# ============================================================
# MAIN COLLECTION FUNCTION
# ============================================================

def phase1_data_collection():
    """
    Phase 1: Collect all raw data.
    Each source has independent error handling.
    Failed sources fall back to JSON cache.
    Engine NEVER aborts because a source is unavailable.
    """
    print("=" * 50)
    print("[Phase 1] Data Collection starting...")
    print("=" * 50)

    raw_data = {}
    errors = []
    start = time.time()

    # --- FRED ---
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            raw_data["fred"] = fetch_fred(fred_key)
            if raw_data["fred"]:
                save_cache("fred", raw_data["fred"])
        except Exception as e:
            errors.append({"source": "FRED", "error": str(e), "severity": "MEDIUM"})
            raw_data["fred"] = load_cached("fred")
    else:
        print("  FRED: No API key — using cache")
        raw_data["fred"] = load_cached("fred")
        if not raw_data["fred"]:
            errors.append({"source": "FRED", "error": "No API key and no cache", "severity": "MEDIUM"})

    # --- IMF WEO ---
    try:
        raw_data["imf_weo"] = fetch_imf_weo()
        if raw_data["imf_weo"]:
            save_cache("imf_weo", raw_data["imf_weo"])
    except Exception as e:
        errors.append({"source": "IMF_WEO", "error": str(e), "severity": "MEDIUM"})
        raw_data["imf_weo"] = load_cached("imf_weo")

    # --- IMF COFER ---
    try:
        raw_data["imf_cofer"] = fetch_imf_cofer()
        if raw_data["imf_cofer"]:
            save_cache("imf_cofer", raw_data["imf_cofer"])
    except Exception as e:
        errors.append({"source": "IMF_COFER", "error": str(e), "severity": "MEDIUM"})
        raw_data["imf_cofer"] = load_cached("imf_cofer")

    # --- World Bank ---
    try:
        raw_data["worldbank"] = fetch_worldbank()
        if raw_data["worldbank"]:
            save_cache("worldbank", raw_data["worldbank"])
    except Exception as e:
        errors.append({"source": "WorldBank", "error": str(e), "severity": "LOW"})
        raw_data["worldbank"] = load_cached("worldbank")

    # --- yfinance ---
    try:
        raw_data["yfinance"] = fetch_yfinance()
        if raw_data["yfinance"]:
            save_cache("yfinance", raw_data["yfinance"])
    except Exception as e:
        errors.append({"source": "yfinance", "error": str(e), "severity": "MEDIUM"})
        raw_data["yfinance"] = load_cached("yfinance")

    # --- UN Population (always from cache — updated ~every 2 years) ---
    raw_data["un_pop"] = load_cached("un_pop")
    if not raw_data["un_pop"]:
        # Create minimal placeholder — will be populated manually
        raw_data["un_pop"] = {
            "note": "UN Population data not yet cached. Run initial data load.",
            "last_date": "2024-01-01",
        }

    # --- GPR Index ---
    try:
        raw_data["gpr"] = fetch_gpr()
        if raw_data["gpr"]:
            save_cache("gpr", raw_data["gpr"])
    except Exception as e:
        errors.append({"source": "GPR", "error": str(e), "severity": "MEDIUM"})
        raw_data["gpr"] = load_cached("gpr")

    # --- ACLED (optional) ---
    try:
        raw_data["acled"] = fetch_acled()
        if raw_data["acled"]:
            save_cache("acled", raw_data["acled"])
    except Exception as e:
        errors.append({"source": "ACLED", "error": str(e), "severity": "LOW"})
        raw_data["acled"] = load_cached("acled")

    # --- Polymarket (optional) ---
    try:
        raw_data["polymarket"] = fetch_polymarket()
        if raw_data["polymarket"]:
            save_cache("polymarket", raw_data["polymarket"])
    except Exception as e:
        errors.append({"source": "Polymarket", "error": str(e), "severity": "LOW"})
        raw_data["polymarket"] = None

    # --- WorldMonitor (optional) ---
    try:
        raw_data["worldmonitor"] = fetch_worldmonitor()
        if raw_data["worldmonitor"]:
            save_cache("worldmonitor", raw_data["worldmonitor"])
    except Exception as e:
        errors.append({"source": "WorldMonitor", "error": str(e), "severity": "LOW"})
        raw_data["worldmonitor"] = None

    # --- Summary ---
    duration = time.time() - start
    sources_ok = sum(1 for v in raw_data.values() if v is not None)
    sources_total = len(raw_data)

    print(f"\n[Phase 1] Complete: {sources_ok}/{sources_total} sources, "
          f"{len(errors)} errors, {duration:.1f}s")

    return {
        "raw_data": raw_data,
        "errors": errors,
        "sources_available": sources_ok,
        "sources_failed": len(errors),
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration, 1),
    }
