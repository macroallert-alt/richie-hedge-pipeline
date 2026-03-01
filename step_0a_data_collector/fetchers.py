"""
fetchers.py — Phase 1: Data Acquisition
=========================================
4-Stufen-Fallback pro Feld:
  Stufe 1: Primaere API (FRED, yfinance, CFTC)
  Stufe 2: Sekundaere API (EODHD, alt. FRED-Serie)
  Stufe 3: Web Scraping (BeautifulSoup, requests)
  Stufe 4: LLM + Web Search (Claude Haiku)
  Stufe 5: Stale Cache (letzter bekannter Wert)

Output: Dict[field_name -> FetchResult]
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("data_collector.fetchers")

# ═══════════════════════════════════════════════════════
# FETCH RESULT
# ═══════════════════════════════════════════════════════

class FetchResult:
    """Standardisiertes Ergebnis eines Fetch-Vorgangs."""
    def __init__(self, field_name: str, value=None, source: str = None,
                 source_method: str = "ALL_FAILED", source_date=None,
                 extra: dict = None):
        self.field_name = field_name
        self.value = value
        self.source = source
        self.source_method = source_method  # API_PRIMARY|API_SECONDARY|SCRAPE|LLM_FALLBACK|STALE_CACHE|ALL_FAILED
        self.source_date = source_date or datetime.now().date()
        self.extra = extra or {}  # For additional data (e.g. volume, high, low)
        self.success = value is not None

    def __repr__(self):
        return f"FetchResult({self.field_name}={self.value}, {self.source_method})"


# ═══════════════════════════════════════════════════════
# FRED FETCHER
# ═══════════════════════════════════════════════════════

class FredFetcher:
    """Batch-fetcher fuer FRED API."""

    def __init__(self, api_key: str):
        from fredapi import Fred
        self.fred = Fred(api_key=api_key)
        self.call_count = 0

    def fetch_series(self, series_id: str, lookback_days: int = 30) -> Optional[pd.Series]:
        """Holt eine FRED-Serie. Gibt pd.Series zurueck oder None."""
        try:
            end = datetime.now()
            start = end - timedelta(days=lookback_days)
            data = self.fred.get_series(series_id, observation_start=start, observation_end=end)
            self.call_count += 1
            if data is not None and len(data) > 0:
                return data.dropna()
            return None
        except Exception as e:
            logger.warning(f"FRED {series_id} failed: {e}")
            return None

    def fetch_latest(self, series_id: str, lookback_days: int = 30) -> Optional[Tuple[float, date]]:
        """Holt den letzten Wert einer FRED-Serie. Returns (value, date) or None."""
        data = self.fetch_series(series_id, lookback_days)
        if data is not None and len(data) > 0:
            return float(data.iloc[-1]), data.index[-1].date()
        return None

    def fetch_batch(self, series_ids: List[str], lookback_days: int = 30) -> Dict[str, Optional[Tuple[float, date]]]:
        """Batch-fetch mehrerer FRED-Serien."""
        results = {}
        for sid in series_ids:
            results[sid] = self.fetch_latest(sid, lookback_days)
            time.sleep(0.1)  # Rate limiting
        return results


# ═══════════════════════════════════════════════════════
# YFINANCE FETCHER
# ═══════════════════════════════════════════════════════

class YFinanceFetcher:
    """Fetcher fuer yfinance Marktdaten."""

    def __init__(self):
        import yfinance as yf
        self.yf = yf
        self.call_count = 0

    def fetch_ticker(self, ticker: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """Holt OHLCV-Daten fuer einen Ticker."""
        try:
            data = self.yf.download(ticker, period=period, progress=False, timeout=15)
            self.call_count += 1
            if data is not None and len(data) > 0:
                return data
            return None
        except Exception as e:
            logger.warning(f"yfinance {ticker} failed: {e}")
            return None

    def fetch_close(self, ticker: str, period: str = "5d") -> Optional[Tuple[float, date]]:
        """Holt den letzten Close-Preis."""
        time.sleep(0.5)  # Rate limit
        data = self.fetch_ticker(ticker, period)
        if data is not None and len(data) > 0:
            close_col = 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                close_col = ('Close', ticker)
                if close_col not in data.columns:
                    # Flatten
                    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                    close_col = 'Close'
            val = float(data[close_col].dropna().iloc[-1])
            dt = data.index[-1]
            if hasattr(dt, 'date'):
                dt = dt.date()
            return val, dt
        return None

    def fetch_ohlcv(self, ticker: str, period: str = "5d") -> Optional[Dict]:
        """Holt OHLCV als Dict mit letztem Tag."""
        time.sleep(0.5)  # Rate limit
        data = self.fetch_ticker(ticker, period)
        if data is not None and len(data) > 0:
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            last = data.iloc[-1]
            dt = data.index[-1]
            if hasattr(dt, 'date'):
                dt = dt.date()
            return {
                "open": float(last.get("Open", np.nan)),
                "high": float(last.get("High", np.nan)),
                "low": float(last.get("Low", np.nan)),
                "close": float(last.get("Close", np.nan)),
                "volume": float(last.get("Volume", 0)),
                "date": dt
            }
        return None

    def fetch_batch_close(self, tickers: List[str], period: str = "5d") -> Dict[str, Optional[Tuple[float, date]]]:
        """Batch-Download mehrerer Ticker. Splits into chunks to avoid rate limits."""
        results = {}
        chunk_size = 10
        chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]

        for ci, chunk in enumerate(chunks):
            try:
                data = self.yf.download(chunk, period=period, progress=False, timeout=30, group_by='ticker')
                self.call_count += 1
                if data is not None and len(data) > 0:
                    for ticker in chunk:
                        try:
                            if len(chunk) == 1:
                                # Single ticker: no MultiIndex
                                if isinstance(data.columns, pd.MultiIndex):
                                    close_series = data[('Close', ticker)].dropna()
                                else:
                                    close_series = data['Close'].dropna()
                            elif isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.get_level_values(0):
                                close_series = data[(ticker, 'Close')].dropna()
                            else:
                                close_series = pd.Series(dtype=float)

                            if len(close_series) > 0:
                                val = float(close_series.iloc[-1])
                                dt = close_series.index[-1]
                                if hasattr(dt, 'date'):
                                    dt = dt.date()
                                results[ticker] = (val, dt)
                            else:
                                results[ticker] = None
                        except Exception:
                            results[ticker] = None
            except Exception as e:
                logger.warning(f"yfinance batch chunk {ci+1} failed: {e}")
                for ticker in chunk:
                    results.setdefault(ticker, None)

            if ci < len(chunks) - 1:
                time.sleep(2)  # Rate limit pause between chunks

        return results

    def fetch_sp500_breadth(self, sp500_tickers: List[str]) -> Optional[Dict]:
        """
        Berechnet Breadth-Metriken aus S&P 500 Komponenten.
        Returns: pct_above_200dma, nh_nl, trin, oder None.
        Batch: 10x50 Ticker, 2s Pause.
        """
        all_data = {}
        batch_size = 25
        batches = [sp500_tickers[i:i+batch_size] for i in range(0, len(sp500_tickers), batch_size)]

        for i, batch in enumerate(batches):
            try:
                data = self.yf.download(batch, period="1y", progress=False, timeout=60)
                self.call_count += 1
                if data is not None and len(data) > 0:
                    if isinstance(data.columns, pd.MultiIndex):
                        for ticker in batch:
                            try:
                                close = data[('Close', ticker)].dropna()
                                if len(close) > 0:
                                    all_data[ticker] = close
                            except (KeyError, Exception):
                                pass
                    elif 'Close' in data.columns:
                        # Single ticker fallback
                        all_data[batch[0]] = data['Close'].dropna()
            except Exception as e:
                logger.warning(f"Breadth batch {i+1} failed: {e}")

            if i < len(batches) - 1:
                time.sleep(5)  # Longer pause for breadth batches

        if len(all_data) < len(sp500_tickers) * 0.80:
            logger.warning(f"Breadth: only {len(all_data)}/{len(sp500_tickers)} tickers valid")
            if len(all_data) < len(sp500_tickers) * 0.50:
                return None

        # Build DataFrame
        prices_df = pd.DataFrame(all_data)
        if len(prices_df) < 200:
            logger.warning(f"Breadth: only {len(prices_df)} days of data, need 200")
            return None

        valid_tickers = prices_df.columns.tolist()
        latest = prices_df.iloc[-1]
        prev = prices_df.iloc[-2] if len(prices_df) > 1 else latest

        # --- % above 200 DMA ---
        sma200 = prices_df.rolling(200, min_periods=180).mean().iloc[-1]
        both_valid = latest.dropna().index.intersection(sma200.dropna().index)
        n_valid = len(both_valid)
        if n_valid > 0:
            pct_above = float((latest[both_valid] > sma200[both_valid]).sum() / n_valid * 100)
        else:
            pct_above = None

        # --- New Highs - New Lows ---
        high_52w = prices_df.tail(252).max()
        low_52w = prices_df.tail(252).min()
        new_highs = int((latest.dropna() >= high_52w.reindex(latest.dropna().index) * 0.98).sum())
        new_lows = int((latest.dropna() <= low_52w.reindex(latest.dropna().index) * 1.02).sum())
        nh_nl = new_highs - new_lows

        # --- TRIN ---
        returns_today = (latest / prev - 1).dropna()
        advancing = returns_today[returns_today > 0]
        declining = returns_today[returns_today < 0]
        adv_count = len(advancing)
        dec_count = len(declining)
        if dec_count > 0 and adv_count > 0:
            # Use price as volume proxy (we don't have individual volumes cheaply)
            adv_price_sum = latest.reindex(advancing.index).sum()
            dec_price_sum = latest.reindex(declining.index).sum()
            if dec_price_sum > 0:
                trin = float((adv_count / dec_count) / (adv_price_sum / dec_price_sum))
            else:
                trin = None
        else:
            trin = None

        return {
            "pct_above_200dma": round(pct_above, 2) if pct_above is not None else None,
            "nh_nl": nh_nl,
            "trin": round(trin, 4) if trin is not None else None,
            "valid_tickers": n_valid,
            "total_tickers": len(sp500_tickers)
        }


# ═══════════════════════════════════════════════════════
# SCRAPE FETCHER
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# FMP FETCHER (Financial Modeling Prep)
# ═══════════════════════════════════════════════════════

class FMPFetcher:
    """Primary price source via Financial Modeling Prep API (stable endpoints)."""

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str):
        import requests
        self.api_key = api_key
        self.session = requests.Session()
        self.call_count = 0

    def _get(self, endpoint: str, params: dict = None) -> Optional[list]:
        params = params or {}
        params['apikey'] = self.api_key
        try:
            resp = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=15)
            self.call_count += 1
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data
                elif isinstance(data, dict) and not data.get('Error Message'):
                    return [data]
            else:
                logger.warning(f"FMP {endpoint}: HTTP {resp.status_code}")
            return None
        except Exception as e:
            logger.warning(f"FMP {endpoint} failed: {e}")
            return None

    def fetch_quote(self, symbol: str) -> Optional[Tuple[float, date]]:
        """Holt aktuellen Preis fuer ein Symbol."""
        data = self._get("quote", {"symbol": symbol})
        if data and data[0].get('price'):
            price = float(data[0]['price'])
            ts = data[0].get('timestamp')
            if ts:
                from datetime import datetime as dt
                d = dt.fromtimestamp(ts).date()
            else:
                d = datetime.now().date()
            return price, d
        return None

    def fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, Optional[Tuple[float, date]]]:
        """Fetch quotes one by one (stable API doesn't support comma-separated batch)."""
        results = {}
        for sym in symbols:
            result = self.fetch_quote(sym)
            if result:
                results[sym] = result
            else:
                results[sym] = None
            time.sleep(0.15)  # ~6 per second, stays within limits
        return results

    def fetch_forex(self, pair: str) -> Optional[Tuple[float, date]]:
        """Holt Forex-Kurs, z.B. EURUSD."""
        data = self._get("quote", {"symbol": f"{pair}"})
        if data and data[0].get('price'):
            return float(data[0]['price']), datetime.now().date()
        return None


class ScrapeFetcher:
    """Web Scraping fuer CBOE, NAAIM, AAII, Atlanta Fed, etc."""

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_cboe_pc_ratio(self) -> Optional[Tuple[float, date]]:
        """CBOE Equity Put/Call Ratio."""
        try:
            url = "https://www.cboe.com/us/options/market_statistics/daily/"
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                # CBOE liefert CSV-Downloads — versuche direkten CSV-Link
                # Fallback: Parse HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Suche nach Equity P/C Ratio im Text
                text = soup.get_text()
                # Einfacher Regex-Ansatz
                import re
                match = re.search(r'equity.*?put.*?call.*?ratio.*?([\d.]+)', text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    if 0.1 <= val <= 3.0:
                        return val, datetime.now().date()
            return None
        except Exception as e:
            logger.warning(f"CBOE scrape failed: {e}")
            return None

    def fetch_naaim(self) -> Optional[Tuple[float, date]]:
        """NAAIM Exposure Index."""
        try:
            url = "https://www.naaim.org/programs/naaim-exposure-index/"
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                import re
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text()
                # Suche nach Exposure Index Zahl
                match = re.search(r'(?:exposure|index).*?([\d.]+)', text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    if -50 <= val <= 200:
                        return val, datetime.now().date()
            return None
        except Exception as e:
            logger.warning(f"NAAIM scrape failed: {e}")
            return None

    def fetch_aaii(self) -> Optional[Tuple[float, date]]:
        """AAII Bull-Bear Spread."""
        try:
            url = "https://www.aaii.com/sentimentsurvey"
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                import re
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text()
                # Suche Bull und Bear Werte
                bull_match = re.search(r'bullish.*?([\d.]+)%', text, re.IGNORECASE)
                bear_match = re.search(r'bearish.*?([\d.]+)%', text, re.IGNORECASE)
                if bull_match and bear_match:
                    spread = float(bull_match.group(1)) - float(bear_match.group(1))
                    if -60 <= spread <= 60:
                        return spread, datetime.now().date()
            return None
        except Exception as e:
            logger.warning(f"AAII scrape failed: {e}")
            return None

    def fetch_gdpnow(self) -> Optional[Tuple[float, date]]:
        """Atlanta Fed GDPNow."""
        try:
            url = "https://www.atlantafed.org/cqer/research/gdpnow"
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                import re
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text()
                match = re.search(r'([-\d.]+)\s*percent', text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    if -10 <= val <= 15:
                        return val, datetime.now().date()
            return None
        except Exception as e:
            logger.warning(f"GDPNow scrape failed: {e}")
            return None

    def fetch_baltic_dry(self) -> Optional[Tuple[float, date]]:
        """Baltic Dry Index."""
        try:
            url = "https://tradingeconomics.com/commodity/baltic"
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                import re
                soup = BeautifulSoup(resp.text, 'html.parser')
                # TradingEconomics hat den Wert im Titel oder in einem spezifischen Element
                text = soup.get_text()
                match = re.search(r'baltic.*?dry.*?([\d,]+)', text, re.IGNORECASE)
                if match:
                    val = float(match.group(1).replace(',', ''))
                    if 100 <= val <= 15000:
                        return val, datetime.now().date()
            return None
        except Exception as e:
            logger.warning(f"Baltic Dry scrape failed: {e}")
            return None


# ═══════════════════════════════════════════════════════
# LLM FALLBACK FETCHER
# ═══════════════════════════════════════════════════════

class LLMFetcher:
    """Claude Haiku mit Web Search als Stufe-4-Fallback."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.call_count = 0
        self.max_calls = 25  # Budget pro Run

    def fetch(self, field_name: str, llm_config: dict) -> Optional[Tuple[float, date]]:
        """
        LLM + Web Search als Datenpunkt-Extraktor.
        Nur fuer Felder mit source_llm_eligible=true.
        """
        if self.call_count >= self.max_calls:
            logger.warning(f"LLM budget exhausted ({self.call_count}/{self.max_calls})")
            return None

        # Rate limit: wait between calls to avoid 429
        if self.call_count > 0:
            time.sleep(12)  # 12s between calls = ~5 calls/min, safe under 50k tokens/min

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                temperature=0,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": llm_config["extraction_prompt"]}]
            )

            self.call_count += 1

            # Parse response
            result_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    result_text += block.text

            # Extract JSON from response
            result = self._parse_json(result_text)
            if result and result.get("value") is not None:
                value = float(result["value"])
                p = llm_config.get("plausibility_check", {})
                if p.get("min", -9e9) <= value <= p.get("max", 9e9):
                    logger.info(f"LLM extracted {field_name}={value}")
                    return value, datetime.now().date()
                else:
                    logger.warning(f"LLM value {value} outside plausibility for {field_name}")

            return None
        except Exception as e:
            logger.warning(f"LLM fallback failed for {field_name}: {e}")
            return None

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extrahiert JSON aus LLM-Antwort."""
        import re
        # Suche nach JSON-Block
        patterns = [
            r'\{[^{}]*"value"[^{}]*\}',
            r'\{.*?\}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
        return None


# ═══════════════════════════════════════════════════════
# CFTC / COT FETCHER
# ═══════════════════════════════════════════════════════

class COTFetcher:
    """CFTC Commitments of Traders — Disaggregated Reports."""

    def __init__(self):
        import requests
        self.session = requests.Session()

    def fetch_leveraged_net(self, contract_code: str) -> Optional[Tuple[float, date]]:
        """
        Holt Net Position der Leveraged Funds fuer einen Contract.
        contract_code: z.B. '13874A' (ES), '043602' (ZN)
        """
        try:
            url = "https://www.cftc.gov/dea/newcot/deafut.txt"
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                return None

            lines = resp.text.strip().split('\n')
            if len(lines) < 2:
                return None

            # Parse header
            header = lines[0].split(',')
            # Finde relevante Spalten
            code_idx = None
            long_idx = None
            short_idx = None
            date_idx = None

            for i, h in enumerate(header):
                h_clean = h.strip().strip('"').lower()
                if 'cftc contract market code' in h_clean:
                    code_idx = i
                elif 'money manager' in h_clean and 'long' in h_clean and 'all' not in h_clean:
                    if long_idx is None:
                        long_idx = i
                elif 'money manager' in h_clean and 'short' in h_clean and 'all' not in h_clean:
                    if short_idx is None:
                        short_idx = i
                elif 'as of date' in h_clean:
                    date_idx = i

            if None in [code_idx, long_idx, short_idx]:
                # Try "Lev Money" columns instead
                for i, h in enumerate(header):
                    h_clean = h.strip().strip('"').lower()
                    if 'lev' in h_clean and 'long' in h_clean:
                        long_idx = i
                    elif 'lev' in h_clean and 'short' in h_clean:
                        short_idx = i

            if None in [code_idx]:
                logger.warning("COT: Could not find contract code column")
                return None

            # Suche Contract
            for line in lines[1:]:
                fields = line.split(',')
                if len(fields) > max(code_idx, long_idx or 0, short_idx or 0):
                    code = fields[code_idx].strip().strip('"')
                    if contract_code in code:
                        try:
                            long_val = int(fields[long_idx].strip().strip('"').replace(',', ''))
                            short_val = int(fields[short_idx].strip().strip('"').replace(',', ''))
                            net = long_val - short_val
                            report_date = datetime.now().date()
                            if date_idx and len(fields) > date_idx:
                                try:
                                    report_date = datetime.strptime(
                                        fields[date_idx].strip().strip('"'), '%Y-%m-%d'
                                    ).date()
                                except ValueError:
                                    pass
                            return float(net), report_date
                        except (ValueError, IndexError):
                            continue

            return None
        except Exception as e:
            logger.warning(f"COT fetch failed for {contract_code}: {e}")
            return None


# ═══════════════════════════════════════════════════════
# V16 SHEET PRICE FETCHER
# ═══════════════════════════════════════════════════════

class V16PriceFetcher:
    """
    Holt die 27 Asset-Preise fuer das V16 Sheet (DATA_Prices Tab).
    Primaer: FMP API. Fallback: yfinance.
    """

    # FMP symbols (FMP nutzt andere Ticker fuer Futures/Crypto)
    FMP_TICKERS = {
        'GLD': 'GLD', 'SLV': 'SLV', 'GDX': 'GDX', 'GDXJ': 'GDXJ', 'SIL': 'SIL',
        'SPY': 'SPY', 'XLY': 'XLY', 'XLI': 'XLI', 'XLF': 'XLF', 'XLE': 'XLE',
        'IWM': 'IWM', 'XLV': 'XLV', 'XLP': 'XLP', 'XLU': 'XLU', 'VNQ': 'VNQ',
        'XLK': 'XLK', 'EEM': 'EEM', 'VGK': 'VGK',
        'TLT': 'TLT', 'TIP': 'TIP', 'LQD': 'LQD', 'HYG': 'HYG',
        'DBC': 'DBC',
        'BTC': 'BTCUSD', 'ETH': 'ETHUSD',
        'PLATINUM': 'PLUSD', 'COPPER': 'HGUSD',
    }

    # No yfinance fallback needed anymore
    YF_ONLY = {}

    def __init__(self, fmp_fetcher: FMPFetcher, yf_fetcher: YFinanceFetcher):
        self.fmp = fmp_fetcher
        self.yf = yf_fetcher

    def fetch_all(self) -> Dict[str, Optional[Tuple[float, date]]]:
        """Holt alle 27 V16 Preise. FMP primaer, yfinance Fallback."""
        mapped = {}

        # FMP batch (25 auf einmal)
        fmp_symbols = list(self.FMP_TICKERS.values())
        logger.info(f"  FMP: fetching {len(fmp_symbols)} symbols...")
        fmp_results = self.fmp.fetch_batch_quotes(fmp_symbols)

        for sheet_name, fmp_sym in self.FMP_TICKERS.items():
            if fmp_sym in fmp_results and fmp_results[fmp_sym] is not None:
                mapped[sheet_name] = fmp_results[fmp_sym]
            else:
                mapped[sheet_name] = None
                logger.warning(f"V16 price missing from FMP: {sheet_name} ({fmp_sym})")

        # Futures via yfinance (nur 2, kein Rate-Limit-Problem)
        for sheet_name, yf_ticker in self.YF_ONLY.items():
            result = self.yf.fetch_close(yf_ticker, '5d')
            if result:
                mapped[sheet_name] = result
            else:
                mapped[sheet_name] = None
                logger.warning(f"V16 price missing: {sheet_name} ({yf_ticker})")

        ok = sum(1 for v in mapped.values() if v is not None)
        logger.info(f"  V16 Prices: {ok}/27 OK (FMP: {self.fmp.call_count} calls)")
        return mapped


# ═══════════════════════════════════════════════════════
# V16 LIQUIDITY FETCHER
# ═══════════════════════════════════════════════════════

class V16LiquidityFetcher:
    """
    Holt die Liquiditaets-Daten fuer DATA_Liquidity Tab.
    Spalten: Fed_Net_Liq, ECB_USD, BOJ_USD, China_M2_USD, US_M2
    """

    FRED_SERIES = {
        'WALCL': 'WALCL',         # Fed Balance Sheet (Mio USD)
        'TGA': 'WTREGEN',          # Treasury General Account (Mio USD)
        'RRP': 'RRPONTSYD',        # Reverse Repo (Mio USD)
        'ECB': 'ECBASSETSW',       # ECB Total Assets (Mio EUR)
        'US_M2': 'WM2NS',          # US M2 (Bil USD)
    }

    def __init__(self, fred_fetcher: FredFetcher, yf_fetcher: YFinanceFetcher, fmp_fetcher: FMPFetcher = None):
        self.fred = fred_fetcher
        self.yf = yf_fetcher
        self.fmp = fmp_fetcher

    def _fetch_fx(self, pair_fmp: str, pair_yf: str) -> Optional[Tuple[float, date]]:
        """FX holen: FMP primaer, yfinance Fallback."""
        if self.fmp:
            result = self.fmp.fetch_forex(pair_fmp)
            if result:
                return result
        return self.yf.fetch_close(pair_yf, '5d')

    def fetch_all(self) -> Dict[str, Any]:
        """
        Holt alle Liquidity-Komponenten.
        Returns Dict mit allen Werten fuer eine Sheet-Zeile.
        """
        results = {}

        # FRED Batch
        fred_data = self.fred.fetch_batch(
            list(self.FRED_SERIES.values()), lookback_days=30
        )

        # Fed Net Liq = WALCL - TGA - RRP
        walcl = fred_data.get('WALCL')
        tga = fred_data.get('WTREGEN')
        rrp = fred_data.get('RRPONTSYD')

        if walcl and tga and rrp:
            net_liq = walcl[0] - tga[0] - rrp[0]
            results['Fed_Net_Liq'] = net_liq
            results['_walcl'] = walcl[0]
            results['_tga'] = tga[0]
            results['_rrp'] = rrp[0]
        else:
            results['Fed_Net_Liq'] = None
            logger.warning("Fed Net Liq: missing components")

        # ECB in USD (ECB Assets in EUR * EURUSD)
        ecb_eur = fred_data.get('ECBASSETSW')
        eurusd = self._fetch_fx('EURUSD', 'EURUSD=X')
        if ecb_eur and eurusd:
            results['ECB_USD'] = ecb_eur[0] * eurusd[0]
        else:
            results['ECB_USD'] = None

        # BOJ in USD — FRED JPNASSETS nicht immer verfuegbar, yfinance als Proxy
        boj = self.fred.fetch_latest('JPNASSETS', lookback_days=90)
        usdjpy = self._fetch_fx('USDJPY', 'USDJPY=X')
        if boj and usdjpy and usdjpy[0] > 0:
            # BOJ assets in Mrd JPY, convert to USD
            results['BOJ_USD'] = boj[0] * 1e6 / usdjpy[0]  # Mio JPY -> USD
        else:
            results['BOJ_USD'] = None

        # China M2 in USD
        china_m2 = self.fred.fetch_latest('MYAGM2CNM189N', lookback_days=90)
        usdcnh = self._fetch_fx('USDCNH', 'USDCNH=X')
        if china_m2 and usdcnh and usdcnh[0] > 0:
            results['China_M2_USD'] = china_m2[0] * 1e8 / usdcnh[0]  # 100M CNY -> USD
        else:
            results['China_M2_USD'] = None

        # US M2
        us_m2 = fred_data.get('WM2NS')
        results['US_M2'] = us_m2[0] if us_m2 else None

        return results


# ═══════════════════════════════════════════════════════
# CALCULATED FIELDS
# ═══════════════════════════════════════════════════════

class CalculatedFields:
    """Berechnet abgeleitete Felder aus Rohdaten."""

    def __init__(self, yf_fetcher: YFinanceFetcher, fmp_fetcher: FMPFetcher = None):
        self.yf = yf_fetcher
        self.fmp = fmp_fetcher

    def _get_price(self, fmp_sym: str, yf_sym: str) -> Optional[Tuple[float, date]]:
        """FMP primaer, yfinance Fallback."""
        if self.fmp:
            result = self.fmp.fetch_quote(fmp_sym)
            if result:
                return result
        return self.yf.fetch_close(yf_sym, '5d')

    def compute_vix_term_ratio(self) -> Optional[Tuple[float, date]]:
        """VIX3M / VIX — Term Structure."""
        vix = self._get_price('^VIX', '^VIX')
        vix3m = self._get_price('^VIX3M', '^VIX3M')
        if vix and vix3m and vix[0] > 0:
            ratio = vix3m[0] / vix[0]
            return round(ratio, 4), vix[1]
        return None

    def compute_iv_rv_spread(self) -> Optional[Tuple[float, date]]:
        """VIX - Realized Vol (21d) via FMP."""
        vix = self._get_price('^VIX', '^VIX')
        if not vix or not self.fmp:
            return None
        try:
            spy_hist = self.fmp._get("historical-price-eod/light", {"symbol": "SPY", "from": (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')})
            if spy_hist and len(spy_hist) >= 21:
                import pandas as pd
                closes = pd.DataFrame(spy_hist).set_index('date')['price'].sort_index()
                returns = closes.pct_change().dropna().tail(21)
                rv = float(returns.std() * (252 ** 0.5) * 100)
                spread = vix[0] - rv
                return round(spread, 2), datetime.now().date()
        except Exception as e:
            logger.warning(f"IV-RV spread failed: {e}")
        return None

    def compute_spy_tlt_corr(self) -> Optional[Tuple[float, date]]:
        """SPY-TLT 21d Korrelation via FMP historical."""
        if not self.fmp:
            return None
        try:
            spy_hist = self.fmp._get("historical-price-eod/light", {"symbol": "SPY", "from": (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')})
            tlt_hist = self.fmp._get("historical-price-eod/light", {"symbol": "TLT", "from": (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')})
            if spy_hist and tlt_hist and len(spy_hist) >= 15 and len(tlt_hist) >= 15:
                import pandas as pd
                spy_df = pd.DataFrame(spy_hist).set_index('date')['price'].sort_index().pct_change().dropna().tail(21)
                tlt_df = pd.DataFrame(tlt_hist).set_index('date')['price'].sort_index().pct_change().dropna().tail(21)
                # Align on common dates
                common = spy_df.index.intersection(tlt_df.index)
                if len(common) >= 15:
                    corr = float(spy_df[common].corr(tlt_df[common]))
                    return round(corr, 4), datetime.now().date()
        except Exception as e:
            logger.warning(f"SPY-TLT corr failed: {e}")
        return None

    def compute_cu_au_ratio(self) -> Optional[Tuple[float, date]]:
        """Copper/Gold Ratio * 1000."""
        cu = self._get_price('HGUSD', 'HG=F')
        au = self._get_price('GCUSD', 'GC=F')
        if cu and au and au[0] > 0:
            ratio = cu[0] / au[0] * 1000
            return round(ratio, 4), cu[1]
        return None

    def compute_hyg_tlt_ratio(self) -> Optional[Tuple[float, date]]:
        """HYG/TLT Ratio."""
        hyg = self._get_price('HYG', 'HYG')
        tlt = self._get_price('TLT', 'TLT')
        if hyg and tlt and tlt[0] > 0:
            ratio = hyg[0] / tlt[0]
            return round(ratio, 4), hyg[1]
        return None

    def compute_spy_metrics(self) -> Optional[Dict]:
        """SPY Intraday Range + Relative Volume via FMP."""
        if not self.fmp:
            return None
        try:
            # Get today's quote for intraday range
            quote = self.fmp._get("quote", {"symbol": "SPY"})
            if not quote:
                return None
            q = quote[0]
            price = q.get('price', 0)
            day_high = q.get('dayHigh', 0)
            day_low = q.get('dayLow', 0)
            volume = q.get('volume', 0)

            intraday_range = None
            if price > 0 and day_high > 0 and day_low > 0:
                intraday_range = (day_high - day_low) / price * 100

            # Get historical for relative volume
            rel_volume = None
            hist = self.fmp._get("historical-price-eod/light", {"symbol": "SPY", "from": (datetime.now() - timedelta(days=35)).strftime('%Y-%m-%d')})
            if hist and len(hist) >= 20 and volume > 0:
                import pandas as pd
                df = pd.DataFrame(hist)
                if 'volume' in df.columns:
                    avg_vol = float(df['volume'].tail(20).mean())
                    if avg_vol > 0:
                        rel_volume = volume / avg_vol

            ts = q.get('timestamp')
            if ts:
                d = datetime.fromtimestamp(ts).date()
            else:
                d = datetime.now().date()

            return {
                'spy_intraday_range': round(intraday_range, 4) if intraday_range else None,
                'spy_rel_volume': round(rel_volume, 4) if rel_volume else None,
                'date': d
            }
        except Exception as e:
            logger.warning(f"SPY metrics failed: {e}")
            return None

    def compute_wti_curve(self) -> Optional[Tuple[float, date, str]]:
        """WTI Contango/Backwardation — CL second month / CL front."""
        front = self._get_price('CLUSD', 'CL=F')
        if not front:
            return None
        second = self._get_price('CLUSD', 'CL=F')  # FMP only has front month
        # If only front available, skip curve calc
        if not second or front[0] == 0:
            return None
        today = datetime.now().date()
        roll_flag = "ROLL_PERIOD" if self._is_roll_period(today) else "OK"
        # Without second month, return 1.0 (flat)
        return 1.0, front[1], roll_flag

    def compute_brent_wti_spread(self) -> Optional[Tuple[float, date]]:
        """Brent - WTI Spread."""
        brent = self._get_price('BZUSD', 'BZ=F')
        wti = self._get_price('CLUSD', 'CL=F')
        if brent and wti:
            spread = brent[0] - wti[0]
            return round(spread, 2), brent[1]
        return None

    def compute_sofr_ff_spread(self, fred: FredFetcher) -> Optional[Tuple[float, date]]:
        """SOFR - Fed Funds Spread in bps."""
        sofr = fred.fetch_latest('SOFR', 14)
        dff = fred.fetch_latest('DFF', 14)
        if sofr and dff:
            spread_bps = (sofr[0] - dff[0]) * 100  # % -> bps
            return round(spread_bps, 1), sofr[1]
        return None

    def compute_cpi_yoy(self, fred: FredFetcher) -> Optional[Tuple[float, date]]:
        """CPI Year-over-Year % Change."""
        cpi_data = fred.fetch_series('CPIAUCSL', lookback_days=400)
        if cpi_data is not None and len(cpi_data) >= 13:
            latest = float(cpi_data.iloc[-1])
            year_ago = float(cpi_data.iloc[-13])
            if year_ago > 0:
                yoy = (latest / year_ago - 1) * 100
                return round(yoy, 2), cpi_data.index[-1].date()
        return None

    def compute_fedwatch(self, fred: FredFetcher) -> Optional[Tuple[float, date]]:
        """FedWatch implied cut probability."""
        dff = fred.fetch_latest('DFF', 14)
        zq = self._get_price('ZQ=F', 'ZQ=F')  # Fed Funds futures
        if dff and zq:
            current_rate = dff[0]
            implied_rate = 100 - zq[0]
            if implied_rate < current_rate:
                prob_cut = min(100, (current_rate - implied_rate) / 0.25 * 100)
            else:
                prob_cut = 0.0
            return round(prob_cut, 1), zq[1]
        return None

    @staticmethod
    def _get_wti_second_month_ticker(today: date) -> str:
        """Berechnet den Ticker des 2. Monats WTI Contract."""
        months = ['F','G','H','J','K','M','N','Q','U','V','X','Z']
        front_month = today.month + 1
        second_month = front_month + 1
        year = today.year
        if second_month > 12:
            second_month -= 12
            year += 1
        month_code = months[second_month - 1]
        year_code = str(year)[-2:]
        return f'CL{month_code}{year_code}.NYM'

    @staticmethod
    def _is_roll_period(today: date) -> bool:
        """Prueft ob wir in der WTI Roll-Periode sind."""
        return 15 <= today.day <= 22


# ═══════════════════════════════════════════════════════
# CALENDAR FIELDS
# ═══════════════════════════════════════════════════════

class CalendarFields:
    """Berechnet Kalender-basierte Felder aus config/event_calendar.json."""

    def __init__(self, calendar_path: str):
        with open(calendar_path) as f:
            self.calendar = json.load(f)

    def days_to_next(self, event_key: str, today: date = None) -> Optional[int]:
        """Tage bis zum naechsten Event."""
        today = today or datetime.now().date()
        dates = self.calendar.get(event_key, [])
        for d in sorted(dates):
            event_date = datetime.strptime(d, '%Y-%m-%d').date()
            if event_date >= today:
                return (event_date - today).days
        return None

    def days_to_opex(self, today: date = None) -> int:
        """Tage bis zum naechsten OpEx (3. Freitag des Monats)."""
        today = today or datetime.now().date()
        # Finde 3. Freitag dieses oder naechsten Monats
        for month_offset in range(0, 3):
            year = today.year
            month = today.month + month_offset
            if month > 12:
                month -= 12
                year += 1
            # Finde 3. Freitag
            first_day = date(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            if third_friday >= today:
                return (third_friday - today).days
        return 0

    def is_earnings_season(self, today: date = None) -> bool:
        """Prueft ob heute Earnings Season ist."""
        today = today or datetime.now().date()
        windows = self.calendar.get('EARNINGS_SEASON_WINDOWS_2026', [])
        for w in windows:
            start = datetime.strptime(w['start'], '%Y-%m-%d').date()
            end = datetime.strptime(w['end'], '%Y-%m-%d').date()
            if start <= today <= end:
                return True
        return False

    def compute_all(self, today: date = None) -> Dict[str, Any]:
        """Berechnet alle Kalender-Felder."""
        today = today or datetime.now().date()
        return {
            'days_to_fomc': self.days_to_next('FOMC_2026', today),
            'days_to_opex': self.days_to_opex(today),
            'days_to_cpi': self.days_to_next('CPI_2026', today),
            'days_to_nfp': self.days_to_next('NFP_2026', today),
            'days_to_ecb': self.days_to_next('ECB_2026', today),
            'days_to_boj': self.days_to_next('BOJ_2026', today),
            'earnings_season_flag': self.is_earnings_season(today),
        }


# ═══════════════════════════════════════════════════════
# MASTER FETCHER — ORCHESTRIERT ALLES
# ═══════════════════════════════════════════════════════

class MasterFetcher:
    """
    Orchestriert alle Fetcher. Implementiert 4-Stufen-Fallback.
    Output: Dict[field_name -> FetchResult]
    """

    def __init__(self, config_dir: str = "config"):
        # Load configs
        self.config_dir = config_dir
        with open(os.path.join(config_dir, 'field_registry.json')) as f:
            registry_data = json.load(f)
        self.registry = {f['name']: f for f in registry_data['fields']}

        # Init fetchers
        self.fred = FredFetcher(os.environ.get('FRED_API_KEY', ''))
        self.yf = YFinanceFetcher()
        self.fmp = FMPFetcher(os.environ.get('FMP_API_KEY', os.environ.get('EODHD_API_KEY', '')))
        self.scraper = ScrapeFetcher()
        self.llm = LLMFetcher(os.environ.get('ANTHROPIC_API_KEY', ''))
        self.cot = COTFetcher()
        self.v16_prices = V16PriceFetcher(self.fmp, self.yf)
        self.v16_liquidity = V16LiquidityFetcher(self.fred, self.yf, self.fmp)
        self.calc = CalculatedFields(self.yf, self.fmp)
        self.calendar = CalendarFields(os.path.join(config_dir, 'event_calendar.json'))

        # SP500 tickers
        sp500_path = os.path.join(config_dir, 'sp500_tickers.json')
        if os.path.exists(sp500_path):
            with open(sp500_path) as f:
                data = json.load(f)
            # Handle both raw list and {"tickers": [...]} format
            self.sp500_tickers = data.get('tickers', data) if isinstance(data, dict) else data
        else:
            self.sp500_tickers = []

    def fetch_all(self, cache: dict = None) -> Dict[str, FetchResult]:
        """
        Holt ALLE Felder. Returns Dict[field_name -> FetchResult].
        Zusaetzlich: v16_prices und v16_liquidity fuer Sheet-Befuellung.
        """
        cache = cache or {}
        results = {}
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("PHASE 1: DATEN HOLEN")
        logger.info("=" * 60)

        # ─── 1. V16 Sheet Preise (27 Assets) ───
        logger.info("Fetching V16 prices (27 assets)...")
        v16_prices = self.v16_prices.fetch_all()
        results['_v16_prices'] = FetchResult('_v16_prices', v16_prices, 'yfinance', 'API_PRIMARY')

        # ─── 2. V16 Liquidity Data ───
        logger.info("Fetching V16 liquidity data...")
        v16_liq = self.v16_liquidity.fetch_all()
        results['_v16_liquidity'] = FetchResult('_v16_liquidity', v16_liq, 'FRED+yfinance', 'API_PRIMARY')

        # ─── 3. FRED Batch (T1+T2 Macro Fields) ───
        logger.info("Fetching FRED batch...")
        fred_fields = {
            'spread_2y10y': 'T10Y2Y', 'spread_3m10y': 'T10Y3M',
            'real_10y_yield': 'DFII10', 'hy_oas': 'BAMLH0A0HYM2',
            'ig_oas': 'BAMLC0A4CBBB', 'nfci': 'NFCI', 'anfci': 'ANFCI',
            'discount_window': 'WDFOL', 'mmf_assets': 'WRMFNS',
            'initial_claims': 'ICSA', 'breakeven_5y5y': 'T5YIFR',
        }
        fred_batch = self.fred.fetch_batch(list(fred_fields.values()))
        for field_name, series_id in fred_fields.items():
            data = fred_batch.get(series_id)
            if data:
                results[field_name] = FetchResult(field_name, data[0], 'FRED', 'API_PRIMARY', data[1])
            else:
                results[field_name] = self._try_fallback(field_name, cache)

        # ─── 4. Net Liquidity (WALCL - TGA - RRP) — separate FRED fetch ───
        logger.info("Fetching Net Liquidity components...")
        liq_series = {'WALCL': 'WALCL', 'WTREGEN': 'WTREGEN', 'RRPONTSYD': 'RRPONTSYD'}
        liq_batch = self.fred.fetch_batch(list(liq_series.values()), lookback_days=30)
        walcl_r = liq_batch.get('WALCL')
        tga_r = liq_batch.get('WTREGEN')
        rrp_r = liq_batch.get('RRPONTSYD')
        if walcl_r and tga_r and rrp_r:
            net_liq = walcl_r[0] - tga_r[0] - rrp_r[0]
            results['net_liquidity'] = FetchResult('net_liquidity', net_liq, 'CALC', 'API_PRIMARY', walcl_r[1])
            results['walcl'] = FetchResult('walcl', walcl_r[0], 'FRED', 'API_PRIMARY', walcl_r[1])
            results['tga'] = FetchResult('tga', tga_r[0], 'FRED', 'API_PRIMARY', tga_r[1])
            results['rrp'] = FetchResult('rrp', rrp_r[0], 'FRED', 'API_PRIMARY', rrp_r[1])
        else:
            for f in ['net_liquidity', 'walcl', 'tga', 'rrp']:
                if f not in results:
                    results[f] = self._try_fallback(f, cache)

        # ─── 5. Helper FRED fields ───
        logger.info("Fetching helper FRED fields...")
        helpers = {'fed_funds_rate': 'DFF', 'sofr_rate': 'SOFR'}
        helper_batch = self.fred.fetch_batch(list(helpers.values()))
        for field_name, series_id in helpers.items():
            data = helper_batch.get(series_id)
            if data:
                results[field_name] = FetchResult(field_name, data[0], 'FRED', 'API_PRIMARY', data[1])
            else:
                results[field_name] = FetchResult(field_name, None, None, 'ALL_FAILED')

        # ─── 6. Market Fields (FMP primary, yfinance fallback) ───
        logger.info("Fetching market fields via FMP...")
        fmp_fields = {
            'vix': '^VIX', 'dxy': 'DX-Y.NYB', 'usdjpy': 'USDJPY',
            'usdcnh': 'USDCNH', 'move_index': '^MOVE',
        }
        # FMP batch
        fmp_quotes = self.fmp.fetch_batch_quotes(list(fmp_fields.values()))
        for field_name, fmp_sym in fmp_fields.items():
            data = fmp_quotes.get(fmp_sym)
            if data:
                results[field_name] = FetchResult(field_name, data[0], 'FMP', 'API_PRIMARY', data[1])
            else:
                # yfinance fallback
                yf_map = {
                    'vix': '^VIX', 'dxy': 'DX-Y.NYB', 'usdjpy': 'USDJPY=X',
                    'usdcnh': 'USDCNH=X', 'move_index': '^MOVE',
                }
                yf_data = self.yf.fetch_close(yf_map.get(field_name, fmp_sym), '5d')
                if yf_data:
                    results[field_name] = FetchResult(field_name, yf_data[0], 'yfinance', 'API_SECONDARY', yf_data[1])
                else:
                    results[field_name] = self._try_fallback(field_name, cache)

        # China 10Y via FMP
        china_data = self.fmp.fetch_quote('CN10Y')
        if china_data:
            results['china_10y'] = FetchResult('china_10y', china_data[0], 'FMP', 'API_PRIMARY', china_data[1])
        else:
            results['china_10y'] = self._try_fallback('china_10y', cache)

        # ─── 7. Calculated Fields ───
        logger.info("Computing calculated fields...")

        vix_term = self.calc.compute_vix_term_ratio()
        results['vix_term_ratio'] = FetchResult('vix_term_ratio', vix_term[0] if vix_term else None,
                                                 'CALC', 'API_PRIMARY' if vix_term else 'ALL_FAILED',
                                                 vix_term[1] if vix_term else None)

        # iv_rv_spread and spy_tlt_corr need 30d history — skip if yfinance rate-limited
        for calc_field, calc_fn in [
            ('iv_rv_spread', self.calc.compute_iv_rv_spread),
            ('spy_tlt_corr_21d', self.calc.compute_spy_tlt_corr),
        ]:
            try:
                val = calc_fn()
                results[calc_field] = FetchResult(calc_field, val[0] if val else None,
                                                   'CALC', 'API_PRIMARY' if val else 'ALL_FAILED',
                                                   val[1] if val else None)
            except Exception:
                results[calc_field] = self._try_fallback(calc_field, cache)

        cu_au = self.calc.compute_cu_au_ratio()
        results['cu_au_ratio'] = FetchResult('cu_au_ratio', cu_au[0] if cu_au else None,
                                              'CALC', 'API_PRIMARY' if cu_au else 'ALL_FAILED',
                                              cu_au[1] if cu_au else None)

        hyg_tlt = self.calc.compute_hyg_tlt_ratio()
        results['hyg_tlt_ratio'] = FetchResult('hyg_tlt_ratio', hyg_tlt[0] if hyg_tlt else None,
                                                'CALC', 'API_PRIMARY' if hyg_tlt else 'ALL_FAILED',
                                                hyg_tlt[1] if hyg_tlt else None)

        try:
            spy_metrics = self.calc.compute_spy_metrics()
        except Exception:
            spy_metrics = None
        if spy_metrics:
            results['spy_intraday_range'] = FetchResult('spy_intraday_range', spy_metrics['spy_intraday_range'],
                                                         'CALC', 'API_PRIMARY', spy_metrics['date'])
            results['spy_rel_volume'] = FetchResult('spy_rel_volume', spy_metrics['spy_rel_volume'],
                                                     'CALC', 'API_PRIMARY', spy_metrics['date'])
        else:
            results['spy_intraday_range'] = FetchResult('spy_intraday_range', None, None, 'ALL_FAILED')
            results['spy_rel_volume'] = FetchResult('spy_rel_volume', None, None, 'ALL_FAILED')

        wti = self.calc.compute_wti_curve()
        results['wti_curve'] = FetchResult('wti_curve', wti[0] if wti else None,
                                            'CALC', 'API_PRIMARY' if wti else 'ALL_FAILED',
                                            wti[1] if wti else None,
                                            {'roll_flag': wti[2]} if wti else {})

        brent_wti = self.calc.compute_brent_wti_spread()
        results['brent_wti_spread'] = FetchResult('brent_wti_spread', brent_wti[0] if brent_wti else None,
                                                    'CALC', 'API_PRIMARY' if brent_wti else 'ALL_FAILED',
                                                    brent_wti[1] if brent_wti else None)

        sofr_ff = self.calc.compute_sofr_ff_spread(self.fred)
        results['sofr_ff_spread'] = FetchResult('sofr_ff_spread', sofr_ff[0] if sofr_ff else None,
                                                  'CALC', 'API_PRIMARY' if sofr_ff else 'ALL_FAILED',
                                                  sofr_ff[1] if sofr_ff else None)

        cpi = self.calc.compute_cpi_yoy(self.fred)
        if cpi:
            results['cpi_yoy'] = FetchResult('cpi_yoy', cpi[0], 'CALC', 'API_PRIMARY', cpi[1])
        else:
            results['cpi_yoy'] = self._try_fallback('cpi_yoy', cache)

        fedwatch = self.calc.compute_fedwatch(self.fred)
        results['fedwatch_cut_prob'] = FetchResult('fedwatch_cut_prob', fedwatch[0] if fedwatch else None,
                                                    'CALC', 'API_PRIMARY' if fedwatch else 'ALL_FAILED',
                                                    fedwatch[1] if fedwatch else None)

        # ─── 8. Breadth (S&P 500) — DISABLED until FMP historical API ───
        logger.info("Breadth: skipped (requires FMP historical API — coming soon)")
        for f in ['pct_above_200dma', 'nh_nl', 'trin']:
            results[f] = self._try_fallback(f, cache)

        # ─── 9. Sentiment Scraping ───
        logger.info("Scraping sentiment sources...")
        for field_name, scrape_fn in [
            ('naaim_exposure', self.scraper.fetch_naaim),
            ('aaii_bull_bear', self.scraper.fetch_aaii),
            ('gdpnow', self.scraper.fetch_gdpnow),
            ('baltic_dry', self.scraper.fetch_baltic_dry),
        ]:
            data = scrape_fn()
            if data:
                results[field_name] = FetchResult(field_name, data[0], 'SCRAPE', 'SCRAPE', data[1])
            else:
                results[field_name] = self._try_fallback(field_name, cache)

        # CBOE P/C Ratio
        pc = self.scraper.fetch_cboe_pc_ratio()
        if pc:
            results['pc_ratio_equity'] = FetchResult('pc_ratio_equity', pc[0], 'CBOE', 'SCRAPE', pc[1])
        else:
            results['pc_ratio_equity'] = self._try_fallback('pc_ratio_equity', cache)

        # ─── 10. COT Positioning ───
        logger.info("Fetching COT data...")
        cot_es = self.cot.fetch_leveraged_net('13874A')
        if cot_es:
            results['cot_es_leveraged'] = FetchResult('cot_es_leveraged', cot_es[0], 'CFTC', 'API_PRIMARY', cot_es[1])
        else:
            results['cot_es_leveraged'] = self._try_fallback('cot_es_leveraged', cache)

        cot_zn = self.cot.fetch_leveraged_net('043602')
        if cot_zn:
            results['cot_zn_leveraged'] = FetchResult('cot_zn_leveraged', cot_zn[0], 'CFTC', 'API_PRIMARY', cot_zn[1])
        else:
            results['cot_zn_leveraged'] = self._try_fallback('cot_zn_leveraged', cache)

        # ─── 11. VIX Call/Put Volume ───
        results['vix_call_put_vol'] = self._try_fallback('vix_call_put_vol', cache)

        # ─── 12. Calendar Fields ───
        logger.info("Computing calendar fields...")
        cal = self.calendar.compute_all()
        for field_name, value in cal.items():
            results[field_name] = FetchResult(field_name, value, 'CALC', 'API_PRIMARY')

        # ─── 13. Context T3 Fields ───
        # ISM Manufacturing PMI — No FRED series available, use LLM
        results['ism_mfg'] = self._try_fallback('ism_mfg', cache)

        # GPR Index — LLM fallback
        results['gpr_index'] = self._try_fallback('gpr_index', cache)

        # SPY Forward P/E — LLM fallback
        results['spy_fwd_pe'] = self._try_fallback('spy_fwd_pe', cache)

        # VIX Call/Put Vol — LLM fallback (already set earlier, ensure present)
        if 'vix_call_put_vol' not in results:
            results['vix_call_put_vol'] = self._try_fallback('vix_call_put_vol', cache)

        # Helper: JGB 10Y — try multiple FMP symbols
        jgb = self.fmp.fetch_quote('JP10Y')
        if not jgb:
            jgb = self.fmp.fetch_quote('^TNX')
        results['jgb_10y'] = FetchResult('jgb_10y', jgb[0] if jgb else None,
                                          'FMP', 'API_PRIMARY' if jgb else 'ALL_FAILED',
                                          jgb[1] if jgb else None)

        # Insider ratio — LLM fallback
        results['insider_ratio'] = self._try_fallback('insider_ratio', cache)

        # ─── DONE ───
        elapsed = time.time() - start_time
        ok = sum(1 for r in results.values() if r.success)
        total = len(results)
        logger.info(f"PHASE 1 COMPLETE: {ok}/{total} fields OK in {elapsed:.1f}s")
        logger.info(f"  FRED calls: {self.fred.call_count}, yfinance calls: {self.yf.call_count}, LLM calls: {self.llm.call_count}")

        return results

    def _try_fallback(self, field_name: str, cache: dict) -> FetchResult:
        """4-Stufen-Fallback fuer ein Feld."""
        field_cfg = self.registry.get(field_name, {})

        # Stufe 4: LLM Fallback
        if field_cfg.get('source_llm_eligible') and field_cfg.get('source_llm_config'):
            data = self.llm.fetch(field_name, field_cfg['source_llm_config'])
            if data:
                return FetchResult(field_name, data[0], 'LLM', 'LLM_FALLBACK', data[1])

        # Stufe 5: Stale Cache
        if field_name in cache:
            cached = cache[field_name]
            logger.info(f"Using stale cache for {field_name}")
            return FetchResult(field_name, cached.get('value'), cached.get('source'),
                             'STALE_CACHE', cached.get('date'))

        return FetchResult(field_name, None, None, 'ALL_FAILED')
