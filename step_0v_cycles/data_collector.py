"""
Cycles Circle — Data Collector
Baldur Creek Capital | Step 0v (V3.4 FINAL)

Two-layer architecture:
  Layer 1 (Local JSON): Raw historical time series, incrementally updated.
  Layer 2 (Cycles Sheet): Only computed results (written by main.py).

Sources: V16 Sheet, DW Sheet, FRED API (14 series). No FMP, no EODHD.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, date

logger = logging.getLogger("cycles.data_collector")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .config import (
    V16_SHEET_ID, DW_SHEET_ID,
    V16_DATA_PRICES_TAB, V16_DATA_LIQUIDITY_TAB,
    V16_CYCLES_HOWELL_TAB, V16_CALC_MACRO_STATE_TAB,
    DW_RAW_MARKET_TAB, DW_MARKET_INDICATORS,
    FRED_BASE_URL, FRED_SERIES,
    FRED_OBSERVATION_COUNT, FRED_BACKFILL_START, FRED_BACKFILL_LIMIT,
    PRICE_TICKERS, PRICE_ROW_LIMIT, V16_BACKFILL_ROW_LIMIT,
    DATA_DIR,
)

# ---------------------------------------------------------------------------
# Local history file paths
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(DATA_DIR, "raw")

HISTORY_FILES = {
    "prices":        os.path.join(RAW_DIR, "prices_history.json"),
    "liquidity":     os.path.join(RAW_DIR, "liquidity_history.json"),
    "howell":        os.path.join(RAW_DIR, "howell_history.json"),
    "macro_state":   os.path.join(RAW_DIR, "macro_state_history.json"),
    "fred":          os.path.join(RAW_DIR, "fred_history.json"),
    "dw_indicators": os.path.join(RAW_DIR, "dw_indicators_history.json"),
}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_history(key):
    path = HISTORY_FILES.get(key)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Load {key} failed: {e}")
        return None


def _save_history(key, data):
    path = HISTORY_FILES.get(key)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False, default=str)
        logger.info(f"Saved {key} history")
    except Exception as e:
        logger.error(f"Save {key} failed: {e}")


def _needs_backfill(key):
    return _load_history(key) is None


def _merge_ts(existing, new_data, dk="date"):
    if not existing:
        existing = []
    if not new_data:
        return existing
    dates = {e[dk] for e in existing if dk in e}
    added = 0
    for e in new_data:
        d = e.get(dk)
        if d and d not in dates:
            existing.append(e)
            dates.add(d)
            added += 1
    existing.sort(key=lambda x: x.get(dk, ""), reverse=True)
    if added:
        logger.info(f"  +{added} new (total {len(existing)})")
    return existing


def _merge_prices(existing, new):
    if not existing:
        existing = {}
    for t, s in (new or {}).items():
        existing[t] = _merge_ts(existing.get(t, []), s)
    return existing


def _merge_fred(existing, new):
    if not existing:
        existing = {}
    for k, s in (new or {}).items():
        existing[k] = _merge_ts(existing.get(k, []), s)
    return existing


def _merge_dw(existing, snap):
    if not existing:
        existing = []
    if not snap:
        return existing
    today = date.today().isoformat()
    for e in existing:
        if e.get("date") == today:
            e["indicators"] = snap
            return existing
    existing.append({"date": today, "indicators": snap})
    existing.sort(key=lambda x: x.get("date", ""), reverse=True)
    return existing


# ---------------------------------------------------------------------------
# Google Sheets service
# ---------------------------------------------------------------------------

def _get_sheets_service():
    try:
        from googleapiclient.discovery import build
        from google.oauth2.service_account import Credentials
    except ImportError:
        logger.error("googleapiclient not installed")
        return None

    sa_key = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
    if not sa_key:
        logger.error("No GCP_SA_KEY or GOOGLE_CREDENTIALS")
        return None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(sa_key)
            tmp_path = f.name
        creds = Credentials.from_service_account_file(
            tmp_path,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        os.unlink(tmp_path)
        return build("sheets", "v4", credentials=creds, cache_discovery=False)
    except Exception as e:
        logger.error(f"Sheets auth failed: {e}")
        return None


def _read_sheet(service, sheet_id, range_str):
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=range_str,
        ).execute()
        return result.get("values", [])
    except Exception as e:
        logger.error(f"Sheet read failed [{range_str}]: {e}")
        return []


# ---------------------------------------------------------------------------
# Type conversions
# ---------------------------------------------------------------------------

def _sf(val):
    """Safe float — handles German comma notation."""
    if val is None or val == "" or val == "—" or val == "---":
        return None
    try:
        o = str(val).strip()
        if "." in o and "," in o:
            s = o.replace(".", "").replace(",", ".")
        elif "," in o and "." not in o:
            s = o.replace(",", ".")
        else:
            s = o.replace(",", "")
        return float(s)
    except (ValueError, TypeError):
        return None


def _sd(val):
    """Safe date parse."""
    if not val:
        return None
    s = str(val).strip()
    if " " in s:
        s = s.split(" ")[0]
    if s.endswith(".0"):
        s = s[:-2]
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 1. V16 Prices
# ---------------------------------------------------------------------------

def _collect_v16_prices(service, backfill=False):
    row_limit = V16_BACKFILL_ROW_LIMIT if backfill else PRICE_ROW_LIMIT
    label = "BACKFILL" if backfill else "INCREMENT"
    logger.info(f"V16 Prices ({label}, max {row_limit})...")

    rows = _read_sheet(service, V16_SHEET_ID, f"{V16_DATA_PRICES_TAB}!A1:AB{row_limit + 1}")
    if len(rows) < 3:
        return {}

    headers = [str(h).strip() for h in rows[0]]
    col_map = {t: headers.index(t) for t in PRICE_TICKERS if t in headers}
    prices = {t: [] for t in PRICE_TICKERS}

    for row in rows[1:]:
        if not row or len(row) < 2:
            continue
        d = _sd(row[0])
        if not d:
            continue
        for t, ci in col_map.items():
            v = _sf(row[ci] if ci < len(row) else None)
            if v and v > 0:
                prices[t].append({"date": d, "price": v})

    found = {t: len(v) for t, v in prices.items() if v}
    logger.info(f"  {len(found)} tickers, {sum(found.values())} points")
    return prices


# ---------------------------------------------------------------------------
# 2. V16 Liquidity
# ---------------------------------------------------------------------------

def _collect_v16_liquidity(service, backfill=False):
    row_limit = V16_BACKFILL_ROW_LIMIT if backfill else PRICE_ROW_LIMIT
    logger.info(f"V16 Liquidity ({'BACKFILL' if backfill else 'INCREMENT'})...")

    rows = _read_sheet(service, V16_SHEET_ID, f"{V16_DATA_LIQUIDITY_TAB}!A1:O{row_limit + 2}")
    if len(rows) < 4:
        return []

    headers = [str(h).strip() for h in rows[0]]
    data = []
    for row in rows[2:]:
        if not row:
            continue
        d = _sd(row[0])
        if not d:
            continue
        entry = {"date": d}
        for i, h in enumerate(headers):
            if i == 0:
                continue
            entry[h] = _sf(row[i] if i < len(row) else None)
        data.append(entry)

    logger.info(f"  {len(data)} rows")
    return data


# ---------------------------------------------------------------------------
# 3. V16 Howell
# ---------------------------------------------------------------------------

def _collect_v16_howell(service, backfill=False):
    row_limit = V16_BACKFILL_ROW_LIMIT if backfill else PRICE_ROW_LIMIT
    logger.info(f"V16 Howell ({'BACKFILL' if backfill else 'INCREMENT'})...")

    rows = _read_sheet(service, V16_SHEET_ID, f"{V16_CYCLES_HOWELL_TAB}!A1:O{row_limit + 2}")
    if len(rows) < 4:
        return []

    headers = [str(h).strip() for h in rows[0]]
    text_cols = {"Phase_Name", "Cycle_Interpretation"}
    data = []
    for row in rows[2:]:
        if not row:
            continue
        d = _sd(row[0])
        if not d:
            continue
        entry = {"date": d}
        for i, h in enumerate(headers):
            if i == 0:
                continue
            val = row[i] if i < len(row) else None
            if h in text_cols:
                entry[h] = str(val).strip() if val else None
            else:
                entry[h] = _sf(val)
        data.append(entry)

    logger.info(f"  {len(data)} rows")
    return data


# ---------------------------------------------------------------------------
# 4. V16 Macro State
# ---------------------------------------------------------------------------

def _collect_v16_macro_state(service, backfill=False):
    row_limit = V16_BACKFILL_ROW_LIMIT if backfill else 62
    logger.info(f"V16 Macro State ({'BACKFILL' if backfill else 'INCREMENT'})...")

    rows = _read_sheet(service, V16_SHEET_ID, f"{V16_CALC_MACRO_STATE_TAB}!A1:O{row_limit + 2}")
    if len(rows) < 4:
        return []

    headers = [str(h).strip() for h in rows[0]]
    state_col = headers.index("Macro_State_Name") if "Macro_State_Name" in headers else 9
    conf_col = headers.index("State_Confidence") if "State_Confidence" in headers else 10

    data = []
    for row in rows[2:]:
        if not row or len(row) <= state_col:
            continue
        d = _sd(row[0])
        state = str(row[state_col]).strip() if state_col < len(row) and row[state_col] else None
        conf = _sf(row[conf_col] if conf_col < len(row) else None)
        if d and state:
            data.append({"date": d, "state": state, "confidence": conf})

    logger.info(f"  {len(data)} rows, current: {data[0]['state'] if data else 'UNKNOWN'}")
    return data


# ---------------------------------------------------------------------------
# 5. DW Indicators (current values)
# ---------------------------------------------------------------------------

def _collect_dw_indicators(service):
    logger.info("DW Indicators...")
    result = {}

    rows = _read_sheet(service, DW_SHEET_ID, f"{DW_RAW_MARKET_TAB}!A1:J200")
    if len(rows) < 3:
        return result

    headers = rows[1]
    ic = headers.index("INDICATOR") if "INDICATOR" in headers else 1
    vc = headers.index("VALUE") if "VALUE" in headers else 3

    for row in rows[2:]:
        if not row or len(row) <= max(ic, vc):
            continue
        name = str(row[ic]).strip().upper() if ic < len(row) else ""
        for our_key, sheet_name in DW_MARKET_INDICATORS.items():
            if name == sheet_name.upper():
                result[our_key] = _sf(row[vc] if vc < len(row) else None)

    logger.info(f"  {result}")
    return result


# ---------------------------------------------------------------------------
# 6. FRED API (all 14 series)
# ---------------------------------------------------------------------------

def _collect_fred(backfill=False):
    if not HAS_REQUESTS:
        logger.error("requests not installed")
        return {}

    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        logger.error("FRED_API_KEY not set")
        return {}

    label = "BACKFILL" if backfill else "INCREMENT"
    logger.info(f"FRED ({label}, {len(FRED_SERIES)} series)...")
    result = {}

    for key, series_id in FRED_SERIES.items():
        try:
            params = {
                "series_id": series_id,
                "api_key": fred_key,
                "file_type": "json",
                "sort_order": "desc",
            }
            if backfill:
                params["observation_start"] = FRED_BACKFILL_START
                params["limit"] = FRED_BACKFILL_LIMIT
            else:
                params["limit"] = FRED_OBSERVATION_COUNT

            resp = requests.get(f"{FRED_BASE_URL}/series/observations",
                                params=params, timeout=20)
            resp.raise_for_status()

            series = []
            for o in resp.json().get("observations", []):
                d, v = o.get("date"), o.get("value", ".")
                if v == "." or not d:
                    continue
                try:
                    series.append({"date": d, "value": float(v)})
                except (ValueError, TypeError):
                    continue

            result[key] = series
            logger.info(f"  {key} ({series_id}): {len(series)} obs")

        except Exception as e:
            logger.warning(f"  {key} ({series_id}) failed: {e}")
            result[key] = []

    return result


# ---------------------------------------------------------------------------
# 7. latest.json
# ---------------------------------------------------------------------------

def _load_latest_json():
    paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "dashboard", "latest.json"),
        "data/dashboard/latest.json",
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Load {path} failed: {e}")
    return None


# ---------------------------------------------------------------------------
# MAIN: Collect All Data
# ---------------------------------------------------------------------------

def collect_all_data(force_backfill=False):
    """
    Collect all data with incremental persistence.

    First run (or force_backfill): fetches full 20y history.
    Subsequent runs: loads existing + fetches recent + merges.
    """
    logger.info("=" * 60)
    logger.info("CYCLES DATA COLLECTION START")
    logger.info("=" * 60)

    needs_bf = force_backfill or _needs_backfill("prices") or _needs_backfill("fred")
    if needs_bf:
        logger.info("*** BACKFILL MODE ***")
    else:
        logger.info("Incremental mode")

    service = _get_sheets_service()
    if not service:
        logger.error("No Sheets service — aborting")
        return None

    # V16 Prices
    existing = _load_history("prices") or {}
    new = _collect_v16_prices(service, backfill=needs_bf)
    merged_prices = _merge_prices(existing, new)
    _save_history("prices", merged_prices)

    # V16 Liquidity
    existing = _load_history("liquidity") or []
    new = _collect_v16_liquidity(service, backfill=needs_bf)
    merged_liq = _merge_ts(existing, new)
    _save_history("liquidity", merged_liq)

    # V16 Howell
    existing = _load_history("howell") or []
    new = _collect_v16_howell(service, backfill=needs_bf)
    merged_howell = _merge_ts(existing, new)
    _save_history("howell", merged_howell)

    # V16 Macro State
    existing = _load_history("macro_state") or []
    new = _collect_v16_macro_state(service, backfill=needs_bf)
    merged_state = _merge_ts(existing, new)
    _save_history("macro_state", merged_state)

    # DW Indicators
    existing = _load_history("dw_indicators") or []
    new_dw = _collect_dw_indicators(service)
    merged_dw = _merge_dw(existing, new_dw)
    _save_history("dw_indicators", merged_dw)

    # FRED
    existing = _load_history("fred") or {}
    new = _collect_fred(backfill=needs_bf)
    merged_fred = _merge_fred(existing, new)
    _save_history("fred", merged_fred)

    # latest.json
    latest = _load_latest_json()

    # Current regime
    current_regime = "UNKNOWN"
    regime_confidence = 0
    if merged_state:
        current_regime = merged_state[0].get("state", "UNKNOWN")
        regime_confidence = merged_state[0].get("confidence", 0)

    # Current DW values
    current_dw = merged_dw[0].get("indicators", {}) if merged_dw else {}

    result = {
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "backfill_mode": needs_bf,
        "prices": merged_prices,
        "liquidity": merged_liq,
        "howell": merged_howell,
        "macro_state": merged_state,
        "fred": merged_fred,
        "dw_indicators_history": merged_dw,
        "current_regime": current_regime,
        "regime_confidence": regime_confidence,
        "current_dw": current_dw,
        "latest_json": latest,
    }

    # Summary
    n_prices = sum(len(v) for v in merged_prices.values())
    n_fred = sum(len(v) for v in merged_fred.values())
    logger.info("=" * 60)
    logger.info(f"COLLECTION COMPLETE ({'BACKFILL' if needs_bf else 'INCREMENT'}):")
    logger.info(f"  Prices: {n_prices} pts")
    logger.info(f"  Liquidity: {len(merged_liq)} | Howell: {len(merged_howell)} | State: {len(merged_state)}")
    logger.info(f"  FRED: {n_fred} obs ({len([k for k,v in merged_fred.items() if v])}/{len(FRED_SERIES)} series)")
    logger.info(f"  Regime: {current_regime}")
    logger.info("=" * 60)

    return result
