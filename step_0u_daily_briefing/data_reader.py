"""
Daily Briefing System — Data Reader
Baldur Creek Capital | Step 0u
Based on: DAILY_BRIEFING_SYSTEM_SPEC_TEIL1.md §3.1

Reads all data sources and builds the two dicts that composite.py needs:
  - indicator_values: raw market indicator values
  - pipeline_data: pipeline context flags and scores

Data Sources:
  1. latest.json (from Drive CURRENT or local) — V16, IC, Risk, CIO, Exec, etc.
  2. DW Sheet RAW_MARKET tab — L2/L4/L5/L7 raw indicator values
  3. DW Sheet RAW_MACRO tab — L6/L8 raw indicator values
"""

import json
import logging
import os
from datetime import date

logger = logging.getLogger("data_reader")


# ---------------------------------------------------------------------------
# Helper: Read latest.json
# ---------------------------------------------------------------------------

def load_latest_json(path="data/dashboard/latest.json"):
    """Load latest.json from local path. Fallback to CURRENT Drive folder."""
    if not os.path.exists(path):
        # Try pipeline root
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "dashboard", "latest.json")
        if os.path.exists(alt):
            path = alt
        else:
            logger.error(f"latest.json not found at {path} or {alt}")
            return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load latest.json: {e}")
        return None


# ---------------------------------------------------------------------------
# Helper: Read DW Sheet tabs via Google Sheets API
# ---------------------------------------------------------------------------

def read_dw_sheet_tab(tab_name, creds_json=None):
    """
    Read a tab from the DW Sheet and return as list of dicts (header row = keys).

    Args:
        tab_name: e.g. "RAW_MARKET" or "RAW_MACRO"
        creds_json: path to GCP SA key JSON, or None to use env var

    Returns:
        list of dicts, one per row, or empty list on failure.
    """
    from .config import DW_SHEET_ID

    try:
        from googleapiclient.discovery import build
        from google.oauth2.service_account import Credentials
    except ImportError:
        logger.error("googleapiclient not available — cannot read DW Sheet")
        return []

    try:
        # Credentials
        if creds_json and os.path.exists(creds_json):
            creds = Credentials.from_service_account_file(
                creds_json,
                scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
            )
        else:
            # Try env var (GitHub Actions pattern)
            sa_key = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
            if sa_key:
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    f.write(sa_key)
                    tmp_path = f.name
                creds = Credentials.from_service_account_file(
                    tmp_path,
                    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
                )
                os.unlink(tmp_path)
            else:
                logger.error("No GCP credentials found for Sheets API")
                return []

        service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        result = service.spreadsheets().values().get(
            spreadsheetId=DW_SHEET_ID,
            range=f"{tab_name}!A:Z",
        ).execute()

        rows = result.get("values", [])
        if len(rows) < 3:
            logger.warning(f"DW Sheet {tab_name}: no data rows")
            return []

        # Row 0 is a title row (e.g. "█ RAW MARKET DATA — ..."),
        # Row 1 is the actual header (DATE, INDICATOR, LAYER, VALUE, ...),
        # Row 2+ is data.
        headers = rows[1]
        data = []
        for row in rows[2:]:
            entry = {}
            for i, h in enumerate(headers):
                entry[h] = row[i] if i < len(row) else None
            data.append(entry)

        logger.info(f"DW Sheet {tab_name}: {len(data)} rows read")
        return data

    except Exception as e:
        logger.error(f"DW Sheet {tab_name} read failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Parse DW Sheet raw data into indicator_values dict
# ---------------------------------------------------------------------------

def _safe_float(val):
    """Convert string/None to float, return None on failure."""
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def parse_raw_market_indicators(raw_market_rows, raw_macro_rows):
    """
    Parse RAW_MARKET and RAW_MACRO tab rows into a flat indicator_values dict.

    The DW Sheet RAW_MARKET/RAW_MACRO tabs have columns:
      INDICATOR, VALUE, DELTA_1D, DELTA_5D, DELTA_21D, PCTL_1Y, ZSCORE_2Y,
      DIRECTION, FRESHNESS, CONFIDENCE, ANOMALY_FLAG, SOURCE, TIMING_CLASS

    We extract the VALUE column for each indicator we need.
    """
    # Build lookup: indicator_name -> row
    lookup = {}
    for row in raw_market_rows + raw_macro_rows:
        name = (row.get("INDICATOR") or row.get("indicator") or "").strip().upper()
        if name:
            lookup[name] = row

    # Debug: log what we found
    if raw_market_rows:
        logger.info(f"RAW_MARKET first row keys: {list(raw_market_rows[0].keys())}")
        logger.info(f"RAW_MARKET first row: {raw_market_rows[0]}")
    if raw_macro_rows:
        logger.info(f"RAW_MACRO first row keys: {list(raw_macro_rows[0].keys())}")
    logger.info(f"Lookup keys ({len(lookup)}): {list(lookup.keys())[:15]}...")

    indicators = {}

    # --- Map DW Sheet indicator names to our keys ---
    # These mappings depend on what the Layer Collectors write.
    # We try multiple possible names for each indicator.

    mapping = {
        # --- RAW_MARKET (L2/L4/L5/L7) ---
        "HY_OAS":       ["HY_OAS_SPREAD"],
        "VIX":          ["VIX_LEVEL"],
        "VIX_TERM":     ["VIX_TERM_STRUCTURE"],   # L2 version (direct ratio)
        "VIX_TERM_L5":  ["VIX_TERM_STRUCTURE"],   # L5 also has one — dedup via lookup
        "MOVE":         ["MOVE_INDEX"],
        "PUT_CALL":     ["PUT_CALL_RATIO"],
        "BREADTH":      ["INSIDER_BUY_SELL"],      # Proxy — no direct A/D in sheet
        "CNN_FG":       ["CNN_FEAR_GREED"],
        "AAII_BULL":    ["AAII_BULL_PCTL"],
        "AAII_BEAR":    ["AAII_BEAR_PCTL"],
        "MARGIN_DEBT":  ["MARGIN_DEBT_YOY_CHG"],
        # L4
        "COT_SP500":    ["COT_SP500_COMM_NET"],
        "COT_GOLD":     ["COT_GOLD_COMM_NET"],
        "COT_TREASURY": ["COT_TREASURY_COMM_NET"],
        "FUND_FLOWS":   ["FUND_FLOWS_EQUITY"],
        "CRYPTO_FUND":  ["CRYPTO_FUNDING_RATE"],
        "OPTIONS_GEX":  ["OPTIONS_GEX"],
        # L5
        "RESERVE_DRAIN":["RESERVE_DRAIN_RATE"],
        "SOFR_FFR":     ["SOFR_FFR_SPREAD"],
        "FIN_STRESS":   ["FIN_STRESS_INDEX"],
        "RRP":          ["ON_RRP_USAGE"],
        "SPY_CONC":     ["SPY_CONCENTRATION"],
        "LIQUIDITY_AMI":["LIQUIDITY_AMIHUD"],
        "PAIRWISE_CORR":["AVG_PAIRWISE_CORR"],
        # L7
        "BOND_EQ_CORR": ["BOND_EQUITY_CORR_60D"],
        "GOLD_RET_20D": ["GOLD_RETURN_20D"],
        "DXY_RET_20D":  ["DXY_RETURN_20D"],
        "2Y10Y":        ["YIELD_CURVE_10Y2Y"],
        "REAL_YIELD_TREND": ["REAL_YIELD_10Y_TREND"],
        # --- RAW_MACRO (L6/L8) ---
        "GPR":          ["GPR_INDEX"],
        "OIL_VOL":      ["OIL_VOLATILITY_20D"],
        "BALTIC_DRY":   ["BALTIC_DRY_INDEX"],
        "EM_FX_STRESS": ["EM_FX_STRESS_BASKET"],
        "HOWELL_CYCLE": ["HOWELL_CYCLE_POS"],
        "OPEX_PROX":    ["OPEX_PROXIMITY"],
    }

    for our_key, possible_names in mapping.items():
        for name in possible_names:
            row = lookup.get(name)
            if row:
                val = _safe_float(row.get("VALUE") or row.get("value"))
                if val is not None:
                    indicators[our_key] = val
                    # Also grab prev_7d if available
                    p7 = _safe_float(row.get("PREV_7D") or row.get("prev_7d"))
                    if p7 is not None:
                        indicators[f"{our_key}_PREV_7D"] = p7
                    break

    # --- Derived indicators ---

    # VIX Term Structure: already a direct ratio in the sheet (VIX_TERM_STRUCTURE)
    # No need to compute VIX / VIX3M — it's already there as VIX_TERM

    # Net Liquidity: RESERVE_DRAIN_RATE is a proxy ($B 4-week),
    # ON_RRP_USAGE is the RRP level. Full Net Liq needs WALCL+TGA from FRED
    # which aren't in the DW Sheet. Use RESERVE_DRAIN as proxy.
    reserve_drain = indicators.get("RESERVE_DRAIN")
    if reserve_drain is not None:
        # RESERVE_DRAIN_RATE is in $B, positive = drain. Convert to rough NET_LIQ proxy.
        # Higher drain = lower liquidity = worse
        indicators["NET_LIQ"] = reserve_drain * 1e9  # Scale for normalization

    # 2Y10Y: YIELD_CURVE_10Y2Y is in bps in the sheet, convert to pct for normalization
    y2y10 = indicators.get("2Y10Y")
    if y2y10 is not None:
        indicators["2Y10Y"] = y2y10 / 100.0  # bps → pct

    # CU_AU ratio: not directly in sheet. Could derive from GOLD_RETURN_20D
    # and COPPER_SMA50_TREND but that's text. Skip for now — will be MISSING.

    # HY OAS z-score: not in sheet, skip — warning trigger won't fire without it

    logger.info(f"Parsed {len(indicators)} indicator values from DW Sheet")
    return indicators


# ---------------------------------------------------------------------------
# Extract pipeline_data from latest.json
# ---------------------------------------------------------------------------

def extract_pipeline_data(latest):
    """
    Extract pipeline context dict from latest.json for the composite engine.

    Returns dict with all keys that compute_composite_scores() expects.
    """
    if latest is None:
        return _empty_pipeline_data()

    pd = {}

    # --- V16 ---
    v16 = latest.get("v16", {})
    pd["v16_regime"] = v16.get("regime", "UNKNOWN")
    pd["v16_drawdown"] = v16.get("current_drawdown", 0.0)
    pd["v16_dd_protect"] = v16.get("dd_protect_status", "INACTIVE")
    pd["v16_weights"] = v16.get("current_weights", {})
    pd["v16_top5"] = v16.get("top_5_weights", [])
    pd["v16_available"] = v16.get("status") != "UNAVAILABLE"

    # --- Risk ---
    risk = latest.get("risk", {})
    pd["risk_status"] = risk.get("portfolio_status", "UNKNOWN")
    alerts = risk.get("alerts", [])
    pd["risk_critical_count"] = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
    pd["risk_warning_count"] = sum(1 for a in alerts if a.get("severity") == "WARNING")
    pd["risk_alerts"] = alerts

    emergency = risk.get("emergency_triggers", {})
    pd["risk_emergency_active"] = any(emergency.values()) if isinstance(emergency, dict) else False

    # Regime conflict detection from ongoing_conditions
    ongoing = risk.get("ongoing_conditions", [])
    pd["regime_conflict"] = any(
        c.get("check_id", "").startswith("INT_REGIME_CONFLICT") for c in ongoing
    )
    pd["risk_available"] = True

    # --- Intelligence ---
    intel = latest.get("intelligence", {})
    pd["ic_available"] = intel.get("status") != "UNAVAILABLE"

    # IC net bearish score (sum of directional signals)
    consensus = intel.get("consensus", {})
    bearish_count = sum(1 for v in consensus.values() if v.get("direction") == "BEARISH")
    bullish_count = sum(1 for v in consensus.values() if v.get("direction") == "BULLISH")
    pd["ic_net_bearish_score"] = -(bearish_count - bullish_count)  # negative = net bearish

    # Cross-system contradictions
    cross = intel.get("cross_system", [])
    pd["ic_contradicting_count"] = sum(1 for c in cross if c.get("alignment") == "CONTRADICTING")
    pd["ic_diverging_count"] = sum(1 for c in cross if c.get("alignment") == "DIVERGING")

    # Pre-mortems
    pms = intel.get("pre_mortems", [])
    pd["pre_mortem_high_count"] = sum(1 for p in pms if p.get("aggregate_risk") == "HIGH")
    pd["pre_mortem_count"] = len(pms)

    # Cadence anomalies
    pd["cadence_anomaly_count"] = len(intel.get("cadence_anomalies", []))

    # Expert disagreements
    pd["expert_disagreement_count"] = len(intel.get("expert_disagreements", []))

    # IC Temperature elevated count (from source_cards if available)
    source_cards = intel.get("source_cards", [])
    pd["ic_temp_elevated_count"] = sum(
        1 for sc in source_cards
        if (sc.get("temperature_above_baseline", 0) or 0) >= 2
    )

    # Belief state summary
    pd["belief_state"] = intel.get("belief_state", {})
    pd["belief_shifts"] = intel.get("belief_shifts", [])

    # Active threads
    threads = intel.get("active_threads", [])
    pd["active_thread_count"] = len(threads)
    pd["threatening_thread_count"] = sum(
        1 for t in threads if t.get("portfolio_alignment") == "THREATENING"
    )

    # --- Execution ---
    exe = latest.get("execution", {})
    pd["execution_level"] = exe.get("execution_level", "UNKNOWN")
    pd["execution_score"] = exe.get("total_score", 0)
    pd["execution_available"] = True

    # Events today (HIGH impact within next 12h)
    cal = exe.get("calendar_upcoming", [])
    pd["events_today_high_impact"] = sum(
        1 for c in cal
        if c.get("impact") == "HIGH" and (c.get("hours_until", 99) <= 12 or c.get("days_until", 99) == 0)
    )
    pd["calendar_upcoming"] = cal

    # --- CIO ---
    briefing = latest.get("briefing", {})
    pd["cio_available"] = briefing.get("status") != "UNAVAILABLE" if isinstance(briefing, dict) else bool(briefing)
    pd["cio_digest"] = latest.get("digest", "")

    # --- Header / Meta ---
    hdr = latest.get("header", {})
    pd["briefing_type"] = hdr.get("briefing_type", "ROUTINE")
    pd["system_conviction"] = hdr.get("system_conviction", "UNKNOWN")
    pd["risk_ampel"] = hdr.get("risk_ampel", "UNKNOWN")
    pd["data_quality"] = hdr.get("data_quality", "UNKNOWN")

    # --- Layers ---
    layers = latest.get("layers", {})
    pd["system_regime"] = layers.get("system_regime", "UNKNOWN")
    pd["fragility_state"] = layers.get("fragility_state", "UNKNOWN")
    pd["layer_scores"] = layers.get("layer_scores", {})

    # --- Regime duration ---
    # Not directly in latest.json; estimate from deltas or default
    pd["regime_duration_days"] = _estimate_regime_duration(latest)

    # --- Pipeline coherence ---
    pd["pipeline_coherence_pct"] = _compute_pipeline_coherence(pd)

    # --- Signals / Router ---
    signals = latest.get("signals", {})
    pd["router_state"] = signals.get("router_state", "UNKNOWN")
    pd["max_proximity"] = signals.get("max_proximity", 0)
    pd["max_proximity_trigger"] = signals.get("max_proximity_trigger", "")

    # --- G7 ---
    g7 = latest.get("g7", {})
    g7_summary = latest.get("g7_summary", {})
    pd["g7_ewi_score"] = g7_summary.get("ewi_score") if g7_summary else None

    # --- Rotation ---
    pd["rotation"] = latest.get("rotation", {})

    # --- Breaking news (will be filled by news scanner, default empty) ---
    pd["breaking_news_high_count"] = 0
    pd["breaking_news"] = []

    # --- Performance (for portfolio attribution) ---
    pd["v16_performance"] = v16.get("performance", {})
    pd["spy_performance"] = v16.get("spy_performance", {})

    # --- Action items ---
    pd["action_items"] = latest.get("action_items", {})

    return pd


def _estimate_regime_duration(latest):
    """Estimate how many days current V16 regime has been active."""
    # Check if deltas has yesterday info
    deltas = latest.get("deltas", {})
    if not deltas.get("has_yesterday", False):
        return 1  # First day or no comparison data

    # Check if regime changed from CIO text or header
    # Simple heuristic: if regime is mentioned as "seit X Tagen" in digest
    digest = latest.get("digest", "")
    if "seit" in digest.lower() and "tagen" in digest.lower():
        import re
        match = re.search(r"seit\s+(\d+)\s+Tag", digest, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Default: assume fresh (conservative)
    return 2


def _compute_pipeline_coherence(pd):
    """
    Estimate pipeline coherence as percentage.
    100% = all systems agree. Lower = more divergence.
    """
    score = 100

    # Regime conflict
    if pd.get("regime_conflict"):
        score -= 15

    # IC contradicting V16
    if pd.get("ic_contradicting_count", 0) > 0:
        score -= 10 * pd["ic_contradicting_count"]

    # IC diverging from V16
    if pd.get("ic_diverging_count", 0) > 0:
        score -= 3 * pd["ic_diverging_count"]

    # Risk RED but V16 says Risk-On
    v16_regime = pd.get("v16_regime", "")
    risk_on_regimes = {"FULL_EXPANSION", "STEADY_GROWTH", "REFLATION",
                       "EARLY_RECOVERY", "FRAGILE_EXPANSION", "LATE_EXPANSION"}
    if pd.get("risk_status") == "RED" and v16_regime in risk_on_regimes:
        score -= 10

    # Data quality DEGRADED
    if pd.get("data_quality") == "DEGRADED":
        score -= 5

    return max(0, min(100, score))


def _empty_pipeline_data():
    """Return empty pipeline_data dict with all keys set to defaults."""
    return {
        "v16_regime": "UNKNOWN",
        "v16_drawdown": 0.0,
        "v16_dd_protect": "INACTIVE",
        "v16_weights": {},
        "v16_top5": [],
        "v16_available": False,
        "risk_status": "UNKNOWN",
        "risk_critical_count": 0,
        "risk_warning_count": 0,
        "risk_alerts": [],
        "risk_emergency_active": False,
        "regime_conflict": False,
        "risk_available": False,
        "ic_available": False,
        "ic_net_bearish_score": 0,
        "ic_contradicting_count": 0,
        "ic_diverging_count": 0,
        "pre_mortem_high_count": 0,
        "pre_mortem_count": 0,
        "cadence_anomaly_count": 0,
        "expert_disagreement_count": 0,
        "ic_temp_elevated_count": 0,
        "belief_state": {},
        "belief_shifts": [],
        "active_thread_count": 0,
        "threatening_thread_count": 0,
        "execution_level": "UNKNOWN",
        "execution_score": 0,
        "execution_available": False,
        "events_today_high_impact": 0,
        "calendar_upcoming": [],
        "cio_available": False,
        "cio_digest": "",
        "briefing_type": "ROUTINE",
        "system_conviction": "UNKNOWN",
        "risk_ampel": "UNKNOWN",
        "data_quality": "UNKNOWN",
        "system_regime": "UNKNOWN",
        "fragility_state": "UNKNOWN",
        "layer_scores": {},
        "regime_duration_days": 1,
        "pipeline_coherence_pct": 50,
        "router_state": "UNKNOWN",
        "max_proximity": 0,
        "max_proximity_trigger": "",
        "g7_ewi_score": None,
        "rotation": {},
        "breaking_news_high_count": 0,
        "breaking_news": [],
        "v16_performance": {},
        "spy_performance": {},
        "action_items": {},
    }


# ---------------------------------------------------------------------------
# MAIN: Collect all data
# ---------------------------------------------------------------------------

def collect_all_data(latest_json_path=None, creds_json=None):
    """
    Collect all data from latest.json + DW Sheets.

    Returns:
        (indicator_values, pipeline_data) tuple ready for composite engine.
    """
    # 1. Load latest.json
    path = latest_json_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "dashboard", "latest.json",
    )
    latest = load_latest_json(path)
    if latest is None:
        logger.error("Cannot proceed without latest.json")
        return {}, _empty_pipeline_data()

    # 2. Extract pipeline data
    pipeline_data = extract_pipeline_data(latest)

    # 3. Read DW Sheet raw indicators
    raw_market = read_dw_sheet_tab("RAW_MARKET", creds_json)
    raw_macro = read_dw_sheet_tab("RAW_MACRO", creds_json)

    # 4. Parse indicators
    indicator_values = parse_raw_market_indicators(raw_market, raw_macro)

    # 5. If DW Sheet failed, try to fill in what we can from latest.json layer scores
    if not indicator_values:
        logger.warning("DW Sheet read failed — falling back to latest.json layer data")
        indicator_values = _fallback_indicators_from_latest(latest)

    logger.info(
        f"Data collection complete: {len(indicator_values)} indicators, "
        f"regime={pipeline_data.get('v16_regime')}, "
        f"coherence={pipeline_data.get('pipeline_coherence_pct')}%"
    )

    return indicator_values, pipeline_data


def _fallback_indicators_from_latest(latest):
    """
    If DW Sheet is unavailable, extract minimal indicator proxies from latest.json.
    This is DEGRADED mode — no raw values, only layer scores as rough proxies.
    """
    indicators = {}
    layers = latest.get("layers", {}).get("layer_scores", {})

    # Map layer scores (-10 to +10) to rough indicator health (0-100)
    # This is a very rough proxy
    for name, layer in layers.items():
        score = layer.get("score", 0)
        # Convert -10..+10 to 0..100
        health = round(50 + score * 5, 1)
        health = max(0, min(100, health))

        if "Liquidity" in name:
            # Proxy for NET_LIQ
            indicators["NET_LIQ_PROXY"] = health
        elif "Macro" in name:
            # Proxy for 2Y10Y/HY_OAS
            indicators["MACRO_PROXY"] = health
        elif "Volatility" in name:
            indicators["VIX_TERM_PROXY"] = health

    logger.warning(f"Fallback indicators: {len(indicators)} proxies (DEGRADED)")
    return indicators
