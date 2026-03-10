"""
step_0s_g7_monitor/overlays.py
Phase 4: Overlay Computation — Etappe 3

Berechnet 9 Cross-Dimensionale Overlays in fester Reihenfolge:
  1. Feedback Loop Quantifizierung (7 Loops)
  2. Supply Chain Stress Index (SCSI)
  3. De-Dollarization Index (DDI)
  4. Fiscal Dominance Proximity Score (FDP)
  5. Sanctions Intensity Tracker (SIT) — LLM + Brave Search (Etappe 3)
  6. Early Warning Index (EWI, 10 Canary Signals)
  7. Geopolitical Attractiveness Ranking (V1 simplified)
  8. Liquidity Distribution Map — Placeholder (Etappe 3 LLM)
  9. Correlation Regime Monitor — Placeholder (braucht Zeitreihen-Erweiterung)

Referenz: G7_WORLD_ORDER_MONITOR_SPEC_TEIL3.md
Thresholds: config/G7_THRESHOLDS.json
"""

import os
import json
import math
import time
import traceback
from datetime import datetime, timezone

# ============================================================
# CONSTANTS
# ============================================================

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
THRESHOLDS_PATH = os.path.join(CONFIG_DIR, "G7_THRESHOLDS.json")


def _load_thresholds():
    """Load G7_THRESHOLDS.json."""
    try:
        with open(THRESHOLDS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current


# ============================================================
# 1. FEEDBACK LOOP QUANTIFIZIERUNG (Spec Teil 3 §1)
# ============================================================

# Loop definitions — which dimensions and regions each loop covers
FEEDBACK_LOOP_DEFS = {
    "debt_demographics": {
        "name": "Debt-Demographics Spiral",
        "dimensions": ["D6_fiscal", "D2_demographics"],
        "regions_at_risk": ["USA", "JP_KR_TW", "EU", "CHINA"],
    },
    "thucydides_trap": {
        "name": "Thucydides Trap",
        "dimensions": ["usa_china_gap", "D5_military", "D11_geopolitical"],
        "regions_at_risk": ["USA", "CHINA"],
    },
    "currency_fiscal": {
        "name": "Currency-Fiscal Doom Loop",
        "dimensions": ["D7_currency", "D6_fiscal"],
        "regions_at_risk": ["REST_EM", "CHINA", "EU"],
    },
    "tech_security": {
        "name": "Tech-Security Dilemma",
        "dimensions": ["D3_technology", "D11_geopolitical"],
        "regions_at_risk": ["USA", "CHINA"],
    },
    "energy_conflict": {
        "name": "Energy-Conflict Nexus",
        "dimensions": ["D4_energy", "D11_geopolitical"],
        "regions_at_risk": ["EU", "JP_KR_TW", "INDIA", "CHINA"],
    },
    "social_political": {
        "name": "Social-Political Instability Loop",
        "dimensions": ["D10_social", "D6_fiscal"],
        "regions_at_risk": ["REST_EM", "CHINA", "EU"],
    },
    "financial_contagion": {
        "name": "Financial Contagion Risk",
        "dimensions": ["D8_capital_mkt", "D9_flows"],
        "regions_at_risk": ["REST_EM", "CHINA", "EU"],
    },
}


def _compute_loop_severity(momentum_a, momentum_b, mode, threshold=0):
    """
    Compute loop severity on 0-10 scale.
    Spec Teil 3 §1: compute_loop_severity()

    mode="both_negative":
      Both momenta negative AND accelerating = ACTIVE + INTENSIFYING
      One negative, one stable = LATENT
      Both stable/positive = INACTIVE

    mode="product_above_threshold":
      Product > threshold+3 = ACTIVE
      Product > threshold = LATENT
      Else INACTIVE
    """
    severity = 0
    status = "INACTIVE"

    if mode == "both_negative":
        if momentum_a < -1 and momentum_b < -1:
            severity = min(10, abs(momentum_a) * abs(momentum_b) * 0.5)
            status = "ACTIVE"
        elif momentum_a < -1 or momentum_b < -1:
            severity = min(5, max(abs(momentum_a), abs(momentum_b)) * 0.3)
            status = "LATENT"
        else:
            severity = 0
            status = "INACTIVE"

    elif mode == "product_above_threshold":
        product = momentum_a * momentum_b
        if product > threshold + 3:
            severity = min(10, product * 0.4)
            status = "ACTIVE"
        elif product > threshold:
            severity = min(5, product * 0.2)
            status = "LATENT"
        else:
            severity = 0
            status = "INACTIVE"

    trend = "INTENSIFYING" if severity > 3 else "STABLE" if severity > 0 else "INACTIVE"

    return {
        "severity": round(severity, 1),
        "status": status,
        "trend": trend,
    }


def _compute_thucydides_severity(gap_momentum, china_military_momentum, us_gpr_score):
    """
    Special computation for Thucydides Trap.
    Spec Teil 3 §1: compute_thucydides_severity()

    Active when: Gap closing + China military rising + GPR elevated
    """
    gap_closing = max(0, -gap_momentum)        # Positive when gap shrinks
    military_buildup = max(0, china_military_momentum)
    gpr_elevated = max(0, (us_gpr_score - 100) / 50)  # Normalized from GPR 100 baseline

    raw_severity = gap_closing * 1.5 + military_buildup * 1.0 + gpr_elevated * 2.0
    severity = min(10, raw_severity)

    if severity > 5:
        status = "ACTIVE"
    elif severity > 2:
        status = "LATENT"
    else:
        status = "INACTIVE"

    return {
        "severity": round(severity, 1),
        "status": status,
        "trend": "INTENSIFYING" if gap_closing > 1 else "STABLE",
    }


def _compute_divergence_severity(usa_tech_momentum, china_tech_momentum, us_gpr_score):
    """
    Tech-Security Dilemma: divergence between USA and China tech + geopolitical tension.
    Spec Teil 3 §1: tech_security loop uses compute_divergence_severity()
    """
    # Divergence: if both moving in opposite directions = tension
    divergence = abs(usa_tech_momentum - china_tech_momentum)
    gpr_factor = max(0, (us_gpr_score - 50) / 50)  # Normalized

    raw_severity = divergence * gpr_factor * 0.8
    severity = min(10, raw_severity)

    if severity > 3:
        status = "ACTIVE"
    elif severity > 1:
        status = "LATENT"
    else:
        status = "INACTIVE"

    trend = "INTENSIFYING" if severity > 3 else "STABLE" if severity > 0 else "INACTIVE"

    return {
        "severity": round(severity, 1),
        "status": status,
        "trend": trend,
    }


def compute_feedback_loops(scores_result, gap_data):
    """
    Compute all 7 feedback loops.
    Returns sorted list of active/latent loops (highest severity first).

    Spec Teil 3 §1: FEEDBACK_LOOPS + aggregate_feedback_loops()
    """
    momenta = scores_result.get("momenta", {})
    scores = scores_result.get("scores", {})

    def _mom(dim, region):
        return momenta.get(dim, {}).get(region, 0.0)

    def _score(dim, region):
        return scores.get(dim, {}).get(region, 50.0)

    all_results = {}

    # 1. debt_demographics: D6 momentum x D2 momentum, mode=both_negative
    loop_id = "debt_demographics"
    loop_def = FEEDBACK_LOOP_DEFS[loop_id]
    results = {}
    for region in loop_def["regions_at_risk"]:
        results[region] = _compute_loop_severity(
            _mom("D6_fiscal", region),
            _mom("D2_demographics", region),
            mode="both_negative",
        )
    all_results[loop_id] = results

    # 2. thucydides_trap: Special computation
    loop_id = "thucydides_trap"
    gap_mom = gap_data.get("gap_momentum", 0)
    thuc = _compute_thucydides_severity(
        gap_mom,
        _mom("D5_military", "CHINA"),
        _score("D11_geopolitical", "USA"),
    )
    all_results[loop_id] = {"USA": thuc, "CHINA": thuc}

    # 3. currency_fiscal: D7 momentum x D6 momentum, mode=both_negative
    loop_id = "currency_fiscal"
    loop_def = FEEDBACK_LOOP_DEFS[loop_id]
    results = {}
    for region in loop_def["regions_at_risk"]:
        results[region] = _compute_loop_severity(
            _mom("D7_currency", region),
            _mom("D6_fiscal", region),
            mode="both_negative",
        )
    all_results[loop_id] = results

    # 4. tech_security: USA/China tech divergence x D11 tension
    loop_id = "tech_security"
    tech_sec = _compute_divergence_severity(
        _mom("D3_technology", "USA"),
        _mom("D3_technology", "CHINA"),
        _score("D11_geopolitical", "USA"),
    )
    all_results[loop_id] = {"USA": tech_sec, "CHINA": tech_sec}

    # 5. energy_conflict: -D4 momentum x D11 score, mode=product_above_threshold
    loop_id = "energy_conflict"
    loop_def = FEEDBACK_LOOP_DEFS[loop_id]
    results = {}
    for region in loop_def["regions_at_risk"]:
        results[region] = _compute_loop_severity(
            -_mom("D4_energy", region),
            _score("D11_geopolitical", region),
            mode="product_above_threshold",
            threshold=0,
        )
    all_results[loop_id] = results

    # 6. social_political: -D10 momentum x D6 momentum, mode=both_negative
    loop_id = "social_political"
    loop_def = FEEDBACK_LOOP_DEFS[loop_id]
    results = {}
    for region in loop_def["regions_at_risk"]:
        results[region] = _compute_loop_severity(
            -_mom("D10_social", region),
            _mom("D6_fiscal", region),
            mode="both_negative",
        )
    all_results[loop_id] = results

    # 7. financial_contagion: D8 momentum x D9 momentum, mode=both_negative
    loop_id = "financial_contagion"
    loop_def = FEEDBACK_LOOP_DEFS[loop_id]
    results = {}
    for region in loop_def["regions_at_risk"]:
        results[region] = _compute_loop_severity(
            _mom("D8_capital_mkt", region),
            _mom("D9_flows", region),
            mode="both_negative",
        )
    all_results[loop_id] = results

    # --- Aggregate: sorted list of active/latent loops ---
    active = []
    for loop_name, results_by_region in all_results.items():
        for region, result in results_by_region.items():
            if result["status"] in ("ACTIVE", "LATENT"):
                active.append({
                    "name": FEEDBACK_LOOP_DEFS[loop_name]["name"],
                    "loop_id": loop_name,
                    "region": region,
                    "severity": result["severity"],
                    "status": result["status"],
                    "trend": result["trend"],
                })

    active.sort(key=lambda x: x["severity"], reverse=True)
    return active


# ============================================================
# 2. SUPPLY CHAIN STRESS INDEX — SCSI (Spec Teil 3 §2)
# ============================================================

# Chokepoint weights (Spec Teil 3 §2)
CHOKEPOINT_WEIGHTS = {
    "suez": 0.20,
    "malacca": 0.20,
    "hormuz": 0.25,
    "bab_el_mandeb": 0.10,
    "panama": 0.10,
    "taiwan_strait": 0.15,
}


def _compute_zscore(value, baseline_mean, baseline_std):
    """Compute Z-Score given mean and std. Returns 0 if std is 0."""
    if baseline_std is None or baseline_std == 0:
        return 0.0
    return (value - baseline_mean) / baseline_std


def compute_scsi(validated_data, previous_overlays):
    """
    Supply Chain Stress Index — Composite 0-100.
    Spec Teil 3 §2.

    Components:
      - Chokepoint Composite (40%): WorldMonitor — currently unavailable, fallback 0
      - Shipping Rate Anomalies (30%): FRED PCU483111483111 Z-Score
      - Cascade Impact (30%): WorldMonitor — currently unavailable, fallback 0

    Also computes SCSI Dimension Modifiers (returned but NOT applied in Etappe 2).
    """
    # --- Component 1: Chokepoint Composite (40%) ---
    worldmonitor = validated_data.get("worldmonitor")
    chokepoints = {
        "suez": 0, "malacca": 0, "hormuz": 0,
        "bab_el_mandeb": 0, "panama": 0, "taiwan_strait": 0,
    }

    if worldmonitor and isinstance(worldmonitor.get("chokepoints"), dict):
        raw_cp = worldmonitor["chokepoints"]
        for cp_name in chokepoints:
            val = raw_cp.get(cp_name)
            if isinstance(val, (int, float)):
                chokepoints[cp_name] = max(0, min(100, val))

    # Weighted mean of chokepoints
    chokepoint_composite = sum(
        chokepoints[cp] * weight
        for cp, weight in CHOKEPOINT_WEIGHTS.items()
    )
    active_alerts = sum(1 for score in chokepoints.values() if score > 50)

    # --- Component 2: Shipping Rate Anomalies (30%) ---
    fred = validated_data.get("fred", {})

    # Deep Sea Freight PPI — use as shipping stress proxy
    freight_ppi = fred.get("PCU483111483111")
    freight_ppi_val = freight_ppi.get("value") if isinstance(freight_ppi, dict) else None

    # We don't have historical baseline in our data — use structural estimates
    # Freight PPI baseline ~120 (index), std ~15 (historical range)
    FREIGHT_PPI_BASELINE_MEAN = 120.0
    FREIGHT_PPI_BASELINE_STD = 15.0

    if freight_ppi_val is not None:
        freight_zscore = _compute_zscore(
            freight_ppi_val, FREIGHT_PPI_BASELINE_MEAN, FREIGHT_PPI_BASELINE_STD
        )
    else:
        freight_zscore = 0.0

    # BDI: not in our yfinance tickers, and not available via FRED
    # Use freight PPI Z-score alone for shipping stress
    bdi_zscore = 0.0  # Unavailable — would need ^BDIY or FRED equivalent

    # Shipping stress: max of absolute Z-scores * 20, capped at 100
    shipping_stress = max(abs(freight_zscore), abs(bdi_zscore)) * 20
    shipping_stress = min(100, max(0, shipping_stress))

    # --- Component 3: Cascade Impact (30%) ---
    # WorldMonitor cascade model — currently unavailable
    cascade_score = 0.0

    # --- Composite ---
    composite = (
        chokepoint_composite * 0.40
        + shipping_stress * 0.30
        + cascade_score * 0.30
    )

    # --- Trend vs previous ---
    prev_composite = _safe_get(
        previous_overlays, "scsi", "composite", default=0
    )
    trend = _compute_trend(composite, prev_composite)

    # --- SCSI Dimension Modifiers (Spec Teil 3 §2) ---
    # Computed here, returned but NOT applied in Etappe 2
    dimension_modifiers = _compute_scsi_dimension_modifiers(chokepoints)

    return {
        "composite": round(composite, 1),
        "trend": trend,
        "active_chokepoint_alerts": active_alerts,
        "chokepoints": chokepoints,
        "shipping_stress": round(shipping_stress, 1),
        "bdi_zscore": round(bdi_zscore, 2),
        "freight_ppi_zscore": round(freight_zscore, 2),
        "cascade_score": round(cascade_score, 1),
        "dimension_modifiers": dimension_modifiers,
        "data_note": _scsi_data_note(worldmonitor),
    }


def _scsi_data_note(worldmonitor):
    """Describe data availability for SCSI."""
    parts = []
    if worldmonitor is None:
        parts.append("WorldMonitor unavailable (HTTP 403) — chokepoints and cascade at 0")
    parts.append("BDI not in yfinance tickers — using Freight PPI only for shipping stress")
    return "; ".join(parts) if parts else "All sources available"


def _compute_scsi_dimension_modifiers(chokepoints):
    """
    SCSI modifies dimension scores for affected regions.
    Spec Teil 3 §2: compute_scsi_dimension_modifiers()

    Returned but NOT applied to scores in Etappe 2.
    Will be applied when Phase flow is refactored in Etappe 3.
    """
    modifiers = {}

    # Taiwan Strait > 40
    ts = chokepoints.get("taiwan_strait", 0)
    if ts > 40:
        severity = ts / 100
        modifiers.setdefault("JP_KR_TW", {})
        modifiers["JP_KR_TW"]["D3_tech_boost"] = round(-severity * 15, 1)
        modifiers["JP_KR_TW"]["D4_energy_boost"] = round(-severity * 8, 1)
        modifiers.setdefault("CHINA", {})
        modifiers["CHINA"]["D4_energy_boost"] = round(-severity * 5, 1)
        modifiers.setdefault("USA", {})
        modifiers["USA"]["D3_tech_boost"] = round(-severity * 5, 1)

    # Strait of Hormuz > 40
    hz = chokepoints.get("hormuz", 0)
    if hz > 40:
        severity = hz / 100
        modifiers.setdefault("JP_KR_TW", {})
        modifiers["JP_KR_TW"]["D4_energy_boost"] = round(
            modifiers.get("JP_KR_TW", {}).get("D4_energy_boost", 0) - severity * 18, 1
        )
        modifiers.setdefault("INDIA", {})
        modifiers["INDIA"]["D4_energy_boost"] = round(-severity * 12, 1)
        modifiers.setdefault("CHINA", {})
        modifiers["CHINA"]["D4_energy_boost"] = round(
            modifiers.get("CHINA", {}).get("D4_energy_boost", 0) - severity * 8, 1
        )

    # Suez Canal > 40
    sz = chokepoints.get("suez", 0)
    if sz > 40:
        severity = sz / 100
        modifiers.setdefault("EU", {})
        modifiers["EU"]["D9_flows_boost"] = round(-severity * 10, 1)
        modifiers["EU"]["D4_energy_boost"] = round(-severity * 5, 1)

    # Bab el-Mandeb > 40
    bm = chokepoints.get("bab_el_mandeb", 0)
    if bm > 40:
        severity = bm / 100
        modifiers.setdefault("EU", {})
        modifiers["EU"]["D9_flows_boost"] = round(
            modifiers.get("EU", {}).get("D9_flows_boost", 0) - severity * 6, 1
        )

    # Malacca Strait > 40
    ml = chokepoints.get("malacca", 0)
    if ml > 40:
        severity = ml / 100
        modifiers.setdefault("CHINA", {})
        modifiers["CHINA"]["D4_energy_boost"] = round(
            modifiers.get("CHINA", {}).get("D4_energy_boost", 0) - severity * 10, 1
        )
        modifiers.setdefault("JP_KR_TW", {})
        modifiers["JP_KR_TW"]["D9_flows_boost"] = round(-severity * 8, 1)

    # Panama Canal > 40
    pn = chokepoints.get("panama", 0)
    if pn > 40:
        severity = pn / 100
        modifiers.setdefault("USA", {})
        modifiers["USA"]["D9_flows_boost"] = round(-severity * 5, 1)

    return modifiers


# ============================================================
# 3. DE-DOLLARIZATION INDEX — DDI (Spec Teil 3 §3)
# ============================================================

def compute_ddi(validated_data, previous_overlays):
    """
    De-Dollarization Index — Composite 0-100.
    Spec Teil 3 §3.

    Components:
      1. COFER USD Reserve Share (25%) — from IMF COFER data
      2. SWIFT USD Payment Share (20%) — Placeholder (no automated source)
      3. CB Gold Purchases (20%) — Placeholder (WGC not automated)
      4. Bilateral Non-USD Agreements (15%) — Placeholder (Etappe 3 LLM)
      5. DXY Trend (10%) — from yfinance
      6. USD Invoicing Share (10%) — Placeholder (Etappe 3 LLM)
    """
    cofer = validated_data.get("imf_cofer", {})
    yf = validated_data.get("yfinance", {})

    # --- Component 1: COFER USD Reserve Share (25%) ---
    usd_reserve_share = cofer.get("USD_share", 58.4)
    # Score: 100 when USD at 40% (strongly de-dollarized), 0 when at 70%
    cofer_score = max(0, min(100, (70 - usd_reserve_share) / 30 * 100))

    # --- Component 2: SWIFT USD Payment Share (20%) ---
    # Not available via automated source — use structural estimate
    # USD SWIFT share ~47% as of late 2025
    swift_usd_estimate = 47.0
    swift_score = max(0, min(100, (55 - swift_usd_estimate) / 25 * 100))

    # --- Component 3: CB Gold Purchases (20%) ---
    # WGC data not automated — use structural estimate
    # ~1000 tonnes/year recently, vs 5y avg ~650
    cb_gold_ytd_estimate = 1000
    gold_score = min(100, (cb_gold_ytd_estimate / 1200) * 100)

    # --- Component 4: Bilateral Non-USD Agreements (15%) ---
    # LLM-assessed in Etappe 3 — neutral default
    bilateral_score = 40  # Moderate: BRICS expanding but slowly

    # --- Component 5: DXY Trend (10%) ---
    dxy_data = yf.get("DX-Y.NYB")
    dxy_pct_1m = 0
    if isinstance(dxy_data, dict):
        dxy_pct_1m = dxy_data.get("pct_change_1m", 0) or 0

    # Falling dollar = higher DDI score (use 6m proxy: 1m * ~2 rough scaling)
    dxy_6m_proxy = dxy_pct_1m * 2
    dxy_score = max(0, min(100, 50 + (-dxy_6m_proxy * 10)))

    # --- Component 6: USD Invoicing Share (10%) ---
    # Still ~80%, very stable — counter-narrative data point
    invoicing_score = 15  # Low: invoicing barely de-dollarized

    # --- Composite ---
    composite = (
        cofer_score * 0.25
        + swift_score * 0.20
        + gold_score * 0.20
        + bilateral_score * 0.15
        + dxy_score * 0.10
        + invoicing_score * 0.10
    )

    # --- Trend & Acceleration ---
    prev_composite = _safe_get(previous_overlays, "ddi", "composite", default=0)
    momentum = composite - prev_composite
    prev_momentum = _safe_get(previous_overlays, "ddi", "momentum", default=0)
    acceleration = momentum - prev_momentum

    if momentum > 0.5:
        trend = "RISING"
    elif momentum < -0.5:
        trend = "FALLING"
    else:
        trend = "STABLE"

    # --- Portfolio implications (Spec Teil 3 §3) ---
    gold_bias = _compute_gold_bias(composite)
    permopt_signal = _compute_permopt_signal(composite)

    return {
        "composite": round(composite, 1),
        "trend": trend,
        "acceleration": round(acceleration, 2),
        "momentum": round(momentum, 2),
        "components": {
            "cofer_usd_share": {
                "value": usd_reserve_share,
                "score": round(cofer_score, 1),
                "source": "imf_cofer" if cofer.get("USD_share") else "estimate",
            },
            "swift_usd_share": {
                "value": swift_usd_estimate,
                "score": round(swift_score, 1),
                "source": "structural_estimate",
            },
            "cb_gold_purchases_ytd": {
                "tonnes": cb_gold_ytd_estimate,
                "score": round(gold_score, 1),
                "source": "structural_estimate",
            },
            "bilateral_non_usd": {
                "score": bilateral_score,
                "source": "placeholder_etappe3",
            },
            "dxy_trend": {
                "pct_1m": dxy_pct_1m,
                "score": round(dxy_score, 1),
                "source": "yfinance",
            },
            "usd_invoicing_share": {
                "value_pct": 80,
                "score": invoicing_score,
                "source": "structural_estimate",
                "note": "Stable at ~80% — counter-narrative data point",
            },
        },
        "portfolio_implications": {
            "gold_bias": gold_bias,
            "permopt_sizing": permopt_signal,
        },
    }


def _compute_gold_bias(ddi_composite):
    """Spec Teil 3 §3: compute_gold_bias()"""
    if ddi_composite > 70:
        return "+5-8% overweight strongly supported"
    elif ddi_composite > 60:
        return "+3-5% overweight supported"
    elif ddi_composite > 50:
        return "+1-3% moderate overweight"
    else:
        return "Neutral — de-dollarization not accelerating"


def _compute_permopt_signal(ddi_composite):
    """Spec Teil 3 §3: compute_permopt_signal()"""
    if ddi_composite > 70:
        return "Consider increasing PermOpt from 3% to 5%"
    elif ddi_composite > 60:
        return "Current 3% PermOpt adequate"
    elif ddi_composite > 50:
        return "Current 3% PermOpt may be oversized"
    else:
        return "Consider reducing PermOpt to 1-2%"


# ============================================================
# 4. FISCAL DOMINANCE PROXIMITY — FDP (Spec Teil 3 §4)
# ============================================================

# Japan special case note
JAPAN_FDP_NOTE = (
    "Japan has operated in managed fiscal dominance for decades. "
    "High proximity score does NOT imply imminent crisis — structural "
    "differences (domestic debt, BOJ holdings, deflationary environment) "
    "make Japan's situation unique. Monitor for BOJ normalization as "
    "potential regime change catalyst."
)


def compute_fdp(validated_data, previous_overlays):
    """
    Fiscal Dominance Proximity Score per region.
    Spec Teil 3 §4.

    Sub-indicators:
      1. Interest/Revenue Ratio (35%) — FRED for USA, IMF for others
      2. CB Balance Sheet/GDP (25%) — Placeholder (hard to automate)
      3. Debt Trajectory (20%) — from IMF/WB debt growth
      4. Real Rate Sustainability (20%) — nominal rate - inflation
    """
    fred = validated_data.get("fred", {})
    imf = validated_data.get("imf_weo", {})
    yf = validated_data.get("yfinance", {})

    result = {}

    for region in REGIONS:
        # --- Sub 1: Interest/Revenue Ratio (35%) ---
        itr = _compute_itr(region, fred, imf)
        itr_proximity = min(1.0, itr / 0.25) if itr is not None else 0.3

        # --- Sub 2: CB Balance Sheet/GDP (25%) ---
        # Hard to automate — use structural estimates
        cb_proximity = _cb_balance_estimate(region)

        # --- Sub 3: Debt Trajectory (20%) ---
        debt_gdp = _get_debt_gdp(region, imf)
        # Use IMF forecast delta as proxy for debt growth
        trajectory_score = _compute_debt_trajectory(region, imf)

        # --- Sub 4: Real Rate Sustainability (20%) ---
        real_rate_score = _compute_real_rate_score(region, fred, imf, yf)

        # --- Composite ---
        composite = (
            itr_proximity * 0.35
            + cb_proximity * 0.25
            + trajectory_score * 0.20
            + real_rate_score * 0.20
        )

        # --- Trend ---
        prev_comp = _safe_get(
            previous_overlays, "fdp", region, "composite_proximity", default=0
        )
        trend = _compute_trend(composite, prev_comp)

        # --- Quarters to threshold ---
        quarterly_change = composite - prev_comp
        if quarterly_change > 0 and composite < 0.95:
            remaining = 1.0 - composite
            quarters_to_threshold = round(remaining / quarterly_change) if quarterly_change > 0.001 else "N/A"
        elif quarterly_change <= 0:
            quarters_to_threshold = "N/A — improving"
        else:
            quarters_to_threshold = "N/A"

        # --- Implication ---
        implication = _generate_fdp_implication(region, composite, quarters_to_threshold)

        entry = {
            "composite_proximity": round(composite, 2),
            "trend": trend,
            "interest_to_revenue": round(itr, 3) if itr is not None else None,
            "cb_balance_to_gdp_est": round(cb_proximity * 0.40, 3),
            "debt_trajectory": round(trajectory_score, 2),
            "real_rate_score": round(real_rate_score, 2),
            "estimated_quarters_to_threshold": quarters_to_threshold,
            "implication": implication,
        }

        # Japan special case
        if region == "JP_KR_TW":
            entry["special_note"] = JAPAN_FDP_NOTE

        result[region] = entry

    return result


def _compute_itr(region, fred, imf):
    """Interest-to-Revenue ratio. USA from FRED, others estimated."""
    if region == "USA":
        interest = fred.get("A091RC1Q027SBEA")
        revenue = fred.get("FGRECPT")
        int_val = interest.get("value") if isinstance(interest, dict) else None
        rev_val = revenue.get("value") if isinstance(revenue, dict) else None
        if int_val is not None and rev_val is not None and rev_val > 0:
            return int_val / rev_val
        return 0.18  # Recent estimate ~18%

    # Structural estimates for other regions
    estimates = {
        "CHINA": 0.06,
        "EU": 0.04,
        "INDIA": 0.25,
        "JP_KR_TW": 0.15,
        "GULF": 0.03,
        "REST_EM": 0.20,
    }
    return estimates.get(region, 0.10)


def _cb_balance_estimate(region):
    """CB Balance Sheet / GDP — structural estimates as proximity 0-1."""
    # Threshold: 0.40 (40% GDP = CB heavily involved)
    estimates = {
        "USA": 0.55,    # Fed BS ~30% GDP → proximity 0.75 → scaled
        "CHINA": 0.45,  # PBOC ~35% GDP
        "EU": 0.50,     # ECB ~40% GDP (QT underway)
        "INDIA": 0.20,  # RBI ~20% GDP
        "JP_KR_TW": 0.95,  # BOJ ~130% GDP — extreme
        "GULF": 0.15,   # Low CB involvement
        "REST_EM": 0.25,
    }
    return estimates.get(region, 0.30)


def _get_debt_gdp(region, imf):
    """Get Debt/GDP from IMF WEO data."""
    key = f"GGXWDG_NGDP_{region}"
    entry = imf.get(key)
    if isinstance(entry, dict) and entry.get("value") is not None:
        return entry["value"]
    # Structural estimates
    estimates = {
        "USA": 125, "CHINA": 85, "EU": 90, "INDIA": 82,
        "JP_KR_TW": 260, "GULF": 25, "REST_EM": 65,
    }
    return estimates.get(region, 80)


def _compute_debt_trajectory(region, imf):
    """Debt trajectory score 0-1. Accelerating debt = higher proximity."""
    debt_gdp = _get_debt_gdp(region, imf)
    # Use known growth rates as structural estimates
    growth_estimates = {
        "USA": 4.5,     # ~4-5% annual debt growth
        "CHINA": 6.0,   # Rapid
        "EU": 1.5,      # Slow
        "INDIA": 3.0,   # Moderate
        "JP_KR_TW": 2.0,  # Slow (already high)
        "GULF": -2.0,   # Declining (oil revenue)
        "REST_EM": 3.5,
    }
    growth = growth_estimates.get(region, 3.0)
    return min(1.0, max(0, growth / 10))


def _compute_real_rate_score(region, fred, imf, yf):
    """
    Real rate sustainability score 0-1.
    Negative real rates = CB subsidizing government.
    Spec Teil 3 §4.
    """
    nominal_rate = None
    inflation = None

    if region == "USA":
        dgs10 = fred.get("DGS10")
        if isinstance(dgs10, dict) and dgs10.get("value") is not None:
            nominal_rate = dgs10["value"]
        inf_data = imf.get("PCPIPCH_USA")
        if isinstance(inf_data, dict) and inf_data.get("value") is not None:
            inflation = inf_data["value"]

    if nominal_rate is None or inflation is None:
        # Structural estimates
        rate_estimates = {
            "USA": (4.3, 2.8),
            "CHINA": (2.5, 0.5),
            "EU": (2.8, 2.2),
            "INDIA": (7.0, 4.5),
            "JP_KR_TW": (1.0, 2.5),
            "GULF": (5.0, 2.5),
            "REST_EM": (8.0, 5.0),
        }
        est = rate_estimates.get(region, (4.0, 3.0))
        if nominal_rate is None:
            nominal_rate = est[0]
        if inflation is None:
            inflation = est[1]

    real_rate = nominal_rate - inflation

    if real_rate < -2:
        return 0.9   # Already in financial repression
    elif real_rate < 0:
        return 0.6   # Mild financial repression
    elif real_rate < 2:
        return 0.3   # Normal
    else:
        return 0.1   # Hawkish — far from fiscal dominance


def _generate_fdp_implication(region, proximity, qtrs):
    """Human-readable FDP context for CIO. Spec Teil 3 §4."""
    if proximity > 0.90:
        return (
            f"{region}: Already in or near fiscal dominance. "
            f"CB policy constrained by fiscal needs."
        )
    elif proximity > 0.75:
        q_str = f"~{qtrs} quarters to threshold" if isinstance(qtrs, (int, float)) else str(qtrs)
        return (
            f"{region}: Fiscal dominance approaching. At current trajectory, "
            f"{q_str}. Bond market repricing risk."
        )
    elif proximity > 0.60:
        return (
            f"{region}: Fiscal space narrowing. Monitor interest/revenue "
            f"ratio and CB balance sheet trends."
        )
    else:
        return (
            f"{region}: Fiscal position manageable. No near-term "
            f"fiscal dominance concern."
        )


# ============================================================
# 5. SANCTIONS INTENSITY TRACKER — SIT (Spec Teil 3 §5)
# ============================================================

# Pre-configured baseline expectations (helps LLM calibration)
SANCTIONS_BASELINE = {
    "USA":      {"role": "PRIMARY_IMPOSER", "packages_est": 30},
    "EU":       {"role": "SECONDARY_IMPOSER", "packages_est": 20},
    "CHINA":    {"role": "TARGET_RETALIATOR", "packages_est": 15},
    "INDIA":    {"role": "MOSTLY_NEUTRAL", "packages_est": 2},
    "JP_KR_TW": {"role": "US_ALIGNED_IMPOSER", "packages_est": 10},
    "GULF":     {"role": "SELECTIVE_COMPLIANCE", "packages_est": 3},
    "REST_EM":  {"role": "MIXED", "packages_est": 5},
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SIT_CACHE_PATH = os.path.join(CACHE_DIR, "sit.json")
SIT_CACHE_MAX_AGE_DAYS = 90


def _load_sit_cache():
    """Load SIT cache. Returns None if stale or missing."""
    try:
        if not os.path.exists(SIT_CACHE_PATH):
            return None
        with open(SIT_CACHE_PATH, "r") as f:
            cache = json.load(f)
        cached_date = cache.get("cached_date", "")
        if cached_date:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached_date)).days
            if age <= SIT_CACHE_MAX_AGE_DAYS:
                return cache
    except Exception:
        pass
    return None


def _save_sit_cache(data):
    """Save SIT result to cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        data["cached_date"] = datetime.now(timezone.utc).isoformat()
        with open(SIT_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  SIT cache save error: {e}")


def _run_brave_sit_queries():
    """
    Run Brave Search queries for sanctions intelligence.
    7 queries — one per region, focused on recent sanctions activity.
    """
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        print("  SIT: No BRAVE_API_KEY — skipping web search")
        return None

    import requests

    queries = {}
    for region in REGIONS:
        region_name = {
            "USA": "United States", "CHINA": "China", "EU": "European Union",
            "INDIA": "India", "JP_KR_TW": "Japan South Korea Taiwan",
            "GULF": "Saudi Arabia UAE Gulf states", "REST_EM": "emerging markets",
        }.get(region, region)

        queries[region] = f"sanctions imposed on by {region_name} 2025 2026 SWIFT"

    all_results = {}
    total_results = 0

    for region, query in queries.items():
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                params={"q": query, "count": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("web", {}).get("results", [])
                snippets = [r.get("description", "")[:300] for r in results[:5]]
                all_results[region] = " | ".join(snippets)
                total_results += len(results)
            else:
                all_results[region] = ""
        except Exception as e:
            print(f"  SIT Brave error for {region}: {e}")
            all_results[region] = ""
        time.sleep(0.25)

    print(f"  SIT: {total_results} Brave results across {len(queries)} queries")
    return all_results


def _call_sit_llm(search_results):
    """
    Call Claude Sonnet to assess sanctions intensity per region.
    Returns structured JSON with per-region assessments.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  SIT: No ANTHROPIC_API_KEY — using fallback")
        return None

    import requests as req

    # Build context block from search results
    context_parts = []
    for region in REGIONS:
        baseline = SANCTIONS_BASELINE.get(region, {})
        snippets = search_results.get(region, "No data available")
        context_parts.append(
            f"--- {region} (baseline role: {baseline.get('role', 'UNKNOWN')}, "
            f"est. {baseline.get('packages_est', 0)} active packages) ---\n"
            f"Web search snippets: {snippets[:800]}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are the Sanctions Intensity Tracker for a geopolitical intelligence system.

Assess the current sanctions landscape for each of the 7 regions based on the web search data below.

SEARCH DATA:
{context}

Respond ONLY with valid JSON (no markdown, no preamble). Use this exact schema:
{{
  "regions": {{
    "USA": {{
      "imposed_by": {{"active_packages": int, "new_last_90d": int, "severity": "LOW|MEDIUM|HIGH"}},
      "imposed_on": {{"active_packages": int, "new_last_90d": int, "severity": "LOW|MEDIUM|HIGH"}},
      "swift_disconnection_risk": "LOW|MEDIUM|HIGH|ACTIVE",
      "reserve_freeze_risk": "LOW|MEDIUM|HIGH|ACTIVE",
      "escalation_trend": "DE-ESCALATING|STABLE|ESCALATING|CRITICAL",
      "severity_score": float 0-10,
      "highlight": "one sentence"
    }},
    ... (same for CHINA, EU, INDIA, JP_KR_TW, GULF, REST_EM)
  }},
  "global_escalation_trend": "DE-ESCALATING|STABLE|ESCALATING|CRITICAL",
  "global_highlight": "one sentence summary"
}}

RULES:
- severity_score: 0=no sanctions exposure, 5=moderate, 10=maximum (like Russia 2022)
- For PRIMARY_IMPOSERS (USA, EU): severity_score reflects THEIR vulnerability to retaliation
- Be calibrated: USA imposing sanctions on others is NORMAL (severity_score 1-2 for USA itself)
- CHINA receiving tech sanctions is current reality (severity_score 4-6)
- REST_EM varies hugely — use weighted average of major EMs
- Base your assessment on the search data provided, not assumptions"""

    try:
        resp = req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 3000,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"  SIT LLM error: HTTP {resp.status_code}")
            return None

        data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Parse JSON from response
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        print(f"  SIT LLM: {len(text)} chars response, {len(result.get('regions', {}))} regions")
        return result

    except json.JSONDecodeError as e:
        print(f"  SIT LLM JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  SIT LLM error: {e}")
        return None


def _build_sit_fallback():
    """Deterministic fallback when Brave/LLM unavailable."""
    regions = {}
    fallback_data = {
        "USA":      {"imposed_by_pkg": 30, "imposed_on_pkg": 2, "sev": 1.5, "esc": "STABLE",
                     "swift": "LOW", "reserve": "LOW", "hl": "Primary sanctions imposer; minimal own exposure."},
        "CHINA":    {"imposed_by_pkg": 5, "imposed_on_pkg": 15, "sev": 5.0, "esc": "ESCALATING",
                     "swift": "MEDIUM", "reserve": "MEDIUM", "hl": "Increasing tech/semiconductor sanctions from US and allies."},
        "EU":       {"imposed_by_pkg": 20, "imposed_on_pkg": 1, "sev": 1.0, "esc": "STABLE",
                     "swift": "LOW", "reserve": "LOW", "hl": "Active sanctions imposer aligned with US; low own exposure."},
        "INDIA":    {"imposed_by_pkg": 0, "imposed_on_pkg": 1, "sev": 0.5, "esc": "STABLE",
                     "swift": "LOW", "reserve": "LOW", "hl": "Mostly neutral; minor secondary sanctions risk from Russia trade."},
        "JP_KR_TW": {"imposed_by_pkg": 8, "imposed_on_pkg": 1, "sev": 1.0, "esc": "STABLE",
                     "swift": "LOW", "reserve": "LOW", "hl": "US-aligned imposers; chip export controls on China active."},
        "GULF":     {"imposed_by_pkg": 1, "imposed_on_pkg": 2, "sev": 1.5, "esc": "STABLE",
                     "swift": "LOW", "reserve": "LOW", "hl": "Selective compliance; balancing US and China relationships."},
        "REST_EM":  {"imposed_by_pkg": 1, "imposed_on_pkg": 5, "sev": 3.0, "esc": "STABLE",
                     "swift": "MEDIUM", "reserve": "LOW", "hl": "Mixed exposure; Russia/Iran sanctions affect some EMs."},
    }

    for region in REGIONS:
        fb = fallback_data.get(region, {})
        regions[region] = {
            "imposed_by": {"active_packages": fb.get("imposed_by_pkg", 0), "new_last_90d": 0, "severity": "LOW"},
            "imposed_on": {"active_packages": fb.get("imposed_on_pkg", 0), "new_last_90d": 0, "severity": "LOW"},
            "swift_disconnection_risk": fb.get("swift", "LOW"),
            "reserve_freeze_risk": fb.get("reserve", "LOW"),
            "escalation_trend": fb.get("esc", "STABLE"),
            "severity_score": fb.get("sev", 1.0),
            "highlight": fb.get("hl", "No data available."),
        }

    return {
        "regions": regions,
        "global_escalation_trend": "STABLE",
        "global_highlight": "Sanctions landscape stable; China tech restrictions dominant theme. (FALLBACK DATA)",
    }


def compute_sit(run_type="WEEKLY"):
    """
    Sanctions Intensity Tracker — LLM + Brave Search.
    Spec Teil 3 §5.

    Measures TWO dimensions per region:
      1. Sanctions IMPOSED BY this region (power projection)
      2. Sanctions IMPOSED ON this region (vulnerability)

    Cache: 90 days (sanctions shift slowly). Force refresh on AD_HOC/QUARTERLY.
    Fallback: Deterministic baseline when Brave/LLM unavailable.
    """
    print("  SIT: Computing Sanctions Intensity...")

    # --- Check cache ---
    force = run_type in ("AD_HOC", "QUARTERLY")
    if not force:
        cache = _load_sit_cache()
        if cache and cache.get("regions"):
            print("  SIT: Using cached data (still valid)")
            return _format_sit_output(cache, source="CACHE")

    # --- Brave Search ---
    ps = time.time()
    search_results = _run_brave_sit_queries()

    if search_results:
        # --- LLM Assessment ---
        llm_result = _call_sit_llm(search_results)
        if llm_result and llm_result.get("regions"):
            # Validate: must have all 7 regions
            if len(llm_result["regions"]) >= 5:
                duration = round(time.time() - ps, 1)
                print(f"  SIT: LLM assessment complete ({duration}s)")
                _save_sit_cache(llm_result)
                return _format_sit_output(llm_result, source="BRAVE_SEARCH_LLM")

    # --- Fallback ---
    print("  SIT: Using deterministic fallback")
    fallback = _build_sit_fallback()
    return _format_sit_output(fallback, source="FALLBACK")


def _format_sit_output(raw, source="UNKNOWN"):
    """Format SIT data into the standard overlay output dict."""
    regions = raw.get("regions", {})

    # Compute aggregate escalation trend from regions
    escalation_counts = {"DE-ESCALATING": 0, "STABLE": 0, "ESCALATING": 0, "CRITICAL": 0}
    for region_data in regions.values():
        trend = region_data.get("escalation_trend", "STABLE")
        if trend in escalation_counts:
            escalation_counts[trend] += 1

    # Global trend: use raw value if provided, else derive from regions
    global_trend = raw.get("global_escalation_trend")
    if not global_trend:
        if escalation_counts["CRITICAL"] >= 1:
            global_trend = "CRITICAL"
        elif escalation_counts["ESCALATING"] >= 2:
            global_trend = "ESCALATING"
        elif escalation_counts["DE-ESCALATING"] >= 4:
            global_trend = "DE-ESCALATING"
        else:
            global_trend = "STABLE"

    # Per-region severity scores (for Attractiveness and Scenario Engine)
    severity_scores = {}
    for region in REGIONS:
        rd = regions.get(region, {})
        severity_scores[region] = rd.get("severity_score", 1.0)

    return {
        "escalation_trend": global_trend,
        "highlight": raw.get("global_highlight", ""),
        "regions": regions,
        "severity_scores": severity_scores,
        "source": source,
        "data_note": f"SIT assessed via {source}.",
    }


# ============================================================
# 6. EARLY WARNING INDEX — EWI (Spec Teil 3 §6)
# ============================================================

def compute_ewi(validated_data, scsi_result, previous_overlays, thresholds):
    """
    Early Warning Index — 10 Canary Signals.
    Spec Teil 3 §6.

    Checks each signal against its threshold.
    Returns count of active signals + details.
    """
    ewi_config = thresholds.get("ewi_canary_signals", {})

    fred = validated_data.get("fred", {})
    yf = validated_data.get("yfinance", {})
    gpr_data = validated_data.get("gpr")
    polymarket = validated_data.get("polymarket")
    acled = validated_data.get("acled")
    worldmonitor = validated_data.get("worldmonitor")

    active = []

    # 1. bdi_anomaly: Z-Score vs 90d baseline
    # BDI not available — use Freight PPI as proxy
    freight_zscore = scsi_result.get("freight_ppi_zscore", 0)
    bdi_threshold = ewi_config.get("bdi_anomaly", {}).get("threshold", 1.5)
    if abs(freight_zscore) > bdi_threshold:
        active.append({
            "signal": "Shipping Rate Anomaly",
            "signal_id": "bdi_anomaly",
            "value": round(freight_zscore, 2),
            "threshold": bdi_threshold,
            "description": f"Freight PPI Z-Score {freight_zscore:+.2f} exceeds {bdi_threshold} sigma threshold",
            "temporal": "TACTICAL",
            "lead_time": "Days to weeks",
        })

    # 2. chokepoint_alert: Any chokepoint > 50
    cp_threshold = ewi_config.get("chokepoint_alert", {}).get("threshold", 50)
    chokepoints = scsi_result.get("chokepoints", {})
    for cp_name, cp_score in chokepoints.items():
        if cp_score > cp_threshold:
            active.append({
                "signal": f"Chokepoint Alert: {cp_name}",
                "signal_id": "chokepoint_alert",
                "value": cp_score,
                "threshold": cp_threshold,
                "description": f"{cp_name} disruption score {cp_score} exceeds {cp_threshold}",
                "temporal": "TACTICAL",
                "lead_time": "Days",
            })
            break  # One alert is enough for the signal

    # 3. polymarket_shift: Any tracked market >5pp/week
    poly_threshold = ewi_config.get("polymarket_shift", {}).get("threshold", 5)
    if isinstance(polymarket, dict) and polymarket.get("markets"):
        for mkt in polymarket["markets"]:
            prob = mkt.get("probability")
            if isinstance(prob, (int, float)) and prob > 0:
                # We don't have week-over-week delta — skip for now
                # Would need historical polymarket data to compute shift
                pass

    # 4. gpr_spike: Z-Score >2 vs 90d MA
    gpr_threshold = ewi_config.get("gpr_spike", {}).get("threshold", 2.0)
    if isinstance(gpr_data, dict) and gpr_data.get("gpr_global") is not None:
        gpr_val = gpr_data["gpr_global"]
        # GPR baseline ~100-120, std ~40 (historical)
        gpr_zscore = _compute_zscore(gpr_val, 110, 40)
        if abs(gpr_zscore) > gpr_threshold:
            active.append({
                "signal": "Geopolitical Risk Spike",
                "signal_id": "gpr_spike",
                "value": round(gpr_zscore, 2),
                "threshold": gpr_threshold,
                "description": f"GPR Index {gpr_val:.0f} (Z={gpr_zscore:+.2f}) exceeds {gpr_threshold} sigma",
                "temporal": "TACTICAL",
                "lead_time": "Days to weeks",
            })

    # 5. swift_shift: Monthly delta >1pp — not available (no automated source)
    # Skip — would need SWIFT RMB Tracker data

    # 6. gold_acceleration: Quarterly purchases >150% of 5y avg — not available
    # Skip — would need WGC automated data

    # 7. tic_treasury_shift: Top-10 net selling >$20B/month — not available
    # Skip — would need TIC data integration

    # 8. vvix_spike: VVIX > 120 or Z-Score > 2
    vvix_threshold = ewi_config.get("vvix_spike", {}).get("threshold", 120)
    vvix_data = yf.get("^VVIX")
    if isinstance(vvix_data, dict) and vvix_data.get("close") is not None:
        vvix_val = vvix_data["close"]
        # Check absolute level
        if vvix_val > vvix_threshold:
            active.append({
                "signal": "Volatility of Volatility Spike",
                "signal_id": "vvix_spike",
                "value": round(vvix_val, 1),
                "threshold": vvix_threshold,
                "description": f"VVIX at {vvix_val:.1f} exceeds {vvix_threshold}",
                "temporal": "TACTICAL",
                "lead_time": "Days",
            })
        else:
            # Check Z-Score (baseline ~90, std ~15)
            vvix_zscore = _compute_zscore(vvix_val, 90, 15)
            if vvix_zscore > 2.0:
                active.append({
                    "signal": "Volatility of Volatility Spike",
                    "signal_id": "vvix_spike",
                    "value": round(vvix_val, 1),
                    "threshold": f"Z>{2.0}",
                    "description": f"VVIX at {vvix_val:.1f} (Z={vvix_zscore:+.2f})",
                    "temporal": "TACTICAL",
                    "lead_time": "Days",
                })

    # 9. worldmonitor_convergence: 3+ signal types in same grid — not available
    # Skip — WorldMonitor unavailable

    # 10. acled_protest_spike: Welford Z-Score >2 — not available
    # Skip — ACLED key pending

    # --- Severity classification ---
    n_active = len(active)
    severity_levels = ewi_config.get("ewi_severity_levels", {})

    if n_active >= severity_levels.get("HIGH", {}).get("min_active", 4):
        severity = "HIGH"
    elif n_active >= severity_levels.get("MEDIUM", {}).get("min_active", 2):
        severity = "MEDIUM"
    elif n_active >= severity_levels.get("LOW", {}).get("min_active", 1):
        severity = "LOW"
    else:
        severity = "NONE"

    return {
        "active_signals": n_active,
        "total_signals": 10,
        "severity": severity,
        "active_details": active,
        "signals_unavailable": [
            "swift_shift", "gold_acceleration", "tic_treasury_shift",
            "worldmonitor_convergence", "acled_protest_spike",
        ],
        "signals_checked": [
            "bdi_anomaly", "chokepoint_alert", "polymarket_shift",
            "gpr_spike", "vvix_spike",
        ],
    }


# ============================================================
# 7. GEOPOLITICAL ATTRACTIVENESS RANKING (Spec Teil 3 §7)
# ============================================================

# Demographic dividend scores (structural, slow-changing)
DEMOGRAPHIC_DIVIDEND = {
    "USA": 0.55,
    "CHINA": 0.25,
    "EU": 0.30,
    "INDIA": 0.85,
    "JP_KR_TW": 0.15,
    "GULF": 0.50,
    "REST_EM": 0.65,
}

# MCap/GDP proxies (structural estimates)
MCAP_TO_GDP = {
    "USA": 1.80,
    "CHINA": 0.70,
    "EU": 0.60,
    "INDIA": 0.95,
    "JP_KR_TW": 1.10,
    "GULF": 0.80,
    "REST_EM": 0.40,
}


def compute_attractiveness(power_scores, sanctions_result=None):
    """
    Geopolitical Attractiveness Ranking — V1 simplified.
    Spec Teil 3 §7.

    Formula (V2):
      score = power_score * 0.30
            + max(0, momentum) * 10 * 0.25
            + -sanctions_severity * 0.15
            + -(mcap/gdp / 2) * 0.15
            + demographic_dividend * 0.15
    """
    ranking = []

    # Extract per-region severity scores from SIT (default 0 if not available)
    sev_scores = {}
    if sanctions_result and isinstance(sanctions_result.get("severity_scores"), dict):
        sev_scores = sanctions_result["severity_scores"]

    for region in REGIONS:
        ps = power_scores.get(region, {})
        score = ps.get("score", 50)
        momentum = ps.get("momentum", 0)
        sanctions_sev = sev_scores.get(region, 0)

        attractiveness = (
            score * 0.30
            + max(0, momentum) * 10 * 0.25
            + -sanctions_sev * 0.15
            + -(MCAP_TO_GDP.get(region, 0.80) / 2) * 0.15
            + DEMOGRAPHIC_DIVIDEND.get(region, 0.50) * 0.15
        )

        ranking.append({
            "region": region,
            "composite_score": round(attractiveness, 2),
            "components": {
                "power_score_weighted": round(score * 0.30, 2),
                "momentum_weighted": round(max(0, momentum) * 10 * 0.25, 2),
                "sanctions_discount": round(-sanctions_sev * 0.15, 2),
                "valuation_edge": round(-(MCAP_TO_GDP.get(region, 0.80) / 2) * 0.15, 2),
                "demographic_premium": round(DEMOGRAPHIC_DIVIDEND.get(region, 0.50) * 0.15, 2),
            },
        })

    # Sort by composite score descending
    ranking.sort(key=lambda x: x["composite_score"], reverse=True)

    # Assign ranks
    for i, entry in enumerate(ranking):
        entry["rank"] = i + 1

    return ranking


# ============================================================
# 8. LIQUIDITY DISTRIBUTION MAP (Spec Teil 3 §8)
# ============================================================

def compute_liquidity_map():
    """
    Liquidity Distribution Map — V1 Placeholder.
    Spec Teil 3 §8.

    Requires V16 production data + TIC + BIS + LLM assessment.
    Will be implemented in Etappe 3.
    """
    return {
        "method": "V1: Placeholder — requires V16 production read + LLM assessment.",
        "global_liquidity_regime": "UNKNOWN",
        "regional_flow_direction": {},
        "data_note": "Liquidity Map requires cross-system data (V16) and LLM — Etappe 3.",
    }


# ============================================================
# 9. CORRELATION REGIME MONITOR (Spec Teil 3 §9)
# ============================================================

def compute_correlation_regime():
    """
    Correlation Regime Monitor — V1 Placeholder.
    Spec Teil 3 §9.

    Requires daily close time-series for correlation calculation.
    Current data_collection.py only stores latest snapshots.
    Will be implemented when data_collection is extended with history arrays.
    """
    return {
        "method": "V1: Placeholder — requires time-series data extension in data_collection.py.",
        "current": "NORMAL",
        "key_correlations": {},
        "regime_stability": None,
        "transition_probability": None,
        "data_note": "Correlation regime requires daily close history — not yet available.",
    }


# ============================================================
# HELPER: TREND COMPUTATION
# ============================================================

def _compute_trend(current, previous, threshold=0.5):
    """Classify trend based on delta."""
    if previous is None:
        return "STABLE"
    delta = current - previous
    if delta > threshold:
        return "RISING"
    elif delta < -threshold:
        return "FALLING"
    return "STABLE"


# ============================================================
# GPR INDEX EXTRACTION
# ============================================================

def _extract_gpr(validated_data, previous_overlays):
    """
    Extract GPR Index current value, trend, and Z-Score.
    Used by Phase 5 (Status Determination) via overlay return dict.
    """
    gpr_data = validated_data.get("gpr")

    if isinstance(gpr_data, dict) and gpr_data.get("gpr_global") is not None:
        gpr_current = gpr_data["gpr_global"]
    else:
        gpr_current = 100  # Baseline default

    # Z-Score (baseline ~110, std ~40)
    gpr_zscore = _compute_zscore(gpr_current, 110, 40)

    # Trend vs previous
    prev_gpr = _safe_get(previous_overlays, "gpr_index_current", default=100)
    gpr_trend = _compute_trend(gpr_current, prev_gpr, threshold=10)

    return gpr_current, gpr_trend, gpr_zscore


# ============================================================
# MAIN ENTRY POINT — phase4_overlay_computation()
# ============================================================

def phase4_overlay_computation(scores, validated_data, previous_overlays, run_type="WEEKLY"):
    """
    Phase 4: Compute all 9 overlays.
    Called by main.py after Phase 3 (Scoring Engine).

    Args:
        scores: Full Phase 3 result dict containing:
            - scores: dict[dim][region] = normalized score
            - momenta: dict[dim][region] = momentum
            - accelerations: dict[dim][region] = acceleration
            - power_scores: dict[region] = {score, momentum, acceleration}
            - gap_data: {gap, trend, gap_momentum}
            - data_quality: dict[region] = quality info
        validated_data: Phase 2 validated raw data (FRED, yfinance, etc.)
        previous_overlays: Previous overlay results (from Sheet or empty dict)
        run_type: "WEEKLY", "QUARTERLY", or "AD_HOC"

    Returns:
        dict with all overlay results, consumed by Phase 5 + Phase 6 + Sheet Writer.
    """
    print("[Phase 4] Overlay Computation (Etappe 3)...")

    thresholds = _load_thresholds()

    # Extract sub-dicts from scores (handle both full result and empty fallback)
    gap_data = scores.get("gap_data", {"gap": 50, "trend": "STABLE", "gap_momentum": 0})
    power_scores = scores.get("power_scores", {r: {"score": 50, "momentum": 0, "acceleration": 0} for r in REGIONS})

    # --- 1. Feedback Loops ---
    try:
        feedback_loops = compute_feedback_loops(scores, gap_data)
        active_loop_count = len([l for l in feedback_loops if l["status"] == "ACTIVE"])
        print(f"  Feedback Loops: {len(feedback_loops)} active/latent, {active_loop_count} ACTIVE")
    except Exception as e:
        print(f"  Feedback Loops ERROR: {e}")
        feedback_loops = []

    # --- 2. SCSI ---
    try:
        scsi = compute_scsi(validated_data, previous_overlays)
        print(f"  SCSI: {scsi['composite']} ({scsi['trend']}), {scsi['active_chokepoint_alerts']} alerts")
    except Exception as e:
        print(f"  SCSI ERROR: {e}")
        scsi = {"composite": 0, "trend": "STABLE", "active_chokepoint_alerts": 0,
                "chokepoints": {}, "shipping_stress": 0, "bdi_zscore": 0,
                "freight_ppi_zscore": 0, "cascade_score": 0, "dimension_modifiers": {}}

    # --- 3. DDI ---
    try:
        ddi = compute_ddi(validated_data, previous_overlays)
        print(f"  DDI: {ddi['composite']} ({ddi['trend']})")
    except Exception as e:
        print(f"  DDI ERROR: {e}")
        ddi = {"composite": 0, "trend": "STABLE", "acceleration": 0, "momentum": 0, "components": {}}

    # --- 4. FDP ---
    try:
        fdp = compute_fdp(validated_data, previous_overlays)
        usa_fdp = fdp.get("USA", {}).get("composite_proximity", 0)
        print(f"  FDP USA: {usa_fdp:.2f}")
    except Exception as e:
        print(f"  FDP ERROR: {e}")
        fdp = {r: {"composite_proximity": 0, "trend": "STABLE"} for r in REGIONS}

    # --- 5. SIT (Brave Search + LLM) ---
    try:
        sanctions = compute_sit(run_type=run_type)
        src = sanctions.get("source", "?")
        print(f"  SIT: {sanctions['escalation_trend']} [{src}]")
    except Exception as e:
        print(f"  SIT ERROR: {e}"); traceback.print_exc()
        sanctions = _format_sit_output(_build_sit_fallback(), source="ERROR_FALLBACK")

    # --- 6. EWI ---
    try:
        ewi = compute_ewi(validated_data, scsi, previous_overlays, thresholds)
        print(f"  EWI: {ewi['active_signals']}/{ewi['total_signals']} active, severity={ewi['severity']}")
    except Exception as e:
        print(f"  EWI ERROR: {e}")
        ewi = {"active_signals": 0, "total_signals": 10, "severity": "NONE", "active_details": []}

    # --- 7. Attractiveness (now uses SIT severity scores) ---
    try:
        attractiveness = compute_attractiveness(power_scores, sanctions_result=sanctions)
        top = attractiveness[0] if attractiveness else {}
        print(f"  Attractiveness: #1 = {top.get('region', '?')} ({top.get('composite_score', 0):.1f})")
    except Exception as e:
        print(f"  Attractiveness ERROR: {e}")
        attractiveness = []

    # --- 8. Liquidity Map (Placeholder) ---
    liquidity_map = compute_liquidity_map()

    # --- 9. Correlation Regime (Placeholder) ---
    correlation_regime = compute_correlation_regime()

    # --- GPR Index ---
    gpr_current, gpr_trend, gpr_zscore = _extract_gpr(validated_data, previous_overlays)
    print(f"  GPR: {gpr_current:.0f} ({gpr_trend}), Z={gpr_zscore:+.2f}")

    # --- Max scenario shift (from previous — computed here as info) ---
    max_scenario_shift = _safe_get(previous_overlays, "max_scenario_shift", default=0)

    # --- Summary ---
    print(f"  Phase 4 complete — all 9 overlays computed")

    return {
        # Feedback Loops
        "feedback_loops": feedback_loops,

        # SCSI
        "scsi": scsi,

        # DDI
        "ddi": ddi,

        # FDP
        "fdp": fdp,

        # SIT (Brave Search + LLM)
        "sanctions": sanctions,

        # EWI
        "ewi": ewi,

        # Attractiveness Ranking
        "attractiveness": attractiveness,

        # Liquidity Map (Placeholder)
        "liquidity_map": liquidity_map,

        # Correlation Regime (Placeholder)
        "correlation_regime": correlation_regime,

        # GPR (extracted for Phase 5)
        "gpr_index_current": gpr_current,
        "gpr_index_trend": gpr_trend,
        "gpr_index_zscore": gpr_zscore,

        # SCSI Dimension Modifiers (computed, NOT applied yet)
        "scsi_dimension_modifiers": scsi.get("dimension_modifiers", {}),

        # Scenario shift (carried from previous)
        "max_scenario_shift": max_scenario_shift,
    }
