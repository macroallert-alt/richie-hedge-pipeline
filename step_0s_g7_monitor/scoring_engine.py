"""
step_0s_g7_monitor/scoring_engine.py
Phase 3: Scoring Engine

Berechnet alle 84 Signal-Triplets (12 Dimensionen x 7 Regionen).
Jeder Datenpunkt bekommt ein SIGNAL-TRIPLET:
  - score:        Wo stehen wir? (0-100, normalisiert relativ)
  - momentum:     Wohin bewegen wir uns? (Delta/Quartal)
  - acceleration: Beschleunigt oder bremst die Bewegung? (Delta-Delta)

QUANT_DIMENSIONS (Stufe 1+2, berechnet hier):
  D1_economic, D2_demographics, D4_energy, D6_fiscal, D7_currency,
  D8_capital_mkt, D9_flows

LLM_DIMENSIONS (Stufe 3, Placeholder — Etappe 3):
  D3_technology, D5_military, D10_social, D11_geopolitical, D12_feedback

Normalisierung: RELATIV innerhalb jeder Dimension (10-95 Range).
  Beste Region ~90-95, schlechteste ~10-15.
  Direkte Vergleichbarkeit: USA D1=85 vs China D1=78 = 7 Punkte Vorsprung.

Power Score = gewichteter Durchschnitt aller 12 Dimension-Scores.
USA-China Gap = USA Power Score - China Power Score.
"""

import os
import json
from datetime import datetime, timezone

# ============================================================
# CONSTANTS
# ============================================================

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

DIMENSION_WEIGHTS = {
    "D1_economic":      0.12,
    "D2_demographics":  0.10,
    "D3_technology":    0.12,
    "D4_energy":        0.08,
    "D5_military":      0.08,
    "D6_fiscal":        0.10,
    "D7_currency":      0.10,
    "D8_capital_mkt":   0.10,
    "D9_flows":         0.05,
    "D10_social":       0.05,
    "D11_geopolitical": 0.05,
    "D12_feedback":     0.05,
}

QUANT_DIMENSIONS = [
    "D1_economic", "D2_demographics", "D4_energy",
    "D6_fiscal", "D7_currency", "D8_capital_mkt", "D9_flows",
]

LLM_DIMENSIONS = [
    "D3_technology", "D5_military", "D10_social",
    "D11_geopolitical", "D12_feedback",
]

# Momentum periods (how many periods back for delta)
DIMENSION_MOMENTUM_PERIODS = {
    "D1_economic":     1,   # QoQ
    "D2_demographics": 4,   # Annualized / 4
    "D4_energy":       1,   # 3-month avg
    "D6_fiscal":       1,   # QoQ
    "D7_currency":     1,   # 3-month avg
    "D8_capital_mkt":  1,   # 3-month avg
    "D9_flows":        1,   # QoQ
}

# Freshness confidence multipliers
FRESHNESS_CONFIDENCE = {
    "FRESH":       1.0,
    "RECENT":      0.9,
    "STALE":       0.7,
    "STRUCTURAL":  0.9,
    "UNAVAILABLE": 0.5,
}

# Known reliability discounts per region
REGION_RELIABILITY_DISCOUNT = {
    "USA":      0.00,
    "CHINA":    0.25,
    "EU":       0.00,
    "INDIA":    0.15,
    "JP_KR_TW": 0.00,
    "GULF":     0.30,
    "REST_EM":  0.35,
}


# ============================================================
# RAW SCORE EXTRACTION
# ============================================================

def extract_d1_economic(validated_data):
    """D1 Economic Weight & Productivity — raw values per region."""
    raw = {}
    imf = validated_data.get("imf_weo", {})
    fred = validated_data.get("fred", {})
    wb = validated_data.get("worldbank", {})

    for region in REGIONS:
        score = 50.0  # Default neutral
        components = []

        # GDP Growth (IMF WEO)
        gdp_growth = imf.get(f"NGDP_RPCH_{region}", {})
        if gdp_growth and gdp_growth.get("value") is not None:
            components.append(gdp_growth["value"])

        # GDP Nominal (IMF WEO) — as share of total
        gdp_nominal = imf.get(f"NGDPD_{region}", {})
        if gdp_nominal and gdp_nominal.get("value") is not None:
            components.append(gdp_nominal["value"] / 1000)  # Scale down

        # US-specific: FRED GDP
        if region == "USA":
            gdp_fred = fred.get("GDP", {})
            if gdp_fred and gdp_fred.get("value") is not None:
                components.append(gdp_fred["value"] / 1000)

        # World Bank GDP Growth
        wb_growth = wb.get(f"NY.GDP.MKTP.KD.ZG_{region}", {})
        if wb_growth and wb_growth.get("value") is not None:
            components.append(wb_growth["value"])

        if components:
            score = sum(components) / len(components)

        raw[region] = score

    return raw


def extract_d2_demographics(validated_data):
    """D2 Demographics & Human Capital — raw values per region."""
    raw = {}
    wb = validated_data.get("worldbank", {})

    for region in REGIONS:
        components = []

        # Fertility Rate (higher = younger population = higher score)
        fert = wb.get(f"SP.DYN.TFRT.IN_{region}", {})
        if fert and fert.get("value") is not None:
            components.append(fert["value"] * 10)  # Scale: 2.0 -> 20

        # Dependency Ratio (INVERSE — lower = better)
        dep = wb.get(f"SP.POP.DPND_{region}", {})
        if dep and dep.get("value") is not None:
            components.append(100 - dep["value"])  # Invert

        # Education Spending/GDP
        edu = wb.get(f"SE.XPD.TOTL.GD.ZS_{region}", {})
        if edu and edu.get("value") is not None:
            components.append(edu["value"] * 5)  # Scale: 5% -> 25

        if components:
            raw[region] = sum(components) / len(components)
        else:
            raw[region] = 50.0

    return raw


def extract_d4_energy(validated_data):
    """D4 Energy & Resource Sovereignty — raw values per region."""
    raw = {}
    yf = validated_data.get("yfinance", {})

    # Oil price context (global)
    oil = yf.get("CL=F", {})
    oil_price = oil.get("close", 70) if oil else 70

    # Energy exporters get higher scores when oil is high
    # Energy importers get lower scores when oil is high
    oil_factor = oil_price / 70  # Normalized to ~$70 baseline

    energy_profile = {
        "USA":      0.8,   # Net exporter now
        "CHINA":   -0.6,   # Major importer
        "EU":      -0.5,   # Major importer
        "INDIA":   -0.7,   # Heavy importer
        "JP_KR_TW": -0.8,  # Very heavy importer
        "GULF":     1.0,   # Maximum exporter
        "REST_EM":  0.2,   # Mixed (Brazil exporter, others import)
    }

    for region in REGIONS:
        profile = energy_profile.get(region, 0)
        # Higher oil * exporter = good, higher oil * importer = bad
        score = 50 + (profile * oil_factor * 20)
        raw[region] = max(10, min(90, score))

    return raw


def extract_d6_fiscal(validated_data):
    """D6 Fiscal Health & Debt Dynamics — raw values per region (INVERSE)."""
    raw = {}
    imf = validated_data.get("imf_weo", {})
    fred = validated_data.get("fred", {})
    wb = validated_data.get("worldbank", {})

    for region in REGIONS:
        components = []

        # Govt Debt/GDP (INVERSE — higher debt = lower score)
        debt_gdp = imf.get(f"GGXWDG_NGDP_{region}", {})
        if debt_gdp and debt_gdp.get("value") is not None:
            # 0% debt = score 100, 200% debt = score 0
            components.append(max(0, 100 - debt_gdp["value"] / 2))

        # WB Central Govt Debt
        wb_debt = wb.get(f"GC.DOD.TOTL.GD.ZS_{region}", {})
        if wb_debt and wb_debt.get("value") is not None:
            components.append(max(0, 100 - wb_debt["value"] / 2))

        # US-specific: Interest Payments / Revenue
        if region == "USA":
            interest = fred.get("A091RC1Q027SBEA", {})
            revenue = fred.get("FGRECPT", {})
            if (interest and interest.get("value") is not None
                    and revenue and revenue.get("value") is not None
                    and revenue["value"] > 0):
                itr = interest["value"] / revenue["value"]
                # 0% = score 100, 30% = score 0
                components.append(max(0, 100 - itr * 100 / 0.30 * 100))

            # Deficit/GDP (INVERSE)
            deficit = fred.get("FYFSGDA188S", {})
            if deficit and deficit.get("value") is not None:
                # 0% deficit = 100, -10% deficit = 0
                components.append(max(0, 100 + deficit["value"] * 10))

        # Inflation (moderate = good, high = bad)
        inflation = imf.get(f"PCPIPCH_{region}", {})
        if inflation and inflation.get("value") is not None:
            inf_val = inflation["value"]
            # 2% = perfect (100), deviation penalized
            inf_score = max(0, 100 - abs(inf_val - 2) * 15)
            components.append(inf_score)

        if components:
            raw[region] = sum(components) / len(components)
        else:
            raw[region] = 50.0

    return raw


def extract_d7_currency(validated_data):
    """D7 Currency & Reserve Status — raw values per region."""
    raw = {}
    cofer = validated_data.get("imf_cofer", {})
    yf = validated_data.get("yfinance", {})

    # COFER reserve shares
    usd_share = cofer.get("USD_share", 58.4)
    eur_share = cofer.get("EUR_share", 19.8)
    cny_share = cofer.get("CNY_share", 2.3)

    # DXY strength
    dxy_data = yf.get("DX-Y.NYB", {})
    dxy = dxy_data.get("close", 100) if dxy_data else 100

    # Gold price (indicator of reserve diversification)
    gold_data = yf.get("GC=F", {})
    gold = gold_data.get("close", 2000) if gold_data else 2000

    # Reserve currency dominance drives score
    reserve_scores = {
        "USA":      usd_share,       # ~58
        "CHINA":    cny_share * 10,  # ~23 (scaled up since share is small)
        "EU":       eur_share * 2,   # ~40
        "INDIA":    5,               # Minimal reserve currency
        "JP_KR_TW": 15,             # JPY ~5% reserves
        "GULF":     8,              # Pegged currencies
        "REST_EM":  3,              # Minimal
    }

    for region in REGIONS:
        raw[region] = reserve_scores.get(region, 10)

    return raw


def extract_d8_capital_mkt(validated_data):
    """D8 Capital Market Depth & Openness — raw values per region."""
    raw = {}
    yf = validated_data.get("yfinance", {})
    fred = validated_data.get("fred", {})

    # Market indices as proxy for market depth
    index_map = {
        "USA":      "^GSPC",
        "CHINA":    "^HSI",
        "EU":       "^STOXX50E",
        "INDIA":    "^NSEI",
        "JP_KR_TW": "^N225",
    }

    # Financial stress (US)
    stress = fred.get("STLFSI4", {})
    stress_val = stress.get("value", 0) if stress else 0

    # HY Spread
    hy = fred.get("BAMLH0A0HYM2", {})
    hy_val = hy.get("value", 400) if hy else 400

    for region in REGIONS:
        components = []

        # Market cap proxy (from index level)
        ticker = index_map.get(region)
        if ticker and yf.get(ticker):
            idx_data = yf[ticker]
            if idx_data and idx_data.get("close"):
                # Normalize: higher index = deeper market
                components.append(min(90, idx_data["close"] / 100))

        # US-specific: Financial Stress and HY Spread
        if region == "USA":
            # Stress: negative = loose, positive = tight
            stress_score = max(0, 80 - stress_val * 20)
            components.append(stress_score)
            # HY Spread: lower = healthier (300bp = good, 800bp = bad)
            hy_score = max(0, 100 - (hy_val - 300) / 5)
            components.append(hy_score)

        # Capital openness penalty for China
        if region == "CHINA":
            components.append(30)  # Capital controls penalty

        if components:
            raw[region] = sum(components) / len(components)
        else:
            raw[region] = 50.0

    return raw


def extract_d9_flows(validated_data):
    """D9 Capital Flow Dynamics — raw values per region."""
    raw = {}
    wb = validated_data.get("worldbank", {})

    for region in REGIONS:
        components = []

        # Current Account/GDP
        ca = wb.get(f"BN.CAB.XOKA.GD.ZS_{region}", {})
        if ca and ca.get("value") is not None:
            # Positive CA = inflows, negative = outflows
            # +5% = score 80, -5% = score 20
            components.append(50 + ca["value"] * 6)

        # Total Reserves
        reserves = wb.get(f"FI.RES.TOTL.CD_{region}", {})
        if reserves and reserves.get("value") is not None:
            # Normalize: $1T+ reserves = high score
            res_score = min(90, reserves["value"] / 1e10)
            components.append(res_score)

        if components:
            raw[region] = max(10, min(90, sum(components) / len(components)))
        else:
            raw[region] = 50.0

    return raw


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_relative(raw_values_by_region):
    """
    Normalize RELATIVE within a dimension.
    Best region gets ~90-95, worst gets ~10-15.
    Scale: 10-95 range (not 0-100, avoids extremes).
    """
    values = {r: v for r, v in raw_values_by_region.items() if v is not None}

    if not values:
        return {r: 50.0 for r in REGIONS}

    min_val = min(values.values())
    max_val = max(values.values())
    range_val = max_val - min_val if max_val != min_val else 1

    normalized = {}
    for region in REGIONS:
        raw = values.get(region, (min_val + max_val) / 2)
        normalized[region] = round(10 + ((raw - min_val) / range_val) * 85, 1)

    return normalized


# ============================================================
# MOMENTUM & ACCELERATION
# ============================================================

def compute_momentum(current_score, previous_score, periods=1):
    """Momentum = Delta Score per quarter."""
    if previous_score is None:
        return 0.0
    return round((current_score - previous_score) / max(periods, 1), 2)


def compute_acceleration(current_momentum, previous_momentum):
    """Acceleration = Delta Momentum (second derivative)."""
    if previous_momentum is None:
        return 0.0
    return round(current_momentum - previous_momentum, 2)


# ============================================================
# POWER SCORES
# ============================================================

def compute_power_score(region, dimension_scores, dimension_weights):
    """Power Score = weighted average of all 12 dimension scores."""
    total = 0.0
    weight_sum = 0.0

    for dim, weight in dimension_weights.items():
        score = dimension_scores.get(dim, {}).get(region)
        if score is not None:
            total += score * weight
            weight_sum += weight

    if weight_sum > 0:
        return round(total / weight_sum * (sum(dimension_weights.values()) / weight_sum), 1)
    return 50.0


def compute_all_power_scores(all_scores, all_momenta, all_accelerations):
    """Compute power scores, momenta, accelerations for all regions."""
    result = {}

    for region in REGIONS:
        ps = compute_power_score(region, all_scores, DIMENSION_WEIGHTS)

        # Power momentum = weighted avg of dimension momenta
        mom_total = 0.0
        mom_weight = 0.0
        for dim, weight in DIMENSION_WEIGHTS.items():
            m = all_momenta.get(dim, {}).get(region)
            if m is not None:
                mom_total += m * weight
                mom_weight += weight
        power_momentum = round(mom_total / mom_weight, 2) if mom_weight > 0 else 0.0

        # Power acceleration
        acc_total = 0.0
        acc_weight = 0.0
        for dim, weight in DIMENSION_WEIGHTS.items():
            a = all_accelerations.get(dim, {}).get(region)
            if a is not None:
                acc_total += a * weight
                acc_weight += weight
        power_acceleration = round(acc_total / acc_weight, 2) if acc_weight > 0 else 0.0

        result[region] = {
            "score": ps,
            "momentum": power_momentum,
            "acceleration": power_acceleration,
        }

    return result


def compute_usa_china_gap(power_scores):
    """
    The central metric of the G7 Monitor.
    Gap = USA Power Score - China Power Score.
    Trend from momentum difference.
    """
    usa = power_scores.get("USA", {})
    china = power_scores.get("CHINA", {})

    gap = round(usa.get("score", 50) - china.get("score", 50), 1)
    gap_momentum = round(usa.get("momentum", 0) - china.get("momentum", 0), 2)

    if gap_momentum < -0.5:
        trend = "CLOSING"
    elif gap_momentum > 0.5:
        trend = "WIDENING"
    else:
        trend = "STABLE"

    return {
        "gap": gap,
        "trend": trend,
        "gap_momentum": gap_momentum,
    }


# ============================================================
# DATA QUALITY
# ============================================================

def compute_data_quality(region, freshness_by_dimension):
    """
    Data Quality Score (0-1) per region.
    Based on available sources, freshness, known reliability issues.
    """
    available_dims = 0
    stale_dims = 0

    for dim in DIMENSION_WEIGHTS:
        fresh = freshness_by_dimension.get(dim, "UNAVAILABLE")
        if fresh != "UNAVAILABLE":
            available_dims += 1
        if fresh in ("STALE", "UNAVAILABLE"):
            stale_dims += 1

    total_dims = len(DIMENSION_WEIGHTS)
    stale_pct = stale_dims / total_dims if total_dims > 0 else 1.0
    reliability_discount = REGION_RELIABILITY_DISCOUNT.get(region, 0)

    quality = (available_dims / total_dims) * (1 - stale_pct) * (1 - reliability_discount)
    ci_half_width = (1 - quality) * 15

    if quality > 0.75:
        reliability = "HIGH"
    elif quality > 0.55:
        reliability = "MEDIUM"
    else:
        reliability = "LOW"

    return {
        "quality": round(quality, 2),
        "sources_available": available_dims,
        "stale_pct": round(stale_pct, 2),
        "reliability": reliability,
        "power_score_ci": f"+/-{round(ci_half_width, 1)}",
    }


# ============================================================
# MAIN SCORING FUNCTION
# ============================================================

def phase3_scoring_engine(validated_data, freshness, previous_scores):
    """
    Phase 3: Compute all 84 Signal-Triplets.

    Returns:
      - scores: dict[dim][region] = normalized score (0-100)
      - momenta: dict[dim][region] = momentum (delta/quarter)
      - accelerations: dict[dim][region] = acceleration (delta-delta)
      - power_scores: dict[region] = {score, momentum, acceleration}
      - gap_data: {gap, trend, gap_momentum}
      - data_quality: dict[region] = quality info
    """
    print("[Phase 3] Scoring Engine (84 Triplets)...")

    # ---- Step 1: Extract raw scores per quant dimension ----
    extractors = {
        "D1_economic":    extract_d1_economic,
        "D2_demographics": extract_d2_demographics,
        "D4_energy":      extract_d4_energy,
        "D6_fiscal":      extract_d6_fiscal,
        "D7_currency":    extract_d7_currency,
        "D8_capital_mkt": extract_d8_capital_mkt,
        "D9_flows":       extract_d9_flows,
    }

    raw_scores = {}
    for dim, extractor in extractors.items():
        try:
            raw_scores[dim] = extractor(validated_data)
        except Exception as e:
            print(f"  {dim} extraction failed: {e}")
            raw_scores[dim] = {r: 50.0 for r in REGIONS}

    # ---- Step 2: Normalize (relative within each dimension) ----
    scores = {}
    for dim in QUANT_DIMENSIONS:
        scores[dim] = normalize_relative(raw_scores.get(dim, {}))

    # ---- Step 3: Momentum (vs previous period) ----
    momenta = {}
    for dim in QUANT_DIMENSIONS:
        momenta[dim] = {}
        periods = DIMENSION_MOMENTUM_PERIODS.get(dim, 1)
        for region in REGIONS:
            prev = (
                previous_scores.get(dim, {}).get(region, {}).get("score")
                if isinstance(previous_scores.get(dim, {}).get(region), dict)
                else previous_scores.get(dim, {}).get(region)
            )
            if prev is None:
                prev = scores[dim].get(region, 50)
            momenta[dim][region] = compute_momentum(
                scores[dim].get(region, 50), prev, periods
            )

    # ---- Step 4: Acceleration ----
    accelerations = {}
    for dim in QUANT_DIMENSIONS:
        accelerations[dim] = {}
        for region in REGIONS:
            prev_mom = (
                previous_scores.get(dim, {}).get(region, {}).get("momentum")
                if isinstance(previous_scores.get(dim, {}).get(region), dict)
                else 0
            )
            accelerations[dim][region] = compute_acceleration(
                momenta[dim].get(region, 0), prev_mom
            )

    # ---- Step 5: LLM Dimensions — Placeholder (Etappe 3) ----
    for dim in LLM_DIMENSIONS:
        scores[dim] = {}
        momenta[dim] = {}
        accelerations[dim] = {}
        for region in REGIONS:
            # Use neutral defaults — will be filled by LLM in Etappe 3
            scores[dim][region] = 50.0
            momenta[dim][region] = 0.0
            accelerations[dim][region] = 0.0

    # ---- Step 6: Power Scores ----
    power_scores = compute_all_power_scores(scores, momenta, accelerations)

    # ---- Step 7: USA-China Gap ----
    gap_data = compute_usa_china_gap(power_scores)

    # ---- Step 8: Data Quality ----
    data_quality = {}
    for region in REGIONS:
        data_quality[region] = compute_data_quality(region, freshness)

    # ---- Summary ----
    print(f"  Quant dimensions: {len(QUANT_DIMENSIONS)}")
    print(f"  LLM dimensions (stub): {len(LLM_DIMENSIONS)}")
    print(f"  USA-China Gap: {gap_data['gap']} ({gap_data['trend']})")
    for region in REGIONS:
        ps = power_scores[region]
        dq = data_quality[region]
        print(
            f"  {region:10s}: Score={ps['score']:5.1f}  "
            f"Mom={ps['momentum']:+5.2f}  "
            f"Acc={ps['acceleration']:+5.2f}  "
            f"DQ={dq['reliability']}"
        )

    return {
        "scores": scores,
        "momenta": momenta,
        "accelerations": accelerations,
        "power_scores": power_scores,
        "gap_data": gap_data,
        "data_quality": data_quality,
        "quant_dimensions_complete": len(QUANT_DIMENSIONS),
        "llm_dimensions_pending": len(LLM_DIMENSIONS),
    }
