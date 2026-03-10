"""
step_0s_g7_monitor/scenario_engine.py
Phase 6: Scenario Engine — Etappe 3 VOLLSTAENDIG

Implements the full Scenario Engine per G7_WORLD_ORDER_MONITOR_SPEC_TEIL4.md:
  1. Phase 1: Quantitative Pre-Assessment (deterministic, Flat Prior + Adjustments)
  2. Phase 2: Polymarket Calibration (deterministic, Bayesian Update)
  3. Phase 3: LLM Synthesis (Sonnet T=0.3, final Probabilities + Thesis)
  4. Phase 4: Operator Override Check (deterministic, reads G7_OPERATOR_OVERRIDES)
  5. Tilt Computation (ASSET_EXPOSURE_VECTORS x Scenario Probabilities)
  6. PermOpt Allocation (DDI-based, 2-5%)
  7. Thesis Stress Test (LLM Attack on dominant thesis)
  8. Drift Tracking (vs History, Anomaly Detection)
  9. Interim Trigger Evaluation

4 Scenarios:
  A) Managed Decline (Baseline)
  B) Conflict Escalation
  C) US Renewal
  D) Multipolar Chaos

Run Types:
  QUARTERLY / AD_HOC: Full 9-step run
  WEEKLY: Interim Trigger Check only; if trigger fires -> Phase 1+3+4+Tilts
"""

import os
import json
import time
import traceback
from datetime import datetime, timezone

# ============================================================
# CONSTANTS
# ============================================================

SCENARIOS = ["managed_decline", "conflict_escalation", "us_renewal", "multipolar_chaos"]
REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

MIN_PROBABILITY = 0.05
MAX_PROBABILITY = 0.70
MAX_LLM_SHIFT_PER_SCENARIO = 0.15

PERMOPT_ELIGIBLE = {"VWO", "INDA", "FXI", "KWEB", "EEM", "EMB",
                    "GLD", "SLV", "DBC", "BTC"}

# Asset Exposure Vectors — Spec Teil 4 §6
# Positive = asset benefits from scenario, Negative = asset suffers
ASSET_EXPOSURE_VECTORS = {
    # Equities
    "SPY":  {"managed_decline": -0.3, "conflict_escalation": -0.8, "us_renewal": +0.9, "multipolar_chaos": -0.5},
    "QQQ":  {"managed_decline": -0.4, "conflict_escalation": -0.9, "us_renewal": +1.0, "multipolar_chaos": -0.6},
    "IWM":  {"managed_decline": -0.2, "conflict_escalation": -0.6, "us_renewal": +0.7, "multipolar_chaos": -0.4},
    "EFA":  {"managed_decline": +0.1, "conflict_escalation": -0.5, "us_renewal": -0.2, "multipolar_chaos": -0.3},
    "EEM":  {"managed_decline": +0.3, "conflict_escalation": -0.7, "us_renewal": -0.2, "multipolar_chaos": +0.2},
    "VWO":  {"managed_decline": +0.5, "conflict_escalation": -0.6, "us_renewal": -0.3, "multipolar_chaos": +0.3},
    "INDA": {"managed_decline": +0.4, "conflict_escalation": -0.3, "us_renewal": -0.1, "multipolar_chaos": +0.4},
    "FXI":  {"managed_decline": +0.2, "conflict_escalation": -1.0, "us_renewal": -0.4, "multipolar_chaos": +0.1},
    "KWEB": {"managed_decline": +0.1, "conflict_escalation": -1.0, "us_renewal": -0.5, "multipolar_chaos": +0.0},
    # Fixed Income
    "TLT":  {"managed_decline": +0.2, "conflict_escalation": +0.5, "us_renewal": -0.5, "multipolar_chaos": +0.1},
    "IEF":  {"managed_decline": +0.1, "conflict_escalation": +0.3, "us_renewal": -0.3, "multipolar_chaos": +0.1},
    "SHY":  {"managed_decline": +0.0, "conflict_escalation": +0.1, "us_renewal": -0.1, "multipolar_chaos": +0.0},
    "TIP":  {"managed_decline": +0.3, "conflict_escalation": +0.2, "us_renewal": -0.2, "multipolar_chaos": +0.4},
    "HYG":  {"managed_decline": -0.2, "conflict_escalation": -0.6, "us_renewal": +0.4, "multipolar_chaos": -0.4},
    "EMB":  {"managed_decline": +0.1, "conflict_escalation": -0.5, "us_renewal": -0.1, "multipolar_chaos": -0.2},
    # Commodities & Alternatives
    "GLD":  {"managed_decline": +0.7, "conflict_escalation": +0.9, "us_renewal": -0.3, "multipolar_chaos": +0.8},
    "SLV":  {"managed_decline": +0.5, "conflict_escalation": +0.6, "us_renewal": -0.1, "multipolar_chaos": +0.6},
    "DBC":  {"managed_decline": +0.2, "conflict_escalation": +0.3, "us_renewal": +0.2, "multipolar_chaos": +0.5},
    "USO":  {"managed_decline": +0.1, "conflict_escalation": +0.5, "us_renewal": +0.3, "multipolar_chaos": +0.3},
    # Crypto
    "BTC":  {"managed_decline": +0.4, "conflict_escalation": -0.3, "us_renewal": +0.3, "multipolar_chaos": +0.5},
    # Sector
    "XLK":  {"managed_decline": -0.3, "conflict_escalation": -0.7, "us_renewal": +1.0, "multipolar_chaos": -0.5},
    "XLE":  {"managed_decline": +0.1, "conflict_escalation": +0.4, "us_renewal": +0.2, "multipolar_chaos": +0.3},
    "XLF":  {"managed_decline": -0.2, "conflict_escalation": -0.5, "us_renewal": +0.6, "multipolar_chaos": -0.3},
    "XLV":  {"managed_decline": +0.0, "conflict_escalation": -0.2, "us_renewal": +0.3, "multipolar_chaos": -0.1},
    "XLU":  {"managed_decline": +0.1, "conflict_escalation": +0.0, "us_renewal": -0.1, "multipolar_chaos": +0.1},
}

# Polymarket keyword mapping — Spec Teil 4 §3
SCENARIO_KEYWORDS = {
    "conflict_escalation": [
        "invade", "invasion", "military conflict", "war",
        "taiwan", "south china sea", "nuclear",
        "iran strike", "nato article 5",
    ],
    "us_renewal": [
        "us gdp growth", "us productivity", "ai revolution",
        "us deficit reduction", "reshoring", "chips act success",
    ],
    "managed_decline": [
        "de-dollarization", "brics currency", "reserve currency",
        "us debt ceiling", "government shutdown",
        "dollar share decline",
    ],
    "multipolar_chaos": [
        "global recession", "trade war escalation",
        "eu fragmentation", "multiple conflicts",
        "capital controls",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _apply_constraints(priors):
    """Enforce min/max constraints and renormalize. Spec Teil 4 §2."""
    for k in priors:
        priors[k] = max(MIN_PROBABILITY, min(MAX_PROBABILITY, priors[k]))
    total = sum(priors.values())
    if total > 0:
        priors = {k: round(v / total, 3) for k, v in priors.items()}
    return priors


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
# 1. PHASE 1 — QUANTITATIVE PRE-ASSESSMENT (deterministic)
# ============================================================

def _phase1_quantitative_prior(power_scores, overlays, feedback_loops, gap_data):
    """
    Compute quantitative prior probabilities from hard data.
    Spec Teil 4 §2. Deterministic. Output: 4 priors summing to 1.0.
    """
    priors = {s: 0.25 for s in SCENARIOS}
    adjustments = []

    # --- Scenario A: Managed Decline ---

    # DDI > 55 and rising -> A boost
    ddi_comp = _safe_get(overlays, "ddi", "composite", default=0)
    ddi_trend = _safe_get(overlays, "ddi", "trend", default="STABLE")
    if ddi_comp > 55:
        boost = min(0.10, (ddi_comp - 55) / 100)
        if ddi_trend == "RISING":
            boost *= 1.5
        priors["managed_decline"] += boost
        adjustments.append(f"A: DDI {ddi_comp:.1f} ({ddi_trend}) -> +{boost:.3f}")

    # Gap < 20 and closing -> A boost
    gap = gap_data.get("gap", 50)
    gap_trend = gap_data.get("trend", "STABLE")
    if gap < 20:
        boost = min(0.08, (20 - gap) / 100)
        if gap_trend == "CLOSING":
            boost *= 1.5
        priors["managed_decline"] += boost
        adjustments.append(f"A: Gap {gap:.1f} ({gap_trend}) -> +{boost:.3f}")

    # FDP USA > 0.65 -> A boost
    fdp_usa = _safe_get(overlays, "fdp", "USA", "composite_proximity", default=0)
    if fdp_usa > 0.65:
        boost = min(0.05, (fdp_usa - 0.65) * 0.15)
        priors["managed_decline"] += boost
        adjustments.append(f"A: FDP USA {fdp_usa:.2f} -> +{boost:.3f}")

    # --- Scenario B: Conflict Escalation ---

    # GPR > 200 -> B boost
    gpr = _safe_get(overlays, "gpr_index_current", default=100)
    if gpr > 200:
        boost = min(0.12, (gpr - 200) / 1000)
        priors["conflict_escalation"] += boost
        adjustments.append(f"B: GPR {gpr:.0f} -> +{boost:.3f}")

    # Sanctions ESCALATING or CRITICAL -> B boost
    sanctions_trend = _safe_get(overlays, "sanctions", "escalation_trend", default="STABLE")
    if sanctions_trend == "ESCALATING":
        priors["conflict_escalation"] += 0.03
        adjustments.append("B: Sanctions ESCALATING -> +0.030")
    elif sanctions_trend == "CRITICAL":
        priors["conflict_escalation"] += 0.08
        adjustments.append("B: Sanctions CRITICAL -> +0.080")

    # Thucydides Trap ACTIVE -> B boost
    thucydides = next(
        (l for l in feedback_loops
         if l.get("name") == "Thucydides Trap" and l.get("status") == "ACTIVE"),
        None,
    )
    if thucydides:
        boost = min(0.05, thucydides.get("severity", 0) / 100)
        priors["conflict_escalation"] += boost
        adjustments.append(f"B: Thucydides ACTIVE sev={thucydides['severity']} -> +{boost:.3f}")

    # Chokepoint Alerts >= 2 -> B boost
    scsi_alerts = _safe_get(overlays, "scsi", "active_chokepoint_alerts", default=0)
    if scsi_alerts >= 2:
        priors["conflict_escalation"] += 0.04
        adjustments.append(f"B: {scsi_alerts} chokepoint alerts -> +0.040")

    # EWI severity HIGH -> B boost
    ewi_sev = _safe_get(overlays, "ewi", "severity", default="NONE")
    if ewi_sev == "HIGH":
        priors["conflict_escalation"] += 0.03
        adjustments.append("B: EWI HIGH -> +0.030")

    # --- Scenario C: US Renewal ---

    # USA Power Score momentum > 1.0 -> C boost
    usa_momentum = _safe_get(power_scores, "USA", "momentum", default=0)
    if usa_momentum > 1.0:
        boost = min(0.06, usa_momentum / 20)
        priors["us_renewal"] += boost
        adjustments.append(f"C: USA momentum {usa_momentum:.1f} -> +{boost:.3f}")

    # USA total score > 75 -> C boost
    usa_score = _safe_get(power_scores, "USA", "score", default=50)
    if usa_score > 75:
        priors["us_renewal"] += 0.02
        adjustments.append(f"C: USA score {usa_score:.1f} > 75 -> +0.020")

    # DXY rising -> C boost (indirect via trend)
    dxy_trend = _safe_get(overlays, "ddi", "components", "dxy_trend", "pct_1m", default=0)
    if isinstance(dxy_trend, (int, float)) and dxy_trend > 1.0:
        priors["us_renewal"] += 0.03
        adjustments.append(f"C: DXY rising ({dxy_trend:+.1f}%) -> +0.030")

    # FDP USA improving -> C boost
    fdp_usa_trend = _safe_get(overlays, "fdp", "USA", "trend", default="STABLE")
    if fdp_usa_trend == "FALLING":
        priors["us_renewal"] += 0.04
        adjustments.append("C: FDP USA FALLING -> +0.040")

    # --- Scenario D: Multipolar Chaos ---

    # DDI > 65 + multiple Feedback Loops ACTIVE -> D boost
    active_loop_count = sum(1 for l in feedback_loops if l.get("status") == "ACTIVE")
    if ddi_comp > 65 and active_loop_count >= 3:
        priors["multipolar_chaos"] += 0.06
        adjustments.append(f"D: DDI {ddi_comp:.1f} + {active_loop_count} ACTIVE loops -> +0.060")

    # Gap < 10 -> D boost
    if gap < 10:
        priors["multipolar_chaos"] += 0.04
        adjustments.append(f"D: Gap {gap:.1f} < 10 -> +0.040")

    # SCSI > 60 + Sanctions ESCALATING/CRITICAL -> D boost
    scsi_comp = _safe_get(overlays, "scsi", "composite", default=0)
    if scsi_comp > 60 and sanctions_trend in ("ESCALATING", "CRITICAL"):
        priors["multipolar_chaos"] += 0.05
        adjustments.append(f"D: SCSI {scsi_comp:.1f} + Sanctions {sanctions_trend} -> +0.050")

    # --- Normalize + Constrain ---
    priors = _apply_constraints(priors)

    return {
        "prior_probabilities": priors,
        "source": "QUANTITATIVE",
        "adjustments_applied": adjustments,
    }


# ============================================================
# 2. PHASE 2 — POLYMARKET CALIBRATION (deterministic)
# ============================================================

def _phase2_polymarket_calibration(prior_probs, validated_data):
    """
    Polymarket probabilities as Bayesian update on priors.
    Spec Teil 4 §3.
    """
    polymarket = validated_data.get("polymarket")

    # Get markets list — handle various data structures
    markets = []
    if isinstance(polymarket, dict):
        markets = polymarket.get("markets", [])
    elif isinstance(polymarket, list):
        markets = polymarket

    # Filter: volume > $100k, recent trades
    filtered = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        vol = m.get("volume", 0)
        if isinstance(vol, (int, float)) and vol > 100_000:
            filtered.append(m)

    if not filtered:
        return {
            "calibrated_probabilities": prior_probs,
            "polymarket_available": False,
            "reason": "No liquid markets available for calibration",
        }

    # Map markets to scenarios via keyword matching
    scenario_markets = {s: [] for s in SCENARIOS}
    for market in filtered:
        title_lower = (market.get("title") or market.get("question") or "").lower()
        for scenario, keywords in SCENARIO_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                prob = market.get("probability", market.get("outcomePrices", [None])[0])
                if isinstance(prob, str):
                    try:
                        prob = float(prob)
                    except (ValueError, TypeError):
                        prob = None
                if isinstance(prob, (int, float)) and 0 < prob < 1:
                    scenario_markets[scenario].append({
                        "market": market.get("title", "?"),
                        "prob": prob,
                        "volume": market.get("volume", 0),
                    })
                break

    # Derive scenario probabilities from Polymarket
    poly_implied = {}
    for scenario, mkts in scenario_markets.items():
        if mkts:
            total_vol = sum(m["volume"] for m in mkts)
            if total_vol > 0:
                poly_implied[scenario] = sum(m["prob"] * m["volume"] for m in mkts) / total_vol
            else:
                poly_implied[scenario] = None
        else:
            poly_implied[scenario] = None

    # Bayesian update weight based on total liquidity
    total_market_volume = sum(m.get("volume", 0) for m in filtered)
    poly_weight = min(0.30, max(0.15, total_market_volume / 50_000_000))

    calibrated = {}
    for scenario in SCENARIOS:
        if poly_implied.get(scenario) is not None:
            calibrated[scenario] = (
                prior_probs[scenario] * (1 - poly_weight)
                + poly_implied[scenario] * poly_weight
            )
        else:
            calibrated[scenario] = prior_probs[scenario]

    calibrated = _apply_constraints(calibrated)

    return {
        "calibrated_probabilities": calibrated,
        "polymarket_available": True,
        "poly_weight_applied": round(poly_weight, 3),
        "markets_used": len(filtered),
        "total_volume": total_market_volume,
        "poly_implied_raw": {k: round(v, 3) if v else None for k, v in poly_implied.items()},
        "shift_from_prior": {
            k: round(calibrated[k] - prior_probs[k], 3) for k in SCENARIOS
        },
    }


# ============================================================
# 3. PHASE 3 — LLM SYNTHESIS (Sonnet T=0.3)
# ============================================================

def _phase3_llm_synthesis(power_scores, overlays, feedback_loops, calibrated_priors,
                          gap_data, previous_thesis, scenario_history):
    """
    LLM Synthesis — sets final probabilities + thesis.
    Spec Teil 4 §4. Temperature 0.3 for consistency.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  [Phase 3] No ANTHROPIC_API_KEY — using calibrated priors as final")
        return _build_llm_fallback(calibrated_priors, previous_thesis)

    import requests

    # Build compact context for LLM
    ps_summary = {}
    for r in REGIONS:
        ps = power_scores.get(r, {})
        ps_summary[r] = {"score": ps.get("score", 50), "momentum": ps.get("momentum", 0)}

    overlay_summary = {
        "ddi": {"composite": _safe_get(overlays, "ddi", "composite", default=0),
                "trend": _safe_get(overlays, "ddi", "trend", default="STABLE")},
        "scsi": {"composite": _safe_get(overlays, "scsi", "composite", default=0),
                 "alerts": _safe_get(overlays, "scsi", "active_chokepoint_alerts", default=0)},
        "fdp_usa": _safe_get(overlays, "fdp", "USA", "composite_proximity", default=0),
        "sanctions_trend": _safe_get(overlays, "sanctions", "escalation_trend", default="STABLE"),
        "ewi": {"active": _safe_get(overlays, "ewi", "active_signals", default=0),
                "severity": _safe_get(overlays, "ewi", "severity", default="NONE")},
        "gpr": _safe_get(overlays, "gpr_index_current", default=100),
    }

    loop_summary = [
        {"name": l.get("name"), "region": l.get("region"),
         "severity": l.get("severity"), "status": l.get("status")}
        for l in feedback_loops[:5]
    ]

    # Previous thesis summary (compact)
    prev_summary = {}
    if previous_thesis and isinstance(previous_thesis, dict):
        prev_summary = {
            "dominant": previous_thesis.get("dominant_thesis", "Unknown"),
            "probs": previous_thesis.get("scenario_probabilities", {}),
            "date": previous_thesis.get("date", ""),
        }

    prompt = f"""You are the G7 World Order Monitor's Scenario Synthesis Engine.

TASK: Set final scenario probabilities and generate strategic thesis.

CONSTRAINTS:
- Probabilities MUST sum to 1.00
- No probability below 0.05 or above 0.70
- Max shift from calibrated priors: 0.15 per scenario
- Every shift from priors MUST have explicit justification
- If you agree with priors, you may keep them unchanged

CALIBRATED PRIORS (from Phase 1 quantitative + Phase 2 Polymarket):
{json.dumps(calibrated_priors, indent=2)}

QUANTITATIVE CONTEXT:
Power Scores: {json.dumps(ps_summary, indent=2)}
USA-China Gap: {json.dumps(gap_data, indent=2)}

OVERLAYS:
{json.dumps(overlay_summary, indent=2)}

FEEDBACK LOOPS (top 5 by severity):
{json.dumps(loop_summary, indent=2)}

PREVIOUS THESIS:
{json.dumps(prev_summary, indent=2)}

RESPOND IN THIS EXACT JSON SCHEMA (no markdown, no preamble):
{{
    "final_probabilities": {{
        "managed_decline": float,
        "conflict_escalation": float,
        "us_renewal": float,
        "multipolar_chaos": float
    }},
    "shift_reasons": ["string — each reason for shifting from priors"],
    "dominant_thesis": "Managed Decline|Conflict Escalation|US Renewal|Multipolar Chaos",
    "confidence": "HIGH|MEDIUM|LOW",
    "confidence_reasoning": "string",
    "preferred_targets": {{
        "if_em_broad": ["ticker1", "ticker2"],
        "if_china_stimulus": ["ticker1"],
        "if_commodity_super": "ticker"
    }},
    "active_vetos": ["TICKER — reason"],
    "veto_watch": ["TICKER — reason"],
    "tilt_narrative": {{
        "gold_bias": "string",
        "usd_duration": "string",
        "em_equity": "string",
        "commodities": "string"
    }},
    "key_uncertainties": ["string"]
}}

IMPORTANT:
- Be EVIDENCE-BASED. Every claim must reference a data point from above.
- Be PORTFOLIO-FIRST. Think about what this means for asset allocation.
- If unsure, lean toward priors. Don't shift without strong evidence."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 3000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )

        if resp.status_code != 200:
            print(f"  [Phase 3] LLM error: HTTP {resp.status_code}")
            return _build_llm_fallback(calibrated_priors, previous_thesis)

        data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(text)

        # Validate and constrain final probabilities
        fp = result.get("final_probabilities", {})
        if not fp or not all(s in fp for s in SCENARIOS):
            print("  [Phase 3] LLM missing probabilities — using priors")
            result["final_probabilities"] = calibrated_priors
        else:
            # Enforce max shift constraint
            for s in SCENARIOS:
                shift = fp[s] - calibrated_priors.get(s, 0.25)
                if abs(shift) > MAX_LLM_SHIFT_PER_SCENARIO:
                    fp[s] = calibrated_priors[s] + (MAX_LLM_SHIFT_PER_SCENARIO * (1 if shift > 0 else -1))
            result["final_probabilities"] = _apply_constraints(fp)

        result["probability_source"] = "LLM"
        print(f"  [Phase 3] LLM synthesis OK — dominant: {result.get('dominant_thesis', '?')}")
        return result

    except json.JSONDecodeError as e:
        print(f"  [Phase 3] LLM JSON parse error: {e}")
        return _build_llm_fallback(calibrated_priors, previous_thesis)
    except Exception as e:
        print(f"  [Phase 3] LLM error: {e}")
        return _build_llm_fallback(calibrated_priors, previous_thesis)


def _build_llm_fallback(calibrated_priors, previous_thesis):
    """Fallback when LLM is unavailable — use calibrated priors as final."""
    dominant = max(calibrated_priors, key=calibrated_priors.get)
    dominant_names = {
        "managed_decline": "Managed Decline",
        "conflict_escalation": "Conflict Escalation",
        "us_renewal": "US Renewal",
        "multipolar_chaos": "Multipolar Chaos",
    }
    return {
        "final_probabilities": calibrated_priors,
        "shift_reasons": ["LLM unavailable — using calibrated quantitative priors as final"],
        "dominant_thesis": dominant_names.get(dominant, "Managed Decline"),
        "confidence": "LOW",
        "confidence_reasoning": "No LLM synthesis available — quantitative priors only.",
        "preferred_targets": {"if_em_broad": ["VWO", "INDA"], "if_china_stimulus": ["FXI"], "if_commodity_super": "DBC"},
        "active_vetos": [],
        "veto_watch": [],
        "tilt_narrative": {"gold_bias": "N/A", "usd_duration": "N/A", "em_equity": "N/A", "commodities": "N/A"},
        "key_uncertainties": ["LLM synthesis unavailable — thesis is purely quantitative"],
        "probability_source": "QUANTITATIVE_FALLBACK",
    }


# ============================================================
# 4. PHASE 4 — OPERATOR OVERRIDE CHECK (deterministic)
# ============================================================

def _phase4_operator_override(llm_result, validated_data):
    """
    Check G7_OPERATOR_OVERRIDES for manual overrides.
    Spec Teil 4 §5. Override takes precedence over LLM.
    """
    # Overrides come from the Sheet tab G7_OPERATOR_OVERRIDES
    # For now, check if any override data is in validated_data
    overrides = validated_data.get("operator_overrides")

    if not overrides or not isinstance(overrides, dict) or not overrides.get("active"):
        return {
            **llm_result,
            "override_active": False,
        }

    # Check expiry
    expires = overrides.get("expires", "")
    if expires:
        try:
            exp_date = datetime.fromisoformat(expires).replace(tzinfo=timezone.utc)
            if exp_date < datetime.now(timezone.utc):
                print("  [Phase 4] Override expired — using LLM probabilities")
                return {
                    **llm_result,
                    "override_active": False,
                    "override_expired": True,
                }
        except (ValueError, TypeError):
            pass

    # Override is active
    override_probs = overrides.get("override_probabilities", {})
    if override_probs and all(s in override_probs for s in SCENARIOS):
        override_probs = _apply_constraints(override_probs)
        print(f"  [Phase 4] OPERATOR OVERRIDE active — reason: {overrides.get('reason', '?')}")
        return {
            **llm_result,
            "final_probabilities": override_probs,
            "probability_source": "OPERATOR_OVERRIDE",
            "override_active": True,
            "override_reason": overrides.get("reason", ""),
            "override_date": overrides.get("set_date", ""),
            "llm_suggested_probabilities": llm_result.get("final_probabilities", {}),
        }

    return {
        **llm_result,
        "override_active": False,
    }


# ============================================================
# 5. TILT COMPUTATION (deterministic)
# ============================================================

def _compute_tilts(scenario_probs):
    """
    Compute tilt for each asset: sum(prob * exposure).
    Spec Teil 4 §6.

    Positive tilt = overweight recommended.
    Negative tilt = underweight recommended.
    """
    computed_tilts = {}
    for asset, exposures in ASSET_EXPOSURE_VECTORS.items():
        tilt = sum(
            scenario_probs.get(scenario, 0) * exposure
            for scenario, exposure in exposures.items()
        )
        computed_tilts[asset] = round(tilt, 3)

    return computed_tilts


# ============================================================
# 6. PERMOPT ALLOCATION (deterministic)
# ============================================================

def _compute_permopt(scenario_probs, computed_tilts, ddi_composite, active_vetos):
    """
    PermOpt allocation based on tilts, DDI, and vetos.
    Spec Teil 4 §7.
    """
    # Total PermOpt size based on DDI
    if ddi_composite > 70:
        total_pct = 0.05
        sizing_rationale = "DDI >70 — De-Dollarization accelerating, PermOpt expansion justified"
    elif ddi_composite > 55:
        total_pct = 0.03
        sizing_rationale = "DDI 55-70 — De-Dollarization ongoing, standard PermOpt"
    else:
        total_pct = 0.02
        sizing_rationale = "DDI <55 — De-Dollarization thesis weakening, reduce PermOpt"

    # Build veto set (extract ticker from "TICKER — reason" format)
    veto_tickers = set()
    for v in (active_vetos or []):
        if isinstance(v, str) and " " in v:
            veto_tickers.add(v.split(" ")[0].upper())
        elif isinstance(v, str):
            veto_tickers.add(v.upper())

    # Filter candidates: positive tilt + eligible + not vetoed
    candidates = {}
    for asset, tilt in computed_tilts.items():
        if (tilt > 0.1
                and asset in PERMOPT_ELIGIBLE
                and asset not in veto_tickers):
            candidates[asset] = tilt

    if not candidates:
        return {
            "total_pct": total_pct,
            "composition": {},
            "sizing_rationale": sizing_rationale,
            "rebalance_frequency": "quarterly",
            "vetos_applied": active_vetos or [],
            "candidates_considered": 0,
        }

    # Proportional allocation based on tilt strength
    total_tilt = sum(candidates.values())
    composition = {}
    for asset, tilt in candidates.items():
        weight = round(total_pct * (tilt / total_tilt), 4)
        if weight >= 0.002:  # Minimum 0.2% to be meaningful
            composition[asset] = weight

    # Renormalize to total_pct
    actual_total = sum(composition.values())
    if actual_total > 0 and abs(actual_total - total_pct) > 0.001:
        factor = total_pct / actual_total
        composition = {k: round(v * factor, 4) for k, v in composition.items()}

    return {
        "total_pct": total_pct,
        "composition": composition,
        "sizing_rationale": sizing_rationale,
        "rebalance_frequency": "quarterly",
        "vetos_applied": active_vetos or [],
        "candidates_considered": len(candidates),
    }


# ============================================================
# 7. THESIS STRESS TEST (LLM, quarterly only)
# ============================================================

def _thesis_stress_test(dominant_thesis, scenario_probs, overlays, gap_data):
    """
    LLM attack on dominant thesis. Spec Teil 4 §10.
    Only runs on QUARTERLY/AD_HOC. Temperature 0.3.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"skipped": True, "reason": "No API key"}

    import requests

    ddi = _safe_get(overlays, "ddi", "composite", default=0)
    fdp_usa = _safe_get(overlays, "fdp", "USA", "composite_proximity", default=0)
    scsi = _safe_get(overlays, "scsi", "composite", default=0)

    prompt = f"""You are the G7 World Order Monitor's internal Devil's Advocate.

TASK: Stress-test the dominant thesis. Your job is to ATTACK it.

DOMINANT THESIS: {dominant_thesis}
CURRENT PROBABILITIES: {json.dumps(scenario_probs)}

KEY OVERLAYS:
DDI: {ddi}, FDP USA: {fdp_usa}, SCSI: {scsi}
USA-China Gap: {gap_data.get('gap', '?')} ({gap_data.get('trend', '?')})

RESPOND IN THIS EXACT JSON SCHEMA (no markdown, no preamble):
{{
    "what_would_kill_this_thesis": ["4-5 specific scenarios"],
    "early_indicators_of_thesis_death": ["4-5 measurable indicators with thresholds"],
    "current_status_of_killers": "string",
    "most_dangerous_alternative": "string",
    "blind_spots": ["2-3 things we are NOT measuring"],
    "recommended_hedges": ["2-3 positions that protect against thesis failure"]
}}

BE AGGRESSIVE. Challenge, don't confirm."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 2000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )

        if resp.status_code != 200:
            return {"skipped": True, "reason": f"HTTP {resp.status_code}"}

        data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        result["skipped"] = False
        print(f"  [Stress Test] Complete — most dangerous alternative: {result.get('most_dangerous_alternative', '?')}")
        return result

    except Exception as e:
        print(f"  [Stress Test] Error: {e}")
        return {"skipped": True, "reason": str(e)}


# ============================================================
# 8. DRIFT TRACKING (deterministic)
# ============================================================

def _detect_drift_anomalies(new_probs, previous_thesis, scenario_history):
    """
    Check for suspicious patterns in thesis evolution.
    Spec Teil 4 §11.
    """
    warnings = []

    prev_probs = {}
    if previous_thesis and isinstance(previous_thesis, dict):
        prev_probs = previous_thesis.get("scenario_probabilities", {})

    # 1. Large shift without strong justification
    for scenario in SCENARIOS:
        current_p = new_probs.get(scenario, 0.25)
        previous_p = prev_probs.get(scenario, 0.25)
        shift = abs(current_p - previous_p)
        if shift > 0.10:
            warnings.append(
                f"DRIFT WARNING: {scenario} shifted {shift:.0%} — "
                f"verify shift_reasons are substantial"
            )

    # 2. Complacency check — same thesis dominant for 4+ entries
    if scenario_history and len(scenario_history) >= 4:
        dominant = max(new_probs, key=new_probs.get)
        consecutive = 0
        for entry in reversed(scenario_history[-4:]):
            entry_probs = entry.get("probabilities", {})
            if entry_probs and max(entry_probs, key=entry_probs.get) == dominant:
                consecutive += 1
            else:
                break
        if consecutive >= 4:
            warnings.append(
                f"COMPLACENCY WARNING: {dominant} has been dominant for "
                f"{consecutive}+ periods. Thesis Stress Test required."
            )

    # 3. Stagnation (identical probabilities)
    if prev_probs:
        identical = all(
            abs(new_probs.get(s, 0) - prev_probs.get(s, 0)) < 0.01
            for s in SCENARIOS
        )
        if identical:
            warnings.append(
                "STAGNATION: Probabilities identical to last period. "
                "Verify the engine is actually processing new data."
            )

    # 4. Flip-flop detection
    if scenario_history and len(scenario_history) >= 2:
        two_ago = scenario_history[-1].get("probabilities", {}) if len(scenario_history) >= 2 else {}
        three_ago = scenario_history[-2].get("probabilities", {}) if len(scenario_history) >= 2 else {}
        for scenario in SCENARIOS:
            p_now = new_probs.get(scenario, 0)
            p_prev = prev_probs.get(scenario, 0)
            p_prev2 = three_ago.get(scenario, 0)
            if (p_now > p_prev and p_prev < p_prev2
                    and abs(p_now - p_prev2) < 0.03):
                warnings.append(
                    f"FLIP_FLOP: {scenario} returned to ~{p_prev2:.0%} "
                    f"after dipping to {p_prev:.0%}."
                )

    return warnings


# ============================================================
# 9. INTERIM TRIGGER EVALUATION (deterministic)
# ============================================================

def _evaluate_interim_triggers(overlays, gap_data, previous_thesis):
    """
    Check if a weekly run needs an interim thesis update.
    Spec Teil 4 §12. Returns trigger list or empty.
    """
    triggers_fired = []

    prev_probs = {}
    if previous_thesis and isinstance(previous_thesis, dict):
        prev_probs = previous_thesis.get("scenario_probabilities", {})

    # Trigger 1: GPR Spike > 2 sigma
    gpr_zscore = _safe_get(overlays, "gpr_index_zscore", default=0)
    if abs(gpr_zscore) > 2.0:
        triggers_fired.append({
            "trigger": "GPR_SPIKE",
            "detail": f"GPR Z-Score {gpr_zscore:.1f} (>2.0 threshold)",
            "severity": "HIGH",
        })

    # Trigger 2: Chokepoint > 70
    chokepoints = _safe_get(overlays, "scsi", "chokepoints", default={})
    for cp, score in chokepoints.items():
        if isinstance(score, (int, float)) and score > 70:
            triggers_fired.append({
                "trigger": "CHOKEPOINT_CRITICAL",
                "detail": f"{cp} chokepoint score {score} (>70 threshold)",
                "severity": "HIGH",
            })

    # Trigger 3: FDP jump > 0.1
    for region in ["USA", "CHINA", "EU"]:
        current_fdp = _safe_get(overlays, "fdp", region, "composite_proximity", default=0)
        # We don't have previous FDP easily here — skip if no previous thesis
        # This will become more useful as history accumulates

    # Trigger 4: Sanctions escalation to CRITICAL
    sanctions_trend = _safe_get(overlays, "sanctions", "escalation_trend", default="STABLE")
    if sanctions_trend == "CRITICAL":
        triggers_fired.append({
            "trigger": "SANCTIONS_CRITICAL",
            "detail": "Global sanctions escalation reached CRITICAL",
            "severity": "HIGH",
        })

    # Trigger 5: EWI severity HIGH
    ewi_sev = _safe_get(overlays, "ewi", "severity", default="NONE")
    if ewi_sev == "HIGH":
        triggers_fired.append({
            "trigger": "EWI_HIGH",
            "detail": f"Early Warning Index severity HIGH ({_safe_get(overlays, 'ewi', 'active_signals', default=0)} active)",
            "severity": "MEDIUM",
        })

    if triggers_fired:
        return {
            "interim_required": True,
            "triggers": triggers_fired,
            "most_severe": max(triggers_fired, key=lambda t: 1 if t["severity"] == "HIGH" else 0),
        }
    else:
        return {"interim_required": False}


# ============================================================
# MAIN ENTRY POINT — phase6_scenario_engine()
# ============================================================

def phase6_scenario_engine(scores, overlays, gap_data, validated_data,
                           previous_thesis, scenario_history, run_type):
    """
    Phase 6: Scenario Engine — Full Implementation.

    Run Types:
      QUARTERLY / AD_HOC: Full 9-step run (Phase 1-4 + Tilts + PermOpt + Stress Test + Drift)
      WEEKLY: Interim Trigger Check only; if trigger fires -> Phase 1+3+4+Tilts

    Returns dict with thesis, probabilities, tilts, permopt, stress test, drift warnings.
    """
    print(f"[Phase 6] Scenario Engine [run_type={run_type}]...")
    ps_start = time.time()

    power_scores = scores.get("power_scores", {})
    feedback_loops = overlays.get("feedback_loops", [])
    is_full_run = run_type in ("QUARTERLY", "AD_HOC")

    # --- WEEKLY: Check interim triggers first ---
    if not is_full_run:
        interim = _evaluate_interim_triggers(overlays, gap_data, previous_thesis)
        if not interim.get("interim_required"):
            print(f"  No interim triggers fired — keeping previous thesis")
            return {
                "thesis_updated": False,
                "reason": "No interim triggers — thesis unchanged",
                "current_thesis": previous_thesis,
                "thesis": previous_thesis or _default_thesis(),
                "interim_triggers": interim,
            }
        else:
            trigger_names = [t["trigger"] for t in interim.get("triggers", [])]
            print(f"  INTERIM TRIGGERS FIRED: {trigger_names}")
            # Fall through to Phase 1+3+4+Tilts (no PermOpt, no Stress Test)

    # ---- PHASE 1: Quantitative Pre-Assessment ----
    phase1 = _phase1_quantitative_prior(power_scores, overlays, feedback_loops, gap_data)
    prior_probs = phase1["prior_probabilities"]
    print(f"  Phase 1 Priors: A={prior_probs['managed_decline']:.1%} B={prior_probs['conflict_escalation']:.1%} "
          f"C={prior_probs['us_renewal']:.1%} D={prior_probs['multipolar_chaos']:.1%}")
    print(f"  Adjustments: {len(phase1['adjustments_applied'])}")

    # ---- PHASE 2: Polymarket Calibration (full runs only) ----
    if is_full_run:
        phase2 = _phase2_polymarket_calibration(prior_probs, validated_data)
        calibrated = phase2["calibrated_probabilities"]
        if phase2.get("polymarket_available"):
            print(f"  Phase 2 Polymarket: {phase2['markets_used']} markets, weight={phase2['poly_weight_applied']}")
        else:
            print(f"  Phase 2 Polymarket: skipped ({phase2.get('reason', '?')})")
    else:
        phase2 = {"calibrated_probabilities": prior_probs, "polymarket_available": False, "reason": "Interim run"}
        calibrated = prior_probs

    # ---- PHASE 3: LLM Synthesis ----
    phase3 = _phase3_llm_synthesis(
        power_scores, overlays, feedback_loops, calibrated,
        gap_data, previous_thesis, scenario_history,
    )
    final_probs = phase3.get("final_probabilities", calibrated)
    dominant = phase3.get("dominant_thesis", "Managed Decline")
    confidence = phase3.get("confidence", "LOW")
    prob_source = phase3.get("probability_source", "UNKNOWN")
    print(f"  Phase 3 Final: A={final_probs.get('managed_decline', 0):.1%} "
          f"B={final_probs.get('conflict_escalation', 0):.1%} "
          f"C={final_probs.get('us_renewal', 0):.1%} "
          f"D={final_probs.get('multipolar_chaos', 0):.1%}")
    print(f"  Dominant: {dominant} | Confidence: {confidence} | Source: {prob_source}")

    # ---- PHASE 4: Operator Override ----
    phase4 = _phase4_operator_override(phase3, validated_data)
    final_probs = phase4.get("final_probabilities", final_probs)
    if phase4.get("override_active"):
        print(f"  Phase 4 OVERRIDE ACTIVE: {phase4.get('override_reason', '?')}")
        prob_source = "OPERATOR_OVERRIDE"

    # ---- TILT COMPUTATION ----
    computed_tilts = _compute_tilts(final_probs)
    top_tilts = sorted(computed_tilts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"  Top Tilts: {', '.join(f'{t}={v:+.3f}' for t, v in top_tilts)}")

    # ---- PERMOPT (full runs only) ----
    if is_full_run:
        ddi_comp = _safe_get(overlays, "ddi", "composite", default=0)
        active_vetos = phase4.get("active_vetos", [])
        permopt = _compute_permopt(final_probs, computed_tilts, ddi_comp, active_vetos)
        print(f"  PermOpt: {permopt['total_pct']:.1%} across {len(permopt['composition'])} assets")
    else:
        permopt = {"total_pct": 0, "composition": {}, "sizing_rationale": "Interim run — PermOpt not updated"}

    # ---- STRESS TEST (full runs only) ----
    if is_full_run:
        stress_test = _thesis_stress_test(dominant, final_probs, overlays, gap_data)
    else:
        stress_test = {"skipped": True, "reason": "Interim run"}

    # ---- DRIFT TRACKING ----
    drift_warnings = _detect_drift_anomalies(final_probs, previous_thesis, scenario_history or [])
    if drift_warnings:
        print(f"  Drift Warnings: {len(drift_warnings)}")
        for w in drift_warnings:
            print(f"    {w}")

    # ---- BUILD THESIS OUTPUT ----
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Compute shift vs previous
    prev_probs = {}
    if previous_thesis and isinstance(previous_thesis, dict):
        prev_probs = previous_thesis.get("scenario_probabilities", {})
    shift_vs_previous = {
        s: round(final_probs.get(s, 0.25) - prev_probs.get(s, 0.25), 3) for s in SCENARIOS
    }

    thesis = {
        "date": now_str,
        "dominant_thesis": dominant,
        "confidence": confidence,
        "confidence_reasoning": phase4.get("confidence_reasoning", ""),
        "scenario_probabilities": final_probs,
        "probability_source": phase4.get("probability_source", prob_source),
        "scenario_shift_vs_previous": shift_vs_previous,
        "shift_reasons": phase4.get("shift_reasons", []),
        "preferred_targets": phase4.get("preferred_targets", {}),
        "perm_opt_allocation": permopt,
        "active_vetos": phase4.get("active_vetos", []),
        "veto_watch": phase4.get("veto_watch", []),
        "interim_flag": not is_full_run,
        "computed_tilts": computed_tilts,
        "tilt_narrative": phase4.get("tilt_narrative", {}),
        "key_uncertainties": phase4.get("key_uncertainties", []),
        "thesis_stress_test": stress_test,
        "drift_warnings": drift_warnings,
        "phase1_adjustments": phase1.get("adjustments_applied", []),
        "phase2_polymarket": {
            "available": phase2.get("polymarket_available", False),
            "markets_used": phase2.get("markets_used", 0),
        },
    }

    duration = round(time.time() - ps_start, 1)
    print(f"  Phase 6 complete ({duration}s) — thesis_updated=True")

    return {
        "thesis_updated": True,
        "reason": "Full quarterly run" if is_full_run else "Interim trigger fired",
        "current_thesis": thesis,
        "thesis": thesis,
    }


def _default_thesis():
    """Default thesis when no previous exists."""
    return {
        "date": "",
        "dominant_thesis": "Managed Decline",
        "confidence": "LOW",
        "scenario_probabilities": {
            "managed_decline": 0.40,
            "conflict_escalation": 0.20,
            "us_renewal": 0.25,
            "multipolar_chaos": 0.15,
        },
        "probability_source": "DEFAULT",
        "preferred_targets": {},
        "perm_opt_allocation": {},
        "active_vetos": [],
        "veto_watch": [],
        "interim_flag": False,
        "computed_tilts": {},
        "shift_reasons": ["Initial default — no engine run yet"],
    }
