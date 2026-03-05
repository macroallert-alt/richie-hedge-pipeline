"""
IC Pipeline — Stufe 2: Intelligence Engine Orchestrator
Deterministic computation: consensus, divergence, novelty, catalysts, freshness.
No LLM calls.
"""

import json
import logging
import os
from datetime import date, datetime

from step_0i_ic_pipeline.src.intelligence.consensus import calculate_all_consensus
from step_0i_ic_pipeline.src.intelligence.divergence import calculate_divergences
from step_0i_ic_pipeline.src.intelligence.novelty import (
    load_beliefs,
    save_beliefs,
    update_beliefs_and_validate_novelty,
    compute_freshness,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalyst Timeline Aggregation (Spec Teil 4 §4.8)
# ---------------------------------------------------------------------------
def _aggregate_catalysts(claims: list[dict]) -> list[dict]:
    """Aggregate all claims with catalyst_date into a timeline."""
    catalysts = []
    seen = set()

    for claim in claims:
        cat = claim.get("catalyst")
        cat_date = claim.get("catalyst_date")
        if not cat or not cat_date:
            continue

        key = f"{cat_date}_{cat[:50]}"
        if key in seen:
            # Merge: add source
            for existing in catalysts:
                if existing.get("_key") == key:
                    if claim["source_id"] not in existing["sources_mentioning"]:
                        existing["sources_mentioning"].append(claim["source_id"])
                    break
            continue

        seen.add(key)
        catalysts.append({
            "_key": key,
            "date": cat_date,
            "event": cat,
            "topics": claim.get("topics", []),
            "sources_mentioning": [claim["source_id"]],
            "expected_impact": claim.get("claim_text", "")[:200],
        })

    # Remove internal key and sort
    for c in catalysts:
        c.pop("_key", None)

    return sorted(catalysts, key=lambda x: x.get("date", "9999"))


# ---------------------------------------------------------------------------
# DC Data Loader (AD.2)
# ---------------------------------------------------------------------------
def _load_dc_data() -> dict | None:
    """
    Load DC IC-interface JSON if available.
    Looks for ic_data_YYYY-MM-DD.json in expected locations.
    Returns None if not available (divergence skipped).
    """
    today_str = date.today().isoformat()
    possible_paths = [
        # In the same repo, written by step_0a
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            "step_0a_data_collector", "data", "ic_interface",
            f"ic_data_{today_str}.json",
        ),
        # Flat path in data/
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data",
            f"ic_data_{today_str}.json",
        ),
    ]

    for path in possible_paths:
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            try:
                with open(norm, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded DC data from {norm}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load DC data from {norm}: {e}")

    logger.info("No DC data available — divergences will be empty")
    return None


# ---------------------------------------------------------------------------
# V16 + Risk Context Loader
# ---------------------------------------------------------------------------
def _load_system_context() -> dict:
    """
    Load V16 regime and risk context from Drive CURRENT/.
    Returns partial dict — gracefully degrades if unavailable.
    """
    context = {
        "v16_regime": "UNKNOWN",
        "v16_confidence": None,
        "f6_signals_today": 0,
        "f6_top_signals": [],
        "data_script_warnings": [],
    }

    possible_paths = [
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            "step_0a_data_collector", "data", "system_state.json",
        ),
    ]

    for path in possible_paths:
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            try:
                with open(norm, "r") as f:
                    state = json.load(f)
                context["v16_regime"] = state.get("v16_regime", "UNKNOWN")
                context["v16_confidence"] = state.get("v16_confidence")
                break
            except Exception:
                pass

    return context


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------
def run_intelligence_engine(
    claims: list[dict],
    sources_config: list[dict],
    expertise_matrix: dict,
    taxonomy: dict,
) -> dict:
    """
    Run Stufe 2: Intelligence Engine.
    All deterministic — no LLM.

    Steps (per Spec Teil 4 §4.5):
    1. System relevance already enriched in Stufe 1
    2. Update belief states + validate novelty
    3. Calculate weighted consensus per topic
    4. Load DC data + calculate divergences
    5. Aggregate catalyst timeline
    6. Compute freshness
    7. Assemble intelligence JSON

    Returns: Complete intel dict for today
    """
    today = date.today().isoformat()
    run_id = f"intel_{today.replace('-', '')}_{datetime.utcnow().strftime('%H%M%S')}"

    topics = taxonomy.get("topics", [])
    divergence_config = taxonomy.get("divergence_config", {})

    # --- Step 2: Belief state + novelty ---
    beliefs_data = load_beliefs()
    claims, beliefs_data = update_beliefs_and_validate_novelty(claims, beliefs_data)
    save_beliefs(beliefs_data)

    # --- Step 3: Consensus ---
    consensus = calculate_all_consensus(
        claims, sources_config, expertise_matrix, topics
    )

    # --- Step 4: Divergence ---
    dc_data = _load_dc_data()
    divergences = calculate_divergences(consensus, dc_data, divergence_config)

    # --- Step 5: Catalysts ---
    catalyst_timeline = _aggregate_catalysts(claims)

    # --- Step 6: Freshness ---
    freshness = compute_freshness(sources_config, claims, beliefs_data)

    # --- Step 7: System context ---
    system_context = _load_system_context()

    # --- Extraction summary ---
    opinion_pred = [c for c in claims if c.get("claim_type") in ("OPINION", "PREDICTION")]
    fact_analysis = [c for c in claims if c.get("claim_type") in ("FACT", "ANALYSIS")]
    high_novelty = [c for c in claims if c.get("novelty_score", 0) >= 5]

    # Build high_novelty_claims for briefing
    high_novelty_claims = [
        {
            "id": c.get("id", ""),
            "source_id": c["source_id"],
            "claim_text": c.get("claim_text", ""),
            "novelty_score": c.get("novelty_score", 0),
            "topics": c.get("topics", []),
            "content_date": c.get("content_date", ""),
        }
        for c in high_novelty
    ]

    # --- Assemble output ---
    intel = {
        "date": today,
        "run_id": run_id,
        "extraction_summary": {
            "sources_processed": len(set(c["source_id"] for c in claims)),
            "total_claims": len(claims),
            "claims_opinion_prediction": len(opinion_pred),
            "claims_fact_analysis": len(fact_analysis),
            "high_novelty_claims": len(high_novelty),
        },
        "consensus": consensus,
        "divergences": divergences,
        "high_novelty_claims": high_novelty_claims,
        "catalyst_timeline": catalyst_timeline,
        "freshness": freshness,
        "system_context": system_context,
    }

    logger.info(
        f"Intelligence engine complete: {len(claims)} claims, "
        f"{len(divergences)} divergences, {len(high_novelty)} high-novelty"
    )

    return intel
