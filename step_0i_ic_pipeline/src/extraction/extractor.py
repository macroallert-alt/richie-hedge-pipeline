"""
IC Pipeline — Stufe 1: Claim Extractor
Claude Haiku API for structured claim extraction from content.
"""

import json
import logging
import os
import re
from datetime import date
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Loading
# ---------------------------------------------------------------------------
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Taxonomy Loading (for system_relevance enrichment)
# ---------------------------------------------------------------------------
def _load_taxonomy() -> dict:
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "taxonomy.json"
    )
    with open(os.path.normpath(path), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# System Relevance Enrichment (deterministic, no LLM)
# ---------------------------------------------------------------------------
def _enrich_system_relevance(claim: dict, taxonomy: dict) -> dict:
    """Add system_relevance fields based on deterministic topic mapping."""
    topics = claim.get("topics", [])
    relevance_rules = taxonomy.get("v16_relevance_rules", {})
    layer_map = taxonomy.get("topic_to_layers", {})
    asset_map = taxonomy.get("topic_to_assets", {})

    affected_layers = set()
    affected_assets = set()
    v16_related = False
    f6_related = False
    g7_related = False

    for topic in topics:
        affected_layers.update(layer_map.get(topic, []))
        affected_assets.update(asset_map.get(topic, []))
        if topic in relevance_rules.get("v16_related", []):
            v16_related = True
        if topic in relevance_rules.get("f6_related", []):
            f6_related = True
        if topic in relevance_rules.get("g7_related", []):
            g7_related = True

    claim["system_relevance"] = {
        "v16_related": v16_related,
        "f6_related": f6_related,
        "g7_related": g7_related,
        "affected_layers": sorted(affected_layers),
        "affected_assets": sorted(affected_assets),
    }
    return claim


# ---------------------------------------------------------------------------
# Claim Validation
# ---------------------------------------------------------------------------
VALID_CLAIM_TYPES = {"FACT", "OPINION", "PREDICTION", "ANALYSIS"}
VALID_DIRECTIONS = {"BULLISH", "BEARISH", "NEUTRAL", "MIXED"}
VALID_TIMEFRAMES = {"IMMEDIATE", "SHORT", "MEDIUM", "LONG"}
VALID_TOPICS = {
    "LIQUIDITY", "FED_POLICY", "CREDIT", "RECESSION", "INFLATION",
    "EQUITY_VALUATION", "CHINA_EM", "GEOPOLITICS", "ENERGY",
    "COMMODITIES", "TECH_AI", "CRYPTO", "DOLLAR", "VOLATILITY", "POSITIONING",
}


def _validate_claim(claim: dict, source_id: str, seq: int, extraction_date: str) -> Optional[dict]:
    """Validate and normalize a single claim. Returns None if invalid."""
    # Required fields
    if not claim.get("claim_text"):
        return None

    # Normalize claim_type
    ct = claim.get("claim_type", "OPINION").upper()
    if ct not in VALID_CLAIM_TYPES:
        ct = "OPINION"
    claim["claim_type"] = ct

    # Normalize sentiment
    sent = claim.get("sentiment", {})
    direction = sent.get("direction", "NEUTRAL").upper()
    if direction not in VALID_DIRECTIONS:
        direction = "NEUTRAL"
    intensity = sent.get("intensity", 5)
    if not isinstance(intensity, (int, float)):
        intensity = 5
    intensity = max(1, min(10, int(intensity)))
    claim["sentiment"] = {"direction": direction, "intensity": intensity}

    # Normalize topics
    topics = claim.get("topics", [])
    topics = [t.upper() for t in topics if t.upper() in VALID_TOPICS]
    if not topics:
        topics = ["LIQUIDITY"]  # fallback
    claim["topics"] = topics[:3]

    primary = claim.get("primary_topic", "").upper()
    if primary not in VALID_TOPICS or primary not in topics:
        primary = topics[0]
    claim["primary_topic"] = primary

    # Normalize timeframe
    tf = claim.get("timeframe", "MEDIUM").upper()
    if tf not in VALID_TIMEFRAMES:
        tf = "MEDIUM"
    claim["timeframe"] = tf

    # Normalize novelty
    nov = claim.get("novelty_score", 3)
    if not isinstance(nov, (int, float)):
        nov = 3
    claim["novelty_score"] = max(0, min(10, int(nov)))

    # Normalize confidence
    conf = claim.get("extraction_confidence", 0.8)
    if not isinstance(conf, (int, float)):
        conf = 0.8
    conf = max(0.0, min(1.0, float(conf)))

    # Build full claim with ID
    claim["id"] = f"claim_{extraction_date.replace('-', '')}_{source_id}_{seq:03d}"
    claim["extraction_date"] = extraction_date
    claim["confidence"] = {"extraction_confidence": conf}

    # Optional fields
    if not claim.get("catalyst"):
        claim["catalyst"] = None
    if not claim.get("catalyst_date"):
        claim["catalyst_date"] = None
    if not claim.get("novelty_note"):
        claim["novelty_note"] = ""

    return claim


# ---------------------------------------------------------------------------
# Main Extraction
# ---------------------------------------------------------------------------
def extract_claims(
    content: dict,
    source_config: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """
    Extract structured claims from content using Claude Haiku API.

    Args:
        content: dict with source_id, source_name, content_date, title, text, content_type
        source_config: full source config from sources.json
        model: Anthropic model ID

    Returns:
        list of validated claim dicts
    """
    source_id = content["source_id"]
    extraction_date = date.today().isoformat()

    # Load prompts
    system_prompt = _load_prompt("extraction_system.txt")
    user_template = _load_prompt("extraction_user.txt")

    # Build user prompt
    user_prompt = user_template.format(
        source_id=source_id,
        source_name=content.get("source_name", source_config["source_name"]),
        content_type=content.get("content_type", "blog"),
        content_date=content.get("content_date", extraction_date),
        title=content.get("title", "Untitled"),
        known_bias=source_config.get("known_bias", 0),
        bias_label=source_config.get("bias_label", "unknown"),
        bias_description=source_config.get("bias_description", "No description."),
        extraction_context=source_config.get("extraction_context", ""),
        content_text=content.get("text", ""),
    )

    # Call Claude Haiku API
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = response.content[0].text.strip()

        # Try to parse JSON — handle markdown code fences
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        claims_raw = json.loads(raw_text)

        if not isinstance(claims_raw, list):
            logger.error(f"[{source_id}] API returned non-list: {type(claims_raw)}")
            return []

    except json.JSONDecodeError as e:
        logger.error(f"[{source_id}] JSON parse error: {e}\nRaw: {raw_text[:500]}")
        return []
    except Exception as e:
        logger.error(f"[{source_id}] API call failed: {e}")
        return []

    # Validate and enrich claims
    taxonomy = _load_taxonomy()
    validated = []

    for i, raw_claim in enumerate(claims_raw[:10]):  # max 10
        claim = _validate_claim(raw_claim, source_id, i + 1, extraction_date)
        if claim:
            claim["source_id"] = source_id
            claim["source_name"] = content.get("source_name", source_config["source_name"])
            claim["content_date"] = content.get("content_date", extraction_date)
            claim["content_type"] = content.get("content_type", "blog")
            claim = _enrich_system_relevance(claim, taxonomy)
            validated.append(claim)

    logger.info(
        f"[{source_id}] Extracted {len(validated)} claims from '{content.get('title', 'untitled')}'"
    )
    return validated