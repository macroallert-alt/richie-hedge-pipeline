"""
IC Pipeline — Stufe 1: Claim Extractor V2
Extraction V2: Two-call architecture per IC V2 Spec Kapitel 4.

Call A: Thesis Map + Portfolio Transmission (full text -> structured theses)
Call B: Action Filter + Linguistic Temperature + Second Derivative
        (only if >=2 historical posts exist for source)
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
# Constants & Validation Sets
# ---------------------------------------------------------------------------
VALID_CLAIM_TYPES = {"FACT", "OPINION", "PREDICTION", "ANALYSIS"}
VALID_DIRECTIONS = {"BULLISH", "BEARISH", "NEUTRAL", "MIXED"}
VALID_TIMEFRAMES = {"IMMEDIATE", "SHORT", "MEDIUM", "LONG"}
VALID_TIME_HORIZONS = {"TACTICAL", "CYCLICAL", "STRUCTURAL"}
VALID_ALIGNMENTS = {"CONFIRMING", "THREATENING", "MIXED", "NEUTRAL"}
VALID_ASSET_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
VALID_ACTION_RECS = {"ACT", "MONITOR", "IGNORE"}
VALID_CONVICTION_TRENDS = {"RISING", "STABLE", "FALLING"}
VALID_TOPICS = {
    "LIQUIDITY", "FED_POLICY", "CREDIT", "RECESSION", "INFLATION",
    "EQUITY_VALUATION", "CHINA_EM", "GEOPOLITICS", "ENERGY",
    "COMMODITIES", "TECH_AI", "CRYPTO", "DOLLAR", "VOLATILITY", "POSITIONING",
}


# ---------------------------------------------------------------------------
# JSON Parsing Helper
# ---------------------------------------------------------------------------
def _parse_json_response(raw_text: str, source_id: str, call_label: str) -> list:
    """Parse JSON array from LLM response, handling markdown fences and preamble."""
    raw_text = raw_text.strip()

    # Strip markdown code fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```\s*$", "", raw_text)

    # Find the start of the JSON array
    bracket_pos = raw_text.find("[")
    if bracket_pos == -1:
        logger.error(
            f"[{source_id}] {call_label}: No JSON array found in response: "
            f"{raw_text[:300]}"
        )
        return []

    # Use raw_decode to parse first JSON value, ignore trailing text
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(raw_text, bracket_pos)
    except json.JSONDecodeError as e:
        logger.error(
            f"[{source_id}] {call_label}: JSON parse error: {e}\n"
            f"Raw: {raw_text[:500]}"
        )
        return []

    if not isinstance(parsed, list):
        logger.error(
            f"[{source_id}] {call_label}: API returned non-list: {type(parsed)}"
        )
        return []

    return parsed


# ---------------------------------------------------------------------------
# V16 Context Builder
# ---------------------------------------------------------------------------
def _build_v16_context(v16_context: dict | None) -> tuple[str, str, list[dict]]:
    """Build V16 context strings for Call A prompt.

    Returns: (v16_regime, v16_positions_text, v16_positions_list)
    """
    if not v16_context:
        return "UNKNOWN", "No V16 data available.", []

    regime = v16_context.get("regime", "UNKNOWN")
    weights = v16_context.get("current_weights", {})

    # Filter to active positions (weight > 0)
    active = sorted(
        [(asset, w) for asset, w in weights.items() if w > 0.005],
        key=lambda x: -x[1],
    )

    if not active:
        return regime, "No active positions.", []

    lines = []
    positions_list = []
    for asset, weight in active:
        pct = round(weight * 100, 1)
        lines.append(f"  {asset}: {pct}%")
        positions_list.append({"position": asset, "v16_weight_pct": pct})

    return regime, "\n".join(lines), positions_list


# ---------------------------------------------------------------------------
# Source History Builder
# ---------------------------------------------------------------------------
def _build_source_history(
    source_id: str,
    claims_archive: dict | None,
    current_content_date: str,
) -> tuple[str, int]:
    """Build source history JSON string from claims archive for Call B.

    Returns: (history_json_string, post_count)
    Groups claims by content_date, returns most recent 4 dates.
    Excludes claims that were extracted TODAY (same extraction_date) to avoid
    including claims from the current batch. Uses extraction_date instead of
    content_date for filtering because multiple posts can share the same
    content_date (e.g. ZeroHedge posts 5 articles on the same day).
    """
    if not claims_archive:
        return "[]", 0

    today_str = date.today().isoformat()

    # Collect claims for this source, grouped by content_date
    # Exclude claims extracted today (current run's output)
    date_claims: dict[str, list[dict]] = {}
    for claim in claims_archive.get("claims", []):
        if claim.get("source_id") != source_id:
            continue
        # Skip claims from current extraction run
        if claim.get("extraction_date", "") == today_str:
            continue
        cd = claim.get("content_date", "")
        if not cd:
            continue
        if cd not in date_claims:
            date_claims[cd] = []
        date_claims[cd].append({
            "claim_text": claim.get("claim_text", "")[:200],
            "direction": claim.get("sentiment", {}).get("direction", ""),
            "intensity": claim.get("sentiment", {}).get("intensity", 0),
            "topics": claim.get("topics", []),
            "novelty_score": claim.get("novelty_score", 0),
            "speaker_confidence": claim.get("speaker_confidence"),
            "linguistic_temperature": claim.get("linguistic_temperature"),
        })

    if not date_claims:
        return "[]", 0

    # Sort by date desc, take last 4 dates
    sorted_dates = sorted(date_claims.keys(), reverse=True)[:4]
    history = []
    for d in sorted_dates:
        history.append({
            "content_date": d,
            "claims": date_claims[d],
        })

    return json.dumps(history, indent=2), len(sorted_dates)


# ---------------------------------------------------------------------------
# Call A Validation
# ---------------------------------------------------------------------------
def _validate_call_a_thesis(
    thesis: dict, source_id: str, seq: int, extraction_date: str,
    v16_positions: list[dict],
) -> Optional[dict]:
    """Validate and normalize a single thesis from Call A. Returns None if invalid."""
    # Required: claim_text
    if not thesis.get("claim_text"):
        return None

    # --- V1-compatible fields (downstream needs these) ---

    # claim_type
    ct = thesis.get("claim_type", "OPINION").upper()
    if ct not in VALID_CLAIM_TYPES:
        ct = "OPINION"
    thesis["claim_type"] = ct

    # sentiment
    sent = thesis.get("sentiment", {})
    direction = sent.get("direction", "NEUTRAL").upper()
    if direction not in VALID_DIRECTIONS:
        direction = "NEUTRAL"
    intensity = sent.get("intensity", 5)
    if not isinstance(intensity, (int, float)):
        intensity = 5
    intensity = max(1, min(10, int(intensity)))
    thesis["sentiment"] = {"direction": direction, "intensity": intensity}

    # topics
    topics = thesis.get("topics", [])
    topics = [t.upper() for t in topics if t.upper() in VALID_TOPICS]
    if not topics:
        topics = ["LIQUIDITY"]
    thesis["topics"] = topics[:3]

    primary = thesis.get("primary_topic", "").upper()
    if primary not in VALID_TOPICS or primary not in topics:
        primary = topics[0]
    thesis["primary_topic"] = primary

    # timeframe
    tf = thesis.get("timeframe", "MEDIUM").upper()
    if tf not in VALID_TIMEFRAMES:
        tf = "MEDIUM"
    thesis["timeframe"] = tf

    # novelty
    nov = thesis.get("novelty_score", 3)
    if not isinstance(nov, (int, float)):
        nov = 3
    thesis["novelty_score"] = max(0, min(10, int(nov)))

    # extraction_confidence
    conf = thesis.get("extraction_confidence", 0.8)
    if not isinstance(conf, (int, float)):
        conf = 0.8
    conf = max(0.0, min(1.0, float(conf)))

    # ID + metadata
    thesis["id"] = f"claim_{extraction_date.replace('-', '')}_{source_id}_{seq:03d}"
    thesis["extraction_date"] = extraction_date
    thesis["confidence"] = {"extraction_confidence": conf}

    # catalyst (optional)
    if not thesis.get("catalyst"):
        thesis["catalyst"] = None
    if not thesis.get("catalyst_date"):
        thesis["catalyst_date"] = None
    if not thesis.get("novelty_note"):
        thesis["novelty_note"] = ""

    # --- V2 fields ---

    # reasoning_chain
    rc = thesis.get("reasoning_chain", [])
    if not isinstance(rc, list):
        rc = []
    thesis["reasoning_chain"] = [str(s)[:300] for s in rc[:8]]

    # speaker_confidence
    sc = thesis.get("speaker_confidence")
    if not isinstance(sc, (int, float)):
        sc = 5
    thesis["speaker_confidence"] = max(1, min(10, int(sc)))

    if not thesis.get("speaker_confidence_signals"):
        thesis["speaker_confidence_signals"] = ""

    # time_horizon
    th = thesis.get("time_horizon", "").upper()
    if th not in VALID_TIME_HORIZONS:
        th = "CYCLICAL"
    thesis["time_horizon"] = th

    if not thesis.get("time_horizon_detail"):
        thesis["time_horizon_detail"] = ""

    # affected_assets
    aa = thesis.get("affected_assets", [])
    if not isinstance(aa, list):
        aa = []
    validated_aa = []
    for asset_entry in aa[:6]:
        if not isinstance(asset_entry, dict) or not asset_entry.get("asset"):
            continue
        a_dir = asset_entry.get("direction", "NEUTRAL").upper()
        if a_dir not in VALID_DIRECTIONS:
            a_dir = "NEUTRAL"
        a_conf = asset_entry.get("confidence", "MEDIUM").upper()
        if a_conf not in VALID_ASSET_CONFIDENCE:
            a_conf = "MEDIUM"
        validated_aa.append({
            "asset": str(asset_entry["asset"]).upper(),
            "direction": a_dir,
            "mechanism": str(asset_entry.get("mechanism", ""))[:200],
            "confidence": a_conf,
        })
    thesis["affected_assets"] = validated_aa

    # v16_position_impact
    vpi = thesis.get("v16_position_impact", [])
    if not isinstance(vpi, list):
        vpi = []
    # Validate against actual V16 positions
    v16_assets = {p["position"] for p in v16_positions}
    validated_vpi = []
    for impact in vpi[:6]:
        if not isinstance(impact, dict):
            continue
        pos = str(impact.get("position", "")).upper()
        if pos not in v16_assets:
            continue
        alignment = impact.get("alignment", "NEUTRAL").upper()
        if alignment not in VALID_ALIGNMENTS:
            alignment = "NEUTRAL"
        # Find actual weight
        actual_weight = next(
            (p["v16_weight_pct"] for p in v16_positions if p["position"] == pos),
            0,
        )
        validated_vpi.append({
            "position": pos,
            "v16_weight_pct": actual_weight,
            "alignment": alignment,
            "detail": str(impact.get("detail", ""))[:200],
        })
    thesis["v16_position_impact"] = validated_vpi

    # surprise_flag
    thesis["surprise_flag"] = bool(thesis.get("surprise_flag", False))
    if not thesis.get("surprise_detail"):
        thesis["surprise_detail"] = None

    # --- Call B fields (defaults, overwritten if Call B runs) ---
    thesis["action_recommendation"] = "MONITOR"
    thesis["action_detail"] = None
    thesis["linguistic_temperature"] = None
    thesis["linguistic_detail"] = None
    thesis["is_repeat"] = False
    thesis["repeat_detail"] = None
    thesis["is_position_shift"] = False
    thesis["position_shift_detail"] = None
    thesis["second_derivative"] = None

    return thesis


# ---------------------------------------------------------------------------
# Call B: Action Filter Merge
# ---------------------------------------------------------------------------
def _merge_call_b_results(theses: list[dict], call_b_results: list[dict]) -> list[dict]:
    """Merge Call B action filter results into Call A theses."""
    # Build index map from Call B
    b_map = {}
    for entry in call_b_results:
        idx = entry.get("thesis_index")
        if isinstance(idx, int) and 0 <= idx < len(theses):
            b_map[idx] = entry

    for i, thesis in enumerate(theses):
        b = b_map.get(i)
        if not b:
            continue

        # action_recommendation
        ar = str(b.get("action_recommendation", "MONITOR")).upper()
        if ar in VALID_ACTION_RECS:
            thesis["action_recommendation"] = ar
        thesis["action_detail"] = b.get("action_detail")

        # linguistic_temperature
        lt = b.get("linguistic_temperature")
        if isinstance(lt, (int, float)):
            thesis["linguistic_temperature"] = max(1, min(10, int(lt)))
        thesis["linguistic_detail"] = b.get("linguistic_detail")

        # is_repeat / is_position_shift
        thesis["is_repeat"] = bool(b.get("is_repeat", False))
        thesis["repeat_detail"] = b.get("repeat_detail")
        thesis["is_position_shift"] = bool(b.get("is_position_shift", False))
        thesis["position_shift_detail"] = b.get("position_shift_detail")

        # second_derivative
        sd = b.get("second_derivative")
        if isinstance(sd, dict):
            ct = str(sd.get("conviction_trend", "STABLE")).upper()
            if ct not in VALID_CONVICTION_TRENDS:
                ct = "STABLE"
            thesis["second_derivative"] = {
                "conviction_trend": ct,
                "shift_detected": bool(sd.get("shift_detected", False)),
                "shift_detail": sd.get("shift_detail"),
            }

        # Call B can override novelty_score if it detects repeat
        if thesis["is_repeat"]:
            thesis["novelty_score"] = min(thesis["novelty_score"], 2)

        # Call B can boost novelty if position shift detected
        if thesis["is_position_shift"]:
            thesis["novelty_score"] = max(thesis["novelty_score"], 9)

    return theses


# ---------------------------------------------------------------------------
# Main Extraction Entry Point
# ---------------------------------------------------------------------------
def extract_claims(
    content: dict,
    source_config: dict,
    model: str = "claude-sonnet-4-6",
    v16_context: dict | None = None,
    claims_archive: dict | None = None,
) -> list[dict]:
    """
    Extract structured claims from content using Extraction V2 (two-call architecture).

    Call A: Thesis Map + Portfolio Transmission (always runs)
    Call B: Action Filter + Second Derivative (only if >=2 historical posts exist)

    Args:
        content: dict with source_id, source_name, content_date, title, text, content_type
        source_config: full source config from sources.json
        model: Anthropic model ID
        v16_context: dict with regime, current_weights from latest.json (optional)
        claims_archive: dict with claims[] for source history lookup (optional)

    Returns:
        list of validated claim dicts (V1-compatible + V2 extensions)
    """
    source_id = content["source_id"]
    extraction_date = date.today().isoformat()
    content_date = content.get("content_date", extraction_date)

    # Build V16 context
    v16_regime, v16_positions_text, v16_positions_list = _build_v16_context(v16_context)

    # ===================================================================
    # CALL A: Thesis Map + Portfolio Transmission
    # ===================================================================
    system_prompt_a = _load_prompt("extraction_v2_call_a_system.txt")
    user_template_a = _load_prompt("extraction_v2_call_a_user.txt")

    user_prompt_a = user_template_a.format(
        source_id=source_id,
        source_name=content.get("source_name", source_config["source_name"]),
        content_type=content.get("content_type", "blog"),
        content_date=content_date,
        title=content.get("title", "Untitled"),
        known_bias=source_config.get("known_bias", 0),
        bias_label=source_config.get("bias_label", "unknown"),
        bias_description=source_config.get("bias_description", "No description."),
        extraction_context=source_config.get("extraction_context", ""),
        v16_regime=v16_regime,
        v16_positions_text=v16_positions_text,
        content_text=content.get("text", ""),
    )

    client = anthropic.Anthropic()
    theses = []

    try:
        response_a = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt_a,
            messages=[{"role": "user", "content": user_prompt_a}],
        )
        raw_text_a = response_a.content[0].text
        theses_raw = _parse_json_response(raw_text_a, source_id, "Call A")

    except Exception as e:
        logger.error(f"[{source_id}] Call A failed: {e}")
        return []

    # Validate Call A output
    for i, raw_thesis in enumerate(theses_raw[:5]):  # max 5
        thesis = _validate_call_a_thesis(
            raw_thesis, source_id, i + 1, extraction_date, v16_positions_list
        )
        if thesis:
            thesis["source_id"] = source_id
            thesis["source_name"] = content.get(
                "source_name", source_config["source_name"]
            )
            thesis["content_date"] = content_date
            thesis["content_type"] = content.get("content_type", "blog")
            theses.append(thesis)

    if not theses:
        logger.warning(f"[{source_id}] Call A produced 0 valid theses")
        return []

    logger.info(
        f"[{source_id}] Call A: {len(theses)} theses extracted "
        f"from '{content.get('title', 'untitled')}'"
    )

    # ===================================================================
    # CALL B: Action Filter + Second Derivative (conditional)
    # ===================================================================
    source_history_json, history_post_count = _build_source_history(
        source_id, claims_archive, content_date
    )

    if history_post_count >= 2:
        # Build compact theses summary for Call B
        theses_for_b = []
        for t in theses:
            theses_for_b.append({
                "claim_text": t["claim_text"],
                "direction": t["sentiment"]["direction"],
                "intensity": t["sentiment"]["intensity"],
                "topics": t["topics"],
                "novelty_score": t["novelty_score"],
                "speaker_confidence": t["speaker_confidence"],
            })

        system_prompt_b = _load_prompt("extraction_v2_call_b_system.txt")
        user_template_b = _load_prompt("extraction_v2_call_b_user.txt")

        user_prompt_b = user_template_b.format(
            source_id=source_id,
            source_name=content.get("source_name", source_config["source_name"]),
            known_bias=source_config.get("known_bias", 0),
            bias_label=source_config.get("bias_label", "unknown"),
            theses_json=json.dumps(theses_for_b, indent=2),
            source_history_json=source_history_json,
        )

        try:
            response_b = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt_b,
                messages=[{"role": "user", "content": user_prompt_b}],
            )
            raw_text_b = response_b.content[0].text
            call_b_results = _parse_json_response(raw_text_b, source_id, "Call B")

            if call_b_results:
                theses = _merge_call_b_results(theses, call_b_results)
                logger.info(
                    f"[{source_id}] Call B: action filter merged "
                    f"({history_post_count} historical posts)"
                )

        except Exception as e:
            logger.warning(
                f"[{source_id}] Call B failed (non-fatal, defaults used): {e}"
            )
    else:
        logger.info(
            f"[{source_id}] Call B skipped — only {history_post_count} "
            f"historical posts (need >=2)"
        )

    # ===================================================================
    # Enrich with system relevance (deterministic)
    # ===================================================================
    taxonomy = _load_taxonomy()
    for thesis in theses:
        thesis = _enrich_system_relevance(thesis, taxonomy)

    logger.info(
        f"[{source_id}] Extraction V2 complete: {len(theses)} theses"
    )
    return theses
