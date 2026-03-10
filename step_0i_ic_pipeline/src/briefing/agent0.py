"""
IC Pipeline — Stufe 3: Agent 0 Briefing
Claude Sonnet generates 6-section briefing. Fallback if LLM fails.
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r") as f:
        return f.read()


def _load_yesterday_briefing() -> str:
    """Load yesterday's briefing highlights for continuity."""
    briefings_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "briefings"
    )
    briefings_dir = os.path.normpath(briefings_dir)

    if not os.path.exists(briefings_dir):
        return "No previous briefing available."

    # Find most recent briefing
    files = sorted(
        [f for f in os.listdir(briefings_dir) if f.startswith("briefing_")],
        reverse=True,
    )

    if not files:
        return "No previous briefing available."

    try:
        with open(os.path.join(briefings_dir, files[0]), "r") as f:
            prev = json.load(f)
        # Extract key sections
        text = prev.get("briefing_text", "")
        # Return first 500 chars as highlights
        if text:
            return text[:500] + "..."
        return "Previous briefing had no text content."
    except Exception as e:
        logger.warning(f"Could not load yesterday's briefing: {e}")
        return "Previous briefing unavailable."


def _generate_fallback_briefing(intel: dict) -> str:
    """
    Generate deterministic fallback briefing when LLM fails.
    Per Spec: Template-based, covers all sections.
    """
    today = date.today().isoformat()
    consensus = intel.get("consensus", {})
    divergences = intel.get("divergences", [])
    high_novelty = intel.get("high_novelty_claims", [])
    catalysts = intel.get("catalyst_timeline", [])
    context = intel.get("system_context", {})

    lines = [f"## IC BRIEFING — {today} [FALLBACK — LLM unavailable]\n"]

    # Section 1: What Changed
    lines.append("## WHAT CHANGED\n")
    if high_novelty:
        for hn in high_novelty[:3]:
            lines.append(
                f"- [{hn['source_id']}] (novelty {hn['novelty_score']}): "
                f"{hn['claim_text'][:150]}"
            )
    else:
        lines.append("No significant shifts today. Consensus stable.")

    # Section 2: Consensus Map
    lines.append("\n## CONSENSUS MAP\n")
    for topic, data in consensus.items():
        if data.get("confidence") == "NO_DATA":
            continue
        score = data.get("consensus_score", 0)
        if score > 1:
            direction = "BULLISH"
        elif score < -1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        div_flag = ""
        for d in divergences:
            if d["topic"] == topic:
                div_flag = f" ⚠️ {d['divergence_type']}"
                break
        lines.append(
            f"{topic}: {direction} ({score:+.1f}) "
            f"[{data.get('source_count', 0)} sources]{div_flag}"
        )

    # Section 3: Divergences
    lines.append("\n## DIVERGENCES & BLIND SPOTS\n")
    if divergences:
        for d in divergences[:3]:
            lines.append(
                f"- {d['divergence_type']}: {d['topic']} (severity {d['severity']:.1f}) — "
                f"{d.get('interpretation', '')}"
            )
    else:
        lines.append("No divergences detected.")

    # Section 4: Catalysts
    lines.append("\n## CATALYSTS NEXT 7 DAYS\n")
    if catalysts:
        for c in catalysts[:5]:
            sources = ", ".join(c.get("sources_mentioning", []))
            lines.append(f"- {c['date']} — {c['event']} [{sources}]")
    else:
        lines.append("No catalysts with specific dates identified.")

    # Section 5: System Context
    lines.append("\n## SYSTEM CONTEXT\n")
    lines.append(f"V16 Regime: {context.get('v16_regime', 'UNKNOWN')}")
    lines.append(f"F6 Signals: {context.get('f6_signals_today', 0)}")

    # Section 6: Confirmed Views
    lines.append("\n## CONFIRMED VIEWS\n")
    lines.append("See consensus map above for stable positions.")

    return "\n".join(lines)


def generate_briefing(
    intel: dict,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Generate Agent 0 briefing using Claude Sonnet.

    Args:
        intel: Complete intelligence JSON from Stufe 2
        model: Anthropic model ID

    Returns:
        Briefing dict with briefing_text, metadata, delivery info
    """
    today = date.today().isoformat()
    briefing_id = f"brief_{today.replace('-', '')}_{datetime.utcnow().strftime('%H%M%S')}"

    system_prompt = _load_prompt("agent0_system.txt")
    user_template = _load_prompt("agent0_user.txt")

    # Prepare context
    context = intel.get("system_context", {})
    yesterday = _load_yesterday_briefing()

    # Compact intel for prompt (remove verbose contributor details)
    intel_compact = {
        "consensus": {
            k: {kk: vv for kk, vv in v.items() if kk != "contributors"}
            for k, v in intel.get("consensus", {}).items()
        },
        "divergences": intel.get("divergences", []),
        "high_novelty_claims": intel.get("high_novelty_claims", []),
        "catalyst_timeline": intel.get("catalyst_timeline", []),
        "freshness": intel.get("freshness", {}),
        "extraction_summary": intel.get("extraction_summary", {}),
    }

    user_prompt = user_template.format(
        date=today,
        intel_json_content=json.dumps(intel_compact, indent=2)[:8000],
        v16_regime=context.get("v16_regime", "UNKNOWN"),
        v16_confidence=context.get("v16_confidence", "N/A"),
        f6_signal_count=context.get("f6_signals_today", 0),
        f6_signals=", ".join(context.get("f6_top_signals", [])) or "None",
        data_script_warnings=", ".join(context.get("data_script_warnings", [])) or "None",
        dq_level="UNKNOWN",
        dq_caveat="",
        days_to_fomc="N/A",
        days_to_cpi="N/A",
        days_to_nfp="N/A",
        days_to_ecb="N/A",
        days_to_boj="N/A",
        yesterday_highlights=yesterday,
    )

    # Call Claude Sonnet
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        briefing_text = response.content[0].text.strip()
        generation_model = model
        logger.info(f"Agent 0 briefing generated: {len(briefing_text)} chars")

    except Exception as e:
        logger.error(f"Agent 0 LLM failed: {e}. Using fallback.")
        briefing_text = _generate_fallback_briefing(intel)
        generation_model = "fallback_deterministic"

    # Count metadata
    word_count = len(briefing_text.split())
    topics_covered = sum(
        1 for t, d in intel.get("consensus", {}).items()
        if d.get("confidence") != "NO_DATA"
    )

    briefing = {
        "date": today,
        "briefing_id": briefing_id,
        "generation_model": generation_model,
        "briefing_text": briefing_text,
        "metadata": {
            "word_count": word_count,
            "topics_covered": topics_covered,
            "divergences_flagged": len(intel.get("divergences", [])),
            "catalysts_7d": len(intel.get("catalyst_timeline", [])),
            "action_items": 0,
            "novelty_claims_featured": len(intel.get("high_novelty_claims", [])),
        },
        "delivery": {
            "vercel_pushed": False,
            "email_sent": False,
        },
    }

    return briefing