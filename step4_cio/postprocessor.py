"""
step4_cio/postprocessor.py
CIO Post-Processor — 6 Deterministic Steps + Fallback Briefing
Spec: CIO Spec Teil 5

Step 1: Output Validation
Step 2: Fact-Check
Step 3: Action Items Extraction
Step 4: CIO History Digest Update
Step 5: Action Item Tracking Update
Step 6: (Write — handled in main.py)
Fallback: Deterministic briefing when LLM fails
"""

import json
import logging
import re
from datetime import date

logger = logging.getLogger("cio_postprocessor")


# ==========================================================================
# SECTION PARSING (used by engine.py and dashboard_update.py)
# ==========================================================================

def parse_sections(briefing_text: str) -> dict:
    """
    Parse briefing_text into individual sections.
    Looks for ## S1: through ## S7: headers.
    Returns dict like {"S1_delta": "...", "S2_catalysts": "...", ...}
    Spec: CIO Addendum Dashboard Writer §1
    """
    section_keys = {
        "1": "S1_delta",
        "2": "S2_catalysts",
        "3": "S3_risk",
        "4": "S4_patterns",
        "5": "S5_intelligence",
        "6": "S6_portfolio",
        "7": "S7_actions",
    }

    pattern = re.compile(
        r'##\s*S(\d+)[:\s]+(.*?)(?=##\s*S\d+[:\s]|KEY ASSUMPTIONS:|DEVIL\'S ADVOCATE|---\n|$)',
        re.DOTALL,
    )

    sections = {}
    for match in pattern.finditer(briefing_text):
        num = match.group(1)
        text = match.group(2).strip()
        key = section_keys.get(num)
        if key:
            sections[key] = text

    return sections


def extract_sections(briefing_text: str) -> list:
    """Return list of section markers present, e.g. ['S1', 'S2', ...]."""
    found = []
    for i in range(1, 8):
        if f"## S{i}:" in briefing_text:
            found.append(f"S{i}")
    return found


def extract_key_assumptions(briefing_text: str) -> list:
    """
    Extract KEY ASSUMPTIONS block from briefing text.
    Spec: CIO Addendum DA Rueckwirkungen §1
    """
    assumptions = []
    ka_match = re.search(r'KEY ASSUMPTIONS:(.*?)(?:---|\Z)', briefing_text, re.DOTALL)
    if not ka_match:
        return assumptions

    ka_text = ka_match.group(1)
    # Pattern: KA1: id — assumption text\n     Wenn falsch: ...
    ka_pattern = re.compile(
        r'KA\d+:\s*(\S+)\s*[—-]\s*(.*?)(?:Wenn falsch:\s*(.*?))?(?=KA\d+:|$)',
        re.DOTALL,
    )

    for m in ka_pattern.finditer(ka_text):
        assumptions.append({
            "id": m.group(1).strip(),
            "assumption": m.group(2).strip(),
            "vulnerability": (m.group(3) or "").strip(),
        })

    return assumptions


# ==========================================================================
# STEP 1: OUTPUT VALIDATION (Spec Teil 5 §5.2)
# ==========================================================================

def validate_output(briefing_text: str, is_final: bool = False) -> tuple:
    """
    Check if LLM briefing is structurally complete.
    Returns (valid, errors).
    """
    errors = []

    # Required sections
    for i in range(1, 8):
        if f"## S{i}:" not in briefing_text:
            errors.append(f"MISSING_SECTION: ## S{i}:")

    # Header line present (within first 300 chars)
    header_area = briefing_text[:300]
    if not any(m in header_area for m in ("ROUTINE", "WATCH", "ACTION", "EMERGENCY", "FALLBACK")):
        errors.append("MISSING_HEADER")

    # Final-specific: DA Resolution Summary
    if is_final:
        if ("DEVIL'S ADVOCATE" not in briefing_text
                and "DA RESOLUTION" not in briefing_text.upper()):
            # Only warn — draft-as-final won't have this
            pass

    # Minimum length per section
    for i in range(1, 8):
        marker = f"## S{i}:"
        next_marker = f"## S{i + 1}:" if i < 7 else "KEY ASSUMPTIONS"

        if marker in briefing_text:
            start = briefing_text.index(marker)
            if next_marker and next_marker in briefing_text:
                end = briefing_text.index(next_marker)
            else:
                end = len(briefing_text)

            section_content = briefing_text[start:end]
            word_count = len(section_content.split())

            if word_count < 10:
                errors.append(f"SECTION_TOO_SHORT: S{i} ({word_count} words)")

    return (len(errors) == 0, errors)


# ==========================================================================
# STEP 2: FACT-CHECK (Spec Teil 5 §5.3)
# ==========================================================================

def fact_check_briefing(briefing_text: str, inputs: dict,
                        preprocessor_output: dict) -> list:
    """
    Deterministic fact-check: compare LLM output against input data.
    Only the most critical checks.
    """
    flags = []
    header = preprocessor_output.get("header", {})

    # Check 1: V16 Regime correct?
    actual_regime = inputs.get("v16_production", {}).get("regime", "")
    if actual_regime:
        for regime in ("RISK_ON", "RISK_OFF", "TRANSITION", "SELECTIVE"):
            if (regime != actual_regime
                    and regime in briefing_text
                    and f"V16" in briefing_text[:briefing_text.index(regime) + 50]
                    if regime in briefing_text else False):
                # Only flag if it appears near "V16" context
                pass  # Too many false positives with simple string matching
        # Simpler: check header line
        header_line = briefing_text[:300]
        if actual_regime not in header_line and "V16" in header_line:
            for r in ("RISK_ON", "RISK_OFF", "TRANSITION", "SELECTIVE"):
                if r != actual_regime and r in header_line:
                    flags.append({
                        "type": "REGIME_MISMATCH",
                        "stated": r,
                        "actual": actual_regime,
                        "severity": "CRITICAL",
                    })

    # Check 2: Risk Ampel correct?
    actual_ampel = inputs.get("risk_alerts", {}).get("portfolio_status", "")
    if actual_ampel:
        for ampel in ("GREEN", "YELLOW", "RED", "BLACK"):
            if ampel != actual_ampel and f"Risk: {ampel}" in briefing_text[:300]:
                flags.append({
                    "type": "AMPEL_MISMATCH",
                    "stated": ampel,
                    "actual": actual_ampel,
                    "severity": "CRITICAL",
                })

    # Check 3: Alert severities correct?
    for alert in inputs.get("risk_alerts", {}).get("alerts", []):
        check_id = alert.get("check_id", "")
        actual_sev = alert.get("severity", "")
        if check_id in briefing_text:
            for sev in ("MONITOR", "WARNING", "CRITICAL", "EMERGENCY"):
                if sev != actual_sev and f"{check_id} {sev}" in briefing_text:
                    flags.append({
                        "type": "SEVERITY_MISMATCH",
                        "check": check_id,
                        "stated": sev,
                        "actual": actual_sev,
                        "severity": "HIGH",
                    })

    # Check 4: Fragility State correct?
    actual_frag = header.get("fragility_state", "")
    if actual_frag:
        for state in ("HEALTHY", "ELEVATED", "EXTREME", "CRISIS"):
            if state != actual_frag and f"Fragility: {state}" in briefing_text[:300]:
                flags.append({
                    "type": "FRAGILITY_MISMATCH",
                    "stated": state,
                    "actual": actual_frag,
                    "severity": "HIGH",
                })

    # Check 5: Briefing type in header correct?
    actual_type = header.get("briefing_type", "")
    if actual_type:
        for bt in ("ROUTINE", "WATCH", "ACTION", "EMERGENCY"):
            if bt != actual_type and bt in briefing_text[:200]:
                flags.append({
                    "type": "BRIEFING_TYPE_MISMATCH",
                    "stated": bt,
                    "actual": actual_type,
                    "severity": "MEDIUM",
                })

    return flags


def handle_fact_check_flags(flags: list) -> dict:
    """
    CRITICAL → RETRY_OR_FALLBACK
    HIGH → DELIVER_WITH_FLAGS
    MEDIUM → DELIVER (logged only)
    """
    critical = [f for f in flags if f.get("severity") == "CRITICAL"]
    high = [f for f in flags if f.get("severity") == "HIGH"]

    if critical:
        return {"action": "RETRY_OR_FALLBACK", "reason": critical}

    if high:
        note = "FACT-CHECK FLAGS:\n"
        for f in high:
            note += f"  {f['type']}: stated {f.get('stated')}, actual: {f.get('actual')}\n"
        return {"action": "DELIVER_WITH_FLAGS", "note": note}

    return {"action": "DELIVER", "note": None}


# ==========================================================================
# STEP 3: ACTION ITEMS EXTRACTION (Spec Teil 5 §5.4)
# ==========================================================================

def _extract_type_from_text(text: str):
    """Extract ACT/REVIEW/WATCH from inline markers like (ACT, TODAY, OPEN)."""
    upper = text.upper()
    if "(ACT," in upper or "(ACT " in upper or ", ACT)" in upper:
        return "ACT"
    if "(REVIEW," in upper or "(REVIEW " in upper or ", REVIEW)" in upper:
        return "REVIEW"
    if "(WATCH," in upper or "(WATCH " in upper or ", WATCH)" in upper:
        return "WATCH"
    return None


def extract_action_items(briefing_text: str, preprocessor_output: dict) -> list:
    """Extract action items from S7 into machine-readable JSON.

    Handles three formats:
    1. Legacy: Lines starting with ACT: / REVIEW: / WATCH:
    2. LLM format with section headers (IMMEDIATE/PRE-EVENT/POST-EVENT/ONGOING/WATCHLIST)
    3. LLM self-typed items: **AI-1: Description (ACT, TODAY, OPEN)**
       — extracts type from inline text when header context is ambiguous
    Also handles: **W-1: ...** watchlist items, bullet - **Text:** items
    """
    s7_text = _extract_section_text(briefing_text, "S7")
    items = []

    current_type = None
    import re

    for line in s7_text.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        line_upper = line_stripped.upper()

        # --- Legacy prefix format ---
        if line_upper.startswith("ACT:") or line_upper.startswith("ACT "):
            items.append(_parse_action_item(line_stripped, "ACT", preprocessor_output))
            continue
        if line_upper.startswith("REVIEW:") or line_upper.startswith("REVIEW "):
            items.append(_parse_action_item(line_stripped, "REVIEW", preprocessor_output))
            continue
        if line_upper.startswith("WATCH:") or line_upper.startswith("WATCH "):
            items.append(_parse_action_item(line_stripped, "WATCH", preprocessor_output))
            continue

        # --- Section header detection (LLM format) ---
        is_header = line_stripped.endswith(":**") or line_stripped.endswith(":**)")
        if is_header:
            if "IMMEDIATE" in line_upper or "VOR NFP" in line_upper or "<24H" in line_upper:
                current_type = "ACT"
                continue
            if "PRE-EVENT" in line_upper or "PRE EVENT" in line_upper:
                current_type = "ACT"
                continue
            if "SHORT-TERM" in line_upper or "48H" in line_upper:
                current_type = "REVIEW"
                continue
            if "POST-EVENT" in line_upper or "POST EVENT" in line_upper or "POST-NFP" in line_upper or "POST NFP" in line_upper:
                current_type = "REVIEW"
                continue
            if "ONGOING" in line_upper or "MEDIUM-TERM" in line_upper or "7D-30D" in line_upper:
                current_type = "WATCH"
                continue
            if "DEFERRED" in line_upper:
                current_type = "WATCH"
                continue
            # WATCHLIST header (allow ACTION in surrounding text)
            if "WATCHLIST" in line_upper:
                current_type = "WATCH"
                continue
            # OFFENE/NEUE ACTION ITEMS — items self-type via inline markers
            if "OFFENE" in line_upper or "NEUE" in line_upper or "NEW ACTION" in line_upper:
                current_type = "SELF_TYPE"
                continue

        # --- AI/A-N item detection ---
        item_match = re.match(r'\*\*(?:AI|DF|A)-?\d+[:\s]', line_stripped)
        if item_match:
            desc = re.sub(r'\*\*', '', line_stripped).strip()
            inline_type = _extract_type_from_text(desc)
            item_type = inline_type or (current_type if current_type != "SELF_TYPE" else None) or "REVIEW"
            items.append(_parse_action_item_from_desc(desc, item_type, preprocessor_output))
            continue

        # --- W-N watchlist items ---
        w_match = re.match(r'\*\*W-?\d+[:\s]', line_stripped)
        if w_match:
            desc = re.sub(r'\*\*', '', line_stripped).strip()
            items.append(_parse_action_item_from_desc(desc, "WATCH", preprocessor_output))
            continue

        # --- WL-N watchlist items ---
        wl_match = re.match(r'\*\*WL-?\d+[:\s]', line_stripped)
        if wl_match:
            desc = re.sub(r'\*\*', '', line_stripped).strip()
            items.append(_parse_action_item_from_desc(desc, "WATCH", preprocessor_output))
            continue

        # --- Bullet watchlist items: - **Text:** description ---
        # Exclude known subfield labels that are part of item detail, not items
        bullet_match = re.match(r'^-\s*\*\*(.+?):\*\*', line_stripped)
        if bullet_match and current_type == "WATCH":
            label = bullet_match.group(1).strip()
            subfield_labels = {
                "status", "trigger", "was", "warum", "monitoring",
                "nächster check", "naechster check", "trigger noch aktiv",
                "kontext", "context", "schwelle", "threshold",
                "nächste schritte", "naechste schritte", "next steps",
                "deadline", "warum jetzt", "why now",
            }
            if label.lower() not in subfield_labels:
                items.append(_parse_action_item_from_desc(label, "WATCH", preprocessor_output))

    # Conviction-based upgrade: LOW → REVIEW becomes ACT
    conviction = preprocessor_output.get("header", {}).get("system_conviction", "MODERATE")
    if conviction == "LOW":
        for item in items:
            if item["type"] == "REVIEW":
                item["type"] = "ACT"
                item["conviction_upgrade"] = True
                item["upgrade_reason"] = "LOW System Conviction — REVIEW upgraded to ACT"

    # Assign IDs
    today_str = preprocessor_output.get("date", date.today().isoformat())
    for i, item in enumerate(items):
        item["id"] = f"action_{today_str}_{i + 1:03d}"

    return items


def _parse_action_item_from_desc(description: str, item_type: str,
                                  preprocessor_output: dict) -> dict:
    """Parse an action item from its description text (LLM format)."""
    source_alerts = []
    source_patterns = []

    for alert in preprocessor_output.get("alert_treatment", {}).get("full_treatment", []):
        cid = alert.get("check_id", "")
        if cid and cid.lower() in description.lower():
            source_alerts.append(cid)

    for pattern in preprocessor_output.get("patterns", {}).get("class_a_active", []):
        pname = pattern.get("pattern", "")
        if pname and pname.lower().replace("_", " ") in description.lower():
            source_patterns.append(pname)

    return {
        "type": item_type,
        "description": description,
        "urgency": (
            "TODAY" if item_type == "ACT"
            else "THIS_WEEK" if item_type == "REVIEW"
            else "ONGOING"
        ),
        "source_alerts": source_alerts,
        "source_patterns": source_patterns,
        "conviction_upgrade": False,
    }


def _parse_action_item(line: str, item_type: str, preprocessor_output: dict) -> dict:
    """Parse a single action item line."""
    # Remove type prefix
    for prefix in ("ACT:", "ACT —", "ACT ", "REVIEW:", "REVIEW —", "REVIEW ",
                    "WATCH:", "WATCH —", "WATCH "):
        if line.upper().startswith(prefix):
            description = line[len(prefix):].strip()
            break
    else:
        description = line

    # Try to identify source alerts and patterns
    source_alerts = []
    source_patterns = []

    for alert in preprocessor_output.get("alert_treatment", {}).get("full_treatment", []):
        cid = alert.get("check_id", "")
        if cid and cid.lower() in description.lower():
            source_alerts.append(cid)

    for pattern in preprocessor_output.get("patterns", {}).get("class_a_active", []):
        pname = pattern.get("pattern", "")
        if pname and pname.lower().replace("_", " ") in description.lower():
            source_patterns.append(pname)

    return {
        "type": item_type,
        "description": description,
        "urgency": (
            "TODAY" if item_type == "ACT"
            else "THIS_WEEK" if item_type == "REVIEW"
            else "ONGOING"
        ),
        "source_alerts": source_alerts,
        "source_patterns": source_patterns,
        "conviction_upgrade": False,
    }


# ==========================================================================
# STEP 4: CIO HISTORY DIGEST UPDATE (Spec Teil 5 §5.5)
# ==========================================================================

def update_history_digest(existing_history: dict, briefing_text: str,
                          inputs: dict, preprocessor_output: dict) -> dict:
    """Update CIO History Digest based on today's outputs."""
    today_str = preprocessor_output.get("date", date.today().isoformat())
    bt = preprocessor_output.get("header", {}).get("briefing_type", "WATCH")

    # Consecutive routine days
    if bt == "ROUTINE":
        routine_streak = existing_history.get("consecutive_routine_days", 0) + 1
    else:
        routine_streak = 0

    # Update threads
    active_threads = _update_threads(
        existing_history.get("active_threads", []),
        inputs, preprocessor_output, today_str,
    )

    # Resolved threads
    resolved = _update_resolved_threads(
        existing_history.get("resolved_threads_last_7d", []),
        existing_history.get("active_threads", []),
        active_threads, today_str,
    )

    # Pattern history
    patterns_7d = _update_pattern_history(
        existing_history.get("patterns_last_7d", {}),
        preprocessor_output.get("patterns", {}).get("class_a_active", []),
    )

    return {
        "last_updated": today_str,
        "consecutive_routine_days": routine_streak,
        "active_threads": active_threads,
        "resolved_threads_last_7d": resolved,
        "patterns_last_7d": patterns_7d,
        "open_action_items": [],  # Populated in Step 5
    }


def _update_threads(existing: list, inputs: dict,
                    preprocessor: dict, today_str: str) -> list:
    """Update active threads based on today's data."""
    updated = []
    existing_ids = set()

    # Update existing threads
    for thread in existing:
        tid = thread.get("thread_id", "")
        existing_ids.add(tid)
        still_active = _is_thread_active(thread, inputs, preprocessor)

        if still_active:
            thread["days_active"] = thread.get("days_active", 1) + 1
            thread["last_checked"] = today_str
            updated.append(thread)
        # If not active → will appear in resolved

    # Detect new threads
    new_threads = _detect_new_threads(inputs, preprocessor, existing_ids)
    for nt in new_threads:
        nt["started"] = today_str
        nt["days_active"] = 1
        nt["last_checked"] = today_str
        updated.append(nt)

    return updated


def _is_thread_active(thread: dict, inputs: dict, preprocessor: dict) -> bool:
    """Check if thread source is still active."""
    source = thread.get("source", "")

    if source.startswith("RISK_OFFICER."):
        check_id = source.replace("RISK_OFFICER.", "")
        return any(
            a.get("check_id") == check_id
            for a in inputs.get("risk_alerts", {}).get("alerts", [])
        )

    if source.startswith("SIGNAL_GENERATOR.ROUTER."):
        target = source.replace("SIGNAL_GENERATOR.ROUTER.", "")
        prox = preprocessor.get("temporal_context", {}).get("router_proximity", {})
        return prox.get(target, {}).get("value", 0) > 0.5

    if source.startswith("IC.DIVERGENCE."):
        topic = source.replace("IC.DIVERGENCE.", "")
        return any(
            d.get("topic") == topic
            for d in inputs.get("ic_intelligence", {}).get("divergences", [])
        )

    return True  # Default: keep alive (conservative)


def _detect_new_threads(inputs: dict, preprocessor: dict,
                        existing_ids: set) -> list:
    """Detect new threads from today's data."""
    new = []

    # Risk Officer alerts
    for alert in inputs.get("risk_alerts", {}).get("alerts", []):
        if alert.get("trend") == "NEW":
            tid = f"risk_{alert.get('check_id', '').lower()}"
            if tid not in existing_ids:
                new.append({
                    "thread_id": tid,
                    "source": f"RISK_OFFICER.{alert['check_id']}",
                    "trend": "NEW",
                    "summary": (
                        f"{alert['check_id']} {alert.get('severity', '')}"
                    ),
                })

    # Router proximity
    for target, data in preprocessor.get("temporal_context", {}).get("router_proximity", {}).items():
        if data.get("value", 0) > 0.5 and data.get("trend") == "RISING":
            tid = f"router_{target.lower()}"
            if tid not in existing_ids:
                new.append({
                    "thread_id": tid,
                    "source": f"SIGNAL_GENERATOR.ROUTER.{target}",
                    "trend": "SLOWLY_RISING",
                    "summary": f"Router {target} Proximity: {data['value']:.2f}",
                })

    # IC divergences
    for div in inputs.get("ic_intelligence", {}).get("divergences", []):
        if div.get("severity", 0) >= 3.0:
            topic = div.get("topic", "UNKNOWN")
            tid = f"ic_div_{topic.lower()}"
            if tid not in existing_ids:
                new.append({
                    "thread_id": tid,
                    "source": f"IC.DIVERGENCE.{topic}",
                    "trend": "NEW",
                    "summary": (
                        f"IC vs Market on {topic}: "
                        f"{div.get('divergence_type', 'UNKNOWN')}"
                    ),
                })

    return new


def _update_resolved_threads(existing_resolved: list, old_active: list,
                             new_active: list, today_str: str) -> list:
    """Track threads that were active yesterday but not today."""
    new_active_ids = {t["thread_id"] for t in new_active}
    newly_resolved = []

    for thread in old_active:
        tid = thread.get("thread_id", "")
        if tid not in new_active_ids:
            newly_resolved.append({
                "thread_id": tid,
                "started": thread.get("started", ""),
                "resolved": today_str,
                "duration": thread.get("days_active", 1),
                "resolution": "Thread no longer active",
            })

    # Keep last 7 days of resolved threads
    combined = existing_resolved + newly_resolved
    # Simple: keep last 50
    return combined[-50:]


def _update_pattern_history(existing: dict, active_patterns: list) -> dict:
    """Count pattern activations."""
    updated = {**existing}
    for p in active_patterns:
        name = p.get("pattern", "")
        updated[name] = updated.get(name, 0) + 1
    return updated


# ==========================================================================
# STEP 5: ACTION ITEM TRACKING UPDATE (Spec Teil 5 §5.6)
# ==========================================================================

def update_action_item_tracking(existing_items: list, new_items: list,
                                inputs: dict, preprocessor_output: dict,
                                config: dict) -> list:
    """
    1. Check existing items: trigger still active?
    2. If not: auto-close
    3. If active + > N days: escalation
    4. Add new items (dedup by description similarity)
    """
    escalation_days = config.get("history", {}).get("action_item_escalation_days", 3)
    act_urgent_days = config.get("history", {}).get("act_item_urgent_escalation_days", 1)

    updated = []

    for item in existing_items:
        if _is_trigger_still_active(item, inputs, preprocessor_output):
            item["days_open"] = item.get("days_open", 1) + 1
            item["trigger_still_active"] = True

            # Escalation
            if item["days_open"] >= escalation_days and item.get("type") == "REVIEW":
                item["escalated"] = True
                item["escalation_note"] = (
                    f"REVIEW offen seit {item['days_open']} Tagen — Eskalation"
                )

            if item["days_open"] > act_urgent_days and item.get("type") == "ACT":
                item["escalated"] = True
                item["escalation_note"] = (
                    f"ACT-Item offen seit {item['days_open']} Tagen — DRINGEND"
                )

            updated.append(item)
        else:
            # Auto-close
            logger.info(
                f"Auto-closing action item: {item.get('id', '?')} — "
                f"trigger no longer active"
            )

    # Add new items (dedup by AI-ID prefix like "AI-1:", "W-2:", or description)
    existing_ids = set()
    for i in updated:
        desc = i.get("description", "")
        ai_id = _extract_ai_id(desc)
        if ai_id:
            existing_ids.add(ai_id)
        else:
            existing_ids.add(desc.lower()[:50])

    for new_item in new_items:
        desc = new_item.get("description", "")
        ai_id = _extract_ai_id(desc)
        dedup_key = ai_id if ai_id else desc.lower()[:50]
        if dedup_key not in existing_ids:
            new_item["days_open"] = 1
            new_item["trigger_still_active"] = True
            new_item["status"] = "OPEN"
            updated.append(new_item)
            existing_ids.add(dedup_key)

    return updated


def _extract_ai_id(description: str):
    """Extract AI-1, AI-10, W-1, WL-3 etc. from description for dedup."""
    import re
    match = re.match(r'((?:AI|WL|DF|W|A)-?\d+)', description)
    if match:
        return match.group(1).upper()
    return None


def _is_trigger_still_active(item: dict, inputs: dict,
                             preprocessor: dict) -> bool:
    """Check if trigger conditions are still active."""
    # Check source alerts
    for sa in item.get("source_alerts", []):
        active_checks = {
            a.get("check_id") for a in inputs.get("risk_alerts", {}).get("alerts", [])
            if a.get("severity") in ("WARNING", "CRITICAL", "EMERGENCY")
        }
        if sa in active_checks:
            return True

    # Check source patterns
    active_patterns = {
        p["pattern"]
        for p in preprocessor.get("patterns", {}).get("class_a_active", [])
    }
    for sp in item.get("source_patterns", []):
        if sp in active_patterns:
            return True

    # If no specific sources tracked, default to active (conservative)
    if not item.get("source_alerts") and not item.get("source_patterns"):
        return True

    return False


# ==========================================================================
# DA RESOLUTION EXTRACTION (Spec Teil 5 — only for Final)
# ==========================================================================

def extract_da_resolution(briefing_text: str) -> dict:
    """
    Extract DA Resolution Summary from Final briefing.
    Looks for inline [DA: ...] markers and the resolution summary block.
    """
    resolution = {
        "total_challenges": 0,
        "accepted": 0,
        "noted": 0,
        "rejected": 0,
        "details": [],
        "modified_sections": [],
    }

    # Find inline DA markers: [DA: ... ACCEPTED/NOTED/REJECTED ...]
    da_pattern = re.compile(
        r'\[DA:\s*(.*?)\.\s*(ACCEPTED|NOTED|REJECTED)\s*[—-]\s*(.*?)\]',
        re.DOTALL,
    )

    for match in da_pattern.finditer(briefing_text):
        challenge_summary = match.group(1).strip()
        marker_type = match.group(2).strip()
        cio_response = match.group(3).strip()

        resolution["total_challenges"] += 1
        resolution[marker_type.lower()] += 1

        # Determine which section this DA marker is in
        pos = match.start()
        section = "UNKNOWN"
        for i in range(7, 0, -1):
            marker = f"## S{i}:"
            if marker in briefing_text[:pos]:
                section = f"S{i}"
                break

        if marker_type == "ACCEPTED" and section not in resolution["modified_sections"]:
            resolution["modified_sections"].append(section)

        resolution["details"].append({
            "section": section,
            "marker_type": marker_type,
            "challenge_summary": challenge_summary,
            "cio_response": cio_response,
        })

    return resolution


# ==========================================================================
# DA RESOLUTION WRITE-BACK — Match Resolutions to DA Challenge IDs
# ==========================================================================

def build_cio_resolutions(da_resolution: dict, devils_advocate: dict) -> list:
    """
    Match extracted DA resolutions from the Final briefing text
    with the original DA challenge IDs and types.

    Args:
        da_resolution: Output from extract_da_resolution() — has 'details' list
        devils_advocate: Full DA output JSON — has 'challenges' list

    Returns:
        List of dicts: [{challenge_id, challenge_type, resolution}, ...]
        Ready for update_effectiveness_after_cio_final().
    """
    if not da_resolution or not devils_advocate:
        return []

    da_details = da_resolution.get("details", [])
    da_challenges = devils_advocate.get("challenges", [])

    if not da_details or not da_challenges:
        return []

    cio_resolutions = []
    matched_ids = set()

    for detail in da_details:
        summary = detail.get("challenge_summary", "").strip().lower()
        resolution = detail.get("marker_type", "")  # ACCEPTED / NOTED / REJECTED

        if not summary or not resolution:
            continue

        # Match by text similarity: first 50 chars of summary vs challenge_text
        best_match = None
        best_score = 0

        for challenge in da_challenges:
            cid = challenge.get("id", "")
            if cid in matched_ids:
                continue

            ctext = challenge.get("challenge_text", "").strip().lower()

            # Strategy 1: summary is substring of challenge_text
            if summary[:50] in ctext:
                best_match = challenge
                best_score = 100
                break

            # Strategy 2: challenge_text starts with summary prefix
            if ctext[:50] in summary and len(ctext[:50]) > 10:
                if best_score < 80:
                    best_match = challenge
                    best_score = 80

            # Strategy 3: word overlap (fallback)
            summary_words = set(summary.split())
            ctext_words = set(ctext.split())
            if len(summary_words) > 0:
                overlap = len(summary_words & ctext_words) / len(summary_words)
                if overlap > 0.5 and overlap > best_score / 100:
                    best_match = challenge
                    best_score = int(overlap * 100)

        if best_match:
            matched_ids.add(best_match["id"])
            cio_resolutions.append({
                "challenge_id": best_match["id"],
                "challenge_type": best_match.get("type", "PREMISE_ATTACK"),
                "resolution": resolution,
            })
            logger.info(
                f"  DA Write-back: {best_match['id']} → {resolution} "
                f"(match score: {best_score})"
            )
        else:
            logger.warning(
                f"  DA Write-back: No match for resolution '{summary[:60]}...' "
                f"({resolution})"
            )

    # For unmatched DA challenges (no resolution in text): default to NOTED
    for challenge in da_challenges:
        if challenge.get("id") not in matched_ids:
            cio_resolutions.append({
                "challenge_id": challenge["id"],
                "challenge_type": challenge.get("type", "PREMISE_ATTACK"),
                "resolution": "NOTED",
            })
            logger.info(
                f"  DA Write-back: {challenge['id']} → NOTED (no marker in text)"
            )

    return cio_resolutions


# ==========================================================================
# FALLBACK BRIEFING (Spec Teil 5 §5.8)
# ==========================================================================

def generate_fallback_briefing(preprocessor_output: dict | None,
                               inputs: dict) -> str:
    """
    Deterministic fallback briefing without LLM.
    Structured, complete, but no narrative or Class B observations.
    """
    if preprocessor_output:
        header = preprocessor_output.get("header", {})
        pp_date = preprocessor_output.get("date", date.today().isoformat())
    else:
        header = {}
        pp_date = date.today().isoformat()

    bt = header.get("briefing_type", "WATCH")
    conv = header.get("system_conviction", "LOW")
    ampel = header.get("risk_ampel",
                        inputs.get("risk_alerts", {}).get("portfolio_status", "GREEN"))
    frag = header.get("fragility_state", "UNKNOWN")
    dq = header.get("data_quality", "DEGRADED")
    v16_regime = header.get("v16_regime",
                             inputs.get("v16_production", {}).get("regime", "UNKNOWN"))

    lines = []
    lines.append(
        f"{pp_date} | FALLBACK | {bt} | Conviction: {conv} | "
        f"Risk: {ampel} | Fragility: {frag} | Data: {dq} | V16: {v16_regime}"
    )
    lines.append("")
    lines.append("LLM NICHT VERFUEGBAR — Deterministisches Fallback-Briefing.")
    lines.append("")

    # --- S1: DELTA ---
    lines.append("## S1: DELTA")
    lines.append("")

    v16 = inputs.get("v16_production", {})
    lines.append(
        f"V16: {v16.get('regime', 'UNKNOWN')}. "
        f"DD-Protect: {'AKTIV' if v16.get('dd_protect_status') == 'ACTIVE' else 'Inaktiv'}."
    )

    ma = inputs.get("layer_analysis", {})
    if ma and ma.get("system_regime"):
        lines.append(
            f"Market Analyst: {ma['system_regime']}. "
            f"Fragility: {ma.get('fragility_state', 'N/A')}."
        )

    alerts = inputs.get("risk_alerts", {}).get("alerts", [])
    new_alerts = [a for a in alerts if a.get("trend") == "NEW"]
    esc_alerts = [a for a in alerts if a.get("trend") == "ESCALATING"]
    if new_alerts:
        lines.append(
            f"Neue Alerts: "
            f"{', '.join(a['check_id'] + ' ' + a.get('severity', '') for a in new_alerts)}"
        )
    if esc_alerts:
        lines.append(
            f"Eskalierende Alerts: "
            f"{', '.join(a['check_id'] + ' ' + a.get('severity', '') for a in esc_alerts)}"
        )
    if not new_alerts and not esc_alerts:
        lines.append("Keine neuen oder eskalierenden Alerts.")

    # IC High Novelty
    ic = inputs.get("ic_intelligence", {})
    high_nov = [
        c for c in ic.get("high_novelty_claims", [])
        if c.get("novelty", c.get("novelty_score", 0)) >= 7
    ]
    for claim in high_nov[:5]:
        src = claim.get("source", claim.get("source_id", "?"))
        txt = claim.get("claim", claim.get("claim_text", ""))[:100]
        nov = claim.get("novelty", claim.get("novelty_score", 0))
        lines.append(f"IC Novelty {nov}: {src}: {txt}")

    # --- S2: CATALYSTS & TIMING ---
    lines.append("")
    lines.append("## S2: CATALYSTS & TIMING")
    lines.append("")

    if preprocessor_output:
        tc = preprocessor_output.get("temporal_context", {})
        events_48h = tc.get("events_48h", [])
        if events_48h:
            for ev in events_48h:
                lines.append(
                    f"{ev.get('date', '')} {ev.get('time', '')} — "
                    f"{ev.get('type', '')} (Impact: {ev.get('impact', '?')})"
                )
        else:
            lines.append("Keine Events in 48h.")

        for cc in tc.get("f6_cc_expiry", []):
            lines.append(f"F6 CC: {cc['ticker']} Strike {cc['strike']}, DTE {cc['dte']}")

        for target, data in tc.get("router_proximity", {}).items():
            if data.get("value", 0) > 0.5:
                lines.append(f"Router {target}: {data['value']:.2f} ({data.get('trend', '')})")
    else:
        lines.append("Temporal Context nicht verfuegbar (Pre-Processor nicht gelaufen).")

    # --- S3: RISK & ALERTS ---
    lines.append("")
    lines.append("## S3: RISK & ALERTS")
    lines.append("")

    emg = inputs.get("risk_alerts", {}).get("emergency_triggers", {})
    active_emg = [
        k for k, v in emg.items()
        if isinstance(v, dict) and v.get("status") == "ACTIVE"
    ]
    if active_emg:
        lines.append(f"EMERGENCY TRIGGERS AKTIV: {', '.join(active_emg)}")
    else:
        lines.append("Emergency Triggers: Alle INACTIVE.")

    if preprocessor_output:
        for alert in preprocessor_output.get("alert_treatment", {}).get("full_treatment", []):
            lines.append(
                f"{alert.get('severity', '')} {alert.get('check_id', '')} — "
                f"{alert.get('trend', '?')} (Tag {alert.get('days_active', '?')})"
            )
            if alert.get("recommendation"):
                lines.append(f"  Recommendation: {alert['recommendation']}")

        compressed = preprocessor_output.get("alert_treatment", {}).get("compressed_ongoing", [])
        if compressed:
            lines.append(
                f"Ongoing: {'; '.join(c['one_liner'] for c in compressed)}"
            )
    else:
        for alert in alerts:
            lines.append(
                f"{alert.get('severity', '')} {alert.get('check_id', '')} — "
                f"{alert.get('trend', '?')}"
            )

    # --- S4: PATTERNS & SYNTHESIS ---
    lines.append("")
    lines.append("## S4: PATTERNS & SYNTHESIS")
    lines.append("")

    if preprocessor_output:
        patterns = preprocessor_output.get("patterns", {}).get("class_a_active", [])
        if patterns:
            for p in patterns:
                lines.append(f"PATTERN: {p['pattern']}")
                td = p.get("trigger_data", {})
                lines.append(f"  Trigger: {json.dumps(td, default=str)}")
                lines.append(f"  Urgency: {p.get('urgency_impact', 'N/A')}")
        else:
            lines.append("Keine Klasse A Patterns aktiv.")

        for af in preprocessor_output.get("absence_flags", []):
            lines.append(f"Absenz: {af['type']}: {af.get('interpretation', '')}")
    else:
        lines.append("Pattern-Analyse nicht verfuegbar (Pre-Processor nicht gelaufen).")

    lines.append("Keine Klasse B Observations (LLM nicht verfuegbar).")

    # --- S5: INTELLIGENCE DIGEST ---
    lines.append("")
    lines.append("## S5: INTELLIGENCE DIGEST")
    lines.append("")

    all_claims = ic.get("high_novelty_claims", [])
    if all_claims:
        sorted_claims = sorted(
            all_claims,
            key=lambda c: c.get("novelty", c.get("novelty_score", 0)),
            reverse=True,
        )
        for claim in sorted_claims[:10]:
            src = claim.get("source", claim.get("source_id", "?"))
            nov = claim.get("novelty", claim.get("novelty_score", 0))
            txt = claim.get("claim", claim.get("claim_text", ""))[:150]
            lines.append(f"{src} (Novelty {nov}): {txt}")
    else:
        lines.append("Keine IC-Claims verfuegbar.")

    divs = ic.get("divergences", [])
    if divs:
        lines.append("")
        for d in divs:
            lines.append(
                f"Divergenz: {d.get('divergence_type', '?')} {d.get('topic', '')}"
            )

    # --- S6: PORTFOLIO CONTEXT ---
    lines.append("")
    lines.append("## S6: PORTFOLIO CONTEXT")
    lines.append("")

    weights = v16.get("current_weights", {})
    top5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    if top5:
        parts = [f"{k} {v:.1%}" for k, v in top5 if v > 0]
        lines.append(f"V16 Top-5: {', '.join(parts)}")

    f6 = inputs.get("f6_production", {})
    for pos in f6.get("active_positions", []):
        cc = pos.get("covered_call", {})
        lines.append(
            f"F6: {pos.get('ticker', '?')} "
            f"(CC Strike {cc.get('strike', '?')}, DTE {cc.get('dte', '?')})"
        )

    if not top5 and not f6.get("active_positions"):
        lines.append("Portfolio-Details im Dashboard.")

    # --- S7: ACTION ITEMS ---
    lines.append("")
    lines.append("## S7: ACTION ITEMS & WATCHLIST")
    lines.append("")

    act_items = []
    if preprocessor_output:
        patterns = preprocessor_output.get("patterns", {}).get("class_a_active", [])
        for p in patterns:
            if p.get("urgency_impact") == "ACT":
                act_items.append(f"ACT: Pattern {p['pattern']} — siehe S4")

        for alert in preprocessor_output.get("alert_treatment", {}).get("full_treatment", []):
            sev = alert.get("severity", "")
            if sev in ("CRITICAL", "EMERGENCY"):
                act_items.append(
                    f"ACT: {alert['check_id']} {sev} — "
                    f"{alert.get('recommendation', 'Review erforderlich')}"
                )
            elif sev == "WARNING":
                act_items.append(
                    f"REVIEW: {alert['check_id']} {sev} — "
                    f"{alert.get('recommendation', 'Review empfohlen')}"
                )

        for oi in preprocessor_output.get("history", {}).get("open_action_items", []):
            act_items.append(
                f"OFFEN (Tag {oi.get('days_open', '?')}): "
                f"{oi.get('description', '?')}"
            )

    if act_items:
        lines.extend(act_items)
    else:
        lines.append("KEINE AKTION ERFORDERLICH.")

    lines.append("")
    lines.append("---")
    lines.append("FALLBACK BRIEFING — LLM nicht verfuegbar.")

    return "\n".join(lines)


# ==========================================================================
# HELPERS
# ==========================================================================

def _extract_section_text(briefing_text: str, section_id: str) -> str:
    """Extract text for a specific section (e.g. 'S7')."""
    num = section_id.replace("S", "")
    start_marker = f"## S{num}:"
    next_num = int(num) + 1
    end_marker = f"## S{next_num}:" if next_num <= 7 else "KEY ASSUMPTIONS"

    if start_marker not in briefing_text:
        return ""

    start = briefing_text.index(start_marker) + len(start_marker)

    if end_marker and end_marker in briefing_text:
        end = briefing_text.index(end_marker)
    else:
        # Try other end markers
        for alt_end in ("KEY ASSUMPTIONS", "DEVIL'S ADVOCATE", "---\n"):
            if alt_end in briefing_text[start:]:
                end = start + briefing_text[start:].index(alt_end)
                break
        else:
            end = len(briefing_text)

    return briefing_text[start:end].strip()
