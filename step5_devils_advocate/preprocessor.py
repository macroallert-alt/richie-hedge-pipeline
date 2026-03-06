"""
step5_devils_advocate/preprocessor.py
Devil's Advocate Pre-Processor — 8 Phases, Deterministic

Phase 1: Input Validation + Draft Parsing
Phase 2: Omission Detection
Phase 3: Internal Consistency Check
Phase 4: Drift Detection
Phase 5: Confidence Saturation
Phase 6: DA History Load + Persistent Challenges
Phase 7: Focus Rotation + Perspective Seed
Phase 8: Asymmetry + Output Assembly
"""

import logging
import random
import re
from datetime import date, datetime, timedelta

logger = logging.getLogger("da_preprocessor")


# =============================================================================
# PHASE 1: INPUT VALIDATION + DRAFT PARSING
# =============================================================================

def validate_inputs(inputs: dict) -> dict:
    """Check all inputs. Only DRAFT_MEMO is mandatory."""
    manifest = {}

    if "draft_memo" not in inputs or inputs["draft_memo"] is None:
        return {"can_run": False, "reason": "DRAFT_MEMO missing"}
    manifest["draft_memo"] = {"status": "COMPLETE", "date": inputs["draft_memo"].get("date")}

    for key in ["risk_alerts", "signals", "layer_analysis",
                "ic_intelligence", "v16_production", "f6_production",
                "yesterday_final"]:
        if key in inputs and inputs[key] is not None and inputs[key] != {}:
            manifest[key] = {"status": "COMPLETE"}
        else:
            manifest[key] = {"status": "MISSING"}

    for key in ["da_history"]:
        if key in inputs and inputs[key] is not None:
            manifest[key] = {"status": "COMPLETE"}
        else:
            manifest[key] = {"status": "MISSING", "note": "First run or history unavailable"}

    return {"can_run": True, "manifest": manifest}


def parse_draft(draft_memo: dict, config: dict) -> dict:
    """Extract structured data from CIO Draft for checks."""
    briefing_text = draft_memo.get("briefing_text", "")

    # Extract sections
    sections = {}
    for i in range(1, 8):
        marker = f"S{i}:"
        next_marker = f"S{i+1}:" if i < 7 else "KEY ASSUMPTIONS"
        start = _find_section_start(briefing_text, i)
        if start is not None:
            end = _find_section_start(briefing_text, i + 1) if i < 7 else None
            if end is None:
                # Try KEY ASSUMPTIONS as end marker
                ka_pos = briefing_text.find("KEY ASSUMPTIONS")
                end = ka_pos if ka_pos > start else len(briefing_text)
            sections[f"S{i}"] = briefing_text[start:end]

    # Extract mentioned entities
    entities = config.get("known_entities", {})
    mentioned_alert_ids = [aid for aid in entities.get("alert_ids", []) if aid in briefing_text]
    mentioned_tickers = _extract_tickers(briefing_text, entities)
    mentioned_ic_sources = [s for s in entities.get("ic_sources", []) if s in briefing_text]
    mentioned_patterns = [p for p in entities.get("pattern_names", []) if p in briefing_text]

    key_assumptions = draft_memo.get("key_assumptions", [])
    header = draft_memo.get("preprocessor_output", {}).get("header", {})

    return {
        "sections": sections,
        "mentioned_alert_ids": mentioned_alert_ids,
        "mentioned_tickers": mentioned_tickers,
        "mentioned_ic_sources": mentioned_ic_sources,
        "mentioned_patterns": mentioned_patterns,
        "key_assumptions": key_assumptions,
        "header": header,
        "briefing_type": draft_memo.get("briefing_type"),
        "system_conviction": draft_memo.get("system_conviction"),
        "word_count": len(briefing_text.split()),
    }


def _find_section_start(text: str, section_num: int) -> int | None:
    """Find start of section S1-S7 in briefing text."""
    patterns = [
        f"## S{section_num}:",
        f"**S{section_num}:",
        f"S{section_num}:",
        f"# S{section_num}:",
    ]
    for p in patterns:
        idx = text.find(p)
        if idx >= 0:
            return idx
    return None


def _extract_tickers(text: str, entities: dict) -> list:
    """Extract ticker symbols from text. Only matches known V16/F6 tickers."""
    known = entities.get("v16_tickers", [])
    # Only match known tickers as whole words (avoid substring matches like
    # "GLD" inside "ENGLAND"). Use word boundary regex per ticker.
    mentioned = []
    for ticker in known:
        if re.search(r'\b' + re.escape(ticker) + r'\b', text):
            mentioned.append(ticker)
    return mentioned


# =============================================================================
# PHASE 2: OMISSION DETECTION
# =============================================================================

def run_omission_detection(parsed_draft: dict, inputs: dict, config: dict) -> list:
    """Run all omission checks and aggregate."""
    omissions = []
    omissions += _check_alert_omissions(parsed_draft, inputs.get("risk_alerts"), config)
    omissions += _check_ic_omissions(parsed_draft, inputs.get("ic_intelligence"), config)
    omissions += _check_pattern_omissions(parsed_draft, inputs.get("draft_memo", {}))
    omissions += _check_position_omissions(parsed_draft, inputs.get("risk_alerts"), config)

    sig_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    omissions.sort(key=lambda o: sig_order.get(o.get("significance", "LOW"), 99))
    return omissions


def _check_alert_omissions(parsed_draft: dict, risk_alerts: dict | None, config: dict) -> list:
    """Check if Risk Officer alerts are mentioned in draft."""
    if not risk_alerts:
        return []
    omissions = []
    om_cfg = config.get("omission_detection", {}).get("alert_omission", {})
    must_trends = om_cfg.get("must_mention_trends", ["NEW", "ESCALATING"])
    must_sevs = om_cfg.get("must_mention_severities_if_stable", ["WARNING", "CRITICAL", "EMERGENCY"])

    for alert in risk_alerts.get("alerts", []):
        check_id = alert.get("check_id", "")
        severity = alert.get("severity", "")
        trend = alert.get("trend", "STABLE")
        must_mention = (
            trend in must_trends
            or (trend == "STABLE" and severity in must_sevs)
        )
        if must_mention and check_id not in parsed_draft["mentioned_alert_ids"]:
            omissions.append({
                "flag_type": "ALERT_OMISSION",
                "check_id": check_id,
                "severity": severity,
                "trend": trend,
                "significance": "HIGH" if severity in ["WARNING", "CRITICAL", "EMERGENCY"] else "MEDIUM",
                "detail": f"Risk Officer Alert {check_id} ({severity}, {trend}) nicht im Draft erwaehnt",
            })

    # Emergency triggers
    for trig_id, trig_data in risk_alerts.get("emergency_triggers", {}).items():
        is_active = False
        if isinstance(trig_data, dict):
            is_active = trig_data.get("status") == "ACTIVE"
        elif isinstance(trig_data, bool):
            is_active = trig_data

        if is_active:
            s3 = parsed_draft["sections"].get("S3", "")
            if trig_id not in s3:
                omissions.append({
                    "flag_type": "EMERGENCY_TRIGGER_OMISSION",
                    "trigger_id": trig_id,
                    "significance": "CRITICAL",
                    "detail": f"Emergency Trigger {trig_id} AKTIV aber nicht in S3 erwaehnt",
                })
    return omissions


def _check_ic_omissions(parsed_draft: dict, ic_intel: dict | None, config: dict) -> list:
    """Check if high-novelty IC claims are mentioned."""
    if not ic_intel:
        return []
    omissions = []
    om_cfg = config.get("omission_detection", {}).get("ic_omission", {})
    high_thresh = om_cfg.get("high_novelty_threshold", 7)
    mod_thresh = om_cfg.get("moderate_novelty_threshold", 5)

    for claim in ic_intel.get("high_novelty_claims", []):
        novelty = claim.get("novelty_score", 0)
        source = claim.get("source_name", claim.get("source_id", "Unknown"))
        if novelty >= high_thresh and source not in parsed_draft["mentioned_ic_sources"]:
            omissions.append({
                "flag_type": "IC_HIGH_NOVELTY_OMISSION",
                "source": source,
                "novelty_score": novelty,
                "significance": "HIGH",
                "detail": f"IC Claim von {source} (Novelty {novelty}) nicht im Draft erwaehnt",
            })
        elif novelty >= mod_thresh and source not in parsed_draft["mentioned_ic_sources"]:
            omissions.append({
                "flag_type": "IC_MODERATE_NOVELTY_OMISSION",
                "source": source,
                "novelty_score": novelty,
                "significance": "LOW",
                "detail": f"IC Claim von {source} (Novelty {novelty}) nicht erwaehnt — moeglicherweise relevant",
            })

    # Divergences
    for div in ic_intel.get("divergences", []):
        div_type = div.get("divergence_type", "UNKNOWN")
        topic = div.get("topic", div.get("theme", "UNKNOWN"))
        s5 = parsed_draft["sections"].get("S5", "")
        if div_type not in s5 and topic not in s5:
            sev = div.get("severity", 0)
            omissions.append({
                "flag_type": "IC_DIVERGENCE_OMISSION",
                "divergence_type": div_type,
                "topic": topic,
                "significance": "HIGH" if sev >= 5.0 else "MEDIUM",
                "detail": f"IC Divergenz {div_type} auf {topic} (Severity {sev}) nicht in S5 erwaehnt",
            })
    return omissions


def _check_pattern_omissions(parsed_draft: dict, draft_memo: dict) -> list:
    """Check if Class A patterns from CIO preprocessor are in draft."""
    omissions = []
    active_patterns = draft_memo.get("preprocessor_output", {}).get(
        "patterns", {}
    ).get("class_a_active", [])

    for pattern in active_patterns:
        pname = pattern.get("pattern", "") if isinstance(pattern, dict) else str(pattern)
        if pname and pname not in parsed_draft["mentioned_patterns"]:
            omissions.append({
                "flag_type": "PATTERN_OMISSION",
                "pattern": pname,
                "significance": "HIGH",
                "detail": f"Klasse A Pattern {pname} nicht im Draft erwaehnt — Pflichtinhalt!",
            })
    return omissions


def _check_position_omissions(parsed_draft: dict, risk_alerts: dict | None, config: dict) -> list:
    """Check if tickers from risk alerts are mentioned in draft."""
    if not risk_alerts:
        return []
    omissions = []
    entities = config.get("known_entities", {})

    for alert in risk_alerts.get("alerts", []):
        if alert.get("severity") in ["WARNING", "CRITICAL", "EMERGENCY"]:
            detail_str = str(alert.get("details", {}))
            detail_tickers = _extract_tickers(detail_str, entities)
            for ticker in detail_tickers:
                if ticker not in parsed_draft["mentioned_tickers"]:
                    omissions.append({
                        "flag_type": "POSITION_OMISSION",
                        "ticker": ticker,
                        "context": f"Referenziert in {alert.get('check_id')} {alert.get('severity')}",
                        "significance": "MEDIUM",
                        "detail": f"{ticker} in Risk Alert {alert.get('check_id')} erwaehnt aber nicht im Draft",
                    })
    return omissions


# =============================================================================
# PHASE 3: INTERNAL CONSISTENCY CHECK
# =============================================================================

def check_internal_consistency(parsed_draft: dict, draft_memo: dict, config: dict) -> list:
    """Find contradictions within the CIO Draft."""
    flags = []
    cc_cfg = config.get("consistency_check", {})
    if not cc_cfg.get("enabled", True):
        return flags

    s3 = parsed_draft["sections"].get("S3", "")
    s7 = parsed_draft["sections"].get("S7", "")
    s4 = parsed_draft["sections"].get("S4", "")
    s6 = parsed_draft["sections"].get("S6", "")
    s1 = parsed_draft["sections"].get("S1", "")
    briefing_type = parsed_draft["briefing_type"]
    conviction = parsed_draft["system_conviction"]

    has_warning_plus = any(sev in s3 for sev in ["WARNING", "CRITICAL", "EMERGENCY"])
    has_no_action = "KEINE AKTION" in s7.upper() or "NO ACTION" in s7.upper()

    # Check 1: S3 severity vs S7 action level
    if has_warning_plus and has_no_action:
        flags.append({
            "flag_type": "CONSISTENCY_S3_S7",
            "significance": "HIGH",
            "detail": "S3 enthaelt WARNING+ Alerts, aber S7 sagt keine Aktion. "
                      "Alerts sollten mindestens REVIEW-Items erzeugen.",
        })

    # Check 2: Briefing type vs S7
    if briefing_type == "ACTION" and has_no_action:
        flags.append({
            "flag_type": "CONSISTENCY_TYPE_S7",
            "significance": "HIGH",
            "detail": "Briefing-Typ ist ACTION, aber S7 sagt keine Aktion. Inkonsistenz.",
        })
    if briefing_type == "ROUTINE" and "ACT" in s7 and not has_no_action:
        flags.append({
            "flag_type": "CONSISTENCY_TYPE_S7",
            "significance": "MEDIUM",
            "detail": "Briefing-Typ ist ROUTINE, aber S7 enthaelt ACT-Items. "
                      "Briefing-Typ sollte mindestens WATCH sein.",
        })

    # Check 3: Conviction vs tone
    tone_cfg = cc_cfg.get("tone_analysis", {})
    cautious_words = tone_cfg.get("cautious_words", [])
    confident_words = tone_cfg.get("confident_words", [])
    s4_lower = s4.lower()
    cautious_count = sum(1 for w in cautious_words if w in s4_lower)
    confident_count = sum(1 for w in confident_words if w in s4_lower)

    if conviction == "LOW" and confident_count > cautious_count + 2:
        flags.append({
            "flag_type": "CONSISTENCY_CONVICTION_TONE",
            "significance": "MEDIUM",
            "detail": "System Conviction ist LOW, aber S4 Ton klingt zuversichtlich.",
        })
    if conviction == "HIGH" and cautious_count > confident_count + 2:
        flags.append({
            "flag_type": "CONSISTENCY_CONVICTION_TONE",
            "significance": "LOW",
            "detail": "System Conviction ist HIGH, aber S4 Ton klingt uebervorsichtig.",
        })

    # Check 4: S1 delta vs header
    risk_ampel = draft_memo.get("risk_ampel", "")
    if "GREEN" in s1 and risk_ampel and risk_ampel != "GREEN":
        flags.append({
            "flag_type": "CONSISTENCY_S1_HEADER",
            "significance": "MEDIUM",
            "detail": f"S1 erwaehnt GREEN, aber Header Risk Ampel ist {risk_ampel}.",
        })

    # Check 5: S6 portfolio vs S3 risk
    if has_warning_plus:
        s6_lower = s6.lower()
        if "unveraendert" in s6_lower or "unchanged" in s6_lower:
            flags.append({
                "flag_type": "CONSISTENCY_S6_S3",
                "significance": "MEDIUM",
                "detail": "S6 sagt Portfolio unveraendert, aber S3 hat WARNING+ Alerts.",
            })

    return flags


# =============================================================================
# PHASE 4: DRIFT DETECTION
# =============================================================================

def detect_drift(parsed_draft: dict, draft_memo: dict,
                 yesterday_final: dict | None, inputs: dict, config: dict) -> list:
    """Detect overreaction or complacency drift vs yesterday."""
    if not config.get("drift_detection", {}).get("enabled", True):
        return []
    if yesterday_final is None:
        return []

    flags = []
    drift_cfg = config.get("drift_detection", {})

    # Tone changes
    type_order = {"ROUTINE": 0, "WATCH": 1, "ACTION": 2, "EMERGENCY": 3}
    conv_order = {"HIGH": 2, "MODERATE": 1, "LOW": 0}
    ampel_order = {"GREEN": 0, "YELLOW": 1, "RED": 2, "BLACK": 3}

    yesterday_type = yesterday_final.get("briefing_type", "ROUTINE")
    today_type = parsed_draft["briefing_type"] or "WATCH"
    type_delta = type_order.get(today_type, 0) - type_order.get(yesterday_type, 0)

    yesterday_conv = yesterday_final.get("system_conviction", "MODERATE")
    today_conv = parsed_draft["system_conviction"] or "MODERATE"
    conv_delta = conv_order.get(today_conv, 1) - conv_order.get(yesterday_conv, 1)

    yesterday_ampel = yesterday_final.get("risk_ampel", "GREEN")
    today_ampel = draft_memo.get("risk_ampel", "GREEN")
    ampel_delta = ampel_order.get(today_ampel, 0) - ampel_order.get(yesterday_ampel, 0)

    tone_changes = {
        "briefing_type": {"yesterday": yesterday_type, "today": today_type, "delta": type_delta},
        "conviction": {"yesterday": yesterday_conv, "today": today_conv, "delta": conv_delta},
        "risk_ampel": {"yesterday": yesterday_ampel, "today": today_ampel, "delta": ampel_delta},
    }

    # Data changes
    risk_alerts = inputs.get("risk_alerts", {})
    new_alerts = sum(1 for a in risk_alerts.get("alerts", []) if a.get("trend") == "NEW")
    escalating = sum(1 for a in risk_alerts.get("alerts", []) if a.get("trend") == "ESCALATING")
    regime_changed = draft_memo.get("v16_regime", "") != yesterday_final.get("v16_regime", "")
    frag_changed = draft_memo.get("fragility_state", "HEALTHY") != yesterday_final.get("fragility_state", "HEALTHY")

    data_changes = {
        "new_alerts": new_alerts,
        "escalating_alerts": escalating,
        "regime_changed": regime_changed,
        "fragility_changed": frag_changed,
    }

    total_tone = abs(type_delta) + abs(conv_delta) + abs(ampel_delta)
    total_data = new_alerts + escalating + (2 if regime_changed else 0) + (1 if frag_changed else 0)

    over_cfg = drift_cfg.get("overreaction_threshold", {})
    comp_cfg = drift_cfg.get("complacency_threshold", {})

    # Overreaction
    if total_tone >= over_cfg.get("tone_delta_min", 2) and total_data <= over_cfg.get("data_delta_max", 0):
        flags.append({
            "flag_type": "DRIFT_OVERREACTION",
            "significance": "HIGH",
            "tone_changes": tone_changes,
            "data_changes": data_changes,
            "detail": f"CIO Ton eskaliert ({yesterday_type}->{today_type}, "
                      f"{yesterday_conv}->{today_conv}) aber keine Datenaenderung.",
        })
    elif total_tone >= 1 and total_data == 0:
        flags.append({
            "flag_type": "DRIFT_MILD_OVERREACTION",
            "significance": "LOW",
            "tone_changes": tone_changes,
            "data_changes": data_changes,
            "detail": f"CIO Ton veraendert ({yesterday_type}->{today_type}) ohne messbare Datenaenderung.",
        })

    # Complacency
    if total_data >= comp_cfg.get("data_delta_min", 2) and total_tone <= comp_cfg.get("tone_delta_max", 0):
        flags.append({
            "flag_type": "DRIFT_COMPLACENCY",
            "significance": "HIGH",
            "tone_changes": tone_changes,
            "data_changes": data_changes,
            "detail": f"Daten veraendern sich ({new_alerts} neue Alerts, "
                      f"{'Regime-Wechsel' if regime_changed else 'kein Regime-Wechsel'}) "
                      f"aber CIO Ton bleibt bei {today_type}/{today_conv}.",
        })
    elif total_data >= 1 and total_tone == 0:
        flags.append({
            "flag_type": "DRIFT_MILD_COMPLACENCY",
            "significance": "LOW",
            "tone_changes": tone_changes,
            "data_changes": data_changes,
            "detail": "Datenaenderungen vorhanden aber CIO Ton unveraendert.",
        })

    return flags


# =============================================================================
# PHASE 5: CONFIDENCE SATURATION
# =============================================================================

def check_confidence_saturation(parsed_draft: dict, draft_memo: dict,
                                inputs: dict, config: dict) -> dict:
    """Detect state of maximum system agreement — Illusion of Safety."""
    sat_cfg = config.get("confidence_saturation", {})
    threshold = sat_cfg.get("threshold", 0.85)
    weights = sat_cfg.get("weights", {})
    components = {}

    # 1: All ampeln green
    risk_ampel = draft_memo.get("risk_ampel", "GREEN")
    components["all_ampeln_green"] = (risk_ampel == "GREEN")

    # 2: All layers aligned
    layer_analysis = inputs.get("layer_analysis", {})
    layers = layer_analysis.get("layers", layer_analysis.get("layer_scores", {}))
    if layers:
        scores = []
        for v in layers.values():
            if isinstance(v, dict):
                scores.append(v.get("score", 0))
            elif isinstance(v, (int, float)):
                scores.append(v)
        if scores:
            positive = sum(1 for s in scores if s > 0)
            negative = sum(1 for s in scores if s < 0)
            total = len(scores)
            alignment = max(positive, negative) / total if total > 0 else 0
            components["all_layers_aligned"] = (alignment > 0.70)
        else:
            components["all_layers_aligned"] = False
    else:
        components["all_layers_aligned"] = False

    # 3: IC consensus high
    ic = inputs.get("ic_intelligence", {})
    consensus = ic.get("consensus", ic.get("ic_consensus", {}))
    if consensus and isinstance(consensus, dict):
        high_conf = sum(1 for v in consensus.values()
                        if isinstance(v, dict) and v.get("confidence") == "HIGH")
        total_topics = len(consensus)
        ic_consensus_pct = high_conf / total_topics if total_topics > 0 else 0
        components["ic_consensus_high"] = (ic_consensus_pct > 0.70)
    else:
        components["ic_consensus_high"] = False

    # 4: Conviction HIGH
    components["conviction_high"] = (parsed_draft["system_conviction"] == "HIGH")

    # 5: No divergences
    divergences = ic.get("divergences", [])
    components["no_divergences"] = (len(divergences) == 0)

    # 6: Fragility HEALTHY
    fragility = draft_memo.get("fragility_state", "HEALTHY")
    components["fragility_healthy"] = (fragility == "HEALTHY")

    # 7: Briefing type ROUTINE
    components["briefing_routine"] = (parsed_draft["briefing_type"] == "ROUTINE")

    # Score
    default_weights = {
        "all_ampeln_green": 0.15, "all_layers_aligned": 0.20,
        "ic_consensus_high": 0.15, "conviction_high": 0.15,
        "no_divergences": 0.10, "fragility_healthy": 0.10,
        "briefing_routine": 0.15,
    }
    w = {k: weights.get(k, default_weights.get(k, 0.1)) for k in default_weights}
    score = sum(w[k] * (1.0 if components.get(k, False) else 0.0) for k in w)

    active = score > threshold
    return {
        "active": active,
        "score": round(score, 3),
        "components": components,
        "interpretation": (
            "CONFIDENCE_SATURATION: System ist sich ungewoehnlich einig. "
            "Historisch sind Momente maximaler Einigkeit oft Wendepunkte."
        ) if active else f"Confidence Saturation bei {score:.0%} — unter Schwelle ({threshold:.0%}).",
    }


# =============================================================================
# PHASE 6: DA HISTORY LOAD + PERSISTENT CHALLENGES
# =============================================================================

def load_da_history_and_persistent(da_history: dict | None, config: dict) -> dict:
    """Load DA History, identify persistent and forced-decision challenges."""
    if da_history is None:
        return {
            "open_challenges": [],
            "persistent_challenges": [],
            "forced_decision_challenges": [],
            "acceptance_rate": {},
            "is_first_run": True,
        }

    noted_threshold = config.get("persistence", {}).get("noted_to_forced_decision_days", 3)
    open_challenges = da_history.get("open_challenges", [])
    persistent = []
    forced_decision = []

    for challenge in open_challenges:
        responses = challenge.get("cio_responses", [])
        noted_count = sum(1 for r in responses if r.get("resolution") == "NOTED")

        if noted_count > 0 and noted_count < noted_threshold:
            persistent.append({
                **challenge,
                "noted_count": noted_count,
                "escalation_note": (
                    f"Tag {challenge.get('days_open', 1)}, {noted_count}x NOTED. "
                    f"{'Naechstes NOTED erzwingt Entscheidung.' if noted_count == noted_threshold - 1 else ''}"
                ),
            })
        elif noted_count >= noted_threshold:
            forced_decision.append({
                **challenge,
                "noted_count": noted_count,
                "escalation_note": (
                    f"FORCED DECISION: {noted_count}x NOTED. "
                    f"CIO MUSS ACCEPTED oder REJECTED waehlen."
                ),
            })

    return {
        "open_challenges": open_challenges,
        "persistent_challenges": persistent,
        "forced_decision_challenges": forced_decision,
        "acceptance_rate": da_history.get("challenge_effectiveness", {}),
        "is_first_run": False,
    }


# =============================================================================
# PHASE 7: FOCUS ROTATION + PERSPECTIVE SEED
# =============================================================================

def select_focus(today: date, da_history: dict | None, config: dict) -> dict:
    """Select primary DA focus for today. Quasi-random, weighted by acceptance rate."""
    focuses = config.get("focus_rotation", {}).get("focuses",
                         ["NARRATIVE", "UNASKED_QUESTION", "PREMISE_ATTACK"])
    min_weight = config.get("focus_rotation", {}).get("min_weight_per_focus", 0.10)

    if da_history is None:
        random.seed(today.toordinal())
        primary = random.choice(focuses)
        weights_used = "equal"
    else:
        acc_rates = da_history.get("challenge_effectiveness", {})
        weights = []
        for focus in focuses:
            rate_key = f"acceptance_rate_{focus.lower()}"
            rate = acc_rates.get(rate_key, 0.15)
            weights.append(max(rate, min_weight))
        total = sum(weights)
        weights = [w / total for w in weights]
        random.seed(today.toordinal())
        primary = random.choices(focuses, weights=weights, k=1)[0]
        weights_used = dict(zip(focuses, [round(w, 3) for w in weights]))

    secondary = [f for f in focuses if f != primary]
    return {
        "primary_focus": primary,
        "secondary_focuses": secondary,
        "weights_used": weights_used,
    }


def select_perspective_seed(today: date, da_history: dict | None, config: dict) -> dict:
    """Select perspective seed. Rotates every 20 trading days."""
    seeds = config.get("perspective_seeds", {}).get("seeds", [])
    interval = config.get("perspective_seeds", {}).get("rotation_interval_trading_days", 20)

    if not seeds:
        return {"seed_index": 0, "seed_id": "NONE", "seed_label": "Default",
                "seed_instruction": "", "rotation_due": False}

    if da_history is None:
        seed_index = 0
    else:
        last_rotation = da_history.get("last_seed_rotation")
        current_index = da_history.get("perspective_seed_index", 0)
        if last_rotation:
            try:
                last_date = datetime.strptime(last_rotation, "%Y-%m-%d").date()
                calendar_days = (today - last_date).days
                trading_days_approx = int(calendar_days * 5 / 7)
                if trading_days_approx >= interval:
                    seed_index = (current_index + 1) % len(seeds)
                else:
                    seed_index = current_index
            except (ValueError, TypeError):
                seed_index = current_index
        else:
            seed_index = 0

    seed = seeds[seed_index % len(seeds)]
    return {
        "seed_index": seed_index,
        "seed_id": seed.get("id", ""),
        "seed_label": seed.get("label", ""),
        "seed_instruction": seed.get("instruction", ""),
        "rotation_due": False,
    }


# =============================================================================
# PHASE 8: ASYMMETRY + OUTPUT ASSEMBLY
# =============================================================================

def calculate_asymmetry(parsed_draft: dict, confidence_saturation: dict, config: dict) -> dict:
    """Control how many flags go to LLM based on briefing type + conviction."""
    asym_cfg = config.get("asymmetry", {})
    briefing_type = parsed_draft["briefing_type"]
    conviction = parsed_draft["system_conviction"]
    saturation = confidence_saturation["active"]

    if briefing_type == "ROUTINE" and conviction == "HIGH":
        mode_cfg = asym_cfg.get("routine_high", {})
        guidance = (
            "Heute ist ein ruhiger Tag mit hoher Conviction. "
            "Das ist der GEFAEHRLICHSTE Zustand — Complacency. "
            "Suche besonders nach dem was NICHT passiert, NICHT erwaehnt wird, "
            "NICHT gemessen wird."
        )
    elif briefing_type == "EMERGENCY" or conviction == "LOW":
        mode_cfg = asym_cfg.get("emergency_low", {})
        guidance = (
            "Der CIO ist bereits maximal alarmiert. "
            "Frage stattdessen: Fokussiert der CIO auf das RICHTIGE Risiko? "
            "Gibt es stabilisierende Faktoren die er in der Krise uebersieht?"
        )
    elif briefing_type == "ACTION":
        mode_cfg = asym_cfg.get("action", {})
        guidance = (
            "ACTION-Tag: Der CIO sieht Handlungsbedarf. "
            "Pruefe: Sind die PRIORITAETEN richtig? Fokussiert er auf "
            "das wichtigste Risiko oder auf das lauteste?"
        )
    else:
        mode_cfg = asym_cfg.get("default", {})
        guidance = "Standard-Modus. Pruefe Praemissen, suche nach Omissions, biete Alternativen."

    return {
        "mode": mode_cfg.get("mode", "STANDARD_SCRUTINY"),
        "max_flags_to_llm": mode_cfg.get("max_flags", 10),
        "challenge_guidance": guidance,
        "min_challenges": mode_cfg.get("min_challenges", 1),
        "saturation_bonus": saturation,
    }


def assemble_preprocessor_output(
    input_manifest: dict, parsed_draft: dict,
    omissions: list, consistency_flags: list, drift_flags: list,
    confidence_saturation: dict, da_history_data: dict,
    focus_selection: dict, perspective_seed: dict, asymmetry: dict,
) -> dict:
    """Pack all preprocessor results into structured JSON for LLM call."""
    all_flags = omissions + consistency_flags + drift_flags
    sig_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    all_flags.sort(key=lambda f: sig_order.get(f.get("significance", "LOW"), 99))

    max_flags = asymmetry["max_flags_to_llm"]
    flags_for_llm = all_flags[:max_flags]
    flags_unused = all_flags[max_flags:]

    return {
        "preprocessor_version": "1.0",
        "date": str(date.today()),
        "input_status": input_manifest,
        "flags": {
            "for_llm": flags_for_llm,
            "total_found": len(all_flags),
            "sent_to_llm": len(flags_for_llm),
            "unused": flags_unused,
        },
        "confidence_saturation": confidence_saturation,
        "persistent_challenges": da_history_data["persistent_challenges"],
        "forced_decision_challenges": da_history_data["forced_decision_challenges"],
        "focus": focus_selection,
        "perspective_seed": perspective_seed,
        "asymmetry": asymmetry,
        "draft_metadata": {
            "briefing_type": parsed_draft["briefing_type"],
            "system_conviction": parsed_draft["system_conviction"],
            "word_count": parsed_draft["word_count"],
            "sections_present": list(parsed_draft["sections"].keys()),
            "key_assumptions_count": len(parsed_draft["key_assumptions"]),
        },
        "acceptance_rate": da_history_data["acceptance_rate"],
    }
