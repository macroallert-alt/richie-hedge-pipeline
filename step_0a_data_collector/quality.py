"""
quality.py — Phase 3: Data Quality
====================================
Stufe 1: Plausibilitaets-Checks (Range, Non-Zero, Max-Move)
Stufe 2: Anomaly Detection (OK/JUMP/DRIFT/ANOMALY)
Stufe 3: Cross-Validation (Feldpaar-Konsistenz)
Stufe 4: System-Level Coherence
Output: anomaly_flag pro Feld + DQ_SUMMARY
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("data_collector.quality")


# ═══════════════════════════════════════════════════════
# STUFE 1: PLAUSIBILITAETS-CHECKS
# ═══════════════════════════════════════════════════════

def plausibility_check(field_name: str, new_value: float, previous_value: Optional[float],
                       registry_entry: dict) -> str:
    if new_value is None:
        return "OK"
    p = registry_entry.get("plausibility", {})
    if not p:
        return "OK"

    if p.get("min") is not None and new_value < p["min"]:
        logger.warning(f"{field_name}: {new_value} < min {p['min']}")
        return "IMPLAUSIBLE"
    if p.get("max") is not None and new_value > p["max"]:
        logger.warning(f"{field_name}: {new_value} > max {p['max']}")
        return "IMPLAUSIBLE"
    if p.get("non_negative") and new_value < 0:
        return "IMPLAUSIBLE"
    if p.get("non_zero") and new_value == 0:
        return "IMPLAUSIBLE"

    max_move = p.get("max_daily_move_pct", 999)
    if previous_value and previous_value != 0 and max_move < 999:
        pct_change = abs(new_value - previous_value) / abs(previous_value)
        if pct_change > max_move:
            logger.warning(f"{field_name}: {pct_change:.2%} move > {max_move:.2%}")
            return "SUSPICIOUS"
    return "OK"


# ═══════════════════════════════════════════════════════
# STUFE 2: ANOMALY DETECTION
# ═══════════════════════════════════════════════════════

def detect_anomaly(field_name: str, value_today: float,
                   value_yesterday: Optional[float],
                   history_2y: Optional[pd.Series]) -> str:
    if value_today is None:
        return "OK"
    if history_2y is None or len(history_2y.dropna()) < 20:
        return "OK"

    valid = history_2y.dropna()
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    if std == 0:
        return "OK"

    z = (value_today - mean) / std
    if abs(z) > 5.0:
        logger.warning(f"{field_name}: z={z:.2f} > 5 → ANOMALY")
        return "ANOMALY"

    if value_yesterday is not None and value_yesterday != 0:
        daily_change = abs(value_today - value_yesterday)
        daily_std = valid.diff().dropna().std()
        if daily_std and daily_std > 0:
            if daily_change / daily_std > 3.5:
                return "JUMP"

    if len(valid) >= 15:
        last_15 = valid.tail(15).dropna()
        if len(last_15) >= 10:
            diffs = last_15.diff().dropna()
            up = (diffs > 0).sum()
            down = (diffs < 0).sum()
            if up >= 13 or down >= 13:
                cumulative = abs(float(last_15.iloc[-1] - last_15.iloc[0]))
                if std > 0 and cumulative / std > 1.5:
                    return "DRIFT"
    return "OK"


# ═══════════════════════════════════════════════════════
# STUFE 3: CROSS-VALIDATION
# ═══════════════════════════════════════════════════════

def cross_validate(transformed: dict) -> Dict[str, str]:
    flags = {}

    def _v(name, attr='value'):
        tf = transformed.get(name)
        return getattr(tf, attr, None) if tf else None

    # VIX vs HY divergence
    vix_d5 = _v('vix', 'delta_5d')
    hy_d5 = _v('hy_oas', 'delta_5d')
    if vix_d5 is not None and hy_d5 is not None:
        if vix_d5 > 5 and hy_d5 < -10:
            flags["VIX_vs_HY_divergence"] = "DIVERGENT"

    # Net Liq component check
    net, walcl, tga, rrp = _v('net_liquidity'), _v('walcl'), _v('tga'), _v('rrp')
    if all(v is not None for v in [net, walcl, tga, rrp]):
        if abs(net - (walcl - tga - rrp)) > 1.0:
            flags["NetLiq_component_check"] = "ANOMALY"

    # Credit spread inversion (IG > HY = Datenfehler)
    ig, hy = _v('ig_oas'), _v('hy_oas')
    if ig is not None and hy is not None and ig > hy:
        flags["Credit_spread_inversion"] = "ANOMALY"

    # VIX term extreme backwardation
    vtr = _v('vix_term_ratio')
    if vtr is not None and vtr < 0.7:
        flags["VIX_term_extreme"] = "WARNING"

    # Extreme calm
    corr, vix = _v('spy_tlt_corr_21d'), _v('vix')
    if corr is not None and vix is not None:
        if corr > 0.8 and vix < 12:
            flags["Extreme_calm"] = "DIVERGENT"

    return flags


# ═══════════════════════════════════════════════════════
# STUFE 4: SYSTEM-LEVEL COHERENCE
# ═══════════════════════════════════════════════════════

def coherence_check(transformed: dict) -> dict:
    significant = 0
    checkable = 0
    for tf in transformed.values():
        if tf.delta_5d is not None and tf.direction is not None:
            checkable += 1
            if tf.direction != "FLAT":
                significant += 1
    move_ratio = significant / checkable if checkable > 0 else 0
    if move_ratio > 0.70:
        return {"status": "CHECK_REQUIRED", "move_ratio": round(move_ratio, 2),
                "note": "70%+ fields moving significantly"}
    return {"status": "OK", "move_ratio": round(move_ratio, 2)}


# ═══════════════════════════════════════════════════════
# QUALITY ENGINE
# ═══════════════════════════════════════════════════════

class QualityEngine:
    def __init__(self, registry: dict, history_cache: dict):
        self.registry = registry
        self.history = history_cache

    def run_all(self, transformed: dict, fetch_results: dict) -> dict:
        logger.info("=" * 60)
        logger.info("PHASE 3: QUALITAET")
        logger.info("=" * 60)

        # Stufe 1: Plausibility
        for field_name, tf in transformed.items():
            cfg = self.registry.get(field_name, {})
            hist = self.history.get(field_name)
            prev = float(hist.dropna().iloc[-2]) if hist is not None and len(hist.dropna()) >= 2 else None
            result = plausibility_check(field_name, tf.value, prev, cfg)
            if result == "IMPLAUSIBLE":
                tf.anomaly_flag = "ANOMALY"
                tf.value = None
            elif result == "SUSPICIOUS":
                tf.anomaly_flag = "SUSPICIOUS"

        # Stufe 2: Anomaly Detection
        for field_name, tf in transformed.items():
            if tf.anomaly_flag == "ANOMALY":
                continue
            hist = self.history.get(field_name)
            prev = float(hist.dropna().iloc[-2]) if hist is not None and len(hist.dropna()) >= 2 else None
            flag = detect_anomaly(field_name, tf.value, prev, hist)
            if flag != "OK":
                tf.anomaly_flag = flag

        # Stufe 3 + 4
        xv_flags = cross_validate(transformed)
        coherence = coherence_check(transformed)

        dq = self._build_summary(transformed, xv_flags, coherence)
        logger.info(f"PHASE 3 COMPLETE: level={dq['data_quality_level']}, "
                    f"ok={dq['fields_ok']}, stale={dq['fields_stale']}, failed={dq['fields_failed']}")
        return dq

    def _build_summary(self, transformed, xv_flags, coherence):
        total = len(transformed)
        ok = stale = failed = llm = anomaly = 0
        t1_issues = []
        sources = {}
        alerts = []

        for fn, tf in transformed.items():
            if tf.source_method == "ALL_FAILED":
                failed += 1
                if tf.tier == "T1":
                    t1_issues.append(fn)
            elif tf.source_method == "STALE_CACHE":
                stale += 1
            elif tf.source_method == "LLM_FALLBACK":
                llm += 1; ok += 1
            else:
                ok += 1
            if tf.anomaly_flag not in ["OK", None]:
                anomaly += 1
            if tf.source:
                sources.setdefault(tf.source, "OK")
                if tf.source_method in ["ALL_FAILED", "STALE_CACHE"]:
                    sources[tf.source] = "DEGRADED"

        ratio = ok / total if total > 0 else 0
        if ratio >= 0.90 and len(t1_issues) == 0:
            level = "FULL"
        elif ratio >= 0.70 and len(t1_issues) <= 2:
            level = "DEGRADED"
        else:
            level = "CRITICAL"

        if stale > 10:
            alerts.append(f"WARN: {stale} stale fields")
        if len(t1_issues) > 3:
            alerts.append(f"CRITICAL: {len(t1_issues)} T1 sources down")
        for rule, flag in xv_flags.items():
            if flag in ["ANOMALY", "DIVERGENT"]:
                alerts.append(f"XV: {rule} → {flag}")
        if coherence["status"] != "OK":
            alerts.append(f"COHERENCE: {coherence['note']}")

        return {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "fields_total": total, "fields_ok": ok, "fields_stale": stale,
            "fields_failed": failed, "fields_llm_fallback": llm,
            "fields_anomaly": anomaly, "data_quality_level": level,
            "t1_issues": t1_issues, "sources_status": sources,
            "coherence": coherence, "cross_validation": xv_flags,
            "alerts": alerts,
        }
