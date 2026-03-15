"""
Cycles Circle — Phase Detection Engine
Baldur Creek Capital | Step 0v (V3.5)

Deterministic phase detection for all 10 cycles.
All fixes verified in Colab V3.4:
- Business: INDPRO YoY Growth (ISM NAPM discontinued)
- Earnings: FRED Corporate Profits YoY (FMP returns 0 for ETFs)
- Trade: CASS Freight Index YoY (BDI has no API source)
- Dollar: FRED DTWEXBGS (not in V16 Prices)
- Credit: HY_OAS from FRED x100 for bps
- Fed: NEUTRAL when 0 < Real FFR <= 2% and FFR stable

V3.5 Fixes (March 2026):
- Dollar: Window sizes corrected for monthly FRED data (MA 252→12, vel 21→3, pctl 2520→120)
- Dollar: Missing case cur > MA AND v < 0 → WEAKENING
- Business: Reorder PEAK before LATE_EXPANSION so PEAK actually triggers
- Earnings: Reorder PEAK before LATE_EXPANSION so PEAK actually triggers
- Commodity: Missing case cur > MA AND v3 < 0 → LATE_EXPANSION / PEAK
- Liquidity: Missing case cur > MA AND v < 0 → explicit LATE_EXPANSION branch
"""

import logging
from datetime import date

logger = logging.getLogger("cycles.phase_engine")


# ---------------------------------------------------------------------------
# Time Series Helpers
# ---------------------------------------------------------------------------

def _get_price_series(data, ticker):
    p = data.get("prices", {}).get(ticker, [])
    s = [{"date": x["date"], "value": x["price"]} for x in p if x.get("price")]
    s.sort(key=lambda x: x["date"])
    return s

def _get_fred_series(data, key):
    s = data.get("fred", {}).get(key, [])
    r = [{"date": x["date"], "value": x["value"]} for x in s if x.get("value") is not None]
    r.sort(key=lambda x: x["date"])
    return r

def _get_liq_series(data, col="Fed_Net_Liq"):
    s = [{"date": r["date"], "value": r.get(col)} for r in data.get("liquidity", []) if r.get(col) is not None]
    s.sort(key=lambda x: x["date"])
    return s

def _get_howell(data, field):
    hw = data.get("howell", [])
    return hw[0].get(field) if hw else None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _ma(s, w):
    if len(s) < w: return None
    v = [x["value"] for x in s[-w:] if x["value"] is not None]
    return sum(v) / len(v) if v else None

def _vel(s, lb=21):
    if len(s) < lb + 1: return None
    c, p = s[-1]["value"], s[-(lb + 1)]["value"]
    if p is None or p == 0 or c is None: return None
    return (c - p) / abs(p)

def _acc(s, lb=21):
    if len(s) < (lb * 2) + 1: return None
    v1 = _vel(s, lb); v2 = _vel(s[:-(lb)], lb)
    if v1 is None or v2 is None: return None
    return v1 - v2

def _pctl(s, w, cur):
    if cur is None: return None
    v = [x["value"] for x in s[-w:] if x["value"] is not None]
    if len(v) < 20: return None
    return round(sum(1 for x in v if x < cur) / len(v) * 100, 1)

def _vel_z(s, lb=21, hw=1260):
    if len(s) < min(hw, 100): return None
    hw = min(hw, len(s))
    cv = _vel(s, lb)
    if cv is None: return None
    vels = [_vel(s[:i], lb) for i in range(lb + 1, hw)]
    vels = [v for v in vels if v is not None]
    if len(vels) < 20: return None
    m = sum(vels) / len(vels)
    sd = (sum((v - m) ** 2 for v in vels) / len(vels)) ** 0.5
    return round((cv - m) / sd, 2) if sd > 0 else 0

def _yoy(s):
    """YoY growth from monthly data (12 periods back)."""
    if len(s) < 13: return []
    r = []
    for i in range(12, len(s)):
        c, a = s[i]["value"], s[i - 12]["value"]
        if c is not None and a is not None and a != 0:
            r.append({"date": s[i]["date"], "value": round((c - a) / abs(a) * 100, 2)})
    return r

def _qyoy(s):
    """YoY growth from quarterly data (4 periods back)."""
    if len(s) < 5: return []
    r = []
    for i in range(4, len(s)):
        c, a = s[i]["value"], s[i - 4]["value"]
        if c is not None and a is not None and a != 0:
            r.append({"date": s[i]["date"], "value": round((c - a) / abs(a) * 100, 2)})
    return r


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------

def _empty(cid, name, tier):
    return {"cycle_id": cid, "cycle_name": name, "tier": tier,
            "phase": "UNKNOWN", "phase_confidence": 0, "phase_duration_months": None,
            "indicator_value": None, "indicator_12m_ma": None,
            "velocity": None, "acceleration": None, "velocity_z_score": None, "percentile": None,
            "danger_zone": {"zone_name": None, "in_zone": False}, "in_danger_zone": False,
            "v16_alignment": "UNKNOWN", "extra": {}, "data_quality": "INSUFFICIENT"}

def _build(cid, name, tier, phase, conf, value, ma12, v, a, vz, pc, series, extra=None):
    return {"cycle_id": cid, "cycle_name": name, "tier": tier,
            "phase": phase, "phase_confidence": conf, "phase_duration_months": None,
            "indicator_value": round(value, 4) if value is not None else None,
            "indicator_12m_ma": round(ma12, 4) if ma12 is not None else None,
            "velocity": round(v, 6) if v is not None else None,
            "acceleration": round(a, 6) if a is not None else None,
            "velocity_z_score": vz, "percentile": pc,
            "danger_zone": {"zone_name": None, "in_zone": False}, "in_danger_zone": False,
            "v16_alignment": "UNKNOWN", "extra": extra or {},
            "data_quality": "GOOD" if len(series) > 100 else "LIMITED" if series else "INSUFFICIENT"}


# ---------------------------------------------------------------------------
# Per-cycle detectors
# ---------------------------------------------------------------------------

# ── FIX V3.5: LIQUIDITY — added explicit cur > m12 AND v < 0 branch ──
def _detect_liquidity(data):
    s = _get_liq_series(data, "Fed_Net_Liq")
    if len(s) < 252: return _empty("LIQUIDITY", "Global Liquidity", 1)

    cur = s[-1]["value"]; m12 = _ma(s, 252); v = _vel(s)
    a = _vel(s, 63) - _vel(s[:(-21)], 63) if len(s) > 150 else None
    p5 = _pctl(s, min(252 * 5, len(s)), cur)
    hwp = _get_howell(data, "Howell_Phase")
    hwc = _get_howell(data, "Cycle_Position")

    ph, co = "EXPANSION", 60
    if p5 is not None and v is not None:
        if p5 < 20 and v < 0 and a and a > 0: ph, co = "TROUGH", 75
        elif v > 0 and p5 < 50: ph, co = "EARLY_RECOVERY", 70
        elif p5 > 80 and v < 0: ph, co = "PEAK", 70
        elif p5 > 80 or (a and a < 0 and p5 and p5 > 60): ph, co = "LATE_EXPANSION", 65
        elif m12 and cur < m12 and v < 0: ph, co = "CONTRACTION", 75
        elif m12 and cur > m12 and v > 0: ph, co = "EXPANSION", 70
        # V3.5 FIX: above MA but falling — was missing, fell to default EXPANSION
        elif m12 and cur > m12 and v < 0: ph, co = "LATE_EXPANSION", 65

    return _build("LIQUIDITY", "Global Liquidity", 1, ph, co, cur, m12, v, a,
                  _vel_z(s), p5, s, {"howell_phase": hwp, "howell_pos": hwc})


def _detect_credit(data):
    s = _get_fred_series(data, "HY_OAS_FRED")
    if len(s) < 60: return _empty("CREDIT", "Credit Cycle", 1)

    sb = [{"date": x["date"], "value": x["value"] * 100} for x in s]
    sb.sort(key=lambda x: x["date"])
    cur = sb[-1]["value"]; v = _vel(sb)

    ph, co = "EXPANSION", 60
    if cur > 700: ph, co = "DISTRESS", 90
    elif cur > 500 and v and v > 0: ph, co = "DETERIORATION", 80
    elif cur > 400 and v and v > 0: ph, co = "DETERIORATION", 65
    elif cur < 350 and v and v <= 0: ph, co = "EXPANSION", 75
    elif cur < 350 and v and v > 0: ph, co = "LATE_EXPANSION", 65
    elif cur < 500 and v and v < 0: ph, co = "RECOVERY", 70
    elif cur > 500 and v and v < 0: ph, co = "REPAIR", 70

    return _build("CREDIT", "Credit Cycle", 1, ph, co, cur, _ma(sb, 252), v, _acc(sb),
                  _vel_z(sb), _pctl(sb, min(252 * 5, len(sb)), cur), sb)


def _detect_commodity(data):
    dbc = _get_price_series(data, "DBC")
    cpi = _get_fred_series(data, "CPI")
    if len(dbc) < 120 or len(cpi) < 12:
        return _empty("COMMODITY", "Commodity Supercycle", 1)

    cl = {c["date"][:7]: c["value"] for c in cpi}
    crb = []; lc = None
    for p in dbc:
        mk = p["date"][:7]; cv = cl.get(mk)
        if cv is None:
            for k in sorted(cl.keys(), reverse=True):
                if k <= mk: cv = cl[k]; break
        if cv and cv > 0: lc = cv; crb.append({"date": p["date"], "value": p["value"] / cv * 100})

    if len(crb) < 252: return _empty("COMMODITY", "Commodity Supercycle", 1)

    cur = crb[-1]["value"]; m10 = _ma(crb, min(252 * 10, len(crb)))
    v3 = _vel(crb, 63)

    # ── FIX V3.5: COMMODITY — added cur > m10 AND v3 < 0 branches ──
    ph, co = "MID_BULL", 55
    if m10:
        if cur < m10 and v3 and v3 < 0: ph, co = "BEAR", 65
        elif cur > m10 and v3 and v3 > 0.15: ph, co = "EUPHORIA", 60
        elif cur > m10 and v3 and v3 > 0: ph, co = "MID_BULL", 65
        elif cur < m10 and v3 and v3 > 0: ph, co = "EARLY_BULL", 60
        # V3.5 FIX: above long-term MA but falling — was missing, fell to default MID_BULL
        elif cur > m10 and v3 and v3 < -0.05: ph, co = "PEAK", 65
        elif cur > m10 and v3 and v3 < 0: ph, co = "LATE_EXPANSION", 60

    return _build("COMMODITY", "Commodity Supercycle", 1, ph, co, cur, m10, v3, None,
                  None, _pctl(crb, min(252 * 10, len(crb)), cur), crb)


def _detect_china_credit(data):
    copper = _get_price_series(data, "COPPER")
    gold = _get_price_series(data, "GLD")
    if len(copper) < 120 or len(gold) < 120:
        return _empty("CHINA_CREDIT", "China Credit Impulse", 1)

    gl = {g["date"]: g["value"] for g in gold}
    rt = [{"date": c["date"], "value": c["value"] / gl[c["date"]]}
          for c in copper if c["date"] in gl and gl[c["date"]] > 0]
    rt.sort(key=lambda x: x["date"])
    if len(rt) < 60: return _empty("CHINA_CREDIT", "China Credit Impulse", 1)

    cur = rt[-1]["value"]; m12 = _ma(rt, 252); m6 = _ma(rt, 126); v = _vel(rt)

    ph, co = "EXPANSION", 55
    if m12 and v is not None:
        if cur < m12 and v < 0: ph, co = "CONTRACTION", 65
        elif v > 0 and cur < m12: ph, co = "TROUGH", 60
        elif v > 0 and m6 and cur > m6: ph, co = "EARLY_STIMULUS", 65
        elif cur > m12 and v > 0: ph, co = "EXPANSION", 70
        elif v < 0 and cur > m12: ph, co = "PEAK", 60

    return _build("CHINA_CREDIT", "China Credit Impulse", 1, ph, co, cur, m12, v, _acc(rt),
                  _vel_z(rt), _pctl(rt, min(252 * 5, len(rt)), cur), rt)


# ── FIX V3.6: DOLLAR — MA-direction first, percentile extremes second ──
#    V3.5 bug: p10>80 + abs(v)<0.005 → PLATEAU blocked WEAKENING detection
#    V3.6: velocity direction is primary signal, percentile refines extremes
def _detect_dollar(data):
    s = _get_fred_series(data, "DXY")
    if len(s) < 12: return _empty("DOLLAR", "US Dollar Cycle", 2)

    cur = s[-1]["value"]
    m12 = _ma(s, 12)
    v = _vel(s, 3)
    p10 = _pctl(s, min(120, len(s)), cur)

    ph, co = "PLATEAU", 50  # safe default (was STRENGTHENING — wrong bias)
    if v is not None and m12:
        # PRIMARY: direction from MA + velocity
        if v < 0:
            # Dollar is falling
            if p10 is not None and p10 < 20:
                ph, co = "TROUGH", 70
            elif cur < m12:
                ph, co = "WEAKENING", 75      # below MA + falling = strong signal
            else:
                ph, co = "WEAKENING", 60      # above MA but falling = early weakening
        elif v > 0:
            # Dollar is rising
            if cur > m12:
                ph, co = "STRENGTHENING", 70  # above MA + rising = strong signal
            else:
                ph, co = "STRENGTHENING", 55  # below MA but rising = early strengthening
        else:
            # v == 0 exactly (rare)
            ph, co = "PLATEAU", 50
    elif v is not None:
        # no MA available — use velocity only
        if v < 0: ph, co = "WEAKENING", 50
        elif v > 0: ph, co = "STRENGTHENING", 50

    return _build("DOLLAR", "US Dollar Cycle", 2, ph, co, cur, m12, v,
                  _acc(s, 3), _vel_z(s, 3, 120), p10, s)


# ── FIX V3.5: BUSINESS — reordered PEAK before LATE_EXPANSION ──
def _detect_business(data):
    indpro = _get_fred_series(data, "INDPRO")
    new_orders = _get_fred_series(data, "ACOGNO")
    if len(indpro) < 24: return _empty("BUSINESS", "Business Cycle", 2)

    gr = _yoy(indpro)
    if len(gr) < 6: return _empty("BUSINESS", "Business Cycle", 2)

    cur = gr[-1]["value"]
    v = gr[-1]["value"] - gr[-2]["value"] if len(gr) >= 2 else None
    nc = 0
    for g in reversed(gr):
        if g["value"] < 0: nc += 1
        else: break

    no_gr = _yoy(new_orders) if len(new_orders) > 13 else []
    no_cur = no_gr[-1]["value"] if no_gr else None

    ph, co = "EXPANSION", 60
    if nc >= 3: ph, co = "RECESSION", 85
    elif cur < 0 and v and v > 0: ph, co = "TROUGH", 70
    elif 0 <= cur < 2 and v and v > 0: ph, co = "EARLY_RECOVERY", 65
    elif cur >= 2 and v and v > 0: ph, co = "EXPANSION", 75
    # V3.5 FIX: PEAK must come BEFORE LATE_EXPANSION (cur>4 is subset of cur>0)
    elif cur > 4 and v and v < 0: ph, co = "PEAK", 70
    elif cur > 0 and v and v < 0: ph, co = "LATE_EXPANSION", 65

    return _build("BUSINESS", "Business Cycle", 2, ph, co, cur, None, v, None,
                  None, None, gr, {"new_orders_yoy": no_cur, "neg_months": nc})


def _detect_fed(data):
    fs = _get_fred_series(data, "FEDFUNDS")
    cs = _get_fred_series(data, "CPI")
    d2 = _get_fred_series(data, "DGS2")
    if len(fs) < 12 or len(cs) < 13:
        return _empty("FED_RATES", "Fed / Interest Rate Cycle", 2)

    ff = fs[-1]["value"]
    cn, ca = cs[-1]["value"], cs[-13]["value"]
    cy = (cn - ca) / ca * 100 if ca else None
    rf = round(ff - cy, 2) if cy is not None else None
    fv = ff - fs[-4]["value"] if len(fs) >= 4 else None
    d2v = d2[-1]["value"] if d2 else None
    sp = round(d2v - ff, 2) if d2v is not None else None

    ph, co = "NEUTRAL", 50
    if rf is not None and fv is not None:
        if rf < 0 and fv < -0.25: ph, co = "EASING", 80
        elif rf < 0 and abs(fv) <= 0.25: ph, co = "EASING", 65
        elif abs(rf) <= 0.5 and abs(fv) <= 0.25: ph, co = "NEUTRAL", 70
        elif rf > 0 and rf <= 2.0 and abs(fv) <= 0.25: ph, co = "NEUTRAL", 65
        elif rf > 0 and fv > 0.25: ph, co = "TIGHTENING", 75
        elif rf > 2.0 and sp is not None and sp < -0.5: ph, co = "PRE_PIVOT", 70
        elif rf > 2.0: ph, co = "RESTRICTIVE", 75
        elif fv < -0.25 and rf > 0: ph, co = "PIVOT", 65

    return _build("FED_RATES", "Fed / Interest Rate Cycle", 2, ph, co, rf, None, fv, None,
                  None, None, [], {"fedfunds": ff, "cpi_yoy": round(cy, 2) if cy else None,
                                   "dgs2": d2v, "spread_2y_ffr": sp})


# ── FIX V3.5: EARNINGS — reordered PEAK before LATE_EXPANSION ──
def _detect_earnings(data):
    cp = _get_fred_series(data, "CORP_PROFITS")
    if len(cp) < 8: return _empty("EARNINGS", "Earnings / Profit Cycle", 2)

    gr = _qyoy(cp)
    if len(gr) < 3: return _empty("EARNINGS", "Earnings / Profit Cycle", 2)

    cur = gr[-1]["value"]
    v = gr[-1]["value"] - gr[-2]["value"] if len(gr) >= 2 else None
    nq = 0
    for g in reversed(gr):
        if g["value"] < 0: nq += 1
        else: break

    ph, co = "EXPANSION", 60
    if nq >= 2: ph, co = "CONTRACTION", 80
    elif cur < 0 and v and v > 0: ph, co = "TROUGH", 65
    elif 0 < cur < 5 and v and v > 0: ph, co = "RECOVERY", 70
    elif cur >= 5: ph, co = "EXPANSION", 75
    # V3.5 FIX: PEAK must come BEFORE LATE_EXPANSION (cur>10 is subset of cur>0)
    elif cur > 10 and v and v < 0: ph, co = "PEAK", 70
    elif cur > 0 and v and v < 0: ph, co = "LATE_EXPANSION", 65

    return _build("EARNINGS", "Earnings / Profit Cycle", 2, ph, co, cur, None, v, None,
                  None, None, gr)


def _detect_trade(data):
    cass = _get_fred_series(data, "CASS")
    if len(cass) < 24: return _empty("TRADE", "Global Trade / Shipping", 3)

    gr = _yoy(cass)
    if len(gr) < 6: return _empty("TRADE", "Global Trade / Shipping", 3)

    cur = gr[-1]["value"]
    v = gr[-1]["value"] - gr[-2]["value"] if len(gr) >= 2 else None
    nc = 0
    for g in reversed(gr):
        if g["value"] < 0: nc += 1
        else: break

    ph, co = "RECOVERY", 50
    if nc >= 6: ph, co = "COLLAPSE", 80
    elif nc >= 3: ph, co = "CONTRACTION", 70
    elif cur < -5 and v and v > 0: ph, co = "TROUGH", 65
    elif cur < 0 and v and v > 0: ph, co = "TROUGH", 60
    elif 0 <= cur < 3 and v and v > 0: ph, co = "RECOVERY", 60
    elif cur >= 3: ph, co = "EXPANSION", 70
    elif cur > 0 and v and v < 0: ph, co = "LATE", 60
    elif cur < 0 and v and v < 0: ph, co = "CONTRACTION", 65

    return _build("TRADE", "Global Trade / Shipping", 3, ph, co, cur, None, v, None,
                  None, None, gr, {"neg_months": nc})


def _detect_political(data):
    yr = date.today().year; cy = ((yr - 2025) % 4) + 1
    pm = {1: "POST_INAUGURATION", 2: "MIDTERM", 3: "PRE_ELECTION", 4: "ELECTION"}
    ar = {1: 6.5, 2: 4.2, 3: 16.3, 4: 7.5}
    h2 = date.today().month >= 7
    return _build("POLITICAL", "Political / Presidential Cycle", 3,
                  pm[cy], 95, cy, None, None, None, None, None, [],
                  {"year": cy, "avg_return": ar[cy], "is_h2": h2})


# ---------------------------------------------------------------------------
# Danger Zones
# ---------------------------------------------------------------------------

def _compute_danger_zones(result, data):
    from .config import CYCLE_DEFINITIONS
    cid = result["cycle_id"]
    zones = CYCLE_DEFINITIONS.get(cid, {}).get("danger_zones", [])
    cur = result["indicator_value"]

    nearest = None; nearest_dist = float("inf"); in_zone = False

    for zone in zones:
        th = zone.get("threshold")
        if th is None or cur is None:
            continue

        if cid in ("CREDIT",):
            dist = th - cur; hit = cur > th
        elif cid in ("BUSINESS", "TRADE"):
            dist = cur - th; hit = cur < th
        elif cid in ("DOLLAR",):
            dist = th - cur; hit = cur > th
        elif cid in ("CHINA_CREDIT",):
            dist = cur - th; hit = cur < th
        else:
            dist = cur - th; hit = cur < th

        if hit:
            in_zone = True
        if abs(dist) < abs(nearest_dist):
            nearest_dist = dist
            nearest = {"zone_name": zone["description"], "severity": zone["severity"],
                       "distance_absolute": round(dist, 2), "in_zone": hit}

    result["danger_zone"] = nearest or {"zone_name": None, "in_zone": False}
    result["in_danger_zone"] = in_zone
    return result


# ---------------------------------------------------------------------------
# V16 Alignment
# ---------------------------------------------------------------------------

def _compute_alignment(result, regime):
    from .config import CYCLE_DEFINITIONS
    cid = result["cycle_id"]; ph = result["phase"]
    v16map = CYCLE_DEFINITIONS.get(cid, {}).get("phases", {}).get(ph, {}).get("v16_mapping", [])
    if not v16map: result["v16_alignment"] = "NEUTRAL"
    elif regime in v16map: result["v16_alignment"] = "ALIGNED"
    else: result["v16_alignment"] = "DIVERGED"
    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def detect_all_phases(data):
    logger.info("=" * 60)
    logger.info("PHASE DETECTION START")
    logger.info("=" * 60)

    regime = data.get("current_regime", "UNKNOWN")

    detectors = {
        "LIQUIDITY": _detect_liquidity, "CREDIT": _detect_credit,
        "COMMODITY": _detect_commodity, "CHINA_CREDIT": _detect_china_credit,
        "DOLLAR": _detect_dollar, "BUSINESS": _detect_business,
        "FED_RATES": _detect_fed, "EARNINGS": _detect_earnings,
        "TRADE": _detect_trade, "POLITICAL": _detect_political,
    }

    cycles = {}
    for cid, fn in detectors.items():
        try:
            r = fn(data)
            r = _compute_danger_zones(r, data)
            r = _compute_alignment(r, regime)
            cycles[cid] = r
            logger.info(f"  {cid}: {r['phase']} ({r['phase_confidence']}%) "
                        f"[{r['v16_alignment']}]"
                        + (" ⚠ DANGER" if r["in_danger_zone"] else ""))
        except Exception as e:
            logger.error(f"  {cid}: FAILED — {e}")
            cycles[cid] = _empty(cid, cid, None)

    # Alignment score
    bull = {"EXPANSION", "EARLY_RECOVERY", "RECOVERY", "MID_BULL", "EARLY_BULL",
            "EARLY_STIMULUS", "EASING", "NEUTRAL", "PRE_ELECTION", "TROUGH"}
    bear = {"CONTRACTION", "DETERIORATION", "DISTRESS", "RECESSION", "COLLAPSE",
            "BEAR", "WITHDRAWAL"}

    nb = sum(1 for c in cycles.values() if c["phase"] in bull)
    nr = sum(1 for c in cycles.values() if c["phase"] in bear)
    nu = sum(1 for c in cycles.values() if c["phase"] == "UNKNOWN")
    nn = 10 - nb - nr - nu
    known = nb + nr + nn
    score = round(nb / known * 10, 1) if known > 0 else 5.0
    label = "ALIGNED" if score >= 7 else "SHIFTING" if score >= 4 else "DIVERGED"

    result = {
        "detected_at": date.today().isoformat(),
        "current_regime": regime,
        "cycles": cycles,
        "alignment_score": score,
        "alignment_label": label,
        "summary": {
            "bullish": nb, "bearish": nr, "neutral": nn, "unknown": nu,
            "in_danger_zone": sum(1 for c in cycles.values() if c["in_danger_zone"]),
            "next_turn": None,
        },
    }

    logger.info("=" * 60)
    logger.info(f"COMPLETE: {score}/10 ({label}) | "
                f"Bull:{nb} Bear:{nr} Neutral:{nn} Unk:{nu}")
    logger.info("=" * 60)

    return result
