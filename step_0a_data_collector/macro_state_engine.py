"""
macro_state_engine.py — CALC_Macro_State_V2 Engine
====================================================
Berechnet den kompletten Macro State aus Rohdaten.
Single Source of Truth: CALC_MACRO_STATE_V2_SPEC_v2.1_FINAL.md

Inputs: transformed dict (aus Phase 2) + Sheet-Daten (DATA_Liquidity, DATA_Prices)
Output: dict mit allen Spalten fuer CALC_Macro_State_V2 + DATA_K16_K17

Aufgerufen von main.py nach Phase 2 (transforms), vor Phase 4 (writers).
"""

import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("data_collector.macro_state")


# ═══════════════════════════════════════════════════════
# KONSTANTEN (aus Spec V2.1 FINAL)
# ═══════════════════════════════════════════════════════

# --- Growth Sub-Vote Thresholds ---
G1_THRESH_POS = 0.01    # MANEMP 10M Momentum
G1_THRESH_NEG = -0.01
G2_THRESH_POS = -0.05   # Claims 6M Momentum (INVERTIERT: negativ = gut)
G2_THRESH_NEG = 0.10    # Claims steigen = schlecht
G3_THRESH_POS = 2.0     # GDPNow Level (geglättet)
G3_THRESH_NEG = 0.5
G4_THRESH_POS = 0.015   # INDPRO 10M Momentum
G4_THRESH_NEG = -0.005

# --- Stress Thresholds ---
VIX_THRESH = 30.0
HY_THRESH_GROWTH_POS = 5.0   # Engerer Threshold wenn Growth=+1
HY_THRESH_DEFAULT = 7.0
NFCI_THRESH = 0.5
MOVE_THRESH = 120.0

# --- K16 Cu/Au Thresholds ---
K16_THRESH = 0.02   # symmetrisch

# --- K17 HY OAS Momentum Thresholds ---
K17_THRESH_POS = -0.10   # Spreads sinken = bullisch (INVERTIERT)
K17_THRESH_NEG = 0.10    # Spreads steigen = bearisch

# --- K4 GLP Acceleration Thresholds ---
K4_THRESH = 0.5   # symmetrisch

# --- K5 Yield Curve Thresholds ---
K5_THRESH = 0.20   # pp, symmetrisch

# --- Howell Vote Thresholds ---
HOWELL_MOM_THRESH = 0.5   # fuer Phase 3

# --- Confirmation ---
CONFIRM_DAYS = 5

# --- Stale Data Timeouts (Tage) ---
STALE_MONTHLY = 30
STALE_WEEKLY = 14
STALE_DAILY = 5

# --- Howell Cycle ---
CYCLE_LEN = 65
HOWELL_TROUGHS = [
    datetime(2006, 7, 1),
    datetime(2011, 12, 1),
    datetime(2017, 5, 1),
    datetime(2022, 10, 1),
]

# --- State Map (Growth, Liq, Stress) -> State Num ---
STATE_MAP = {
    (+1, +1, 0):  1,  (+1,  0, 0):  2,  (+1, -1, 0):  3,
    ( 0, +1, 0):  5,  ( 0,  0, 0):  6,  ( 0, -1, 0):  7,
    (-1, +1, 0):  8,  (-1,  0, 0):  9,  (-1, -1, 0): 10,
    (+1, +1, 1):  4,  (+1,  0, 1):  4,  (+1, -1, 1):  4,
    ( 0, +1, 1):  5,  ( 0,  0, 1):  6,  ( 0, -1, 1):  7,
    (-1, +1, 1):  9,  (-1,  0, 1):  9,  (-1, -1, 1): 10,
    (+1, +1, 2): 11,  (+1,  0, 2): 11,  (+1, -1, 2): 11,
    ( 0, +1, 2): 11,  ( 0,  0, 2): 11,  ( 0, -1, 2): 11,
    (-1, +1, 2): 11,  (-1,  0, 2): 11,  (-1, -1, 2): 11,
    (+1, +1, 3): 12,  (+1,  0, 3): 12,  (+1, -1, 3): 12,
    ( 0, +1, 3): 12,  ( 0,  0, 3): 12,  ( 0, -1, 3): 12,
    (-1, +1, 3): 12,  (-1,  0, 3): 12,  (-1, -1, 3): 12,
    (+1, +1, 4): 12,  (+1,  0, 4): 12,  (+1, -1, 4): 12,
    ( 0, +1, 4): 12,  ( 0,  0, 4): 12,  ( 0, -1, 4): 12,
    (-1, +1, 4): 12,  (-1,  0, 4): 12,  (-1, -1, 4): 12,
}

STATE_NAMES = {
    1: 'FULL_EXPANSION', 2: 'STEADY_GROWTH', 3: 'LATE_EXPANSION',
    4: 'FRAGILE_EXPANSION', 5: 'REFLATION', 6: 'NEUTRAL',
    7: 'SOFT_LANDING', 8: 'EARLY_RECOVERY', 9: 'CONTRACTION',
    10: 'DEEP_CONTRACTION', 11: 'STRESS_ELEVATED', 12: 'FINANCIAL_CRISIS',
}

OFFENSIVE_STATES = {1, 2, 5, 8}
DEFENSIVE_STATES = {4, 7, 10, 11, 12}
NEUTRAL_STATES = {3, 6, 9}


# ═══════════════════════════════════════════════════════
# HELPER: safe value extraction
# ═══════════════════════════════════════════════════════

def _val(transformed: dict, field_name: str) -> Optional[float]:
    """Extrahiert den Wert eines TransformedField, oder None."""
    tf = transformed.get(field_name)
    if tf is None:
        return None
    val = tf.value if hasattr(tf, 'value') else tf
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _sign(x):
    """Vorzeichen-Funktion: >0 -> +1, <0 -> -1, 0 -> 0."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


# ═══════════════════════════════════════════════════════
# GROWTH SIGNAL
# ═══════════════════════════════════════════════════════

def compute_growth_signal(
    manemp_mom_10m: Optional[float],
    claims_mom_6m: Optional[float],
    gdpnow_smooth: Optional[float],
    indpro_mom_10m: Optional[float],
) -> Tuple[int, str]:
    """
    Berechnet Growth Signal aus 4 Sub-Votes.
    Returns: (growth_signal, growth_detail)
    """
    # G1: MANEMP (ISM Proxy)
    if manemp_mom_10m is not None:
        g1 = 1 if manemp_mom_10m > G1_THRESH_POS else (-1 if manemp_mom_10m < G1_THRESH_NEG else 0)
    else:
        g1 = 0

    # G2: Initial Claims (INVERTIERT)
    if claims_mom_6m is not None:
        g2 = 1 if claims_mom_6m < G2_THRESH_POS else (-1 if claims_mom_6m > G2_THRESH_NEG else 0)
    else:
        g2 = 0

    # G3: GDPNow (geglättet)
    if gdpnow_smooth is not None:
        g3 = 1 if gdpnow_smooth > G3_THRESH_POS else (-1 if gdpnow_smooth < G3_THRESH_NEG else 0)
    else:
        g3 = 0

    # G4: Industrial Production
    if indpro_mom_10m is not None:
        g4 = 1 if indpro_mom_10m > G4_THRESH_POS else (-1 if indpro_mom_10m < G4_THRESH_NEG else 0)
    else:
        g4 = 0

    growth_sum = g1 + g2 + g3 + g4
    growth_signal = _sign(growth_sum)
    growth_detail = f"ISM:{g1}|CL:{g2}|GDP:{g3}|IP:{g4}={growth_sum}"

    return growth_signal, growth_detail


# ═══════════════════════════════════════════════════════
# STRESS SCORE
# ═══════════════════════════════════════════════════════

def compute_stress_score(
    vix: Optional[float],
    hy_oas: Optional[float],
    nfci: Optional[float],
    move: Optional[float],
    growth_signal: int,
) -> Tuple[int, str, float]:
    """
    Berechnet Stress Score (0-4).
    Returns: (stress_score, stress_detail, hy_threshold)
    """
    hy_threshold = HY_THRESH_GROWTH_POS if growth_signal == 1 else HY_THRESH_DEFAULT

    triggers = []
    if vix is not None and vix > VIX_THRESH:
        triggers.append(f"VIX={vix:.1f}>{VIX_THRESH}")
    if hy_oas is not None and hy_oas > hy_threshold:
        triggers.append(f"HY={hy_oas:.1f}>{hy_threshold}")
    if nfci is not None and nfci > NFCI_THRESH:
        triggers.append(f"NFCI={nfci:.2f}>{NFCI_THRESH}")
    if move is not None and move > MOVE_THRESH:
        triggers.append(f"MOVE={move:.1f}>{MOVE_THRESH}")

    stress_score = len(triggers)
    stress_detail = "+".join(triggers) if triggers else "NONE"

    return stress_score, stress_detail, hy_threshold


# ═══════════════════════════════════════════════════════
# LIQUIDITY DIRECTION (5-Vote System)
# ═══════════════════════════════════════════════════════

def compute_k16_vote(cu_au_mom_6m: Optional[float]) -> int:
    """K16: Cu/Au Momentum."""
    if cu_au_mom_6m is None:
        return 0
    if cu_au_mom_6m > K16_THRESH:
        return 1
    elif cu_au_mom_6m < -K16_THRESH:
        return -1
    return 0


def compute_k17_vote(hy_oas_mom_6m: Optional[float]) -> int:
    """K17: HY OAS Momentum (INVERTIERT: sinkend = bullisch)."""
    if hy_oas_mom_6m is None:
        return 0
    if hy_oas_mom_6m < K17_THRESH_POS:    # Spreads sinken stark
        return 1
    elif hy_oas_mom_6m > K17_THRESH_NEG:  # Spreads steigen stark
        return -1
    return 0


def compute_k4_vote(glp_acceleration: Optional[float]) -> int:
    """K4: GLP Acceleration."""
    if glp_acceleration is None:
        return 0
    if glp_acceleration > K4_THRESH:
        return 1
    elif glp_acceleration < -K4_THRESH:
        return -1
    return 0


def compute_howell_vote(howell_phase: Optional[int], momentum_6m: Optional[float]) -> int:
    """Howell Vote basierend auf Phase."""
    if howell_phase is None:
        return 0
    if howell_phase in (1, 2):
        return 1
    elif howell_phase == 4:
        return -1
    elif howell_phase == 3:
        if momentum_6m is not None:
            if momentum_6m > HOWELL_MOM_THRESH:
                return 1
            elif momentum_6m < -HOWELL_MOM_THRESH:
                return -1
        return 0
    return 0


def compute_k5_vote(yc_mom_3m: Optional[float]) -> int:
    """K5: Yield Curve 3M Veränderung."""
    if yc_mom_3m is None:
        return 0
    if yc_mom_3m > K5_THRESH:
        return 1
    elif yc_mom_3m < -K5_THRESH:
        return -1
    return 0


def compute_liq_direction(
    k16: int, k17: int, k4: int, howell_vote: int, k5: int,
    howell_phase: Optional[int],
    prev_confirmed: int,
    pending_value: Optional[int],
    pending_streak: int,
) -> Tuple[int, int, int, int, int, str]:
    """
    Volle Liq_Dir Pipeline: Sum → Raw → Veto → Confirm.
    Returns: (vote_sum, liq_raw, liq_final, liq_confirmed, new_streak, liq_detail)
    """
    vote_sum = k16 + k17 + k4 + howell_vote + k5
    liq_raw = _sign(vote_sum)

    # VETO_H1
    if howell_phase == 1 and liq_raw == -1:
        liq_final = 0
    else:
        liq_final = liq_raw

    # 5-Tage Bestätigung
    if liq_final != prev_confirmed:
        if pending_value is not None and liq_final == pending_value:
            new_streak = pending_streak + 1
        else:
            new_streak = 1
        new_pending = liq_final

        if new_streak >= CONFIRM_DAYS:
            liq_confirmed = liq_final
            new_streak = 0
            confirm_status = "CONF"
        else:
            liq_confirmed = prev_confirmed
            confirm_status = f"PEND({liq_final}):{new_streak}/{CONFIRM_DAYS}"
    else:
        liq_confirmed = prev_confirmed
        new_streak = 0
        new_pending = None
        confirm_status = "CONF"

    liq_detail = (
        f"K16:{k16}|HY:{k17}|GLP:{k4}|H:{howell_vote}|YC:{k5}"
        f"={vote_sum}({confirm_status})"
    )

    return vote_sum, liq_raw, liq_final, liq_confirmed, new_streak, liq_detail


# ═══════════════════════════════════════════════════════
# HOWELL PHASE
# ═══════════════════════════════════════════════════════

def compute_howell_phase(current_date: date, trend: Optional[float],
                         momentum_6m: Optional[float],
                         prev_phase: Optional[int]) -> int:
    """Berechnet Howell Phase 1-4 aus Trend + Momentum."""
    if trend is not None and momentum_6m is not None:
        if trend > 0 and momentum_6m > 0:
            return 2   # CALM
        elif trend > 0 and momentum_6m < 0:
            return 3   # SPECULATION
        elif trend < 0 and momentum_6m > 0:
            return 1   # REBOUND
        elif trend < 0 and momentum_6m < 0:
            return 4   # TURBULENCE

    # Forward-fill wenn Trend oder Momentum genau 0 oder None
    if prev_phase is not None:
        return prev_phase
    return 2  # Default CALM


# ═══════════════════════════════════════════════════════
# TRANSITION MODIFIER
# ═══════════════════════════════════════════════════════

def compute_transition(
    curr_state: int, prev_state: int, growth_signal: int
) -> Tuple[str, float]:
    """
    Berechnet Transition Direction und Modifier.
    Returns: (transition_dir, modifier)
    """
    if prev_state == curr_state:
        return "STABLE", 1.00

    if curr_state in OFFENSIVE_STATES and prev_state in DEFENSIVE_STATES:
        if growth_signal >= 0:
            return "WITH_TREND", 1.10
        return "AGAINST_TREND", 0.85

    if curr_state in DEFENSIVE_STATES and prev_state in OFFENSIVE_STATES:
        if growth_signal <= 0:
            return "WITH_TREND", 1.10
        return "AGAINST_TREND", 0.85

    return "LATERAL", 1.00


# ═══════════════════════════════════════════════════════
# VELOCITY-ADJUSTED CONFIDENCE
# ═══════════════════════════════════════════════════════

def compute_velocity_confidence(
    growth_stable: bool, liq_stable: bool, stress_stable: bool,
    vix_5d_change: Optional[float], hy_5d_change: Optional[float],
    curr_state: int,
) -> Tuple[int, int]:
    """
    Berechnet Velocity Score und State Confidence.
    Returns: (velocity_score, state_confidence)
    """
    score = 0
    if growth_stable:
        score += 1
    if liq_stable:
        score += 1
    if stress_stable:
        score += 1

    # VIX consistency
    if vix_5d_change is not None:
        if curr_state in OFFENSIVE_STATES:
            if vix_5d_change <= 0:
                score += 1
        elif curr_state in DEFENSIVE_STATES:
            if vix_5d_change >= 0:
                score += 1
        else:
            if abs(vix_5d_change) < 3:
                score += 1

    # HY consistency
    if hy_5d_change is not None:
        if curr_state in OFFENSIVE_STATES:
            if hy_5d_change <= 0:
                score += 1
        elif curr_state in DEFENSIVE_STATES:
            if hy_5d_change >= 0:
                score += 1
        else:
            if abs(hy_5d_change) < 0.2:
                score += 1

    confidence = 50 + score * 9
    return score, confidence


# ═══════════════════════════════════════════════════════
# UNCERTAINTY FLAG
# ═══════════════════════════════════════════════════════

def compute_uncertainty(
    growth_changed_10d: bool,
    liq_changed_10d: bool,
    stress_oscillated_5d: bool,
    state_confidence: int,
) -> bool:
    """Returns True wenn Unsicherheit hoch."""
    if growth_changed_10d:
        return True
    if liq_changed_10d:
        return True
    if stress_oscillated_5d:
        return True
    if state_confidence < 60:
        return True
    return False


# ═══════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════

class MacroStateEngine:
    """
    Berechnet den kompletten Macro State fuer einen einzelnen Tag.
    Haelt internen State fuer 5-Tage-Confirmation und Prev_State.
    """

    def __init__(self):
        # Persistenter State (ueberlebt zwischen Tagen)
        self.prev_confirmed_liq = 0
        self.pending_liq_value = None
        self.pending_liq_streak = 0
        self.prev_state = 6
        self.prev_howell_phase = 2
        # History fuer Velocity/Uncertainty (letzte 10 Tage)
        self.history_growth = []
        self.history_liq = []
        self.history_stress = []
        self.history_vix = []
        self.history_hy = []

    def load_prev_state(self, sheet_data: dict):
        """Laedt den letzten State aus dem Sheet (fuer Tagesstart)."""
        self.prev_confirmed_liq = sheet_data.get('liq_dir_confirmed', 0)
        self.prev_state = sheet_data.get('macro_state_num', 6)
        self.prev_howell_phase = sheet_data.get('howell_phase', 2)
        # Pending state muss ggf. aus Liq_Detail geparst werden
        # Fuer den Daily Runner reicht es wenn wir mit CONF starten

    def compute(
        self,
        trade_date: date,
        transformed: dict,
        glp_data: Optional[dict] = None,
        sheet_prev: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Berechnet alle Macro State V2 Werte fuer einen Tag.

        Args:
            trade_date: Handelstag
            transformed: Dict[field_name -> TransformedField] aus Phase 2
            glp_data: Optional dict mit GLP-Daten (trend, momentum_6m, acceleration)
            sheet_prev: Optional dict mit vorherigen Sheet-Werten

        Returns:
            (macro_state_row, k16_k17_row) — dicts mit allen Spalten
        """
        if sheet_prev:
            self.load_prev_state(sheet_prev)

        # ─── Rohdaten extrahieren ───
        vix = _val(transformed, 'vix')
        hy_oas = _val(transformed, 'hy_oas')
        nfci = _val(transformed, 'nfci')
        move = _val(transformed, 'move_index')
        spread_2y10y = _val(transformed, 'spread_2y10y')

        # Growth Inputs
        manemp_mom_10m = _val(transformed, '_manemp_mom_10m')
        claims_mom_6m = _val(transformed, '_claims_mom_6m')
        gdpnow_smooth = _val(transformed, '_gdpnow_smooth')
        indpro_mom_10m = _val(transformed, '_indpro_mom_10m')

        # K16: Cu/Au
        cu_au_ratio = _val(transformed, 'cu_au_ratio')
        cu_au_mom_6m = _val(transformed, '_cu_au_mom_6m')

        # K17: HY OAS Momentum
        hy_oas_mom_6m = _val(transformed, '_hy_oas_mom_6m')

        # K4: GLP Acceleration
        glp_acceleration = None
        glp_trend = None
        glp_momentum_6m = None
        if glp_data:
            glp_acceleration = glp_data.get('acceleration')
            glp_trend = glp_data.get('trend')
            glp_momentum_6m = glp_data.get('momentum_6m')

        # K5: Yield Curve 3M diff
        yc_mom_3m = _val(transformed, '_yc_mom_3m')

        # ─── SCHRITT 2: Howell Phase ───
        howell_phase = compute_howell_phase(
            trade_date, glp_trend, glp_momentum_6m, self.prev_howell_phase
        )

        # ─── SCHRITT 3: Growth Signal ───
        growth_signal, growth_detail = compute_growth_signal(
            manemp_mom_10m, claims_mom_6m, gdpnow_smooth, indpro_mom_10m
        )

        # ─── SCHRITT 4: Liq Votes ───
        k16 = compute_k16_vote(cu_au_mom_6m)
        k17 = compute_k17_vote(hy_oas_mom_6m)
        k4 = compute_k4_vote(glp_acceleration)
        howell_vote = compute_howell_vote(howell_phase, glp_momentum_6m)
        k5 = compute_k5_vote(yc_mom_3m)

        # ─── SCHRITT 5: Liq Dir Pipeline ───
        (vote_sum, liq_raw, liq_final, liq_confirmed,
         new_streak, liq_detail) = compute_liq_direction(
            k16, k17, k4, howell_vote, k5,
            howell_phase,
            self.prev_confirmed_liq,
            self.pending_liq_value,
            self.pending_liq_streak,
        )

        # Update internal state
        if liq_confirmed != self.prev_confirmed_liq:
            self.pending_liq_value = None
            self.pending_liq_streak = 0
        else:
            self.pending_liq_value = liq_final if liq_final != liq_confirmed else None
            self.pending_liq_streak = new_streak

        self.prev_confirmed_liq = liq_confirmed

        # ─── SCHRITT 6: Stress Score ───
        stress_score, stress_detail, hy_threshold = compute_stress_score(
            vix, hy_oas, nfci, move, growth_signal
        )

        # ─── SCHRITT 7: State Map Lookup ───
        # Clamp stress for STATE_MAP (max 4)
        stress_clamped = min(stress_score, 4)
        macro_state_num = STATE_MAP.get(
            (growth_signal, liq_confirmed, stress_clamped), 6
        )
        macro_state_name = STATE_NAMES.get(macro_state_num, 'NEUTRAL')

        # ─── SCHRITT 8: Abgeleitete Werte ───
        prev_state = self.prev_state

        # Transition
        transition_dir, transition_modifier = compute_transition(
            macro_state_num, prev_state, growth_signal
        )

        # Velocity + Confidence
        self.history_growth.append(growth_signal)
        self.history_liq.append(liq_confirmed)
        self.history_stress.append(stress_score)
        if vix is not None:
            self.history_vix.append(vix)
        if hy_oas is not None:
            self.history_hy.append(hy_oas)

        # Trim history to 10 entries
        for h in [self.history_growth, self.history_liq, self.history_stress,
                   self.history_vix, self.history_hy]:
            if len(h) > 10:
                h.pop(0)

        # Stability checks
        growth_stable = (len(self.history_growth) >= 6 and
                         self.history_growth[-1] == self.history_growth[-6]
                         if len(self.history_growth) >= 6 else True)
        liq_stable = (len(self.history_liq) >= 6 and
                      self.history_liq[-1] == self.history_liq[-6]
                      if len(self.history_liq) >= 6 else True)
        stress_stable = (len(self.history_stress) >= 6 and
                         self.history_stress[-1] == self.history_stress[-6]
                         if len(self.history_stress) >= 6 else True)

        vix_5d = None
        if len(self.history_vix) >= 6:
            vix_5d = self.history_vix[-1] - self.history_vix[-6]
        hy_5d = None
        if len(self.history_hy) >= 6:
            hy_5d = self.history_hy[-1] - self.history_hy[-6]

        velocity_score, state_confidence = compute_velocity_confidence(
            growth_stable, liq_stable, stress_stable,
            vix_5d, hy_5d, macro_state_num
        )

        # Uncertainty
        growth_changed_10d = (len(set(self.history_growth[-10:])) > 1
                              if len(self.history_growth) >= 2 else False)
        liq_changed_10d = (len(set(self.history_liq[-10:])) > 1
                           if len(self.history_liq) >= 2 else False)
        stress_oscillated_5d = (len(set(self.history_stress[-5:])) > 1
                                if len(self.history_stress) >= 2 else False)

        uncertainty = compute_uncertainty(
            growth_changed_10d, liq_changed_10d, stress_oscillated_5d,
            state_confidence
        )

        # Update prev state
        self.prev_state = macro_state_num
        self.prev_howell_phase = howell_phase

        # ─── Build Output Rows ───
        macro_state_row = {
            'Date': trade_date.strftime('%Y-%m-%d'),
            'Growth_Signal': growth_signal,
            'Growth_Detail': growth_detail,
            'Liq_Direction': liq_confirmed,
            'Liq_Detail': liq_detail,
            'Stress_Score': stress_score,
            'Stress_Detail': stress_detail,
            'HY_Threshold': hy_threshold,
            'Macro_State_Num': macro_state_num,
            'Macro_State_Name': macro_state_name,
            'State_Confidence': state_confidence,
            'Prev_State': prev_state,
            'Transition_Dir': transition_dir,
            'Transition_Modifier': transition_modifier,
            'Uncertainty_Flag': 'TRUE' if uncertainty else 'FALSE',
            'Velocity_Score': velocity_score,
            'Howell_Phase': howell_phase,
            'VIX': round(vix, 2) if vix is not None else '',
            'HY_Spread': round(hy_oas, 2) if hy_oas is not None else '',
            'NFCI': round(nfci, 3) if nfci is not None else '',
            'MOVE': round(move, 2) if move is not None else '',
        }

        k16_k17_row = {
            'Date': trade_date.strftime('%Y-%m-%d'),
            'Cu_Au_Ratio': round(cu_au_ratio, 6) if cu_au_ratio is not None else '',
            'Cu_Au_Mom6M': round(cu_au_mom_6m, 6) if cu_au_mom_6m is not None else '',
            'K16_Vote': k16,
            'HY_OAS_Mom6M': round(hy_oas_mom_6m, 6) if hy_oas_mom_6m is not None else '',
            'K17_Vote': k17,
            'GLP_Acceleration': round(glp_acceleration, 4) if glp_acceleration is not None else '',
            'K4_Vote': k4,
            'Howell_Phase': howell_phase,
            'Howell_Mom6M': round(glp_momentum_6m, 4) if glp_momentum_6m is not None else '',
            'Howell_Vote': howell_vote,
            'YC_Spread': round(spread_2y10y, 4) if spread_2y10y is not None else '',
            'YC_Mom3M': round(yc_mom_3m, 4) if yc_mom_3m is not None else '',
            'K5_Vote': k5,
            'Vote_Sum': vote_sum,
            'Liq_Dir_Raw': liq_raw,
            'Liq_Dir_Final': liq_final,
            'Liq_Dir_Confirmed': liq_confirmed,
            'Vote_Sum_Magnitude': abs(vote_sum),
        }

        logger.info(
            f"MacroState V2: State={macro_state_num} ({macro_state_name}) "
            f"Growth={growth_signal} Liq={liq_confirmed} Stress={stress_score} "
            f"Conf={state_confidence} Uncert={uncertainty}"
        )

        return macro_state_row, k16_k17_row
