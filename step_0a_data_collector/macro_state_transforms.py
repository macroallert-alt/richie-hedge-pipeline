"""
macro_state_transforms.py — Berechnet abgeleitete Inputs fuer MacroStateEngine
================================================================================
Wird zwischen Phase 2 (transforms) und macro_state_engine aufgerufen.

Berechnet Momentum/Acceleration-Werte die der Engine braucht:
  - MANEMP 10M Momentum (aus history cache)
  - Claims 6M Momentum (aus history cache)
  - GDPNow geglättet (aus history cache)
  - INDPRO 10M Momentum (aus history cache)
  - Cu/Au 6M Momentum (aus history cache)
  - HY OAS 6M Momentum (aus history cache)
  - YC 3M diff (aus history cache)
  - GLP data (aus Sheet oder berechnet)

Ergebnisse werden als '_xyz' Felder in transformed dict eingehängt.
"""

import logging
import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, Optional, Any

logger = logging.getLogger("data_collector.macro_transforms")


class FetchResult:
    """Minimal FetchResult fuer injizierte Werte."""
    def __init__(self, value):
        self.value = value
        self.source = 'MACRO_CALC'
        self.source_method = 'API_PRIMARY'
        self.source_date = None
        self.success = value is not None
        self.field_name = ''
        self.extra = {}


class MacroStateTransforms:
    """
    Berechnet alle abgeleiteten Inputs die der MacroStateEngine braucht.
    Nutzt den History-Cache fuer Momentum/Acceleration Berechnungen.
    """

    def __init__(self, history_cache: dict):
        self.history = history_cache

    def compute_all(self, transformed: dict, trade_date: date) -> dict:
        """
        Berechnet alle Macro-State-spezifischen Transforms.
        Haengt Ergebnisse als '_xxx' keys in transformed ein.
        Returns: dict mit GLP-Daten (fuer Engine).
        """
        logger.info("Computing Macro State V2 transforms...")

        # --- G1: MANEMP 10M Momentum ---
        manemp_mom = self._monthly_momentum('ism_mfg', 10)
        transformed['_manemp_mom_10m'] = FetchResult(manemp_mom)

        # --- G2: Claims 6M Momentum (4W-MA, dann 26W pct_change) ---
        claims_mom = self._claims_momentum()
        transformed['_claims_mom_6m'] = FetchResult(claims_mom)

        # --- G3: GDPNow geglättet (rolling 10d) ---
        gdpnow_smooth = self._gdpnow_smooth()
        transformed['_gdpnow_smooth'] = FetchResult(gdpnow_smooth)

        # --- G4: INDPRO 10M Momentum ---
        indpro_mom = self._monthly_momentum('indpro', 10)
        transformed['_indpro_mom_10m'] = FetchResult(indpro_mom)

        # --- K16: Cu/Au 6M Momentum ---
        cu_au_mom = self._daily_momentum('cu_au_ratio', 126)
        transformed['_cu_au_mom_6m'] = FetchResult(cu_au_mom)

        # --- K17: HY OAS 6M Momentum ---
        hy_mom = self._daily_momentum('hy_oas', 126)
        transformed['_hy_oas_mom_6m'] = FetchResult(hy_mom)

        # --- K5: Yield Curve 3M diff ---
        yc_diff = self._daily_diff('spread_2y10y', 63)
        transformed['_yc_mom_3m'] = FetchResult(yc_diff)

        # --- GLP Data (Trend, Momentum, Acceleration) ---
        glp_data = self._compute_glp_data()

        ok = sum(1 for k, v in transformed.items()
                 if k.startswith('_') and v.value is not None)
        logger.info(f"Macro transforms: {ok} derived values computed")

        return glp_data

    def _monthly_momentum(self, field_name: str, periods: int) -> Optional[float]:
        """
        Berechnet pct_change(periods) auf MONATLICHER Frequenz.
        History cache hat daily Werte — wir samplen monatlich.
        """
        hist = self.history.get(field_name)
        if hist is None or len(hist) < 30:
            return None

        try:
            series = hist.dropna().sort_index()
            # Resample auf monatlich (letzter bekannter Wert pro Monat)
            monthly = series.resample('ME').last().dropna()
            if len(monthly) < periods + 1:
                return None
            mom = monthly.iloc[-1] / monthly.iloc[-(periods + 1)] - 1
            return float(mom)
        except Exception as e:
            logger.warning(f"Monthly momentum for {field_name} failed: {e}")
            return None

    def _claims_momentum(self) -> Optional[float]:
        """
        Berechnet Claims 4W-MA dann 26W pct_change auf WÖCHENTLICHER Frequenz.
        """
        hist = self.history.get('initial_claims')
        if hist is None or len(hist) < 30:
            return None

        try:
            series = hist.dropna().sort_index()
            # Resample auf wöchentlich
            weekly = series.resample('W').last().dropna()
            if len(weekly) < 30:
                return None
            # 4-Wochen MA
            claims_4w = weekly.rolling(4, min_periods=2).mean()
            # 26-Wochen pct_change
            if len(claims_4w.dropna()) < 27:
                return None
            mom = claims_4w.iloc[-1] / claims_4w.iloc[-27] - 1
            return float(mom)
        except Exception as e:
            logger.warning(f"Claims momentum failed: {e}")
            return None

    def _gdpnow_smooth(self) -> Optional[float]:
        """GDPNow geglättet mit rolling(10).mean() auf daily."""
        hist = self.history.get('gdpnow')
        if hist is None or len(hist) < 5:
            return None

        try:
            series = hist.dropna().sort_index()
            smoothed = series.rolling(10, min_periods=3).mean()
            if len(smoothed.dropna()) == 0:
                return None
            return float(smoothed.dropna().iloc[-1])
        except Exception as e:
            logger.warning(f"GDPNow smooth failed: {e}")
            return None

    def _daily_momentum(self, field_name: str, lookback: int) -> Optional[float]:
        """Berechnet pct_change(lookback) auf daily Daten."""
        hist = self.history.get(field_name)
        if hist is None or len(hist) < lookback + 5:
            return None

        try:
            series = hist.dropna().sort_index()
            if len(series) < lookback + 1:
                return None
            mom = series.iloc[-1] / series.iloc[-(lookback + 1)] - 1
            return float(mom)
        except Exception as e:
            logger.warning(f"Daily momentum for {field_name} failed: {e}")
            return None

    def _daily_diff(self, field_name: str, lookback: int) -> Optional[float]:
        """Berechnet diff(lookback) auf daily Daten (absolute Veränderung)."""
        hist = self.history.get(field_name)
        if hist is None or len(hist) < lookback + 5:
            return None

        try:
            series = hist.dropna().sort_index()
            if len(series) < lookback + 1:
                return None
            diff = float(series.iloc[-1]) - float(series.iloc[-(lookback + 1)])
            return diff
        except Exception as e:
            logger.warning(f"Daily diff for {field_name} failed: {e}")
            return None

    def _compute_glp_data(self) -> dict:
        """
        Berechnet GLP Trend, Momentum und Acceleration.
        Versucht zuerst aus Geschichte zu lesen, sonst None.
        """
        # Wir brauchen die GLP-Daten idealerweise aus DATA_Liquidity Sheet.
        # Im Daily Runner haben wir diese nicht direkt — aber wir koennen
        # approximieren wenn wir net_liquidity als Proxy nehmen.
        # Fuer den vollstaendigen Backfill wird GLP aus dem Sheet gelesen.

        # Fuer den taeglichen Lauf: GLP-Data wird von main.py separat
        # aus dem Sheet gelesen und uebergeben.
        # Hier: Return leeres dict, main.py fuellt es.
        return {
            'trend': None,
            'momentum_6m': None,
            'acceleration': None,
        }
