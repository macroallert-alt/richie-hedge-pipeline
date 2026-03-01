"""
writers.py — Phase 4: Schreiben
================================
Output-Kanaele:
  1. V16 Google Sheet (DATA_Prices, DATA_Liquidity, DATA_K16_K17)
  2. RAW_MARKET.json (alle T1+T2+T3 Felder)
  3. RAW_MACRO.json (Macro Snapshot fuer Agent0/IC)
  4. IC_INTERFACE.json (Divergenz-Themen fuer Intelligence Cycle)
  5. DQ_SUMMARY.json (Data Quality Report)
"""

import json
import logging
import os
from datetime import datetime, date
from typing import Dict, Any, Optional

logger = logging.getLogger("data_collector.writers")


# ═══════════════════════════════════════════════════════
# V16 GOOGLE SHEET WRITER
# ═══════════════════════════════════════════════════════

class V16SheetWriter:
    """
    Schreibt Daten in die V16 Google Sheet DATA-Tabs.
    Nutzt gspread mit Service Account.
    """

    def __init__(self, sheet_id: str, credentials_path: str = None):
        self.sheet_id = sheet_id
        self.credentials_path = credentials_path or os.environ.get(
            'GOOGLE_APPLICATION_CREDENTIALS', 'credentials/gcp_service_account.json'
        )
        self._client = None
        self._workbook = None

    def _connect(self):
        """Lazy-Connect zu Google Sheets."""
        if self._client is not None:
            return
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(self.credentials_path, scopes=scopes)
        self._client = gspread.authorize(creds)
        self._workbook = self._client.open_by_key(self.sheet_id)
        logger.info(f"Connected to V16 Sheet: {self.sheet_id}")

    def write_prices(self, v16_prices: dict, trade_date: date) -> bool:
        """
        Schreibt 27 Asset-Preise in DATA_Prices Tab.
        Insert als Zeile 3 (nach Header + Beschreibungszeile).
        Deutsches Zahlenformat (Komma als Dezimal).
        """
        try:
            self._connect()
            ws = self._workbook.worksheet("DATA_Prices")

            # Check ob Datum schon existiert
            date_str = trade_date.strftime('%Y-%m-%d')
            existing = ws.col_values(1)
            if date_str in existing:
                logger.info(f"DATA_Prices: {date_str} already exists, skipping")
                return True

            # Build row: Date + 27 prices
            column_order = [
                'GLD', 'SLV', 'GDX', 'GDXJ', 'SIL',
                'SPY', 'XLY', 'XLI', 'XLF', 'XLE', 'IWM',
                'XLV', 'XLP', 'XLU', 'VNQ', 'XLK',
                'EEM', 'VGK',
                'TLT', 'TIP', 'LQD', 'HYG',
                'DBC', 'PLATINUM', 'COPPER', 'BTC', 'ETH'
            ]

            row = [date_str]
            for col_name in column_order:
                price_data = v16_prices.get(col_name)
                if price_data and price_data[0] is not None:
                    # Deutsches Format: 1.234,56
                    row.append(self._to_de_number(price_data[0]))
                else:
                    row.append('')

            # Insert at row 3 (pushes existing data down)
            ws.insert_row(row, index=3, value_input_option='RAW')
            logger.info(f"DATA_Prices: wrote {date_str}, {sum(1 for r in row[1:] if r)}/27 prices")
            return True

        except Exception as e:
            logger.error(f"DATA_Prices write failed: {e}")
            return False

    def write_liquidity(self, liq_data: dict, trade_date: date) -> bool:
        """Schreibt Liquidity-Daten in DATA_Liquidity Tab."""
        try:
            self._connect()
            ws = self._workbook.worksheet("DATA_Liquidity")

            date_str = trade_date.strftime('%Y-%m-%d')
            existing = ws.col_values(1)
            if date_str in existing:
                logger.info(f"DATA_Liquidity: {date_str} already exists, skipping")
                return True

            row = [
                date_str,
                self._to_de_number(liq_data.get('Fed_Net_Liq')),
                self._to_de_number(liq_data.get('ECB_USD')),
                self._to_de_number(liq_data.get('BOJ_USD')),
                self._to_de_number(liq_data.get('China_M2_USD')),
                self._to_de_number(liq_data.get('US_M2')),
            ]

            ws.insert_row(row, index=3, value_input_option='RAW')
            logger.info(f"DATA_Liquidity: wrote {date_str}")
            return True

        except Exception as e:
            logger.error(f"DATA_Liquidity write failed: {e}")
            return False

    def write_k16(self, transformed: dict, trade_date: date) -> bool:
        """Schreibt Cu/Au, Credit Impulse, GLI in DATA_K16_K17."""
        try:
            self._connect()
            ws = self._workbook.worksheet("DATA_K16_K17")

            date_str = trade_date.strftime('%Y-%m-%d')
            existing = ws.col_values(1)
            if date_str in existing:
                logger.info(f"DATA_K16_K17: {date_str} already exists, skipping")
                return True

            cu_au = transformed.get('cu_au_ratio')
            cu_au_val = cu_au.value if cu_au else None

            # Sparse row: nur A, B befuellen, rest leer (Sheet-Formeln)
            row = [date_str, self._to_de_number(cu_au_val)]
            # Pad to column E (Credit Impulse) and G (GLI)
            row += ['', '', '', '', '']  # C, D, E, F, G — Sheet-Formeln

            ws.insert_row(row, index=3, value_input_option='RAW')
            logger.info(f"DATA_K16_K17: wrote {date_str}")
            return True

        except Exception as e:
            logger.error(f"DATA_K16_K17 write failed: {e}")
            return False

    def write_all(self, v16_prices: dict, liq_data: dict, transformed: dict,
                  trade_date: date) -> dict:
        """Schreibt alle 3 DATA-Tabs. Returns Status-Dict."""
        status = {
            "DATA_Prices": self.write_prices(v16_prices, trade_date),
            "DATA_Liquidity": self.write_liquidity(liq_data, trade_date),
            "DATA_K16_K17": self.write_k16(transformed, trade_date),
        }
        ok = sum(1 for v in status.values() if v)
        logger.info(f"V16 Sheet: {ok}/3 tabs updated")
        return status

    @staticmethod
    def _to_de_number(value) -> str:
        """Konvertiert float zu deutschem Zahlenformat."""
        if value is None:
            return ''
        if isinstance(value, bool):
            return str(int(value))
        try:
            v = float(value)
            if abs(v) >= 1000:
                # 1234567.89 -> 1.234.567,89
                formatted = f"{v:,.2f}"
                formatted = formatted.replace(',', 'TEMP').replace('.', ',').replace('TEMP', '.')
            else:
                formatted = f"{v:.6g}".replace('.', ',')
            return formatted
        except (ValueError, TypeError):
            return str(value)


# ═══════════════════════════════════════════════════════
# JSON FILE WRITERS
# ═══════════════════════════════════════════════════════

class JSONWriter:
    """Schreibt alle JSON-Outputs."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_raw_market(self, transformed: dict, dq_summary: dict) -> str:
        """RAW_MARKET.json — Alle Felder mit vollen Transformationen."""
        output = {
            "_meta": {
                "version": "2.0",
                "generated_at": datetime.now().isoformat(),
                "data_quality_level": dq_summary.get("data_quality_level", "UNKNOWN"),
                "fields_ok": dq_summary.get("fields_ok", 0),
                "fields_total": dq_summary.get("fields_total", 0),
            },
            "fields": {}
        }

        for field_name, tf in sorted(transformed.items()):
            output["fields"][field_name] = tf.to_dict()

        path = os.path.join(self.output_dir, "RAW_MARKET.json")
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Wrote RAW_MARKET.json ({len(output['fields'])} fields)")
        return path

    def write_raw_macro(self, transformed: dict) -> str:
        """RAW_MACRO.json — Macro Snapshot fuer Agent0/IC."""
        # Group by IC divergence theme
        themes = {}
        for field_name, tf in transformed.items():
            # Get IC theme from registry (via category mapping)
            theme = _get_ic_theme(field_name)
            if theme:
                if theme not in themes:
                    themes[theme] = []
                themes[theme].append({
                    "field": field_name,
                    "value": tf.value,
                    "direction": tf.direction,
                    "pctl_1y": tf.pctl_1y,
                    "zscore_2y": getattr(tf, 'zscore_2y', None),
                    "confidence": tf.confidence,
                    "anomaly": tf.anomaly_flag,
                })

        output = {
            "_meta": {
                "version": "2.0",
                "generated_at": datetime.now().isoformat(),
            },
            "macro_snapshot": {
                "regime_indicators": {
                    "vix": _extract_val(transformed, 'vix'),
                    "hy_oas": _extract_val(transformed, 'hy_oas'),
                    "nfci": _extract_val(transformed, 'nfci'),
                    "net_liquidity": _extract_val(transformed, 'net_liquidity'),
                },
                "directional_signals": {
                    "dxy": _extract_direction(transformed, 'dxy'),
                    "spread_2y10y": _extract_direction(transformed, 'spread_2y10y'),
                    "cu_au_ratio": _extract_direction(transformed, 'cu_au_ratio'),
                    "fedwatch_cut_prob": _extract_direction(transformed, 'fedwatch_cut_prob'),
                },
                "stress_flags": self._compute_stress_flags(transformed),
            },
            "ic_themes": themes,
        }

        path = os.path.join(self.output_dir, "RAW_MACRO.json")
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Wrote RAW_MACRO.json ({len(themes)} themes)")
        return path

    def write_ic_interface(self, transformed: dict) -> str:
        """IC_INTERFACE.json — Divergenz-Themen fuer Intelligence Cycle."""
        divergences = []

        # Check each IC theme for internal divergence
        theme_fields = _get_all_theme_fields()
        for theme, fields in theme_fields.items():
            directions = []
            for fn in fields:
                tf = transformed.get(fn)
                if tf and tf.direction:
                    directions.append(tf.direction)

            if len(directions) >= 2:
                up = directions.count("UP")
                down = directions.count("DOWN")
                if up > 0 and down > 0:
                    divergences.append({
                        "theme": theme,
                        "fields": fields,
                        "directions": {fn: transformed[fn].direction
                                       for fn in fields if fn in transformed and transformed[fn].direction},
                        "severity": "HIGH" if abs(up - down) <= 1 else "LOW",
                    })

        output = {
            "_meta": {
                "version": "2.0",
                "generated_at": datetime.now().isoformat(),
            },
            "divergences": divergences,
            "extreme_readings": self._find_extremes(transformed),
        }

        path = os.path.join(self.output_dir, "IC_INTERFACE.json")
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Wrote IC_INTERFACE.json ({len(divergences)} divergences)")
        return path

    def write_dq_summary(self, dq: dict) -> str:
        """DQ_SUMMARY.json."""
        path = os.path.join(self.output_dir, "DQ_SUMMARY.json")
        with open(path, 'w') as f:
            json.dump(dq, f, indent=2, default=str)
        logger.info(f"Wrote DQ_SUMMARY.json (level={dq.get('data_quality_level')})")
        return path

    def write_all(self, transformed: dict, dq_summary: dict) -> dict:
        """Schreibt alle JSON-Outputs. Returns Pfade."""
        return {
            "RAW_MARKET": self.write_raw_market(transformed, dq_summary),
            "RAW_MACRO": self.write_raw_macro(transformed),
            "IC_INTERFACE": self.write_ic_interface(transformed),
            "DQ_SUMMARY": self.write_dq_summary(dq_summary),
        }

    def _compute_stress_flags(self, transformed: dict) -> dict:
        flags = {}
        vix = transformed.get('vix')
        if vix and vix.value is not None:
            flags["vix_elevated"] = vix.value > 25
            flags["vix_extreme"] = vix.value > 35

        hy = transformed.get('hy_oas')
        if hy and hy.value is not None:
            flags["hy_stress"] = hy.value > 500
            flags["hy_crisis"] = hy.value > 800

        vtr = transformed.get('vix_term_ratio')
        if vtr and vtr.value is not None:
            flags["backwardation"] = vtr.value < 0.9

        nfci = transformed.get('nfci')
        if nfci and nfci.value is not None:
            flags["nfci_tightening"] = nfci.value > 0
        return flags

    def _find_extremes(self, transformed: dict) -> list:
        extremes = []
        for fn, tf in transformed.items():
            if tf.pctl_1y is not None:
                if tf.pctl_1y >= 95 or tf.pctl_1y <= 5:
                    extremes.append({
                        "field": fn, "value": tf.value,
                        "pctl_1y": tf.pctl_1y, "direction": tf.direction,
                        "extreme_type": "HIGH" if tf.pctl_1y >= 95 else "LOW",
                    })
        return sorted(extremes, key=lambda x: abs(x['pctl_1y'] - 50), reverse=True)


# ═══════════════════════════════════════════════════════
# IC THEME MAPPING
# ═══════════════════════════════════════════════════════

IC_THEME_MAP = {
    'net_liquidity': 'LIQUIDITY', 'walcl': 'LIQUIDITY', 'tga': 'LIQUIDITY', 'rrp': 'LIQUIDITY',
    'mmf_assets': 'LIQUIDITY',
    'fedwatch_cut_prob': 'FED_POLICY', 'fed_funds_rate': 'FED_POLICY', 'sofr_ff_spread': 'FED_POLICY',
    'hy_oas': 'CREDIT', 'ig_oas': 'CREDIT', 'hyg_tlt_ratio': 'CREDIT',
    'initial_claims': 'RECESSION', 'gdpnow': 'RECESSION', 'ism_mfg': 'RECESSION',
    'breakeven_5y5y': 'INFLATION', 'cpi_yoy': 'INFLATION',
    'vix': 'VOLATILITY', 'vix_term_ratio': 'VOLATILITY', 'move_index': 'VOLATILITY',
    'vix_call_put_vol': 'VOLATILITY', 'iv_rv_spread': 'VOLATILITY',
    'pc_ratio_equity': 'POSITIONING', 'naaim_exposure': 'POSITIONING',
    'aaii_bull_bear': 'POSITIONING', 'cot_es_leveraged': 'POSITIONING', 'cot_zn_leveraged': 'POSITIONING',
    'dxy': 'DOLLAR', 'usdjpy': 'DOLLAR',
    'china_10y': 'CHINA_EM', 'usdcnh': 'CHINA_EM',
    'wti_curve': 'ENERGY', 'brent_wti_spread': 'ENERGY',
    'cu_au_ratio': 'COMMODITIES',
}

def _get_ic_theme(field_name: str) -> Optional[str]:
    return IC_THEME_MAP.get(field_name)

def _get_all_theme_fields() -> Dict[str, list]:
    themes = {}
    for fn, theme in IC_THEME_MAP.items():
        themes.setdefault(theme, []).append(fn)
    return themes

def _extract_val(transformed, fn):
    tf = transformed.get(fn)
    if tf and tf.value is not None:
        return {"value": tf.value, "pctl_1y": tf.pctl_1y, "confidence": tf.confidence}
    return None

def _extract_direction(transformed, fn):
    tf = transformed.get(fn)
    if tf:
        return {"value": tf.value, "direction": tf.direction, "delta_5d": tf.delta_5d}
    return None
