"""
Ratio-Kontext-Modul — Baldur Creek Capital
Step 0x Pre-Step | V1.1

Berechnet für 30 Ratio-Paare aus V16 Sheet DATA_Prices:
- Z-Score (Full + Rolling 5J) auf LOG-Ratios
- Perzentil
- Momentum (Z-Score Veränderung letzte 63 Tage)
- Headroom (% der historischen Range zum Mean)
- ADF Stationaritätstest
- Halflife (Ornstein-Uhlenbeck, nur wenn stationär)

Output: step_0x_theses/data/ratio_context.json
Gelesen von: theses_agent.py (Step 3b Prompt)

Usage:
  python -m step_0x_theses.ratio_context [--skip-sheet]
"""

import json
import logging
import os
import warnings
import numpy as np
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ratio_context")

from .config import (
    DATA_DIR, DW_SHEET_ID, DW_PRICES_TAB, V16_ETF_MAP, RATIO_PAIRS,
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

OUTPUT_FILE = os.path.join(DATA_DIR, "ratio_context.json")

IS_START = "2007-01-01"
IS_END   = "2020-12-31"
OOS_START = "2021-01-01"

ROLLING_YEARS = 5
ROLLING_WINDOW_DAYS = ROLLING_YEARS * 252

MOMENTUM_WINDOW = 63  # ~3 Monate

ADF_STATIONARY_P = 0.05
ADF_TRENDING_P   = 0.10

Z_EXTREME = 2.0
Z_ELEVATED = 1.5


# ═══════════════════════════════════════════════════════════════
# GCP CREDENTIALS
# ═══════════════════════════════════════════════════════════════

def _get_gcp_credentials():
    """GCP Credentials aus Environment."""
    sa_key_json = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
    if not sa_key_json:
        logger.warning("Kein GCP_SA_KEY/GOOGLE_CREDENTIALS im Environment")
        return None
    try:
        from google.oauth2.service_account import Credentials
        sa_info = json.loads(sa_key_json)
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        return Credentials.from_service_account_info(sa_info, scopes=scopes)
    except Exception as e:
        logger.warning(f"GCP Credentials Fehler: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# SHEET LESEN
# ═══════════════════════════════════════════════════════════════

def _parse_european_number(val):
    """'2.123,80' → 2123.80"""
    val = val.strip()
    if not val or val == "-":
        return None
    try:
        if "," in val:
            val_clean = val.replace(".", "").replace(",", ".")
        else:
            val_clean = val
        price = float(val_clean)
        return price if price > 0 else None
    except ValueError:
        return None


def _parse_date(val):
    """Parsed Datum aus Sheet."""
    val = val.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def read_all_prices():
    """Liest ALLE Zeilen aus DATA_Prices.
    Returns: (dates[], prices{ticker: [float|None]}) — chronologisch."""
    import gspread

    creds = _get_gcp_credentials()
    if not creds:
        return None, None

    logger.info(f"Sheet öffnen: {DW_SHEET_ID}, Tab '{DW_PRICES_TAB}'...")
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(DW_SHEET_ID)
    ws = sheet.worksheet(DW_PRICES_TAB)

    logger.info("Lade alle Werte...")
    all_values = ws.get_all_values()
    logger.info(f"{len(all_values)} Zeilen geladen")

    if len(all_values) < 3:
        logger.error(f"Sheet hat nur {len(all_values)} Zeilen")
        return None, None

    headers = all_values[0]
    ticker_cols = {}
    for i, h in enumerate(headers):
        ticker = h.strip().upper()
        if ticker in V16_ETF_MAP:
            ticker_cols[ticker] = i

    logger.info(f"{len(ticker_cols)} Ticker gefunden")

    dates = []
    prices = {t: [] for t in ticker_cols}
    skipped = 0

    for row in all_values[2:]:
        if not row or len(row) < 2:
            skipped += 1
            continue
        dt = _parse_date(row[0])
        if dt is None:
            skipped += 1
            continue
        dates.append(dt)
        for ticker, col_idx in ticker_cols.items():
            if col_idx < len(row):
                prices[ticker].append(_parse_european_number(row[col_idx]))
            else:
                prices[ticker].append(None)

    logger.info(f"{len(dates)} Datenzeilen ({skipped} übersprungen)")

    # Absteigend → chronologisch
    if len(dates) > 1 and dates[0] > dates[-1]:
        dates = dates[::-1]
        for t in prices:
            prices[t] = prices[t][::-1]

    logger.info(f"Zeitraum: {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
    return dates, prices


# ═══════════════════════════════════════════════════════════════
# RATIO-BERECHNUNG (LOG-RATIOS)
# ═══════════════════════════════════════════════════════════════

def compute_log_ratio_series(dates, prices, num_ticker, den_ticker):
    """Log-Ratio ln(A/B) Zeitreihe."""
    num_p = prices.get(num_ticker, [])
    den_p = prices.get(den_ticker, [])
    if not num_p or not den_p:
        return None, None

    valid_dates = []
    log_ratios = []

    for i, dt in enumerate(dates):
        if i >= len(num_p) or i >= len(den_p):
            break
        n, d = num_p[i], den_p[i]
        if n is not None and d is not None and n > 0 and d > 0:
            valid_dates.append(dt)
            log_ratios.append(np.log(n / d))

    if len(log_ratios) < 100:
        return None, None

    return valid_dates, np.array(log_ratios)


# ═══════════════════════════════════════════════════════════════
# STATISTIK
# ═══════════════════════════════════════════════════════════════

def _zscore(values, current):
    if len(values) < 10:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    return float((current - mean) / std) if std > 0 else 0.0


def _percentile(values, current):
    if len(values) < 10:
        return 50.0
    return float(np.sum(values <= current) / len(values) * 100)


def _headroom(values, current):
    """% der Range Richtung Mean. 0%=am Extrem, 100%=am Mean."""
    hist_min, hist_max = np.min(values), np.max(values)
    hist_range = hist_max - hist_min
    if hist_range <= 0:
        return 50.0
    mean = np.mean(values)
    if current >= mean:
        return float((hist_max - current) / (hist_max - mean) * 100) if hist_max != mean else 0.0
    else:
        return float((current - hist_min) / (mean - hist_min) * 100) if mean != hist_min else 0.0


def _momentum(log_ratios, window=MOMENTUM_WINDOW):
    """Z-Score der 3M-Veränderung vs. historische Veränderungen."""
    if len(log_ratios) < window + 50:
        return 0.0
    changes = log_ratios[window:] - log_ratios[:-window]
    current_change = changes[-1]
    mean_change = np.mean(changes[:-1])
    std_change = np.std(changes[:-1])
    return float((current_change - mean_change) / std_change) if std_change > 0 else 0.0


def _adf_test(values):
    """ADF Test. Returns p-value oder None."""
    from statsmodels.tsa.stattools import adfuller
    if len(values) < 100:
        return None
    try:
        result = adfuller(values, maxlag=20, autolag="AIC")
        return float(result[1])
    except Exception:
        return None


def _halflife(values):
    """Ornstein-Uhlenbeck Halflife in Tagen."""
    if len(values) < 100:
        return None
    try:
        y = values[1:] - values[:-1]
        x = values[:-1]
        x_dm = x - np.mean(x)
        y_dm = y - np.mean(y)
        beta = np.sum(x_dm * y_dm) / np.sum(x_dm * x_dm)
        if beta >= 0:
            return None
        hl = -np.log(2) / np.log(1 + beta)
        if hl < 1 or hl > 2520:
            return None
        return round(float(hl), 0)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# PRO-PAAR ANALYSE
# ═══════════════════════════════════════════════════════════════

def analyze_pair(dates, log_ratios):
    """Vollständige Analyse für ein Ratio-Paar."""
    is_start_dt = datetime.strptime(IS_START, "%Y-%m-%d")
    is_end_dt = datetime.strptime(IS_END, "%Y-%m-%d")

    current = float(log_ratios[-1])
    n_total = len(log_ratios)

    is_mask = np.array([(is_start_dt <= d <= is_end_dt) for d in dates])
    oos_mask = np.array([(d > is_end_dt) for d in dates])
    is_values = log_ratios[is_mask]
    oos_values = log_ratios[oos_mask]

    z_full = _zscore(log_ratios, current)
    pctl_full = _percentile(log_ratios, current)
    headroom = _headroom(log_ratios, current)

    rolling = log_ratios[-ROLLING_WINDOW_DAYS:] if n_total > ROLLING_WINDOW_DAYS else log_ratios
    z_5y = _zscore(rolling, current)

    momentum = _momentum(log_ratios)

    adf_full = _adf_test(log_ratios)
    adf_is = _adf_test(is_values) if len(is_values) >= 100 else None
    adf_oos = _adf_test(oos_values) if len(oos_values) >= 100 else None

    halflife = None
    if adf_full is not None and adf_full < ADF_STATIONARY_P:
        halflife = _halflife(log_ratios)

    # Regime (für Frontend)
    if adf_full is not None and adf_full < ADF_STATIONARY_P:
        regime = "REGIME_BREAK" if (adf_oos is not None and adf_oos > ADF_TRENDING_P) else "STATIONARY"
    elif adf_full is not None and adf_full > ADF_TRENDING_P:
        regime = "TRENDING"
    else:
        regime = "UNCLEAR"

    # Signal (für Frontend)
    if abs(z_full) >= Z_EXTREME:
        signal = "EXTREM_TEUER" if z_full > 0 else "EXTREM_GÜNSTIG"
    elif abs(z_full) >= Z_ELEVATED:
        signal = "ERHÖHT" if z_full > 0 else "GEDRÜCKT"
    else:
        signal = "NEUTRAL"

    # IS-Parameter
    is_params = None
    if len(is_values) >= 100:
        is_mean, is_std = float(np.mean(is_values)), float(np.std(is_values))
        if is_std > 0:
            is_params = {
                "z_score_is_params": round(float((current - is_mean) / is_std), 3),
                "percentile_is_params": round(float(np.sum(is_values <= current) / len(is_values) * 100), 1),
            }

    return {
        "z_full": round(z_full, 2),
        "z_5y": round(z_5y, 2),
        "pctl": round(pctl_full, 1),
        "momentum_3m": round(momentum, 2),
        "headroom": round(headroom, 0),
        "adf_p": round(adf_full, 4) if adf_full is not None else None,
        "halflife": int(halflife) if halflife is not None else None,
        "signal": signal,
        "regime": regime,
        "n_observations": n_total,
        "date_range": f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
        "adf_is": round(adf_is, 4) if adf_is is not None else None,
        "adf_oos": round(adf_oos, 4) if adf_oos is not None else None,
        "is_params": is_params,
    }


# ═══════════════════════════════════════════════════════════════
# LLM-FORMAT (gruppiert nach Ziel-Asset)
# ═══════════════════════════════════════════════════════════════

def format_ratios_for_llm(ratio_results):
    """Gruppiertes Format für den Thesen-Agent Prompt.
    Gruppiert nach Numerator (Ziel-Asset). Nur Fakten."""

    # Gruppiere nach Numerator
    groups = {}
    for r in ratio_results:
        num = r["numerator"]
        if num not in groups:
            groups[num] = []
        groups[num].append(r)

    lines = [
        "=== RELATIVE-VALUE KONTEXT (Log-Ratio Z-Scores, 30 Paare) ===",
        "Z=Position (+ teuer, - günstig). Mom=Momentum 3M. Raum=% zum Mean.",
        ""
    ]

    for num_ticker in sorted(groups.keys()):
        entries = groups[num_ticker]
        num_name = V16_ETF_MAP.get(num_ticker, num_ticker)
        lines.append(f"{num_ticker} ({num_name}) gegen:")

        for r in entries:
            a = r["analysis"]
            den = r["denominator"]

            if a["halflife"] is not None:
                stat_str = f"Stationär HL={a['halflife']}d"
            elif a["adf_p"] is not None and a["adf_p"] > ADF_TRENDING_P:
                stat_str = "Trend"
            else:
                stat_str = "Unklar"

            lines.append(
                f"  vs {den:8s} Z={a['z_full']:+5.2f} 5J={a['z_5y']:+5.2f} "
                f"P={a['pctl']:5.1f}% Mom={a['momentum_3m']:+5.2f} "
                f"Raum={a['headroom']:3.0f}% | {stat_str}"
            )

        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    """Hauptfunktion. Returns output dict oder None."""
    logger.info("=" * 60)
    logger.info("RATIO-KONTEXT-MODUL V1.1")
    logger.info("=" * 60)

    dates, prices = read_all_prices()
    if dates is None or prices is None:
        logger.error("Sheet-Read fehlgeschlagen — kein Ratio-Kontext")
        return None

    ratio_results = []
    skipped = []

    for num, den, description in RATIO_PAIRS:
        valid_dates, log_ratios = compute_log_ratio_series(dates, prices, num, den)
        if log_ratios is None:
            skipped.append(f"{num}/{den}")
            continue

        analysis = analyze_pair(valid_dates, log_ratios)

        ratio_results.append({
            "pair": f"{num}/{den}",
            "numerator": num,
            "denominator": den,
            "description": description,
            "numerator_name": V16_ETF_MAP.get(num, num),
            "denominator_name": V16_ETF_MAP.get(den, den),
            "analysis": analysis,
        })

        a = analysis
        hl_str = f"HL={a['halflife']}d" if a['halflife'] else "—"
        logger.info(f"  {num}/{den:8s} Z={a['z_full']:+5.2f} P={a['pctl']:5.1f}% "
                     f"Raum={a['headroom']:3.0f}% {hl_str} → {a['signal']}")

    if skipped:
        logger.warning(f"Übersprungen: {', '.join(skipped)}")

    llm_text = format_ratios_for_llm(ratio_results)

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "module_version": "1.1",
            "method": "Log-Ratios, ADF, Ornstein-Uhlenbeck Halflife",
            "total_rows": len(dates),
            "date_range": f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
            "is_period": f"{IS_START} → {IS_END}",
            "oos_period": f"{OOS_START} → {dates[-1].strftime('%Y-%m-%d')}",
            "pairs_computed": len(ratio_results),
            "pairs_skipped": skipped,
        },
        "ratios": ratio_results,
        "llm_prompt_text": llm_text,
    }

    # JSON schreiben
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    logger.info(f"Geschrieben: {OUTPUT_FILE} ({size_kb:.1f} KB)")
    logger.info(f"{len(ratio_results)} Paare berechnet")

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ratio-Kontext-Modul V1.1")
    parser.add_argument("--skip-sheet", action="store_true", help="Sheet-Read überspringen")
    args = parser.parse_args()

    if args.skip_sheet:
        logger.info("--skip-sheet: Übersprungen")
        return

    run()


if __name__ == "__main__":
    main()
