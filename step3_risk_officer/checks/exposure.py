"""
step_3_risk_officer/checks/exposure.py
Exposure Checks — Portfolio-Zusammensetzung pruefen.
Spec: Risk Officer Teil 2 §10 (EXP_SECTOR_CONCENTRATION, EXP_GEOGRAPHY,
      EXP_SINGLE_NAME, EXP_ASSET_CLASS)
"""

from ..utils.helpers import make_alert
from ..utils.mappings import get_sector_breakdown, get_asset_class, is_international_asset


# ═══════════════════════════════════════════════════════════════════
# EXP_SECTOR_CONCENTRATION — Effektive Sektor-Exposure (V16 + F6)
# Spec Teil 2 §10.1
# ═══════════════════════════════════════════════════════════════════

def check_sector_concentration(v16_weights, f6_positions, config):
    """
    Berechnet effektive Sektor-Exposure durch Kombination von
    V16 ETF-Gewichten und F6 Einzelaktien-Gewichten.

    Returns: (sector_exposure_dict, list_of_alerts)
    """
    sector_exposure = {}

    # V16 Beitrag: ETF-Gewichte nach Sektor aufteilen
    for asset, weight in v16_weights.items():
        breakdown = get_sector_breakdown(asset)
        for sector, pct in breakdown.items():
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight * pct

    # F6 Beitrag: Einzelaktien nach Sektor zuordnen
    for position in (f6_positions or []):
        sector = position.get("sector", "Other")
        weight = position.get("current_weight", 0.0)
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

    # Pruefung gegen Schwellen
    alerts = []
    overrides = config.get("overrides", {})

    for sector, exposure in sector_exposure.items():
        # Sektor-spezifische Overrides (z.B. Tech strenger)
        if sector in overrides:
            cfg = overrides[sector]
        else:
            cfg = config

        warning_band = cfg.get("warning_band", 0.35)
        max_threshold = cfg.get("max", 0.40)
        critical_threshold = cfg.get("critical", 0.50)

        if exposure >= critical_threshold:
            alerts.append(make_alert(
                severity="CRITICAL",
                message=(
                    f"Effective {sector} Exposure {exposure:.1%} exceeds "
                    f"CRITICAL threshold ({critical_threshold:.0%}). "
                    f"Review combined V16+F6 {sector} allocation."
                ),
                check_id="EXP_SECTOR_CONCENTRATION",
                affected_systems=["V16", "F6"] if f6_positions else ["V16"],
                trade_class="B" if f6_positions else "A",
                current_value=round(exposure, 4),
                threshold=critical_threshold,
                recommendation=(
                    f"Immediate review recommended. {sector} exposure at {exposure:.1%} "
                    f"is significantly above limit. Discuss with Agent R whether "
                    f"positions in {sector} should be reduced or hedged."
                )
            ))
        elif exposure >= max_threshold:
            alerts.append(make_alert(
                severity="WARNING",
                message=(
                    f"Effective {sector} Exposure {exposure:.1%} exceeds "
                    f"MAX threshold ({max_threshold:.0%})."
                ),
                check_id="EXP_SECTOR_CONCENTRATION",
                affected_systems=["V16", "F6"] if f6_positions else ["V16"],
                trade_class="B" if f6_positions else "A",
                current_value=round(exposure, 4),
                threshold=max_threshold,
                recommendation=(
                    f"CIO should review combined {sector} exposure across V16 and F6."
                )
            ))
        elif exposure >= warning_band:
            alerts.append(make_alert(
                severity="MONITOR",
                message=(
                    f"Effective {sector} Exposure {exposure:.1%} approaching "
                    f"warning level ({warning_band:.0%})."
                ),
                check_id="EXP_SECTOR_CONCENTRATION",
                current_value=round(exposure, 4),
                threshold=warning_band,
                recommendation="No action required. Monitor for further increases."
            ))

    return sector_exposure, alerts


# ═══════════════════════════════════════════════════════════════════
# EXP_GEOGRAPHY — International Exposure Hard Cap
# Spec Teil 2 §10.2
# ═══════════════════════════════════════════════════════════════════

def check_geography(v16_weights, f6_positions, config):
    """
    Berechnet gesamte International-Exposure aus V16 + F6.
    Prueft gegen 25% Hard Cap (Stage 1).

    Returns: (alert_or_None, breakdown_dict)
    """
    international_exposure = 0.0
    breakdown = {}

    # V16 International-Assets
    for asset, weight in v16_weights.items():
        if is_international_asset(asset):
            international_exposure += weight
            breakdown[f"V16_{asset}"] = weight

    # F6 International (selten, aber pruefen)
    for position in (f6_positions or []):
        ticker = position.get("ticker", "")
        if is_international_asset(ticker):
            weight = position.get("current_weight", 0.0)
            international_exposure += weight
            breakdown[f"F6_{ticker}"] = weight

    # Pruefung
    international_max = config.get("international_max", 0.25)
    warning_band = config.get("international_warning_band", 0.20)
    critical = config.get("international_critical", 0.30)

    alert = None

    if international_exposure >= critical:
        alert = make_alert(
            severity="CRITICAL",
            message=(
                f"International exposure {international_exposure:.1%} significantly "
                f"exceeds Hard Cap ({international_max:.0%}) by "
                f"{(international_exposure - international_max):.1%}pp."
            ),
            check_id="EXP_GEOGRAPHY",
            current_value=round(international_exposure, 4),
            threshold=international_max,
            recommendation=(
                f"International exposure {international_exposure:.1%} significantly "
                f"above Hard Cap. Immediate Agent R consultation for position reduction."
            )
        )
    elif international_exposure >= international_max:
        alert = make_alert(
            severity="WARNING",
            message=(
                f"International exposure {international_exposure:.1%} exceeds "
                f"{international_max:.0%} Hard Cap."
            ),
            check_id="EXP_GEOGRAPHY",
            current_value=round(international_exposure, 4),
            threshold=international_max,
            recommendation=(
                f"Review with Agent R which international positions to reduce."
            )
        )
    elif international_exposure >= warning_band:
        alert = make_alert(
            severity="MONITOR",
            message=(
                f"International exposure {international_exposure:.1%}, "
                f"approaching {international_max:.0%} cap."
            ),
            check_id="EXP_GEOGRAPHY",
            current_value=round(international_exposure, 4),
            threshold=international_max,
            recommendation="International exposure approaching cap. Monitor."
        )

    return alert, breakdown


# ═══════════════════════════════════════════════════════════════════
# EXP_SINGLE_NAME — Einzelpositions-Limit
# Spec Teil 2 §10.3
# ═══════════════════════════════════════════════════════════════════

def check_single_name(v16_weights, f6_positions, config):
    """
    Prueft ob eine einzelne Position das Limit ueberschreitet.

    Returns: list_of_alerts
    """
    all_positions = {}

    # V16 Assets
    for asset, weight in v16_weights.items():
        all_positions[asset] = {"weight": weight, "source": "V16"}

    # F6 Positionen
    for position in (f6_positions or []):
        ticker = position.get("ticker", "UNKNOWN")
        all_positions[ticker] = {
            "weight": position.get("current_weight", 0.0),
            "source": "F6"
        }

    warning_band = config.get("warning_band", 0.20)
    max_threshold = config.get("max", 0.25)
    critical_threshold = config.get("critical", 0.30)

    alerts = []
    for name, data in all_positions.items():
        w = data["weight"]
        src = data["source"]

        if w >= critical_threshold:
            alerts.append(make_alert(
                severity="CRITICAL",
                message=f"Single position {name} ({src}) at {w:.1%} exceeds {critical_threshold:.0%}.",
                check_id="EXP_SINGLE_NAME",
                affected_positions=[name],
                affected_systems=[src],
                current_value=round(w, 4),
                threshold=critical_threshold,
                recommendation=f"Single position {name} at {w:.1%} is critically high."
            ))
        elif w >= max_threshold:
            alerts.append(make_alert(
                severity="WARNING",
                message=f"Single position {name} ({src}) at {w:.1%} exceeds {max_threshold:.0%}.",
                check_id="EXP_SINGLE_NAME",
                affected_positions=[name],
                affected_systems=[src],
                current_value=round(w, 4),
                threshold=max_threshold
            ))
        elif w >= warning_band:
            alerts.append(make_alert(
                severity="MONITOR",
                message=f"Single position {name} ({src}) at {w:.1%} approaching limit.",
                check_id="EXP_SINGLE_NAME",
                affected_positions=[name],
                current_value=round(w, 4),
                threshold=warning_band
            ))

    return alerts


# ═══════════════════════════════════════════════════════════════════
# EXP_ASSET_CLASS — Asset-Class Verteilung
# Spec Teil 2 §10.4
# ═══════════════════════════════════════════════════════════════════

def check_asset_class(v16_weights, f6_positions, config):
    """
    Berechnet Asset-Class-Verteilung des Gesamtportfolios.

    Returns: (asset_classes_dict, list_of_alerts)
    """
    asset_classes = {
        "Equity_US": 0.0,
        "Equity_International": 0.0,
        "Bonds": 0.0,
        "Commodities": 0.0,
        "Cash_Equivalent": 0.0
    }

    for asset, weight in v16_weights.items():
        ac = get_asset_class(asset)
        asset_classes[ac] = asset_classes.get(ac, 0.0) + weight

    for position in (f6_positions or []):
        ticker = position.get("ticker", "")
        weight = position.get("current_weight", 0.0)
        ac = get_asset_class(ticker)
        asset_classes[ac] = asset_classes.get(ac, 0.0) + weight

    # Pruefung
    total_equity = asset_classes["Equity_US"] + asset_classes["Equity_International"]
    hedge_assets = asset_classes["Bonds"] + asset_classes["Commodities"]

    equity_warning = config.get("total_equity_warning", 0.85)
    equity_critical = config.get("total_equity_critical", 0.92)
    min_hedge = config.get("min_hedge_assets", 0.08)

    alerts = []

    if total_equity >= equity_critical:
        alerts.append(make_alert(
            severity="WARNING",
            message=(
                f"Total equity exposure {total_equity:.1%} exceeds critical threshold. "
                f"Bonds: {asset_classes['Bonds']:.1%}, "
                f"Commodities: {asset_classes['Commodities']:.1%}, "
                f"Cash: {asset_classes['Cash_Equivalent']:.1%}."
            ),
            check_id="EXP_ASSET_CLASS",
            current_value=round(total_equity, 4),
            threshold=equity_critical,
            recommendation=(
                "Portfolio heavily concentrated in equities. "
                "Limited downside protection beyond V16 DD-Protect."
            )
        ))
    elif total_equity >= equity_warning:
        alerts.append(make_alert(
            severity="MONITOR",
            message=f"Total equity exposure {total_equity:.1%} above {equity_warning:.0%}.",
            check_id="EXP_ASSET_CLASS",
            current_value=round(total_equity, 4),
            threshold=equity_warning
        ))

    if hedge_assets < min_hedge:
        alerts.append(make_alert(
            severity="MONITOR",
            message=(
                f"Hedge assets (Bonds + Commodities) at {hedge_assets:.1%}, "
                f"below minimum {min_hedge:.0%}. "
                f"Portfolio has limited downside protection beyond V16 DD-Protect."
            ),
            check_id="EXP_ASSET_CLASS",
            current_value=round(hedge_assets, 4),
            threshold=min_hedge
        ))

    return asset_classes, alerts