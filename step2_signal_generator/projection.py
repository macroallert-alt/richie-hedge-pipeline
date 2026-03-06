"""
step2_signal_generator/projection.py
Portfolio Projection — Baseline (V16-only in V1)

V1 Scope: Only Baseline projection from V16 weights.
Full projection (with F6/PermOpt/Router) comes in V2.
Source: Signal Generator Spec Teil 3 §16
"""

import logging

logger = logging.getLogger("signal_generator.projection")


def calculate_baseline_projection(v16_trades: dict) -> dict:
    """
    Baseline Projection: What the portfolio holds RIGHT NOW.
    V1: Only V16 weights, no F6 substitutions.
    Spec Teil 3 §16.2

    Returns:
        {
            "positions": {asset: {"weight": float, "source": str, "type": str}, ...},
            "sector_exposure": {sector: float, ...},
            "total_weight": float,
            "type": "BASELINE",
            "description": str
        }
    """
    weights = v16_trades.get("weights", {})
    positions = {}

    for asset, wdata in weights.items():
        w = wdata.get("weight", 0.0) if isinstance(wdata, dict) else 0.0
        if w > 0.0001:
            positions[asset] = {
                "weight": round(w, 6),
                "source": "V16",
                "type": "ETF",
            }

    # Sector exposure = same as positions for V1 (no F6 substitutions)
    sector_exposure = {
        asset: data["weight"]
        for asset, data in positions.items()
    }

    total_weight = round(sum(d["weight"] for d in positions.values()), 4)

    return {
        "positions": positions,
        "sector_exposure": sector_exposure,
        "total_weight": total_weight,
        "type": "BASELINE",
        "description": "Aktuelle Portfolio-Zusammensetzung (V16-only, V1)",
    }


def calculate_effective_concentration(projection: dict, config: dict) -> dict:
    """
    Measure effective Tech concentration across all positions.
    Spec Teil 3 §16.4

    SPY has ~30% Tech, QQQ ~50%, XLK is 100% Tech, etc.
    """
    rules = config.get("compilation_rules", {}).get("projection_rules", {})
    tech_map = rules.get("tech_exposure_map", {})
    warning_threshold = rules.get("concentration_warning_threshold", 0.35)

    positions = projection.get("positions", {})
    effective_tech_pct = 0.0

    for asset, data in positions.items():
        weight = data.get("weight", 0.0)
        tech_factor = tech_map.get(asset, tech_map.get("_DEFAULT", 0.10))
        effective_tech_pct += weight * tech_factor

    # Top-5 concentration
    weights_sorted = sorted(
        [(a, d["weight"]) for a, d in positions.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    top5_weight = sum(w for _, w in weights_sorted[:5])
    top5_assets = [a for a, _ in weights_sorted[:5]]

    has_warning = effective_tech_pct > warning_threshold
    warning_msg = (
        f"Effective Tech exposure {effective_tech_pct:.1%} exceeds {warning_threshold:.0%} threshold"
        if has_warning else None
    )

    return {
        "effective_tech_pct": round(effective_tech_pct, 4),
        "top5_concentration_pct": round(top5_weight, 4),
        "top5_assets": top5_assets,
        "warning": has_warning,
        "warning_message": warning_msg,
    }


def build_projections_output(baseline: dict, concentration: dict) -> dict:
    """
    Assemble projections output block.
    V1: Only baseline + concentration. No full projection.
    """
    return {
        "baseline": baseline,
        "full": {
            "status": "UNAVAILABLE",
            "note": "Full projection (V16+F6+PermOpt+Router) available in V2",
        },
        "concentration_check": {
            "baseline": concentration,
            "full": {
                "status": "UNAVAILABLE",
            },
        },
        "delta": {
            "new_assets": [],
            "removed_assets": [],
            "weight_changes": {},
            "total_changes": 0,
            "has_material_changes": False,
            "note": "Delta available in V2 when Full projection is implemented",
        },
    }
