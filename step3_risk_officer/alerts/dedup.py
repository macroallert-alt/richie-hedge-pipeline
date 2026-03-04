"""
step_3_risk_officer/alerts/dedup.py
Alert-Deduplizierung, Trend-Berechnung, Ongoing Conditions.
Spec: Risk Officer Teil 3 §20
"""

from ..utils.helpers import SEVERITY_ORDER


def deduplicate_and_trend(today_alerts, yesterday_alerts):
    """
    Vergleicht heutige Alerts mit gestrigen fuer Trend-Erkennung.

    REGELN (Spec Teil 3 §20.1):
    1. Gleicher check_id + gleiche Severity → STABLE (erste 3 Tage) oder ONGOING (ab Tag 4)
    2. Gleicher check_id + hoehere Severity → ESCALATING
    3. Gleicher check_id + niedrigere Severity → DEESCALATING
    4. Neuer check_id → NEW
    5. Gestriger check_id weg → RESOLVED

    Returns: (active_alerts, ongoing_conditions)
    """
    yesterday_by_check = {}
    for a in (yesterday_alerts or []):
        cid = a.get("check_id")
        if cid:
            yesterday_by_check[cid] = a

    active_alerts = []
    ongoing_conditions = []

    for alert in today_alerts:
        cid = alert.get("check_id", "")

        if cid not in yesterday_by_check:
            # Neuer Alert
            alert["trend"] = "NEW"
            alert["previous_severity"] = None
            alert["days_active"] = 1
            active_alerts.append(alert)
        else:
            prev = yesterday_by_check[cid]
            prev_sev = SEVERITY_ORDER.get(prev.get("severity", "MONITOR"), 1)
            curr_sev = SEVERITY_ORDER.get(alert.get("severity", "MONITOR"), 1)
            days = prev.get("days_active", 1) + 1

            if curr_sev > prev_sev:
                # Eskaliert
                alert["trend"] = "ESCALATING"
                alert["previous_severity"] = prev.get("severity")
                alert["days_active"] = days
                active_alerts.append(alert)

            elif curr_sev < prev_sev:
                # De-eskaliert
                alert["trend"] = "DEESCALATING"
                alert["previous_severity"] = prev.get("severity")
                alert["days_active"] = days
                active_alerts.append(alert)

            else:
                # Gleiche Severity
                alert["previous_severity"] = prev.get("severity")
                alert["days_active"] = days

                if days <= 3:
                    # Erste 3 Tage: Noch als Alert anzeigen
                    alert["trend"] = "STABLE"
                    active_alerts.append(alert)
                else:
                    # Ab Tag 4: In ongoing_conditions verschieben
                    alert["trend"] = "ONGOING"
                    ongoing_conditions.append(alert)

    # Resolved: Gestrige Alerts die heute nicht mehr existieren
    today_check_ids = {a.get("check_id") for a in today_alerts}
    for cid, prev_alert in yesterday_by_check.items():
        if cid not in today_check_ids:
            # Ueberspringe ONGOING die gestern schon ongoing waren
            if prev_alert.get("trend") == "ONGOING":
                continue
            resolved = {
                "check_id": cid,
                "severity": "RESOLVED",
                "trend": "RESOLVED",
                "previous_severity": prev_alert.get("severity"),
                "message": (
                    f"Previously active alert '{cid}' has resolved. "
                    f"Was {prev_alert.get('severity', '?')} for "
                    f"{prev_alert.get('days_active', 1)} day(s)."
                ),
                "days_active": 0,
                "id": prev_alert.get("id", ""),
                "context": prev_alert.get("context", {}),
                "recommendation": "",
                "base_severity": "RESOLVED",
                "boost_applied": None
            }
            active_alerts.append(resolved)

    return active_alerts, ongoing_conditions