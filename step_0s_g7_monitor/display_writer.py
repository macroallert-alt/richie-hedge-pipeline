"""
step_0s_g7_monitor/display_writer.py
Phase 10 Extension: Display Writer for 11 Layout Tabs

Uses batchUpdate to write all cells per tab in ONE API call.
Row numbers verified from actual Google Sheets CSV exports 2026-03-08.

VOLLSTAENDIG — schreibt Enrichment-Daten in ALLE vorher leeren Zellen.
"""

from datetime import datetime, timezone
import os
import tempfile

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

CYCLE_PHASES = {
    "USA": "Late Decline?", "CHINA": "Peak / Early Decline?",
    "EU": "Managed Decline", "INDIA": "Early Rise",
    "JP_KR_TW": "Tech Power, Geo Risk", "GULF": "Rising Wealth",
    "REST_EM": "Frontier Growth",
}

KEY_RISKS = {
    "USA": "Fiscal / Polarization", "CHINA": "Demographics / Property",
    "EU": "Energy / Demographics", "INDIA": "Institutions / Infrastructure",
    "JP_KR_TW": "Taiwan / Demographics", "GULF": "Oil Transition / Stability",
    "REST_EM": "Governance / Debt",
}


def _fmt(val, decimals=1):
    if val is None: return "—"
    if isinstance(val, float): return f"{val:.{decimals}f}"
    return str(val)

def _fmt_pct(val, decimals=1):
    if val is None: return "—"
    if isinstance(val, (int, float)): return f"{val:.{decimals}f}%"
    return str(val)

def _fmt_delta(val, decimals=2):
    if val is None or val == 0: return "—"
    if isinstance(val, (int, float)): return f"{val:+.{decimals}f}"
    return str(val)

def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _quarter_label():
    now = datetime.now(timezone.utc)
    return f"Q{(now.month - 1) // 3 + 1} {now.year}"


# ============================================================
# ENRICHMENT HELPER
# ============================================================

def _ev(enr, block, field, region):
    """Extract enrichment value: enr[block][data][region] or enr[block][data][region][field]."""
    b = enr.get(block, {})
    data = b.get("data", {})
    entry = data.get(region)
    if isinstance(entry, dict) and field:
        v = entry.get(field)
        return _fmt(v) if v is not None else "—"
    elif isinstance(entry, (int, float)):
        return _fmt(entry)
    return "—"

def _ev_row(enr, block, field=None):
    """Build a row of enrichment values for all 7 regions."""
    return [_ev(enr, block, field, r) for r in REGIONS]


class G7DisplayWriter:

    def __init__(self, spreadsheet_id):
        self.sheet_id = spreadsheet_id
        self.service = None

    def connect(self):
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            creds_json = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
            if not creds_json:
                print("[G7DisplayWriter] No GCP_SA_KEY or GOOGLE_CREDENTIALS")
                return False
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(creds_json)
                creds_path = f.name
            try:
                creds = service_account.Credentials.from_service_account_file(
                    creds_path, scopes=["https://www.googleapis.com/auth/spreadsheets"])
                self.service = build("sheets", "v4", credentials=creds)
                return True
            finally:
                os.unlink(creds_path)
        except Exception as e:
            print(f"[G7DisplayWriter] Auth failed: {e}")
            return False

    def _batch_write(self, data_list):
        if not data_list:
            return
        body = {"valueInputOption": "USER_ENTERED",
                "data": [{"range": rng, "values": vals} for rng, vals in data_list]}
        try:
            self.service.spreadsheets().values().batchUpdate(
                spreadsheetId=self.sheet_id, body=body).execute()
        except Exception as e:
            print(f"  [DisplayWriter] batchUpdate failed: {e}")

    def _extract_probs(self, scenario_result):
        if not scenario_result or not isinstance(scenario_result, dict):
            return [0.40, 0.20, 0.25, 0.15]
        probs = None
        # Try current_thesis first, then thesis
        for key in ("current_thesis", "thesis"):
            ct = scenario_result.get(key)
            if isinstance(ct, dict):
                probs = ct.get("scenario_probabilities") or ct.get("probabilities")
                if probs and isinstance(probs, dict):
                    break
        if not probs or not isinstance(probs, dict):
            return [0.40, 0.20, 0.25, 0.15]
        return [
            probs.get("managed_decline", probs.get("S1_managed_decline", probs.get("S1_status_quo", 0.40))),
            probs.get("conflict_escalation", probs.get("S2_conflict", probs.get("S2_bifurcation", 0.20))),
            probs.get("us_renewal", probs.get("S3_us_renewal", probs.get("S3_fragmentation", 0.25))),
            probs.get("multipolar_chaos", probs.get("S4_multipolar", probs.get("S4_fiscal_dominance", 0.15))),
        ]

    def _extract_tilts(self, scenario_result):
        """Extract computed tilts from scenario result."""
        for key in ("current_thesis", "thesis"):
            ct = scenario_result.get(key) if isinstance(scenario_result, dict) else None
            if isinstance(ct, dict) and ct.get("computed_tilts"):
                return ct["computed_tilts"]
        return {}

    # ============================================================
    # DASHBOARD
    # ============================================================

    def write_dashboard(self, power_scores, gap_data, overlays, g7_status, scenario_result):
        print("  [Display] DASHBOARD...")
        d = []

        # --- Power Scores table (B5:G11) ---
        tilts = self._extract_tilts(scenario_result)
        ps_rows = []
        for region in REGIONS:
            ps = power_scores.get(region, {})
            score = ps.get("score")
            mom = ps.get("momentum", 0)
            trend = "▲" if mom > 0.3 else "▼" if mom < -0.3 else "►"
            # Column G: dominant tilt direction for this region's key ETF
            region_etf = {"USA": "SPY", "CHINA": "FXI", "EU": "EFA", "INDIA": "INDA",
                          "JP_KR_TW": "EFA", "GULF": "EEM", "REST_EM": "VWO"}
            etf = region_etf.get(region, "EEM")
            tilt_val = tilts.get(etf, 0)
            tilt_str = f"{tilt_val:+.2f}" if tilt_val != 0 else "—"
            ps_rows.append([_fmt(score), trend, _fmt_delta(mom),
                           CYCLE_PHASES.get(region, "—"), KEY_RISKS.get(region, "—"), tilt_str])
        d.append(("DASHBOARD!B5:G11", ps_rows))

        # --- Gap section (D14:D17) ---
        gap = gap_data.get("gap", 0)
        thuc = "LOW"
        for loop in overlays.get("feedback_loops", []):
            if loop.get("loop_id") == "thucydides_trap":
                thuc = "HIGH" if loop["status"] == "ACTIVE" else "MEDIUM" if loop["status"] == "LATENT" else "LOW"
                break
        d.append(("DASHBOARD!D14:D17", [
            [_fmt(gap)], [gap_data.get("trend", "STABLE")], [thuc],
            ["N/A" if gap_data.get("trend") != "CLOSING" else "Monitor"]]))

        # --- Scenario Probabilities (B21:B24) ---
        probs = self._extract_probs(scenario_result)
        d.append(("DASHBOARD!B21:B24", [[_fmt_pct(p * 100, 0)] for p in probs]))

        # --- Tilt Direction Table (B27:C34) — top 8 assets by absolute tilt ---
        if tilts:
            sorted_tilts = sorted(tilts.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            tilt_rows = []
            for asset, tilt_val in sorted_tilts:
                direction = "OW" if tilt_val > 0.05 else "UW" if tilt_val < -0.05 else "N"
                tilt_rows.append([asset, f"{direction} ({tilt_val:+.2f})"])
            d.append(("DASHBOARD!B27:C34", tilt_rows))

        # --- Run info (D39:D42) ---
        d.append(("DASHBOARD!D39:D42", [
            [_now_iso()], ["Next scheduled run"], ["Weekly + Quarterly"],
            [g7_status.get("attention_flag", "NONE")]]))

        self._batch_write(d)

    # ============================================================
    # POWER_SCORES
    # ============================================================

    def write_power_scores(self, scores, momenta, power_scores, scenario_result=None):
        print("  [Display] POWER_SCORES...")
        d = []
        def _rv(dim):
            return [_fmt(scores.get(dim, {}).get(r)) for r in REGIONS]

        d.append(("POWER_SCORES!C7:I11", [_rv("D1_economic"), _rv("D2_demographics"),
                  _rv("D3_technology"), _rv("D4_energy"), _rv("D5_military")]))
        d.append(("POWER_SCORES!C15:I17", [_rv("D6_fiscal"), _rv("D7_currency"), _rv("D8_capital_mkt")]))
        d.append(("POWER_SCORES!C21:I24", [_rv("D9_flows"), _rv("D10_social"),
                  _rv("D11_geopolitical"), _rv("D12_feedback")]))
        d.append(("POWER_SCORES!C28:I28", [[_fmt(power_scores.get(r, {}).get("score")) for r in REGIONS]]))
        d.append(("POWER_SCORES!C29:I29", [[_fmt_delta(power_scores.get(r, {}).get("momentum", 0)) for r in REGIONS]]))
        d.append(("POWER_SCORES!C30:I30", [[CYCLE_PHASES.get(r, "—") for r in REGIONS]]))

        # --- Regime Sensitivity Check (C34:I37) ---
        # Shows how each region's "effective score" shifts under each pure scenario
        # Uses tilt as proxy: region_etf tilt under each scenario
        if scenario_result:
            from step_0s_g7_monitor.scenario_engine import ASSET_EXPOSURE_VECTORS
            region_etf = {"USA": "SPY", "CHINA": "FXI", "EU": "EFA", "INDIA": "INDA",
                          "JP_KR_TW": "EFA", "GULF": "EEM", "REST_EM": "VWO"}
            scenario_names = ["managed_decline", "conflict_escalation", "us_renewal", "multipolar_chaos"]
            regime_rows = []
            for sn in scenario_names:
                row = []
                for r in REGIONS:
                    etf = region_etf.get(r, "EEM")
                    exp = ASSET_EXPOSURE_VECTORS.get(etf, {})
                    val = exp.get(sn, 0)
                    row.append(f"{val:+.1f}")
                regime_rows.append(row)
            d.append(("POWER_SCORES!C34:I37", regime_rows))

        self._batch_write(d)

    # ============================================================
    # STRUCTURAL — ALL CELLS
    # ============================================================

    def write_structural(self, validated_data, scores):
        print("  [Display] STRUCTURAL...")
        imf = validated_data.get("imf_weo", {})
        wb = validated_data.get("worldbank", {})
        enr = validated_data.get("enrichment", {})
        now = _today()
        d = []

        def _iv(ind, r):
            e = imf.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"
        def _wv(ind, r):
            e = wb.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        # D1 rows 5-9: Economic Weight
        d.append(("STRUCTURAL!D5:J5", [[_iv("NGDPD", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L5", [[now]]))
        d.append(("STRUCTURAL!D6:J6", [[_iv("NGDP_RPCH", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L6", [[now]]))
        d.append(("STRUCTURAL!D7:J7", [_ev_row(enr, "gdp_per_capita_ppp")])); d.append(("STRUCTURAL!L7", [[now]]))
        d.append(("STRUCTURAL!D8:J8", [_ev_row(enr, "labor_productivity_growth")])); d.append(("STRUCTURAL!L8", [[now]]))
        d.append(("STRUCTURAL!D9:J9", [_ev_row(enr, "manufacturing_gdp_pct")])); d.append(("STRUCTURAL!L9", [[now]]))

        # D2 rows 13-18: Demographics
        d.append(("STRUCTURAL!D13:J13", [_ev_row(enr, "working_age_pop_growth")])); d.append(("STRUCTURAL!L13", [[now]]))
        d.append(("STRUCTURAL!D14:J14", [[_wv("SP.POP.DPND", r) for r in REGIONS]])); d.append(("STRUCTURAL!L14", [[now]]))
        d.append(("STRUCTURAL!D15:J15", [_ev_row(enr, "median_age")])); d.append(("STRUCTURAL!L15", [[now]]))
        d.append(("STRUCTURAL!D16:J16", [[_wv("SP.DYN.TFRT.IN", r) for r in REGIONS]])); d.append(("STRUCTURAL!L16", [[now]]))
        d.append(("STRUCTURAL!D17:J17", [_ev_row(enr, "net_migration_rate")])); d.append(("STRUCTURAL!L17", [[now]]))
        d.append(("STRUCTURAL!D18:J18", [_ev_row(enr, "youth_unemployment")])); d.append(("STRUCTURAL!L18", [[now]]))

        # D3 rows 22-28: Technology / Innovation
        d.append(("STRUCTURAL!D22:J22", [_ev_row(enr, "rd_spend_gdp_pct")])); d.append(("STRUCTURAL!L22", [[now]]))
        d.append(("STRUCTURAL!D23:J23", [_ev_row(enr, "wipo_patents")])); d.append(("STRUCTURAL!L23", [[now]]))
        d.append(("STRUCTURAL!D24:J24", [_ev_row(enr, "ai_papers_published")])); d.append(("STRUCTURAL!L24", [[now]]))
        d.append(("STRUCTURAL!D25:J25", [_ev_row(enr, "semiconductor_revenue_share")])); d.append(("STRUCTURAL!L25", [[now]]))
        d.append(("STRUCTURAL!D26:J26", [_ev_row(enr, "vc_deep_tech_bn")])); d.append(("STRUCTURAL!L26", [[now]]))
        d.append(("STRUCTURAL!D27:J27", [_ev_row(enr, "top_100_tech_hq_count")])); d.append(("STRUCTURAL!L27", [[now]]))
        d.append(("STRUCTURAL!D28:J28", [_ev_row(enr, "stem_graduates_thousands")])); d.append(("STRUCTURAL!L28", [[now]]))

        # D4 rows 32-36: Energy Sovereignty
        d.append(("STRUCTURAL!D32:J32", [_ev_row(enr, "energy_import_dependency")])); d.append(("STRUCTURAL!L32", [[now]]))
        d.append(("STRUCTURAL!D33:J33", [_ev_row(enr, "renewable_electricity_share")])); d.append(("STRUCTURAL!L33", [[now]]))
        d.append(("STRUCTURAL!D34:J34", [_ev_row(enr, "critical_mineral_processing_share")])); d.append(("STRUCTURAL!L34", [[now]]))
        d.append(("STRUCTURAL!D35:J35", [_ev_row(enr, "strategic_petroleum_reserves_days")])); d.append(("STRUCTURAL!L35", [[now]]))
        d.append(("STRUCTURAL!D36:J36", [_ev_row(enr, "lng_export_capacity_bcm")])); d.append(("STRUCTURAL!L36", [[now]]))

        # D5 rows 40-44: Military / Projection
        d.append(("STRUCTURAL!D40:J40", [[_wv("MS.MIL.XPND.GD.ZS", r) for r in REGIONS]])); d.append(("STRUCTURAL!L40", [[now]]))
        d.append(("STRUCTURAL!D41:J41", [_ev_row(enr, "sipri_military", "absolute_bn_usd")])); d.append(("STRUCTURAL!L41", [[now]]))
        d.append(("STRUCTURAL!D42:J42", [_ev_row(enr, "nuclear_warheads")])); d.append(("STRUCTURAL!L42", [[now]]))
        d.append(("STRUCTURAL!D43:J43", [_ev_row(enr, "aircraft_carriers")])); d.append(("STRUCTURAL!L43", [[now]]))
        d.append(("STRUCTURAL!D44:J44", [_ev_row(enr, "foreign_military_bases")])); d.append(("STRUCTURAL!L44", [[now]]))

        self._batch_write(d)

    # ============================================================
    # FINANCIAL — ALL CELLS
    # ============================================================

    def write_financial(self, validated_data, overlays):
        print("  [Display] FINANCIAL...")
        fred = validated_data.get("fred", {})
        imf = validated_data.get("imf_weo", {})
        cofer = validated_data.get("imf_cofer", {})
        yf = validated_data.get("yfinance", {})
        enr = validated_data.get("enrichment", {})
        now = _today()
        d = []

        def _iv(ind, r):
            e = imf.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        # D6 Row 5: Debt/GDP
        d.append(("FINANCIAL!D5:J5", [[_iv("GGXWDG_NGDP", r) for r in REGIONS]])); d.append(("FINANCIAL!L5", [[now]]))
        # Row 6: Debt/GDP 5Y Trajectory — placeholder (need IMF history computation)
        d.append(("FINANCIAL!D6:J6", [["—"] * 7]))
        # Row 7: ITR
        itr_row = ["—"] * 7
        i_d = fred.get("A091RC1Q027SBEA"); r_d = fred.get("FGRECPT")
        if (isinstance(i_d, dict) and i_d.get("value") is not None
                and isinstance(r_d, dict) and r_d.get("value") is not None and r_d["value"] > 0):
            itr_row[0] = _fmt(i_d["value"] / r_d["value"] * 100)
        d.append(("FINANCIAL!D7:J7", [itr_row])); d.append(("FINANCIAL!L7", [[now]]))
        # Row 8: Deficit
        def_row = ["—"] * 7
        deficit = fred.get("FYFSGDA188S")
        if isinstance(deficit, dict) and deficit.get("value") is not None:
            def_row[0] = _fmt(deficit["value"])
        d.append(("FINANCIAL!D8:J8", [def_row]))
        # Row 9: CB Balance Sheet / GDP (enrichment)
        d.append(("FINANCIAL!D9:J9", [_ev_row(enr, "cb_balance_sheet_gdp_pct")])); d.append(("FINANCIAL!L9", [[now]]))
        # Row 10: M2 vs GDP — placeholder
        d.append(("FINANCIAL!D10:J10", [["—"] * 7]))
        # Row 11: Real Rate
        rr_row = ["—"] * 7
        dgs = fred.get("DGS10"); inf_u = imf.get("PCPIPCH_USA")
        if (isinstance(dgs, dict) and dgs.get("value") is not None
                and isinstance(inf_u, dict) and inf_u.get("value") is not None):
            rr_row[0] = _fmt(dgs["value"] - inf_u["value"])
        d.append(("FINANCIAL!D11:J11", [rr_row]))
        # Row 12: NFCI
        nfci_row = ["—"] * 7
        st = fred.get("STLFSI4")
        if isinstance(st, dict) and st.get("value") is not None:
            nfci_row[0] = _fmt(st["value"], 2)
        d.append(("FINANCIAL!D12:J12", [nfci_row]))
        # Row 13: ANFCI
        anfci_row = ["—"] * 7
        anfci = fred.get("ANFCI")
        if isinstance(anfci, dict) and anfci.get("value") is not None:
            anfci_row[0] = _fmt(anfci["value"], 2)
        d.append(("FINANCIAL!D13:J13", [anfci_row]))

        # D7 Row 15: COFER
        c_row = ["—"] * 7
        c_row[0] = _fmt(cofer.get("USD_share"))
        c_row[1] = _fmt(cofer.get("CNY_share"))
        c_row[2] = _fmt(cofer.get("EUR_share"))
        d.append(("FINANCIAL!D15:J15", [c_row])); d.append(("FINANCIAL!L15", [[now]]))
        # Row 16: COFER 5Y Trend — placeholder
        d.append(("FINANCIAL!D16:J16", [["—"] * 7]))
        # Row 17: CB Gold Holdings (enrichment)
        d.append(("FINANCIAL!D17:J17", [_ev_row(enr, "cb_gold_holdings_tonnes")])); d.append(("FINANCIAL!L17", [[now]]))
        # Row 18: Gold Purchases
        gp = enr.get("cb_gold_purchases_tonnes_yr", {})
        gp_total = gp.get("global_total")
        d.append(("FINANCIAL!D18:J18", [[_fmt(gp_total) if gp_total else "—"] + ["—"] * 6])); d.append(("FINANCIAL!L18", [[now]]))
        # Row 19: SWIFT Payment Share (enrichment)
        d.append(("FINANCIAL!D19:J19", [_ev_row(enr, "swift_payment_share_pct")])); d.append(("FINANCIAL!L19", [[now]]))
        # Row 20: Currency vs USD 5Y (enrichment)
        d.append(("FINANCIAL!D20:J20", [_ev_row(enr, "currency_vs_usd_5y_pct")])); d.append(("FINANCIAL!L20", [[now]]))
        # Row 21: DXY
        dxy_row = ["—"] * 7
        dxy = yf.get("DX-Y.NYB")
        if isinstance(dxy, dict) and dxy.get("close") is not None:
            dxy_row[0] = _fmt(dxy["close"])
        d.append(("FINANCIAL!D21:J21", [dxy_row])); d.append(("FINANCIAL!L21", [[now]]))

        # D8 Row 25-31: Capital Market Depth (ALL from enrichment)
        d.append(("FINANCIAL!D25:J25", [_ev_row(enr, "market_cap_gdp_pct")])); d.append(("FINANCIAL!L25", [[now]]))
        d.append(("FINANCIAL!D26:J26", [_ev_row(enr, "bond_market_gdp_pct")])); d.append(("FINANCIAL!L26", [[now]]))
        d.append(("FINANCIAL!D27:J27", [["—"] * 7]))  # Daily Trading Volume — no good source yet
        d.append(("FINANCIAL!D28:J28", [_ev_row(enr, "property_rights_score")])); d.append(("FINANCIAL!L28", [[now]]))
        d.append(("FINANCIAL!D29:J29", [_ev_row(enr, "capital_controls_severity")])); d.append(("FINANCIAL!L29", [[now]]))
        d.append(("FINANCIAL!D30:J30", [_ev_row(enr, "rule_of_law_score")])); d.append(("FINANCIAL!L30", [[now]]))
        d.append(("FINANCIAL!D31:J31", [_ev_row(enr, "fdi_gdp_pct")])); d.append(("FINANCIAL!L31", [[now]]))

        self._batch_write(d)

    # ============================================================
    # LEADING — ALL CELLS
    # ============================================================

    def write_leading(self, scores, momenta, overlays, validated_data):
        print("  [Display] LEADING...")
        enr = validated_data.get("enrichment", {})
        now = _today()
        d = []

        # D9 rows 5-9: Capital Flows
        d.append(("LEADING!D5:J5", [["—"] * 7]))  # ETF Flows — no source yet
        d.append(("LEADING!D6:J6", [_ev_row(enr, "fdi_inflows_bn")])); d.append(("LEADING!L6", [[now]]))
        d.append(("LEADING!D7:J7", [_ev_row(enr, "treasury_holdings_bn")])); d.append(("LEADING!L7", [[now]]))
        d.append(("LEADING!D8:J8", [["—"] * 7]))  # Portfolio Flows — no source yet
        d.append(("LEADING!D9:J9", [["—"] * 7]))  # SWF Activity — qualitative

        # D10 rows 13-18: Social Cohesion
        d.append(("LEADING!D13:J13", [_ev_row(enr, "trust_in_government_pct")])); d.append(("LEADING!L13", [[now]]))
        d.append(("LEADING!D14:J14", [_ev_row(enr, "political_polarization_score")])); d.append(("LEADING!L14", [[now]]))
        d.append(("LEADING!D15:J15", [_ev_row(enr, "gini_coefficient")])); d.append(("LEADING!L15", [[now]]))
        d.append(("LEADING!D16:J16", [_ev_row(enr, "social_mobility_rank")])); d.append(("LEADING!L16", [[now]]))
        d.append(("LEADING!D17:J17", [["—"] * 7]))  # Consumer Conf — no source yet
        d.append(("LEADING!D18:J18", [["—"] * 7]))  # Protest Index — needs ACLED

        # D11 rows 22-27: Geopolitical Dynamics
        d.append(("LEADING!D22:J22", [_ev_row(enr, "alliance_strength_score")])); d.append(("LEADING!L22", [[now]]))
        d.append(("LEADING!D23:J23", [_ev_row(enr, "sanctions_active_count")])); d.append(("LEADING!L23", [[now]]))
        d.append(("LEADING!D24:J24", [_ev_row(enr, "trade_gdp_ratio")])); d.append(("LEADING!L24", [[now]]))
        d.append(("LEADING!D25:J25", [["—"] * 7]))  # Reshoring Index — Kearney annual only
        # Row 26: GPR
        gpr_row = ["—"] * 7
        gpr = overlays.get("gpr_index_current")
        if gpr is not None:
            gpr_row[0] = _fmt(gpr, 0)
        d.append(("LEADING!D26:J26", [gpr_row])); d.append(("LEADING!L26", [[now]]))
        d.append(("LEADING!D27:J27", [_ev_row(enr, "conflict_proximity_score")])); d.append(("LEADING!L27", [[now]]))

        # D12 rows 31-35: Feedback Loop Intensity
        feedback_loops = overlays.get("feedback_loops", [])
        ls = {}
        for loop in feedback_loops:
            lid = loop.get("loop_id", "")
            r = loop.get("region", "")
            s = loop.get("severity", 0)
            ls.setdefault(lid, {})[r] = s

        def _lr(lid):
            return [_fmt(ls.get(lid, {}).get(r, 0)) if ls.get(lid, {}).get(r, 0) > 0 else "—" for r in REGIONS]

        net = []
        for r in REGIONS:
            t = sum(ls.get(lid, {}).get(r, 0) for lid in ls)
            net.append(_fmt(t) if t > 0 else "—")

        d.append(("LEADING!D31:J35", [
            _lr("debt_demographics"), _lr("energy_conflict"),
            _lr("tech_security"), _lr("currency_fiscal"), net]))
        d.append(("LEADING!L31:L35", [[now]] * 5))

        self._batch_write(d)

    # ============================================================
    # FEEDBACK_LOOPS
    # ============================================================

    def write_feedback_loops(self, overlays):
        print("  [Display] FEEDBACK_LOOPS...")
        feedback_loops = overlays.get("feedback_loops", [])
        d = []

        ms = {}
        for loop in feedback_loops:
            lid = loop.get("loop_id", "")
            s = loop.get("severity", 0)
            if lid not in ms or s > ms[lid]:
                ms[lid] = s

        neg_rows = []
        for lid in ["debt_demographics", "currency_fiscal", "thucydides_trap",
                     "energy_conflict", "financial_contagion", "social_political"]:
            sev = ms.get(lid, 0)
            neg_rows.append([_fmt(sev) if sev > 0 else "INACTIVE",
                           "FAST" if sev > 5 else "SLOW" if sev > 0 else "—"])
        d.append(("FEEDBACK_LOOPS!D5:E10", neg_rows))
        d.append(("FEEDBACK_LOOPS!D14:E17", [["—", "—"]] * 4))

        lbr = {}
        for loop in feedback_loops:
            r = loop.get("region", "")
            lid = loop.get("loop_id", "")
            s = loop.get("severity", 0)
            lbr.setdefault(r, {})[lid] = s

        neg_cols = ["debt_demographics", "currency_fiscal", "thucydides_trap",
                    "energy_conflict", "financial_contagion", "social_political"]
        matrix = []
        for region in REGIONS:
            rl = lbr.get(region, {})
            cells = []
            net = 0
            for lid in neg_cols:
                sv = rl.get(lid, 0)
                cells.append(_fmt(sv) if sv > 0 else "—")
                net += sv
            cells.extend(["—"] * 4)
            cells.append(_fmt(net) if net > 0 else "—")
            matrix.append(cells)
        d.append(("FEEDBACK_LOOPS!B21:L27", matrix))
        self._batch_write(d)

    # ============================================================
    # SCENARIOS
    # ============================================================

    def write_scenarios(self, scenario_result):
        print("  [Display] SCENARIOS...")
        probs = self._extract_probs(scenario_result)
        d = [
            ("SCENARIOS!B4", [[_fmt_pct(probs[0] * 100, 0)]]),
            ("SCENARIOS!B13", [[_fmt_pct(probs[1] * 100, 0)]]),
            ("SCENARIOS!B22", [[_fmt_pct(probs[2] * 100, 0)]]),
            ("SCENARIOS!B31", [[_fmt_pct(probs[3] * 100, 0)]]),
        ]

        # --- Probability-Weighted Tilt Table (SCENARIOS!B40:G48) ---
        # 9 key assets: SPY, QQQ, EEM, TLT, GLD, FXI, INDA, BTC, DBC
        # Columns: Asset, Managed Decline, Conflict, Renewal, Multipolar, Weighted Tilt
        from step_0s_g7_monitor.scenario_engine import ASSET_EXPOSURE_VECTORS
        display_assets = ["SPY", "QQQ", "EEM", "TLT", "GLD", "FXI", "INDA", "BTC", "DBC"]
        tilt_rows = []
        for asset in display_assets:
            exp = ASSET_EXPOSURE_VECTORS.get(asset, {})
            a_exp = exp.get("managed_decline", 0)
            b_exp = exp.get("conflict_escalation", 0)
            c_exp = exp.get("us_renewal", 0)
            d_exp = exp.get("multipolar_chaos", 0)
            weighted = (probs[0] * a_exp + probs[1] * b_exp
                        + probs[2] * c_exp + probs[3] * d_exp)
            tilt_rows.append([
                asset,
                f"{a_exp:+.1f}", f"{b_exp:+.1f}", f"{c_exp:+.1f}", f"{d_exp:+.1f}",
                f"{weighted:+.3f}",
            ])
        d.append(("SCENARIOS!B40:G48", tilt_rows))

        # --- Thesis metadata (SCENARIOS!B51:B54) ---
        thesis = None
        for key in ("current_thesis", "thesis"):
            t = scenario_result.get(key) if isinstance(scenario_result, dict) else None
            if isinstance(t, dict) and t.get("dominant_thesis"):
                thesis = t
                break
        if thesis:
            d.append(("SCENARIOS!B51:B54", [
                [thesis.get("dominant_thesis", "—")],
                [thesis.get("confidence", "—")],
                [thesis.get("probability_source", "—")],
                ["Yes" if thesis.get("interim_flag") else "No"],
            ]))

        self._batch_write(d)

    # ============================================================
    # SCORING
    # ============================================================

    def write_scoring(self, validated_data, g7_status):
        print("  [Display] SCORING...")
        fred = validated_data.get("fred", {})
        yf = validated_data.get("yfinance", {})
        gpr_data = validated_data.get("gpr")

        def _yc(t):
            e = yf.get(t)
            return _fmt(e["close"]) if isinstance(e, dict) and e.get("close") is not None else "—"
        def _fv(s, dc=2):
            e = fred.get(s)
            return _fmt(e["value"], dc) if isinstance(e, dict) and e.get("value") is not None else "—"

        gv = gpr_data.get("gpr_global") if isinstance(gpr_data, dict) else None
        gold = yf.get("GC=F")
        gp = gold.get("pct_change_1m") if isinstance(gold, dict) else None

        self._batch_write([("SCORING!B5:B16", [
            [_yc("^VIX")],              # VIX
            [_yc("^VIX3M")],            # VIX3M
            [_fv("BAMLH0A0HYM2")],      # HY Spread
            [_fmt(gv, 0) if gv else "—"],# GPR
            [_fv("T10Y2Y")],            # 2Y10Y
            [_fv("BAMLC0A0CM")],        # Credit Stress IG OAS
            [_fmt_pct(gp) if gp else "—"],# Gold 1M%
            [_yc("DX-Y.NYB")],          # DXY
            ["—"],                       # V16 State — cross-read pending
            [_yc("^MOVE")],              # MOVE Index
            [_fv("STLFSI4")],           # NFCI
            [_fv("ANFCI")],             # ANFCI
        ])])

    # ============================================================
    # SOURCES
    # ============================================================

    def write_sources(self, freshness_by_source, collection_errors):
        print("  [Display] SOURCES...")
        d = []
        for src, row in [("fred", 5), ("imf_weo", 6), ("imf_cofer", 7),
                         ("worldbank", 8), ("un_pop", 9), ("yfinance", 10)]:
            f = freshness_by_source.get(src, "UNAVAILABLE")
            d.append((f"SOURCES!H{row}", [["LIVE" if f in ("FRESH", "RECENT") else "STALE" if f == "STALE" else "DOWN"]]))
        for src, row in [("acled", 21), ("gpr", 22)]:
            f = freshness_by_source.get(src, "UNAVAILABLE")
            d.append((f"SOURCES!H{row}", [["LIVE" if f in ("FRESH", "RECENT") else "DOWN"]]))
        self._batch_write(d)

    # ============================================================
    # HISTORY
    # ============================================================

    def write_history(self, power_scores, scenario_result):
        print("  [Display] HISTORY...")
        quarter = _quarter_label()
        qr = {"Q1 2026": 5, "Q2 2026": 6, "Q3 2026": 7, "Q4 2026": 8,
              "Q1 2027": 9, "Q2 2027": 10, "Q3 2027": 11, "Q4 2027": 12}
        sr = {"Q1 2026": 16, "Q2 2026": 17, "Q3 2026": 18, "Q4 2026": 19,
              "Q1 2027": 20, "Q2 2027": 21, "Q3 2027": 22, "Q4 2027": 23}
        d = []
        row = qr.get(quarter)
        if row:
            d.append((f"HISTORY!B{row}:H{row}", [[_fmt(power_scores.get(r, {}).get("score")) for r in REGIONS]]))
        sc_row = sr.get(quarter)
        if sc_row:
            probs = self._extract_probs(scenario_result)
            d.append((f"HISTORY!B{sc_row}:E{sc_row}", [[_fmt_pct(p * 100, 0) for p in probs]]))
        if d:
            self._batch_write(d)

    # ============================================================
    # UNIVERSE_MAP
    # ============================================================

    def write_universe_map(self, overlays):
        print("  [Display] UNIVERSE_MAP...")
        att = overlays.get("attractiveness", [])
        if not att:
            return
        rbr = {e["region"]: e["rank"] for e in att}
        self._batch_write([
            ("UNIVERSE_MAP!G5", [[f"Core (#{rbr.get('USA', '?')})"]]),
            ("UNIVERSE_MAP!G17", [[f"Tilt: #{rbr.get('INDIA', '?')} India"]]),
            ("UNIVERSE_MAP!G18", [[f"#{rbr.get('EU', '?')}"]]),
        ])

    # ============================================================
    # MASTER
    # ============================================================

    def write_all(self, scoring_result, overlays, g7_status, scenario_result,
                  validated_data, freshness_by_source, collection_errors):
        scores = scoring_result.get("scores", {})
        momenta = scoring_result.get("momenta", {})
        power_scores = scoring_result.get("power_scores", {})
        gap_data = scoring_result.get("gap_data", {})

        for name, fn, args in [
            ("DASHBOARD", self.write_dashboard, (power_scores, gap_data, overlays, g7_status, scenario_result)),
            ("POWER_SCORES", self.write_power_scores, (scores, momenta, power_scores, scenario_result)),
            ("STRUCTURAL", self.write_structural, (validated_data, scores)),
            ("FINANCIAL", self.write_financial, (validated_data, overlays)),
            ("LEADING", self.write_leading, (scores, momenta, overlays, validated_data)),
            ("FEEDBACK_LOOPS", self.write_feedback_loops, (overlays,)),
            ("SCENARIOS", self.write_scenarios, (scenario_result,)),
            ("SCORING", self.write_scoring, (validated_data, g7_status)),
            ("SOURCES", self.write_sources, (freshness_by_source, collection_errors)),
            ("HISTORY", self.write_history, (power_scores, scenario_result)),
            ("UNIVERSE_MAP", self.write_universe_map, (overlays,)),
        ]:
            try:
                fn(*args)
            except Exception as e:
                print(f"  [Display] {name} ERROR: {e}")

        print("  [Display] All 11 layout tabs written")
