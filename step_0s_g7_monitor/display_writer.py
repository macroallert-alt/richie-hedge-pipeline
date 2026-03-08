"""
step_0s_g7_monitor/display_writer.py
Phase 10 Extension: Display Writer for 11 Layout Tabs

Writes computed data from Phase 3 (Scoring) and Phase 4 (Overlays)
into the 11 pre-formatted layout tabs with Bloomberg Terminal styling.

The layout tabs have FIXED structure (headers, section dividers, label columns).
This module ONLY writes into the data cells (replacing '—' placeholders).
It does NOT touch headers, labels, section titles, or formatting.

Tabs written:
  DASHBOARD, POWER_SCORES, STRUCTURAL, FINANCIAL, LEADING,
  FEEDBACK_LOOPS, SCENARIOS, SCORING, SOURCES, HISTORY, UNIVERSE_MAP

Uses googleapiclient.discovery (NOT gspread) — same pattern as sheet_writer.py.
Auth via GCP_SA_KEY / GOOGLE_CREDENTIALS environment variable.
"""

from datetime import datetime, timezone
import os
import json
import tempfile

# ============================================================
# CONSTANTS
# ============================================================

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]

# Display names for regions (as they appear in the sheet tabs)
REGION_DISPLAY = {
    "USA": "USA",
    "CHINA": "CHINA",
    "EU": "EU / EUROZONE",
    "INDIA": "INDIA",
    "JP_KR_TW": "JAPAN / KOREA / TAIWAN",
    "GULF": "GULF / MIDDLE EAST",
    "REST_EM": "REST EM",
}

# Dimension display names
DIM_KEYS = [
    "D1_economic", "D2_demographics", "D3_technology", "D4_energy",
    "D5_military", "D6_fiscal", "D7_currency", "D8_capital_mkt",
    "D9_flows", "D10_social", "D11_geopolitical", "D12_feedback",
]

# Cycle phase estimates (structural, slow-changing)
CYCLE_PHASES = {
    "USA": "Late Decline?",
    "CHINA": "Peak / Early Decline?",
    "EU": "Managed Decline",
    "INDIA": "Early Rise",
    "JP_KR_TW": "Tech Power, Geo Risk",
    "GULF": "Rising Wealth",
    "REST_EM": "Frontier Growth",
}

KEY_RISKS = {
    "USA": "Fiscal / Polarization",
    "CHINA": "Demographics / Property",
    "EU": "Energy / Demographics",
    "INDIA": "Institutions / Infrastructure",
    "JP_KR_TW": "Taiwan / Demographics",
    "GULF": "Oil Transition / Stability",
    "REST_EM": "Governance / Debt",
}


def _fmt(val, decimals=1):
    """Format number for sheet display. None -> '—'."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _fmt_pct(val, decimals=1):
    """Format as percentage string."""
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}%"
    return str(val)


def _fmt_delta(val, decimals=2):
    """Format as signed delta."""
    if val is None or val == 0:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:+.{decimals}f}"
    return str(val)


def _trend_arrow(trend):
    """Convert trend string to arrow symbol."""
    arrows = {
        "RISING": "▲",
        "WIDENING": "▲",
        "CLOSING": "▼",
        "FALLING": "▼",
        "STABLE": "►",
        "INTENSIFYING": "▲▲",
        "INACTIVE": "—",
    }
    return arrows.get(trend, trend or "—")


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _quarter_label():
    """Current quarter label like 'Q1 2026'."""
    now = datetime.now(timezone.utc)
    q = (now.month - 1) // 3 + 1
    return f"Q{q} {now.year}"


# ============================================================
# MAIN DISPLAY WRITER CLASS
# ============================================================

class G7DisplayWriter:
    """
    Writes computed data into the 11 layout tabs.
    Uses own auth (same pattern as G7SheetWriter).
    """

    def __init__(self, spreadsheet_id):
        """
        Args:
            spreadsheet_id: G7 Sheet ID
        """
        self.sheet_id = spreadsheet_id
        self.service = None

    def connect(self):
        """Authenticate and build Sheets service. Returns True on success."""
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
                    creds_path,
                    scopes=["https://www.googleapis.com/auth/spreadsheets"],
                )
                self.service = build("sheets", "v4", credentials=creds)
                return True
            finally:
                os.unlink(creds_path)
        except Exception as e:
            print(f"[G7DisplayWriter] Auth failed: {e}")
            return False

    def _write(self, range_str, values):
        """Write values to a range. values = list of lists."""
        try:
            self.service.spreadsheets().values().update(
                spreadsheetId=self.sheet_id,
                range=range_str,
                valueInputOption="USER_ENTERED",
                body={"values": values},
            ).execute()
        except Exception as e:
            print(f"  [DisplayWriter] Write failed {range_str}: {e}")

    # ============================================================
    # DASHBOARD TAB
    # ============================================================

    def write_dashboard(self, power_scores, gap_data, overlays, g7_status, scenario_result):
        """
        DASHBOARD tab — rows 5-11 (power scores), 14-17 (differential),
        21-24 (scenarios), 28-36 (tilts), 39-42 (schedule).
        """
        print("  [Display] DASHBOARD...")

        # --- Power Scores table: rows 5-11, columns B-G ---
        # Row order: USA, CHINA, EU, INDIA, JP_KR_TW, GULF, REST_EM
        ps_rows = []
        for region in REGIONS:
            ps = power_scores.get(region, {})
            score = ps.get("score")
            momentum = ps.get("momentum", 0)
            trend = "▲" if momentum > 0.3 else "▼" if momentum < -0.3 else "►"
            ps_rows.append([
                _fmt(score),
                trend,
                _fmt_delta(momentum),
                CYCLE_PHASES.get(region, "—"),
                KEY_RISKS.get(region, "—"),
                "—",  # Tilt — Etappe 3/4
            ])
        self._write("DASHBOARD!B5:G11", ps_rows)

        # --- Power Differential: rows 14-17, column D ---
        gap = gap_data.get("gap", 0)
        gap_trend = gap_data.get("trend", "STABLE")

        # Thucydides trap risk from feedback loops
        thuc_risk = "LOW"
        for loop in overlays.get("feedback_loops", []):
            if loop.get("loop_id") == "thucydides_trap" or loop.get("name") == "Thucydides Trap":
                if loop["status"] == "ACTIVE":
                    thuc_risk = "HIGH"
                elif loop["status"] == "LATENT":
                    thuc_risk = "MEDIUM"
                break

        crossover_est = "N/A" if gap_trend != "CLOSING" else "Monitor"

        self._write("DASHBOARD!D14:D17", [
            [_fmt(gap)],
            [gap_trend],
            [thuc_risk],
            [crossover_est],
        ])

        # --- Scenarios: rows 21-24, column B (probability only) ---
        scenario = scenario_result or {}
        probs = scenario.get("current_thesis", {}).get("probabilities", {})
        if not probs:
            probs = scenario.get("thesis", {}).get("probabilities", {})
        if not probs:
            # Default stubs
            probs = {"S1_managed_decline": 0.40, "S2_conflict": 0.20,
                     "S3_us_renewal": 0.25, "S4_multipolar": 0.15}

        self._write("DASHBOARD!B21:B24", [
            [_fmt_pct((probs.get("S1_managed_decline", probs.get("S1_status_quo", 0.40))) * 100, 0)],
            [_fmt_pct((probs.get("S2_conflict", probs.get("S2_bifurcation", 0.20))) * 100, 0)],
            [_fmt_pct((probs.get("S3_us_renewal", probs.get("S3_fragmentation", 0.25))) * 100, 0)],
            [_fmt_pct((probs.get("S4_multipolar", probs.get("S4_fiscal_dominance", 0.15))) * 100, 0)],
        ])

        # --- Review Schedule: rows 39-42, column D ---
        self._write("DASHBOARD!D39:D42", [
            [_now_iso()],
            ["Next scheduled run"],
            ["Weekly + Quarterly"],
            [g7_status.get("attention_flag", "NONE")],
        ])

    # ============================================================
    # POWER_SCORES TAB
    # ============================================================

    def write_power_scores(self, scores, momenta, power_scores):
        """
        POWER_SCORES tab — dimension scores in rows 7-11, 15-17, 21-24.
        Total/Trend/Phase in rows 28-30. Columns C-I (7 regions).
        """
        print("  [Display] POWER_SCORES...")

        def _region_vals(dim):
            """Get score values for all 7 regions for a dimension."""
            return [_fmt(scores.get(dim, {}).get(r)) for r in REGIONS]

        # D1-D5: rows 7-11, columns C-I
        self._write("POWER_SCORES!C7:I11", [
            _region_vals("D1_economic"),
            _region_vals("D2_demographics"),
            _region_vals("D3_technology"),
            _region_vals("D4_energy"),
            _region_vals("D5_military"),
        ])

        # D6-D8: rows 15-17, columns C-I
        self._write("POWER_SCORES!C15:I17", [
            _region_vals("D6_fiscal"),
            _region_vals("D7_currency"),
            _region_vals("D8_capital_mkt"),
        ])

        # D9-D12: rows 21-24, columns C-I
        self._write("POWER_SCORES!C21:I24", [
            _region_vals("D9_flows"),
            _region_vals("D10_social"),
            _region_vals("D11_geopolitical"),
            _region_vals("D12_feedback"),
        ])

        # Total Power Score: row 28, columns C-I
        totals = [_fmt(power_scores.get(r, {}).get("score")) for r in REGIONS]
        self._write("POWER_SCORES!C28:I28", [totals])

        # Trend: row 29, columns C-I
        trends = []
        for r in REGIONS:
            mom = power_scores.get(r, {}).get("momentum", 0)
            trends.append(_fmt_delta(mom))
        self._write("POWER_SCORES!C29:I29", [trends])

        # Cycle Phase: row 30, columns C-I
        phases = [CYCLE_PHASES.get(r, "—") for r in REGIONS]
        self._write("POWER_SCORES!C30:I30", [phases])

    # ============================================================
    # STRUCTURAL TAB (D1-D5 raw data)
    # ============================================================

    def write_structural(self, validated_data, scores):
        """
        STRUCTURAL tab — D1 rows 5-9, D2 rows 12-17, D3 rows 20-26,
        D4 rows 29-33, D5 rows 36-40. Columns D-J (7 regions).
        Only writes data we actually have.
        """
        print("  [Display] STRUCTURAL...")

        imf = validated_data.get("imf_weo", {})
        wb = validated_data.get("worldbank", {})
        fred = validated_data.get("fred", {})

        def _imf_val(indicator, region):
            entry = imf.get(f"{indicator}_{region}")
            if isinstance(entry, dict) and entry.get("value") is not None:
                return _fmt(entry["value"])
            return "—"

        def _wb_val(indicator, region):
            entry = wb.get(f"{indicator}_{region}")
            if isinstance(entry, dict) and entry.get("value") is not None:
                return _fmt(entry["value"])
            return "—"

        # --- D1: Economic Weight, rows 5-9, cols D-J ---
        # Row 5: GDP Share (PPP) — not directly available, use NGDPD as proxy
        # Row 6: GDP Growth — NGDP_RPCH
        # Row 7: GDP per Capita — not available
        # Row 8: Productivity — not available
        # Row 9: Manufacturing — not available
        d1_data = []
        # GDP Share (use nominal GDP as proxy)
        d1_data.append([_imf_val("NGDPD", r) for r in REGIONS])
        # GDP Growth
        d1_data.append([_imf_val("NGDP_RPCH", r) for r in REGIONS])
        # GDP per Capita — placeholder
        d1_data.append(["—"] * 7)
        # Productivity — placeholder
        d1_data.append(["—"] * 7)
        # Manufacturing — placeholder
        d1_data.append(["—"] * 7)
        self._write("STRUCTURAL!D5:J9", d1_data)

        # --- D2: Demographics, rows 12-17, cols D-J ---
        d2_data = []
        # Working-Age Pop Growth — placeholder
        d2_data.append(["—"] * 7)
        # Dependency Ratio
        d2_data.append([_wb_val("SP.POP.DPND", r) for r in REGIONS])
        # Median Age — placeholder
        d2_data.append(["—"] * 7)
        # Fertility Rate
        d2_data.append([_wb_val("SP.DYN.TFRT.IN", r) for r in REGIONS])
        # Net Migration — placeholder
        d2_data.append(["—"] * 7)
        # Youth Unemployment — placeholder
        d2_data.append(["—"] * 7)
        self._write("STRUCTURAL!D12:J17", d2_data)

        # --- D3: Technology, rows 20-26, cols D-J ---
        d3_data = []
        # R&D Spend
        d3_data.append([_wb_val("GB.XPD.RSDV.GD.ZS", r) for r in REGIONS])
        # Patent Filings — placeholder
        d3_data.append(["—"] * 7)
        # AI Papers — placeholder
        d3_data.append(["—"] * 7)
        # Semi Revenue — placeholder
        d3_data.append(["—"] * 7)
        # VC Investment — placeholder
        d3_data.append(["—"] * 7)
        # Top 100 Tech — placeholder
        d3_data.append(["—"] * 7)
        # STEM Grads — placeholder
        d3_data.append(["—"] * 7)
        self._write("STRUCTURAL!D20:J26", d3_data)

        # --- D4: Energy, rows 29-33, cols D-J ---
        # All placeholder except scores show energy profile
        d4_data = [["—"] * 7 for _ in range(5)]
        self._write("STRUCTURAL!D29:J33", d4_data)

        # --- D5: Military, rows 36-40, cols D-J ---
        d5_data = []
        # Defense Spend %GDP
        d5_data.append([_wb_val("MS.MIL.XPND.GD.ZS", r) for r in REGIONS])
        # Rest placeholder
        for _ in range(4):
            d5_data.append(["—"] * 7)
        self._write("STRUCTURAL!D36:J40", d5_data)

        # --- UPDATED column (K) for rows with data ---
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # D1 row 5-6: IMF data
        self._write("STRUCTURAL!K5:K6", [[now], [now]])
        # D2 row 13, 15: WB data
        self._write("STRUCTURAL!K13", [[now]])
        self._write("STRUCTURAL!K15", [[now]])
        # D3 row 20: WB R&D
        self._write("STRUCTURAL!K20", [[now]])
        # D5 row 36: WB military
        self._write("STRUCTURAL!K36", [[now]])

    # ============================================================
    # FINANCIAL TAB (D6-D8 raw data)
    # ============================================================

    def write_financial(self, validated_data, overlays):
        """
        FINANCIAL tab — D6 rows 5-13, D7 rows 14-20, D8 rows 23-29.
        Columns D-J (7 regions).
        """
        print("  [Display] FINANCIAL...")

        fred = validated_data.get("fred", {})
        imf = validated_data.get("imf_weo", {})
        cofer = validated_data.get("imf_cofer", {})
        yf = validated_data.get("yfinance", {})

        def _imf_val(indicator, region):
            entry = imf.get(f"{indicator}_{region}")
            if isinstance(entry, dict) and entry.get("value") is not None:
                return _fmt(entry["value"])
            return "—"

        def _fred_val(series):
            entry = fred.get(series)
            if isinstance(entry, dict) and entry.get("value") is not None:
                return _fmt(entry["value"])
            return "—"

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # --- D6: Fiscal Health, rows 5-13 ---
        # Row 5: Gross Debt/GDP
        self._write("FINANCIAL!D5:J5", [[_imf_val("GGXWDG_NGDP", r) for r in REGIONS]])
        self._write("FINANCIAL!K5", [[now]])

        # Row 6: Debt/GDP 5Y Trajectory — placeholder
        self._write("FINANCIAL!D6:J6", [["—"] * 7])

        # Row 7: Interest Cost / Tax Revenue — USA only from FRED
        itr_row = ["—"] * 7
        int_data = fred.get("A091RC1Q027SBEA")
        rev_data = fred.get("FGRECPT")
        if (isinstance(int_data, dict) and int_data.get("value") is not None
                and isinstance(rev_data, dict) and rev_data.get("value") is not None
                and rev_data["value"] > 0):
            itr = int_data["value"] / rev_data["value"] * 100
            itr_row[0] = _fmt(itr)
        self._write("FINANCIAL!D7:J7", [itr_row])
        self._write("FINANCIAL!K7", [[now]])

        # Row 8: Fiscal Deficit/GDP — USA from FRED
        deficit_row = ["—"] * 7
        deficit = fred.get("FYFSGDA188S")
        if isinstance(deficit, dict) and deficit.get("value") is not None:
            deficit_row[0] = _fmt(deficit["value"])
        self._write("FINANCIAL!D8:J8", [deficit_row])

        # Row 9: CB Balance/GDP — placeholder
        self._write("FINANCIAL!D9:J9", [["—"] * 7])

        # Row 10: M2 Growth vs Nominal GDP — placeholder
        self._write("FINANCIAL!D10:J10", [["—"] * 7])

        # Row 11: Real Interest Rate — USA from FRED DGS10 - IMF inflation
        real_row = ["—"] * 7
        dgs10 = fred.get("DGS10")
        inf_usa = imf.get("PCPIPCH_USA")
        if (isinstance(dgs10, dict) and dgs10.get("value") is not None
                and isinstance(inf_usa, dict) and inf_usa.get("value") is not None):
            real_rate = dgs10["value"] - inf_usa["value"]
            real_row[0] = _fmt(real_rate)
        self._write("FINANCIAL!D11:J11", [real_row])

        # Row 12: NFCI
        nfci_row = ["—"] * 7
        stlfsi = fred.get("STLFSI4")
        if isinstance(stlfsi, dict) and stlfsi.get("value") is not None:
            nfci_row[0] = _fmt(stlfsi["value"], 2)
        self._write("FINANCIAL!D12:J12", [nfci_row])

        # Row 13: ANFCI — placeholder
        self._write("FINANCIAL!D13:J13", [["—"] * 7])

        # --- D7: Currency / Reserve Status, rows 14-20 ---
        # Row 14: COFER Reserve Share
        cofer_row = ["—"] * 7
        cofer_row[0] = _fmt(cofer.get("USD_share"))  # USA = USD share
        cofer_row[2] = _fmt(cofer.get("EUR_share"))   # EU = EUR share
        cofer_row[1] = _fmt(cofer.get("CNY_share"))   # China = CNY share
        self._write("FINANCIAL!D14:J14", [cofer_row])
        self._write("FINANCIAL!K14", [[now]])

        # Row 15: COFER 5Y Trend — placeholder
        self._write("FINANCIAL!D15:J15", [["—"] * 7])

        # Row 16-17: Gold Holdings/Purchases — placeholder
        self._write("FINANCIAL!D16:J17", [["—"] * 7, ["—"] * 7])

        # Row 18: SWIFT Payment Share — placeholder
        self._write("FINANCIAL!D18:J18", [["—"] * 7])

        # Row 19: Currency vs USD — placeholder
        self._write("FINANCIAL!D19:J19", [["—"] * 7])

        # Row 20: DXY — USA column only
        dxy_row = ["—"] * 7
        dxy = yf.get("DX-Y.NYB")
        if isinstance(dxy, dict) and dxy.get("close") is not None:
            dxy_row[0] = _fmt(dxy["close"])
        self._write("FINANCIAL!D20:J20", [dxy_row])
        self._write("FINANCIAL!K20", [[now]])

        # --- D8: Capital Market Depth, rows 23-29 --- all placeholder
        for row in range(23, 30):
            self._write(f"FINANCIAL!D{row}:J{row}", [["—"] * 7])

    # ============================================================
    # LEADING TAB (D9-D12 + EWI)
    # ============================================================

    def write_leading(self, scores, momenta, overlays, validated_data):
        """
        LEADING tab — D9 rows 5-9, D10 rows 12-17, D11 rows 20-25,
        D12 rows 28-32. Columns D-J (7 regions).
        """
        print("  [Display] LEADING...")

        def _score_vals(dim):
            return [_fmt(scores.get(dim, {}).get(r)) for r in REGIONS]

        # --- D9: Capital Flows, rows 5-9 ---
        # Row 5: Regional ETF Flows — placeholder
        self._write("LEADING!D5:J5", [["—"] * 7])
        # Row 6: FDI Net Inflows — placeholder
        self._write("LEADING!D6:J6", [["—"] * 7])
        # Row 7: Foreign Holdings US Treasuries — placeholder
        self._write("LEADING!D7:J7", [["—"] * 7])
        # Row 8: Portfolio Flows — placeholder
        self._write("LEADING!D8:J8", [["—"] * 7])
        # Row 9: SWF Activity — placeholder
        self._write("LEADING!D9:J9", [["—"] * 7])

        # --- D10: Social Cohesion, rows 12-17 --- all placeholder
        for row in range(12, 18):
            self._write(f"LEADING!D{row}:J{row}", [["—"] * 7])

        # --- D11: Geopolitical Dynamics, rows 20-25 ---
        # Row 20: Alliance Strength — placeholder
        self._write("LEADING!D20:J20", [["—"] * 7])
        # Row 21: Active Sanctions — placeholder
        self._write("LEADING!D21:J21", [["—"] * 7])
        # Row 22: Trade/GDP — placeholder
        self._write("LEADING!D22:J22", [["—"] * 7])
        # Row 23: Reshoring Index — placeholder
        self._write("LEADING!D23:J23", [["—"] * 7])
        # Row 24: GPR Index — USA column
        gpr_row = ["—"] * 7
        gpr_current = overlays.get("gpr_index_current")
        if gpr_current is not None:
            gpr_row[0] = _fmt(gpr_current, 0)
        self._write("LEADING!D24:J24", [gpr_row])
        # Row 25: Conflict Proximity — placeholder
        self._write("LEADING!D25:J25", [["—"] * 7])

        # --- D12: Feedback Loop Intensity, rows 28-32 ---
        # Rows 28-31: Individual loop scores, Row 32: NET
        # Map our 7 loops to the 4 display rows + NET
        feedback_loops = overlays.get("feedback_loops", [])

        # Build severity lookup: loop_id -> {region: severity}
        loop_severity = {}
        for loop in feedback_loops:
            lid = loop.get("loop_id", loop.get("name", ""))
            region = loop.get("region", "")
            sev = loop.get("severity", 0)
            if lid not in loop_severity:
                loop_severity[lid] = {}
            loop_severity[lid][region] = sev

        def _loop_row(loop_id):
            row = []
            for r in REGIONS:
                sev = loop_severity.get(loop_id, {}).get(r)
                row.append(_fmt(sev) if sev and sev > 0 else "—")
            return row

        # Row 28: Debt x Demographics
        self._write("LEADING!D28:J28", [_loop_row("debt_demographics")])
        # Row 29: Geopolitics x Energy
        self._write("LEADING!D29:J29", [_loop_row("energy_conflict")])
        # Row 30: Tech x Capital (tech_security)
        self._write("LEADING!D30:J30", [_loop_row("tech_security")])
        # Row 31: Currency x Fiscal
        self._write("LEADING!D31:J31", [_loop_row("currency_fiscal")])

        # Row 32: NET FEEDBACK SCORE — sum of all severities per region
        net_row = []
        for r in REGIONS:
            total = 0
            for lid, regions in loop_severity.items():
                total += regions.get(r, 0)
            net_row.append(_fmt(total) if total > 0 else "—")
        self._write("LEADING!D32:J32", [net_row])

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._write("LEADING!K24", [[now]])
        self._write("LEADING!K28:K32", [[now]] * 5)

    # ============================================================
    # FEEDBACK_LOOPS TAB
    # ============================================================

    def write_feedback_loops(self, overlays):
        """
        FEEDBACK_LOOPS tab.
        Negative Spirals severity: rows 5-10, column D (SEVERITY), E (SPEED).
        Positive Spirals: rows 14-17, column D, E.
        Interaction Matrix: rows 21-27, columns B-K.
        """
        print("  [Display] FEEDBACK_LOOPS...")

        feedback_loops = overlays.get("feedback_loops", [])

        # Build lookup: loop_id -> max severity across regions
        max_sev = {}
        max_status = {}
        for loop in feedback_loops:
            lid = loop.get("loop_id", loop.get("name", ""))
            sev = loop.get("severity", 0)
            status = loop.get("status", "INACTIVE")
            if lid not in max_sev or sev > max_sev[lid]:
                max_sev[lid] = sev
                max_status[lid] = status

        # Map our loop IDs to the display rows
        # Row 5: Debt-Demo = debt_demographics
        # Row 6: Fiscal Dominance — not a direct loop in our system, use FDP
        # Row 7: Thucydides = thucydides_trap
        # Row 8: Energy Dependency = energy_conflict
        # Row 9: Capital Flight = financial_contagion
        # Row 10: Inequality-Populism = social_political
        neg_map = [
            ("debt_demographics", 5),
            ("currency_fiscal", 6),     # Closest to "Fiscal Dominance"
            ("thucydides_trap", 7),
            ("energy_conflict", 8),
            ("financial_contagion", 9),
            ("social_political", 10),
        ]

        for loop_id, row in neg_map:
            sev = max_sev.get(loop_id, 0)
            status = max_status.get(loop_id, "INACTIVE")
            speed = "FAST" if sev > 5 else "SLOW" if sev > 0 else "—"
            self._write(f"FEEDBACK_LOOPS!D{row}:E{row}", [
                [_fmt(sev) if sev > 0 else "INACTIVE", speed]
            ])

        # Positive spirals: rows 14-17 — these are structural estimates
        # Row 14: Tech-Capital (USA/China tech score as proxy)
        # Row 15: Demographic Dividend (India score as proxy)
        # Row 16: Reserve Currency Privilege
        # Row 17: Energy Independence
        # Leave as structural — no computed severity yet
        for row in range(14, 18):
            self._write(f"FEEDBACK_LOOPS!D{row}:E{row}", [["—", "—"]])

        # --- Interaction Matrix: rows 21-27, columns B-K ---
        # 7 regions x 10 loop columns + NET
        loop_severity_by_region = {}
        for loop in feedback_loops:
            r = loop.get("region", "")
            lid = loop.get("loop_id", "")
            sev = loop.get("severity", 0)
            if r not in loop_severity_by_region:
                loop_severity_by_region[r] = {}
            loop_severity_by_region[r][lid] = sev

        # Column order: Debt-Demo, Fiscal Dom, Thucydides, Energy Dep,
        #               Capital Fl, Ineq-Pop, Tech-Cap+, Demo Div+, Reserve+, Energy+, NET
        loop_cols = [
            "debt_demographics", "currency_fiscal", "thucydides_trap",
            "energy_conflict", "financial_contagion", "social_political",
            None, None, None, None,  # Positive loops = placeholder
        ]

        for i, region in enumerate(REGIONS):
            row_num = 21 + i
            region_loops = loop_severity_by_region.get(region, {})
            cells = []
            net = 0
            for lid in loop_cols:
                if lid is None:
                    cells.append("—")
                else:
                    sev = region_loops.get(lid, 0)
                    if sev > 0:
                        cells.append(_fmt(sev))
                        net += sev
                    else:
                        cells.append("—")
            cells.append(_fmt(net) if net > 0 else "—")
            self._write(f"FEEDBACK_LOOPS!B{row_num}:L{row_num}", [cells])

    # ============================================================
    # SOURCES TAB
    # ============================================================

    def write_sources(self, freshness_by_source, collection_errors):
        """
        SOURCES tab — update STATUS column (H) for automatable sources.
        Rows 5-12 (API sources), 15-24 (semi-auto), 28-32 (discretionary).
        """
        print("  [Display] SOURCES...")

        # Map source names to row numbers
        source_rows = {
            "fred": 5,
            "imf_weo": 6,
            "imf_cofer": 7,
            "worldbank": 8,
            "un_pop": 9,
            "yfinance": 10,
            # "bis": 11,  # FRED (BIS) — not separate
            # "nfci": 12, # Chicago Fed — via FRED
        }

        semi_auto_rows = {
            "gpr": 22,      # Caldara-Iacoviello
            "acled": 21,     # ACLED
        }

        for source, row in source_rows.items():
            freshness = freshness_by_source.get(source, "UNAVAILABLE")
            status = "LIVE" if freshness in ("FRESH", "RECENT") else "STALE" if freshness == "STALE" else "DOWN"
            self._write(f"SOURCES!H{row}", [[status]])

        for source, row in semi_auto_rows.items():
            freshness = freshness_by_source.get(source, "UNAVAILABLE")
            status = "LIVE" if freshness in ("FRESH", "RECENT") else "DOWN"
            self._write(f"SOURCES!H{row}", [[status]])

    # ============================================================
    # HISTORY TAB
    # ============================================================

    def write_history(self, power_scores, scenario_result):
        """
        HISTORY tab — append current quarter's scores.
        Power Score History: rows 5-12 (Q1 2026 - Q4 2027), cols B-H.
        Scenario History: rows 16-23, cols B-E.
        Finds the current quarter row and writes.
        """
        print("  [Display] HISTORY...")

        quarter = _quarter_label()

        # Map quarter to row
        quarter_rows = {
            "Q1 2026": 5, "Q2 2026": 6, "Q3 2026": 7, "Q4 2026": 8,
            "Q1 2027": 9, "Q2 2027": 10, "Q3 2027": 11, "Q4 2027": 12,
        }
        scenario_quarter_rows = {
            "Q1 2026": 16, "Q2 2026": 17, "Q3 2026": 18, "Q4 2026": 19,
            "Q1 2027": 20, "Q2 2027": 21, "Q3 2027": 22, "Q4 2027": 23,
        }

        row = quarter_rows.get(quarter)
        if row:
            ps_vals = [_fmt(power_scores.get(r, {}).get("score")) for r in REGIONS]
            self._write(f"HISTORY!B{row}:H{row}", [ps_vals])

        sc_row = scenario_quarter_rows.get(quarter)
        if sc_row:
            scenario = scenario_result or {}
            probs = scenario.get("current_thesis", {}).get("probabilities", {})
            if not probs:
                probs = scenario.get("thesis", {}).get("probabilities", {})
            if probs:
                sc_vals = [
                    _fmt_pct(probs.get("S1_managed_decline", probs.get("S1_status_quo", 0)) * 100, 0),
                    _fmt_pct(probs.get("S2_conflict", probs.get("S2_bifurcation", 0)) * 100, 0),
                    _fmt_pct(probs.get("S3_us_renewal", probs.get("S3_fragmentation", 0)) * 100, 0),
                    _fmt_pct(probs.get("S4_multipolar", probs.get("S4_fiscal_dominance", 0)) * 100, 0),
                ]
                self._write(f"HISTORY!B{sc_row}:E{sc_row}", [sc_vals])

    # ============================================================
    # UNIVERSE_MAP TAB
    # ============================================================

    def write_universe_map(self, overlays):
        """
        UNIVERSE_MAP tab — attractiveness ranking drives G7 TILT column.
        Current V16 Assets: rows 5-30, column G (G7 TILT).
        """
        print("  [Display] UNIVERSE_MAP...")

        attractiveness = overlays.get("attractiveness", [])
        if not attractiveness:
            return

        # Build region->rank lookup
        rank_by_region = {}
        for entry in attractiveness:
            rank_by_region[entry["region"]] = entry["rank"]

        # Map assets to regions and assign tilt hints
        # Only write G7 TILT column (G) for key assets
        asset_region_map = {
            5: "USA",      # SPY
            6: "USA",      # QQQ
            7: "USA",      # IWM
            17: None,      # EEM — EM Broad
            18: "EU",      # VGK
            19: None,      # GLD — Neutral
            29: None,      # BTC — Neutral
        }

        # For now, write the top-ranked region info
        # Detailed tilts come in Etappe 3/4 with scenario engine
        top_region = attractiveness[0]["region"] if attractiveness else "—"
        self._write("UNIVERSE_MAP!G5", [[f"Core (#{rank_by_region.get('USA', '?')})"]])
        self._write("UNIVERSE_MAP!G17", [[f"Tilt: #{rank_by_region.get('INDIA', '?')} India"]])
        self._write("UNIVERSE_MAP!G18", [[f"#{rank_by_region.get('EU', '?')}"]])

    # ============================================================
    # SCENARIOS TAB (mostly static, write probabilities)
    # ============================================================

    def write_scenarios(self, scenario_result):
        """
        SCENARIOS tab — write probability values.
        Row 3 (Sc A prob), Row 10 (Sc B), Row 17 (Sc C), Row 24 (Sc D).
        Column B.
        """
        print("  [Display] SCENARIOS...")

        scenario = scenario_result or {}
        probs = scenario.get("current_thesis", {}).get("probabilities", {})
        if not probs:
            probs = scenario.get("thesis", {}).get("probabilities", {})
        if not probs:
            probs = {"S1_managed_decline": 0.40, "S2_conflict": 0.20,
                     "S3_us_renewal": 0.25, "S4_multipolar": 0.15}

        self._write("SCENARIOS!B4", [[_fmt_pct(probs.get("S1_managed_decline", probs.get("S1_status_quo", 0.40)) * 100, 0)]])
        self._write("SCENARIOS!B11", [[_fmt_pct(probs.get("S2_conflict", probs.get("S2_bifurcation", 0.20)) * 100, 0)]])
        self._write("SCENARIOS!B18", [[_fmt_pct(probs.get("S3_us_renewal", probs.get("S3_fragmentation", 0.25)) * 100, 0)]])
        self._write("SCENARIOS!B25", [[_fmt_pct(probs.get("S4_multipolar", probs.get("S4_fiscal_dominance", 0.15)) * 100, 0)]])

    # ============================================================
    # SCORING TAB (regime detection outputs)
    # ============================================================

    def write_scoring(self, validated_data, g7_status):
        """
        SCORING tab — write live indicator values into regime detection inputs.
        Rows 4-14 (indicator values), column B.
        Active Regime Output section.
        """
        print("  [Display] SCORING...")

        fred = validated_data.get("fred", {})
        yf = validated_data.get("yfinance", {})

        def _yf_val(ticker):
            entry = yf.get(ticker)
            if isinstance(entry, dict) and entry.get("close") is not None:
                return _fmt(entry["close"])
            return "—"

        def _fred_val(series):
            entry = fred.get(series)
            if isinstance(entry, dict) and entry.get("value") is not None:
                return _fmt(entry["value"], 2)
            return "—"

        # Row 4: VIX
        self._write("SCORING!B4", [[_yf_val("^VIX")]])
        # Row 5: VIX 3M — not available
        self._write("SCORING!B5", [["—"]])
        # Row 6: HY Spread
        self._write("SCORING!B6", [[_fred_val("BAMLH0A0HYM2")]])
        # Row 7: GPR Index
        gpr = validated_data.get("gpr")
        gpr_val = gpr.get("gpr_global") if isinstance(gpr, dict) else None
        self._write("SCORING!B7", [[_fmt(gpr_val, 0) if gpr_val else "—"]])
        # Row 8: 2Y10Y Spread
        self._write("SCORING!B8", [[_fred_val("T10Y2Y")]])
        # Row 9: Credit Stress IG OAS — not in our FRED series
        self._write("SCORING!B9", [["—"]])
        # Row 10: Gold 1M %
        gold = yf.get("GC=F")
        gold_pct = gold.get("pct_change_1m") if isinstance(gold, dict) else None
        self._write("SCORING!B10", [[_fmt_pct(gold_pct) if gold_pct else "—"]])
        # Row 11: DXY
        self._write("SCORING!B11", [[_yf_val("DX-Y.NYB")]])
        # Row 12: V16 State — placeholder
        self._write("SCORING!B12", [["—"]])
        # Row 13: MOVE Index — not in our data
        self._write("SCORING!B13", [["—"]])
        # Row 14: NFCI
        self._write("SCORING!B14", [[_fred_val("STLFSI4")]])
        # Row 15: ANFCI — not separate series
        self._write("SCORING!B15", [["—"]])

    # ============================================================
    # MASTER WRITE METHOD
    # ============================================================

    def write_all(self, scoring_result, overlays, g7_status, scenario_result,
                  validated_data, freshness_by_source, collection_errors):
        """
        Write all 11 layout tabs.

        Args:
            scoring_result: Phase 3 full result
            overlays: Phase 4 full result
            g7_status: Phase 5 result
            scenario_result: Phase 6 result
            validated_data: Phase 2 validated data
            freshness_by_source: Phase 2 freshness tags
            collection_errors: Phase 1 errors
        """
        scores = scoring_result.get("scores", {})
        momenta = scoring_result.get("momenta", {})
        power_scores = scoring_result.get("power_scores", {})
        gap_data = scoring_result.get("gap_data", {})

        try:
            self.write_dashboard(power_scores, gap_data, overlays, g7_status, scenario_result)
        except Exception as e:
            print(f"  [Display] DASHBOARD ERROR: {e}")

        try:
            self.write_power_scores(scores, momenta, power_scores)
        except Exception as e:
            print(f"  [Display] POWER_SCORES ERROR: {e}")

        try:
            self.write_structural(validated_data, scores)
        except Exception as e:
            print(f"  [Display] STRUCTURAL ERROR: {e}")

        try:
            self.write_financial(validated_data, overlays)
        except Exception as e:
            print(f"  [Display] FINANCIAL ERROR: {e}")

        try:
            self.write_leading(scores, momenta, overlays, validated_data)
        except Exception as e:
            print(f"  [Display] LEADING ERROR: {e}")

        try:
            self.write_feedback_loops(overlays)
        except Exception as e:
            print(f"  [Display] FEEDBACK_LOOPS ERROR: {e}")

        try:
            self.write_scenarios(scenario_result)
        except Exception as e:
            print(f"  [Display] SCENARIOS ERROR: {e}")

        try:
            self.write_scoring(validated_data, g7_status)
        except Exception as e:
            print(f"  [Display] SCORING ERROR: {e}")

        try:
            self.write_sources(freshness_by_source, collection_errors)
        except Exception as e:
            print(f"  [Display] SOURCES ERROR: {e}")

        try:
            self.write_history(power_scores, scenario_result)
        except Exception as e:
            print(f"  [Display] HISTORY ERROR: {e}")

        try:
            self.write_universe_map(overlays)
        except Exception as e:
            print(f"  [Display] UNIVERSE_MAP ERROR: {e}")

        print("  [Display] All 11 layout tabs written")
