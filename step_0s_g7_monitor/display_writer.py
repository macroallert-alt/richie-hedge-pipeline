"""
step_0s_g7_monitor/display_writer.py
Phase 10 Extension: Display Writer for 11 Layout Tabs

Writes computed data from Phase 3 (Scoring) and Phase 4 (Overlays)
into the 11 pre-formatted layout tabs with Bloomberg Terminal styling.

The layout tabs have FIXED structure (headers, section dividers, label columns).
This module ONLY writes into the data cells (replacing '—' placeholders).
It does NOT touch headers, labels, section titles, or formatting.

Row numbers verified from actual Google Sheets CSV exports 2026-03-08.

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
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _fmt_pct(val, decimals=1):
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}%"
    return str(val)


def _fmt_delta(val, decimals=2):
    if val is None or val == 0:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:+.{decimals}f}"
    return str(val)


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _quarter_label():
    now = datetime.now(timezone.utc)
    q = (now.month - 1) // 3 + 1
    return f"Q{q} {now.year}"


# ============================================================
# DISPLAY WRITER CLASS
# ============================================================

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
    # DASHBOARD — rows 5-11 scores, 14-17 diff, 21-24 scenarios, 39-42 schedule
    # ============================================================

    def write_dashboard(self, power_scores, gap_data, overlays, g7_status, scenario_result):
        print("  [Display] DASHBOARD...")

        ps_rows = []
        for region in REGIONS:
            ps = power_scores.get(region, {})
            score = ps.get("score")
            momentum = ps.get("momentum", 0)
            trend = "▲" if momentum > 0.3 else "▼" if momentum < -0.3 else "►"
            ps_rows.append([
                _fmt(score), trend, _fmt_delta(momentum),
                CYCLE_PHASES.get(region, "—"),
                KEY_RISKS.get(region, "—"),
                "—",
            ])
        self._write("DASHBOARD!B5:G11", ps_rows)

        gap = gap_data.get("gap", 0)
        thuc_risk = "LOW"
        for loop in overlays.get("feedback_loops", []):
            if loop.get("loop_id") == "thucydides_trap":
                thuc_risk = "HIGH" if loop["status"] == "ACTIVE" else "MEDIUM" if loop["status"] == "LATENT" else "LOW"
                break
        self._write("DASHBOARD!D14:D17", [
            [_fmt(gap)], [gap_data.get("trend", "STABLE")], [thuc_risk],
            ["N/A" if gap_data.get("trend") != "CLOSING" else "Monitor"],
        ])

        probs = self._extract_probs(scenario_result)
        self._write("DASHBOARD!B21:B24", [[_fmt_pct(p * 100, 0)] for p in probs])

        self._write("DASHBOARD!D39:D42", [
            [_now_iso()], ["Next scheduled run"], ["Weekly + Quarterly"],
            [g7_status.get("attention_flag", "NONE")],
        ])

    # ============================================================
    # POWER_SCORES — D1-5 rows 7-11, D6-8 rows 15-17, D9-12 rows 21-24
    # Total row 28, Trend row 29, Phase row 30. Cols C-I.
    # ============================================================

    def write_power_scores(self, scores, momenta, power_scores):
        print("  [Display] POWER_SCORES...")

        def _rv(dim):
            return [_fmt(scores.get(dim, {}).get(r)) for r in REGIONS]

        self._write("POWER_SCORES!C7:I11", [_rv("D1_economic"), _rv("D2_demographics"),
                     _rv("D3_technology"), _rv("D4_energy"), _rv("D5_military")])
        self._write("POWER_SCORES!C15:I17", [_rv("D6_fiscal"), _rv("D7_currency"), _rv("D8_capital_mkt")])
        self._write("POWER_SCORES!C21:I24", [_rv("D9_flows"), _rv("D10_social"),
                     _rv("D11_geopolitical"), _rv("D12_feedback")])

        self._write("POWER_SCORES!C28:I28", [[_fmt(power_scores.get(r, {}).get("score")) for r in REGIONS]])
        self._write("POWER_SCORES!C29:I29", [[_fmt_delta(power_scores.get(r, {}).get("momentum", 0)) for r in REGIONS]])
        self._write("POWER_SCORES!C30:I30", [[CYCLE_PHASES.get(r, "—") for r in REGIONS]])

    # ============================================================
    # STRUCTURAL — D1 rows 5-9, D2 rows 13-18, D3 rows 22-28,
    #              D4 rows 32-36, D5 rows 40-44. Data cols D-J, updated col L.
    # ============================================================

    def write_structural(self, validated_data, scores):
        print("  [Display] STRUCTURAL...")

        imf = validated_data.get("imf_weo", {})
        wb = validated_data.get("worldbank", {})
        now = _today()

        def _iv(ind, r):
            e = imf.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        def _wv(ind, r):
            e = wb.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        # D1: rows 5-9
        self._write("STRUCTURAL!D5:J5", [[_iv("NGDPD", r) for r in REGIONS]])
        self._write("STRUCTURAL!L5", [[now]])
        self._write("STRUCTURAL!D6:J6", [[_iv("NGDP_RPCH", r) for r in REGIONS]])
        self._write("STRUCTURAL!L6", [[now]])
        for row in [7, 8, 9]:
            self._write(f"STRUCTURAL!D{row}:J{row}", [["—"] * 7])

        # D2: rows 13-18
        self._write("STRUCTURAL!D13:J13", [["—"] * 7])
        self._write("STRUCTURAL!D14:J14", [[_wv("SP.POP.DPND", r) for r in REGIONS]])
        self._write("STRUCTURAL!L14", [[now]])
        self._write("STRUCTURAL!D15:J15", [["—"] * 7])
        self._write("STRUCTURAL!D16:J16", [[_wv("SP.DYN.TFRT.IN", r) for r in REGIONS]])
        self._write("STRUCTURAL!L16", [[now]])
        for row in [17, 18]:
            self._write(f"STRUCTURAL!D{row}:J{row}", [["—"] * 7])

        # D3: rows 22-28
        self._write("STRUCTURAL!D22:J22", [[_wv("GB.XPD.RSDV.GD.ZS", r) for r in REGIONS]])
        self._write("STRUCTURAL!L22", [[now]])
        for row in range(23, 29):
            self._write(f"STRUCTURAL!D{row}:J{row}", [["—"] * 7])

        # D4: rows 32-36
        for row in range(32, 37):
            self._write(f"STRUCTURAL!D{row}:J{row}", [["—"] * 7])

        # D5: rows 40-44
        self._write("STRUCTURAL!D40:J40", [[_wv("MS.MIL.XPND.GD.ZS", r) for r in REGIONS]])
        self._write("STRUCTURAL!L40", [[now]])
        for row in [41, 42, 43, 44]:
            self._write(f"STRUCTURAL!D{row}:J{row}", [["—"] * 7])

    # ============================================================
    # FINANCIAL — D6 rows 5-13, D7 rows 15-21, D8 rows 25-31
    # Data cols D-J, updated col L.
    # ============================================================

    def write_financial(self, validated_data, overlays):
        print("  [Display] FINANCIAL...")

        fred = validated_data.get("fred", {})
        imf = validated_data.get("imf_weo", {})
        cofer = validated_data.get("imf_cofer", {})
        yf = validated_data.get("yfinance", {})
        now = _today()

        def _iv(ind, r):
            e = imf.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        def _fv(s):
            e = fred.get(s)
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        # D6 Row 5: Debt/GDP
        self._write("FINANCIAL!D5:J5", [[_iv("GGXWDG_NGDP", r) for r in REGIONS]])
        self._write("FINANCIAL!L5", [[now]])
        # Row 6: placeholder
        self._write("FINANCIAL!D6:J6", [["—"] * 7])
        # Row 7: Interest/Revenue — USA
        itr_row = ["—"] * 7
        i_d = fred.get("A091RC1Q027SBEA")
        r_d = fred.get("FGRECPT")
        if (isinstance(i_d, dict) and i_d.get("value") is not None
                and isinstance(r_d, dict) and r_d.get("value") is not None and r_d["value"] > 0):
            itr_row[0] = _fmt(i_d["value"] / r_d["value"] * 100)
        self._write("FINANCIAL!D7:J7", [itr_row])
        self._write("FINANCIAL!L7", [[now]])
        # Row 8: Deficit
        d_row = ["—"] * 7
        deficit = fred.get("FYFSGDA188S")
        if isinstance(deficit, dict) and deficit.get("value") is not None:
            d_row[0] = _fmt(deficit["value"])
        self._write("FINANCIAL!D8:J8", [d_row])
        # Rows 9-10: placeholder
        self._write("FINANCIAL!D9:J10", [["—"] * 7, ["—"] * 7])
        # Row 11: Real Rate
        rr_row = ["—"] * 7
        dgs = fred.get("DGS10")
        inf_u = imf.get("PCPIPCH_USA")
        if (isinstance(dgs, dict) and dgs.get("value") is not None
                and isinstance(inf_u, dict) and inf_u.get("value") is not None):
            rr_row[0] = _fmt(dgs["value"] - inf_u["value"])
        self._write("FINANCIAL!D11:J11", [rr_row])
        # Row 12: NFCI
        nfci_row = ["—"] * 7
        st = fred.get("STLFSI4")
        if isinstance(st, dict) and st.get("value") is not None:
            nfci_row[0] = _fmt(st["value"], 2)
        self._write("FINANCIAL!D12:J12", [nfci_row])
        # Row 13: ANFCI placeholder
        self._write("FINANCIAL!D13:J13", [["—"] * 7])

        # D7 Row 15: COFER
        c_row = ["—"] * 7
        c_row[0] = _fmt(cofer.get("USD_share"))
        c_row[1] = _fmt(cofer.get("CNY_share"))
        c_row[2] = _fmt(cofer.get("EUR_share"))
        self._write("FINANCIAL!D15:J15", [c_row])
        self._write("FINANCIAL!L15", [[now]])
        # Rows 16-20: placeholder
        for row in range(16, 21):
            self._write(f"FINANCIAL!D{row}:J{row}", [["—"] * 7])
        # Row 21: DXY
        dxy_row = ["—"] * 7
        dxy = yf.get("DX-Y.NYB")
        if isinstance(dxy, dict) and dxy.get("close") is not None:
            dxy_row[0] = _fmt(dxy["close"])
        self._write("FINANCIAL!D21:J21", [dxy_row])
        self._write("FINANCIAL!L21", [[now]])

        # D8 rows 25-31: placeholder
        for row in range(25, 32):
            self._write(f"FINANCIAL!D{row}:J{row}", [["—"] * 7])

    # ============================================================
    # LEADING — D9 rows 5-9, D10 rows 13-18, D11 rows 22-27, D12 rows 31-35
    # ============================================================

    def write_leading(self, scores, momenta, overlays, validated_data):
        print("  [Display] LEADING...")
        now = _today()

        # D9: rows 5-9 placeholder
        for row in range(5, 10):
            self._write(f"LEADING!D{row}:J{row}", [["—"] * 7])

        # D10: rows 13-18 placeholder
        for row in range(13, 19):
            self._write(f"LEADING!D{row}:J{row}", [["—"] * 7])

        # D11: rows 22-27
        for row in range(22, 26):
            self._write(f"LEADING!D{row}:J{row}", [["—"] * 7])
        # Row 26: GPR
        gpr_row = ["—"] * 7
        gpr = overlays.get("gpr_index_current")
        if gpr is not None:
            gpr_row[0] = _fmt(gpr, 0)
        self._write("LEADING!D26:J26", [gpr_row])
        self._write("LEADING!L26", [[now]])
        self._write("LEADING!D27:J27", [["—"] * 7])

        # D12: rows 31-35
        feedback_loops = overlays.get("feedback_loops", [])
        ls = {}
        for loop in feedback_loops:
            lid = loop.get("loop_id", "")
            r = loop.get("region", "")
            s = loop.get("severity", 0)
            if lid not in ls:
                ls[lid] = {}
            ls[lid][r] = s

        def _lr(lid):
            return [_fmt(ls.get(lid, {}).get(r, 0)) if ls.get(lid, {}).get(r, 0) > 0 else "—" for r in REGIONS]

        self._write("LEADING!D31:J31", [_lr("debt_demographics")])
        self._write("LEADING!D32:J32", [_lr("energy_conflict")])
        self._write("LEADING!D33:J33", [_lr("tech_security")])
        self._write("LEADING!D34:J34", [_lr("currency_fiscal")])

        net = []
        for r in REGIONS:
            t = sum(ls.get(lid, {}).get(r, 0) for lid in ls)
            net.append(_fmt(t) if t > 0 else "—")
        self._write("LEADING!D35:J35", [net])
        self._write("LEADING!L31:L35", [[now]] * 5)

    # ============================================================
    # FEEDBACK_LOOPS — Neg rows 5-10 cols D-E, Pos rows 14-17,
    # Interaction Matrix rows 21-27 cols B-L
    # ============================================================

    def write_feedback_loops(self, overlays):
        print("  [Display] FEEDBACK_LOOPS...")

        feedback_loops = overlays.get("feedback_loops", [])
        ms = {}
        mst = {}
        for loop in feedback_loops:
            lid = loop.get("loop_id", "")
            s = loop.get("severity", 0)
            if lid not in ms or s > ms[lid]:
                ms[lid] = s
                mst[lid] = loop.get("status", "INACTIVE")

        neg_map = [
            ("debt_demographics", 5), ("currency_fiscal", 6), ("thucydides_trap", 7),
            ("energy_conflict", 8), ("financial_contagion", 9), ("social_political", 10),
        ]
        for lid, row in neg_map:
            sev = ms.get(lid, 0)
            sev_str = _fmt(sev) if sev > 0 else "INACTIVE"
            speed = "FAST" if sev > 5 else "SLOW" if sev > 0 else "—"
            self._write(f"FEEDBACK_LOOPS!D{row}:E{row}", [[sev_str, speed]])

        for row in range(14, 18):
            self._write(f"FEEDBACK_LOOPS!D{row}:E{row}", [["—", "—"]])

        # Interaction Matrix
        lbr = {}
        for loop in feedback_loops:
            r = loop.get("region", "")
            lid = loop.get("loop_id", "")
            s = loop.get("severity", 0)
            if r not in lbr:
                lbr[r] = {}
            lbr[r][lid] = s

        neg_cols = ["debt_demographics", "currency_fiscal", "thucydides_trap",
                    "energy_conflict", "financial_contagion", "social_political"]

        for i, region in enumerate(REGIONS):
            rl = lbr.get(region, {})
            cells = []
            net = 0
            for lid in neg_cols:
                sv = rl.get(lid, 0)
                cells.append(_fmt(sv) if sv > 0 else "—")
                net += sv
            cells.extend(["—"] * 4)  # positive loops placeholder
            cells.append(_fmt(net) if net > 0 else "—")
            self._write(f"FEEDBACK_LOOPS!B{21 + i}:L{21 + i}", [cells])

    # ============================================================
    # SOURCES — API rows 5-12 col H, Semi rows 15-24 col H
    # ============================================================

    def write_sources(self, freshness_by_source, collection_errors):
        print("  [Display] SOURCES...")
        for src, row in {"fred": 5, "imf_weo": 6, "imf_cofer": 7, "worldbank": 8,
                         "un_pop": 9, "yfinance": 10}.items():
            f = freshness_by_source.get(src, "UNAVAILABLE")
            self._write(f"SOURCES!H{row}", [["LIVE" if f in ("FRESH", "RECENT") else "STALE" if f == "STALE" else "DOWN"]])
        for src, row in {"acled": 21, "gpr": 22}.items():
            f = freshness_by_source.get(src, "UNAVAILABLE")
            self._write(f"SOURCES!H{row}", [["LIVE" if f in ("FRESH", "RECENT") else "DOWN"]])

    # ============================================================
    # HISTORY — PS rows 5-12 cols B-H, Scenarios rows 16-23 cols B-E
    # ============================================================

    def write_history(self, power_scores, scenario_result):
        print("  [Display] HISTORY...")
        quarter = _quarter_label()
        qr = {"Q1 2026": 5, "Q2 2026": 6, "Q3 2026": 7, "Q4 2026": 8,
              "Q1 2027": 9, "Q2 2027": 10, "Q3 2027": 11, "Q4 2027": 12}
        sr = {"Q1 2026": 16, "Q2 2026": 17, "Q3 2026": 18, "Q4 2026": 19,
              "Q1 2027": 20, "Q2 2027": 21, "Q3 2027": 22, "Q4 2027": 23}

        row = qr.get(quarter)
        if row:
            self._write(f"HISTORY!B{row}:H{row}", [[_fmt(power_scores.get(r, {}).get("score")) for r in REGIONS]])

        sc_row = sr.get(quarter)
        if sc_row:
            probs = self._extract_probs(scenario_result)
            self._write(f"HISTORY!B{sc_row}:E{sc_row}", [[_fmt_pct(p * 100, 0) for p in probs]])

    # ============================================================
    # UNIVERSE_MAP — col G for tilt hints
    # ============================================================

    def write_universe_map(self, overlays):
        print("  [Display] UNIVERSE_MAP...")
        att = overlays.get("attractiveness", [])
        if not att:
            return
        rbr = {e["region"]: e["rank"] for e in att}
        self._write("UNIVERSE_MAP!G5", [[f"Core (#{rbr.get('USA', '?')})"]])
        self._write("UNIVERSE_MAP!G17", [[f"Tilt: #{rbr.get('INDIA', '?')} India"]])
        self._write("UNIVERSE_MAP!G18", [[f"#{rbr.get('EU', '?')}"]])

    # ============================================================
    # SCENARIOS — Prob in col B: ScA row 4, ScB row 13, ScC row 22, ScD row 31
    # ============================================================

    def write_scenarios(self, scenario_result):
        print("  [Display] SCENARIOS...")
        probs = self._extract_probs(scenario_result)
        self._write("SCENARIOS!B4", [[_fmt_pct(probs[0] * 100, 0)]])
        self._write("SCENARIOS!B13", [[_fmt_pct(probs[1] * 100, 0)]])
        self._write("SCENARIOS!B22", [[_fmt_pct(probs[2] * 100, 0)]])
        self._write("SCENARIOS!B31", [[_fmt_pct(probs[3] * 100, 0)]])

    # ============================================================
    # SCORING — Live inputs rows 5-16 col B
    # ============================================================

    def write_scoring(self, validated_data, g7_status):
        print("  [Display] SCORING...")
        fred = validated_data.get("fred", {})
        yf = validated_data.get("yfinance", {})
        gpr_data = validated_data.get("gpr")

        def _yc(t):
            e = yf.get(t)
            return _fmt(e["close"]) if isinstance(e, dict) and e.get("close") is not None else "—"

        def _fv(s, d=2):
            e = fred.get(s)
            return _fmt(e["value"], d) if isinstance(e, dict) and e.get("value") is not None else "—"

        self._write("SCORING!B5", [[_yc("^VIX")]])
        self._write("SCORING!B6", [["—"]])
        self._write("SCORING!B7", [[_fv("BAMLH0A0HYM2")]])
        gv = gpr_data.get("gpr_global") if isinstance(gpr_data, dict) else None
        self._write("SCORING!B8", [[_fmt(gv, 0) if gv else "—"]])
        self._write("SCORING!B9", [[_fv("T10Y2Y")]])
        self._write("SCORING!B10", [["—"]])
        gold = yf.get("GC=F")
        gp = gold.get("pct_change_1m") if isinstance(gold, dict) else None
        self._write("SCORING!B11", [[_fmt_pct(gp) if gp else "—"]])
        self._write("SCORING!B12", [[_yc("DX-Y.NYB")]])
        self._write("SCORING!B13", [["—"]])
        self._write("SCORING!B14", [["—"]])
        self._write("SCORING!B15", [[_fv("STLFSI4")]])
        self._write("SCORING!B16", [["—"]])

    # ============================================================
    # HELPER
    # ============================================================

    def _extract_probs(self, scenario_result):
        scenario = scenario_result or {}
        probs = scenario.get("current_thesis", {}).get("probabilities", {})
        if not probs:
            probs = scenario.get("thesis", {}).get("probabilities", {})
        return [
            probs.get("S1_managed_decline", probs.get("S1_status_quo", 0.40)),
            probs.get("S2_conflict", probs.get("S2_bifurcation", 0.20)),
            probs.get("S3_us_renewal", probs.get("S3_fragmentation", 0.25)),
            probs.get("S4_multipolar", probs.get("S4_fiscal_dominance", 0.15)),
        ]

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
            ("POWER_SCORES", self.write_power_scores, (scores, momenta, power_scores)),
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
