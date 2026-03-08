"""
step_0s_g7_monitor/display_writer.py
Phase 10 Extension: Display Writer for 11 Layout Tabs

Uses batchUpdate to write all cells per tab in ONE API call.
Total API writes: ~11 (one per tab) instead of ~100+.
Google Sheets limit: 60 writes/min/user.

Row numbers verified from actual Google Sheets CSV exports 2026-03-08.
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
        """
        Write multiple ranges in ONE API call.
        data_list = [("SHEET!A1:B2", [[val, val], [val, val]]), ...]
        """
        if not data_list:
            return
        body = {
            "valueInputOption": "USER_ENTERED",
            "data": [
                {"range": rng, "values": vals}
                for rng, vals in data_list
            ],
        }
        try:
            self.service.spreadsheets().values().batchUpdate(
                spreadsheetId=self.sheet_id, body=body
            ).execute()
        except Exception as e:
            print(f"  [DisplayWriter] batchUpdate failed: {e}")

    def _extract_probs(self, scenario_result):
        if not scenario_result or not isinstance(scenario_result, dict):
            return [0.40, 0.20, 0.25, 0.15]
        probs = None
        ct = scenario_result.get("current_thesis")
        if isinstance(ct, dict):
            probs = ct.get("probabilities")
        if not probs:
            th = scenario_result.get("thesis")
            if isinstance(th, dict):
                probs = th.get("probabilities")
        if not probs or not isinstance(probs, dict):
            return [0.40, 0.20, 0.25, 0.15]
        return [
            probs.get("S1_managed_decline", probs.get("S1_status_quo", 0.40)),
            probs.get("S2_conflict", probs.get("S2_bifurcation", 0.20)),
            probs.get("S3_us_renewal", probs.get("S3_fragmentation", 0.25)),
            probs.get("S4_multipolar", probs.get("S4_fiscal_dominance", 0.15)),
        ]

    # ============================================================
    # DASHBOARD
    # ============================================================

    def write_dashboard(self, power_scores, gap_data, overlays, g7_status, scenario_result):
        print("  [Display] DASHBOARD...")
        d = []

        # Power Scores rows 5-11, cols B-G
        ps_rows = []
        for region in REGIONS:
            ps = power_scores.get(region, {})
            score = ps.get("score")
            mom = ps.get("momentum", 0)
            trend = "▲" if mom > 0.3 else "▼" if mom < -0.3 else "►"
            ps_rows.append([_fmt(score), trend, _fmt_delta(mom),
                           CYCLE_PHASES.get(region, "—"), KEY_RISKS.get(region, "—"), "—"])
        d.append(("DASHBOARD!B5:G11", ps_rows))

        # Differential rows 14-17, col D
        gap = gap_data.get("gap", 0)
        thuc = "LOW"
        for loop in overlays.get("feedback_loops", []):
            if loop.get("loop_id") == "thucydides_trap":
                thuc = "HIGH" if loop["status"] == "ACTIVE" else "MEDIUM" if loop["status"] == "LATENT" else "LOW"
                break
        d.append(("DASHBOARD!D14:D17", [
            [_fmt(gap)], [gap_data.get("trend", "STABLE")], [thuc],
            ["N/A" if gap_data.get("trend") != "CLOSING" else "Monitor"]]))

        # Scenarios rows 21-24, col B
        probs = self._extract_probs(scenario_result)
        d.append(("DASHBOARD!B21:B24", [[_fmt_pct(p * 100, 0)] for p in probs]))

        # Schedule rows 39-42, col D
        d.append(("DASHBOARD!D39:D42", [
            [_now_iso()], ["Next scheduled run"], ["Weekly + Quarterly"],
            [g7_status.get("attention_flag", "NONE")]]))

        self._batch_write(d)

    # ============================================================
    # POWER_SCORES
    # ============================================================

    def write_power_scores(self, scores, momenta, power_scores):
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

        self._batch_write(d)

    # ============================================================
    # STRUCTURAL
    # ============================================================

    def write_structural(self, validated_data, scores):
        print("  [Display] STRUCTURAL...")
        imf = validated_data.get("imf_weo", {})
        wb = validated_data.get("worldbank", {})
        now = _today()
        d = []

        def _iv(ind, r):
            e = imf.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        def _wv(ind, r):
            e = wb.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        # D1 rows 5-9
        d.append(("STRUCTURAL!D5:J5", [[_iv("NGDPD", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L5", [[now]]))
        d.append(("STRUCTURAL!D6:J6", [[_iv("NGDP_RPCH", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L6", [[now]]))
        d.append(("STRUCTURAL!D7:J9", [["—"] * 7] * 3))

        # D2 rows 13-18
        d.append(("STRUCTURAL!D13:J13", [["—"] * 7]))
        d.append(("STRUCTURAL!D14:J14", [[_wv("SP.POP.DPND", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L14", [[now]]))
        d.append(("STRUCTURAL!D15:J15", [["—"] * 7]))
        d.append(("STRUCTURAL!D16:J16", [[_wv("SP.DYN.TFRT.IN", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L16", [[now]]))
        d.append(("STRUCTURAL!D17:J18", [["—"] * 7] * 2))

        # D3 rows 22-28
        d.append(("STRUCTURAL!D22:J22", [[_wv("GB.XPD.RSDV.GD.ZS", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L22", [[now]]))
        d.append(("STRUCTURAL!D23:J28", [["—"] * 7] * 6))

        # D4 rows 32-36
        d.append(("STRUCTURAL!D32:J36", [["—"] * 7] * 5))

        # D5 rows 40-44
        d.append(("STRUCTURAL!D40:J40", [[_wv("MS.MIL.XPND.GD.ZS", r) for r in REGIONS]]))
        d.append(("STRUCTURAL!L40", [[now]]))
        d.append(("STRUCTURAL!D41:J44", [["—"] * 7] * 4))

        self._batch_write(d)

    # ============================================================
    # FINANCIAL
    # ============================================================

    def write_financial(self, validated_data, overlays):
        print("  [Display] FINANCIAL...")
        fred = validated_data.get("fred", {})
        imf = validated_data.get("imf_weo", {})
        cofer = validated_data.get("imf_cofer", {})
        yf = validated_data.get("yfinance", {})
        now = _today()
        d = []

        def _iv(ind, r):
            e = imf.get(f"{ind}_{r}")
            return _fmt(e["value"]) if isinstance(e, dict) and e.get("value") is not None else "—"

        # D6 Row 5: Debt/GDP
        d.append(("FINANCIAL!D5:J5", [[_iv("GGXWDG_NGDP", r) for r in REGIONS]]))
        d.append(("FINANCIAL!L5", [[now]]))
        # Row 6: placeholder
        d.append(("FINANCIAL!D6:J6", [["—"] * 7]))
        # Row 7: ITR
        itr_row = ["—"] * 7
        i_d = fred.get("A091RC1Q027SBEA")
        r_d = fred.get("FGRECPT")
        if (isinstance(i_d, dict) and i_d.get("value") is not None
                and isinstance(r_d, dict) and r_d.get("value") is not None and r_d["value"] > 0):
            itr_row[0] = _fmt(i_d["value"] / r_d["value"] * 100)
        d.append(("FINANCIAL!D7:J7", [itr_row]))
        d.append(("FINANCIAL!L7", [[now]]))
        # Row 8: Deficit
        def_row = ["—"] * 7
        deficit = fred.get("FYFSGDA188S")
        if isinstance(deficit, dict) and deficit.get("value") is not None:
            def_row[0] = _fmt(deficit["value"])
        d.append(("FINANCIAL!D8:J8", [def_row]))
        # Rows 9-10: placeholder
        d.append(("FINANCIAL!D9:J10", [["—"] * 7] * 2))
        # Row 11: Real Rate
        rr_row = ["—"] * 7
        dgs = fred.get("DGS10")
        inf_u = imf.get("PCPIPCH_USA")
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
        # Row 13: placeholder
        d.append(("FINANCIAL!D13:J13", [["—"] * 7]))

        # D7 Row 15: COFER
        c_row = ["—"] * 7
        c_row[0] = _fmt(cofer.get("USD_share"))
        c_row[1] = _fmt(cofer.get("CNY_share"))
        c_row[2] = _fmt(cofer.get("EUR_share"))
        d.append(("FINANCIAL!D15:J15", [c_row]))
        d.append(("FINANCIAL!L15", [[now]]))
        # Rows 16-20: placeholder
        d.append(("FINANCIAL!D16:J20", [["—"] * 7] * 5))
        # Row 21: DXY
        dxy_row = ["—"] * 7
        dxy = yf.get("DX-Y.NYB")
        if isinstance(dxy, dict) and dxy.get("close") is not None:
            dxy_row[0] = _fmt(dxy["close"])
        d.append(("FINANCIAL!D21:J21", [dxy_row]))
        d.append(("FINANCIAL!L21", [[now]]))

        # D8 rows 25-31: placeholder
        d.append(("FINANCIAL!D25:J31", [["—"] * 7] * 7))

        self._batch_write(d)

    # ============================================================
    # LEADING
    # ============================================================

    def write_leading(self, scores, momenta, overlays, validated_data):
        print("  [Display] LEADING...")
        now = _today()
        d = []

        # D9 rows 5-9
        d.append(("LEADING!D5:J9", [["—"] * 7] * 5))
        # D10 rows 13-18
        d.append(("LEADING!D13:J18", [["—"] * 7] * 6))
        # D11 rows 22-25
        d.append(("LEADING!D22:J25", [["—"] * 7] * 4))
        # Row 26: GPR
        gpr_row = ["—"] * 7
        gpr = overlays.get("gpr_index_current")
        if gpr is not None:
            gpr_row[0] = _fmt(gpr, 0)
        d.append(("LEADING!D26:J26", [gpr_row]))
        d.append(("LEADING!L26", [[now]]))
        # Row 27
        d.append(("LEADING!D27:J27", [["—"] * 7]))

        # D12 rows 31-35
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

        # Neg spirals rows 5-10
        neg_rows = []
        for lid in ["debt_demographics", "currency_fiscal", "thucydides_trap",
                     "energy_conflict", "financial_contagion", "social_political"]:
            sev = ms.get(lid, 0)
            neg_rows.append([_fmt(sev) if sev > 0 else "INACTIVE",
                           "FAST" if sev > 5 else "SLOW" if sev > 0 else "—"])
        d.append(("FEEDBACK_LOOPS!D5:E10", neg_rows))

        # Pos spirals rows 14-17
        d.append(("FEEDBACK_LOOPS!D14:E17", [["—", "—"]] * 4))

        # Interaction matrix rows 21-27
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
        self._batch_write([
            ("SCENARIOS!B4", [[_fmt_pct(probs[0] * 100, 0)]]),
            ("SCENARIOS!B13", [[_fmt_pct(probs[1] * 100, 0)]]),
            ("SCENARIOS!B22", [[_fmt_pct(probs[2] * 100, 0)]]),
            ("SCENARIOS!B31", [[_fmt_pct(probs[3] * 100, 0)]]),
        ])

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
            [_yc("^VIX")], ["—"], [_fv("BAMLH0A0HYM2")],
            [_fmt(gv, 0) if gv else "—"], [_fv("T10Y2Y")], ["—"],
            [_fmt_pct(gp) if gp else "—"], [_yc("DX-Y.NYB")],
            ["—"], ["—"], [_fv("STLFSI4")], ["—"],
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
