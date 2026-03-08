"""
step_0s_g7_monitor/sheet_writer.py
Google Sheets Interface fuer G7 World Order Monitor

Pattern: Identisch mit step3_risk_officer/main.py
  - googleapiclient.discovery (nicht gspread)
  - GCP_SA_KEY / GOOGLE_CREDENTIALS aus Environment
  - Sheets API v4: values().get(), values().update(), values().append(), values().clear()

G7 Sheet ID: 1TVl-GNYxK7Sppn8Tv8lSlMVgFfCwr8WslWSwABpOybk

Tabs die G7 SCHREIBT:
  G7_STATUS              — Woechentlich ueberschrieben (clear+write)
  G7_THESIS              — Quartalsweise/Interim ueberschrieben (clear+write)
  G7_NARRATIVE           — Woechentlich ueberschrieben (clear+write)
  G7_THESIS_HISTORY      — Append, max 12 Eintraege
  G7_POWER_SCORE_HISTORY — Append, woechentliche Snapshots
  G7_RUN_LOG             — Append, Metadata pro Run
  G7_DATA_CACHE          — Write-Through Cache (clear+write)

Tabs die G7 LIEST:
  G7_STATUS              — Vorherige Woche (fuer Drift-Tracking)
  G7_THESIS              — Vorheriges Quartal
  G7_THESIS_HISTORY      — Drift-Tracking
  G7_POWER_SCORE_HISTORY — Vorherige Scores (fuer Momentum)
  G7_OPERATOR_OVERRIDES  — Manuelle Szenario-Overrides
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timezone

REGIONS = ["USA", "CHINA", "EU", "INDIA", "JP_KR_TW", "GULF", "REST_EM"]


class G7SheetWriter:
    """
    Google Sheets read/write interface for G7 World Order Monitor.
    Uses googleapiclient.discovery — same pattern as step3_risk_officer.
    """

    def __init__(self, sheet_id):
        self.sheet_id = sheet_id
        self._sheets = None

    # ═══════════════════════════════════════════════════════
    # CONNECTION
    # ═══════════════════════════════════════════════════════

    def connect(self):
        """Connect to Google Sheets API. Returns True on success."""
        try:
            creds = self._get_credentials()
            if not creds:
                print("[G7SheetWriter] No credentials found")
                return False

            from googleapiclient.discovery import build
            service = build("sheets", "v4", credentials=creds)
            self._sheets = service.spreadsheets()
            print(f"[G7SheetWriter] Connected to Sheet {self.sheet_id}")
            return True
        except Exception as e:
            print(f"[G7SheetWriter] Connection failed: {e}")
            return False

    def _get_credentials(self):
        """Create credentials from environment variable (same as step3)."""
        from google.oauth2.service_account import Credentials

        creds_json = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_CREDENTIALS")
        if not creds_json:
            print("[G7SheetWriter] No GCP_SA_KEY or GOOGLE_CREDENTIALS in env")
            return None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(creds_json)
            creds_path = f.name

        creds = Credentials.from_service_account_file(
            creds_path,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        os.unlink(creds_path)
        return creds

    # ═══════════════════════════════════════════════════════
    # LOW-LEVEL HELPERS (same as step3)
    # ═══════════════════════════════════════════════════════

    def _read_range(self, range_str):
        """Read range from Sheet. Returns list of lists."""
        try:
            result = self._sheets.values().get(
                spreadsheetId=self.sheet_id, range=range_str
            ).execute()
            return result.get("values", [])
        except Exception as e:
            print(f"[G7SheetWriter] Read failed {range_str}: {e}")
            return []

    def _write_range(self, range_str, values):
        """Write values to Sheet."""
        try:
            self._sheets.values().update(
                spreadsheetId=self.sheet_id,
                range=range_str,
                valueInputOption="RAW",
                body={"values": values},
            ).execute()
            return True
        except Exception as e:
            print(f"[G7SheetWriter] Write failed {range_str}: {e}")
            return False

    def _clear_and_write(self, range_str, values):
        """Clear range then write new values."""
        try:
            self._sheets.values().clear(
                spreadsheetId=self.sheet_id, range=range_str, body={}
            ).execute()
        except Exception:
            pass  # Tab may not exist yet — that is OK
        return self._write_range(range_str, values)

    def _append_row(self, tab_name, row):
        """Append a single row to a tab."""
        try:
            self._sheets.values().append(
                spreadsheetId=self.sheet_id,
                range=f"{tab_name}!A:Z",
                valueInputOption="RAW",
                body={"values": [row]},
            ).execute()
            return True
        except Exception as e:
            print(f"[G7SheetWriter] Append failed {tab_name}: {e}")
            return False

    def _ensure_header(self, tab_name, header):
        """Write header row if tab is empty."""
        existing = self._read_range(f"{tab_name}!A1:A1")
        if not existing:
            self._write_range(f"{tab_name}!A1", [header])

    # ═══════════════════════════════════════════════════════
    # WRITERS
    # ═══════════════════════════════════════════════════════

    def write_g7_status(self, status):
        """
        Write G7_STATUS tab — woechentlich ueberschrieben.
        Format: Key-Value pairs in columns A (key) and B (value).
        Konsument: Risk Officer, Dashboard Writer, Agent R.
        """
        rows = [
            ["G7_STATUS", ""],
            ["date", status.get("date", "")],
            ["g7_status", status.get("g7_status", "STABLE")],
            ["available", str(status.get("available", True))],
            ["last_update", status.get("last_update", "")],
            ["attention_flag", status.get("attention_flag", "NONE")],
            ["portfolio_relevance", status.get("portfolio_relevance") or ""],
            ["active_shifts", json.dumps(status.get("active_shifts", []))],
            ["status_changed", str(status.get("status_changed", False))],
            ["previous_status", status.get("previous_status", "STABLE")],
            [""],
            ["POWER SCORES", ""],
        ]

        # Power scores summary
        ps = status.get("power_scores_summary", {})
        for region in REGIONS:
            rows.append([f"power_score_{region}", str(ps.get(region, 0))])

        rows.append([""])
        rows.append(["POWER SCORE MOMENTA", ""])
        pm = status.get("power_score_momenta", {})
        for region in REGIONS:
            rows.append([f"momentum_{region}", str(pm.get(region, 0))])

        rows.append([""])
        rows.append(["USA-CHINA GAP", ""])
        rows.append(["usa_china_gap", str(status.get("usa_china_gap", 0))])
        rows.append(["usa_china_gap_trend", status.get("usa_china_gap_trend", "STABLE")])

        rows.append([""])
        rows.append(["GPR", ""])
        rows.append(["gpr_index_current", str(status.get("gpr_index_current", 0))])
        rows.append(["gpr_index_trend", status.get("gpr_index_trend", "STABLE")])
        rows.append(["gpr_index_zscore", str(status.get("gpr_index_zscore", 0))])

        rows.append([""])
        rows.append(["OVERLAYS", ""])
        scsi = status.get("supply_chain_stress_index", {})
        rows.append(["scsi_composite", str(scsi.get("composite", 0))])
        rows.append(["scsi_trend", scsi.get("trend", "STABLE")])
        rows.append(["scsi_chokepoint_alerts", str(scsi.get("active_chokepoint_alerts", 0))])

        ddi = status.get("dedollarization_index", {})
        rows.append(["ddi_composite", str(ddi.get("composite", 0))])
        rows.append(["ddi_trend", ddi.get("trend", "STABLE")])

        ewi = status.get("early_warning_index", {})
        rows.append(["ewi_active_signals", str(ewi.get("active_signals", 0))])
        rows.append(["ewi_severity", ewi.get("severity", "NONE")])

        rows.append([""])
        rows.append(["FDP (Fiscal Dominance Proximity)", ""])
        fdp = status.get("fiscal_dominance_proximity", {})
        for region in REGIONS:
            rows.append([f"fdp_{region}", str(fdp.get(region, 0))])

        rows.append([""])
        rows.append(["FEEDBACK LOOPS", ""])
        loops = status.get("dominant_feedback_loops", [])
        rows.append(["dominant_loops_json", json.dumps(loops)])

        return self._clear_and_write("G7_STATUS!A1", rows)

    def write_g7_thesis(self, thesis):
        """Write G7_THESIS tab — quartalsweise/interim ueberschrieben."""
        rows = [
            ["G7_THESIS", ""],
            ["date", thesis.get("date", "")],
            ["dominant_thesis", thesis.get("dominant_thesis", "")],
            ["confidence", thesis.get("confidence", "")],
            ["probability_source", thesis.get("probability_source", "")],
            ["interim_flag", str(thesis.get("interim_flag", False))],
            [""],
            ["SCENARIO PROBABILITIES", ""],
        ]

        probs = thesis.get("scenario_probabilities", {})
        for scenario, prob in probs.items():
            rows.append([scenario, str(prob)])

        rows.append([""])
        rows.append(["PREFERRED TARGETS", ""])
        targets = thesis.get("preferred_targets", {})
        rows.append(["preferred_targets_json", json.dumps(targets)])

        rows.append([""])
        rows.append(["VETOS", ""])
        rows.append(["active_vetos_json", json.dumps(thesis.get("active_vetos", []))])
        rows.append(["veto_watch_json", json.dumps(thesis.get("veto_watch", []))])

        rows.append([""])
        rows.append(["PERMOPT", ""])
        permopt = thesis.get("perm_opt_allocation", {})
        rows.append(["permopt_json", json.dumps(permopt)])

        rows.append([""])
        rows.append(["TILTS", ""])
        tilts = thesis.get("computed_tilts", {})
        rows.append(["tilts_json", json.dumps(tilts)])

        rows.append([""])
        rows.append(["SHIFT REASONS", ""])
        reasons = thesis.get("shift_reasons", [])
        rows.append(["shift_reasons_json", json.dumps(reasons)])

        return self._clear_and_write("G7_THESIS!A1", rows)

    def write_g7_narrative(self, narrative):
        """Write G7_NARRATIVE tab — woechentlich ueberschrieben."""

        def _serialize(val):
            """Ensure value is a string — serialize lists/dicts to JSON."""
            if val is None:
                return ""
            if isinstance(val, (list, dict)):
                return json.dumps(val, default=str)
            return str(val)

        rows = [
            ["G7_NARRATIVE", ""],
            ["headline", _serialize(narrative.get("headline", ""))],
            ["weekly_shift_narrative", _serialize(narrative.get("weekly_shift_narrative", ""))],
            ["scenario_implications", _serialize(narrative.get("scenario_implications", ""))],
            ["portfolio_context", _serialize(narrative.get("portfolio_context", ""))],
            ["unasked_question", _serialize(narrative.get("unasked_question", ""))],
            ["cascade_watch", _serialize(narrative.get("cascade_watch", []))],
            ["attention_flag", narrative.get("attention_flag", "NONE")],
            [""],
            ["REGIME CONGRUENCE", ""],
            ["regime_congruence_json", json.dumps(narrative.get("regime_congruence", {}), default=str)],
            ["regime_congruence_note", _serialize(narrative.get("regime_congruence_note", ""))],
            [""],
            ["COUNTER NARRATIVE", ""],
            ["counter_narrative_json", json.dumps(narrative.get("counter_narrative", {}), default=str)],
            [""],
            ["TOP SIGNALS", ""],
            ["top_signals_json", json.dumps(narrative.get("top_signals", []), default=str)],
            [""],
            ["SCHEMA-NOW", ""],
            ["historical_analog_json", json.dumps(narrative.get("historical_analog", {}), default=str)],
            ["liquidity_map_json", json.dumps(narrative.get("liquidity_distribution_map", {}), default=str)],
            ["correlation_regime_json", json.dumps(narrative.get("correlation_regime", {}), default=str)],
            [""],
            ["DASHBOARD EXPLANATIONS", ""],
        ]

        # Dashboard explanations — written as key-value pairs
        explanations = narrative.get("dashboard_explanations", {})
        if isinstance(explanations, dict):
            for key, text in explanations.items():
                rows.append([key, _serialize(text)])

        rows.append([""])
        rows.append(["META", ""])
        rows.append(["word_count", str(narrative.get("word_count", 0))])
        rows.append(["llm_model", narrative.get("llm_model", "")])
        rows.append(["generation_time_seconds", str(narrative.get("generation_time_seconds", 0))])

        return self._clear_and_write("G7_NARRATIVE!A1", rows)

    def write_g7_power_score_history(self, power_scores, gap_data):
        """Append weekly power score snapshot to G7_POWER_SCORE_HISTORY."""
        header = (
            ["date", "usa_china_gap", "usa_china_gap_trend"]
            + [f"{r}_score" for r in REGIONS]
            + [f"{r}_momentum" for r in REGIONS]
            + [f"{r}_acceleration" for r in REGIONS]
        )
        self._ensure_header("G7_POWER_SCORE_HISTORY", header)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = [
            now,
            str(gap_data.get("gap", 0)),
            gap_data.get("trend", "STABLE"),
        ]
        for region in REGIONS:
            ps = power_scores.get(region, {})
            row.append(str(round(ps.get("score", 0), 1)))
        for region in REGIONS:
            ps = power_scores.get(region, {})
            row.append(str(round(ps.get("momentum", 0), 2)))
        for region in REGIONS:
            ps = power_scores.get(region, {})
            row.append(str(round(ps.get("acceleration", 0), 2)))

        return self._append_row("G7_POWER_SCORE_HISTORY", row)

    def write_g7_thesis_history(self, history):
        """Write full thesis history (max 12 entries) to G7_THESIS_HISTORY."""
        header = [
            "date", "type", "dominant_thesis", "confidence",
            "probability_source", "probabilities_json",
            "shift_reasons_json", "key_overlays_json",
        ]
        rows = [header]

        for entry in history[-12:]:  # Max 12
            probs = entry.get("probabilities", {})
            overlays_snapshot = entry.get("key_overlays_at_time", {})
            rows.append([
                entry.get("date", ""),
                entry.get("type", ""),
                entry.get("dominant_thesis", ""),
                entry.get("confidence", ""),
                entry.get("probability_source", ""),
                json.dumps(probs),
                json.dumps(entry.get("shift_reasons", [])),
                json.dumps(overlays_snapshot),
            ])

        return self._clear_and_write("G7_THESIS_HISTORY!A1", rows)

    def write_g7_run_log(self, run_log):
        """Append run metadata to G7_RUN_LOG."""
        header = [
            "run_id", "run_type", "started", "completed",
            "total_duration_s", "final_status", "errors_count",
            "phases_json", "errors_json",
        ]
        self._ensure_header("G7_RUN_LOG", header)

        row = [
            run_log.get("run_id", ""),
            run_log.get("run_type", ""),
            run_log.get("started", ""),
            run_log.get("completed", ""),
            str(run_log.get("total_duration_s", 0)),
            run_log.get("final_status", ""),
            str(run_log.get("errors_count", 0)),
            json.dumps(run_log.get("phases", {})),
            json.dumps(run_log.get("errors", [])),
        ]

        return self._append_row("G7_RUN_LOG", row)

    def write_g7_data_cache(self, validated_data):
        """Write-through cache of validated API data to G7_DATA_CACHE."""
        rows = [
            ["G7_DATA_CACHE", ""],
            ["last_updated", datetime.now(timezone.utc).isoformat()],
            ["sources_count", str(len(validated_data))],
            [""],
        ]

        for source, data in validated_data.items():
            try:
                data_str = json.dumps(data, default=str)
                # Truncate to fit Sheet cell limit (~50k chars)
                if len(data_str) > 49000:
                    data_str = data_str[:49000] + "...(truncated)"
                rows.append([source, data_str])
            except Exception:
                rows.append([source, "SERIALIZATION_ERROR"])

        return self._clear_and_write("G7_DATA_CACHE!A1", rows)

    # ═══════════════════════════════════════════════════════
    # READERS
    # ═══════════════════════════════════════════════════════

    def read_previous_g7_status(self):
        """Read previous G7_STATUS for drift tracking."""
        rows = self._read_range("G7_STATUS!A1:B50")
        if not rows:
            return None

        result = {}
        for row in rows:
            if len(row) >= 2 and row[0]:
                result[row[0]] = row[1]

        if "g7_status" not in result:
            return None

        return result

    def read_previous_g7_scores(self):
        """Read previous power scores from G7_POWER_SCORE_HISTORY (last row)."""
        rows = self._read_range("G7_POWER_SCORE_HISTORY!A1:Z500")
        if not rows or len(rows) < 2:
            return None

        header = rows[0]
        last_row = rows[-1]

        # Parse into dict keyed by region
        scores = {}
        for region in REGIONS:
            score_col = f"{region}_score"
            mom_col = f"{region}_momentum"
            acc_col = f"{region}_acceleration"

            score_val = 50.0
            mom_val = 0.0
            acc_val = 0.0

            if score_col in header:
                idx = header.index(score_col)
                if idx < len(last_row):
                    try:
                        score_val = float(last_row[idx])
                    except (ValueError, TypeError):
                        pass

            if mom_col in header:
                idx = header.index(mom_col)
                if idx < len(last_row):
                    try:
                        mom_val = float(last_row[idx])
                    except (ValueError, TypeError):
                        pass

            if acc_col in header:
                idx = header.index(acc_col)
                if idx < len(last_row):
                    try:
                        acc_val = float(last_row[idx])
                    except (ValueError, TypeError):
                        pass

            scores[region] = {
                "score": score_val,
                "momentum": mom_val,
                "acceleration": acc_val,
            }

        return scores

    def read_previous_g7_thesis(self):
        """Read current G7_THESIS."""
        rows = self._read_range("G7_THESIS!A1:B50")
        if not rows:
            return None

        result = {}
        for row in rows:
            if len(row) >= 2 and row[0]:
                result[row[0]] = row[1]

        if "dominant_thesis" not in result:
            return None

        # Parse JSON fields
        for json_key in [
            "preferred_targets_json", "active_vetos_json",
            "veto_watch_json", "permopt_json", "tilts_json",
            "shift_reasons_json",
        ]:
            if json_key in result:
                try:
                    result[json_key] = json.loads(result[json_key])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Reconstruct scenario_probabilities
        probs = {}
        for key in [
            "managed_decline", "conflict_escalation",
            "us_renewal", "multipolar_chaos",
        ]:
            if key in result:
                try:
                    probs[key] = float(result[key])
                except (ValueError, TypeError):
                    pass
        if probs:
            result["scenario_probabilities"] = probs

        return result

    def read_g7_thesis_history(self):
        """Read G7_THESIS_HISTORY for drift tracking."""
        rows = self._read_range("G7_THESIS_HISTORY!A1:H20")
        if not rows or len(rows) < 2:
            return []

        header = rows[0]
        history = []

        for row in rows[1:]:
            entry = {}
            for i, col in enumerate(header):
                if i < len(row):
                    entry[col] = row[i]
            # Parse JSON fields
            for json_key in ["probabilities_json", "shift_reasons_json", "key_overlays_json"]:
                if json_key in entry:
                    try:
                        parsed = json.loads(entry[json_key])
                        # Store under clean key name
                        clean_key = json_key.replace("_json", "")
                        entry[clean_key] = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
            history.append(entry)

        return history

    def read_g7_operator_overrides(self):
        """Read G7_OPERATOR_OVERRIDES tab for manual scenario overrides."""
        rows = self._read_range("G7_OPERATOR_OVERRIDES!A1:B20")
        if not rows:
            return None

        result = {}
        for row in rows:
            if len(row) >= 2 and row[0]:
                result[row[0]] = row[1]

        if result.get("active", "").lower() != "true":
            return None

        # Parse override probabilities
        override_probs = {}
        for key in [
            "managed_decline", "conflict_escalation",
            "us_renewal", "multipolar_chaos",
        ]:
            if key in result:
                try:
                    override_probs[key] = float(result[key])
                except (ValueError, TypeError):
                    pass

        if override_probs:
            result["override_probabilities"] = override_probs

        return result
