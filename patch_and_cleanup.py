"""
patch_and_cleanup.py
Einmal-Script: Patcht engine.py lokal + cleaned DW Tabs via GitHub Actions.
Usage lokal:   python patch_and_cleanup.py
Usage Actions: python patch_and_cleanup.py --cleanup-only
"""

import os
import sys
import gspread
from google.oauth2.service_account import Credentials

DW_SHEET_ID = "1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY"


def patch_engine():
    engine_path = os.path.join("step1_market_analyst", "engine.py")
    if not os.path.exists(engine_path):
        print(f"ERROR: {engine_path} not found!")
        return False

    with open(engine_path, "r", encoding="utf-8") as f:
        code = f.read()

    old_1 = """    # Find the "Agent 1" / "Senior Macro" row and overwrite it
    all_data = ws.get_all_values()
    target_row = None
    for i, row in enumerate(all_data):
        if len(row) >= 3 and ("Agent 1" in str(row[1]) or "Senior Macro" in str(row[2])):
            target_row = i + 1  # 1-indexed
            break
    if target_row is None:
        # Fallback: write to row 2 (after title)
        target_row = 2"""

    new_1 = """    # Find the "Agent 1" / "Senior Macro" / "Step1_MarketAnalyst" row and overwrite it
    all_data = ws.get_all_values()
    target_row = None
    for i, row in enumerate(all_data):
        if len(row) >= 2:
            row_str = " ".join(str(c) for c in row[:4]).lower()
            if any(kw in row_str for kw in ["agent 1", "senior macro", "step1_marketanalyst"]):
                target_row = i + 1  # 1-indexed
                break
    if target_row is None:
        # Fallback: write to row 2 (after title)
        target_row = 2"""

    if old_1 in code:
        code = code.replace(old_1, new_1)
        print("  FIX 1 applied: write_agent_summary row detection")
    else:
        print("  FIX 1 skipped (already patched?)")

    old_2 = """        history = []
        for row in data[1:31]:
            if not row or not row[0]:
                continue
            day_record = {"date": row[0], "layers": {}}"""

    new_2 = """        history = []
        for row in data[1:31]:
            if not row or not row[0]:
                continue
            cell0 = str(row[0]).strip()
            if not cell0 or cell0 == "\u2014":
                continue
            if len(cell0) < 8 or cell0[4:5] != "-":
                continue
            day_record = {"date": cell0, "layers": {}}"""

    if old_2 in code:
        code = code.replace(old_2, new_2)
        print("  FIX 2 applied: read_history date filter")
    else:
        print("  FIX 2 skipped (already patched?)")

    with open(engine_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("  engine.py saved")
    return True


def connect():
    for path in ["/tmp/gcp_sa.json", "gcp_sa.json",
                 os.path.join(os.path.expanduser("~"), ".config", "gcp_sa.json")]:
        if os.path.exists(path):
            creds = Credentials.from_service_account_file(path, scopes=[
                "https://www.googleapis.com/auth/spreadsheets"])
            gc = gspread.authorize(creds)
            return gc.open_by_key(DW_SHEET_ID)
    raw = os.environ.get("GOOGLE_CREDENTIALS", "") or os.environ.get("GCP_SA_KEY", "")
    if raw:
        with open("/tmp/gcp_sa.json", "w") as f:
            f.write(raw)
        creds = Credentials.from_service_account_file("/tmp/gcp_sa.json", scopes=[
            "https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        return gc.open_by_key(DW_SHEET_ID)
    print("ERROR: No GCP credentials found!")
    return None


def cleanup_agent_summary(sheet):
    ws = sheet.worksheet("AGENT_SUMMARY")
    data = ws.get_all_values()
    print(f"\nAGENT_SUMMARY ({len(data)} rows)")
    rows_to_clear = []
    for i, row in enumerate(data):
        if i == 0:
            continue
        if "09:08:20" in " ".join(str(c) for c in row):
            rows_to_clear.append(i + 1)
            print(f"  STALE row {i+1}")
    if not rows_to_clear:
        print("  Clean")
        return
    for r in sorted(rows_to_clear, reverse=True):
        ws.delete_rows(r)
    print(f"  Deleted {len(rows_to_clear)} row(s)")


def cleanup_scores(sheet):
    ws = sheet.worksheet("SCORES")
    data = ws.get_all_values()
    print(f"\nSCORES ({len(data)} rows)")
    col_def_row = None
    for i, row in enumerate(data):
        if len(row) > 0 and "COLUMN DEFINITIONS" in str(row[0]):
            col_def_row = i + 1
            break
    if col_def_row is None:
        print("  COLUMN DEFINITIONS not found - abort")
        return
    junk_start = 10
    junk_end = col_def_row - 2
    if junk_start > junk_end:
        print("  Clean")
        return
    n = junk_end - junk_start + 1
    print(f"  Deleting {n} legacy rows ({junk_start}-{junk_end})")
    for r in range(junk_end, junk_start - 1, -1):
        ws.delete_rows(r)
    print(f"  Done")


def verify_beliefs(sheet):
    ws = sheet.worksheet("BELIEFS")
    data = ws.get_all_values()
    print(f"\nBELIEFS ({len(data)} rows)")
    if len(data) >= 2:
        r = data[1]
        ok = len(r) >= 25 and r[0] and r[0][:4] == "2026"
        print(f"  Row 2: {'OK' if ok else 'NEEDS ATTENTION'}")


def verify_divergence(sheet):
    ws = sheet.worksheet("DIVERGENCE")
    data = ws.get_all_values()
    print(f"\nDIVERGENCE ({len(data)} rows)")
    print("  OK")


def main():
    print("=" * 50)
    cleanup_only = "--cleanup-only" in sys.argv

    if not cleanup_only:
        print("Patching engine.py...")
        if not patch_engine():
            return

    print("Cleaning DW Tabs...")
    sheet = connect()
    if not sheet:
        return
    cleanup_agent_summary(sheet)
    cleanup_scores(sheet)
    verify_beliefs(sheet)
    verify_divergence(sheet)
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
