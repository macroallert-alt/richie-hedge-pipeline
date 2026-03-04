import os, sys


def patch_templates():
    path = os.path.join('step1_market_analyst', 'modules', 'templates.py')
    if not os.path.exists(path):
        print(f'ERROR: {path} not found!')
        return
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()

    old = '    GENERIC_MAP = {\n        "value": "value",\n        "pctl": "pctl_1y",\n        "delta_5d": "delta_5d",\n        "direction": "direction",\n        "pctl_1y": "pctl_1y",\n    }'
    alias_block = '\n\n    ALIAS_MAP = {\n        "vts_value": ("vix_term_struct", "value"),\n        "vix_pctl": ("vix", "pctl_1y"),\n        "ivrv_value": ("iv_rv_spread", "value"),\n        "hy_velocity": ("hy_oas", "delta_5d"),\n        "nfci_value": ("nfci", "value"),\n        "nfci_direction": ("nfci", "direction"),\n        "spread_value": ("spread_2y10y", "value"),\n        "real_yield_value": ("real_10y_yield", "value"),\n        "hy_pctl": ("hy_oas", "pctl_1y"),\n        "ig_pctl": ("ig_oas", "pctl_1y"),\n        "hy_delta_5d": ("hy_oas", "delta_5d"),\n        "naaim_pctl": ("naaim_exposure", "pctl_1y"),\n        "aaii_pctl": ("aaii_bull_bear", "pctl_1y"),\n        "cot_es_pctl": ("cot_es_leveraged", "pctl_1y"),\n        "usdcnh_pctl": ("usdcnh", "pctl_1y"),\n        "china_10y_direction": ("china_10y", "direction"),\n        "rrp_pctl": ("rrp", "pctl_1y"),\n        "rrp_delta_5d": ("rrp", "delta_5d"),\n        "walcl_direction": ("walcl", "direction"),\n        "tga_delta_5d": ("tga", "delta_5d"),\n        "disc_window_direction": ("disc_window", "direction"),\n    }'
    if 'ALIAS_MAP' not in code and old in code:
        code = code.replace(old, old + alias_block)
        print('  FIX 1: ALIAS_MAP added')
    else:
        print('  FIX 1 skipped')

    old2 = '    def replacer(match):\n        key = match.group(1)\n\n        if key in GENERIC_KEYS and ctx_data:'
    new2 = '    def replacer(match):\n        key = match.group(1)\n\n        if key in ALIAS_MAP:\n            fn, dk = ALIAS_MAP[key]\n            fd = raw_data.get(fn, {})\n            if isinstance(fd, dict) and dk in fd:\n                return str(fd[dk])\n\n        if key in GENERIC_KEYS and ctx_data:'
    if 'ALIAS_MAP[key]' not in code and old2 in code:
        code = code.replace(old2, new2)
        print('  FIX 2: alias lookup added')
    else:
        print('  FIX 2 skipped')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)
    print('  templates.py saved')


def cleanup_scores():
    import gspread
    from google.oauth2.service_account import Credentials
    cp = '/tmp/gcp_sa.json'
    if not os.path.exists(cp):
        raw = os.environ.get('GOOGLE_CREDENTIALS', '') or os.environ.get('GCP_SA_KEY', '')
        if raw:
            with open(cp, 'w') as f:
                f.write(raw)
        else:
            print('ERROR: No credentials')
            return
    creds = Credentials.from_service_account_file(cp, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    ws = gc.open_by_key('1sZeZ4VVztAqjBjyfXcCfhpSWJ4pCGF8ip1ksu_TYMHY').worksheet('SCORES')
    data = ws.get_all_values()
    print(f'\nSCORES ({len(data)} rows)')
    prefixes = ['L2 Sentiment','L3 Intelligence','L4 Positioning','L5 Fragility','L6 Geopolitik','L7 Cross-Asset','L8 Seasonality']
    td = [i+1 for i,r in enumerate(data) if r and any(str(r[0]).startswith(p) for p in prefixes)]
    if not td:
        print('  SCORES clean')
        return
    for r in sorted(td, reverse=True):
        print(f'  Deleting legacy row {r}')
        ws.delete_rows(r)
    print(f'  Deleted {len(td)} rows')


if __name__ == '__main__':
    print('='*50)
    patch_templates()
    if '--cleanup' in sys.argv:
        cleanup_scores()
    print('='*50)
    print('DONE')