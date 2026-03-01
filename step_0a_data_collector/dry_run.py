#!/usr/bin/env python3
"""
dry_run.py — Lokaler Test der Data Collector Pipeline
======================================================
Fuehrt alle 4 Phasen aus, aber OHNE ins V16 Sheet zu schreiben.
Outputs: Console + data/*.json Dateien lokal.

Voraussetzungen:
  1. pip install -r requirements.txt
  2. FRED_API_KEY als Environment-Variable oder .env Datei
  3. Optional: ANTHROPIC_API_KEY (fuer LLM-Fallback)

Ausfuehrung:
  cd step_0a_data_collector
  export FRED_API_KEY=dein_key_hier
  python dry_run.py

  Oder mit .env:
  echo "FRED_API_KEY=dein_key_hier" > .env
  python dry_run.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# ─── .env Support (optional) ───
def load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())
        print("📁 .env geladen")

load_dotenv()

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dry_run")

# ─── Pre-Flight Checks ───
def preflight():
    print("\n" + "=" * 60)
    print("  DRY RUN — Pre-Flight Checks")
    print("=" * 60)

    checks = {
        "FRED_API_KEY": bool(os.environ.get('FRED_API_KEY')),
        "ANTHROPIC_API_KEY": bool(os.environ.get('ANTHROPIC_API_KEY')),
        "FMP_API_KEY": bool(os.environ.get('FMP_API_KEY') or os.environ.get('EODHD_API_KEY')),
        "EODHD_API_KEY": bool(os.environ.get('EODHD_API_KEY')),
    }

    for name, ok in checks.items():
        status = "✅" if ok else "⚠️  MISSING (optional)"
        if name == "FRED_API_KEY" and not ok:
            status = "❌ REQUIRED"
        print(f"  {name}: {status}")

    # Check configs
    config_files = [
        'config/field_registry.json',
        'config/event_calendar.json',
        'config/us_holidays.json',
        'config/sp500_tickers.json',
    ]
    print()
    for cf in config_files:
        exists = os.path.exists(cf)
        print(f"  {cf}: {'✅' if exists else '❌ MISSING'}")
        if not exists:
            print(f"    → Kopiere die Config-Dateien nach {os.path.dirname(cf)}/")

    # Check dependencies
    print()
    deps = ['fredapi', 'yfinance', 'pandas', 'numpy', 'scipy', 'requests', 'bs4']
    missing = []
    for dep in deps:
        try:
            __import__(dep)
            print(f"  {dep}: ✅")
        except ImportError:
            print(f"  {dep}: ❌ MISSING → pip install {dep}")
            missing.append(dep)

    if not os.environ.get('FRED_API_KEY'):
        print("\n❌ FRED_API_KEY ist erforderlich. Setze ihn:")
        print("   export FRED_API_KEY=dein_key_hier")
        print("   oder erstelle eine .env Datei")
        return False

    if missing:
        print(f"\n❌ Fehlende Pakete: {', '.join(missing)}")
        print(f"   pip install {' '.join(missing)}")
        return False

    if not os.path.exists('config/field_registry.json'):
        print("\n❌ Config-Dateien fehlen. Lege sie unter config/ ab.")
        return False

    print("\n✅ Pre-Flight OK — starte Pipeline...\n")
    return True


def run_dry():
    if not preflight():
        sys.exit(1)

    today = datetime.now().date()
    start = time.time()
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)

    # ─── Load Registry ───
    with open('config/field_registry.json') as f:
        registry_data = json.load(f)
    registry = {f['name']: f for f in registry_data['fields']}
    print(f"📋 Registry: {len(registry)} Felder geladen")

    # ─── Load Cache ───
    from cache import HistoryCache, FetchCache
    history_cache = HistoryCache('data/cache')
    history_data = history_cache.load()
    fetch_cache = FetchCache('data/cache')
    fetch_cache_data = fetch_cache.load()

    # ═════════════════════════════════════════════
    # PHASE 1: FETCH
    # ═════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 1: FETCH")
    print("=" * 60)
    t1 = time.time()

    from fetchers import MasterFetcher
    fetcher = MasterFetcher(config_dir='config')
    fetch_results = fetcher.fetch_all(cache=fetch_cache_data)

    elapsed_p1 = time.time() - t1
    ok_count = sum(1 for r in fetch_results.values() if r.success)
    total = len(fetch_results)
    print(f"\n📊 Phase 1: {ok_count}/{total} Felder geholt in {elapsed_p1:.1f}s")
    print(f"   FRED calls: {fetcher.fred.call_count}, "
          f"yfinance calls: {fetcher.yf.call_count}, "
          f"LLM calls: {fetcher.llm.call_count}")

    # Show failed fields
    failed = [name for name, r in fetch_results.items()
              if not r.success and not name.startswith('_')]
    if failed:
        print(f"\n⚠️  Fehlgeschlagen ({len(failed)}):")
        for f in sorted(failed):
            print(f"   • {f}")

    # Show V16 prices status
    v16_prices = fetch_results.get('_v16_prices')
    if v16_prices and v16_prices.success:
        prices = v16_prices.value
        ok_prices = sum(1 for v in prices.values() if v is not None)
        print(f"\n📈 V16 Preise: {ok_prices}/27 Assets")
        missing_prices = [k for k, v in prices.items() if v is None]
        if missing_prices:
            print(f"   Missing: {', '.join(missing_prices)}")

    # ═════════════════════════════════════════════
    # PHASE 2: TRANSFORM
    # ═════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: TRANSFORM")
    print("=" * 60)
    t2 = time.time()

    from transforms import TransformEngine
    engine = TransformEngine(registry, history_data, 'config/us_holidays.json')
    transformed = engine.transform_all(fetch_results, today)

    elapsed_p2 = time.time() - t2
    ok_transformed = sum(1 for t in transformed.values() if t.value is not None)
    print(f"\n📊 Phase 2: {ok_transformed}/{len(transformed)} Felder transformiert in {elapsed_p2:.1f}s")

    # ═════════════════════════════════════════════
    # PHASE 3: QUALITY
    # ═════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 3: QUALITY")
    print("=" * 60)
    t3 = time.time()

    from quality import QualityEngine
    quality = QualityEngine(registry, history_data)
    dq_summary = quality.run_all(transformed, fetch_results)

    elapsed_p3 = time.time() - t3
    print(f"\n📊 Phase 3: Quality Level = {dq_summary['data_quality_level']} in {elapsed_p3:.1f}s")
    if dq_summary.get('alerts'):
        print("   Alerts:")
        for a in dq_summary['alerts']:
            print(f"   ⚠️  {a}")

    # ═════════════════════════════════════════════
    # PHASE 4: WRITE (JSON only, NO Sheet!)
    # ═════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 4: WRITE (JSON only — Sheet SKIPPED)")
    print("=" * 60)
    t4 = time.time()

    from writers import JSONWriter
    json_writer = JSONWriter('data')
    paths = json_writer.write_all(transformed, dq_summary)

    elapsed_p4 = time.time() - t4
    print(f"\n📁 Geschriebene Dateien:")
    for name, path in paths.items():
        size = os.path.getsize(path)
        print(f"   {name}: {path} ({size:,} bytes)")

    # ─── Update Cache ───
    engine.update_history(transformed, today)
    history_cache.data = history_data
    history_cache.prune()
    history_cache.save()
    fetch_cache.update_from_results(fetch_results)
    fetch_cache.save()
    print(f"\n💾 Cache gespeichert ({len(history_data)} Felder)")

    # ═════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════
    total_elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  DRY RUN COMPLETE")
    print("=" * 60)
    print(f"  Total:     {total_elapsed:.1f}s")
    print(f"  Phase 1:   {elapsed_p1:.1f}s (Fetch)")
    print(f"  Phase 2:   {elapsed_p2:.1f}s (Transform)")
    print(f"  Phase 3:   {elapsed_p3:.1f}s (Quality)")
    print(f"  Phase 4:   {elapsed_p4:.1f}s (Write JSON)")
    print(f"  Quality:   {dq_summary['data_quality_level']}")
    print(f"  Fields:    {dq_summary['fields_ok']}/{dq_summary['fields_total']} OK")
    print(f"  Stale:     {dq_summary['fields_stale']}")
    print(f"  Failed:    {dq_summary['fields_failed']}")
    print(f"  Anomaly:   {dq_summary['fields_anomaly']}")

    # ─── Sample Output ───
    print("\n" + "=" * 60)
    print("  SAMPLE — Erste 10 T1 Felder")
    print("=" * 60)
    t1_fields = [(fn, tf) for fn, tf in sorted(transformed.items()) if tf.tier == "T1" and tf.value is not None]
    for fn, tf in t1_fields[:10]:
        d5 = f"d5={tf.delta_5d:+.2f}" if tf.delta_5d is not None else "d5=n/a"
        pctl = f"p={tf.pctl_1y:.0f}%" if tf.pctl_1y is not None else "p=n/a"
        z = f"z={tf.zscore_2y:+.2f}" if tf.zscore_2y is not None else "z=n/a"
        direction = tf.direction or "?"
        print(f"  {fn:25s} = {tf.value:12.2f}  {d5:15s}  {pctl:8s}  {z:10s}  {direction:5s}  conf={tf.confidence:.2f}")

    print("\n" + "=" * 60)
    print("  SAMPLE — Erste 10 T2 Felder")
    print("=" * 60)
    t2_fields = [(fn, tf) for fn, tf in sorted(transformed.items()) if tf.tier == "T2" and tf.value is not None]
    for fn, tf in t2_fields[:10]:
        d5 = f"d5={tf.delta_5d:+.2f}" if tf.delta_5d is not None else "d5=n/a"
        pctl = f"p={tf.pctl_1y:.0f}%" if tf.pctl_1y is not None else "p=n/a"
        print(f"  {fn:25s} = {tf.value:12.2f}  {d5:15s}  {pctl:8s}  conf={tf.confidence:.2f}")

    # ─── V16 Sheet Preview ───
    if v16_prices and v16_prices.success:
        print("\n" + "=" * 60)
        print("  V16 SHEET PREVIEW — DATA_Prices Row")
        print("=" * 60)
        prices = v16_prices.value
        print(f"  Date: {today}")
        for ticker in ['GLD', 'SPY', 'TLT', 'BTC', 'VIX']:
            if ticker == 'VIX':
                vix_tf = transformed.get('vix')
                val = vix_tf.value if vix_tf else None
            else:
                p = prices.get(ticker)
                val = p[0] if p else None
            print(f"  {ticker:8s}: {val}" if val else f"  {ticker:8s}: MISSING")

    v16_liq = fetch_results.get('_v16_liquidity')
    if v16_liq and v16_liq.success:
        liq = v16_liq.value
        print(f"\n  V16 SHEET PREVIEW — DATA_Liquidity Row")
        print(f"  Fed_Net_Liq: {liq.get('Fed_Net_Liq', 'MISSING')}")
        print(f"  ECB_USD:     {liq.get('ECB_USD', 'MISSING')}")
        print(f"  BOJ_USD:     {liq.get('BOJ_USD', 'MISSING')}")
        print(f"  US_M2:       {liq.get('US_M2', 'MISSING')}")

    print("\n🏁 Dry Run fertig. Prüfe data/*.json fuer Details.")
    print("   Wenn alles gut aussieht → GitHub Actions Workflow aktivieren.")
    print("   Sheet-Write wird erst im Production-Run (main.py) ausgefuehrt.\n")


if __name__ == "__main__":
    try:
        run_dry()
    except KeyboardInterrupt:
        print("\n\nAbgebrochen.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
