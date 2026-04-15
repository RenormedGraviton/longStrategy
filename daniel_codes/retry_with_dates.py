#!/usr/bin/env python3
"""
Retry failed symbols by querying Tardis for each symbol's actual
availableSince date and using that as from_date.
"""

import os
import sys
import json
import requests
from datetime import datetime

sys.path.insert(0, '.')
from fetch_universe import process_symbol, API_KEY, EXCHANGE

TO_DATE = "2026-04-07"
RAW_DIR = "data/raw"
BARS_DIR = "data/bars"
STATUS_LOG = "data/fetch_status.jsonl"

# Force line-buffered
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass


def get_symbol_availability():
    """Fetch all symbol availability dates from Tardis API."""
    print("Fetching symbol availability from Tardis...")
    r = requests.get(
        f"https://api.tardis.dev/v1/exchanges/{EXCHANGE}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    avail = {}
    for s in data.get('availableSymbols', []):
        sym = s['id'].upper()
        avail[sym] = {
            'since': s.get('availableSince', '2020-01-01')[:10],
            'to': s.get('availableTo', '9999-12-31')[:10],
        }
    return avail


def main():
    # Get symbols still missing bars
    bars_on_disk = set(f.replace('.csv', '') for f in os.listdir(BARS_DIR) if f.endswith('.csv'))
    missing = [l.strip() for l in open('data/missing_symbols.txt') if l.strip()]
    still_missing = [s for s in missing if s not in bars_on_disk]
    # Filter out unicode symbols that won't work
    still_missing = [s for s in still_missing if s.isascii()]

    print(f"{len(still_missing)} symbols still need bars")
    if not still_missing:
        print("Nothing to do!")
        return

    avail = get_symbol_availability()

    os.makedirs(os.path.dirname(STATUS_LOG) or '.', exist_ok=True)
    status_f = open(STATUS_LOG, 'a', buffering=1)
    def log_status(sym, status, **kw):
        rec = {"ts": datetime.utcnow().isoformat(), "symbol": sym, "status": status, **kw}
        status_f.write(json.dumps(rec) + "\n")

    succeeded = []
    failed = []
    skipped = []

    log_status("__run__", "start", n_symbols=len(still_missing), note="retry_with_dates")

    for i, sym in enumerate(still_missing):
        sym_lower = sym.lower()
        sym_upper = sym.upper()
        info = avail.get(sym_upper) or avail.get(sym_lower)

        if not info:
            print(f"\n[{i+1}/{len(still_missing)}] {sym} — NOT FOUND on Tardis, skipping")
            skipped.append(sym)
            log_status(sym, "skipped", reason="not_on_tardis")
            continue

        from_date = max("2025-10-07", info['since'])
        print(f"\n[{i+1}/{len(still_missing)}] {sym} — available since {info['since']}, using from_date={from_date}")
        log_status(sym, "begin", idx=i+1, total=len(still_missing), from_date=from_date)

        t0 = datetime.utcnow()
        try:
            ok = process_symbol(sym, from_date, TO_DATE, RAW_DIR, BARS_DIR)
            elapsed = (datetime.utcnow() - t0).total_seconds()
            if ok:
                succeeded.append(sym)
                log_status(sym, "ok", elapsed_s=round(elapsed, 1))
            else:
                failed.append(sym)
                log_status(sym, "failed", elapsed_s=round(elapsed, 1))
        except Exception as e:
            elapsed = (datetime.utcnow() - t0).total_seconds()
            print(f"  EXCEPTION: {e}")
            failed.append(sym)
            log_status(sym, "exception", elapsed_s=round(elapsed, 1), error=str(e))

        if (i + 1) % 10 == 0:
            print(f"\n  Progress: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Skipped:   {len(skipped)} (not on Tardis)")
    if failed:
        print(f"  Failed: {failed}")
    if skipped:
        print(f"  Skipped: {skipped}")
    log_status("__run__", "done", succeeded=len(succeeded), failed=len(failed), skipped=len(skipped))
    status_f.close()


if __name__ == '__main__':
    main()
