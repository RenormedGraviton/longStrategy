#!/usr/bin/env python3
"""
Retry symbols that failed in the main fetch by querying Tardis for each
symbol's actual available date range and clamping the request to it.

The main fetch failed these symbols with HTTP 400 because we asked for the
full 2025-10-07 → 2026-04-07 window, but their Tardis data starts later
(new listings) or ends earlier (delistings). Clamping to the available
range lets us recover them.

Symbols whose entire window is shorter than `MIN_TOTAL_DAYS` will be skipped
(too little data to be useful even for OOS). All others are fetched into
data/bars/ with whatever rows they have.
"""

import os
import sys
import json
import requests
import datetime as dt
from fetch_universe import (
    process_symbol, EXCHANGE, API_KEY, _rss_mb,
)


TARGET_FROM = dt.date(2025, 10, 7)
TARGET_TO   = dt.date(2026, 4, 7)
MIN_TOTAL_DAYS = 5  # bail on anything shorter — not enough for any feature use


def load_failed_symbols(status_log='data/fetch_status.jsonl'):
    """Read fetch_status.jsonl and return symbols whose LAST status is 'failed'."""
    last_status = {}
    with open(status_log) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = rec.get('symbol')
            st = rec.get('status')
            if sym and sym != '__run__' and st in ('ok', 'failed', 'exception'):
                last_status[sym] = st
    return sorted([s for s, st in last_status.items() if st in ('failed', 'exception')])


def get_tardis_availability(symbols):
    """Hit Tardis exchange-info endpoint, return dict[sym -> (since, to)]."""
    print(f"Querying Tardis for {EXCHANGE} symbol availability...")
    r = requests.get(
        f"https://api.tardis.dev/v1/exchanges/{EXCHANGE}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    # Tardis stores ids lowercase
    lookup = {s['id'].upper(): s for s in data.get('availableSymbols', [])}
    out = {}
    for sym in symbols:
        meta = lookup.get(sym)
        if meta is None:
            out[sym] = None
            continue
        since = meta.get('availableSince', '')[:10]
        to    = meta.get('availableTo', '')[:10] or None
        out[sym] = (since, to)
    return out


def main():
    failed = load_failed_symbols()
    print(f"Found {len(failed)} failed symbols in status log\n")

    avail = get_tardis_availability(failed)

    plan = []   # (sym, from_date, to_date)
    skips = []  # (sym, reason)
    for sym in failed:
        meta = avail.get(sym)
        if meta is None:
            skips.append((sym, 'not on tardis'))
            continue
        since_s, to_s = meta
        if not since_s:
            skips.append((sym, 'no availableSince'))
            continue
        since = dt.date.fromisoformat(since_s)
        to    = dt.date.fromisoformat(to_s) if to_s else TARGET_TO
        clamped_from = max(since, TARGET_FROM)
        clamped_to   = min(to,    TARGET_TO)
        days = (clamped_to - clamped_from).days
        if days < MIN_TOTAL_DAYS:
            skips.append((sym, f'only {days}d available'))
            continue
        plan.append((sym, clamped_from.isoformat(), clamped_to.isoformat()))

    print(f"\nPlan: fetch {len(plan)} symbols, skip {len(skips)}")
    if skips:
        print("\nSkipping:")
        for sym, reason in skips:
            print(f"  {sym}: {reason}")
    print(f"\nWill fetch:")
    for sym, f, t in plan:
        days = (dt.date.fromisoformat(t) - dt.date.fromisoformat(f)).days
        print(f"  {sym:<14} {f} → {t}  ({days}d)")
    print()

    raw_dir = 'data/raw'
    bars_dir = 'data/bars'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(bars_dir, exist_ok=True)

    succeeded = []
    failures  = []
    for i, (sym, fd, td) in enumerate(plan):
        print(f"\n{'#'*70}\n# [{i+1}/{len(plan)}] {sym}  ({fd} → {td})\n{'#'*70}")
        try:
            ok = process_symbol(sym, fd, td, raw_dir, bars_dir)
            (succeeded if ok else failures).append(sym)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failures.append(sym)
        print(f"  RSS={_rss_mb():.0f} MB  (running tally: {len(succeeded)} ok / {len(failures)} failed)")

    print(f"\n{'='*70}\nRetry done.  ok={len(succeeded)}  failed={len(failures)}")
    if failures:
        print(f"Still failed: {failures}")


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    main()
