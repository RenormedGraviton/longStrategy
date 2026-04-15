#!/usr/bin/env python3
"""
Fetch incremental data (append to existing bars).
Downloads new days and appends to existing bar CSVs.
"""

import os
import sys
import json
import glob
import shutil
import gc
import requests
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, '.')
from fetch_universe import (
    process_symbol, API_KEY, EXCHANGE,
    aggregate_trades_day, aggregate_deriv_day,
)
from tardis_dev import datasets

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass


def get_symbol_availability():
    r = requests.get(
        f"https://api.tardis.dev/v1/exchanges/{EXCHANGE}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    avail = {}
    for s in r.json().get('availableSymbols', []):
        avail[s['id'].upper()] = {
            'since': s.get('availableSince', '2020-01-01')[:10],
            'to': s.get('availableTo', '9999-12-31')[:10],
        }
    return avail


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_date', default='2026-04-07')
    parser.add_argument('--to_date', default='2026-04-13')
    parser.add_argument('--bars_dir', default='data/bars')
    parser.add_argument('--raw_dir', default='data/raw')
    parser.add_argument('--symbols_file', help='File with one symbol per line')
    args = parser.parse_args()

    if args.symbols_file:
        symbols = [l.strip() for l in open(args.symbols_file) if l.strip()]
    else:
        # Default: all symbols that already have bars
        symbols = [f.replace('.csv', '') for f in sorted(os.listdir(args.bars_dir))
                   if f.endswith('.csv')]

    print(f"Incremental fetch: {len(symbols)} symbols, {args.from_date} to {args.to_date}")

    avail = get_symbol_availability()
    succeeded = 0
    failed = 0
    skipped = 0

    for i, sym in enumerate(symbols):
        existing_path = os.path.join(args.bars_dir, f"{sym}.csv")

        # Check if symbol still available
        info = avail.get(sym)
        if not info:
            skipped += 1
            continue

        from_date = max(args.from_date, info['since'])
        to_date = min(args.to_date, info['to'])
        if from_date >= to_date:
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(symbols)}] {sym} ({from_date} to {to_date})")

        sym_raw = os.path.join(args.raw_dir, sym)
        os.makedirs(sym_raw, exist_ok=True)

        try:
            datasets.download(
                exchange=EXCHANGE,
                data_types=["trades", "derivative_ticker"],
                from_date=from_date,
                to_date=to_date,
                symbols=[sym],
                api_key=API_KEY,
                download_dir=sym_raw,
            )
        except Exception as e:
            print(f"  DOWNLOAD FAILED: {e}")
            shutil.rmtree(sym_raw, ignore_errors=True)
            failed += 1
            continue

        # Process trade files
        trade_files = sorted([
            os.path.join(sym_raw, f) for f in os.listdir(sym_raw)
            if 'trades_' in f and sym in f and f.endswith('.csv.gz')
        ])
        deriv_files = sorted([
            os.path.join(sym_raw, f) for f in os.listdir(sym_raw)
            if 'derivative_ticker_' in f and sym in f and f.endswith('.csv.gz')
        ])

        if not trade_files:
            print(f"  No trade files")
            shutil.rmtree(sym_raw, ignore_errors=True)
            failed += 1
            continue

        bars_list = []
        for f in trade_files:
            try:
                bars_list.append(aggregate_trades_day(f))
            except Exception as e:
                print(f"    Error: {e}")

        if not bars_list:
            shutil.rmtree(sym_raw, ignore_errors=True)
            failed += 1
            continue

        new_bars = pd.concat(bars_list).sort_index()
        del bars_list
        gc.collect()

        deriv_list = []
        for f in deriv_files:
            try:
                deriv_list.append(aggregate_deriv_day(f))
            except Exception as e:
                pass
        if deriv_list:
            deriv = pd.concat(deriv_list).sort_index()
            del deriv_list
            new_bars = new_bars.join(deriv, how='left')
            del deriv
            if 'open_interest' in new_bars.columns:
                new_bars['open_interest'] = new_bars['open_interest'].ffill()
            if 'funding_rate' in new_bars.columns:
                new_bars['funding_rate'] = new_bars['funding_rate'].ffill()

        new_bars = new_bars.dropna(subset=['close'])
        new_bars['symbol'] = sym

        # Merge with existing
        if os.path.exists(existing_path):
            old = pd.read_csv(existing_path, index_col=0, parse_dates=True)
            combined = pd.concat([old, new_bars])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            del old
        else:
            combined = new_bars

        combined.to_csv(existing_path)
        print(f"  OK: {len(new_bars)} new bars, {len(combined)} total")
        succeeded += 1

        del new_bars, combined
        gc.collect()
        shutil.rmtree(sym_raw, ignore_errors=True)

        if (i + 1) % 20 == 0:
            print(f"\n  Progress: {succeeded} ok, {failed} failed, {skipped} skipped")

    print(f"\nDONE: {succeeded} ok, {failed} failed, {skipped} skipped")


if __name__ == '__main__':
    main()
