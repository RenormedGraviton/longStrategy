#!/usr/bin/env python3
"""
Fetch universe of Binance perp symbols from Tardis.

Strategy: stream per-symbol to minimize disk usage:
  1. Download one symbol's 6 months of data
  2. Process to hourly bars
  3. Save bars to data/bars/{symbol}.csv
  4. Delete raw files
  5. Move to next symbol

Excludes BTC, ETH, SOL, XRP, DOGE, BNB, BCH (per user request — most efficient
pricing / lowest predictability for this strategy).
"""

import os
import re
import gc
import sys
import json
import shutil
import argparse
import resource
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tardis_dev import datasets


def _rss_mb():
    """Return current process RSS in MB (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


_DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def _file_date_window(path):
    """Extract the YYYY-MM-DD from a Tardis filename and return (day_start, day_end).

    Tardis daily files occasionally contain a handful of ticks from adjacent UTC days
    (likely because file partitioning uses local_timestamp, not exchange timestamp).
    We use this window to drop those stray ticks before resampling so each day file
    contributes exactly the hours within its labeled UTC date.
    """
    m = _DATE_RE.search(os.path.basename(path))
    if not m:
        return None, None
    day = pd.Timestamp(m.group(1))
    return day, day + pd.Timedelta(days=1)
import warnings
warnings.filterwarnings('ignore')

API_KEY = "TD.iqyl7P-p9TcTzy7X.IEPM2eFvy19lUXK.Rf95c4SQ9zi81Gj.21aHZ-ODndsFMla.7mkNuVn1zYioBsm.EElq"
EXCHANGE = "binance-futures"


EXCLUDED_SYMBOLS = (
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'BNBUSDT', 'BCHUSDT',
)


def get_universe(exclude=EXCLUDED_SYMBOLS, top_n=200):
    """Return list of active USDT perp symbols, excluding given list."""
    print(f"Fetching {EXCHANGE} symbol list from Tardis...")
    r = requests.get(
        f"https://api.tardis.dev/v1/exchanges/{EXCHANGE}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    syms = data.get('availableSymbols', [])
    active = [
        s['id'].upper() for s in syms
        if s['type'] == 'perpetual'
        and s['id'].endswith('usdt')
        and s.get('availableTo', '9999') > '2026-03-01'
        and s['id'].upper() not in exclude
    ]
    print(f"  Active USDT perps (excluding {exclude}): {len(active)}")
    if top_n and len(active) > top_n:
        print(f"  Capping at {top_n}")
        active = active[:top_n]
    return active


def aggregate_trades_day(f):
    # Memory-efficient: load only needed cols with explicit dtypes (price as float32,
    # amount as float32 — sufficient for aggregation precision, halves RAM vs default
    # float64). High-volume coins (e.g. SIRENUSDT) can have ~10M trades per day,
    # so we drop the str `side` column ASAP after deriving the buy mask.
    df = pd.read_csv(
        f, compression='gzip',
        usecols=['timestamp', 'side', 'price', 'amount'],
        dtype={'timestamp': np.int64, 'side': 'category',
               'price': np.float32, 'amount': np.float32},
    )
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us')
    df.drop(columns=['timestamp'], inplace=True)

    # Drop stray ticks from adjacent UTC days (Tardis day-file boundary spillover)
    day_start, day_end = _file_date_window(f)
    if day_start is not None:
        df = df[(df['ts'] >= day_start) & (df['ts'] < day_end)]
    if df.empty:
        return pd.DataFrame()

    # Derive helper columns (kept as float32 to avoid widening)
    df['quote_volume'] = (df['price'] * df['amount']).astype(np.float32)
    df['buy_volume'] = np.where(df['side'] == 'buy',
                                df['amount'], np.float32(0)).astype(np.float32)
    df.drop(columns=['side'], inplace=True)
    df.set_index('ts', inplace=True)

    # Resample once via .agg() — single pass over the data, much cheaper than
    # 8 separate resample calls each of which scans the full series.
    price_bars = df['price'].resample('1h').agg(['first', 'max', 'min', 'last', 'count'])
    price_bars.columns = ['open', 'high', 'low', 'close', 'trades_count']
    sums = df[['amount', 'quote_volume', 'buy_volume']].resample('1h').sum()

    bars = pd.concat([price_bars, sums], axis=1)
    bars.rename(columns={'amount': 'volume'}, inplace=True)
    bars['vwap'] = bars['quote_volume'] / bars['volume'].replace(0, np.nan)

    # Enforce a stable column order matching the existing schema in data/bars/*.csv
    bars = bars[['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                 'trades_count', 'buy_volume', 'vwap']]

    # Free the big intermediate
    del df, price_bars, sums
    return bars


def aggregate_deriv_day(f):
    # Only read the columns we actually need. Tardis derivative_ticker has 11 cols
    # (including exchange/symbol/local_timestamp/index_price/last_price etc) and we
    # ignore most of them — loading them all wastes RAM on big files.
    wanted = ['timestamp', 'open_interest', 'funding_rate', 'mark_price']
    df = pd.read_csv(
        f, compression='gzip',
        usecols=lambda c: c in wanted,
        dtype={'timestamp': np.int64,
               'open_interest': np.float32,
               'funding_rate': np.float32,
               'mark_price': np.float32},
    )
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us')
    df.drop(columns=['timestamp'], inplace=True)
    # Drop stray ticks from adjacent UTC days (Tardis day-file boundary spillover)
    day_start, day_end = _file_date_window(f)
    if day_start is not None:
        df = df[(df['ts'] >= day_start) & (df['ts'] < day_end)]
    if df.empty:
        return pd.DataFrame()
    df.set_index('ts', inplace=True)

    bars = pd.DataFrame()
    if 'open_interest' in df.columns:
        bars['open_interest'] = df['open_interest'].resample('1h').last()
    # IMPORTANT — column naming gotcha for binance-futures on Tardis:
    #
    #   `funding_rate`           = the LIVE PREDICTED/ESTIMATED funding rate, sourced
    #                              from Binance's markPriceUpdate WebSocket `r` field.
    #                              Updates ~every 60s based on the premium index.
    #                              THIS IS THE LIVE SIGNAL WE WANT.
    #   `predicted_funding_rate` = exists in Tardis schema but is ALWAYS NaN for
    #                              binance-futures (verified across multiple symbols
    #                              and dates 2025-10-07 to 2026-04-01). Do NOT capture.
    #
    # Verified by: BTCUSDT 2026-04-01 had 1022 changes/day to `funding_rate`
    # with median ~61s gap → that is live, not stepwise realized.
    if 'funding_rate' in df.columns:
        bars['funding_rate'] = df['funding_rate'].resample('1h').last()
    if 'mark_price' in df.columns:
        bars['mark_price'] = df['mark_price'].resample('1h').last()
    del df
    return bars


def process_symbol(symbol, from_date, to_date, raw_dir, bars_dir):
    """Download → process → save → cleanup for one symbol."""
    print(f"\n{'='*70}")
    print(f"  {symbol}")
    print(f"{'='*70}")

    bars_path = os.path.join(bars_dir, f"{symbol}.csv")
    if os.path.exists(bars_path):
        print(f"  Bars exist, skipping: {bars_path}")
        return True

    # Download
    print(f"  Downloading {from_date} to {to_date}...")
    sym_raw = os.path.join(raw_dir, symbol)
    os.makedirs(sym_raw, exist_ok=True)
    try:
        datasets.download(
            exchange=EXCHANGE,
            data_types=["trades", "derivative_ticker"],
            from_date=from_date,
            to_date=to_date,
            symbols=[symbol],
            api_key=API_KEY,
            download_dir=sym_raw,
        )
    except Exception as e:
        print(f"  DOWNLOAD FAILED: {e}")
        shutil.rmtree(sym_raw, ignore_errors=True)
        return False

    # Process
    trade_files = sorted([
        os.path.join(sym_raw, f) for f in os.listdir(sym_raw)
        if 'trades_' in f and symbol in f and f.endswith('.csv.gz')
    ])
    deriv_files = sorted([
        os.path.join(sym_raw, f) for f in os.listdir(sym_raw)
        if 'derivative_ticker_' in f and symbol in f and f.endswith('.csv.gz')
    ])

    if not trade_files:
        print(f"  No trade files, skipping")
        shutil.rmtree(sym_raw, ignore_errors=True)
        return False

    print(f"  Processing {len(trade_files)} trade days, {len(deriv_files)} ticker days...")
    print(f"  RSS at start: {_rss_mb():.0f} MB")

    # Memory bail-out: if RSS climbs above this, abort the symbol cleanly
    # rather than getting OOM-killed mid-loop. 3000 MB leaves headroom on a
    # ~16 GB machine that has Chrome/Cursor/etc also running.
    RSS_BAIL_MB = 3000

    bars_list = []
    for i, f in enumerate(trade_files):
        try:
            bars_list.append(aggregate_trades_day(f))
        except Exception as e:
            print(f"    Error processing {os.path.basename(f)}: {e}")
        if (i + 1) % 30 == 0:
            gc.collect()
            rss = _rss_mb()
            print(f"    [{i+1}/{len(trade_files)} trade days] RSS={rss:.0f} MB")
            if rss > RSS_BAIL_MB:
                print(f"    RSS BAIL ({rss:.0f} > {RSS_BAIL_MB} MB), aborting symbol")
                del bars_list
                gc.collect()
                shutil.rmtree(sym_raw, ignore_errors=True)
                return False

    if not bars_list:
        print(f"  No bars produced (all {len(trade_files)} trade days failed to aggregate), skipping")
        shutil.rmtree(sym_raw, ignore_errors=True)
        return False

    bars = pd.concat(bars_list).sort_index()
    del bars_list
    gc.collect()
    print(f"  RSS after trade concat: {_rss_mb():.0f} MB  (trade bars: {len(bars)})")

    deriv_list = []
    for i, f in enumerate(deriv_files):
        try:
            deriv_list.append(aggregate_deriv_day(f))
        except Exception as e:
            print(f"    Error processing {os.path.basename(f)}: {e}")
        if (i + 1) % 30 == 0:
            gc.collect()

    if deriv_list:
        deriv_hourly = pd.concat(deriv_list).sort_index()
        del deriv_list
        gc.collect()
        bars = bars.join(deriv_hourly, how='left')
        del deriv_hourly
        if 'open_interest' in bars.columns:
            bars['open_interest'] = bars['open_interest'].ffill()
        if 'funding_rate' in bars.columns:
            bars['funding_rate'] = bars['funding_rate'].ffill()

    bars = bars.dropna(subset=['close'])
    bars['symbol'] = symbol

    os.makedirs(bars_dir, exist_ok=True)
    bars.to_csv(bars_path)
    print(f"  Saved {len(bars)} bars to {bars_path}")
    print(f"  DONE: {symbol} ✓ ({len(bars)} bars)")
    print(f"  RSS at end: {_rss_mb():.0f} MB")

    # Free everything before next symbol
    del bars
    gc.collect()

    # Cleanup raw
    shutil.rmtree(sym_raw, ignore_errors=True)
    return True


def main():
    # Force line-buffered stdout/stderr so logs flush per line under nohup.
    # Without this, Python block-buffers when stdout isn't a tty and a crash
    # loses the entire in-flight buffer (which is exactly what bit us last run).
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_date', default='2025-10-07')
    parser.add_argument('--to_date', default='2026-04-07')
    parser.add_argument('--top_n', type=int, default=150,
                        help='Number of symbols to process (default 150)')
    parser.add_argument('--raw_dir', default='data/raw')
    parser.add_argument('--bars_dir', default='data/bars')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Override universe with specific symbols')
    parser.add_argument('--status_log', default='data/fetch_status.jsonl',
                        help='JSONL per-symbol status log (independent of stdout)')
    args = parser.parse_args()

    if args.symbols:
        universe = args.symbols
        print(f"Using {len(universe)} provided symbols")
    else:
        universe = get_universe(top_n=args.top_n)

    print(f"\nProcessing {len(universe)} symbols, {args.from_date} to {args.to_date}")
    print(f"Bars output: {args.bars_dir}")

    os.makedirs(os.path.dirname(args.status_log) or '.', exist_ok=True)
    status_f = open(args.status_log, 'a', buffering=1)  # line-buffered
    def log_status(sym, status, **kw):
        rec = {"ts": datetime.utcnow().isoformat(), "symbol": sym, "status": status, **kw}
        status_f.write(json.dumps(rec) + "\n")

    succeeded = []
    failed = []
    log_status("__run__", "start", n_symbols=len(universe), from_date=args.from_date, to_date=args.to_date)
    for i, sym in enumerate(universe):
        print(f"\n[{i+1}/{len(universe)}] {sym}")
        log_status(sym, "begin", idx=i+1, total=len(universe))
        t0 = datetime.utcnow()
        try:
            ok = process_symbol(sym, args.from_date, args.to_date,
                                args.raw_dir, args.bars_dir)
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
            print(f"\n  Progress: {len(succeeded)} ok, {len(failed)} failed")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print(f"  Failed symbols: {failed[:20]}")
    log_status("__run__", "done", succeeded=len(succeeded), failed=len(failed))
    status_f.close()


if __name__ == '__main__':
    main()
