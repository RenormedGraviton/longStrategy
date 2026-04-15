#!/usr/bin/env python3
"""
Fetch historical data from Tardis.dev for Binance perpetual futures.

Tardis provides:
- trades        — every trade (used to build candles)
- derivative_ticker — funding rate, open interest, mark price, index price (snapshots)

We download these CSV files (gzipped) for a date range, then resample to 1-hour bars
for the strategy. Tardis has YEARS of history vs Binance API's 30-day OI limit.
"""

import os
import argparse
import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tardis_dev import datasets

API_KEY = "TD.BcdwoEMKCC2YKW-Z.cbwZZsGxT2tOHGP.VpRFIrFcmScXfui.wNcJsHQDYfzfp6g.NlBGOyqmsIIqT4E.HzZk"

def download(symbol='BTCUSDT', from_date='2024-01-01', to_date='2024-12-31', out_dir='data/raw'):
    """Download trades and derivative_ticker CSVs from Tardis."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {symbol} from {from_date} to {to_date}...")
    print(f"  Data types: trades, derivative_ticker")
    print(f"  Output dir: {out_dir}")

    datasets.download(
        exchange="binance-futures",
        data_types=["trades", "derivative_ticker"],
        from_date=from_date,
        to_date=to_date,
        symbols=[symbol],
        api_key=API_KEY,
        download_dir=out_dir,
    )
    print("Download complete.")


def _aggregate_trades_day(f):
    """Build 1h bars from a single day's trade file. Memory-efficient."""
    df = pd.read_csv(f, compression='gzip',
                     usecols=['timestamp', 'side', 'price', 'amount'])
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us')
    df['price'] = df['price'].astype(float)
    df['amount'] = df['amount'].astype(float)
    df['quote_volume'] = df['price'] * df['amount']
    df['is_buy'] = (df['side'] == 'buy').astype(int)
    df['buy_volume'] = df['amount'] * df['is_buy']
    df['sell_volume'] = df['amount'] * (1 - df['is_buy'])
    df = df.set_index('ts')

    bars = pd.DataFrame()
    bars['open'] = df['price'].resample('1h').first()
    bars['high'] = df['price'].resample('1h').max()
    bars['low'] = df['price'].resample('1h').min()
    bars['close'] = df['price'].resample('1h').last()
    bars['volume'] = df['amount'].resample('1h').sum()
    bars['quote_volume'] = df['quote_volume'].resample('1h').sum()
    bars['trades_count'] = df['price'].resample('1h').count()
    bars['buy_volume'] = df['buy_volume'].resample('1h').sum()
    bars['sell_volume'] = df['sell_volume'].resample('1h').sum()
    bars['vwap'] = bars['quote_volume'] / bars['volume'].replace(0, np.nan)
    return bars


def _aggregate_deriv_day(f):
    """Build 1h OI + funding bars from a single day's derivative_ticker file."""
    df = pd.read_csv(f, compression='gzip')
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us')
    df = df.set_index('ts')

    bars = pd.DataFrame()
    if 'open_interest' in df.columns:
        bars['open_interest'] = df['open_interest'].astype(float).resample('1h').last()
    if 'funding_rate' in df.columns:
        bars['funding_rate'] = df['funding_rate'].astype(float).resample('1h').last()
    if 'predicted_funding_rate' in df.columns:
        bars['predicted_funding_rate'] = df['predicted_funding_rate'].astype(float).resample('1h').last()
    if 'mark_price' in df.columns:
        bars['mark_price'] = df['mark_price'].astype(float).resample('1h').last()
    if 'index_price' in df.columns:
        bars['index_price'] = df['index_price'].astype(float).resample('1h').last()
    return bars


def build_hourly_bars(symbol='BTCUSDT', raw_dir='data/raw', out='data/hourly.csv'):
    """Build 1-hour OHLCV + OI + funding rate bars from raw Tardis CSVs.
    Processes day-by-day to limit memory usage."""
    print(f"\nBuilding hourly bars for {symbol} (memory-efficient day-by-day)...")

    trade_files = sorted([
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if 'trades_' in f and symbol in f and f.endswith('.csv.gz')
    ])
    deriv_files = sorted([
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if 'derivative_ticker_' in f and symbol in f and f.endswith('.csv.gz')
    ])
    print(f"  {len(trade_files)} trade files, {len(deriv_files)} ticker files")

    print("  Aggregating trade files day-by-day...")
    bars_list = []
    for i, f in enumerate(trade_files):
        if (i + 1) % 30 == 0 or i == len(trade_files) - 1:
            print(f"    [{i+1}/{len(trade_files)}] {os.path.basename(f)}")
        bars_list.append(_aggregate_trades_day(f))
    bars = pd.concat(bars_list).sort_index()
    del bars_list
    print(f"    {len(bars)} hourly bars from trades")

    print("  Aggregating derivative_ticker files day-by-day...")
    deriv_list = []
    for i, f in enumerate(deriv_files):
        if (i + 1) % 30 == 0 or i == len(deriv_files) - 1:
            print(f"    [{i+1}/{len(deriv_files)}] {os.path.basename(f)}")
        deriv_list.append(_aggregate_deriv_day(f))
    deriv_hourly = pd.concat(deriv_list).sort_index()
    del deriv_list
    print(f"    {len(deriv_hourly)} hourly OI/funding bars")

    print("  Merging...")
    combined = bars.join(deriv_hourly, how='left')
    combined['open_interest'] = combined['open_interest'].ffill()
    combined['funding_rate'] = combined['funding_rate'].ffill()
    combined = combined.dropna(subset=['close'])

    combined.to_csv(out)
    print(f"\nSaved {len(combined)} bars to {out}")
    print(f"  Range: {combined.index.min()} to {combined.index.max()}")
    print(f"  Columns: {list(combined.columns)}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    # 6 months: 4 in-sample + 2 OOS
    parser.add_argument('--from_date', default='2025-10-07')
    parser.add_argument('--to_date', default='2026-04-07')
    parser.add_argument('--raw_dir', default='data/raw')
    parser.add_argument('--out', default='data/hourly.csv')
    parser.add_argument('--skip_download', action='store_true')
    args = parser.parse_args()

    if not args.skip_download:
        download(args.symbol, args.from_date, args.to_date, args.raw_dir)

    build_hourly_bars(args.symbol, args.raw_dir, args.out)


if __name__ == '__main__':
    main()
