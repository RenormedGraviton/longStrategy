#!/usr/bin/env python3
"""
Fetch hourly data for Binance perpetual futures.
- Klines (OHLCV) — 1000 bars per request
- Open Interest history — 30-day rolling window via API, hourly granularity
- Funding rate — 1000 records per request
"""

import requests
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta
import os

BASE = "https://fapi.binance.com"

def fetch_klines(symbol, interval='1h', start_ms=None, end_ms=None, limit=1000):
    """Fetch klines (OHLCV) from Binance."""
    url = f"{BASE}/fapi/v1/klines"
    all_data = []
    cursor = start_ms

    while True:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if cursor: params['startTime'] = cursor
        if end_ms: params['endTime'] = end_ms

        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        all_data.extend(data)
        if len(data) < limit:
            break

        cursor = data[-1][0] + 1
        if end_ms and cursor >= end_ms:
            break
        time.sleep(0.1)

    cols = ['open_time','open','high','low','close','volume','close_time',
            'quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
    df = pd.DataFrame(all_data, columns=cols)
    if df.empty:
        return df
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open','high','low','close','volume','quote_volume','taker_buy_base','taker_buy_quote']:
        df[c] = df[c].astype(float)
    df['trades'] = df['trades'].astype(int)
    return df.drop(columns=['ignore', 'close_time'])


def fetch_open_interest(symbol, period='1h', start_ms=None, end_ms=None, limit=500):
    """Fetch open interest history. Note: only last 30 days available via this endpoint."""
    url = f"{BASE}/futures/data/openInterestHist"
    all_data = []
    cursor = start_ms

    while True:
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        if cursor: params['startTime'] = cursor
        if end_ms: params['endTime'] = end_ms

        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"  OI fetch error: {r.status_code} {r.text}")
            break
        data = r.json()
        if not data:
            break

        all_data.extend(data)
        if len(data) < limit:
            break

        cursor = data[-1]['timestamp'] + 1
        if end_ms and cursor >= end_ms:
            break
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
    df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
    return df


def fetch_funding_rate(symbol, start_ms=None, end_ms=None, limit=1000):
    """Fetch funding rate history. Funding is typically every 8h (or 4h for some)."""
    url = f"{BASE}/fapi/v1/fundingRate"
    all_data = []
    cursor = start_ms

    while True:
        params = {'symbol': symbol, 'limit': limit}
        if cursor: params['startTime'] = cursor
        if end_ms: params['endTime'] = end_ms

        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        all_data.extend(data)
        if len(data) < limit:
            break

        cursor = data[-1]['fundingTime'] + 1
        if end_ms and cursor >= end_ms:
            break
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    return df[['fundingTime', 'fundingRate']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--days', type=int, default=30, help='days of history (OI capped at 30)')
    parser.add_argument('--out', default='data')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    end = int(time.time() * 1000)
    start = end - args.days * 24 * 3600 * 1000

    print(f"Fetching {args.symbol} data for last {args.days} days...")

    print("  Klines (1h)...")
    klines = fetch_klines(args.symbol, '1h', start, end)
    print(f"    {len(klines)} bars")
    klines.to_csv(f"{args.out}/{args.symbol}_klines_1h.csv", index=False)

    print("  Open Interest (1h)...")
    oi = fetch_open_interest(args.symbol, '1h', start, end)
    print(f"    {len(oi)} bars")
    if not oi.empty:
        oi.to_csv(f"{args.out}/{args.symbol}_oi_1h.csv", index=False)

    print("  Funding Rate...")
    funding = fetch_funding_rate(args.symbol, start, end)
    print(f"    {len(funding)} records")
    if not funding.empty:
        funding.to_csv(f"{args.out}/{args.symbol}_funding.csv", index=False)

    print(f"\nDone. Data saved to {args.out}/")
    print(f"  Klines:  {len(klines)} rows, {klines['open_time'].min()} to {klines['open_time'].max()}")
    if not oi.empty:
        print(f"  OI:      {len(oi)} rows, {oi['timestamp'].min()} to {oi['timestamp'].max()}")
    if not funding.empty:
        print(f"  Funding: {len(funding)} rows, {funding['fundingTime'].min()} to {funding['fundingTime'].max()}")


if __name__ == '__main__':
    main()
