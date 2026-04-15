#!/usr/bin/env python3
"""
Evaluate the winner model with funding rate adjustment.
Funding settles every 8h on Binance (00:00, 08:00, 16:00 UTC).
Over a 24h hold, a position crosses 3 settlements.
Long pays funding if positive, receives if negative.
Short receives funding if positive, pays if negative.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from adaboost_clean import load_all, resample_4h, compute_features, FEATURE_NAMES

MODEL_DIR = 'models/winner_clf_20260411'
COST_BPS = 10
TAIL_PCT = 5
TARGET = 'target_24h'
MIN_ABS_RET = 0.008


def compute_funding_map(symbols_list, bars_dir='data/bars'):
    """For each (symbol, 4h_timestamp), compute sum of funding at settlement
    times in the next 24h window."""
    funding_map = {}
    for sym in symbols_list:
        path = os.path.join(bars_dir, f'{sym}.csv')
        if not os.path.exists(path):
            continue
        df_1h = pd.read_csv(path, index_col=0, parse_dates=True)
        if 'funding_rate' not in df_1h.columns:
            continue

        # Settlement times only (00, 08, 16 UTC)
        settle = df_1h[df_1h.index.hour.isin([0, 8, 16])]['funding_rate']

        # 4h bar timestamps
        bars4h = resample_4h(df_1h)

        for t in bars4h.index:
            t_end = t + pd.Timedelta(hours=24)
            mask = (settle.index >= t) & (settle.index < t_end)
            funding_map[(sym, t)] = float(settle[mask].sum())

    return funding_map


def eval_with_funding(df, score, funding_map, label, cost_bps=10, tail_pct=5):
    """Evaluate with and without funding, print comparison table."""
    y = df[TARGET].values
    cost = cost_bps / 1e4

    p_lo = np.percentile(score, tail_pct)
    p_hi = np.percentile(score, 100 - tail_pct)
    long_mask = score >= p_hi
    short_mask = score <= p_lo

    # Map funding to rows
    df = df.copy()
    df['funding_24h'] = df.apply(
        lambda row: funding_map.get((row['symbol'], row['ts']), 0.0), axis=1
    )

    long_funding = df.loc[long_mask, 'funding_24h'].values
    short_funding = df.loc[short_mask, 'funding_24h'].values

    # Raw PnL (no funding)
    long_pnl_raw = y[long_mask] - cost
    short_pnl_raw = -y[short_mask] - cost
    allp_raw = np.concatenate([long_pnl_raw, short_pnl_raw])

    # Adjusted PnL (with funding)
    long_pnl_adj = y[long_mask] - long_funding - cost
    short_pnl_adj = -y[short_mask] + short_funding - cost
    allp_adj = np.concatenate([long_pnl_adj, short_pnl_adj])

    sharpe_raw = allp_raw.mean() / allp_raw.std() * np.sqrt(6 * 365)
    sharpe_adj = allp_adj.mean() / allp_adj.std() * np.sqrt(6 * 365)

    sep = '=' * 70
    dash = '-' * 30 + ' ' + '-' * 12 + ' ' + '-' * 14 + ' ' + '-' * 10
    print(f'\n{sep}')
    print(f'  {label} — FUNDING RATE ADJUSTMENT')
    print(f'{sep}')
    print(f'  {"Metric":<30} {"No Funding":>12} {"With Funding":>14} {"Delta":>10}')
    print(f'  {dash}')
    print(f'  {"Per-trade Sharpe":<30} {sharpe_raw:>+12.3f} {sharpe_adj:>+14.3f} {sharpe_adj - sharpe_raw:>+10.3f}')
    print(f'  {"Mean bps/trade":<30} {allp_raw.mean() * 1e4:>+12.1f} {allp_adj.mean() * 1e4:>+14.1f} {(allp_adj.mean() - allp_raw.mean()) * 1e4:>+10.1f}')
    print(f'  {"Long bps":<30} {long_pnl_raw.mean() * 1e4:>+12.1f} {long_pnl_adj.mean() * 1e4:>+14.1f} {(long_pnl_adj.mean() - long_pnl_raw.mean()) * 1e4:>+10.1f}')
    print(f'  {"Short bps":<30} {short_pnl_raw.mean() * 1e4:>+12.1f} {short_pnl_adj.mean() * 1e4:>+14.1f} {(short_pnl_adj.mean() - short_pnl_raw.mean()) * 1e4:>+10.1f}')
    print(f'  {"Win rate":<30} {(allp_raw > 0).mean():>12.1%} {(allp_adj > 0).mean():>14.1%}')
    print(f'  {"N long":<30} {int(long_mask.sum()):>12,} {int(long_mask.sum()):>14,}')
    print(f'  {"N short":<30} {int(short_mask.sum()):>12,} {int(short_mask.sum()):>14,}')
    print(f'')
    print(f'  Funding cost breakdown (bps):')
    print(f'    Long  mean:  {long_funding.mean() * 1e4:>+.2f}   (positive = long pays)')
    print(f'    Short mean:  {short_funding.mean() * 1e4:>+.2f}   (positive = short receives)')
    print(f'    Net impact on long:  {-long_funding.mean() * 1e4:>+.2f} bps/trade')
    print(f'    Net impact on short: {short_funding.mean() * 1e4:>+.2f} bps/trade')
    print(f'{sep}')

    return {
        'sharpe_raw': sharpe_raw, 'sharpe_adj': sharpe_adj,
        'mean_bps_raw': allp_raw.mean() * 1e4, 'mean_bps_adj': allp_adj.mean() * 1e4,
        'long_bps_raw': long_pnl_raw.mean() * 1e4, 'long_bps_adj': long_pnl_adj.mean() * 1e4,
        'short_bps_raw': short_pnl_raw.mean() * 1e4, 'short_bps_adj': short_pnl_adj.mean() * 1e4,
    }


def main():
    model = joblib.load(f'{MODEL_DIR}/model.joblib')
    meta = json.load(open(f'{MODEL_DIR}/metadata.json'))
    caps = json.load(open('data/splits/cap_thresholds.json'))

    # ── 1. Original OOS (57 symbols, frozen splits) ──
    print('Loading original OOS splits...')
    oos = pd.read_parquet('data/splits/oos_data.parquet')
    for fc, b in caps.items():
        oos[fc] = oos[fc].clip(b['p1'], b['p99'])
    oos = oos[oos[TARGET].notna()].copy()
    oos = oos.reset_index()
    oos.rename(columns={oos.columns[0]: 'ts'}, inplace=True)

    oos_syms = sorted(oos['symbol'].unique())
    print(f'OOS: {len(oos):,} rows, {len(oos_syms)} symbols')

    print('Building funding map for OOS symbols...')
    funding_map_oos = compute_funding_map(oos_syms)
    print(f'  {len(funding_map_oos):,} entries')

    X_oos = oos[FEATURE_NAMES].values
    score_oos = model.predict_proba(X_oos)[:, 1] - 0.5

    eval_with_funding(oos, score_oos, funding_map_oos,
                      f'ORIGINAL OOS ({len(oos_syms)} symbols)')

    # ── 2. New 161-symbol holdout ──
    print('\nLoading new symbols for holdout...')
    orig_syms = set(l.strip() for l in open('data/universe_top200.txt') if l.strip())
    all_symbols = load_all('data/bars')
    new_symbols = {s: df for s, df in all_symbols.items() if s not in orig_syms}

    # Volatility filter
    filtered = {}
    for sym, df_1h in new_symbols.items():
        ret = df_1h['close'].pct_change().dropna()
        if len(ret) >= 100 and ret.abs().mean() > MIN_ABS_RET:
            filtered[sym] = df_1h

    print(f'New volatile symbols: {len(filtered)}')

    all_features = []
    for sym, df_1h in filtered.items():
        bars4h = resample_4h(df_1h)
        if len(bars4h) < 10:
            continue
        f = compute_features(bars4h)
        f['symbol'] = sym
        all_features.append(f)

    holdout = pd.concat(all_features).dropna(subset=FEATURE_NAMES)
    holdout = holdout[holdout[TARGET].notna()].copy()
    holdout = holdout.reset_index()
    holdout.rename(columns={holdout.columns[0]: 'ts'}, inplace=True)

    holdout_syms = sorted(holdout['symbol'].unique())
    print(f'Holdout: {len(holdout):,} rows, {len(holdout_syms)} symbols')

    for fc, b in caps.items():
        holdout[fc] = holdout[fc].clip(b['p1'], b['p99'])

    print('Building funding map for holdout symbols...')
    funding_map_holdout = compute_funding_map(holdout_syms)
    print(f'  {len(funding_map_holdout):,} entries')

    X_hold = holdout[FEATURE_NAMES].values
    score_hold = model.predict_proba(X_hold)[:, 1] - 0.5

    eval_with_funding(holdout, score_hold, funding_map_holdout,
                      f'NEW HOLDOUT ({len(holdout_syms)} symbols)')


if __name__ == '__main__':
    main()
