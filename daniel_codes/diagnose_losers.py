#!/usr/bin/env python3
"""
Diagnose what makes systematically-bad symbols bad.

Reads `results/per_symbol_oos.csv` (output of train_rf.py) and the raw
bar files in `data/bars/`, then compares structural properties between
symbols where the model wins and where it loses.

The model is "wrong" on losers in a *consistent* way (not noise) — so
there should be a measurable structural difference: liquidity, listing
age, volatility regime, funding behavior, taker imbalance, etc. If we
can identify it, that's a feature/filter to add.
"""

import os
import sys
import pandas as pd
import numpy as np


def symbol_stats(bars_path):
    """Compute structural summary stats for one symbol from its bar file."""
    df = pd.read_csv(bars_path, index_col=0, parse_dates=True)
    if len(df) < 50:
        return None

    notional = (df['close'] * df['volume']).mean()  # USDT/hour traded
    ret = df['close'].pct_change()

    # Tardis 'open_interest' is in TOKENS (contracts). Convert to USD with mark.
    oi_tokens = df['open_interest'] if 'open_interest' in df.columns else None
    if oi_tokens is not None:
        oi_usd = (oi_tokens * df['close']).mean()
        oi_vol = oi_tokens.pct_change().std()
    else:
        oi_usd = np.nan
        oi_vol = np.nan

    return {
        'n_bars':              len(df),
        'first_bar':           df.index.min(),
        'last_bar':            df.index.max(),
        'days_listed':         (df.index.max() - df.index.min()).days,
        'avg_hourly_notional': notional,
        'mean_close':          df['close'].mean(),
        'mean_vol_24h':        ret.rolling(24).std().mean(),
        'mean_abs_ret':        ret.abs().mean(),
        'kurtosis_ret':        ret.kurt(),
        'skew_ret':            ret.skew(),
        'mean_funding':        df['funding_rate'].mean() if 'funding_rate' in df.columns else np.nan,
        'std_funding':         df['funding_rate'].std() if 'funding_rate' in df.columns else np.nan,
        'pct_extreme_funding': (df['funding_rate'].abs() > 0.005).mean() if 'funding_rate' in df.columns else np.nan,
        'mean_oi_usd':         oi_usd,
        'oi_volatility':       oi_vol,
        'mean_buy_ratio':      (df['buy_volume'] / df['volume'].replace(0, np.nan)).mean() if 'buy_volume' in df.columns else np.nan,
        'mean_hl_range_pct':   ((df['high'] - df['low']) / df['close']).mean(),
    }


def compare_groups(winners, losers, label_w='WINNERS', label_l='LOSERS'):
    """Print side-by-side group means + relative difference."""
    cols = [c for c in winners.columns if c not in
            ('symbol', 'first_bar', 'last_bar')]
    print(f"\n{'metric':<24} {label_w:>14} {label_l:>14}     diff      ratio")
    print('-' * 78)
    for c in cols:
        w = winners[c].astype(float).mean()
        l = losers[c].astype(float).mean()
        if pd.isna(w) or pd.isna(l):
            continue
        if l != 0:
            ratio = w / l
        else:
            ratio = float('inf') if w > 0 else float('nan')
        diff = w - l
        if abs(w) > 1e9 or abs(l) > 1e9:
            print(f"  {c:<22} {w:>14.2e} {l:>14.2e} {diff:>+10.2e}   {ratio:>6.2f}x")
        elif abs(w) > 100 or abs(l) > 100:
            print(f"  {c:<22} {w:>14.2f} {l:>14.2f} {diff:>+10.2f}   {ratio:>6.2f}x")
        else:
            print(f"  {c:<22} {w:>14.6f} {l:>14.6f} {diff:>+10.6f}   {ratio:>6.2f}x")


def main():
    per_sym_path = sys.argv[1] if len(sys.argv) > 1 else 'results/per_symbol_oos.csv'
    bars_dir = 'data/bars'

    if not os.path.exists(per_sym_path):
        sys.exit(f'missing {per_sym_path} — run train_rf.py first')

    per_sym = pd.read_csv(per_sym_path)
    print(f"Loaded {len(per_sym)} symbols from {per_sym_path}")
    print(f"  Sharpe range: {per_sym['sharpe'].min():.2f} to {per_sym['sharpe'].max():.2f}")
    print(f"  Trade count range: {per_sym['n_trades'].min()} to {per_sym['n_trades'].max()}")

    # Compute structural stats for each symbol
    rows = []
    for _, r in per_sym.iterrows():
        sym = r['symbol']
        bp = os.path.join(bars_dir, f'{sym}.csv')
        if not os.path.exists(bp):
            continue
        s = symbol_stats(bp)
        if s is None:
            continue
        s['symbol'] = sym
        s['sharpe'] = r['sharpe']
        s['n_trades'] = r['n_trades']
        s['dir_acc'] = r['dir_acc']
        s['total_return'] = r['total_return']
        rows.append(s)

    df = pd.DataFrame(rows)
    print(f"\nLoaded structural stats for {len(df)} symbols")

    # Split: winners (sharpe > 0.5) vs losers (sharpe < 0)
    winners = df[df['sharpe'] > 0.5]
    losers = df[df['sharpe'] < 0.0]
    middle = df[(df['sharpe'] >= 0.0) & (df['sharpe'] <= 0.5)]

    print(f"\nWINNERS (sharpe > 0.5): {len(winners)} symbols")
    print(f"  Top 10: {winners.nlargest(10, 'sharpe')[['symbol','sharpe','n_trades']].to_string(index=False)}")
    print(f"\nLOSERS  (sharpe < 0.0): {len(losers)} symbols")
    print(f"  Worst 10: {losers.nsmallest(10, 'sharpe')[['symbol','sharpe','n_trades']].to_string(index=False)}")
    print(f"\nMIDDLE  (0 ≤ sharpe ≤ 0.5): {len(middle)} symbols")

    # Group means comparison
    compare_groups(winners, losers)

    # Specific symbols of interest
    print(f"\n\n{'='*78}")
    print(f"  SYMBOLS OF INTEREST")
    print(f"{'='*78}")
    of_interest = ['BDXNUSDT', 'JCTUSDT', 'AIAUSDT', 'PIPPINUSDT', 'SIRENUSDT',
                   'A2ZUSDT', 'BANKUSDT', 'PTBUSDT', '币安人生USDT', 'BLUAIUSDT']
    interest = df[df['symbol'].isin(of_interest)].sort_values('sharpe', ascending=False)
    if len(interest):
        cols_show = ['symbol', 'sharpe', 'n_trades', 'dir_acc', 'days_listed',
                     'avg_hourly_notional', 'mean_vol_24h', 'mean_funding',
                     'pct_extreme_funding', 'mean_buy_ratio']
        cols_show = [c for c in cols_show if c in interest.columns]
        print(interest[cols_show].to_string(index=False))

    # Save full table for further inspection
    out_path = 'results/loser_diagnosis.csv'
    df.to_csv(out_path, index=False)
    print(f"\nFull stats saved: {out_path}")


if __name__ == '__main__':
    main()
