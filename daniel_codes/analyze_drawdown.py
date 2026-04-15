#!/usr/bin/env python3
"""
Drawdown forensics for the 4h hybrid strategy.

For each saved per-bar prediction file:
1. Replay the strategy with the same threshold + cost as training
2. Build the time-aggregated equity curve (sum across all symbols per bar)
3. Identify the worst N drawdown periods (peak-to-trough)
4. For each bad period, report: which symbols contributed most loss,
   what BTC was doing, what cross-symbol return correlation looked like

The goal is to learn whether the deep DDs are:
  (a) idiosyncratic to one bad symbol → fix: per-symbol risk caps
  (b) correlated regime breaks → fix: market-wide vol gate
  (c) just bad luck spread evenly → fix: position sizing

Run: python3 analyze_drawdown.py results/preds_seed42.parquet
"""

import os
import sys
import pandas as pd
import numpy as np


def replay_strategy(df, threshold, cost_bps):
    """Convert per-bar predictions into per-bar strategy returns (per symbol).

    Position rules: long if pred > +threshold, short if pred < -threshold,
    flat otherwise. Costs charged per leg (entry/exit), where each unit
    of |delta_position| = one leg = cost_bps/2 cents.
    """
    cost_per_leg = (cost_bps / 2.0) / 1e4
    out = []
    for sym, g in df.groupby('symbol', sort=False):
        g = g.sort_index()
        pos = np.zeros(len(g))
        pos[g['pred'].values >  threshold] =  1
        pos[g['pred'].values < -threshold] = -1
        gross = pos * g['target_return'].values
        legs = np.abs(np.diff(np.concatenate([[0.0], pos])))
        costs = legs * cost_per_leg
        net = gross - costs

        gg = g.copy()
        gg['position'] = pos
        gg['gross_ret'] = gross
        gg['cost'] = costs
        gg['net_ret'] = net
        out.append(gg)
    return pd.concat(out)


def aggregate_equity(per_bar):
    """Sum strategy returns across all symbols at each timestamp.

    Each timestamp's value is the *average* per-symbol return — i.e., as if
    you allocated equal capital to every symbol every bar. NaNs (when a
    symbol has no data at that timestamp) are excluded from the mean.
    """
    g = per_bar.groupby(per_bar.index)
    bar_ret = g['net_ret'].mean()           # equal-weight across active symbols
    n_active = g['symbol'].nunique()
    bar_ret = bar_ret.sort_index()
    n_active = n_active.sort_index()
    cum = bar_ret.cumsum()
    return pd.DataFrame({
        'bar_ret': bar_ret,
        'n_active': n_active,
        'cum_ret': cum,
    })


def find_drawdowns(eq, top_n=5):
    """Find the top-N largest peak-to-trough drawdowns in the equity curve.

    Returns a DataFrame with peak_time, trough_time, peak_val, trough_val,
    drawdown (negative), duration_bars, recovery_time (or NaT).
    """
    cum = eq['cum_ret'].values
    times = eq.index
    peaks = np.maximum.accumulate(cum)
    dd = cum - peaks  # always <= 0

    # Find local drawdown periods: between consecutive new highs
    new_high = (cum == peaks)
    high_idx = np.where(new_high)[0].tolist()
    high_idx.append(len(cum))  # sentinel

    periods = []
    for i in range(len(high_idx) - 1):
        start = high_idx[i]
        end = high_idx[i + 1]
        if end - start < 2:
            continue
        segment = dd[start:end]
        trough_off = np.argmin(segment)
        trough = segment[trough_off]
        if trough >= 0:
            continue
        periods.append({
            'peak_time': times[start],
            'peak_val': cum[start],
            'trough_time': times[start + trough_off],
            'trough_val': cum[start + trough_off],
            'drawdown': trough,
            'duration_bars': trough_off,
        })

    if not periods:
        return pd.DataFrame()
    df = pd.DataFrame(periods).sort_values('drawdown')
    return df.head(top_n).reset_index(drop=True)


def analyze_dd_period(per_bar, peak_time, trough_time, top_n_symbols=10):
    """For one drawdown period, report which symbols lost the most + market state."""
    mask = (per_bar.index >= peak_time) & (per_bar.index <= trough_time)
    period = per_bar[mask]

    # Per-symbol contribution
    sym_pnl = period.groupby('symbol')['net_ret'].sum().sort_values()
    n_active = period['symbol'].nunique()

    # Cross-symbol correlation of UNDERLYING returns during the period
    underlying = period.pivot_table(
        index=period.index, columns='symbol', values='target_return')
    avg_pairwise_corr = (underlying.corr().values[np.triu_indices_from(underlying.corr().values, k=1)].mean()
                         if underlying.shape[1] > 1 else np.nan)

    # Market direction proxy: equal-weight average of underlying returns
    market_avg = underlying.mean(axis=1)
    market_total = market_avg.sum()

    return {
        'duration_hours': int((trough_time - peak_time).total_seconds() / 3600),
        'n_symbols_active': n_active,
        'market_total_ret': market_total,           # average underlying token's return
        'avg_pairwise_corr': avg_pairwise_corr,
        'worst_5_symbols': sym_pnl.head(5).to_dict(),
        'best_5_symbols': sym_pnl.tail(5).to_dict(),
        'sym_pnl_full': sym_pnl,
    }


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_drawdown.py <preds.parquet> [threshold] [cost_bps]")
    path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0007
    cost_bps  = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    print(f"Loading {path}")
    df = pd.read_parquet(path)
    print(f"  {len(df)} bars, {df['symbol'].nunique()} symbols")
    print(f"  Time range: {df.index.min()} → {df.index.max()}")
    print(f"  Replaying strategy with threshold={threshold}, cost={cost_bps} bps RT")

    per_bar = replay_strategy(df, threshold=threshold, cost_bps=cost_bps)
    eq = aggregate_equity(per_bar)

    print(f"\n{'='*70}")
    print(f"  EQUITY CURVE SUMMARY")
    print(f"{'='*70}")
    print(f"  Bars in equity series: {len(eq)}")
    print(f"  Final cum_ret:         {eq['cum_ret'].iloc[-1]*100:+.1f}%")
    print(f"  Max cum_ret:           {eq['cum_ret'].max()*100:+.1f}%")
    print(f"  Min cum_ret:           {eq['cum_ret'].min()*100:+.1f}%")
    print(f"  Equal-weight Sharpe:   {eq['bar_ret'].mean()/eq['bar_ret'].std()*np.sqrt(365*24/4):.2f}")

    dds = find_drawdowns(eq, top_n=5)
    if dds.empty:
        print("  No drawdowns found.")
        return

    print(f"\n{'='*70}")
    print(f"  TOP 5 DRAWDOWN PERIODS")
    print(f"{'='*70}")
    print(dds.to_string(index=False))

    # Detailed look at worst drawdown
    worst = dds.iloc[0]
    print(f"\n{'='*70}")
    print(f"  WORST DRAWDOWN: {worst['peak_time']} → {worst['trough_time']}")
    print(f"  Magnitude: {worst['drawdown']*100:+.1f}%")
    print(f"{'='*70}")

    info = analyze_dd_period(per_bar, worst['peak_time'], worst['trough_time'])
    print(f"  Duration:               {info['duration_hours']} hours ({info['duration_hours']/24:.1f} days)")
    print(f"  Symbols active:         {info['n_symbols_active']}")
    print(f"  Avg market return:      {info['market_total_ret']*100:+.2f}% (equal-weighted underlying)")
    print(f"  Avg pairwise corr:      {info['avg_pairwise_corr']:+.3f}")
    print(f"\n  Worst 5 symbol contributions (cumulative net_ret over the period):")
    for sym, pnl in info['worst_5_symbols'].items():
        print(f"    {sym:20s}  {pnl*100:+8.2f}%")
    print(f"\n  Best 5 symbol contributions:")
    for sym, pnl in info['best_5_symbols'].items():
        print(f"    {sym:20s}  {pnl*100:+8.2f}%")

    # Distribution: how concentrated is the loss?
    sym_pnl = info['sym_pnl_full']
    losing = sym_pnl[sym_pnl < 0]
    print(f"\n  Symbol P&L distribution during this DD:")
    print(f"    Losing symbols:  {len(losing)} of {len(sym_pnl)} ({len(losing)/len(sym_pnl)*100:.0f}%)")
    print(f"    Top-3 losers contribute {sym_pnl.head(3).sum()*100:+.2f}% (of total {sym_pnl.sum()*100:+.2f}%)")
    if sym_pnl.sum() != 0:
        concentration = sym_pnl.head(3).sum() / sym_pnl[sym_pnl<0].sum() * 100
        print(f"    Top-3 losers = {concentration:.0f}% of all loss")


if __name__ == '__main__':
    main()
