#!/usr/bin/env python3
"""
Plot the daily-aggregated equity curve, daily return bars, and drawdown
of the saved benchmark predictions.

The strategy returns are computed as:
  position[t] = sign(pred[t]) if |pred[t]| > threshold else 0
  net[t]      = position * realized_next_bar - leg_cost * |pos_change|

Per-bar returns are equal-weight averaged across all currently-active
symbols (≈ as if you allocated 1/N capital per symbol). Then aggregated
to daily by summing the per-bar means within each calendar day.

Run: python3 plot_daily_returns.py results/preds_benchmark_seed42.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def replay(df, threshold, cost_bps):
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
        gg['net_ret'] = net
        out.append(gg)
    return pd.concat(out)


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: plot_daily_returns.py <preds.parquet> [out_png] [oos_start]')
    in_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else 'results/daily_returns.png'
    oos_start = pd.Timestamp(sys.argv[3]) if len(sys.argv) > 3 else None

    df = pd.read_parquet(in_path)
    threshold = float(df['threshold'].iloc[0])
    cost_bps  = float(df['cost_bps'].iloc[0])
    horizon   = int(df['horizon'].iloc[0])
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    print(f"  Time: {df.index.min()} → {df.index.max()}")
    print(f"  Threshold: {threshold}, cost: {cost_bps} bps RT, horizon: {horizon}h")
    if oos_start:
        print(f"  Restricting to OOS window: {oos_start} →")

    # Replay strategy → per-bar net returns per symbol (always over full data
    # so the gate state and per-symbol cost accounting stay correct)
    per_bar = replay(df, threshold=threshold, cost_bps=cost_bps)

    # Equal-weight cross-sectional avg per timestamp (1 portfolio, 1/N each)
    bar_avg = per_bar.groupby(per_bar.index)['net_ret'].mean().sort_index()

    # Optional restriction to true walk-forward OOS window
    if oos_start is not None:
        bar_avg = bar_avg[bar_avg.index >= oos_start]

    # Aggregate to daily (sum of per-bar means within each calendar day).
    # Keep ALL days including ones where the gate fired everywhere — those
    # are genuine "no trade" days, not missing data. Dropping them would
    # compress the timeline and inflate the daily Sharpe.
    daily = bar_avg.resample('1D').sum()
    # Drop only days that had no bars at all (true missing)
    bars_per_day = bar_avg.resample('1D').count()
    daily = daily[bars_per_day > 0]
    cum_daily = daily.cumsum()
    drawdown = cum_daily - cum_daily.cummax()

    # Stats
    n_days        = len(daily)
    n_traded_days = (daily != 0).sum()
    n_flat_days   = n_days - n_traded_days
    win_rate_traded = (daily > 0).sum() / max(n_traded_days, 1)
    avg_day       = daily.mean()
    std_day       = daily.std()
    sharpe        = avg_day / std_day * np.sqrt(365) if std_day > 0 else 0
    cagr_proxy    = (1 + cum_daily.iloc[-1]) ** (365 / n_days) - 1 if n_days else 0

    print(f"\n  Days in series:    {n_days}")
    print(f"  Trading days:      {n_traded_days}  ({n_traded_days/n_days*100:.0f}%)")
    print(f"  Flat (gated) days: {n_flat_days}  ({n_flat_days/n_days*100:.0f}%)")
    print(f"  Final cum return:  {cum_daily.iloc[-1]*100:+.2f}%")
    print(f"  Max cum return:    {cum_daily.max()*100:+.2f}%")
    print(f"  Max drawdown:      {drawdown.min()*100:+.2f}%")
    print(f"  Win rate (traded): {win_rate_traded*100:.1f}%")
    print(f"  Avg daily return:  {avg_day*100:+.3f}%")
    print(f"  Std daily return:  {std_day*100:.3f}%")
    print(f"  Daily Sharpe (×√365):  {sharpe:.2f}")
    print(f"  Annualized return proxy: {cagr_proxy*100:+.1f}%")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 2, 2]},
                             sharex=True)

    # Vertical line at filter_through_date / in-sample boundary (2026-02-07)
    boundary = pd.Timestamp('2026-02-07')

    # 1. Cumulative equity
    ax = axes[0]
    ax.plot(cum_daily.index, cum_daily.values * 100,
            color='#2ecc71', linewidth=1.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(boundary, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.6)
    ax.text(boundary, ax.get_ylim()[1] * 0.95, '  filter cutoff\n  (2026-02-07)',
            color='#e74c3c', fontsize=8, va='top')
    ax.set_ylabel('Cumulative return (%)')
    ax.set_title(f'Daily-aggregated equity curve — benchmark seed 42\n'
                 f'Final {cum_daily.iloc[-1]*100:+.1f}% | Daily Sharpe {sharpe:.2f} | '
                 f'Win rate (traded) {win_rate_traded*100:.0f}% | '
                 f'Trading days {n_traded_days}/{n_days} | '
                 f'Max DD {drawdown.min()*100:.1f}%',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # 2. Daily returns bars (red/green)
    ax = axes[1]
    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in daily]
    ax.bar(daily.index, daily.values * 100, color=colors, edgecolor='none', width=0.9)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(boundary, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_ylabel('Daily return (%)')
    ax.set_title('Daily returns', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 3. Drawdown curve
    ax = axes[2]
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                    color='#e74c3c', alpha=0.4)
    ax.plot(drawdown.index, drawdown.values * 100, color='#c0392b', linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(boundary, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.set_title('Drawdown from peak', fontweight='bold')
    ax.grid(alpha=0.3)

    # Format x-axis
    for a in axes:
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"\nSaved: {out_png}")

    # Also save the daily returns as CSV for further inspection
    csv_path = out_png.replace('.png', '.csv')
    out_df = pd.DataFrame({
        'daily_ret': daily,
        'cum_ret': cum_daily,
        'drawdown': drawdown,
    })
    out_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
