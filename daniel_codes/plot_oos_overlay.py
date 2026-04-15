#!/usr/bin/env python3
"""
Overlay daily-aggregated cumulative equity curves from multiple seed
benchmark runs, restricted to a true walk-forward OOS window.

Run: python3 plot_oos_overlay.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


SEEDS = [42, 1, 7, 13, 100, 2024, 12345]
OOS_START = pd.Timestamp('2026-02-07')
OUT = 'results/oos_overlay.png'


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
        net = gross - legs * cost_per_leg
        gg = g.copy()
        gg['net_ret'] = net
        out.append(gg)
    return pd.concat(out)


def daily_oos_curve(parquet_path):
    df = pd.read_parquet(parquet_path)
    threshold = float(df['threshold'].iloc[0])
    cost_bps  = float(df['cost_bps'].iloc[0])
    per_bar = replay(df, threshold=threshold, cost_bps=cost_bps)
    bar_avg = per_bar.groupby(per_bar.index)['net_ret'].mean().sort_index()
    bar_avg = bar_avg[bar_avg.index >= OOS_START]
    daily = bar_avg.resample('1D').sum()
    bars_per_day = bar_avg.resample('1D').count()
    daily = daily[bars_per_day > 0]
    return daily


def main():
    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={'height_ratios': [3, 2]},
                             sharex=True)

    cmap = plt.cm.viridis
    finals = []
    sharpes = []
    dds = []

    for i, seed in enumerate(SEEDS):
        path = f'results/preds_benchmark_seed{seed}.parquet'
        if not os.path.exists(path):
            print(f"  missing {path} — skipping")
            continue
        daily = daily_oos_curve(path)
        cum = daily.cumsum()
        dd = cum - cum.cummax()
        n_traded = (daily != 0).sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
        finals.append(cum.iloc[-1])
        sharpes.append(sharpe)
        dds.append(dd.min())

        color = cmap(i / max(len(SEEDS) - 1, 1))
        label = f'seed {seed}: +{cum.iloc[-1]*100:.1f}% / Sharpe {sharpe:.2f} / DD {dd.min()*100:.1f}%'
        axes[0].plot(cum.index, cum.values * 100, color=color, linewidth=1.6, label=label)
        axes[1].plot(dd.index, dd.values * 100, color=color, linewidth=1.2, alpha=0.8)

    # Compute mean equity curve (interpolate to common index)
    print("\nOOS-only stats (post-2026-02-07):")
    print(f"  Mean final return: {np.mean(finals)*100:+.2f}%")
    print(f"  Mean Sharpe:       {np.mean(sharpes):.2f}")
    print(f"  Std Sharpe:        {np.std(sharpes):.2f}")
    print(f"  Min Sharpe:        {np.min(sharpes):.2f}")
    print(f"  Max Sharpe:        {np.max(sharpes):.2f}")
    print(f"  Mean max DD:       {np.mean(dds)*100:+.2f}%")
    print(f"  Worst max DD:      {np.min(dds)*100:+.2f}%")

    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_ylabel('Cumulative return (%)')
    axes[0].set_title(f'OOS-only daily equity curves (post {OOS_START.date()}) — 7 random seeds\n'
                      f'Mean Sharpe {np.mean(sharpes):.2f} (range {np.min(sharpes):.2f}-{np.max(sharpes):.2f}) | '
                      f'Mean final {np.mean(finals)*100:+.1f}% over 59 days | '
                      f'Mean DD {np.mean(dds)*100:.1f}%',
                      fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Drawdown from peak (per seed)', fontweight='bold')
    axes[1].grid(alpha=0.3)

    for a in axes:
        a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
