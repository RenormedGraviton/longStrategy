#!/usr/bin/env python3
"""
Refit the winning AdaBoost classifier and plot OOS daily returns.
Winner: clf n=100 d=4 leaf=2000 lr=0.10, top/bottom 5% OOS predictions.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

FEATURE_NAMES = [
    'price_change_4h', 'oi_change_pct', 'oi_change_dollar',
    'volume_change_pct', 'volume_dollar', 'cvd_dollar',
]
TARGET = 'target_24h'
COST_BPS = 10
SEED = 42

# Winning config
N_EST = 100
DEPTH = 4
LEAF = 2000
LR = 0.10
TAIL_PCT = 5


def load_splits():
    is_df = pd.read_parquet('data/splits/is_data.parquet')
    oos_df = pd.read_parquet('data/splits/oos_data.parquet')
    caps = json.load(open('data/splits/cap_thresholds.json'))
    for fc, b in caps.items():
        is_df[fc] = is_df[fc].clip(b['p1'], b['p99'])
        oos_df[fc] = oos_df[fc].clip(b['p1'], b['p99'])
    return is_df, oos_df


def main():
    print('Loading frozen splits...')
    is_df, oos_df = load_splits()
    is_df = is_df[is_df[TARGET].notna()]
    oos_df = oos_df[oos_df[TARGET].notna()]
    print(f'IS: {len(is_df):,} rows | OOS: {len(oos_df):,} rows')

    # Fit
    print(f'\nFitting AdaBoostClassifier(n={N_EST}, d={DEPTH}, leaf={LEAF}, lr={LR})...')
    X_tr = is_df[FEATURE_NAMES].values
    y_tr = (is_df[TARGET].values > 0).astype(int)
    base = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF, random_state=SEED)
    model = AdaBoostClassifier(estimator=base, n_estimators=N_EST, learning_rate=LR, random_state=SEED)
    model.fit(X_tr, y_tr)

    # Predict on OOS
    X_te = oos_df[FEATURE_NAMES].values
    y_te = oos_df[TARGET].values
    score = model.predict_proba(X_te)[:, 1] - 0.5

    p_lo = np.percentile(score, TAIL_PCT)
    p_hi = np.percentile(score, 100 - TAIL_PCT)
    long_mask  = score >= p_hi
    short_mask = score <= p_lo

    cost = COST_BPS / 1e4

    # Build per-trade PnL with timestamps
    oos_df = oos_df.copy()
    oos_df['signal'] = 0
    oos_df.loc[long_mask,  'signal'] = 1
    oos_df.loc[short_mask, 'signal'] = -1
    oos_df['gross_pnl'] = oos_df['signal'] * oos_df[TARGET]
    oos_df['net_pnl'] = oos_df['gross_pnl'] - cost * (oos_df['signal'] != 0).astype(float)

    # Only traded rows
    trades = oos_df[oos_df['signal'] != 0].copy()
    trades['date'] = trades.index.date

    # Equal-weight daily aggregation across symbols
    daily = trades.groupby('date').agg(
        n_trades=('signal', 'size'),
        n_long=('signal', lambda s: (s > 0).sum()),
        n_short=('signal', lambda s: (s < 0).sum()),
        mean_gross=('gross_pnl', 'mean'),
        mean_net=('net_pnl', 'mean'),
        long_mean=('gross_pnl', lambda _: None),   # placeholder
    )
    # Compute long/short means separately
    long_daily  = trades[trades['signal'] > 0].groupby('date')['net_pnl'].mean()
    short_daily = trades[trades['signal'] < 0].groupby('date')['net_pnl'].mean()
    daily['long_mean']  = long_daily.reindex(daily.index).fillna(0.0)
    daily['short_mean'] = short_daily.reindex(daily.index).fillna(0.0)
    daily['combined']   = daily['mean_net']

    # Convert index to datetime for plotting
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Cumulative
    daily['cum_combined'] = daily['combined'].cumsum()
    daily['cum_long']     = daily['long_mean'].cumsum()
    daily['cum_short']    = daily['short_mean'].cumsum()

    # Metrics
    combined_pnl = trades['net_pnl'].values
    sharpe_combined = combined_pnl.mean() / combined_pnl.std() * np.sqrt(6 * 365)
    daily_sharpe = daily['combined'].mean() / daily['combined'].std() * np.sqrt(365) if daily['combined'].std() > 0 else 0
    win_rate = (combined_pnl > 0).mean()
    total_ret = combined_pnl.sum()
    mean_bps = combined_pnl.mean() * 1e4
    n_trades = len(trades)
    n_long = (trades['signal'] > 0).sum()
    n_short = (trades['signal'] < 0).sum()

    # Max drawdown on daily cumulative
    cum = daily['cum_combined'].values
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max()

    print(f'\nMetrics:')
    print(f'  Trades: {n_trades:,} ({n_long} long, {n_short} short)')
    print(f'  Win rate: {win_rate:.1%}')
    print(f'  Mean bps/trade: {mean_bps:+.2f}')
    print(f'  Per-trade Sharpe (annualized ×√(6×365)): {sharpe_combined:+.3f}')
    print(f'  Daily Sharpe (annualized ×√365):        {daily_sharpe:+.3f}')
    print(f'  Total OOS return: {total_ret*100:+.2f}%  (sum of per-trade pnl)')
    print(f'  Max daily-cumulative DD: {max_dd*100:.2f}%')
    print(f'  Days with trades: {len(daily)}')

    # ─── PLOT ───
    out_path = 'results/winner_oos_daily.pdf'
    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

        # Panel 1: cumulative return
        ax = axes[0]
        ax.plot(daily.index, daily['cum_combined'] * 100, color='black', lw=2, label='Combined')
        ax.plot(daily.index, daily['cum_long']  * 100, color='green', lw=1.2, alpha=0.7, label='Long leg')
        ax.plot(daily.index, daily['cum_short'] * 100, color='red',   lw=1.2, alpha=0.7, label='Short leg')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('Cumulative return (%)')
        ax.set_title(
            f'OOS daily performance — AdaBoostClassifier '
            f'(n={N_EST}, d={DEPTH}, leaf={LEAF}, lr={LR}), '
            f'top/bottom {TAIL_PCT}% predictions\n'
            f'{n_trades} trades, {n_long} L / {n_short} S, '
            f'per-trade Sharpe={sharpe_combined:+.2f}, '
            f'daily Sharpe={daily_sharpe:+.2f}, '
            f'win={win_rate:.1%}, '
            f'total={total_ret*100:+.1f}%, '
            f'maxDD={max_dd*100:.1f}%',
            fontsize=10, loc='left',
        )
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 2: daily net return bars
        ax = axes[1]
        colors = ['green' if v > 0 else 'red' for v in daily['combined']]
        ax.bar(daily.index, daily['combined'] * 100, color=colors, width=0.9)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Daily mean return (%)')
        ax.set_title('Daily mean net return per trade (equal-weighted across symbols that day)',
                     fontsize=10, loc='left')
        ax.grid(alpha=0.3, axis='y')

        # Panel 3: trade counts per day
        ax = axes[2]
        ax.bar(daily.index, daily['n_long'],  color='green', width=0.9, label='Long')
        ax.bar(daily.index, -daily['n_short'], color='red',   width=0.9, label='Short')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('# trades (long +, short −)')
        ax.set_xlabel('Date')
        ax.set_title('Daily trade counts (long=top, short=bottom)', fontsize=10, loc='left')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Also save as PNG for quick preview
        fig2, axes2 = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
        axes2[0].plot(daily.index, daily['cum_combined'] * 100, color='black', lw=2, label='Combined')
        axes2[0].plot(daily.index, daily['cum_long']  * 100, color='green', lw=1.2, alpha=0.7, label='Long')
        axes2[0].plot(daily.index, daily['cum_short'] * 100, color='red',   lw=1.2, alpha=0.7, label='Short')
        axes2[0].axhline(0, color='gray', lw=0.5)
        axes2[0].set_ylabel('Cumulative return (%)')
        axes2[0].set_title(
            f'OOS daily performance — AdaBoost clf(n={N_EST}, d={DEPTH}, leaf={LEAF}, lr={LR}), top/bottom {TAIL_PCT}%\n'
            f'per-trade Sharpe={sharpe_combined:+.2f}, daily Sharpe={daily_sharpe:+.2f}, '
            f'total={total_ret*100:+.1f}%, maxDD={max_dd*100:.1f}%',
            fontsize=10, loc='left')
        axes2[0].legend(loc='upper left', fontsize=9)
        axes2[0].grid(alpha=0.3)
        colors2 = ['green' if v > 0 else 'red' for v in daily['combined']]
        axes2[1].bar(daily.index, daily['combined'] * 100, color=colors2, width=0.9)
        axes2[1].axhline(0, color='black', lw=0.5)
        axes2[1].set_ylabel('Daily mean return (%)')
        axes2[1].grid(alpha=0.3, axis='y')
        axes2[2].bar(daily.index, daily['n_long'], color='green', width=0.9, label='Long')
        axes2[2].bar(daily.index, -daily['n_short'], color='red', width=0.9, label='Short')
        axes2[2].axhline(0, color='black', lw=0.5)
        axes2[2].set_ylabel('# trades')
        axes2[2].set_xlabel('Date')
        axes2[2].legend(loc='upper left', fontsize=9)
        axes2[2].grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('results/winner_oos_daily.png', dpi=140, bbox_inches='tight')
        plt.close(fig2)

    # Save the daily table too
    daily.to_csv('results/winner_oos_daily.csv')
    print(f'\nSaved:')
    print(f'  {out_path}')
    print(f'  results/winner_oos_daily.png')
    print(f'  results/winner_oos_daily.csv')


if __name__ == '__main__':
    main()
