#!/usr/bin/env python3
"""
Equal-weight daily portfolio: 1/N capital across all trades on that day.
Realistic single-account compounding return for the winning classifier.
"""

import json
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

# Winner
N_EST, DEPTH, LEAF, LR = 100, 4, 2000, 0.10
TAIL_PCT = 5


def main():
    print('Loading frozen splits...')
    is_df  = pd.read_parquet('data/splits/is_data.parquet')
    oos_df = pd.read_parquet('data/splits/oos_data.parquet')
    caps = json.load(open('data/splits/cap_thresholds.json'))
    for fc, b in caps.items():
        is_df[fc]  = is_df[fc].clip(b['p1'], b['p99'])
        oos_df[fc] = oos_df[fc].clip(b['p1'], b['p99'])
    is_df  = is_df[is_df[TARGET].notna()]
    oos_df = oos_df[oos_df[TARGET].notna()].copy()

    print(f'Fitting clf(n={N_EST}, d={DEPTH}, leaf={LEAF}, lr={LR})...')
    X_tr = is_df[FEATURE_NAMES].values
    y_tr = (is_df[TARGET].values > 0).astype(int)
    base = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF, random_state=SEED)
    model = AdaBoostClassifier(estimator=base, n_estimators=N_EST,
                                learning_rate=LR, random_state=SEED)
    model.fit(X_tr, y_tr)

    X_te = oos_df[FEATURE_NAMES].values
    y_te = oos_df[TARGET].values
    score = model.predict_proba(X_te)[:, 1] - 0.5
    p_lo, p_hi = np.percentile(score, TAIL_PCT), np.percentile(score, 100 - TAIL_PCT)
    long_mask  = score >= p_hi
    short_mask = score <= p_lo

    cost = COST_BPS / 1e4
    oos_df['signal'] = 0
    oos_df.loc[long_mask,  'signal'] = 1
    oos_df.loc[short_mask, 'signal'] = -1
    oos_df['trade_pnl'] = oos_df['signal'] * oos_df[TARGET] - cost * (oos_df['signal'] != 0)
    oos_df['date'] = oos_df.index.date

    trades = oos_df[oos_df['signal'] != 0].copy()
    n_trades = len(trades)
    n_long  = (trades['signal'] > 0).sum()
    n_short = (trades['signal'] < 0).sum()

    # ── Equal-weight daily portfolio ──
    # Each day, split capital equally across all trades that day.
    # Daily return = mean of per-trade returns that day.
    daily_eqwt = trades.groupby('date')['trade_pnl'].mean()
    daily_eqwt.index = pd.to_datetime(daily_eqwt.index)
    daily_eqwt = daily_eqwt.sort_index()

    # Fill missing days with 0 return
    all_dates = pd.date_range(daily_eqwt.index.min(), daily_eqwt.index.max(), freq='D')
    daily_eqwt = daily_eqwt.reindex(all_dates).fillna(0.0)

    # Long-only and short-only daily portfolios
    long_daily  = trades[trades['signal'] > 0].groupby('date')['trade_pnl'].mean()
    short_daily = trades[trades['signal'] < 0].groupby('date')['trade_pnl'].mean()
    long_daily.index  = pd.to_datetime(long_daily.index)
    short_daily.index = pd.to_datetime(short_daily.index)
    long_daily  = long_daily.reindex(all_dates).fillna(0.0)
    short_daily = short_daily.reindex(all_dates).fillna(0.0)

    # Compound (geometric): equity curve
    eq_combined = (1 + daily_eqwt).cumprod() - 1
    eq_long     = (1 + long_daily).cumprod() - 1
    eq_short    = (1 + short_daily).cumprod() - 1

    # Metrics
    mean_daily = daily_eqwt.mean()
    std_daily  = daily_eqwt.std()
    sharpe_daily_ann = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0
    total_ret_compounded = float(eq_combined.iloc[-1])
    win_days = (daily_eqwt > 0).sum()
    active_days = (daily_eqwt != 0).sum()
    win_day_pct = win_days / active_days if active_days > 0 else 0

    # Max DD on equity curve
    eq = eq_combined.values
    peak = np.maximum.accumulate(1 + eq)
    dd = (1 + eq) / peak - 1
    max_dd = dd.min()

    # Annualized return
    n_days = (daily_eqwt.index.max() - daily_eqwt.index.min()).days + 1
    ann_ret = (1 + total_ret_compounded) ** (365 / n_days) - 1

    print(f'\nEqual-weight daily portfolio:')
    print(f'  Trades: {n_trades:,} ({n_long} L / {n_short} S)')
    print(f'  Active days: {active_days} / {n_days} calendar days')
    print(f'  Daily mean return:  {mean_daily*100:+.3f}%')
    print(f'  Daily std:          {std_daily*100:.3f}%')
    print(f'  Daily win rate:     {win_day_pct:.1%}')
    print(f'  Annualized Sharpe:  {sharpe_daily_ann:+.3f}')
    print(f'  Total return (compounded): {total_ret_compounded*100:+.2f}%')
    print(f'  Annualized return: {ann_ret*100:+.2f}%')
    print(f'  Max drawdown: {max_dd*100:.2f}%')

    # ── PLOT ──
    out_pdf = 'results/winner_oos_eqwt.pdf'
    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

        # Panel 1: compounded equity curves
        ax = axes[0]
        ax.plot(eq_combined.index, eq_combined.values * 100, color='black', lw=2, label='Combined')
        ax.plot(eq_long.index,     eq_long.values * 100,     color='green', lw=1.2, alpha=0.75, label='Long leg')
        ax.plot(eq_short.index,    eq_short.values * 100,    color='red',   lw=1.2, alpha=0.75, label='Short leg')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('Compounded return (%)')
        ax.set_title(
            f'OOS equal-weight portfolio — AdaBoost clf(n={N_EST}, d={DEPTH}, '
            f'leaf={LEAF}, lr={LR}), top/bottom {TAIL_PCT}%\n'
            f'{n_trades} trades, Sharpe(ann)={sharpe_daily_ann:+.2f}, '
            f'total={total_ret_compounded*100:+.1f}%, '
            f'ann={ann_ret*100:+.1f}%, '
            f'maxDD={max_dd*100:.1f}%, '
            f'daily win={win_day_pct:.1%}',
            fontsize=10, loc='left')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 2: daily return bars
        ax = axes[1]
        colors = ['green' if v > 0 else ('red' if v < 0 else 'gray') for v in daily_eqwt.values]
        ax.bar(daily_eqwt.index, daily_eqwt.values * 100, color=colors, width=0.9)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Daily return (%)')
        ax.set_title('Daily equal-weight portfolio return',
                     fontsize=10, loc='left')
        ax.grid(alpha=0.3, axis='y')

        # Panel 3: drawdown
        ax = axes[2]
        ax.fill_between(eq_combined.index, dd * 100, 0, color='red', alpha=0.4)
        ax.plot(eq_combined.index, dd * 100, color='darkred', lw=1)
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.set_title(f'Underwater curve (max DD = {max_dd*100:.2f}%)',
                     fontsize=10, loc='left')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')

        # Save PNG preview
        plt.savefig('results/winner_oos_eqwt.png', dpi=140, bbox_inches='tight')
        plt.close(fig)

    # Save daily CSV
    daily_df = pd.DataFrame({
        'daily_return': daily_eqwt,
        'long_return':  long_daily,
        'short_return': short_daily,
        'eq_combined':  eq_combined,
        'eq_long':      eq_long,
        'eq_short':     eq_short,
    })
    daily_df.to_csv('results/winner_oos_eqwt.csv')

    print(f'\nSaved:')
    print(f'  {out_pdf}')
    print(f'  results/winner_oos_eqwt.png')
    print(f'  results/winner_oos_eqwt.csv')


if __name__ == '__main__':
    main()
