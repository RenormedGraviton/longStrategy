#!/usr/bin/env python3
"""
Winner classifier OOS plot with better date labels + symbol diversity panel.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import joblib

FEATURE_NAMES = [
    'price_change_4h', 'oi_change_pct', 'oi_change_dollar',
    'volume_change_pct', 'volume_dollar', 'cvd_dollar',
]
TARGET = 'target_24h'
COST_BPS = 10
TAIL_PCT = 5


def main():
    print('Loading saved model + frozen splits...')
    model = joblib.load('models/winner_clf_20260411/model.joblib')

    is_df  = pd.read_parquet('data/splits/is_data.parquet')
    oos_df = pd.read_parquet('data/splits/oos_data.parquet')
    caps = json.load(open('data/splits/cap_thresholds.json'))
    for fc, b in caps.items():
        is_df[fc]  = is_df[fc].clip(b['p1'], b['p99'])
        oos_df[fc] = oos_df[fc].clip(b['p1'], b['p99'])
    oos_df = oos_df[oos_df[TARGET].notna()].copy()

    # Predict on OOS
    X_te = oos_df[FEATURE_NAMES].values
    y_te = oos_df[TARGET].values
    score = model.predict_proba(X_te)[:, 1] - 0.5

    p_lo = np.percentile(score, TAIL_PCT)
    p_hi = np.percentile(score, 100 - TAIL_PCT)
    long_mask  = score >= p_hi
    short_mask = score <= p_lo

    cost = COST_BPS / 1e4
    oos_df['signal'] = 0
    oos_df.loc[long_mask,  'signal'] = 1
    oos_df.loc[short_mask, 'signal'] = -1
    oos_df['trade_pnl'] = oos_df['signal'] * oos_df[TARGET] - cost * (oos_df['signal'] != 0)
    oos_df['date'] = oos_df.index.date

    trades = oos_df[oos_df['signal'] != 0].copy()

    # ── Daily aggregates ──
    def gb(df, col):
        return df.groupby('date')[col]

    # Equal-weight daily return (mean per-trade return)
    daily_mean = gb(trades, 'trade_pnl').mean()
    daily_mean.index = pd.to_datetime(daily_mean.index)

    # Long/short split
    long_daily  = gb(trades[trades['signal'] > 0], 'trade_pnl').mean()
    short_daily = gb(trades[trades['signal'] < 0], 'trade_pnl').mean()
    long_daily.index = pd.to_datetime(long_daily.index)
    short_daily.index = pd.to_datetime(short_daily.index)

    # Symbol diversity — unique symbols per day
    sym_count_all   = trades.groupby('date')['symbol'].nunique()
    sym_count_long  = trades[trades['signal'] > 0].groupby('date')['symbol'].nunique()
    sym_count_short = trades[trades['signal'] < 0].groupby('date')['symbol'].nunique()
    trade_count_long  = trades[trades['signal'] > 0].groupby('date').size()
    trade_count_short = trades[trades['signal'] < 0].groupby('date').size()
    for s in [sym_count_all, sym_count_long, sym_count_short,
              trade_count_long, trade_count_short]:
        s.index = pd.to_datetime(s.index)

    # Fill missing dates with 0
    all_dates = pd.date_range(daily_mean.index.min(), daily_mean.index.max(), freq='D')
    daily_mean       = daily_mean.reindex(all_dates).fillna(0.0)
    long_daily       = long_daily.reindex(all_dates).fillna(0.0)
    short_daily      = short_daily.reindex(all_dates).fillna(0.0)
    sym_count_all    = sym_count_all.reindex(all_dates).fillna(0).astype(int)
    sym_count_long   = sym_count_long.reindex(all_dates).fillna(0).astype(int)
    sym_count_short  = sym_count_short.reindex(all_dates).fillna(0).astype(int)
    trade_count_long  = trade_count_long.reindex(all_dates).fillna(0).astype(int)
    trade_count_short = trade_count_short.reindex(all_dates).fillna(0).astype(int)

    # Compounded equity curve (equal-weight daily)
    eq_comb  = (1 + daily_mean).cumprod() - 1
    eq_long  = (1 + long_daily).cumprod() - 1
    eq_short = (1 + short_daily).cumprod() - 1

    # Metrics
    active = (daily_mean != 0)
    sharpe_daily = daily_mean.mean() / daily_mean.std() * np.sqrt(365) if daily_mean.std() > 0 else 0
    total_ret = float(eq_comb.iloc[-1])
    eq = 1 + eq_comb.values
    max_dd = (eq / np.maximum.accumulate(eq) - 1).min()
    win_rate_days = (daily_mean[active] > 0).mean()
    mean_sym_per_day = sym_count_all[active].mean()

    print(f'\n{"="*70}')
    print(f'  SYMBOL DIVERSITY ANALYSIS')
    print(f'{"="*70}')
    print(f'Active days: {int(active.sum())}')
    print(f'Mean distinct symbols per day: {mean_sym_per_day:.1f}')
    print(f'Median distinct symbols per day: {sym_count_all[active].median():.0f}')
    print(f'Min / Max: {sym_count_all[active].min()} / {sym_count_all[active].max()}')
    print()
    print(f'Mean long-only symbols/day:  {sym_count_long[active].mean():.1f}')
    print(f'Mean short-only symbols/day: {sym_count_short[active].mean():.1f}')
    print()
    print('Trades-per-symbol ratio (how often the same symbol is re-selected):')
    print(f'  Long:  {trade_count_long[active].mean():.1f} trades / {sym_count_long[active].mean():.1f} syms = {trade_count_long[active].sum() / max(sym_count_long[active].sum(), 1):.2f}x')
    print(f'  Short: {trade_count_short[active].mean():.1f} trades / {sym_count_short[active].mean():.1f} syms = {trade_count_short[active].sum() / max(sym_count_short[active].sum(), 1):.2f}x')
    print()
    print('Total distinct symbols ever traded in OOS:')
    print(f'  Any:   {trades["symbol"].nunique()}')
    print(f'  Long:  {trades[trades["signal"] > 0]["symbol"].nunique()}')
    print(f'  Short: {trades[trades["signal"] < 0]["symbol"].nunique()}')
    print(f'  Both:  {len(set(trades[trades["signal"] > 0]["symbol"]) & set(trades[trades["signal"] < 0]["symbol"]))}')
    print(f'  Total OOS universe: {oos_df["symbol"].nunique()}')

    # ── PLOT ──
    out_pdf = 'results/winner_oos_v2.pdf'
    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)

        # Panel 1: Equity curves (compounded)
        ax = axes[0]
        ax.plot(eq_comb.index, eq_comb.values * 100, color='black', lw=2, label='Combined')
        ax.plot(eq_long.index, eq_long.values * 100, color='green', lw=1.2, alpha=0.75, label='Long leg')
        ax.plot(eq_short.index, eq_short.values * 100, color='red',   lw=1.2, alpha=0.75, label='Short leg')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('Compounded return (%)')
        ax.set_title(
            f'OOS equal-weight portfolio — AdaBoost clf(n=100, d=4, leaf=2000, lr=0.10), top/bottom {TAIL_PCT}%\n'
            f'Sharpe(ann)={sharpe_daily:+.2f}  |  '
            f'total={total_ret*100:+.1f}%  |  '
            f'maxDD={max_dd*100:.1f}%  |  '
            f'daily win={win_rate_days:.1%}  |  '
            f'mean syms/day={mean_sym_per_day:.1f}',
            fontsize=11, loc='left', fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 2: Daily return bars
        ax = axes[1]
        colors = ['green' if v > 0 else ('red' if v < 0 else 'gray') for v in daily_mean.values]
        ax.bar(daily_mean.index, daily_mean.values * 100, color=colors, width=0.9)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Daily return (%)')
        ax.set_title('Daily equal-weight portfolio return', fontsize=10, loc='left')
        ax.grid(alpha=0.3, axis='y')

        # Panel 3: Symbol diversity per day
        ax = axes[2]
        ax.bar(sym_count_long.index,   sym_count_long.values,  color='green', width=0.9, label='Distinct long syms')
        ax.bar(sym_count_short.index, -sym_count_short.values, color='red',   width=0.9, label='Distinct short syms')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('# distinct symbols')
        ax.set_title(f'Distinct symbols traded per day (long=up, short=down). '
                     f'Mean: {sym_count_long[active].mean():.1f}L / {sym_count_short[active].mean():.1f}S. '
                     f'Total OOS universe: 57 symbols',
                     fontsize=10, loc='left')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3, axis='y')

        # Panel 4: Trade counts vs symbol counts (to show re-selection rate)
        ax = axes[3]
        ax.bar(trade_count_long.index,  trade_count_long.values,  color='darkgreen', width=0.9, alpha=0.6, label='# long trades')
        ax.bar(trade_count_long.index,  sym_count_long.values,    color='lime',      width=0.5,              label='# distinct long syms')
        ax.bar(trade_count_short.index, -trade_count_short.values, color='darkred',  width=0.9, alpha=0.6, label='# short trades')
        ax.bar(trade_count_short.index, -sym_count_short.values,   color='salmon',   width=0.5,              label='# distinct short syms')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('# trades (vs distinct syms)')
        ax.set_xlabel('Date')
        ax.set_title('Trades per day vs distinct symbols per day — '
                     'gap = same symbol traded multiple 4h bars', fontsize=10, loc='left')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(alpha=0.3, axis='y')

        # ── Better date labels: month major ticks, week minor ticks ──
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            for lbl in ax.get_xticklabels(which='major'):
                lbl.set_rotation(0)
                lbl.set_ha('center')
                lbl.set_fontsize(9)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.savefig('results/winner_oos_v2.png', dpi=140, bbox_inches='tight')
        plt.close(fig)

    # Save detailed CSV
    out_csv = 'results/winner_oos_v2.csv'
    pd.DataFrame({
        'daily_return': daily_mean,
        'long_return':  long_daily,
        'short_return': short_daily,
        'eq_combined':  eq_comb,
        'n_distinct_symbols_long':  sym_count_long,
        'n_distinct_symbols_short': sym_count_short,
        'n_trades_long':  trade_count_long,
        'n_trades_short': trade_count_short,
    }).to_csv(out_csv)

    print(f'\nSaved:')
    print(f'  {out_pdf}')
    print(f'  results/winner_oos_v2.png')
    print(f'  {out_csv}')


if __name__ == '__main__':
    main()
