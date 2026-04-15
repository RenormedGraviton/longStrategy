#!/usr/bin/env python3
"""
Quick baseline: apply the winning 24h classifier recipe (n=100, d=4,
leaf=2000, lr=0.10) to target_12h and see how it does. This is just a
sanity check before running a full target_12h sweep.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    'price_change_4h', 'oi_change_pct', 'oi_change_dollar',
    'volume_change_pct', 'volume_dollar', 'cvd_dollar',
]
COST_BPS = 10
SEED = 42
ANNUALIZE = np.sqrt(6 * 365)

# Winning 24h recipe
N_EST, DEPTH, LEAF, LR = 100, 4, 2000, 0.10
TAIL_PCT = 5


def evaluate(target_col):
    print(f'\n{"="*70}')
    print(f'  TARGET: {target_col}')
    print(f'{"="*70}')

    is_df = pd.read_parquet('data/splits/is_data.parquet')
    oos_df = pd.read_parquet('data/splits/oos_data.parquet')
    caps = json.load(open('data/splits/cap_thresholds.json'))
    for fc, b in caps.items():
        is_df[fc] = is_df[fc].clip(b['p1'], b['p99'])
        oos_df[fc] = oos_df[fc].clip(b['p1'], b['p99'])

    is_df = is_df[is_df[target_col].notna()]
    oos_df = oos_df[oos_df[target_col].notna()]
    print(f'  IS rows: {len(is_df):,}  |  OOS rows: {len(oos_df):,}')

    X_tr = is_df[FEATURES].values
    y_tr = (is_df[target_col].values > 0).astype(int)
    base = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF, random_state=SEED)
    model = AdaBoostClassifier(estimator=base, n_estimators=N_EST,
                                learning_rate=LR, random_state=SEED)
    model.fit(X_tr, y_tr)

    X_te = oos_df[FEATURES].values
    y_te = oos_df[target_col].values
    sym_te = oos_df['symbol'].values
    score = model.predict_proba(X_te)[:, 1] - 0.5

    cost = COST_BPS / 1e4

    # All-trades evaluation
    pred_all = np.where(score > 0, 1.0, -1.0)
    pnl_all = pred_all * y_te - cost
    sharpe_all = pnl_all.mean() / pnl_all.std() * ANNUALIZE if pnl_all.std() > 0 else 0
    acc_all = ((score > 0) == (y_te > 0)).mean()

    print(f'\n  ALL-TRADES (sign of score):')
    print(f'    Accuracy:            {acc_all:.4f}')
    print(f'    Mean bps/trade:      {pnl_all.mean()*1e4:+.2f}')
    print(f'    Per-trade Sharpe:    {sharpe_all:+.3f}')

    # Tail evaluation (top/bottom 5%)
    p_lo = np.percentile(score, TAIL_PCT)
    p_hi = np.percentile(score, 100 - TAIL_PCT)
    long_mask = score >= p_hi
    short_mask = score <= p_lo

    long_pnl = y_te[long_mask] - cost
    short_pnl = -y_te[short_mask] - cost
    all_pnl = np.concatenate([long_pnl, short_pnl])

    sharpe_tail = all_pnl.mean() / all_pnl.std() * ANNUALIZE if all_pnl.std() > 0 else 0
    long_sh = long_pnl.mean() / long_pnl.std() * ANNUALIZE if len(long_pnl) > 1 and long_pnl.std() > 0 else 0
    short_sh = short_pnl.mean() / short_pnl.std() * ANNUALIZE if len(short_pnl) > 1 and short_pnl.std() > 0 else 0
    win = (all_pnl > 0).mean()

    print(f'\n  TAIL {TAIL_PCT}% / {TAIL_PCT}%:')
    print(f'    Long trades:         {int(long_mask.sum()):,}  ({long_pnl.mean()*1e4:+.2f} bps, Sharpe={long_sh:+.2f})')
    print(f'    Short trades:        {int(short_mask.sum()):,}  ({short_pnl.mean()*1e4:+.2f} bps, Sharpe={short_sh:+.2f})')
    print(f'    Combined mean bps:   {all_pnl.mean()*1e4:+.2f}')
    print(f'    Combined Sharpe:     {sharpe_tail:+.3f}')
    print(f'    Win rate:            {win*100:.1f}%')

    # Equal-weight daily portfolio (same as target_24h eval)
    df = oos_df.copy()
    df['signal'] = 0
    df.loc[long_mask, 'signal'] = 1
    df.loc[short_mask, 'signal'] = -1
    df['trade_pnl'] = df['signal'] * df[target_col] - cost * (df['signal'] != 0)
    df['date'] = df.index.date
    trades = df[df['signal'] != 0]
    daily = trades.groupby('date')['trade_pnl'].mean()

    mean_d = daily.mean()
    std_d = daily.std()
    sharpe_daily_ann = mean_d / std_d * np.sqrt(365) if std_d > 0 else 0
    eq = (1 + daily).cumprod()
    total_ret = eq.iloc[-1] - 1
    max_dd = (eq / eq.cummax() - 1).min()

    print(f'\n  EQUAL-WEIGHT DAILY PORTFOLIO:')
    print(f'    Daily mean return:   {mean_d*100:+.3f}%')
    print(f'    Daily Sharpe (ann):  {sharpe_daily_ann:+.3f}')
    print(f'    Compounded return:   {total_ret*100:+.1f}%')
    print(f'    Max drawdown:        {max_dd*100:.2f}%')
    print(f'    Active days:         {(daily != 0).sum()}')

    return {
        'target': target_col,
        'all_trade_sharpe': round(sharpe_all, 3),
        'tail_sharpe':      round(sharpe_tail, 3),
        'tail_long_bps':    round(long_pnl.mean()*1e4, 2),
        'tail_short_bps':   round(short_pnl.mean()*1e4, 2),
        'daily_sharpe_ann': round(sharpe_daily_ann, 3),
        'compounded_ret':   round(total_ret, 4),
        'max_dd':           round(max_dd, 4),
    }


def main():
    results = []
    for target in ['target_4h', 'target_8h', 'target_12h', 'target_24h']:
        r = evaluate(target)
        results.append(r)

    print(f'\n\n{"="*70}')
    print('  SUMMARY — same winning classifier recipe across horizons')
    print(f'{"="*70}')
    print(f'{"target":<12} {"all_Sh":>7} {"5%_Sh":>7} {"L_bps":>7} {"S_bps":>7} {"daily_Sh":>10} {"total_ret":>11} {"maxDD":>7}')
    print('-' * 75)
    for r in results:
        print(f'{r["target"]:<12} {r["all_trade_sharpe"]:>+7.3f} {r["tail_sharpe"]:>+7.3f} '
              f'{r["tail_long_bps"]:>+7.1f} {r["tail_short_bps"]:>+7.1f} '
              f'{r["daily_sharpe_ann"]:>+10.3f} {r["compounded_ret"]*100:>+10.1f}% '
              f'{r["max_dd"]*100:>+6.1f}%')

    pd.DataFrame(results).to_csv('results/try_horizons_baseline.csv', index=False)
    print(f'\nSaved: results/try_horizons_baseline.csv')


if __name__ == '__main__':
    main()
